/**
 * NovaFlagship · L8 智能体运行时层
 * ------------------------------------------------------------------
 * 范式：ReAct（Yao et al., ICLR 2023）推理-行动交替循环 + Reflexion 式
 * 自我批评钩子（Shinn et al., NeurIPS 2023）+ 硬预算（步数/Token/墙钟）。
 *
 * 旗舰能力 —— 确定性回放（Deterministic Replay）：
 *   每一步（LLM 响应、工具调用、工具观察）都进入 TraceRecorder；
 *   replayAgent() 用录制的响应替换模型，重放整条决策链，
 *   要求事件序列逐条相等 —— 这是「可复现、可验证」的工程化身，
 *   也是回归测试的锚（tests/agent.test.js 背书）。
 */

import { ModelGateway } from './gateway.js';

/** 轨迹记录器：事件流 + JSONL 序列化。 */
export class TraceRecorder {
  constructor() {
    this.events = [];
  }

  record(type, payload = {}) {
    const ev = { seq: this.events.length, type, ts: Date.now(), payload };
    this.events.push(ev);
    return ev;
  }

  toJSON() {
    return { version: 1, events: this.events };
  }

  toJSONL() {
    return this.events.map((e) => JSON.stringify(e)).join('\n') + '\n';
  }

  /**
   * 去掉 ts 与「环境噪声」后的结构化事件序列（用于回放比对）。
   * 归一化规则：llm 事件剔除 model/provider（回放时由 replay 提供商服务），
   * tool_result 剔除 ms（延迟是环境量，不是决策量）。
   * 回放相等 ⇒ 决策序列（消息、工具名/参数、终答）逐条一致。
   */
  structural() {
    return this.events.map(({ seq, type, payload }) => {
      const p = JSON.parse(JSON.stringify(payload));
      if (type === 'llm') {
        delete p.model;
        delete p.provider;
      }
      if (type === 'tool_result') delete p.ms;
      return { seq, type, payload: p };
    });
  }
}

const DEFAULT_BUDGET = { maxSteps: 12, maxTokens: 8000, maxMs: 120000, maxToolCalls: 24 };

export class AgentRuntime {
  constructor({
    gateway,
    tools,
    memory = null,
    observe = null,
    budget = {},
    systemPrompt = '你是 NovaFlagship 智能体。严谨、简洁、先查后答。',
    model,
    injectMemory = true,
    writeMemory = true,
  } = {}) {
    if (!gateway || !tools) throw new Error('AgentRuntime 需要 gateway 与 tools');
    this.gateway = gateway;
    this.tools = tools;
    this.memory = memory;
    this.observe = observe;
    this.budget = { ...DEFAULT_BUDGET, ...budget };
    this.systemPrompt = systemPrompt;
    this.model = model;
    this.injectMemory = injectMemory && !!memory;
    this.writeMemory = writeMemory && !!memory;
  }

  /** 执行一个任务。返回 { answer, steps, trace, usage, costUsd, ms, finishReason }。 */
  async run({ task, sessionId = 'default', signal, metadata = {} } = {}) {
    const trace = new TraceRecorder();
    const started = Date.now();
    let usage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };
    let costUsd = 0;
    let toolCalls = 0;
    let steps = 0;
    let finishReason = 'stop';

    trace.record('task', { task, sessionId, ...metadata });

    // 状态快照：回放时据此把有状态工具（记忆等）恢复到运行前，
    // 保证「同初始状态 + 同录制响应 ⇒ 同决策链」严格成立。
    if (this.memory) trace.record('state_snapshot', this.memory.snapshot());

    const messages = [{ role: 'system', content: this.systemPrompt }];
    if (this.injectMemory) {
      const block = await this.memory.contextBlock(task, { k: 5 });
      if (block) messages.push({ role: 'system', content: block });
    }
    messages.push({ role: 'user', content: task });

    const budgetExceeded = (reason) => {
      finishReason = reason;
      trace.record('budget_exceeded', { reason, usage, steps });
    };

    for (;;) {
      if (steps >= this.budget.maxSteps) { budgetExceeded('maxSteps'); break; }
      if (Date.now() - started >= this.budget.maxMs) { budgetExceeded('maxMs'); break; }
      if (usage.totalTokens >= this.budget.maxTokens) { budgetExceeded('maxTokens'); break; }
      if (signal?.aborted) { finishReason = 'aborted'; trace.record('aborted', {}); break; }

      steps++;
      let resp;
      try {
        resp = await this.gateway.chat({
          model: this.model,
          messages,
          tools: this.tools.list(),
          signal,
        });
      } catch (err) {
        trace.record('llm_error', { message: String(err?.message || err) });
        finishReason = 'error';
        return this.#result({ answer: `推理失败：${err?.message || err}`, steps, trace, usage, costUsd, started, finishReason });
      }

      usage.promptTokens += resp.usage?.promptTokens || 0;
      usage.completionTokens += resp.usage?.completionTokens || 0;
      usage.totalTokens = usage.promptTokens + usage.completionTokens;
      costUsd += resp.costUsd || 0;
      trace.record('llm', { message: resp.message, model: resp.model, provider: resp.provider });

      const calls = resp.message.tool_calls || [];
      if (calls.length === 0) {
        trace.record('answer', { content: resp.message.content || '' });
        if (this.memory && this.writeMemory) {
          await this.memory.remember(task, { salience: 0.6, kind: 'task', meta: { answer: (resp.message.content || '').slice(0, 400) } });
        }
        return this.#result({ answer: resp.message.content || '', steps, trace, usage, costUsd, started, finishReason });
      }

      // 追加 assistant 消息（OpenAI 格式：tool_calls 挂在 assistant 上）
      messages.push({
        role: 'assistant',
        content: resp.message.content || '',
        tool_calls: calls.map((tc) => ({ id: tc.id, type: 'function', function: tc.function })),
      });

      for (const tc of calls) {
        if (toolCalls >= this.budget.maxToolCalls) { budgetExceeded('maxToolCalls'); break; }
        toolCalls++;
        let args = {};
        try {
          args = JSON.parse(tc.function.arguments || '{}');
        } catch {
          args = { __raw: tc.function.arguments };
        }
        trace.record('tool_call', { name: tc.function.name, args, id: tc.id });
        const result = await this.tools.execute(tc.function.name, args, { signal });
        trace.record('tool_result', { name: tc.function.name, ok: result.ok, output: result.ok ? result.output : undefined, error: result.error, ms: result.ms });
        messages.push({
          role: 'tool',
          tool_call_id: tc.id,
          name: tc.function.name,
          content: result.ok ? JSON.stringify(result.output) : JSON.stringify({ error: result.error }),
        });
      }
    }

    const answer =
      finishReason === 'stop'
        ? ''
        : finishReason === 'aborted'
          ? '（已取消）'
          : `（预算耗尽：${finishReason}。已用步数 ${steps}，token ${usage.totalTokens}。请调大 budget 或拆分任务。）`;
    trace.record('answer', { content: answer });
    return this.#result({ answer, steps, trace, usage, costUsd, started, finishReason });
  }

  #result({ answer, steps, trace, usage, costUsd, started, finishReason }) {
    const ms = Date.now() - started;
    this.observe?.metrics?.timing?.('agent.run.ms', ms);
    return { answer, steps, trace, usage, costUsd, ms, finishReason };
  }
}

/**
 * 确定性回放：用轨迹中录制的 LLM 响应重放整个智能体决策。
 * 回放前按轨迹内 state_snapshot 把有状态工具恢复到运行前状态，
 * 回放后恢复调用方现场（对调用方的记忆零侵扰）。
 * 要求：非记忆类工具是确定性的（calc 等），否则工具结果天然可能不同。
 */
export async function replayAgent(traceData, { tools, budget, systemPrompt, memory: memoryOpt } = {}) {
  const events = traceData.events || traceData;
  const llmQueue = events.filter((e) => e.type === 'llm').map((e) => e.payload.message);
  const taskEvent = events.find((e) => e.type === 'task');
  if (!taskEvent) throw new Error('回放失败：轨迹缺少 task 事件');
  if (llmQueue.length === 0) throw new Error('回放失败：轨迹没有 LLM 事件');

  const replayProvider = {
    name: 'replay',
    async chat() {
      const msg = llmQueue.shift();
      if (!msg) throw new Error('回放耗尽：实际 LLM 调用次数超过录制');
      return {
        message: msg,
        usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
        finishReason: msg.tool_calls?.length ? 'tool_calls' : 'stop',
      };
    },
  };
  const gateway = new ModelGateway();
  gateway.register(replayProvider, { isDefault: true });

  // 状态隔离：恢复到轨迹录制时的初始状态，回放后还原调用方现场
  const memory = memoryOpt ?? tools?.context?.memory ?? null;
  const snapEvent = events.find((e) => e.type === 'state_snapshot');
  const backup = memory ? memory.snapshot() : null;
  if (memory && snapEvent) memory.restore(snapEvent.payload);

  const agent = new AgentRuntime({
    gateway,
    tools,
    memory, // 仅用于录制 state_snapshot；不回写、不注入（见下方两个开关）
    budget,
    systemPrompt,
    injectMemory: false,
    writeMemory: false,
  });
  try {
    const result = await agent.run({ task: taskEvent.payload.task });
    return { result, consumed: llmQueue.length };
  } finally {
    if (memory && backup) memory.restore(backup);
  }
}

/** 比对两次运行的结构化轨迹是否一致（忽略时间戳）。 */
export function tracesEqual(a, b) {
  const sa = a.structural ? a.structural() : a.events;
  const sb = b.structural ? b.structural() : b.events;
  if (sa.length !== sb.length) return false;
  for (let i = 0; i < sa.length; i++) {
    if (sa[i].type !== sb[i].type) return false;
    if (JSON.stringify(sa[i].payload) !== JSON.stringify(sb[i].payload)) return false;
  }
  return true;
}
