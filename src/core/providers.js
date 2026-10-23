/**
 * NovaFlagship · 提供商层
 * ------------------------------------------------------------------
 * 1) openaiCompatible —— 世界事实标准协议客户端（OpenAI / DeepSeek / Moonshot /
 *    llama.cpp / Ollama / vLLM / SGLang 全部兼容），含 SSE 流式与 tool_calls 增量装配。
 * 2) novaLite —— 确定性离线推理模拟器。不伪装成语言模型：它是一个可复现的
 *    「策略函数」，让整条智能体链路（规划→工具→观察→终答）在无网络、无 GPU、
 *    无 API Key 的环境下完整跑通并被测试断言。真实模型随时可原位替换。
 * 3) mockProvider —— 脚本化响应，供单元测试精确控制对话流。
 */

import { ProviderError } from './gateway.js';
import { estimateMessagesTokens } from './tokenize.js';

/** 把内部 tools 定义转换为 OpenAI function calling 格式。 */
export function toOpenAITools(tools = []) {
  if (!tools.length) return undefined;
  return tools.map((t) => ({
    type: 'function',
    function: {
      name: t.name,
      description: t.description || '',
      parameters: t.inputSchema || t.parameters || { type: 'object', properties: {} },
    },
  }));
}

/** OpenAI chat.completions 响应 ⇒ 内部标准化 Completion。 */
function normalizeCompletion(json, model, providerName) {
  const choice = json.choices?.[0] || {};
  const msg = choice.message || {};
  const toolCalls = (msg.tool_calls || []).map((tc, i) => ({
    id: tc.id || `call_${i}_${Math.random().toString(36).slice(2, 8)}`,
    type: 'function',
    function: { name: tc.function?.name || '', arguments: tc.function?.arguments || '{}' },
  }));
  const usage = json.usage
    ? {
        promptTokens: json.usage.prompt_tokens || 0,
        completionTokens: json.usage.completion_tokens || 0,
        totalTokens: json.usage.total_tokens || 0,
      }
    : null;
  return {
    id: json.id,
    model: json.model || model,
    provider: providerName,
    message: {
      role: 'assistant',
      content: msg.content ?? '',
      ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
    },
    usage,
    finishReason: choice.finish_reason || 'stop',
    raw: json,
  };
}

/**
 * OpenAI 兼容提供商。
 * @param {object} opts
 * @param {string} opts.baseURL 例如 https://api.openai.com/v1 或 http://127.0.0.1:8080/v1
 * @param {string} [opts.apiKey]
 * @param {number} [opts.timeoutMs=60000]
 */
export function openaiCompatible({
  name = 'openai-compatible',
  baseURL,
  apiKey = '',
  timeoutMs = 60000,
  fetchImpl = fetch,
  headers = {},
} = {}) {
  if (!baseURL) throw new Error('openaiCompatible 需要 baseURL');
  const url = `${baseURL.replace(/\/+$/, '')}/chat/completions`;

  async function call(request, ctx) {
    const body = {
      model: request.model,
      messages: request.messages,
      ...(request.tools?.length ? { tools: toOpenAITools(request.tools) } : {}),
      ...(request.temperature !== undefined ? { temperature: request.temperature } : {}),
      ...(request.maxTokens ? { max_tokens: request.maxTokens } : {}),
      stream: false,
    };
    const signals = [ctx?.signal, AbortSignal.timeout(timeoutMs)].filter(Boolean);
    const res = await fetchImpl(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...(apiKey ? { authorization: `Bearer ${apiKey}` } : {}),
        ...headers,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.any(signals),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new ProviderError(`HTTP ${res.status}: ${text.slice(0, 300)}`, {
        provider: name,
        status: res.status,
        retryable: [408, 429, 500, 502, 503, 504].includes(res.status),
      });
    }
    const json = await res.json();
    const out = normalizeCompletion(json, request.model, name);
    out.usage =
      out.usage || { promptTokens: estimateMessagesTokens(request.messages), completionTokens: 0, totalTokens: 0 };
    return out;
  }

  async function* callStream(request, ctx) {
    const body = {
      model: request.model,
      messages: request.messages,
      ...(request.tools?.length ? { tools: toOpenAITools(request.tools) } : {}),
      ...(request.temperature !== undefined ? { temperature: request.temperature } : {}),
      ...(request.maxTokens ? { max_tokens: request.maxTokens } : {}),
      stream: true,
      stream_options: { include_usage: true },
    };
    const signals = [ctx?.signal, AbortSignal.timeout(timeoutMs)].filter(Boolean);
    const res = await fetchImpl(url, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...(apiKey ? { authorization: `Bearer ${apiKey}` } : {}),
        ...headers,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.any(signals),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new ProviderError(`HTTP ${res.status}: ${text.slice(0, 300)}`, {
        provider: name,
        status: res.status,
        retryable: [408, 429, 500, 502, 503, 504].includes(res.status),
      });
    }

    const decoder = new TextDecoder();
    let buf = '';
    let content = '';
    const toolBuf = new Map(); // index -> {id,name,args}
    let usage = null;
    let finishReason = 'stop';
    let responseId = null;

    const flushLine = function* (line) {
      if (!line || line.startsWith(':')) return;
      if (!line.startsWith('data:')) return;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') return;
      let json;
      try {
        json = JSON.parse(payload);
      } catch {
        return; // 容忍半包/注释
      }
      responseId = json.id || responseId;
      if (json.usage) usage = { promptTokens: json.usage.prompt_tokens || 0, completionTokens: json.usage.completion_tokens || 0, totalTokens: json.usage.total_tokens || 0 };
      const choice = json.choices?.[0];
      if (!choice) return;
      if (choice.finish_reason) finishReason = choice.finish_reason;
      const delta = choice.delta || {};
      if (delta.content) {
        content += delta.content;
        yield { type: 'content', text: delta.content };
      }
      for (const tc of delta.tool_calls || []) {
        const i = tc.index ?? 0;
        const cur = toolBuf.get(i) || { id: null, name: '', args: '' };
        if (tc.id) cur.id = tc.id;
        if (tc.function?.name) cur.name += tc.function.name;
        if (tc.function?.arguments) cur.args += tc.function.arguments;
        toolBuf.set(i, cur);
      }
    };

    for await (const chunk of res.body) {
      buf += decoder.decode(chunk, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).replace(/\r$/, '');
        buf = buf.slice(nl + 1);
        yield* flushLine(line);
      }
    }
    if (buf.trim()) yield* flushLine(buf.replace(/\r$/, ''));

    const toolCalls = [...toolBuf.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([i, t]) => ({
        id: t.id || `call_${i}_${Math.random().toString(36).slice(2, 8)}`,
        type: 'function',
        function: { name: t.name, arguments: t.args || '{}' },
      }));
    yield {
      type: 'done',
      response: {
        id: responseId,
        model: request.model,
        provider: name,
        message: { role: 'assistant', content, ...(toolCalls.length ? { tool_calls: toolCalls } : {}) },
        usage: usage || { promptTokens: estimateMessagesTokens(request.messages), completionTokens: estimateTokens(content), totalTokens: 0 },
        finishReason,
      },
    };
  }

  return { name, chat: call, chatStream: callStream };
}

/** llama.cpp server 预设（默认 http://127.0.0.1:8080/v1）。 */
export const llamaCppProvider = (opts = {}) =>
  openaiCompatible({ name: 'llama.cpp', baseURL: 'http://127.0.0.1:8080/v1', ...opts });

/** Ollama 预设（OpenAI 兼容端点 /v1）。 */
export const ollamaProvider = (opts = {}) =>
  openaiCompatible({ name: 'ollama', baseURL: 'http://127.0.0.1:11434/v1', ...opts });

/* ------------------------------------------------------------------ */
/* Nova-Lite：确定性离线推理模拟器                                      */
/* ------------------------------------------------------------------ */

const FNV_OFFSET = 0x811c9dc5;
function fnv1a(str) {
  let h = FNV_OFFSET;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function lastUserText(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role === 'user' && typeof m.content === 'string') return m.content;
  }
  return '';
}

function countMemoryHits(messages) {
  const sys = messages.find((m) => m.role === 'system')?.content || '';
  const m = /\[记忆\][\s\S]*?共\s*(\d+)\s*条/.exec(sys);
  return m ? Number(m[1]) : 0;
}

/**
 * 确定性「推理」提供者。决策规则（按优先级）：
 *   R1 显式指令 `@tool <name> {json}`        ⇒ 调用指定工具
 *   R2 上条消息是工具观察结果                 ⇒ 生成引用工具输出的终答（闭环 ReAct）
 *   R3 含算术表达式且 calc 工具可用           ⇒ 调用 calc
 *   R4 记忆召回指令 `@recall <query>`         ⇒ 调用 memory_search
 *   R5 默认                                  ⇒ 确定性模板应答
 * 同一输入 ⇒ 同一输出（可回放、可断言）。它是测试与离线演示的锚，不是语言模型。
 */
export function novaLite({ name = 'nova-lite', seed = 42 } = {}) {
  function decide(request) {
    const messages = request.messages || [];
    const tools = request.tools || [];
    const hasTool = (n) => tools.some((t) => t.name === n);
    const text = lastUserText(messages);
    const last = messages[messages.length - 1];

    // R2：工具观察后 ⇒ 终答
    if (last?.role === 'tool') {
      const obs = messages.filter((m) => m.role === 'tool').slice(-4);
      const lines = obs.map((o) => {
        let out = o.content;
        try {
          out = JSON.stringify(JSON.parse(o.content));
        } catch { /* 原样 */ }
        return `- ${o.name || 'tool'} → ${String(out).slice(0, 240)}`;
      });
      return {
        message: {
          role: 'assistant',
          content: [
            '【Nova-Lite 终答】基于工具观察结果：',
            ...lines,
            '',
            `结论：已基于 ${obs.length} 条工具返回完成推理。（确定性离线提供商；接入真实模型后此段由模型生成）`,
          ].join('\n'),
        },
        finishReason: 'stop',
      };
    }

    // R1：显式 @tool 指令
    const explicit = /@tool\s+([\w.-]+)\s*(\{[\s\S]*\})?/.exec(text);
    if (explicit && hasTool(explicit[1])) {
      let args = {};
      if (explicit[2]) {
        try {
          args = JSON.parse(explicit[2]);
        } catch { /* 留空由 schema 校验报错 */ }
      }
      return {
        message: {
          role: 'assistant',
          content: '',
          tool_calls: [
            { id: `nl_${fnv1a(text).toString(36)}`, type: 'function', function: { name: explicit[1], arguments: JSON.stringify(args) } },
          ],
        },
        finishReason: 'tool_calls',
      };
    }

    // R3：算术表达式 ⇒ calc
    if (hasTool('calc') && /(?:\d|\))\s*[+\-*/^%]\s*(?:\d|\()/.test(text) && !/[?？]$/.test(text.trim())) {
      const expr = text.match(/-?[\d\s()+*/^%.]+/)?.[0]?.trim() || text.trim();
      return {
        message: {
          role: 'assistant',
          content: '',
          tool_calls: [
            { id: `nl_calc_${fnv1a(expr).toString(36)}`, type: 'function', function: { name: 'calc', arguments: JSON.stringify({ expression: expr }) } },
          ],
        },
        finishReason: 'tool_calls',
      };
    }

    // R4：记忆召回指令
    if (hasTool('memory_search') && /@recall\s+(.+)/.exec(text)) {
      const q = /@recall\s+(.+)/.exec(text)[1].trim();
      return {
        message: {
          role: 'assistant',
          content: '',
          tool_calls: [
            { id: `nl_recall_${fnv1a(q).toString(36)}`, type: 'function', function: { name: 'memory_search', arguments: JSON.stringify({ query: q, k: 5 }) } },
          ],
        },
        finishReason: 'tool_calls',
      };
    }

    // R5：确定性模板应答
    const hits = countMemoryHits(messages);
    const toolNames = tools.map((t) => t.name).join(', ') || '（无）';
    const variant = fnv1a(text + '|' + tools.length) % 3;
    const openers = ['已接收任务', '任务已进入推理管线', '推理请求已受理'];
    return {
      message: {
        role: 'assistant',
        content: [
          `【Nova-Lite】${openers[variant]}：「${text.slice(0, 120)}」`,
          `记忆命中 ${hits} 条；可用工具：${toolNames}。`,
          '本回复由确定性模拟提供商生成，用于离线验证与回归测试；接入真实模型后，此处为模型生成内容。',
        ].join('\n'),
      },
      finishReason: 'stop',
    };
  }

  return {
    name,
    kind: 'deterministic-simulator',
    async chat(request) {
      const out = decide(request);
      const promptTokens = estimateMessagesTokens(request.messages);
      const completionTokens = estimateTokens(out.message.content || '');
      return {
        ...out,
        usage: { promptTokens, completionTokens, totalTokens: promptTokens + completionTokens },
      };
    },
    async *chatStream(request) {
      const out = decide(request);
      const text = out.message.content || '';
      for (let i = 0; i < text.length; i += 12) {
        yield { type: 'content', text: text.slice(i, i + 12) };
      }
      for (const tc of out.message.tool_calls || []) yield { type: 'tool_call', toolCall: tc };
      yield { type: 'done', response: out };
    },
  };
}

/** 测试用脚本化提供商：按队列返回预置响应，并记录全部请求。 */
export function mockProvider({ name = 'mock' } = {}) {
  const queue = [];
  const calls = [];
  return {
    name,
    kind: 'mock',
    enqueue(response) {
      queue.push(response);
      return this;
    },
    get pending() {
      return queue.length;
    },
    calls,
    async chat(request) {
      calls.push(request);
      const next = queue.shift();
      if (!next) throw new Error('mockProvider: 响应队列已空');
      return typeof next === 'function' ? next(request) : next;
    },
  };
}

function estimateTokens(text) {
  let cjk = 0;
  for (const ch of text) if (/[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/.test(ch)) cjk++;
  return cjk + Math.ceil((text.length - cjk) / 4);
}

/** 从环境变量构建真实提供商（NOVA_PROVIDER / NOVA_BASE_URL / NOVA_API_KEY / NOVA_MODEL）。 */
export function providerFromEnv(env = process.env) {
  const provider = env.NOVA_PROVIDER || 'nova-lite';
  if (provider === 'nova-lite') return novaLite();
  return openaiCompatible({
    name: provider,
    baseURL: env.NOVA_BASE_URL,
    apiKey: env.NOVA_API_KEY || '',
  });
}
