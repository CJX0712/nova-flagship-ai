// L8 智能体：ReAct 闭环 / 预算 / 取消 / 确定性回放
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createNova, replayAgent, tracesEqual } from '../src/nova.js';
import { ToolRegistry } from '../src/core/tools.js';
import { AgentRuntime } from '../src/core/agent.js';

function buildAgent(agentOpts = {}) {
  const nova = createNova({ allowHttp: false, fsRoots: [], agent: agentOpts });
  return nova;
}

test('ReAct 闭环：calc 工具 + novaLite 两跳终答', async () => {
  const nova = buildAgent();
  const r = await nova.agent.run({ task: '计算 (3+4)*2^10' });
  assert.equal(r.finishReason, 'stop');
  assert.match(r.answer, /7168/);
  const called = r.trace.events.filter((e) => e.type === 'tool_call').map((e) => e.payload.name);
  assert.deepEqual(called, ['calc']);
  assert.ok(r.steps >= 2);
  assert.ok(r.usage.totalTokens > 0);
});

test('显式工具指令 + 记忆召回指令', async () => {
  const nova = buildAgent();
  const r1 = await nova.agent.run({ task: '@tool calc {"expression":"6*7"}' });
  assert.match(r1.answer, /42/);
  const r2 = await nova.agent.run({ task: '@recall 旗舰' });
  const called = r2.trace.events.filter((e) => e.type === 'tool_call').map((e) => e.payload.name);
  assert.deepEqual(called, ['memory_search']);
});

test('预算：maxSteps=1 必然预算耗尽', async () => {
  const nova = buildAgent({ budget: { maxSteps: 1 } });
  const r = await nova.agent.run({ task: '计算 (3+4)*2^10' });
  assert.equal(r.finishReason, 'maxSteps');
  assert.match(r.answer, /预算耗尽/);
  assert.equal(r.steps, 1);
});

test('预算：maxTokens 极小即停', async () => {
  const nova = buildAgent({ budget: { maxTokens: 5 } });
  const r = await nova.agent.run({ task: '计算 (3+4)*2^10' });
  assert.equal(r.finishReason, 'maxTokens');
});

test('取消：abort 立即结束', async () => {
  const nova = buildAgent();
  const ctrl = new AbortController();
  ctrl.abort();
  const r = await nova.agent.run({ task: '计算 (3+4)*2^10', signal: ctrl.signal });
  assert.equal(r.finishReason, 'aborted');
  assert.equal(r.answer, '（已取消）');
});

test('确定性：同任务两次运行轨迹一致（离线栈）', async () => {
  const nova = buildAgent();
  const r1 = await nova.agent.run({ task: '计算 (3+4)*2^10' });
  const nova2 = buildAgent();
  const r2 = await nova2.agent.run({ task: '计算 (3+4)*2^10' });
  assert.ok(tracesEqual(r1.trace, r2.trace));
  assert.equal(r1.answer, r2.answer);
});

test('确定性回放：replayAgent 重放决策链并逐条相等', async () => {
  const nova = buildAgent();
  const r = await nova.agent.run({ task: '计算 (3+4)*2^10' });
  const { result, consumed } = await replayAgent(r.trace.toJSON(), {
    tools: nova.tools,
    budget: nova.agent.budget,
    systemPrompt: nova.agent.systemPrompt,
  });
  assert.equal(consumed, 0); // 录制响应被完整消耗
  assert.ok(tracesEqual(r.trace, result.trace));
  assert.equal(result.answer, r.answer);
});

test('回放比对：长度不同即不相等', () => {
  const t1 = { structural: () => [{ seq: 0, type: 'a', payload: {} }] };
  const t2 = { structural: () => [{ seq: 0, type: 'a', payload: {} }, { seq: 1, type: 'b', payload: {} }] };
  assert.equal(tracesEqual(t1, t2), false);
});

test('回放隔离：回放不污染调用方记忆（快照恢复）', async () => {
  const nova = buildAgent();
  const before = JSON.stringify(nova.memory.stats());
  const r = await nova.agent.run({ task: '计算 (3+4)*2^10' });
  const afterRun = JSON.stringify(nova.memory.stats());
  assert.notEqual(before, afterRun); // 原运行回写了任务记忆
  await replayAgent(r.trace.toJSON(), {
    tools: nova.tools,
    budget: nova.agent.budget,
    systemPrompt: nova.agent.systemPrompt,
  });
  assert.equal(JSON.stringify(nova.memory.stats()), afterRun); // 回放后原样恢复
});

test('轨迹记录器：JSONL 往返', async () => {
  const nova = buildAgent();
  const r = await nova.agent.run({ task: '你好' });
  const jsonl = r.trace.toJSONL();
  const lines = jsonl.trim().split('\n').map((l) => JSON.parse(l));
  assert.equal(lines.length, r.trace.events.length);
  assert.equal(lines[0].type, 'task');
  assert.equal(lines.at(-1).type, 'answer');
});

test('记忆注入：任务完成后写入情节记忆', async () => {
  const nova = buildAgent();
  await nova.agent.run({ task: '你好' });
  assert.ok(nova.memory.episodic.size >= 1);
});

test('AgentRuntime：缺参数报错', () => {
  assert.throws(() => new AgentRuntime({ tools: new ToolRegistry() }), /gateway 与 tools/);
  assert.throws(() => new AgentRuntime({ gateway: {} }), /gateway 与 tools/);
});
