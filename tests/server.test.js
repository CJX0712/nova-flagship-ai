// L10 服务层：零依赖 HTTP API 冒烟（OpenAI 兼容 + 智能体 + 记忆 + 静态驾驶舱）
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createNova } from '../src/nova.js';

let handle;
test.before(async () => {
  const nova = createNova({ allowHttp: false, fsRoots: [] });
  await nova.memory.remember('驾驶舱测试记忆：NovaFlagship 支持 SSE 流式');
  const srv = await nova.serve({ port: 0, host: '127.0.0.1' });
  handle = { nova, srv, base: srv.url };
});
test.after(async () => {
  await new Promise((r) => handle.srv.close(r));
});

test('GET /api/health：拓扑', async () => {
  const h = await (await fetch(`${handle.base}/api/health`)).json();
  assert.equal(h.ok, true);
  assert.equal(h.version, '1.0.0');
  assert.ok(h.providers.includes('nova-lite'));
  assert.ok(h.tools.includes('calc'));
  assert.ok(h.ops > 0);
});

test('POST /v1/chat/completions：OpenAI 兼容形状', async () => {
  const res = await fetch(`${handle.base}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages: [{ role: 'user', content: '你好' }] }),
  });
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.object, 'chat.completion');
  assert.ok(body.choices[0].message.content.includes('Nova-Lite'));
  assert.ok(body.usage.total_tokens > 0);
  assert.equal(body.nova.provider, 'nova-lite');
});

test('POST /v1/chat/completions：stream=SSE', async () => {
  const res = await fetch(`${handle.base}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages: [{ role: 'user', content: '你好' }], stream: true }),
  });
  assert.match(res.headers.get('content-type'), /text\/event-stream/);
  const text = await res.text();
  assert.match(text, /^data: /m);
  assert.match(text, /data: \[DONE\]/);
});

test('POST /v1/agents：智能体运行 + 轨迹', async () => {
  const res = await fetch(`${handle.base}/v1/agents`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ task: '计算 (3+4)*2^10' }),
  });
  const body = await res.json();
  assert.match(body.answer, /7168/);
  assert.equal(body.finishReason, 'stop');
  const trace = await (await fetch(`${handle.base}/v1/agents/${body.id}/trace`)).json();
  assert.ok(trace.events.some((e) => e.type === 'tool_call'));
  assert.equal((await (await fetch(`${handle.base}/v1/agents/nope/trace`)).json()).error !== undefined, true);
});

test('POST /v1/agents?stream=1：SSE 事件流', async () => {
  const res = await fetch(`${handle.base}/v1/agents?stream=1`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ task: '你好' }),
  });
  const text = await res.text();
  assert.match(text, /"type":"trace"/);
  assert.match(text, /"type":"result"/);
});

test('记忆 API：recall / remember', async () => {
  const recall = await (await fetch(`${handle.base}/v1/memory/recall`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ query: 'SSE 流式', k: 3 }),
  })).json();
  assert.ok(recall.hits.length >= 1);
  const remember = await (await fetch(`${handle.base}/v1/memory/remember`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text: '服务层测试：写入一条记忆' }),
  })).json();
  assert.ok(remember.id);
});

test('RAG 上下文 API', async () => {
  const ctx = await (await fetch(`${handle.base}/v1/rag/context`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ query: '记忆', k: 3 }),
  })).json();
  assert.ok(typeof ctx.context === 'string');
  assert.ok(Array.isArray(ctx.sources));
});

test('GET /api/ops + /api/metrics + 静态驾驶舱', async () => {
  const ops = await (await fetch(`${handle.base}/api/ops`)).json();
  assert.ok(ops.ops.length >= 30);
  const metrics = await (await fetch(`${handle.base}/api/metrics`)).json();
  assert.ok(metrics.counters !== undefined);
  const html = await (await fetch(`${handle.base}/`)).text();
  assert.match(html, /NovaFlagship/);
  assert.match(html, /驾驶舱/);
  const notFound = await fetch(`${handle.base}/nope`);
  assert.equal(notFound.status, 404);
});

test('错误处理：缺 task / 缺 messages', async () => {
  const r1 = await fetch(`${handle.base}/v1/agents`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' });
  assert.equal(r1.status, 400);
  const r2 = await fetch(`${handle.base}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' });
  assert.equal(r2.status, 400);
});
