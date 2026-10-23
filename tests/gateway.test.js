// L4 模型网关：重试 / 熔断 / 账本 / OpenAI 兼容 / 流式 / 确定性模拟器
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ModelGateway, ProviderError, retryWithBackoff, CircuitBreaker, CostLedger } from '../src/core/gateway.js';
import { openaiCompatible, novaLite, mockProvider } from '../src/core/providers.js';

test('retryWithBackoff：Full Jitter 边界与次数上限', async () => {
  const delays = [];
  let attempts = 0;
  const out = await retryWithBackoff(
    () => {
      attempts++;
      if (attempts < 3) throw new ProviderError('boom', { retryable: true });
      return 'ok';
    },
    {
      retries: 3,
      baseMs: 100,
      maxMs: 400,
      rng: () => 0.99, // 固定随机 ⇒ 可断言
      sleep: (ms) => {
        delays.push(ms);
      },
    },
  );
  assert.equal(out, 'ok');
  assert.equal(attempts, 3);
  assert.deepEqual(delays, [99, 198]); // cap=min(400, 100*2^n)
  for (const d of delays) assert.ok(d >= 0 && d <= 400);
});

test('retryWithBackoff：耗尽抛错 / abort 立即拒绝', async () => {
  await assert.rejects(
    () => retryWithBackoff(() => { throw new ProviderError('x', { retryable: true }); }, { retries: 2, sleep: () => {} }),
    ProviderError,
  );
  const ctrl = new AbortController();
  ctrl.abort();
  await assert.rejects(
    () =>
      retryWithBackoff(
        () => {
          throw new ProviderError('x', { retryable: true });
        },
        { retries: 5, signal: ctrl.signal, sleep: () => {} },
      ),
    ProviderError,
  );
});

test('CircuitBreaker：closed → open → half-open → closed', () => {
  let now = 0;
  const cb = new CircuitBreaker({ failureThreshold: 3, cooldownMs: 1000, now: () => now });
  assert.equal(cb.canRequest(), true);
  cb.onFailure(); cb.onFailure();
  assert.equal(cb.state, 'closed');
  cb.onFailure();
  assert.equal(cb.state, 'open');
  assert.equal(cb.canRequest(), false);
  now = 1001;
  assert.equal(cb.canRequest(), true); // 半开探测
  assert.equal(cb.state, 'half-open');
  assert.equal(cb.canRequest(), false); // 半开只放 1 个
  cb.onSuccess();
  assert.equal(cb.state, 'closed');
});

test('CostLedger：总额 = Σ in·pin + out·pout', () => {
  const ledger = new CostLedger({ pricing: { 'gpt-x': { in: 0.01, out: 0.03 } } });
  ledger.record({ model: 'gpt-x', promptTokens: 1000, completionTokens: 500 }); // 0.01+0.015=0.025
  ledger.record({ model: 'gpt-x', promptTokens: 2000, completionTokens: 0 }); // 0.02
  assert.ok(Math.abs(ledger.total() - 0.045) < 1e-12);
  const by = ledger.totalsByModel().get('gpt-x');
  assert.equal(by.calls, 2);
  assert.equal(by.promptTokens, 3000);
});

function fakeCompletion(overrides = {}) {
  return {
    ok: true,
    status: 200,
    json: async () => ({
      id: 'cmpl-1',
      model: 'test-model',
      choices: [{ index: 0, message: { role: 'assistant', content: '你好，世界' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 11, completion_tokens: 5, total_tokens: 16 },
      ...overrides,
    }),
  };
}

test('openaiCompatible：非流式归一化', async () => {
  let captured;
  const provider = openaiCompatible({
    baseURL: 'https://api.test/v1',
    apiKey: 'k',
    fetchImpl: async (url, init) => {
      captured = { url, init };
      return fakeCompletion();
    },
  });
  const out = await provider.chat({ model: 'm', messages: [{ role: 'user', content: 'hi' }] });
  assert.equal(captured.url, 'https://api.test/v1/chat/completions');
  assert.equal(captured.init.headers.authorization, 'Bearer k');
  assert.equal(out.message.content, '你好，世界');
  assert.equal(out.usage.totalTokens, 16);
  assert.equal(out.provider, 'openai-compatible');
});

test('openaiCompatible：错误码可重试性', async () => {
  for (const [status, retryable] of [[429, true], [500, true], [400, false], [401, false]]) {
    const provider = openaiCompatible({
      baseURL: 'https://api.test/v1',
      fetchImpl: async () => ({ ok: false, status, text: async () => 'err' }),
    });
    await assert.rejects(() => provider.chat({ model: 'm', messages: [] }), (err) => {
      assert.ok(err instanceof ProviderError);
      assert.equal(err.status, status);
      assert.equal(err.retryable, retryable);
      return true;
    });
  }
});

function sseStream(lines) {
  const enc = new TextEncoder();
  return {
    ok: true,
    status: 200,
    body: (async function* () {
      for (const l of lines) yield enc.encode(l);
    })(),
  };
}

test('openaiCompatible：SSE 流式 + tool_calls 增量装配', async () => {
  const lines = [
    'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"让"}}]}\n\n',
    'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"我算"}}]}\n\n',
    'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"calc","arguments":""}}]}}]}\n\n',
    'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"expression\\""}}]}}]}\n\n',
    'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"2+3\\"}"}}]}}]}\n\n',
    'data: {"id":"c1","choices":[{"index":0,"finish_reason":"tool_calls"}]}\n\n',
    'data: {"id":"c1","usage":{"prompt_tokens":8,"completion_tokens":9,"total_tokens":17}}\n\n',
    'data: [DONE]\n\n',
  ];
  const provider = openaiCompatible({
    baseURL: 'https://api.test/v1',
    fetchImpl: async () => sseStream(lines),
  });
  const events = [];
  let final;
  for await (const ev of provider.chatStream({ model: 'm', messages: [] })) {
    events.push(ev);
    if (ev.type === 'done') final = ev.response;
  }
  assert.deepEqual(events.filter((e) => e.type === 'content').map((e) => e.text), ['让', '我算']);
  assert.equal(final.message.tool_calls.length, 1);
  assert.equal(final.message.tool_calls[0].function.name, 'calc');
  assert.equal(final.message.tool_calls[0].function.arguments, '{"expression":"2+3"}');
  assert.equal(final.finishReason, 'tool_calls');
  assert.equal(final.usage.totalTokens, 17);
});

test('novaLite：确定性 + 三条决策规则', async () => {
  const p = novaLite();
  const req = { model: 'nova-lite', messages: [{ role: 'user', content: '你好' }] };
  const a = await p.chat(req);
  const b = await p.chat(req);
  assert.equal(a.message.content, b.message.content); // 同输入同输出

  // R1 显式工具
  const r1 = await p.chat({ messages: [{ role: 'user', content: '@tool calc {"expression":"1+1"}' }], tools: [{ name: 'calc' }] });
  assert.equal(r1.message.tool_calls[0].function.name, 'calc');
  assert.equal(r1.finishReason, 'tool_calls');

  // R2 工具观察后 ⇒ 终答
  const r2 = await p.chat({
    messages: [
      { role: 'user', content: '算一下' },
      { role: 'assistant', content: '', tool_calls: [{ id: 'c1', function: { name: 'calc', arguments: '{}' } }] },
      { role: 'tool', tool_call_id: 'c1', name: 'calc', content: '{"value":4}' },
    ],
    tools: [{ name: 'calc' }],
  });
  assert.equal(r2.finishReason, 'stop');
  assert.match(r2.message.content, /Nova-Lite 终答/);
  assert.match(r2.message.content, /calc/);

  // R3 算术触发
  const r3 = await p.chat({ messages: [{ role: 'user', content: '计算 6*7 等于几' }], tools: [{ name: 'calc' }] });
  assert.equal(r3.message.tool_calls?.[0].function.name, 'calc');
  assert.equal(r3.usage.totalTokens, r3.usage.promptTokens + r3.usage.completionTokens);
});

test('mockProvider：脚本队列与调用记录', async () => {
  const m = mockProvider();
  m.enqueue({ message: { role: 'assistant', content: '一' }, usage: {} });
  m.enqueue({ message: { role: 'assistant', content: '二' }, usage: {} });
  assert.equal((await m.chat({ messages: [] })).message.content, '一');
  assert.equal((await m.chat({ messages: [] })).message.content, '二');
  assert.equal(m.pending, 0);
  assert.equal(m.calls.length, 2);
  await assert.rejects(() => m.chat({ messages: [] }), /队列已空/);
});

test('ModelGateway：路由 provider:model 与流式退化', async () => {
  const gw = new ModelGateway();
  gw.register(mockProvider().enqueue({ message: { role: 'assistant', content: 'x' }, usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 } }), { isDefault: true });
  const out = await gw.chat({ model: 'mock:m1', messages: [{ role: 'user', content: 'hi' }] });
  assert.equal(out.message.content, 'x');
  assert.equal(out.model, 'm1');
  assert.equal(out.provider, 'mock');

  const gw2 = new ModelGateway();
  gw2.register(novaLite(), { isDefault: true });
  const events = [];
  for await (const ev of gw2.chatStream({ messages: [{ role: 'user', content: '你好' }] })) events.push(ev.type);
  assert.ok(events.includes('content'));
  assert.ok(events.includes('done'));
});
