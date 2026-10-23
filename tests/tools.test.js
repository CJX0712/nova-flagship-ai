// L7 工具层：表达式求值 / 注册表 / 内置工具 / MCP 客户端
import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ToolRegistry, registerBuiltinTools, McpClient, evalExpression } from '../src/core/tools.js';

test('evalExpression：优先级 / 右结合 / 函数 / 错误', () => {
  assert.equal(evalExpression('2+3*4'), 14);
  assert.equal(evalExpression('2^3^2'), 512); // 右结合
  assert.equal(evalExpression('-(3+4)'), -7);
  assert.equal(evalExpression('(1+2)*(3+4)'), 21);
  assert.equal(evalExpression('sqrt(16)'), 4);
  assert.equal(evalExpression('max(1,5,3)'), 5);
  assert.equal(evalExpression('2*pi'), Math.PI * 2);
  assert.equal(evalExpression('10 % 3'), 1);
  assert.throws(() => evalExpression('1/0'), /除零/);
  assert.throws(() => evalExpression('foo(1)'), /未知函数/);
  assert.throws(() => evalExpression('1+'), /意外/);
  assert.throws(() => evalExpression('1 $ 2'), /非法字符/);
  assert.throws(() => evalExpression('(1+2'), /右括号/);
});

test('ToolRegistry：未知工具 / 入参校验 / 正常执行', async () => {
  const reg = new ToolRegistry();
  reg.register({
    name: 'echo',
    description: '回显',
    inputSchema: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] },
    handler: ({ text }) => ({ echo: text }),
  });
  const bad = await reg.execute('nope', {});
  assert.equal(bad.ok, false);
  assert.match(bad.error, /未知工具/);
  const invalid = await reg.execute('echo', { text: 42 });
  assert.equal(invalid.ok, false);
  assert.match(invalid.error, /校验失败/);
  const ok = await reg.execute('echo', { text: 'hi' });
  assert.equal(ok.ok, true);
  assert.deepEqual(ok.output, { echo: 'hi' });
  assert.equal(reg.list().length, 1);
  assert.throws(() => reg.register({ name: 'echo', handler: () => {} }), /重复注册/);
});

test('ToolRegistry：超时控制', async () => {
  const reg = new ToolRegistry({ defaultTimeoutMs: 30 });
  reg.register({ name: 'hang', inputSchema: {}, handler: () => new Promise(() => {}) });
  const r = await reg.execute('hang', {});
  assert.equal(r.ok, false);
  assert.match(r.error, /超时/);
});

test('内置工具：calc / now / hash / uuid', async () => {
  const reg = new ToolRegistry();
  registerBuiltinTools(reg, { memory: null });
  const calc = await reg.execute('calc', { expression: '(3+4)*2^10' });
  assert.equal(calc.output.value, 7168);
  const now = await reg.execute('now', {});
  assert.ok(now.output.iso);
  const hash = await reg.execute('hash', { text: 'abc' });
  assert.equal(hash.output.digest, 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  assert.match((await reg.execute('uuid', {})).output.uuid, /^[0-9a-f-]{36}$/);
});

test('内置工具：memory_search / memory_store 注入', async () => {
  const { MemorySystem } = await import('../src/core/memory.js');
  const memory = new MemorySystem();
  await memory.remember('NovaFlagship 使用 RRF 融合检索', { salience: 0.8 });
  const reg = new ToolRegistry();
  registerBuiltinTools(reg, { memory });
  const found = await reg.execute('memory_search', { query: 'RRF 融合', k: 3 });
  assert.equal(found.ok, true);
  assert.ok(found.output.hits.length >= 1);
  const stored = await reg.execute('memory_store', { text: '熔断器防止雪崩' });
  assert.equal(stored.output.stored, true);
});

test('McpClient：完整 JSON-RPC 握手 + 工具调用 + 错误传播', async () => {
  const fixture = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures', 'mcp-echo-server.mjs');
  const client = new McpClient({ name: 'mcp-echo', command: process.execPath, args: [fixture] });
  await client.start();
  try {
    const tools = await client.listTools();
    assert.equal(tools.length, 2);
    assert.deepEqual(tools.map((t) => t.name).sort(), ['add', 'echo']);
    const echo = await client.callTool('echo', { text: '旗舰' });
    assert.deepEqual(echo, { echo: '旗舰' });
    const add = await client.callTool('add', { a: 2, b: 40 });
    assert.deepEqual(add, { sum: 42 });
    await assert.rejects(() => client.callTool('add', { a: 'x' }), /数值参数/);
    await assert.rejects(() => client.callTool('missing', {}), /未知工具/);

    // 注册进 ToolRegistry：命名空间隔离
    const reg = new ToolRegistry();
    reg.registerMcp(tools.map((t) => ({ ...t, call: (args) => client.callTool(t.name, args) })));
    assert.equal(reg.has('mcp__mcp-echo__echo'), true);
    const viaReg = await reg.execute('mcp__mcp-echo__add', { a: 1, b: 2 });
    assert.deepEqual(viaReg.output, { sum: 3 });
  } finally {
    client.close();
  }
});
