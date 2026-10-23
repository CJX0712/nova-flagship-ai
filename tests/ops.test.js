// L9 微操作目录
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createOps, FAMILIES } from '../src/core/ops.js';
import { createNova } from '../src/nova.js';

test('目录：10 大家族 + 规模', () => {
  assert.equal(FAMILIES.length, 10);
  const ops = createOps();
  assert.ok(ops.size >= 35);
  const families = new Set(ops.list().map((o) => o.family));
  assert.equal(families.size, 10);
  for (const f of FAMILIES) assert.ok(families.has(f.code));
});

test('编号规则：<FAMILY>-<NNNN> 唯一', () => {
  const ops = createOps();
  const ids = ops.list().map((o) => o.id);
  assert.equal(new Set(ids).size, ids.length);
  for (const id of ids) assert.match(id, /^[A-Z]{3}-\d{4}$/);
});

test('RET-0001 rrf_fuse 可执行', () => {
  const ops = createOps();
  const out = ops.run('RET-0001', { rankings: [['a', 'b'], ['c', 'a']] });
  assert.equal(out.fused[0].id, 'a'); // a 两路都靠前 ⇒ 融合分最高
  assert.ok(out.fused[0].score > out.fused[1].score);
  // 边界：n/k 上界
  const two = ops.run('RET-0001', { rankings: [['x'], ['x']] });
  assert.ok(two.fused[0].score <= 2 / 60 + 1e-12);
});

test('RSN-0001 sample：种子确定', () => {
  const ops = createOps();
  const a = ops.run('RSN-0001', { logits: [1, 2, 3], seed: 9 });
  const b = ops.run('RSN-0001', { logits: [1, 2, 3], seed: 9 });
  assert.equal(a.index, b.index);
  assert.equal(ops.run('RSN-0001', { logits: [1, 2, 3], seed: 9, temperature: 0 }).index, 2);
});

test('ACT-0001 calc / VER-0001 schema / MET-0001 budget', () => {
  const ops = createOps();
  assert.equal(ops.run('ACT-0001', { expression: '2^10' }).value, 1024);
  const v = ops.run('VER-0001', { data: { a: 1 }, schema: { type: 'object', properties: { a: { type: 'string' } } } });
  assert.equal(v.errors.length, 1);
  const b = ops.run('MET-0001', { used: { steps: 3 }, budget: { maxSteps: 3 } });
  assert.deepEqual(b.exceeded, ['maxSteps']);
  assert.equal(b.ok, false);
});

test('PLN-0001 decompose_task', () => {
  const ops = createOps();
  const out = ops.run('PLN-0001', { task: '先查资料。然后写代码；最后跑测试' });
  assert.ok(out.subtasks.length >= 3);
});

test('未知操作抛错', () => {
  assert.throws(() => createOps().run('XXX-0001', {}), /未知微操作/);
});

test('MEM 操作经 ctx 注入可用', async () => {
  const nova = createNova({ allowHttp: false, fsRoots: [] });
  await nova.memory.remember('微操作测试记忆');
  const ctx = { memory: nova.memory };
  const stored = await nova.ops.run('MEM-0001', { text: '通过微操作写入' }, ctx);
  assert.ok(stored.id);
  const hits = await nova.ops.run('MEM-0002', { query: '微操作写入', k: 3 }, ctx);
  assert.ok(hits.hits.length >= 1);
  const c = nova.ops.run('MEM-0003', {}, ctx);
  assert.ok(c.size >= 2);
});
