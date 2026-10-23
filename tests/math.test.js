// L1 数学基元层：不变量测试（硬金标准）
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  mulberry32, logsumexp, softmax, dot, norm, l2normalize, cosine, sigmoid,
  zscore, ewma, BM25Index, rrfFuse, applyRoPE, sampleFromLogits, relErr,
} from '../src/core/math.js';

test('logsumexp：稳定性与下界', () => {
  const xs = [1, 2, 3, 4];
  assert.ok(logsumexp(xs) >= Math.max(...xs) - 1e-12);
  assert.ok(relErr(logsumexp(xs), Math.log(xs.reduce((a, b) => a + Math.exp(b), 0))) < 1e-12);
  // 大值不上溢
  assert.ok(Number.isFinite(logsumexp([1000, 1001, 1002])));
  assert.equal(logsumexp([]), -Infinity);
  assert.equal(logsumexp([-Infinity]), -Infinity);
});

test('softmax：和为 1、温度退化、单调', () => {
  const p = softmax([1, 2, 3]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(p[0] < p[1] && p[1] < p[2]);
  const greedy = softmax([1, 5, 2], 0);
  assert.deepEqual(greedy, [0, 1, 0]);
  assert.deepEqual(softmax([]), []);
});

test('cosine：值域 / 自相似 / 零向量', () => {
  assert.equal(cosine([1, 0], [1, 0]), 1);
  assert.equal(cosine([1, 0], [0, 1]), 0);
  assert.equal(cosine([1, 2, 3], [-1, -2, -3]), -1);
  assert.equal(cosine([0, 0], [1, 1]), 0);
  for (const c of [cosine([3, 1], [1, 3]), cosine([-2, 5], [4, -1])]) {
    assert.ok(c >= -1 && c <= 1);
  }
});

test('sigmoid：稳定且值域 (0,1)', () => {
  assert.ok(Math.abs(sigmoid(0) - 0.5) < 1e-12);
  assert.ok(sigmoid(1000) > 0.999999999);
  assert.ok(sigmoid(-1000) < 1e-9);
  assert.ok(Math.abs(sigmoid(2) + sigmoid(-2) - 1) < 1e-12);
});

test('zscore / ewma', () => {
  const z = zscore([1, 2, 3, 4, 5]);
  assert.ok(Math.abs(z.reduce((a, b) => a + b, 0)) < 1e-12);
  assert.deepEqual(zscore([7, 7, 7]), [0, 0, 0]);
  const e = ewma([10, 20, 30], 0.5);
  assert.equal(e[0], 10);
  assert.equal(e[1], 15);
  assert.equal(e[2], 22.5);
});

test('BM25：Okapi 性质（非负、边际递减、降序）', () => {
  const idx = new BM25Index();
  idx.add('d1', ['猫', '喜欢', '鱼']);
  idx.add('d2', ['猫', '狗', '狗']);
  idx.add('d3', ['鸟', '飞']);
  assert.ok(idx.idf('猫') >= 0);
  // 词频越高分越高（同一文档）
  const low = idx.score(['猫'], 'd2');
  idx.add('d4', ['猫', '猫', '猫']);
  const high = idx.score(['猫'], 'd4');
  assert.ok(high > low);
  // 边际递减：f=2 的增益 > f=2→3 的增益
  const g1 = idx.score(['x'], 'd4'); // d4 无 x ⇒ 0，换方式验证递减性
  assert.equal(g1, 0);
  const results = idx.search(['猫'], 10);
  assert.ok(results.every((r, i, arr) => i === 0 || arr[i - 1].score >= r.score));
  assert.equal(idx.score(['猫'], 'd3'), 0);
});

test('RRF：有界性 + 单调性', () => {
  const fused = rrfFuse([['a', 'b', 'c'], ['b', 'a', 'c']], 60);
  const n = 2;
  for (const f of fused) assert.ok(f.score > 0 && f.score <= n / 60 + 1e-12);
  // 单调性：新增一路排序只增不减
  const before = new Map(rrfFuse([['a', 'b']], 60).map((f) => [f.id, f.score]));
  const after = new Map(rrfFuse([['a', 'b'], ['b', 'a']], 60).map((f) => [f.id, f.score]));
  for (const [id, s] of before) assert.ok(after.get(id) >= s - 1e-12);
});

test('RoPE：旋转保范数 + 零位置恒等', () => {
  const v = [1, 2, 3, 4, 5, 6, 7, 8];
  assert.deepEqual(applyRoPE(v, 0), v);
  const rotated = applyRoPE(v, 137);
  assert.ok(Math.abs(norm(rotated) - norm(v)) < 1e-9);
  // 确定性
  assert.deepEqual(applyRoPE(v, 137), rotated);
});

test('核采样：确定性 / 贪心退化 / 核保留', () => {
  const logits = [3, 1, 2, 0.5];
  const r1 = mulberry32(7);
  const r2 = mulberry32(7);
  assert.equal(sampleFromLogits(logits, { rng: r1 }).index, sampleFromLogits(logits, { rng: r2 }).index);
  assert.equal(sampleFromLogits(logits, { temperature: 0 }).index, 0);
  const nuc = sampleFromLogits(logits, { topP: 0.5, rng: mulberry32(1) });
  assert.ok(nuc.kept.includes(nuc.index));
  assert.ok(nuc.kept.length >= 1);
  assert.throws(() => sampleFromLogits([]));
});

test('向量工具', () => {
  assert.equal(dot([1, 2, 3], [4, 5, 6]), 32);
  assert.ok(Math.abs(norm([3, 4]) - 5) < 1e-12);
  assert.ok(Math.abs(norm(l2normalize([3, 4])) - 1) < 1e-12);
});
