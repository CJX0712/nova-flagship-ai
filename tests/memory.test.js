// L5 记忆层：嵌入确定性 / 衰减单调 / 混合检索 / 巩固
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MemorySystem, HashEmbedder, WorkingMemory, EpisodicMemory, SemanticMemory } from '../src/core/memory.js';
import { cosine } from '../src/core/math.js';

test('HashEmbedder：确定性 + 自相似 + 相似性排序', () => {
  const emb = new HashEmbedder({ dim: 256 });
  const a = emb.embed('旗舰 AI 引擎');
  const b = emb.embed('旗舰 AI 引擎');
  assert.deepEqual(a, b); // 确定性 ⇒ 可回放
  assert.equal(cosine(a, a), 1);
  const near = emb.embed('旗舰 AI 引擎优化');
  const far = emb.embed('今天天气不错适合钓鱼');
  assert.ok(cosine(a, near) > cosine(a, far));
});

test('WorkingMemory：容量有界', () => {
  const wm = new WorkingMemory({ capacity: 3 });
  for (let i = 0; i < 10; i++) wm.push({ kind: 'x', text: String(i) });
  assert.equal(wm.size, 3);
  assert.deepEqual(wm.recent(2).map((i) => i.text), ['8', '9']);
});

test('EpisodicMemory：Ebbinghaus 衰减单调', () => {
  let now = 1000;
  const ep = new EpisodicMemory({ lambda: 0.001, now: () => now });
  ep.record({ kind: 'task', content: '新事件', salience: 1 });
  const fresh = ep.recall()[0].decayedScore;
  now += 10000;
  const aged = ep.recall()[0].decayedScore;
  assert.ok(fresh > aged);
  assert.ok(Math.abs(aged - Math.exp(-0.001 * 10000)) < 1e-9);
  // 显著性提升分数
  ep.record({ kind: 'task', content: '重要事件', salience: 1 });
  now += 10000;
  const top = ep.recall()[0];
  assert.equal(top.content, '重要事件');
});

test('SemanticMemory：混合检索双路命中', async () => {
  const mem = new SemanticMemory({ embedder: new HashEmbedder({ dim: 256 }) });
  await mem.add({ text: '猫喜欢吃鱼' }); // 词汇命中
  await mem.add({ text: '狗在院子里跑' });
  await mem.add({ text: '猫吃鱼拌饭' }); // 稠密近似命中（n-gram 重叠）
  const hits = mem.searchHybrid('猫 鱼', { k: 3 });
  assert.ok(hits.length >= 2);
  assert.ok(hits[0].sources.length >= 1);
  assert.ok(hits[0].score >= hits[1].score);
  // 确定性
  assert.deepEqual(mem.searchHybrid('猫 鱼', { k: 3 }), hits);
  // 空库
  assert.deepEqual(new SemanticMemory().searchHybrid('x'), []);
});

test('SemanticMemory：consolidate 合并近重复', async () => {
  const mem = new SemanticMemory({ embedder: new HashEmbedder({ dim: 512 }) });
  await mem.add({ text: 'NovaFlagship 是旗舰 AI 引擎', salience: 0.4 });
  await mem.add({ text: 'NovaFlagship 是旗舰 AI 引擎', salience: 0.6 });
  await mem.add({ text: '完全不同的另一条记忆内容', salience: 0.5 });
  assert.equal(mem.size, 3);
  const r = mem.consolidate({ threshold: 0.95 });
  assert.equal(r.merged, 1);
  assert.equal(mem.size, 2);
});

test('MemorySystem：remember → recall → contextBlock 闭环', async () => {
  const mem = new MemorySystem();
  await mem.remember('晨星是一人公司创始人，坐标深圳', { salience: 0.9 });
  await mem.remember('旗舰架构分为十层', { salience: 0.7 });
  const hits = await mem.recall('晨星 深圳', 3);
  assert.ok(hits.length >= 1);
  assert.match(hits[0].text, /晨星/);
  const block = await mem.contextBlock('晨星 深圳');
  assert.match(block, /\[记忆\] 共 \d+ 条/);
  assert.match(block, /\[1\]/);
  const stats = mem.stats();
  assert.equal(stats.semantic, 2);
  assert.equal(stats.episodic, 2);
  assert.equal(mem.working.size, 2);
  const c = mem.consolidate();
  assert.ok(c.size >= 1);
});

test('snapshot/restore：快照隔离与无损往返', async () => {
  const mem = new MemorySystem();
  await mem.remember('快照测试甲', { salience: 0.8 });
  const snap = mem.snapshot();
  await mem.remember('快照测试乙', { salience: 0.3 });
  assert.equal(mem.stats().semantic, 2);
  mem.restore(snap);
  assert.equal(mem.stats().semantic, 1);
  assert.equal(mem.stats().episodic, 1);
  // 恢复后检索行为不变（确定性 embedder ⇒ 向量无损重建）
  const hits = await mem.recall('快照测试甲', 3);
  assert.equal(hits.length, 1);
  assert.match(hits[0].text, /快照测试甲/);
  // dump/load 往返
  const dump = mem.semantic.dump();
  mem.semantic.load(dump);
  assert.equal(mem.semantic.size, 1);
});
