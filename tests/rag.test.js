// L6 RAG：分块不变量 / 上下文组装 / 引用
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chunkText, RagPipeline } from '../src/core/rag.js';
import { MemorySystem } from '../src/core/memory.js';
import { estimateTokens } from '../src/core/tokenize.js';

function mergedCoverage(chunks, len) {
  const ivs = chunks.map((c) => [c.start, c.end]).sort((a, b) => a[0] - b[0]);
  let covered = 0;
  let curEnd = -1;
  for (const [s, e] of ivs) {
    const from = Math.max(s, curEnd);
    if (e > from) covered += e - from;
    curEnd = Math.max(curEnd, e);
  }
  return covered / len;
}

test('chunkText：小文本单块', () => {
  const c = chunkText('短文本', { size: 100 });
  assert.equal(c.length, 1);
  assert.equal(c[0].text, '短文本');
  assert.equal(c[0].start, 0);
  assert.equal(c[0].end, 3);
});

test('chunkText：覆盖性 + 有界性 + 确定性', () => {
  const text = ('旗舰架构第十层包含1024项微指令。'.repeat(40));
  const chunks = chunkText(text, { size: 120, overlap: 20 });
  assert.ok(chunks.length > 1);
  // 覆盖性 ≥ 99%
  assert.ok(mergedCoverage(chunks, text.length) >= 0.99);
  // 有界性
  for (const c of chunks) assert.ok(c.text.length <= 120, `块超限: ${c.text.length}`);
  // 确定性
  assert.deepEqual(chunkText(text, { size: 120, overlap: 20 }), chunks);
});

test('chunkText：按段落优先切分', () => {
  const text = '第一段内容。\n\n第二段内容。\n\n第三段内容。';
  const chunks = chunkText(text, { size: 12 });
  assert.ok(chunks.length >= 2);
  assert.ok(chunks.some((c) => c.text.includes('第一段')));
});

test('chunkText：空串', () => {
  assert.deepEqual(chunkText(''), []);
  assert.deepEqual(chunkText(null), []);
});

test('RagPipeline：ingest → buildContext 带引用与预算', async () => {
  const mem = new MemorySystem();
  const rag = new RagPipeline({ memory: mem, chunkSize: 100, chunkOverlap: 10, maxContextTokens: 300 });
  const r = await rag.ingest([
    { id: 'doc1', text: 'NovaFlagship 的核心是零依赖智能体引擎，支持确定性回放验证。' },
    { id: 'doc2', text: '记忆系统采用三段式：工作记忆、情节记忆、语义记忆。' },
    { id: 'doc3', text: '检索使用 BM25 与稠密向量经 RRF 融合，论文级算法保证召回质量。' },
  ]);
  assert.equal(r.docs, 3);
  assert.ok(r.chunks >= 3);

  const ctx = await rag.buildContext('回放 验证', { k: 4 });
  assert.ok(ctx.hits >= 1);
  assert.match(ctx.context, /\[1\]/);
  assert.ok(ctx.usedTokens <= 300);
  assert.ok(ctx.sources.length >= 1);
  assert.ok(ctx.sources.every((s) => s.ref >= 1 && s.source));
});

test('RagPipeline：空库返回空上下文', async () => {
  const rag = new RagPipeline({ memory: new MemorySystem() });
  const ctx = await rag.buildContext('任意查询');
  assert.equal(ctx.context, '');
  assert.equal(ctx.hits, 0);
});
