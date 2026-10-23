/**
 * NovaFlagship · L6 检索增强层（RAG）
 * ------------------------------------------------------------------
 * - 递归分块： separators 优先级递降（LangChain RecursiveCharacterTextSplitter
 *   的同款策略，工业验证），带重叠与字符偏移（引用可溯源）。
 * - 上下文组装：按检索分贪心装袋（token 预算内最大化信息量），输出带 [n] 引用编号。
 * - 检索质量由 L5 混合检索保证（BM25 ⊕ 稠密 ⊕ RRF）。
 *
 * 不变量（tests/rag.test.js 背书）：
 *   - 覆盖性：原文所有非分隔符字符都出现在某个分块中
 *   - 有界性：分块 ≤ size（不可再分的原子块除外）
 *   - 预算：组装上下文 token ≤ 预算；引用编号与来源一一对应
 */

import { packToBudget, estimateTokens } from './tokenize.js';

const DEFAULT_SEPARATORS = ['\n\n', '\n', '。', '！', '？', '；', '. ', '! ', '? ', '; ', ' ', ''];

/** 递归分块。返回 [{ text, index, start, end }]（start/end 为原文字符偏移）。 */
export function chunkText(text, { size = 480, overlap = 60, separators = DEFAULT_SEPARATORS } = {}) {
  const src = String(text ?? '');
  if (src.length === 0) return [];
  if (src.length <= size) return [{ text: src, index: 0, start: 0, end: src.length }];

  const chunks = [];
  const splitRec = (str, offset, sepIdx) => {
    if (str.length <= size || sepIdx >= separators.length) {
      chunks.push({ text: str, index: 0, start: offset, end: offset + str.length });
      return;
    }
    const sep = separators[sepIdx];
    const parts = sep === '' ? [...str] : str.split(sep);
    let cursor = 0;
    let buf = '';
    let bufStart = offset;
    for (const part of parts) {
      const candidate = buf ? buf + sep + part : part;
      if (candidate.length > size && buf) {
        splitRec(buf, bufStart, sepIdx + 1);
        // 重叠：把上一块尾部带入下一块
        const tail = buf.slice(Math.max(0, buf.length - overlap));
        buf = tail + (sep === '' ? '' : sep) + part;
        bufStart = offset + cursor - tail.length - (sep === '' ? 0 : sep.length);
      } else {
        buf = candidate;
      }
      cursor += part.length + (sep === '' ? 0 : sep.length);
    }
    if (buf) splitRec(buf, bufStart, sepIdx + 1);
  };

  splitRec(src, 0, 0);
  return chunks.map((c, i) => ({ ...c, index: i }));
}

/** RAG 管道：ingest → 混合检索 → 预算内上下文组装（带引用）。 */
export class RagPipeline {
  /**
   * @param {object} opts { memory: MemorySystem, chunkSize, chunkOverlap, maxContextTokens }
   */
  constructor({ memory, chunkSize = 480, chunkOverlap = 60, maxContextTokens = 1800 } = {}) {
    if (!memory) throw new Error('RagPipeline 需要 memory');
    this.memory = memory;
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;
    this.maxContextTokens = maxContextTokens;
  }

  /** 摄入文档：[{ id, text, meta }] ⇒ 分块写入语义记忆。 */
  async ingest(docs) {
    let total = 0;
    for (const doc of docs) {
      const chunks = chunkText(doc.text, { size: this.chunkSize, overlap: this.chunkOverlap });
      for (const c of chunks) {
        await this.memory.semantic.add({
          text: c.text,
          meta: { docId: doc.id, chunk: c.index, start: c.start, end: c.end, ...(doc.meta || {}) },
          salience: doc.meta?.salience ?? 0.5,
        });
        total++;
      }
    }
    return { docs: docs.length, chunks: total };
  }

  /** 检索 + 组装带引用的上下文。 */
  async buildContext(query, { k = 6, maxTokens = this.maxContextTokens } = {}) {
    const hits = await this.memory.recall(query, Math.max(k * 3, k));
    const items = hits.map((h, i) => ({
      id: h.id,
      text: `[${i + 1}] ${h.text}${h.meta?.docId ? `（来源:${h.meta.docId}#${h.meta.chunk ?? 0}）` : ''}`,
      score: h.score,
      source: h.meta?.docId ? `${h.meta.docId}#${h.meta.chunk ?? 0}` : h.id,
    }));
    const { picked, used } = packToBudget(items, maxTokens, (it) => estimateTokens(it.text));
    const context = picked.map((p) => p.text).join('\n');
    const sources = picked.map((p, i) => ({ ref: i + 1, id: p.id, source: p.source, score: Number(p.score.toFixed(4)) }));
    return { context, sources, hits: picked.length, usedTokens: used };
  }
}
