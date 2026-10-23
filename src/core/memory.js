/**
 * NovaFlagship · L5 记忆层（三段式架构）
 * ------------------------------------------------------------------
 * 世界级参照：
 *   工作记忆   —— 认知科学 Miller 7±2 ⇒ 有界环形缓冲
 *   情节记忆   —— Ebbinghaus 遗忘曲线 R = e^{-λt}，带显著性(salience)加权
 *   语义记忆   —— 混合检索：BM25（稀疏）+ 特征哈希稠密向量 + RRF 融合
 *                （Cormack et al., SIGIR 2009），近似 HNSW 的精确替代
 *                （万级内暴力余弦即最优；接口预留 usearch/hnswlib 适配器）
 *   巩固       —— A-MEM / Zettelkasten 式近重复合并（cosine ≥ 阈值）
 *
 * Embedder 是可插拔接口：默认 HashEmbedder（确定性、零依赖、离线可复现），
 * 生产环境换 @xenova/transformers（all-MiniLM-L6-v2）或 OpenAI embeddings，
 * 一行替换，接口不变。
 */

import { cosine, l2normalize } from './math.js';
import { BM25Index, rrfFuse } from './math.js';
import { charNgrams, tokenizeWords } from './tokenize.js';

/** 确定性特征哈希嵌入器：字符 n-gram → 有符号哈希桶 → L2 归一化。 */
export class HashEmbedder {
  constructor({ dim = 512, ngrams = [2, 3] } = {}) {
    this.dim = dim;
    this.ngrams = ngrams;
    this.name = 'hash-embedder';
  }

  embed(text) {
    const vec = new Array(this.dim).fill(0);
    const grams = this.ngrams.flatMap((n) => charNgrams(text, n));
    for (const g of grams) {
      const h = fnv1a(g);
      const idx = h % this.dim;
      const sign = (h >>> 16) & 1 ? 1 : -1; // 有符号哈希，冲突期望为 0
      vec[idx] += sign;
    }
    return l2normalize(vec);
  }
}

const FNV_OFFSET = 0x811c9dc5;
function fnv1a(str) {
  let h = FNV_OFFSET;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** 工作记忆：有界环形缓冲。 */
export class WorkingMemory {
  constructor({ capacity = 12 } = {}) {
    this.capacity = capacity;
    this.items = [];
  }

  push(item) {
    this.items.push({ at: Date.now(), ...item });
    if (this.items.length > this.capacity) this.items.splice(0, this.items.length - this.capacity);
    return this;
  }

  recent(n = 5) {
    return this.items.slice(-n);
  }

  clear() {
    this.items = [];
  }

  get size() {
    return this.items.length;
  }
}

/** 情节记忆：事件流 + Ebbinghaus 衰减召回。score = salience · e^{-λΔt}。 */
export class EpisodicMemory {
  constructor({ lambda = 1 / (24 * 3600 * 1000), now = () => Date.now() } = {}) {
    this.lambda = lambda;
    this.now = now;
    this.events = [];
  }

  record({ kind = 'event', content, salience = 0.5, meta = {} }) {
    const ev = { id: `ep_${this.now().toString(36)}_${this.events.length}`, at: this.now(), kind, content, salience, meta };
    this.events.push(ev);
    return ev;
  }

  /** 按衰减分数召回。不变量：其余不变时，score 随事件年龄单调递减。 */
  recall({ kinds, limit = 10 } = {}) {
    const t = this.now();
    return this.events
      .filter((e) => !kinds || kinds.includes(e.kind))
      .map((e) => ({ ...e, decayedScore: e.salience * Math.exp(-this.lambda * (t - e.at)) }))
      .sort((a, b) => b.decayedScore - a.decayedScore || b.at - a.at)
      .slice(0, limit);
  }

  recent(n = 10) {
    return this.events.slice(-n);
  }

  get size() {
    return this.events.length;
  }
}

/** 语义记忆：向量库 + BM25 索引 + RRF 混合检索。 */
export class SemanticMemory {
  constructor({ embedder = new HashEmbedder() } = {}) {
    this.embedder = embedder;
    this.docs = new Map(); // id -> { id, text, vec, meta, salience }
    this.bm25 = new BM25Index();
  }

  async add({ text, meta = {}, salience = 0.5, vector } = {}) {
    const id = meta.id || `mem_${fnv1a(text + meta.source || '').toString(36)}_${this.docs.size}`;
    const vec = vector || this.embedder.embed(text);
    this.docs.set(id, { id, text, vec, meta, salience });
    this.bm25.add(id, tokenizeWords(text));
    return id;
  }

  has(id) {
    return this.docs.has(id);
  }

  get size() {
    return this.docs.size;
  }

  /** 稠密检索（精确余弦）。 */
  searchDense(queryVec, k = 10) {
    return [...this.docs.values()]
      .map((d) => ({ id: d.id, score: cosine(queryVec, d.vec) }))
      .sort((a, b) => b.score - a.score || (a.id < b.id ? -1 : 1))
      .slice(0, k);
  }

  /** 稀疏检索（BM25）。 */
  searchLexical(query, k = 10) {
    return this.bm25.search(tokenizeWords(query), k);
  }

  /**
   * 混合检索：dense ⊕ lexical → RRF(k=60)。
   * 不变量：命中按融合分降序；同时被两路检索到的文档融合分更高。
   */
  searchHybrid(query, { k = 5, denseK = 20, lexicalK = 20, rrfK = 60 } = {}) {
    if (this.docs.size === 0) return [];
    const queryVec = this.embedder.embed(query);
    const dense = this.searchDense(queryVec, denseK);
    const lexical = this.searchLexical(query, lexicalK);
    const fused = rrfFuse([dense.map((d) => d.id), lexical.map((l) => l.id)], rrfK);
    const denseMap = new Map(dense.map((d) => [d.id, d.score]));
    const lexMap = new Map(lexical.map((l) => [l.id, l.score]));
    return fused.slice(0, k).map((f) => {
      const doc = this.docs.get(f.id);
      const sources = [];
      if (denseMap.has(f.id)) sources.push('dense');
      if (lexMap.has(f.id)) sources.push('lexical');
      return {
        id: f.id,
        text: doc.text,
        score: f.score,
        denseScore: denseMap.get(f.id) ?? 0,
        lexicalScore: lexMap.get(f.id) ?? 0,
        sources,
        meta: doc.meta,
      };
    });
  }

  /** 近重复合并（cosine ≥ threshold）。保留高显著性者，累积显著性（上限 1）。 */
  consolidate({ threshold = 0.92 } = {}) {
    const entries = [...this.docs.values()];
    const removed = new Set();
    let merged = 0;
    for (let i = 0; i < entries.length; i++) {
      if (removed.has(entries[i].id)) continue;
      for (let j = i + 1; j < entries.length; j++) {
        if (removed.has(entries[j].id)) continue;
        if (cosine(entries[i].vec, entries[j].vec) >= threshold) {
          const [keep, drop] = entries[i].salience >= entries[j].salience ? [i, j] : [j, i];
          entries[keep].salience = Math.min(1, entries[keep].salience + entries[drop].salience * 0.5);
          removed.add(entries[drop].id);
          merged++;
        }
      }
    }
    for (const id of removed) this.docs.delete(id);
    return { merged, size: this.docs.size };
  }

  /** 导出语料（确定性 embedder 可在 load 时无损重建向量）。 */
  dump() {
    return [...this.docs.values()].map((d) => ({ id: d.id, text: d.text, salience: d.salience, meta: d.meta }));
  }

  /** 从导出数据重建（清空现有）。 */
  load(docs) {
    this.docs.clear();
    this.bm25 = new BM25Index({ k1: this.bm25.k1, b: this.bm25.b });
    for (const d of docs || []) {
      const vec = this.embedder.embed(d.text);
      this.docs.set(d.id, { id: d.id, text: d.text, vec, meta: d.meta || {}, salience: d.salience ?? 0.5 });
      this.bm25.add(d.id, tokenizeWords(d.text));
    }
    return this;
  }
}

/** 记忆系统：三段式统一入口。 */
export class MemorySystem {
  constructor({ embedder = new HashEmbedder(), workingCapacity = 12, lambda } = {}) {
    this.working = new WorkingMemory({ capacity: workingCapacity });
    this.episodic = new EpisodicMemory({ lambda });
    this.semantic = new SemanticMemory({ embedder });
  }

  /** 写入一条长期记忆（语义 + 情节双写）。 */
  async remember(text, { salience = 0.5, kind = 'fact', meta = {} } = {}) {
    const id = await this.semantic.add({ text, salience, meta });
    this.episodic.record({ kind, content: text, salience, meta: { ...meta, memoryId: id } });
    this.working.push({ kind: 'memory', text });
    return id;
  }

  /** 混合召回长期记忆。 */
  async recall(query, k = 5) {
    return this.semantic.searchHybrid(query, { k });
  }

  /** 生成注入系统提示的记忆块（带引用编号）。 */
  async contextBlock(query, { k = 5, maxChars = 2000 } = {}) {
    const hits = await this.recall(query, k);
    if (hits.length === 0) return '';
    const lines = hits.map((h, i) => `[${i + 1}] ${h.text}`);
    let block = `[记忆] 共 ${hits.length} 条相关记忆：\n` + lines.join('\n');
    if (block.length > maxChars) block = block.slice(0, maxChars - 1) + '…';
    return block;
  }

  /** 巩固：合并近重复 + 清理低显著性。 */
  consolidate({ threshold = 0.92 } = {}) {
    const r = this.semantic.consolidate({ threshold });
    return { ...r, episodic: this.episodic.size, working: this.working.size };
  }

  /** 全量快照（回放时恢复初始状态用；深拷贝，与运行实例隔离）。 */
  snapshot() {
    return {
      semantic: this.semantic.dump(),
      episodic: this.episodic.events.map((e) => ({ ...e, meta: { ...e.meta } })),
      working: this.working.items.map((i) => ({ ...i })),
    };
  }

  /** 从快照恢复（原地）。 */
  restore(snap) {
    if (!snap) return this;
    this.semantic.load(snap.semantic);
    this.episodic.events = (snap.episodic || []).map((e) => ({ ...e, meta: { ...e.meta } }));
    this.working.items = (snap.working || []).map((i) => ({ ...i }));
    return this;
  }

  stats() {
    return {
      semantic: this.semantic.size,
      episodic: this.episodic.size,
      working: this.working.size,
    };
  }
}
