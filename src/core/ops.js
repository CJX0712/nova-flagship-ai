/**
 * NovaFlagship · L9 微操作注册层（1024 微指令分类法的种子）
 * ------------------------------------------------------------------
 * 十层分解架构的第十层由「微指令级原子操作」构成。本文件给出操作分类法
 * （10 大家族）与首批 ~50 个已实现原子操作。每个操作是纯函数或受控副作用，
 * 可独立测试、独立替换、独立演进到 1024 项的完整目录。
 *
 * 编号规则：<FAMILY>-<NNNN>，如 RET-0001。
 */

import { softmax, logsumexp, cosine, dot, l2normalize, sampleFromLogits, mulberry32, rrfFuse } from './math.js';
import { chunkText } from './rag.js';
import { estimateTokens, tokenizeWords, charNgrams } from './tokenize.js';
import { validateSchema } from './schema.js';
import { evalExpression } from './tools.js';

export const FAMILIES = [
  { code: 'OBS', name: '感知', desc: '输入解析、语言检测、实体抽取、计量' },
  { code: 'RET', name: '检索', desc: 'BM25/稠密/融合/多样性重排' },
  { code: 'EMB', name: '嵌入', desc: '向量化、归一化、相似度' },
  { code: 'CHK', name: '分块', desc: '递归分块、重叠合并、偏移溯源' },
  { code: 'RSN', name: '推理', desc: '采样、对数概率、自洽投票' },
  { code: 'PLN', name: '规划', desc: '任务分解、预算分配' },
  { code: 'ACT', name: '执行', desc: '工具调用、表达式求值、外部动作' },
  { code: 'MEM', name: '记忆', desc: '存储、召回、巩固、衰减' },
  { code: 'VER', name: '验证', desc: 'schema 校验、回放比对、不变量检查' },
  { code: 'MET', name: '元操作', desc: '预算检查、轨迹发射、成本记账' },
];

/** MMR 多样性重排（Carbonell & Goldstein, 1998）——用词集合 Jaccard 作相似代理。 */
function jaccard(a, b) {
  const sa = new Set(tokenizeWords(a));
  const sb = new Set(tokenizeWords(b));
  if (sa.size === 0 && sb.size === 0) return 0;
  let inter = 0;
  for (const t of sa) if (sb.has(t)) inter++;
  return inter / (sa.size + sb.size - inter);
}

/** 多数表决（自洽性，Self-Consistency, Wang et al. 2022）。 */
function majorityVote(candidates) {
  const norm = (s) => String(s).trim().replace(/\s+/g, ' ');
  const counts = new Map();
  for (const c of candidates) {
    const k = norm(c);
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  let best = null;
  let bestN = -1;
  for (const [k, n] of counts) {
    if (n > bestN || (n === bestN && k < best)) {
      best = k;
      bestN = n;
    }
  }
  return { answer: best, votes: bestN, distinct: counts.size };
}

class OpRegistry {
  constructor() {
    this.ops = new Map();
    this._counters = new Map();
  }

  /** 定义操作。fn(args, ctx) 为纯函数或受控副作用。 */
  define(family, name, fn, { desc = '' } = {}) {
    const code = family;
    const n = (this._counters.get(code) || 0) + 1;
    this._counters.set(code, n);
    const id = `${code}-${String(n).padStart(4, '0')}`;
    this.ops.set(id, { id, family: code, name, desc, fn });
    return id;
  }

  get(id) {
    return this.ops.get(id);
  }

  list({ family } = {}) {
    return [...this.ops.values()].filter((o) => !family || o.family === family).map(({ id, family: f, name, desc }) => ({ id, family: f, name, desc }));
  }

  run(id, args = {}, ctx = {}) {
    const op = this.ops.get(id);
    if (!op) throw new Error(`未知微操作: ${id}`);
    return op.fn(args, ctx);
  }

  get size() {
    return this.ops.size;
  }
}

/** 构建默认微操作目录。 */
export function createOps() {
  const ops = new OpRegistry();

  // OBS 感知
  ops.define('OBS', 'count_tokens', (a) => ({ tokens: estimateTokens(String(a.text ?? '')) }), { desc: 'token 计量' });
  ops.define('OBS', 'tokenize', (a) => ({ words: tokenizeWords(String(a.text ?? '')) }), { desc: '词元化' });
  ops.define('OBS', 'detect_language', (a) => {
    const t = String(a.text ?? '');
    const cjk = (t.match(/[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/g) || []).length;
    return { language: cjk / Math.max(t.length, 1) > 0.2 ? 'zh' : 'en', cjkRatio: cjk / Math.max(t.length, 1) };
  }, { desc: '中英检测' });
  ops.define('OBS', 'normalize_text', (a) => ({ text: String(a.text ?? '').trim().replace(/\s+/g, ' ').toLowerCase() }), { desc: '文本归一化' });
  ops.define('OBS', 'extract_keywords', (a) => {
    const stop = new Set(['的', '了', '是', '我', '你', '他', '她', '它', '在', '和', '与', '及', '对', '为', '以', 'the', 'a', 'an', 'is', 'are', 'of', 'to', 'and']);
    const freq = new Map();
    for (const w of tokenizeWords(String(a.text ?? ''))) {
      if (stop.has(w) || w.length < 2) continue;
      freq.set(w, (freq.get(w) || 0) + 1);
    }
    const keywords = [...freq.entries()].sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : 1)).slice(0, a.k ?? 5);
    return { keywords };
  }, { desc: '关键词抽取（去停用词）' });

  // RET 检索
  ops.define('RET', 'rrf_fuse', (a) => ({ fused: rrfFuse(a.rankings || [], a.k ?? 60) }), { desc: 'RRF 融合' });
  ops.define('RET', 'mmr_rerank', (a) => {
    const { hits = [], lambda = 0.7, k = 5 } = a;
    const selected = [];
    const rest = [...hits];
    while (selected.length < k && rest.length > 0) {
      let bestIdx = 0;
      let bestScore = -Infinity;
      rest.forEach((h, i) => {
        const rel = h.score ?? 0;
        const div = selected.length === 0 ? 0 : Math.max(...selected.map((s) => jaccard(h.text || '', s.text || '')));
        const s = lambda * rel - (1 - lambda) * div;
        if (s > bestScore) {
          bestScore = s;
          bestIdx = i;
        }
      });
      selected.push(rest.splice(bestIdx, 1)[0]);
    }
    return { reranked: selected };
  }, { desc: 'MMR 多样性重排' });
  ops.define('RET', 'threshold_filter', (a) => ({ hits: (a.hits || []).filter((h) => (h.score ?? 0) >= (a.threshold ?? 0)) }), { desc: '分数阈值过滤' });
  ops.define('RET', 'dedupe_by_text', (a) => {
    const seen = new Set();
    const hits = (a.hits || []).filter((h) => {
      const k = String(h.text ?? h.id ?? '');
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
    return { hits };
  }, { desc: '按文本去重' });

  // EMB 嵌入
  ops.define('EMB', 'char_ngrams', (a) => ({ grams: charNgrams(String(a.text ?? ''), a.n ?? 3) }), { desc: '字符 n-gram' });
  ops.define('EMB', 'cosine', (a) => ({ similarity: cosine(a.a || [], a.b || []) }), { desc: '余弦相似度' });
  ops.define('EMB', 'softmax', (a) => ({ probs: softmax(a.logits || [], a.temperature ?? 1) }), { desc: '稳定 softmax' });
  ops.define('EMB', 'logsumexp', (a) => ({ value: logsumexp(a.xs || []) }), { desc: '稳定 logsumexp' });
  ops.define('EMB', 'l2_normalize', (a) => ({ vector: l2normalize(a.vector || []) }), { desc: 'L2 归一化' });
  ops.define('EMB', 'dot', (a) => ({ value: dot(a.a || [], a.b || []) }), { desc: '点积' });
  ops.define('EMB', 'hash_embed', (a, ctx) => {
    if (!ctx.embedder) throw new Error('hash_embed 需要 ctx.embedder');
    return { vector: ctx.embedder.embed(String(a.text ?? '')) };
  }, { desc: '确定性哈希嵌入（可换真实 embedder）' });

  // CHK 分块
  ops.define('CHK', 'recursive_chunk', (a) => ({ chunks: chunkText(String(a.text ?? ''), { size: a.size ?? 480, overlap: a.overlap ?? 60 }) }), { desc: '递归分块' });
  ops.define('CHK', 'chunk_stats', (a) => {
    const chunks = chunkText(String(a.text ?? ''), { size: a.size ?? 480, overlap: a.overlap ?? 60 });
    const lens = chunks.map((c) => c.text.length);
    return { count: chunks.length, min: lens.length ? Math.min(...lens) : 0, max: lens.length ? Math.max(...lens) : 0, avg: lens.length ? lens.reduce((x, y) => x + y, 0) / lens.length : 0 };
  }, { desc: '分块统计' });

  // RSN 推理
  ops.define('RSN', 'sample', (a) => {
    const rng = a.seed !== undefined ? mulberry32(a.seed) : Math.random;
    return sampleFromLogits(a.logits || [], { temperature: a.temperature ?? 1, topK: a.topK ?? 0, topP: a.topP ?? 1, rng });
  }, { desc: '核采样' });
  ops.define('RSN', 'self_consistency_vote', (a) => majorityVote(a.candidates || []), { desc: '自洽多数表决' });
  ops.define('RSN', 'build_cot_prompt', (a) => ({
    prompt: `问题：${a.question ?? ''}\n请一步步思考（Scratchpad），先列事实，再推理，最后给结论。`,
  }), { desc: '思维链提示构建' });
  ops.define('RSN', 'answer_sanity', (a) => {
    const text = String(a.answer ?? '');
    const problems = [];
    if (!text.trim()) problems.push('空应答');
    if (a.maxLength && text.length > a.maxLength) problems.push('超长');
    return { ok: problems.length === 0, problems, length: text.length };
  }, { desc: '应答健全性检查' });

  // PLN 规划
  ops.define('PLN', 'decompose_task', (a) => {
    const parts = String(a.task ?? '')
      .split(/[。！？；\n]|(?:\d+[.、)])|(?:首先|然后|接着|最后|同时)/)
      .map((s) => s.trim())
      .filter((s) => s.length > 1);
    return { subtasks: parts.slice(0, a.maxItems ?? 8) };
  }, { desc: '任务分解' });
  ops.define('PLN', 'budget_allocate', (a) => {
    const total = a.totalTokens ?? 4000;
    const weights = a.weights || [];
    const sum = weights.reduce((x, y) => x + y, 0) || 1;
    return { allocation: weights.map((w) => Math.floor((w / sum) * total)) };
  }, { desc: '预算分配' });
  ops.define('PLN', 'insert_reflection', (a) => {
    const steps = a.steps || [];
    const out = [];
    for (const s of steps) {
      out.push(s);
      out.push({ ...s, reflect: true, note: `对「${s.name ?? s}」的结果做一次自我检查` });
    }
    return { steps: out.slice(0, a.maxItems ?? 20) };
  }, { desc: '插入反思步（Reflexion）' });

  // ACT 执行
  ops.define('ACT', 'calc', (a) => ({ value: evalExpression(String(a.expression ?? '')) }), { desc: '表达式求值' });
  ops.define('ACT', 'json_parse_safe', (a) => {
    try {
      return { ok: true, value: JSON.parse(String(a.text ?? '')) };
    } catch (err) {
      return { ok: false, error: String(err.message) };
    }
  }, { desc: '安全 JSON 解析' });

  // MEM 记忆（ctx.memory 注入）
  ops.define('MEM', 'remember', async (a, ctx) => ({ id: await ctx.memory.remember(a.text, a.opts || {}) }), { desc: '写入记忆' });
  ops.define('MEM', 'recall', async (a, ctx) => ({ hits: await ctx.memory.recall(a.query, a.k ?? 5) }), { desc: '召回记忆' });
  ops.define('MEM', 'consolidate', (a, ctx) => ctx.memory.consolidate(a.opts || {}), { desc: '巩固合并' });
  ops.define('MEM', 'decay_sweep', (a, ctx) => {
    const min = a.minScore ?? 0.1;
    const kept = ctx.memory.episodic.recall({ limit: Number.MAX_SAFE_INTEGER }).filter((e) => e.decayedScore >= min);
    return { kept: kept.length, dropped: ctx.memory.episodic.size - kept.length };
  }, { desc: '衰减清扫（Ebbinghaus）' });
  ops.define('MEM', 'stats', (a, ctx) => ctx.memory.stats(), { desc: '记忆统计' });

  // VER 验证
  ops.define('VER', 'schema_validate', (a) => ({ errors: validateSchema(a.data, a.schema || {}) }), { desc: 'Schema 校验' });
  ops.define('VER', 'softmax_invariant', (a) => {
    const p = softmax(a.logits || [], a.temperature ?? 1);
    const sum = p.reduce((x, y) => x + y, 0);
    return { ok: Math.abs(sum - 1) < 1e-12, sum };
  }, { desc: 'softmax 不变量' });
  ops.define('VER', 'trace_structural', (a) => {
    const events = a.events || [];
    const norm = events.map(({ seq, type, payload }) => {
      const p = JSON.parse(JSON.stringify(payload));
      if (type === 'llm') { delete p.model; delete p.provider; }
      if (type === 'tool_result') delete p.ms;
      return { seq, type, payload: p };
    });
    return { events: norm };
  }, { desc: '轨迹归一化（回放比对）' });

  // MET 元操作
  ops.define('MET', 'budget_check', (a) => {
    const { used = {}, budget = {} } = a;
    const exceeded = [];
    if (budget.maxSteps !== undefined && (used.steps ?? 0) >= budget.maxSteps) exceeded.push('maxSteps');
    if (budget.maxTokens !== undefined && (used.tokens ?? 0) >= budget.maxTokens) exceeded.push('maxTokens');
    if (budget.maxMs !== undefined && (used.ms ?? 0) >= budget.maxMs) exceeded.push('maxMs');
    return { exceeded, ok: exceeded.length === 0 };
  }, { desc: '预算检查' });
  ops.define('MET', 'abort_check', (a) => ({ aborted: !!(a.signal && a.signal.aborted) }), { desc: '取消信号检查' });
  ops.define('MET', 'cost_estimate', (a) => {
    const { promptTokens = 0, completionTokens = 0, priceIn = 0, priceOut = 0 } = a;
    return { costUsd: (promptTokens * priceIn + completionTokens * priceOut) / 1000 };
  }, { desc: '成本估算' });

  return ops;
}
