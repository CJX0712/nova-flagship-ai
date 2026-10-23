/**
 * NovaFlagship · L1 数学基元层
 * ------------------------------------------------------------------
 * 每一个函数都对应一篇世界级论文/一个工业级标准实现，并附带可验证不变量：
 *
 *   logsumexp   —— 数值稳定性标准技巧（避免 exp 上溢），机器学习框架通用
 *   softmax     —— 带温度的稳定 softmax；sum=1 为硬不变量
 *   bm25        —— Okapi BM25（Robertson & Zaragoza, 2009），k1=1.5, b=0.75 文献默认
 *   rrf         —— Reciprocal Rank Fusion（Cormack, Clarke & Büttcher, SIGIR 2009），k=60
 *   rope        —— Rotary Position Embedding（Su et al., 2021）；旋转保范数
 *   sample      —— top-k / top-p（Holtzman et al., 2020）核采样
 *   mulberry32  —— 确定性 PRNG，保证「同种子同输出」的可复现性
 *
 * 设计红线：纯函数、零依赖、零副作用。tests/math.test.js 对全部不变量背书。
 */

/** 确定性 PRNG（mulberry32）。同一种子 ⇒ 同一序列，回放验证的基石。 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 稳定 log-sum-exp：log(Σ e^{x_i}) = m + log(Σ e^{x_i - m})，m = max(x)。
 *  不变量：结果 ≥ max(x)；与朴素实现相对误差 < 1e-12。 */
export function logsumexp(xs) {
  if (!Array.isArray(xs) || xs.length === 0) return -Infinity;
  const m = Math.max(...xs);
  if (m === -Infinity) return -Infinity;
  if (m === Infinity) return Infinity;
  let s = 0;
  for (const x of xs) s += Math.exp(x - m);
  return m + Math.log(s);
}

/** 带温度的稳定 softmax。不变量：输出和为 1（误差 < 1e-12）；温度 → 0 时收敛到 argmax one-hot。 */
export function softmax(xs, temperature = 1) {
  if (!Array.isArray(xs) || xs.length === 0) return [];
  const t = Number(temperature);
  if (!(t > 0)) {
    // 贪心：温度非正 ⇒ 确定性 argmax
    const out = new Array(xs.length).fill(0);
    out[xs.reduce((best, v, i) => (v > xs[best] ? i : best), 0)] = 1;
    return out;
  }
  const scaled = xs.map((x) => x / t);
  const m = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** 向量点积。 */
export function dot(a, b) {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

/** L2 范数。 */
export function norm(a) {
  return Math.sqrt(dot(a, a));
}

/** L2 归一化（零向量原样返回）。 */
export function l2normalize(a) {
  const n = norm(a);
  return n === 0 ? a.slice() : a.map((x) => x / n);
}

/** 余弦相似度。不变量：值域 [-1, 1]；自相似 = 1；零向量 = 0。 */
export function cosine(a, b) {
  const na = norm(a);
  const nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  const c = dot(a, b) / (na * nb);
  return Math.max(-1, Math.min(1, c));
}

/** 稳定 sigmoid：大 |x| 不溢出。 */
export function sigmoid(x) {
  if (x >= 0) {
    const e = Math.exp(-x);
    return 1 / (1 + e);
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

/** z-score 标准化。不变量：均值 ≈ 0，标准差 ≈ 1（常数输入时返回全 0）。 */
export function zscore(xs) {
  const n = xs.length;
  if (n === 0) return [];
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  const varc = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const sd = Math.sqrt(varc);
  if (sd === 0) return new Array(n).fill(0);
  return xs.map((x) => (x - mean) / sd);
}

/** 指数加权移动平均。不变量：权重和 = 1；首值等于序列首元素。 */
export function ewma(xs, alpha = 0.3) {
  if (xs.length === 0) return [];
  const a = Math.min(Math.max(Number(alpha) || 0.3, 0), 1);
  const out = [xs[0]];
  for (let i = 1; i < xs.length; i++) out[i] = a * xs[i] + (1 - a) * out[i - 1];
  return out;
}

/**
 * Okapi BM25 索引（Robertson & Zaragoza, 2009）。
 *   idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))        （Lucene 变体，恒 ≥ 0）
 *   score  = Σ_t idf(t) · f·(k1+1) / (f + k1·(1 - b + b·|d|/avgdl))
 * 默认 k1=1.5, b=0.75 为原文推荐值。
 */
export class BM25Index {
  constructor({ k1 = 1.5, b = 0.75 } = {}) {
    this.k1 = k1;
    this.b = b;
    this.docs = new Map(); // id -> { tf: Map<term,freq>, len }
    this.df = new Map(); // term -> 文档频率
  }

  add(id, terms) {
    const tf = new Map();
    for (const t of terms) tf.set(t, (tf.get(t) || 0) + 1);
    this.docs.set(id, { tf, len: terms.length });
    for (const t of new Set(terms)) this.df.set(t, (this.df.get(t) || 0) + 1);
    return this;
  }

  get size() {
    return this.docs.size;
  }

  idf(term) {
    const N = this.docs.size;
    const df = this.df.get(term) || 0;
    return Math.log(1 + (N - df + 0.5) / (df + 0.5));
  }

  /** 对单篇文档的 BM25 打分。不变量：非负；查询词频越高分越高（边际递减）。 */
  score(queryTerms, id) {
    const doc = this.docs.get(id);
    if (!doc || doc.len === 0) return 0;
    const avgdl = this.#avgdl();
    let s = 0;
    for (const t of queryTerms) {
      const f = doc.tf.get(t) || 0;
      if (f === 0) continue;
      s += this.idf(t) * ((f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + (this.b * doc.len) / avgdl)));
    }
    return s;
  }

  /** top-k 检索，按分数降序。不变量：结果长度 ≤ k；分数降序。 */
  search(queryTerms, k = 10) {
    const scored = [...this.docs.keys()]
      .map((id) => ({ id, score: this.score(queryTerms, id) }))
      .filter((r) => r.score > 0)
      .sort((x, y) => y.score - x.score || (x.id < y.id ? -1 : 1));
    return scored.slice(0, k);
  }

  #avgdl() {
    if (this.docs.size === 0) return 0;
    let s = 0;
    for (const d of this.docs.values()) s += d.len;
    return s / this.docs.size;
  }
}

/**
 * Reciprocal Rank Fusion（Cormack, Clarke & Büttcher, SIGIR 2009）。
 *   score(d) = Σ_i 1 / (k + rank_i(d))，k=60 为论文推荐值。
 * 不变量：融合分数 ∈ (0, n/k]；新增一路排序只会使分数不减（单调性）。
 */
export function rrfFuse(rankings, k = 60) {
  const fused = new Map();
  for (const ranking of rankings) {
    ranking.forEach((id, idx) => {
      fused.set(id, (fused.get(id) || 0) + 1 / (k + idx + 1));
    });
  }
  return [...fused.entries()]
    .map(([id, score]) => ({ id, score }))
    .sort((x, y) => y.score - x.score || (x.id < y.id ? -1 : 1));
}

/**
 * RoPE 旋转位置编码（Su et al., 2021, RoFormer）。
 * 对相邻维度对施加位置相关的旋转；每对旋转保范数 ⇒ 整体保范数。
 * 不变量：‖RoPE(x, m)‖ = ‖x‖；RoPE(x, 0) = x。
 */
export function applyRoPE(vec, position, { base = 10000, dim = vec.length } = {}) {
  const out = vec.slice();
  const d = Math.min(dim, vec.length - (vec.length % 2));
  for (let i = 0; i < d; i += 2) {
    const theta = position / base ** (i / d);
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const x0 = vec[i];
    const x1 = vec[i + 1];
    out[i] = x0 * cos - x1 * sin;
    out[i + 1] = x0 * sin + x1 * cos;
  }
  return out;
}

/**
 * 核采样（top-k + top-p，Holtzman et al., ICLR 2020）。
 * 流程：温度缩放 → top-k 截断 → top-p 核截断（保留累积概率 ≥ p 的最小集合）→ rng 抽取。
 * 不变量：返回值 ∈ 保留集合；temperature ≤ 0 时退化为 argmax；同种子 rng 结果确定。
 */
export function sampleFromLogits(logits, { temperature = 1, topK = 0, topP = 1, rng = Math.random } = {}) {
  const n = logits.length;
  if (n === 0) throw new Error('sampleFromLogits: 空 logits');
  const probs = softmax(logits, temperature);
  let kept = probs.map((p, i) => ({ i, p }));
  if (topK > 0 && topK < kept.length) {
    kept.sort((a, b) => b.p - a.p);
    kept = kept.slice(0, topK);
  }
  if (topP < 1) {
    kept.sort((a, b) => b.p - a.p);
    let acc = 0;
    const nucleus = [];
    for (const item of kept) {
      nucleus.push(item);
      acc += item.p;
      if (acc >= topP) break;
    }
    kept = nucleus;
  }
  const total = kept.reduce((a, b) => a + b.p, 0);
  if (!(total > 0)) return { index: kept[0].i, prob: kept[0].p, kept: kept.map((x) => x.i) };
  let r = rng() * total;
  for (const item of kept) {
    r -= item.p;
    if (r <= 0) return { index: item.i, prob: item.p / total, kept: kept.map((x) => x.i) };
  }
  const last = kept[kept.length - 1];
  return { index: last.i, prob: last.p / total, kept: kept.map((x) => x.i) };
}

/** 安全数值比较：相对误差。 */
export function relErr(a, b) {
  const d = Math.abs(a - b);
  const m = Math.max(Math.abs(a), Math.abs(b), Number.MIN_VALUE);
  return d / m;
}
