/**
 * NovaFlagship · 观测层（Tracer + Metrics）
 * ------------------------------------------------------------------
 * OpenTelemetry 启发的轻量 span 模型（父子嵌套、JSONL 落盘），
 * 零依赖、可 grep、可喂给任何日志管道。
 */

import { ewma } from './math.js';

export class Tracer {
  constructor({ sink } = {}) {
    this.spans = [];
    this.sink = sink; // (span) => void，可接 fs/HTTP
    this._stack = [];
    this._seq = 0;
  }

  /** 同步 span：span(name, attrs, fn) ⇒ fn(span) 的返回值。 */
  span(name, attrs = {}, fn) {
    const span = this.start(name, attrs);
    try {
      const out = fn?.(span);
      span.end();
      return out;
    } catch (err) {
      span.end({ error: String(err?.message || err) });
      throw err;
    }
  }

  /** 异步 span。 */
  async spanAsync(name, attrs, fn) {
    const span = this.start(name, attrs);
    try {
      const out = await fn?.(span);
      span.end();
      return out;
    } catch (err) {
      span.end({ error: String(err?.message || err) });
      throw err;
    }
  }

  start(name, attrs = {}) {
    const span = {
      id: `sp_${(this._seq++).toString(36)}`,
      parentId: this._stack[this._stack.length - 1]?.id ?? null,
      name,
      attrs,
      start: Date.now(),
      end: null,
      durationMs: null,
    };
    this.spans.push(span);
    const tracer = this;
    span.end = (endAttrs = {}) => {
      if (span.end !== null) return span;
      span.end = Date.now();
      span.durationMs = span.end - span.start;
      Object.assign(span.attrs, endAttrs);
      tracer.sink?.(span);
      return span;
    };
    return span;
  }

  /** 手动维护嵌套栈（用于 async 场景）。 */
  enter(span) {
    this._stack.push(span);
  }

  exit() {
    this._stack.pop();
  }

  toJSONL() {
    return this.spans.map((s) => JSON.stringify(s)).join('\n') + (this.spans.length ? '\n' : '');
  }
}

/** 指标：计数器 + EWMA 计时。 */
export class Metrics {
  constructor({ alpha = 0.2 } = {}) {
    this.alpha = alpha;
    this.counters = new Map();
    this._timings = new Map(); // name -> number[]
  }

  incr(name, by = 1) {
    this.counters.set(name, (this.counters.get(name) || 0) + by);
    return this.counters.get(name);
  }

  get(name) {
    return this.counters.get(name) || 0;
  }

  timing(name, ms) {
    const arr = this._timings.get(name) || [];
    arr.push(ms);
    this._timings.set(name, arr);
    return arr[arr.length - 1];
  }

  /** EWMA 平滑值（越近的样本权重越高）。 */
  ewma(name) {
    const arr = this._timings.get(name) || [];
    if (arr.length === 0) return null;
    return ewma(arr, this.alpha)[arr.length - 1];
  }

  snapshot() {
    const out = { counters: Object.fromEntries(this.counters), timings: {} };
    for (const name of this._timings.keys()) {
      out.timings[name] = { last: this._timings.get(name).at(-1), ewma: this.ewma(name) };
    }
    return out;
  }
}

/** 观测套件：trace + metrics 一站式。 */
export function createObservability({ sink } = {}) {
  const metrics = new Metrics();
  const tracer = new Tracer({ sink });
  return { tracer, metrics };
}
