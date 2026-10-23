/**
 * NovaFlagship · L4 模型网关层
 * ------------------------------------------------------------------
 * 统一抽象：任何「OpenAI 兼容」端点（OpenAI / DeepSeek / Moonshot / llama.cpp /
 * Ollama / vLLM / SGLang …）与本地确定性提供商走同一条 chat 接口。
 *
 * 世界级复用边界：
 *   - 协议：OpenAI Chat Completions（事实标准，含 tools + SSE 流式）
 *   - 韧性：指数退避 + Full Jitter（AWS 架构博客标准做法）、熔断器（Release It! 模式）
 *   - 计量：usage 账本 + 成本账本（prompt/completion 分开计价）
 *
 * 不变量（tests/gateway.test.js 背书）：
 *   - 重试次数 ≤ retries+1；Full Jitter 延迟 ∈ [0, cap]；abort 立即拒绝
 *   - 熔断器：连续失败达阈值 ⇒ open；冷却后半开探测；成功即闭合
 *   - 账本：total = Σ 每笔 in·priceIn + out·priceOut
 */

import { estimateMessagesTokens } from './tokenize.js';

/** 结构化提供商错误。 */
export class ProviderError extends Error {
  constructor(message, { provider = 'unknown', status = 0, retryable = false, cause } = {}) {
    super(message);
    this.name = 'ProviderError';
    this.provider = provider;
    this.status = status;
    this.retryable = retryable;
    if (cause) this.cause = cause;
  }
}

/** 合并 AbortSignal（Node ≥ 20）。 */
function anySignal(signals) {
  const valid = signals.filter(Boolean);
  if (valid.length === 0) return undefined;
  if (valid.length === 1) return valid[0];
  return AbortSignal.any(valid);
}

/**
 * 指数退避 + Full Jitter 重试。
 * delay_n = random(0, min(cap, base·2^n)) —— AWS "Exponential Backoff And Jitter" 标准方案。
 */
export async function retryWithBackoff(fn, {
  retries = 3,
  baseMs = 250,
  maxMs = 8000,
  signal,
  shouldRetry = (err) => err instanceof ProviderError && err.retryable,
  onRetry,
  rng = Math.random,
  sleep = (ms) => new Promise((r) => setTimeout(r, ms)),
} = {}) {
  let attempt = 0;
  for (;;) {
    try {
      return await fn(attempt);
    } catch (err) {
      if (attempt >= retries || signal?.aborted || !shouldRetry(err)) throw err;
      const cap = Math.min(maxMs, baseMs * 2 ** attempt);
      const delay = cap * rng(); // Full Jitter
      onRetry?.({ attempt: attempt + 1, delay, error: err });
      await sleep(delay);
      attempt++;
    }
  }
}

/** 熔断器：closed → open（连续失败达阈值）→ half-open（冷却后限流探测）→ closed。 */
export class CircuitBreaker {
  constructor({ failureThreshold = 5, cooldownMs = 30000, now = () => Date.now() } = {}) {
    this.failureThreshold = failureThreshold;
    this.cooldownMs = cooldownMs;
    this.now = now;
    this.state = 'closed';
    this.failures = 0;
    this.openedAt = 0;
    this.halfOpenInFlight = 0;
  }

  canRequest() {
    if (this.state === 'open') {
      if (this.now() - this.openedAt >= this.cooldownMs) {
        this.state = 'half-open';
        this.halfOpenInFlight = 0;
      } else {
        return false;
      }
    }
    if (this.state === 'half-open') {
      if (this.halfOpenInFlight >= 1) return false;
      this.halfOpenInFlight++;
    }
    return true;
  }

  onSuccess() {
    this.failures = 0;
    this.state = 'closed';
    this.halfOpenInFlight = 0;
  }

  onFailure() {
    this.failures++;
    this.halfOpenInFlight = 0;
    if (this.state === 'half-open' || this.failures >= this.failureThreshold) {
      this.state = 'open';
      this.openedAt = this.now();
    }
  }
}

/** 成本账本：按模型分别累计 prompt/completion token 与美元成本。 */
export class CostLedger {
  constructor({ pricing = {} } = {}) {
    // pricing: { 'model': { in: 每千 prompt token 美元, out: 每千 completion token 美元 } }
    this.pricing = pricing;
    this.entries = [];
  }

  record({ model, promptTokens = 0, completionTokens = 0 }) {
    const price = this.pricing[model] || { in: 0, out: 0 };
    const cost = (promptTokens * price.in + completionTokens * price.out) / 1000;
    this.entries.push({ model, promptTokens, completionTokens, cost, at: Date.now() });
    return cost;
  }

  total() {
    return this.entries.reduce((a, e) => a + e.cost, 0);
  }

  totalsByModel() {
    const out = new Map();
    for (const e of this.entries) {
      const cur = out.get(e.model) || { promptTokens: 0, completionTokens: 0, cost: 0, calls: 0 };
      cur.promptTokens += e.promptTokens;
      cur.completionTokens += e.completionTokens;
      cur.cost += e.cost;
      cur.calls++;
      out.set(e.model, cur);
    }
    return out;
  }
}

/**
 * 模型网关：注册提供商 → 统一 chat（含流式）→ 自动重试/熔断/记账。
 *
 * provider 契约（世界最小接口）：
 *   { name, chat(request, ctx) -> Promise<Completion>, chatStream?(request, ctx) -> AsyncGenerator<Event> }
 * request: { model, messages, tools?, temperature?, maxTokens?, signal? }
 * Completion: { message:{role,content,tool_calls?}, usage, finishReason, model, provider }
 */
export class ModelGateway {
  constructor({ observe, pricing, retry, breaker } = {}) {
    this.providers = new Map();
    this.defaultProvider = null;
    this.observe = observe;
    this.ledger = new CostLedger({ pricing });
    this.breakers = new Map();
    this.retry = { retries: 2, baseMs: 250, maxMs: 8000, ...retry };
    this.breakerOpts = { failureThreshold: 5, cooldownMs: 30000, ...breaker };
  }

  register(provider, { isDefault = false } = {}) {
    if (!provider?.name || typeof provider.chat !== 'function') {
      throw new Error('provider 必须有 name 与 chat()');
    }
    this.providers.set(provider.name, provider);
    this.breakers.set(provider.name, new CircuitBreaker(this.breakerOpts));
    if (isDefault || !this.defaultProvider) this.defaultProvider = provider.name;
    return this;
  }

  /** 解析 "provider:model" 或裸模型名。 */
  resolve(model) {
    if (typeof model === 'string' && model.includes(':')) {
      const [pname, ...rest] = model.split(':');
      const provider = this.providers.get(pname);
      if (provider) return { provider, model: rest.join(':') };
    }
    const provider = this.providers.get(this.defaultProvider);
    if (!provider) throw new ProviderError('未注册任何提供商', { provider: 'gateway' });
    return { provider, model };
  }

  async chat(request = {}) {
    const { provider, model } = this.resolve(request.model ?? provider_default(this));
    const breaker = this.breakers.get(provider.name);
    if (!breaker.canRequest()) {
      throw new ProviderError(`提供商 ${provider.name} 熔断中`, { provider: provider.name, retryable: true });
    }
    const started = Date.now();
    try {
      const res = await retryWithBackoff(
        (attempt) =>
          provider.chat({ ...request, model }, { signal: request.signal, attempt }),
        {
          ...this.retry,
          signal: request.signal,
          shouldRetry: (err) => err?.retryable === true || (err instanceof ProviderError && err.retryable),
        },
      );
      breaker.onSuccess();
      return this.#finalize(provider, model, res, started);
    } catch (err) {
      breaker.onFailure();
      throw err;
    }
  }

  /** 流式 chat：async generator，依次产出 content / tool_call / usage / done 事件。 */
  async *chatStream(request = {}) {
    const { provider, model } = this.resolve(request.model ?? provider_default(this));
    if (typeof provider.chatStream !== 'function') {
      // 提供商不支持流式 ⇒ 退化为一次性返回
      const res = await this.chat({ ...request, model });
      yield { type: 'content', text: res.message.content || '' };
      for (const tc of res.message.tool_calls || []) yield { type: 'tool_call', toolCall: tc };
      yield { type: 'usage', usage: res.usage };
      yield { type: 'done', response: res };
      return;
    }
    const breaker = this.breakers.get(provider.name);
    if (!breaker.canRequest()) {
      throw new ProviderError(`提供商 ${provider.name} 熔断中`, { provider: provider.name, retryable: true });
    }
    const started = Date.now();
    let response = null;
    try {
      for await (const ev of provider.chatStream({ ...request, model }, { signal: request.signal })) {
        if (ev.type === 'done') response = ev.response;
        else yield ev;
      }
      breaker.onSuccess();
      const finalized = this.#finalize(provider, model, response, started);
      yield { type: 'usage', usage: finalized.usage };
      yield { type: 'done', response: finalized };
    } catch (err) {
      breaker.onFailure();
      throw err;
    }
  }

  #finalize(provider, model, res, started) {
    const usage = res.usage || {
      promptTokens: estimateMessagesTokens(res.__request?.messages || []),
      completionTokens: 0,
      totalTokens: 0,
    };
    usage.totalTokens = usage.totalTokens || (usage.promptTokens || 0) + (usage.completionTokens || 0);
    const cost = this.ledger.record({
      model,
      promptTokens: usage.promptTokens || 0,
      completionTokens: usage.completionTokens || 0,
    });
    const out = {
      id: res.id || `cmpl-${Date.now().toString(36)}`,
      model,
      provider: provider.name,
      message: res.message,
      usage,
      finishReason: res.finishReason || 'stop',
      costUsd: cost,
      ms: Date.now() - started,
    };
    this.observe?.span?.('llm.chat', { model, provider: provider.name, usage }, () => {});
    return out;
  }
}

function provider_default(gw) {
  return gw.providers.get(gw.defaultProvider)?.name || 'default';
}
