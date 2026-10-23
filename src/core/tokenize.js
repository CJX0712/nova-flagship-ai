/**
 * NovaFlagship · L3 分词与计量层
 * ------------------------------------------------------------------
 * 零依赖近似分词器（OpenAI 官方公开的启发式）：
 *   - 英文约 4 字符/token，CJK 约 1 字符/token；
 *   - 消息级开销沿用 OpenAI 公开公式：每消息 4 token + 2 token 回复引导。
 * 精确计量两条路：① 提供商 usage 字段（网关自动采集）；② 适配 tiktoken（可选增强）。
 * 不变量：非负、空串为 0、长度单调、CJK 串 ≈ 字符数。
 */

const CJK_RE = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]/g;

/** 估算单段文本的 token 数。 */
export function estimateTokens(text) {
  if (typeof text !== 'string' || text.length === 0) return 0;
  const cjk = (text.match(CJK_RE) || []).length;
  const ascii = text.length - cjk;
  return cjk + Math.ceil(ascii / 4);
}

/** 估算整条消息列表的 token 数（OpenAI 公开启发式公式）。 */
export function estimateMessagesTokens(messages, { perMessage = 4, replyPrimer = 2 } = {}) {
  if (!Array.isArray(messages)) return 0;
  let total = replyPrimer;
  for (const m of messages) {
    total += perMessage;
    if (typeof m.content === 'string') total += estimateTokens(m.content);
    else if (Array.isArray(m.content)) {
      for (const part of m.content) {
        if (part?.type === 'text') total += estimateTokens(part.text);
        else total += perMessage; // 图片等多模态部分按固定开销估算
      }
    }
    if (m.name) total += estimateTokens(m.name);
    if (Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) {
        total += perMessage + estimateTokens(tc.function?.name || '') + estimateTokens(tc.function?.arguments || '');
      }
    }
  }
  return total;
}

/** 按 token 预算贪心装入文本块（高分优先），返回入选项与总消耗。
 *  不变量：总消耗 ≤ 预算；入选集合是「按分数降序贪心」的最大可行前缀。
 *  tokensOf 默认依次尝试 item.text / item 自身字符串化。 */
export function packToBudget(items, budget, tokensOf = (x) => estimateTokens(typeof x === 'string' ? x : (x?.text ?? JSON.stringify(x)))) {
  const sorted = [...items].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  const picked = [];
  let used = 0;
  for (const item of sorted) {
    const cost = tokensOf(item);
    if (used + cost > budget) continue; // 跳过放不下的，继续尝试更小的（背包贪心）
    picked.push(item);
    used += cost;
  }
  return { picked, used, budget };
}

/** 简易词元化（检索用）：小写化 → CJK 单字 → 英文/数字词。 */
export function tokenizeWords(text) {
  if (typeof text !== 'string') return [];
  const out = [];
  const cjk = text.match(/[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/g) || [];
  out.push(...cjk);
  const words = text.toLowerCase().match(/[a-z0-9][a-z0-9'_\-]*/g) || [];
  out.push(...words);
  return out;
}

/** 字符 n-gram（用于无模型的稠密检索兜底）。 */
export function charNgrams(text, n = 3) {
  const s = String(text || '').toLowerCase().replace(/\s+/g, ' ');
  const out = [];
  for (let i = 0; i + n <= s.length; i++) out.push(s.slice(i, i + n));
  if (out.length === 0 && s.length > 0) out.push(s);
  return out;
}
