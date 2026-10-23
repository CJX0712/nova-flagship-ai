/**
 * NovaFlagship · L7 工具层
 * ------------------------------------------------------------------
 * - ToolRegistry：JSON Schema 入参校验 + 超时 + 结构化结果（失败也作为观察回流给模型）
 * - 内置工具：now / calc（Pratt parser，禁 eval）/ http_get / fs_read（目录围栏）/
 *   uuid / hash / memory_search / memory_store
 * - McpClient：Model Context Protocol（MCP）stdio 传输客户端，JSON-RPC 2.0，
 *   与 Claude Desktop / 官方 SDK 同一协议——世界标准工具接口，零重写。
 */

import { validateSchema } from './schema.js';
import { randomUUID, createHash } from 'node:crypto';
import { readFile, realpath } from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';

/* ------------------------------------------------------------------ */
/* 表达式求值器（Pratt / 优先级爬升，绝不使用 eval）                     */
/* ------------------------------------------------------------------ */

const FUNCS = {
  sqrt: Math.sqrt, abs: Math.abs, sin: Math.sin, cos: Math.cos, tan: Math.tan,
  log: Math.log, ln: Math.log, log10: Math.log10, log2: Math.log2,
  exp: Math.exp, floor: Math.floor, ceil: Math.ceil, round: Math.round,
  min: (...a) => Math.min(...a), max: (...a) => Math.max(...a),
  pow: (a, b) => a ** b, sign: Math.sign, trunc: Math.trunc, cbrt: Math.cbrt,
};
const CONSTS = { pi: Math.PI, e: Math.E, tau: Math.PI * 2 };

function tokenizeExpr(src) {
  const tokens = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (/\s/.test(c)) { i++; continue; }
    if (/[\d.]/.test(c)) {
      let j = i;
      while (j < src.length && /[\d._eE]/.test(src[j])) {
        if ((src[j] === 'e' || src[j] === 'E') && /[+-]/.test(src[j + 1] || '') && /\d/.test(src[j - 1] || '')) j++;
        else if ('eE'.includes(src[j]) && !/\d/.test(src[j + 1] || '') && !/\d/.test(src[j - 1] || '')) break;
        else j++;
      }
      const num = Number(src.slice(i, j).replace(/_/g, ''));
      if (!Number.isFinite(num)) throw new Error(`非法数字: ${src.slice(i, j)}`);
      tokens.push({ type: 'num', value: num });
      i = j;
      continue;
    }
    if (/[a-zA-Z_]/.test(c)) {
      let j = i;
      while (j < src.length && /[\w]/.test(src[j])) j++;
      tokens.push({ type: 'ident', value: src.slice(i, j) });
      i = j;
      continue;
    }
    if ('+-*/%^(),'.includes(c)) { tokens.push({ type: c }); i++; continue; }
    throw new Error(`非法字符: ${c}`);
  }
  return tokens;
}

/** 优先级爬升解析器。不变量：^ 右结合；一元负号最高优先；除零抛错。 */
export function evalExpression(src) {
  const tokens = tokenizeExpr(String(src));
  let pos = 0;
  const peek = () => tokens[pos];
  const eat = (type) => (peek()?.type === type ? tokens[pos++] : null);

  const BP = { '+': 1, '-': 1, '*': 2, '/': 2, '%': 2, '^': 4 };

  function parseExpr(minBp = 0) {
    let lhs = parseUnary();
    for (;;) {
      const t = peek();
      const bp = t && BP[t.type];
      if (!bp || bp < minBp) break;
      pos++;
      const nextMinBp = t.type === '^' ? bp : bp + 1; // ^ 右结合
      const rhs = parseExpr(nextMinBp);
      lhs = apply(t.type, lhs, rhs);
    }
    return lhs;
  }

  function parseUnary() {
    const t = peek();
    if (t?.type === '-') { pos++; return -parseUnary(); }
    if (t?.type === '+') { pos++; return parseUnary(); }
    return parsePrimary();
  }

  function parsePrimary() {
    const t = peek();
    if (!t) throw new Error('表达式意外结束');
    if (t.type === 'num') { pos++; return t.value; }
    if (t.type === '(') {
      pos++;
      const v = parseExpr(0);
      if (!eat(')')) throw new Error('缺少右括号');
      return v;
    }
    if (t.type === 'ident') {
      pos++;
      if (peek()?.type === '(') {
        pos++;
        const args = [];
        if (peek()?.type !== ')') {
          for (;;) {
            args.push(parseExpr(0));
            if (!eat(',')) break;
          }
        }
        if (!eat(')')) throw new Error(`函数 ${t.value} 缺少右括号`);
        const fn = FUNCS[t.value.toLowerCase()];
        if (!fn) throw new Error(`未知函数: ${t.value}`);
        return fn(...args);
      }
      const k = t.value.toLowerCase();
      if (k in CONSTS) return CONSTS[k];
      throw new Error(`未定义的符号: ${t.value}`);
    }
    throw new Error(`意外的记号: ${t.type}`);
  }

  function apply(op, a, b) {
    switch (op) {
      case '+': return a + b;
      case '-': return a - b;
      case '*': return a * b;
      case '/': if (b === 0) throw new Error('除零错误'); return a / b;
      case '%': if (b === 0) throw new Error('除零错误'); return a % b;
      case '^': return a ** b;
      default: throw new Error(`未知运算符: ${op}`);
    }
  }

  const value = parseExpr(0);
  if (pos < tokens.length) throw new Error(`表达式末尾有多余记号: ${tokens[pos].type}`);
  if (!Number.isFinite(value)) throw new Error('结果非有限数');
  return value;
}

/* ------------------------------------------------------------------ */
/* ToolRegistry                                                        */
/* ------------------------------------------------------------------ */

export class ToolRegistry {
  constructor({ defaultTimeoutMs = 15000, context = {} } = {}) {
    this.tools = new Map();
    this.defaultTimeoutMs = defaultTimeoutMs;
    this.context = context; // 注入 memory 等运行时依赖
  }

  /** 注册工具。def: { name, description, inputSchema, handler(args, ctx) } */
  register(def) {
    if (!def?.name || typeof def.handler !== 'function') throw new Error('工具需要 name 与 handler');
    if (this.tools.has(def.name)) throw new Error(`工具重复注册: ${def.name}`);
    this.tools.set(def.name, {
      name: def.name,
      description: def.description || '',
      inputSchema: def.inputSchema || { type: 'object', properties: {} },
      handler: def.handler,
      timeoutMs: def.timeoutMs || this.defaultTimeoutMs,
    });
    return this;
  }

  registerMcp(tools) {
    for (const t of tools) {
      this.register({
        name: `mcp__${t.serverName}__${t.name}`,
        description: `[MCP:${t.serverName}] ${t.description || ''}`,
        inputSchema: t.inputSchema,
        handler: async (args) => t.call(args),
      });
    }
    return this;
  }

  list() {
    return [...this.tools.values()].map(({ name, description, inputSchema }) => ({ name, description, inputSchema }));
  }

  has(name) {
    return this.tools.has(name);
  }

  /** 执行工具：schema 校验 → 超时控制 → 调用 → 结构化结果。失败也是有效观察。 */
  async execute(name, args = {}, { signal, context } = {}) {
    const tool = this.tools.get(name);
    if (!tool) return { ok: false, error: `未知工具: ${name}`, ms: 0 };
    const started = Date.now();
    const errors = validateSchema(args, tool.inputSchema);
    if (errors.length > 0) {
      return { ok: false, error: `入参校验失败: ${errors.map((e) => `${e.path || '<root>'} ${e.message}`).join('; ')}`, ms: Date.now() - started };
    }
    let timer;
    try {
      const result = await Promise.race([
        tool.handler(args, { ...this.context, ...context }),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error(`工具超时(${tool.timeoutMs}ms)`)), tool.timeoutMs);
          signal?.addEventListener('abort', () => reject(new Error('工具调用被取消')), { once: true });
        }),
      ]);
      return { ok: true, output: result, ms: Date.now() - started };
    } catch (err) {
      return { ok: false, error: String(err?.message || err), ms: Date.now() - started };
    } finally {
      clearTimeout(timer);
    }
  }
}

/* ------------------------------------------------------------------ */
/* 内置工具集                                                          */
/* ------------------------------------------------------------------ */

/** 注册全部内置工具（依赖注入：memory、fsRoots 等）。 */
export function registerBuiltinTools(registry, { memory = null, fsRoots = [], allowHttp = true } = {}) {
  // 把 memory 挂到 registry.context：回放引擎据此定位有状态工具并做状态隔离
  if (memory) registry.context = { ...registry.context, memory };
  registry.register({
    name: 'now',
    description: '获取当前日期时间（本地时区人话格式 + ISO + Unix 时间戳）',
    inputSchema: { type: 'object', properties: {} },
    handler: () => {
      const d = new Date();
      return { iso: d.toISOString(), unix: d.getTime(), local: d.toString() };
    },
  });

  registry.register({
    name: 'calc',
    description: '精确计算数学表达式（支持 + - * / % ^、括号、sqrt/sin/cos/log/exp/min/max 等函数与 pi/e 常量）',
    inputSchema: {
      type: 'object',
      properties: { expression: { type: 'string', minLength: 1, description: '如 (3+4)*2^10' } },
      required: ['expression'],
      additionalProperties: false,
    },
    handler: ({ expression }) => ({ expression, value: evalExpression(expression) }),
  });

  registry.register({
    name: 'uuid',
    description: '生成随机 UUID v4',
    inputSchema: { type: 'object', properties: {} },
    handler: () => ({ uuid: randomUUID() }),
  });

  registry.register({
    name: 'hash',
    description: '计算文本的 SHA-256 摘要',
    inputSchema: {
      type: 'object',
      properties: { text: { type: 'string' }, algo: { type: 'string', enum: ['sha256', 'sha1', 'md5'] } },
      required: ['text'],
    },
    handler: ({ text, algo = 'sha256' }) => ({ algo, digest: createHash(algo).update(String(text)).digest('hex') }),
  });

  if (allowHttp) {
    registry.register({
      name: 'http_get',
      description: 'GET 抓取网页/API（仅 http/https，响应截断至 200KB）',
      inputSchema: {
        type: 'object',
        properties: {
          url: { type: 'string', pattern: '^https?://' },
          maxChars: { type: 'integer', minimum: 1, maximum: 200000 },
        },
        required: ['url'],
      },
      timeoutMs: 20000,
      handler: async ({ url, maxChars = 20000 }) => {
        const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
        const text = await res.text();
        return { status: res.status, contentType: res.headers.get('content-type'), body: text.slice(0, maxChars) };
      },
    });
  }

  if (fsRoots.length > 0) {
    registry.register({
      name: 'fs_read',
      description: `读取文件（仅限目录围栏: ${fsRoots.join(', ')}，最大 512KB）`,
      inputSchema: {
        type: 'object',
        properties: { path: { type: 'string' } },
        required: ['path'],
      },
      handler: async ({ path: p }) => {
        const resolved = path.resolve(p);
        const roots = await Promise.all(fsRoots.map((r) => realpath(r)));
        const real = await realpath(resolved);
        if (!roots.some((r) => real === r || real.startsWith(r + path.sep))) {
          throw new Error(`路径越权: ${real} 不在围栏内`);
        }
        const text = await readFile(real, 'utf8');
        return { path: real, chars: text.length, content: text.slice(0, 512 * 1024) };
      },
    });
  }

  if (memory) {
    registry.register({
      name: 'memory_search',
      description: '语义+关键词混合检索长期记忆（BM25 ⊕ 稠密 ⊕ RRF 融合）',
      inputSchema: {
        type: 'object',
        properties: { query: { type: 'string', minLength: 1 }, k: { type: 'integer', minimum: 1, maximum: 50 } },
        required: ['query'],
      },
      handler: async ({ query, k = 5 }) => {
        const hits = await memory.recall(query, k);
        return { count: hits.length, hits: hits.map((h) => ({ id: h.id, text: h.text, score: Number(h.score.toFixed(4)) })) };
      },
    });
    registry.register({
      name: 'memory_store',
      description: '把一条事实/经验写入长期记忆',
      inputSchema: {
        type: 'object',
        properties: { text: { type: 'string', minLength: 1 }, salience: { type: 'number', minimum: 0, maximum: 1 } },
        required: ['text'],
      },
      handler: async ({ text, salience = 0.5 }) => {
        const id = await memory.remember(text, { salience, source: 'agent' });
        return { id, stored: true };
      },
    });
  }

  return registry;
}

/* ------------------------------------------------------------------ */
/* MCP 客户端（stdio 传输，JSON-RPC 2.0）                               */
/* ------------------------------------------------------------------ */

export class McpClient {
  /**
   * @param {object} opts { name, command, args, env, timeoutMs }
   */
  constructor({ name, command, args = [], env = {}, timeoutMs = 20000 }) {
    this.name = name;
    this.command = command;
    this.args = args;
    this.env = env;
    this.timeoutMs = timeoutMs;
    this.proc = null;
    this.nextId = 1;
    this.pending = new Map(); // id -> {resolve, reject, timer}
    this.buf = '';
    this.initialized = false;
  }

  async start() {
    this.proc = spawn(this.command, this.args, {
      env: { ...process.env, ...this.env },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.proc.stdout.on('data', (d) => this.#onData(d));
    this.proc.stderr.on('data', () => {}); // MCP 日志走 stderr，忽略
    this.proc.on('exit', (code) => {
      for (const [, p] of this.pending) {
        clearTimeout(p.timer);
        p.reject(new Error(`MCP 进程退出(code=${code})`));
      }
      this.pending.clear();
    });
    await this.#request('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'nova-flagship', version: '1.0.0' },
    });
    this.#notify('notifications/initialized', {});
    this.initialized = true;
    return this;
  }

  async listTools() {
    const res = await this.#request('tools/list', {});
    return (res.tools || []).map((t) => ({ ...t, serverName: this.name }));
  }

  async callTool(toolName, args = {}) {
    const res = await this.#request('tools/call', { name: toolName, arguments: args });
    const text = (res.content || []).filter((c) => c.type === 'text').map((c) => c.text).join('\n');
    if (res.isError) throw new Error(text || 'MCP 工具执行错误');
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }

  close() {
    this.proc?.kill();
    this.proc = null;
    this.initialized = false;
  }

  #onData(chunk) {
    this.buf += chunk.toString('utf8');
    let nl;
    while ((nl = this.buf.indexOf('\n')) >= 0) {
      const line = this.buf.slice(0, nl).trim();
      this.buf = this.buf.slice(nl + 1);
      if (!line) continue;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        continue;
      }
      if (msg.id !== undefined && this.pending.has(msg.id)) {
        const p = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        clearTimeout(p.timer);
        if (msg.error) p.reject(new Error(`MCP error ${msg.error.code}: ${msg.error.message}`));
        else p.resolve(msg.result);
      }
    }
  }

  #request(method, params) {
    const id = this.nextId++;
    const payload = JSON.stringify({ jsonrpc: '2.0', id, method, params });
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP 请求超时: ${method}`));
      }, this.timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.proc.stdin.write(payload + '\n');
    });
  }

  #notify(method, params) {
    this.proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n');
  }
}
