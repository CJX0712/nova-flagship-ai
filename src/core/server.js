/**
 * NovaFlagship · L10 服务层（零依赖 HTTP API）
 * ------------------------------------------------------------------
 * 端点：
 *   GET  /api/health                  健康检查 + 拓扑
 *   POST /v1/chat/completions         OpenAI 兼容补全（支持 stream=SSE）
 *   POST /v1/agents                   运行智能体（?stream=1 ⇒ SSE 事件流）
 *   GET  /v1/agents/:id/trace         轨迹查询
 *   POST /v1/memory/recall            记忆召回
 *   POST /v1/memory/remember          记忆写入
 *   POST /v1/rag/context              RAG 上下文组装
 *   GET  /api/ops                     微操作目录
 *   GET  /api/metrics                 指标快照
 *   GET  /                            驾驶舱 UI（单文件 cockpit.html）
 */

import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const COCKPIT = path.join(HERE, '..', '..', 'web', 'cockpit.html');
const VERSION = '1.0.0';

export function createServer({ gateway, agent, memory, rag, ops, observe, version = VERSION } = {}) {
  const runs = new Map(); // id -> AgentResult

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const route = `${req.method} ${url.pathname}`;
    try {
      // CORS + JSON 助手
      res.setHeader('access-control-allow-origin', '*');
      res.setHeader('access-control-allow-headers', 'content-type');
      res.setHeader('access-control-allow-methods', 'GET,POST,OPTIONS');
      if (req.method === 'OPTIONS') {
        res.writeHead(204).end();
        return;
      }

      if (route === 'GET /api/health') {
        return json(res, 200, {
          ok: true,
          version,
          providers: gateway ? [...gateway.providers.keys()] : [],
          defaultProvider: gateway?.defaultProvider ?? null,
          tools: agent?.tools?.list().map((t) => t.name) ?? [],
          memory: memory?.stats() ?? null,
          ops: ops?.size ?? 0,
        });
      }

      if (route === 'POST /v1/chat/completions') {
        const body = await readJson(req);
        if (!Array.isArray(body.messages)) return json(res, 400, { error: { message: 'messages 必填' } });
        if (body.stream) {
          res.writeHead(200, sseHeaders());
          for await (const ev of gateway.chatStream({
            model: body.model,
            messages: body.messages,
            tools: body.tools,
            temperature: body.temperature,
            maxTokens: body.max_tokens,
          })) {
            if (ev.type === 'done') {
              sse(res, { id: ev.response.id, object: 'chat.completion', choices: [{ index: 0, message: ev.response.message, finish_reason: ev.response.finishReason }], usage: toOpenAIUsage(ev.response.usage) });
              sse(res, '[DONE]');
            } else if (ev.type === 'content') {
              sse(res, { id: 'chatcmpl-stream', object: 'chat.completion.chunk', choices: [{ index: 0, delta: { content: ev.text } }] });
            }
          }
          return res.end();
        }
        const out = await gateway.chat({ model: body.model, messages: body.messages, tools: body.tools, temperature: body.temperature, maxTokens: body.max_tokens });
        return json(res, 200, {
          id: out.id,
          object: 'chat.completion',
          model: out.model,
          choices: [{ index: 0, message: out.message, finish_reason: out.finishReason }],
          usage: toOpenAIUsage(out.usage),
          nova: { provider: out.provider, costUsd: out.costUsd, ms: out.ms },
        });
      }

      if (route === 'POST /v1/agents') {
        const body = await readJson(req);
        if (!body.task) return json(res, 400, { error: { message: 'task 必填' } });
        const id = `run_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
        if (url.searchParams.get('stream') === '1') {
          res.writeHead(200, sseHeaders());
          // 简化：跑完后按事件流推送（保持协议简单确定）
          const result = await agent.run({ task: body.task, budget: body.budget });
          runs.set(id, result);
          for (const ev of result.trace.events) {
            sse(res, { type: 'trace', event: ev });
          }
          sse(res, { type: 'result', id, answer: result.answer, usage: result.usage, costUsd: result.costUsd, ms: result.ms, finishReason: result.finishReason });
          return res.end();
        }
        const result = await agent.run({ task: body.task, budget: body.budget });
        runs.set(id, result);
        return json(res, 200, { id, answer: result.answer, steps: result.steps, usage: result.usage, costUsd: result.costUsd, ms: result.ms, finishReason: result.finishReason, trace: result.trace.toJSON() });
      }

      const traceMatch = /^GET \/v1\/agents\/([\w-]+)\/trace$/.exec(route);
      if (traceMatch) {
        const run = runs.get(traceMatch[1]);
        if (!run) return json(res, 404, { error: { message: 'run 不存在' } });
        return json(res, 200, run.trace.toJSON());
      }

      if (route === 'POST /v1/memory/recall') {
        const body = await readJson(req);
        const hits = await memory.recall(body.query, body.k ?? 5);
        return json(res, 200, { hits });
      }

      if (route === 'POST /v1/memory/remember') {
        const body = await readJson(req);
        const id = await memory.remember(body.text, { salience: body.salience ?? 0.5 });
        return json(res, 200, { id, stats: memory.stats() });
      }

      if (route === 'POST /v1/rag/context') {
        const body = await readJson(req);
        const ctx = await rag.buildContext(body.query, { k: body.k, maxTokens: body.maxTokens });
        return json(res, 200, ctx);
      }

      if (route === 'GET /api/ops') {
        return json(res, 200, { families: FAMILIES_SAFE, ops: ops.list() });
      }

      if (route === 'GET /api/metrics') {
        return json(res, 200, observe?.metrics?.snapshot() ?? {});
      }

      if (route === 'GET /' || route === 'GET /cockpit.html') {
        const html = await readFile(COCKPIT, 'utf8');
        res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
        return res.end(html);
      }

      return json(res, 404, { error: { message: `未找到路由: ${route}` } });
    } catch (err) {
      return json(res, 500, { error: { message: String(err?.message || err) } });
    }
  });

  server.runs = runs;
  return server;
}

// 避免循环依赖： families 由 ops 模块按需注入
let FAMILIES_SAFE = [];
export function attachOpsMeta(families) {
  FAMILIES_SAFE = families;
}

function sseHeaders() {
  return {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  };
}

function sse(res, data) {
  res.write(`data: ${typeof data === 'string' ? data : JSON.stringify(data)}\n\n`);
}

function json(res, status, obj) {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(obj));
}

async function readJson(req, limit = 4 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const c of req) {
    size += c.length;
    if (size > limit) throw new Error('请求体超过 4MB 限制');
    chunks.push(c);
  }
  if (chunks.length === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

function toOpenAIUsage(usage) {
  if (!usage) return undefined;
  return {
    prompt_tokens: usage.promptTokens || 0,
    completion_tokens: usage.completionTokens || 0,
    total_tokens: usage.totalTokens || 0,
  };
}
