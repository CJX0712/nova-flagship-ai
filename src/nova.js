/**
 * NovaFlagship · 公共 API 总装线
 * ------------------------------------------------------------------
 * createNova() 一键装配完整栈：网关 + 工具 + 三段记忆 + 智能体 + RAG +
 * 微操作目录 + 观测 + HTTP 服务。全部模块亦可单独引入使用。
 */

import { ModelGateway, CostLedger, CircuitBreaker, retryWithBackoff, ProviderError } from './core/gateway.js';
import { novaLite, mockProvider, providerFromEnv, openaiCompatible, llamaCppProvider, ollamaProvider } from './core/providers.js';
import { ToolRegistry, registerBuiltinTools, McpClient, evalExpression } from './core/tools.js';
import { MemorySystem, HashEmbedder, WorkingMemory, EpisodicMemory, SemanticMemory } from './core/memory.js';
import { AgentRuntime, TraceRecorder, replayAgent, tracesEqual } from './core/agent.js';
import { RagPipeline, chunkText } from './core/rag.js';
import { createObservability, Tracer, Metrics } from './core/observe.js';
import { createOps, FAMILIES } from './core/ops.js';
import { createServer, attachOpsMeta } from './core/server.js';
import * as math from './core/math.js';
import * as tokenize from './core/tokenize.js';
import { validateSchema, validateOrThrow } from './core/schema.js';

export const VERSION = '1.0.0';

/**
 * 装配完整 Nova 栈。
 * @param {object} [options]
 * @param {object} [options.provider] 自定义提供商（默认 nova-lite 或环境变量构建）
 * @param {object} [options.gateway]  自定义网关
 * @param {object} [options.tools]    自定义工具注册表
 * @param {object} [options.memory]   自定义记忆系统
 * @param {object} [options.agent]    AgentRuntime 附加参数（systemPrompt/budget/model）
 * @param {string[]} [options.fsRoots] fs_read 工具的目录围栏
 */
export function createNova(options = {}) {
  const observe = options.observe || createObservability();
  const memory = options.memory || new MemorySystem({ embedder: options.embedder });

  // 观测适配器：网关/智能体只依赖 { span, metrics } 两个钩子
  const observeAdapter = options.observeAdapter || {
    span: (name, attrs, fn) => observe.tracer.span(name, attrs, fn),
    metrics: observe.metrics,
  };

  const gateway = options.gateway || new ModelGateway({ observe: observeAdapter, pricing: options.pricing });
  if (!options.gateway) {
    gateway.register(options.provider || providerFromEnv(), { isDefault: true });
  }

  const tools = options.tools || new ToolRegistry();
  if (!options.tools) {
    registerBuiltinTools(tools, { memory, fsRoots: options.fsRoots || [], allowHttp: options.allowHttp !== false });
  }

  const agent = new AgentRuntime({
    gateway,
    tools,
    memory,
    observe: observeAdapter,
    systemPrompt: options.systemPrompt,
    model: options.model,
    budget: options.budget,
    ...(options.agent || {}),
  });

  const rag = new RagPipeline({ memory, ...(options.rag || {}) });
  const ops = createOps();
  attachOpsMeta(FAMILIES);
  const server = createServer({ gateway, agent, memory, rag, ops, observe, version: VERSION });

  return {
    version: VERSION,
    gateway,
    tools,
    memory,
    agent,
    rag,
    ops,
    observe,
    server,
    async serve({ port = 8780, host = '127.0.0.1' } = {}) {
      await new Promise((resolve) => server.listen(port, host, resolve));
      const actualPort = server.address().port; // port=0 时取系统分配的临时端口
      return { url: `http://${host}:${actualPort}`, port: actualPort, host, close: () => server.close() };
    },
  };
}

export {
  // 网关与提供商
  ModelGateway, CostLedger, CircuitBreaker, retryWithBackoff, ProviderError,
  novaLite, mockProvider, providerFromEnv, openaiCompatible, llamaCppProvider, ollamaProvider,
  // 工具
  ToolRegistry, registerBuiltinTools, McpClient, evalExpression,
  // 记忆
  MemorySystem, HashEmbedder, WorkingMemory, EpisodicMemory, SemanticMemory,
  // 智能体
  AgentRuntime, TraceRecorder, replayAgent, tracesEqual,
  // RAG
  RagPipeline, chunkText,
  // 观测
  createObservability, Tracer, Metrics,
  // 微操作
  createOps, FAMILIES,
  // 数学与协议
  math, tokenize, validateSchema, validateOrThrow,
};
