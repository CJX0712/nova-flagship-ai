# ✦ NovaFlagship · 创新旗舰 AI

> **零运行时依赖的智能体引擎**：OpenAI 兼容网关 · MCP 工具协议 · 三段式记忆 · 混合检索 · **确定性回放验证**
>
> 作者：**晨星**（CJX0712） · 许可证：MIT · 版本：1.0.0

[![Node](https://img.shields.io/badge/node-%3E%3D20-brightgreen)](https://nodejs.org)
[![Deps](https://img.shields.io/badge/runtime%20deps-0-blue)](package.json)
[![Tests](https://img.shields.io/badge/tests-81%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## 一句话

把世界顶级成果**组装**成一台可离线复现的 AI 引擎：模型层对齐 OpenAI 事实标准协议，工具层对齐 MCP 世界标准，算法层直接采用 BM25 / RRF / RoPE / ReAct / Reflexion 等论文级实现，验证层用**确定性回放**让每一次决策可重放、可断言、可回归。

## 为什么是"旗舰"

| 维度 | 旗舰做法 | 对比常见做法 |
|---|---|---|
| 依赖 | **运行时零依赖**，`node --test` 直接跑，无需 npm install | 动辄数百 MB node_modules |
| 协议 | OpenAI Chat Completions + MCP JSON-RPC 2.0，双世界标准 | 私有 SDK 锁定 |
| 算法 | BM25(Okapi) ⊕ RRF(SIGIR'09) ⊕ RoPE(RoFormer'21) ⊕ ReAct(ICLR'23) ⊕ Reflexion(NeurIPS'23) | 手拍相似度、裸 RAG |
| 验证 | **确定性回放**：录制响应 → 重放决策链 → 事件序列逐条相等 | 无验证或只测 happy path |
| 记忆 | 三段式（工作/情节/语义）+ Ebbinghaus 衰减 + A-MEM 式巩固 | 单向量库裸存 |
| 韧性 | Full Jitter 指数退避 + 熔断器 + 硬预算（步数/Token/墙钟） | 裸 try/catch |
| UI | 单文件零依赖驾驶舱（流式对话/轨迹/记忆/评估/微操作） | 重量前端框架 |

## 30 秒上手

```bash
git clone https://github.com/CJX0712/nova-flagship-ai.git
cd nova-flagship-ai
node examples/quickstart.mjs        # 看完整闭环：规划→工具→终答→回放验证
```

```js
import { createNova } from './src/nova.js';

const nova = createNova();                       // 默认离线确定性提供商，开箱即跑
await nova.rag.ingest([{ id: 'doc1', text: '……领域知识……' }]);
const r = await nova.agent.run({ task: '计算 (3+4)*2^10' });
console.log(r.answer);                            // 引用工具结果的终答
```

**切真实模型，一行不改业务代码**：

```bash
NOVA_PROVIDER=openai-compatible \
NOVA_BASE_URL=https://api.deepseek.com/v1 \
NOVA_API_KEY=sk-xxx \
node src/cli.js serve
```

兼容 OpenAI / DeepSeek / Moonshot / 智谱 / llama.cpp / Ollama / vLLM / SGLang —— 任何 OpenAI 兼容端点。

## 架构总览（十层）

```
┌────────────────────────────────────────────────────────────────┐
│ L10 服务与观测   server.js(HTTP/SSE)  observe.js(Tracer/Metrics)│
├────────────────────────────────────────────────────────────────┤
│ L9  微操作目录   ops.js  38 项原子操作，1024 微指令分类法种子    │
├────────────────────────────────────────────────────────────────┤
│ L8  智能体运行时 agent.js  ReAct 循环 + 预算 + 取消 + 确定性回放  │
├────────────────────────────────────────────────────────────────┤
│ L7  工具层       tools.js  注册表 + 内置工具 + MCP stdio 客户端   │
├────────────────────────────────────────────────────────────────┤
│ L6  检索增强     rag.js   递归分块 + 预算装袋 + 引用溯源          │
├────────────────────────────────────────────────────────────────┤
│ L5  记忆层       memory.js 工作/情节/语义三段式 + RRF 混合检索    │
├────────────────────────────────────────────────────────────────┤
│ L4  模型网关     gateway.js + providers.js  重试/熔断/账本/流式   │
├────────────────────────────────────────────────────────────────┤
│ L3  分词计量     tokenize.js  近似分词 + 预算贪心装袋             │
├────────────────────────────────────────────────────────────────┤
│ L2  协议层       schema.js  JSON Schema 子集校验（带 JSON Path）  │
├────────────────────────────────────────────────────────────────┤
│ L1  数学基元     math.js   logsumexp/softmax/BM25/RRF/RoPE/核采样 │
└────────────────────────────────────────────────────────────────┘
```

详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 核心能力详解

### 1. 模型网关（L4）

```js
nova.gateway.register(openaiCompatible({ baseURL, apiKey }), { isDefault: true });
const out = await nova.gateway.chat({ model: 'deepseek-chat', messages, tools });
for await (const ev of nova.gateway.chatStream({ messages })) { /* SSE 同协议 */ }
```

内置：Full Jitter 重试、熔断器（closed/open/half-open）、Token+成本账本、流式 tool_calls 增量装配。

### 2. MCP 工具（L7）

```js
const mcp = new McpClient({ name: 'fs', command: 'npx', args: ['-y', '@modelcontextprotocol/server-filesystem', '/path'] });
await mcp.start();
nova.tools.registerMcp(await mcp.listTools());   // 自动命名空间隔离 mcp__fs__read_file
```

内置工具：`calc`（Pratt parser，禁 eval）、`now`、`hash`、`uuid`、`http_get`（沙箱）、`fs_read`（目录围栏）、`memory_search`、`memory_store`。

### 3. 三段式记忆 + 混合检索（L5/L6）

- 稠密（特征哈希/可换 MiniLM/OpenAI）⊕ 稀疏（Okapi BM25）→ **RRF(k=60) 融合**
- 情节记忆按 Ebbinghaus 曲线衰减：`score = salience · e^(−λΔt)`
- 巩固：cosine ≥ 阈值近重复合并，显著性累积
- 检索不变量：同查询同结果（确定性）；双路命中文档融合分更高

### 4. 确定性回放（旗舰验证能力）

```js
const r = await nova.agent.run({ task: '……' });           // 轨迹内嵌 state_snapshot
const { result } = await replayAgent(r.trace.toJSON(), { tools: nova.tools, ... });
tracesEqual(r.trace, result.trace);                        // ⇒ true，逐条相等
```

每一次 LLM 响应、工具调用、工具观察都进入轨迹；回放时用录制响应替换模型、用快照恢复有状态工具，**决策序列必须逐条复现**。这是回归测试的锚，也是"可复现"的工程化身。

### 5. 微操作目录（L9，1024 微指令种子）

38 个已实现原子操作，10 大家族：感知/检索/嵌入/分块/推理/规划/执行/记忆/验证/元操作。编号 `<FAMILY>-<NNNN>`，纯函数、可独立测试、可演进到 1024 项完整目录：

```bash
node src/cli.js ops            # 全目录
node src/cli.js ops --family RET
```

## 验证体系

```bash
node --test          # 81 项测试：数学不变量 + 协议 + 集成 + 端到端 + 回放
node src/cli.js eval # 金标回归套件（evals/golden.jsonl）
node src/cli.js doctor
```

不变量示例（全部有测试背书）：softmax 和为 1、logsumexp ≥ max 且大值不上溢、BM25 非负且边际递减、RRF 有界且单调、RoPE 保范数、分块覆盖性 ≥ 99%、预算硬约束、回放逐条相等。

## 驾驶舱

```bash
node src/cli.js serve   # http://127.0.0.1:8780
```

单文件零依赖 UI：流式对话、智能体轨迹时间线、记忆检索检查器（dense/lexical 来源标注）、金标评估面板、微操作目录、连接设置。

## 项目结构

```
nova-flagship-ai/
├── src/
│   ├── nova.js               # 公共 API 总装线 createNova()
│   ├── cli.js                # nova CLI（serve/chat/eval/doctor/ops/version）
│   └── core/
│       ├── math.js           # L1 数学基元（全部带不变量）
│       ├── tokenize.js       # L3 分词计量
│       ├── schema.js         # L2 JSON Schema 校验
│       ├── gateway.js        # L4 网关（重试/熔断/账本）
│       ├── providers.js      # OpenAI 兼容 / novaLite / mock
│       ├── memory.js         # L5 三段式记忆
│       ├── rag.js            # L6 检索增强
│       ├── tools.js          # L7 工具 + MCP 客户端
│       ├── agent.js          # L8 智能体 + 回放
│       ├── ops.js            # L9 微操作目录
│       ├── observe.js        # Tracer/Metrics
│       └── server.js         # L10 HTTP API
├── web/cockpit.html          # 单文件驾驶舱
├── tests/                    # 81 项测试
├── evals/golden.jsonl        # 金标回归
├── examples/quickstart.mjs
└── docs/                     # ARCHITECTURE / MATH / EVALUATION
```

## 设计红线（本项目自我约束）

1. **零重写**：能对齐世界标准的绝不自创——协议用 OpenAI/MCP，算法用论文原版，测试不变量用数学定义。
2. **可复现**：默认提供商是确定性模拟器（nova-lite），CI 无网络、无 GPU、无 Key 也能全绿。
3. **可替换**：模型、嵌入器、检索器全是接口，世界更强方案出现时原位替换，不动业务代码。
4. **零依赖**：运行时依赖数为 0；只用 Node ≥ 20 内置能力（fetch/AbortSignal/node:test/crypto）。

## 路线图

- [ ] 真实模型基准：接入公开评测集（MMLU/C-Eval 子集）跑分
- [ ] 持久化：SQLite/LevelDB 适配器（记忆与轨迹落盘）
- [ ] 多智能体：A2A 协议协作 + 任务图编排
- [ ] 微操作目录扩展至 1024 项完整指令集
- [ ] HNSW/usearch 适配器（十万级以上向量规模）

## 引用

```bibtex
@software{novaflagship2026,
  title  = {NovaFlagship: 零依赖智能体引擎},
  author = {晨星},
  year   = {2026},
  url    = {https://github.com/CJX0712/nova-flagship-ai}
}
```

## 许可证

MIT © 2026 晨星。详见 [LICENSE](LICENSE)。
