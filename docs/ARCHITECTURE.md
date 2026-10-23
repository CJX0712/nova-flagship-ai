# NovaFlagship 架构文档

> 十层分解架构。每层：职责、世界级参照、对外接口、不变量。

## 分层原则

- **单向依赖**：上层只调下层，`src/nova.js` 是唯一总装线。
- **接口优先**：模型（provider）、嵌入（embedder）、检索（index）都是可替换接口。
- **层内闭环**：每层有自己的不变量测试（`tests/*.test.js`），改坏立刻红。

## L1 数学基元层 `core/math.js`

| 原语 | 世界级参照 | 不变量（测试背书） |
|---|---|---|
| `logsumexp` | 数值稳定性标准技巧 | ≥ max(x)；与朴素实现 relErr < 1e-12 |
| `softmax` | 带温度 softmax | 和为 1；温度 ≤ 0 退化 argmax one-hot |
| `cosine/dot/norm/l2normalize` | 线性代数标准 | 值域 [-1,1]；零向量 → 0 |
| `BM25Index` | Okapi BM25，Robertson & Zaragoza 2009（k1=1.5, b=0.75） | idf ≥ 0；tf 边际递减；结果降序 |
| `rrfFuse` | Cormack, Clarke & Büttcher, SIGIR 2009（k=60） | 分数 ∈ (0, n/k]；新增排序只增不减 |
| `applyRoPE` | Su et al. 2021（RoFormer） | 保范数；位置 0 恒等 |
| `sampleFromLogits` | 核采样，Holtzman et al. ICLR 2020 | 同种子同结果；返回值 ∈ 核 |
| `mulberry32` | 确定性 PRNG | 同种子同序列（回放基石） |
| `zscore/ewma/sigmoid` | 统计标准 | ewma 首值=序列首值；sigmoid(0)=0.5 |

## L2 协议层 `core/schema.js`

JSON Schema 子集校验器（type/enum/const/required/properties/items/minimum/pattern/additionalProperties），错误带 JSON Path（`a.b[0].c`），与 AJV 错误风格对齐。用于工具入参校验（L7）与 VER 族微操作。

**不变量**：类型不符时短路后续检查；错误路径精确定位到出错节点。

## L3 分词计量层 `core/tokenize.js`

- `estimateTokens`：OpenAI 公开启发式（CJK≈1 字符/token，ASCII≈4 字符/token）。
- `estimateMessagesTokens`：每消息 4 + 回复引导 2（OpenAI 公开公式）。
- `packToBudget`：按分数降序的背包贪心，总消耗 ≤ 预算。
- 精确计量两条路：provider `usage` 字段（网关自动采集）/ 适配 tiktoken。

**不变量**：非负、空串 0、长度单调、CJK 串 ≈ 字符数。

## L4 模型网关层 `core/gateway.js` + `core/providers.js`

```
AgentRuntime ──► ModelGateway ──► Provider（OpenAI 兼容 / novaLite / mock）
                     │
                     ├─ retryWithBackoff（Full Jitter，AWS 标准）
                     ├─ CircuitBreaker（closed→open→half-open）
                     └─ CostLedger（按模型 in/out 计价）
```

- **provider 契约（世界最小接口）**：`{ name, chat(request) → Completion, chatStream?(request) → AsyncGenerator }`。
- **openaiCompatible**：事实标准客户端——SSE 解析、tool_calls 按 index 增量装配、`include_usage`、错误码可重试性映射（408/429/5xx 可重试）。
- **novaLite**：确定性模拟提供商（R1 显式 `@tool` / R2 工具观察后终答 / R3 算术触发 / R4 `@recall` / R5 模板）。**不是语言模型**，是让全链路离线可复现的策略函数。
- **mockProvider**：脚本队列，单元测试精确控流。

**不变量**：重试 ≤ retries+1 且延迟 ∈ [0, cap]；abort 立即拒绝；熔断阈值/冷却/半开单探针；账本总额 = Σ in·pin + out·pout。

## L5 记忆层 `core/memory.js`

```
MemorySystem
├── WorkingMemory    环形缓冲（Miller 7±2 的有界化）
├── EpisodicMemory   事件流，召回分 = salience · e^(−λΔt)（Ebbinghaus）
└── SemanticMemory   向量库 + BM25 索引 → RRF 融合；近重复巩固（A-MEM/Zettelkasten）
```

- **Embedder 接口**：默认 `HashEmbedder`（字符 2-3 gram 有符号特征哈希 + L2 归一化，确定性）；生产换 `@xenova/transformers`（all-MiniLM-L6-v2）或 OpenAI embeddings，接口不变。
- **混合检索**：`searchHybrid = dense ⊕ lexical → rrfFuse(k=60)`，命中带 `sources: ['dense','lexical']` 标注。
- **规模说明**：万级内暴力精确余弦即最优；接口预留 usearch/hnswlib 适配位。
- **快照**：`snapshot()/restore()/dump()/load()`——回放状态隔离的基础。

**不变量**：同查询同结果；衰减随年龄单调；容量有界；巩固后 size 减少且高显著性保留。

## L6 检索增强层 `core/rag.js`

- `chunkText`：递归分块（separators 优先级递降：`\n\n → \n → 。！？；→ 空格 → 字符`），重叠 carry-tail，字符偏移可溯源。
- `RagPipeline.buildContext`：`recall → 引用编号 [n] → packToBudget 装袋`，输出 `{context, sources, usedTokens}`。

**不变量**：覆盖性 ≥ 99%（区间合并度量）；分块 ≤ size（原子块除外）；确定性；上下文 token ≤ 预算；引用与来源一一对应。

## L7 工具层 `core/tools.js`

- `ToolRegistry`：注册（重名拒绝）→ schema 校验 → 超时（AbortSignal）→ 结构化结果（失败也是有效观察，回流给模型）。
- `evalExpression`：Pratt/优先级爬升解析器，**禁 eval**；`^` 右结合；除零/未知符号/非法字符/括号不配齐全部抛错。
- 内置工具：`now / calc / uuid / hash / http_get（沙箱+截断）/ fs_read（目录围栏 realpath 校验）/ memory_search / memory_store`。
- `McpClient`：MCP stdio 传输，JSON-RPC 2.0（initialize → notifications/initialized → tools/list → tools/call），id 关联 + 超时；`registerMcp` 自动命名空间隔离（`mcp__<server>__<tool>`）。

**不变量**：未知工具/入参错误/超时都返回 `{ok:false}` 而非崩溃；MCP 握手-列举-调用-错误传播全链路有 fixture 服务器实测。

## L8 智能体运行时层 `core/agent.js`

```
run(task)
  ├─ trace: task → state_snapshot
  └─ loop（预算检查在每轮顶部）:
       llm.chat(messages + tools)
         ├─ tool_calls? → 逐个 execute → 观察消息回流 → 继续
         └─ 无 → answer（回写记忆）→ 结束
```

- **范式**：ReAct（Yao et al., ICLR 2023）；Reflexion 式钩子预留（PLN-insert_reflection 微操作）。
- **硬预算**：`maxSteps / maxTokens / maxMs / maxToolCalls`，触顶即优雅收敛并说明原因。
- **取消**：AbortSignal 贯穿 LLM 调用与工具执行。
- **轨迹**：`task / state_snapshot / llm / tool_call / tool_result / budget_exceeded / aborted / llm_error / answer`。
- **回放**：`replayAgent` 用录制响应换模型、用快照恢复记忆，`tracesEqual` 逐条比对（归一化 llm 的 model/provider、tool_result 的 ms——延迟与供应商是环境量，不是决策量）。

**不变量**：同初始状态 + 同录制响应 ⇒ 同决策链；回放对调用方记忆零侵扰；预算必然收敛。

## L9 微操作目录 `core/ops.js`

10 大家族 × 38 项已实现操作，编号 `<FAMILY>-<NNNN>`：

| 家族 | 例 |
|---|---|
| OBS 感知 | count_tokens / tokenize / detect_language / normalize_text / extract_keywords |
| RET 检索 | rrf_fuse / mmr_rerank / threshold_filter / dedupe_by_text |
| EMB 嵌入 | char_ngrams / cosine / softmax / logsumexp / l2_normalize / dot / hash_embed |
| CHK 分块 | recursive_chunk / chunk_stats |
| RSN 推理 | sample / self_consistency_vote / build_cot_prompt / answer_sanity |
| PLN 规划 | decompose_task / budget_allocate / insert_reflection |
| ACT 执行 | calc / json_parse_safe |
| MEM 记忆 | remember / recall / consolidate / decay_sweep / stats |
| VER 验证 | schema_validate / softmax_invariant / trace_structural |
| MET 元操作 | budget_check / abort_check / cost_estimate |

这是第十层"1024 项微指令级操作"分类法的种子：每个操作纯函数或受控副作用，可独立测试、独立替换、独立演进。

## L10 服务与观测层 `core/server.js` + `core/observe.js`

- HTTP（`node:http`，零依赖）：`/api/health`、`/v1/chat/completions`（OpenAI 兼容 + SSE）、`/v1/agents`（同步 + SSE 轨迹流）、`/v1/agents/:id/trace`、`/v1/memory/*`、`/v1/rag/context`、`/api/ops`、`/api/metrics`、`/`（驾驶舱）。
- 观测：`Tracer`（span 树 + JSONL）+ `Metrics`（计数器 + EWMA 计时），OpenTelemetry 启发式轻量实现。

## 数据流（一次智能体任务）

```
task ─► memory.contextBlock ─► messages
      ─► gateway.chat ────────► novaLite/真实模型
      ─► tool_calls? ─yes─► tools.execute（校验/超时）─► 观察回流 ─┐
                │no                                                │
                └─◄─────────────────────────────────────────────────┘
      ─► answer ─► trace（含 state_snapshot）─► replayAgent 可复现
```

## 扩展点（换世界更强方案的位置）

| 插槽 | 默认实现 | 升级路径 |
|---|---|---|
| Provider | novaLite（离线） | 任何 OpenAI 兼容端点（DeepSeek/vLLM/llama.cpp…） |
| Embedder | HashEmbedder | @xenova/transformers MiniLM / OpenAI embeddings / BGE-M3 |
| 向量索引 | 暴力精确余弦 | usearch / hnswlib（实现 `searchDense` 同签名） |
| 检索融合 | RRF | CombSUM/CombMNZ（实现同签名替换） |
|  rerank | MMR（Jaccard 代理） | cross-encoder（bge-reranker） |
| 持久化 | 内存 | SQLite/LevelDB 适配 `snapshot/restore` |
