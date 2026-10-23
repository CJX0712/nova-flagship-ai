# NovaFlagship 数学基础

> 每个原语：定义、论文出处、**可验证不变量**。不变量即测试断言——数学正确性不靠信仰，靠断言。

## 1. 数值稳定 softmax 与 logsumexp

**定义**

```
softmax(x)_i = e^{x_i} / Σ_j e^{x_j}
logsumexp(x) = m + log Σ_j e^{x_j − m},  m = max(x)
```

**为什么**：直接计算 `e^{x_i}` 在大 logits 下上溢为 Infinity。减去最大值后指数项 ∈ (0,1]，和 ∈ (1, n]，对数有界——这是所有机器学习框架（PyTorch/TF/JAX）的标准实现。

**不变量**
- `logsumexp(x) ≥ max(x)`，且与朴素实现相对误差 < 1e-12
- `Σ softmax(x) = 1`（误差 < 1e-12）
- 温度 → 0 时 softmax 收敛到 argmax one-hot（贪心解码）

**测试**：`tests/math.test.js`「logsumexp：稳定性与下界」「softmax：和为 1、温度退化、单调」

## 2. 余弦相似度与向量运算

```
cos(a,b) = (a·b) / (‖a‖·‖b‖)
```

**不变量**：值域 [-1,1]；`cos(a,a)=1`；正交 → 0；零向量 → 0（定义保护，避免 0/0）。

## 3. Okapi BM25

**定义**（Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009）

```
idf(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))          ≥ 0
score(q, d) = Σ_{t∈q} idf(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1 − b + b·|d|/avgdl))
```

默认 `k1=1.5, b=0.75` 为原文推荐值；idf 采用 Lucene 变体（保证非负，避免负权重倒排）。

**不变量**
- `idf(t) ≥ 0` 对所有 t
- 同一文档内，查询词频 f 越高分越高，但**边际递减**（分母同步增长）
- 不含任何查询词的文档得 0 分
- 检索结果按分数降序

**测试**：`tests/math.test.js`「BM25：Okapi 性质」

## 4. Reciprocal Rank Fusion

**定义**（Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009）

```
score(d) = Σ_{i=1}^{n} 1 / (k + rank_i(d)),   k = 60
```

**性质**：只依赖排名不依赖分数绝对值 ⇒ 对异构检索器（BM25 与稠密向量分数量纲完全不同）天然公平；k=60 为论文推荐值。

**不变量**
- `0 < score(d) ≤ n/k`
- **单调性**：新增一路排序，任何文档的融合分只增不减
- 同时被多路检索到的文档融合分更高

**测试**：`tests/math.test.js`「RRF：有界性 + 单调性」

## 5. RoPE 旋转位置编码

**定义**（Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021）

对维度对 (x_{2i}, x_{2i+1}) 施加位置相关的二维旋转：

```
[x'_{2i}  ]   [cos(mθ_i)  −sin(mθ_i)] [x_{2i}  ]
[x'_{2i+1}] = [sin(mθ_i)   cos(mθ_i)] [x_{2i+1}],   θ_i = base^{−2i/d},  base = 10000
```

**关键性质**：旋转矩阵正交 ⇒ 点积注意力中相对位置信息进入 q·k，且**每个维度对保范数**。

**不变量**
- `‖RoPE(x, m)‖ = ‖x‖`（整体保范数，误差 < 1e-9）
- `RoPE(x, 0) = x`（零位置恒等）
- 确定性：同 (x, m) 同输出

**测试**：`tests/math.test.js`「RoPE：旋转保范数 + 零位置恒等」

## 6. 核采样（top-k / top-p）

**定义**（Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020）

1. 温度缩放：`z_i / T`
2. top-k 截断：保留概率最高的 k 个
3. top-p 核截断：保留累积概率 ≥ p 的**最小**集合（至少 1 个）
4. 在保留集合上按概率抽取

**不变量**：返回值 ∈ 保留集合；`T ≤ 0` 退化为 argmax；同种子 PRNG ⇒ 同结果（可复现解码）。

**测试**：`tests/math.test.js`「核采样」

## 7. Ebbinghaus 遗忘曲线（情节记忆衰减）

**定义**

```
R(t) = e^{−t/S}   ⇒   衰减分数 = salience · e^{−λΔt}
```

λ 为遗忘速率（默认半衰期 24h）。显著性（salience）是强度调制项。

**不变量**：其余不变时，分数随事件年龄**严格单调递减**；显著性线性放大分数。

**测试**：`tests/memory.test.js`「EpisodicMemory：Ebbinghaus 衰减单调」

## 8. MMR 多样性重排

**定义**（Carbonell & Goldstein, *The Use of MMR, Rerank for Diversity-Based Reranking*, SIGIR 1998）

```
MMR = argmax_{d_i ∈ R\S} [ λ·Rel(d_i, q) − (1−λ)·max_{d_j ∈ S} Sim(d_i, d_j) ]
```

本项目用词集合 Jaccard 作 Sim 代理（零依赖、确定性）；λ=0.7 默认。

**不变量**：每轮选出的文档都是当前剩余中的 MMR 最大值；k 控制输出规模。

**测试**：`tests/ops.test.js`（RET-0002 mmr_rerank 可执行）

## 9. Full Jitter 指数退避

**定义**（AWS Architecture Blog, *Exponential Backoff And Jitter*）

```
delay_n = random(0, min(cap, base · 2^n))
```

Full Jitter 相比固定退避最大化分散重试时刻，缓解惊群；相比 Decorrelated Jitter 实现更简。

**不变量**：`delay_n ∈ [0, cap]`；重试次数 ≤ retries+1；abort 信号立即拒绝（不再睡）。

**测试**：`tests/gateway.test.js`「retryWithBackoff：Full Jitter 边界与次数上限」

## 10. 自洽性（Self-Consistency）

**定义**（Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, ICLR 2023）

对同一问题采样多条推理链，对**最终答案**做多数表决（而非投票过程）。

**不变量**：表决结果 ∈ 候选集合；票数 = 最高频答案的出现次数；平局按字典序（保证确定性）。

**测试**：`tests/ops.test.js`（RSN-0002 self_consistency_vote）

---

## 不变量总表（CI 硬门禁）

| 层 | 不变量 | 测试位置 |
|---|---|---|
| L1 | softmax 和=1；logsumexp≥max；BM25 非负边际递减；RRF 有界单调；RoPE 保范数；核采样确定性 | math.test.js |
| L2 | schema 错误带路径；类型不符短路 | schema-tokenize.test.js |
| L3 | token 非负/单调；预算 ≤ 上限 | schema-tokenize.test.js |
| L4 | 重试边界；熔断状态机；账本等式；SSE 装配 | gateway.test.js |
| L5 | 嵌入确定性；衰减单调；检索确定；巩固收敛 | memory.test.js |
| L6 | 分块覆盖 ≥99%；有界；上下文 ≤ 预算 | rag.test.js |
| L7 | 表达式求值正确性；工具失败结构化；MCP 全链路 | tools.test.js |
| L8 | 预算收敛；取消；**回放逐条相等** | agent.test.js |
| L9 | 编号唯一；家族齐全 | ops.test.js |
| L10 | 端点契约；错误码；静态资源 | server.test.js |
| E2E | 金标期望 + 回放一致 + CLI 退出码 | e2e.test.js |
