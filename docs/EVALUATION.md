# NovaFlagship 评估体系

> 评估不是跑分秀，是**回归防线**：每一次代码改动，金标套件 + 回放一致性必须全绿。

## 1. 三层评估金字塔

```
        ┌─────────────────────┐
        │   E2E 金标回归       │  evals/golden.jsonl + CLI eval + 回放一致
        ├─────────────────────┤
        │   集成测试           │  智能体闭环 / HTTP API / MCP 全链路
        ├─────────────────────┤
        │   单元不变量         │  数学原语 / 协议 / 预算 / 衰减 …
        └─────────────────────┘
```

- **单元不变量（81 项测试的地基）**：每个数学原语、每个协议行为都有可执行断言，见 [MATH.md](MATH.md) 不变量总表。
- **集成测试**：智能体 ReAct 闭环、服务端到端、MCP 握手-调用-错误传播（真实子进程）。
- **E2E 金标**：`evals/golden.jsonl`，每条用例带期望（应答包含 / finishReason / 工具调用 / 步数上限）+ **回放一致性**。

## 2. 金标用例格式

```jsonc
// evals/golden.jsonl（每行一条）
{
  "id": "g1-calc",
  "task": "计算 (3+4)*2^10",
  "expect": {
    "answerContains": "7168",      // 终答必须包含
    "toolsCalled": ["calc"],       // 必须调用的工具
    "finishReason": "stop",        // 停止原因
    "maxSteps": 6                  // 步数上限
  }
}
```

运行：

```bash
node src/cli.js eval
# 金标回归: 6/6 通过
#   ✅ g1-calc
#   ✅ g2-explicit-tool
#   ...
```

任一失败 ⇒ 退出码 1，可直接接 CI。

## 3. 回放一致性（旗舰评估维度）

```
录制：agent.run(task) → trace（含 state_snapshot）
重放：replayAgent(trace) → 用录制响应替换模型 + 快照恢复记忆
断言：tracesEqual(原始, 回放) === true
```

**为什么这是硬评估**：它验证的不是"模型聪明"，而是**系统行为的确定性**——规划、工具分发、观察回流、预算控制的每一步都可在相同初态下复现。模型可以换，确定性不能丢。

**回放相等性的归一化规则**（`TraceRecorder.structural()`）：
- 剔除 `llm` 事件的 `model/provider`（回放时由 replay 提供商服务）
- 剔除 `tool_result` 的 `ms`（延迟是环境量，不是决策量）
- 剔除所有 `ts`

**回放的前提**：被回放的工具有状态副作用时，轨迹内 `state_snapshot` 负责恢复（记忆类工具已验证）；墙钟类工具（`now`）天然非确定，不进回放金标。

## 4. 指标与观测

```bash
curl localhost:8780/api/metrics
```

- 计数器：工具调用次数、预算超限次数、LLM 错误次数
- EWMA 计时：`agent.run.ms` 等关键路径延迟
- 成本账本：`gateway.ledger.totalsByModel()`（in/out token 分开计价）

## 5. 接入真实模型后的评估升级路径

| 阶段 | 方法 | 现状 |
|---|---|---|
| 离线回归 | 金标 + 回放 | ✅ 已实现（6 用例） |
| 工具正确率 | 扩充 golden：每个工具 ≥ 3 用例（正常/边界/错误入参） | 🔜 路线图 |
| 检索质量 | Recall@k / MRR on 自建标注集（ingest 文档 → 查询 → 人工标注相关块） | 🔜 |
| 端到端质量 | MMLU / C-Eval / GSM8K 子集，对比直连模型 | 🔜 |
| 成本-质量帕累托 | 同一金标集在不同模型/温度下的成本-通过率曲线 | 🔜 |

**扩展金标的正确姿势**：新用例必须满足三条——期望可由确定性工具验证、步数有界、回放一致。主观开放题不进金标（那是人评，不是回归）。

## 6. CI 接入

```yaml
# .github/workflows/ci.yml（最小可用）
- run: node --version
- run: node --test        # 81 项
- run: node src/cli.js eval
- run: node src/cli.js doctor
```

零依赖 ⇒ 无需 `npm install` 步骤，CI 30 秒内跑完。
