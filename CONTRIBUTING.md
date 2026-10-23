# 贡献指南

感谢参与 NovaFlagship。本项目有明确的工程红线，贡献前请读一遍。

## 工程红线（不可协商）

1. **运行时零依赖**：只用 Node ≥ 20 内置能力（fetch / AbortSignal / node:test / node:crypto / node:http）。新增依赖需要充分论证。
2. **不变量先行**：每个新原语/新行为必须带可执行不变量测试（见 [MATH.md](MATH.md)）。没有不变量的 PR 会被要求补齐。
3. **接口优先**：模型、嵌入器、检索器、reranker 全部走接口，禁止在业务代码里写死某家 SDK。
4. **回放安全**：新增工具有状态副作用时，必须保证 `replayAgent` 可恢复（参考 `state_snapshot` 机制）；墙钟/随机类工具标注为非确定，不进回放金标。
5. **署名**：Git commit 作者使用 `晨星`（协作者请保留署名约定，见下）。

## 开发流程

```bash
node --test              # 全量测试（81 项）
node src/cli.js eval     # 金标回归
node src/cli.js doctor   # 环境自检
node examples/quickstart.mjs
```

## 提交规范

```
<type>: <摘要>

type: feat / fix / refactor / test / docs / perf / chore
```

Commit 作者：`晨星 <CJX0712@users.noreply.github.com>`

## 新增一个工具（示例）

```js
nova.tools.register({
  name: 'my_tool',
  description: '一句话说清能力边界（模型靠它决策）',
  inputSchema: { type: 'object', properties: { q: { type: 'string' } }, required: ['q'] },
  handler: async ({ q }, ctx) => ({ result: await doSomething(q) }),
});
```

清单：schema 完整（required + additionalProperties）、失败返回 throw（Registry 会包成结构化观察）、有副作用就考虑回放快照、补测试。

## 新增一个微操作（L9）

```js
ops.define('RET', 'my_retrieve', (args, ctx) => ({ hits: [] }), { desc: '说明' });
```

编号自动分配（`RET-NNNN`）。要求：纯函数或受控副作用、带不变量测试、能独立替换。

## 报告问题

附最小复现：Node 版本、任务输入、`trace.toJSONL()` 输出、期望 vs 实际。轨迹就是最好的复现材料——这也是本项目设计的初衷。
