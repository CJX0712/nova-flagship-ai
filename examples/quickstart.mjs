/**
 * NovaFlagship 快速上手：20 行看到一个完整智能体闭环
 * 运行: node examples/quickstart.mjs
 */
import { createNova, replayAgent, tracesEqual } from '../src/nova.js';

const nova = createNova({ allowHttp: false, fsRoots: [] });

// 1) 灌入领域知识（混合检索语料）
await nova.rag.ingest([
  { id: 'arch', text: 'NovaFlagship 采用十层架构：数学基元、协议、网关、记忆、检索、工具、智能体、服务、微操作、观测。' },
  { id: 'verify', text: '确定性回放是旗舰验证能力：录制全部 LLM 响应与工具观察，重放时要求事件序列逐条相等。' },
]);

// 2) 跑一个需要工具的任务（ReAct：规划 → 调用 calc → 观察 → 终答）
const result = await nova.agent.run({ task: '计算 (3+4)*2^10' });
console.log('任务: 计算 (3+4)*2^10');
console.log('终答:', result.answer);
console.log(`步数 ${result.steps} · token ${result.usage.totalTokens} · ${result.ms}ms · ${result.finishReason}`);

// 3) 检索增强：带引用的上下文
const ctx = await nova.rag.buildContext('回放 验证');
console.log('\nRAG 上下文（前 2 行）:');
console.log(ctx.context.split('\n').slice(0, 2).join('\n'));

// 4) 确定性回放验证：同轨迹重放，事件序列必须逐条相等
const { result: replayed } = await replayAgent(result.trace.toJSON(), {
  tools: nova.tools,
  budget: nova.agent.budget,
  systemPrompt: nova.agent.systemPrompt,
});
console.log('\n回放一致:', tracesEqual(result.trace, replayed.trace) ? '✅ 是' : '❌ 否');

// 5) 真实模型原位替换（取消注释并设置环境变量即可，其余代码零改动）:
// const nova2 = createNova({ provider: openaiCompatible({ baseURL: process.env.NOVA_BASE_URL, apiKey: process.env.NOVA_API_KEY }) });
