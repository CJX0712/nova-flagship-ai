#!/usr/bin/env node
/**
 * NovaFlagship CLI
 *   nova serve [--port 8780] [--host 127.0.0.1]   启动 HTTP 服务 + 驾驶舱
 *   nova chat --task "..." [--direct]             单轮对话（默认走智能体）
 *   nova eval [--file evals/golden.jsonl]         跑金标回归套件
 *   nova doctor                                   环境自检
 *   nova ops [--family RET]                       微操作目录
 *   nova version
 */

import { parseArgs } from 'node:util';
import { createNova } from './nova.js';
import { VERSION } from './nova.js';

const COMMANDS = {
  serve: cmdServe,
  chat: cmdChat,
  eval: cmdEval,
  doctor: cmdDoctor,
  ops: cmdOps,
  version: () => console.log(`nova ${VERSION}`),
};

async function main() {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      port: { type: 'string', short: 'p' },
      host: { type: 'string', short: 'h' },
      task: { type: 'string', short: 't' },
      model: { type: 'string', short: 'm' },
      file: { type: 'string', short: 'f' },
      family: { type: 'string' },
      direct: { type: 'boolean', default: false },
      help: { type: 'boolean', default: false },
    },
  });
  const cmd = positionals[0];
  if (values.help || !cmd) {
    console.log(`NovaFlagship CLI ${VERSION}
用法: nova <命令> [选项]
  serve [--port 8780] [--host 127.0.0.1]   启动 HTTP 服务 + 驾驶舱
  chat --task "..." [--direct] [--model m]  单轮对话
  eval [--file evals/golden.jsonl]          金标回归套件
  doctor                                    环境自检
  ops [--family RET]                        微操作目录
  version                                   版本`);
    process.exitCode = cmd ? 0 : 1;
    return;
  }
  const fn = COMMANDS[cmd];
  if (!fn) {
    console.error(`未知命令: ${cmd}（可用: ${Object.keys(COMMANDS).join(', ')}）`);
    process.exitCode = 1;
    return;
  }
  await fn(values);
}

function buildStack(extra = {}) {
  return createNova({ fsRoots: [], allowHttp: false, ...extra });
}

async function cmdServe(values) {
  const nova = buildStack();
  const port = Number(values.port || 8780);
  const host = values.host || '127.0.0.1';
  await nova.serve({ port, host });
  console.log(`✦ NovaFlagship ${VERSION} 已启动`);
  console.log(`  驾驶舱   http://${host}:${port}/`);
  console.log(`  API      http://${host}:${port}/v1/chat/completions`);
  console.log(`  提供商   ${nova.gateway.defaultProvider}（默认离线确定性；设 NOVA_BASE_URL/NOVA_API_KEY 切换真实模型）`);
}

async function cmdChat(values) {
  const task = values.task;
  if (!task) {
    console.error('需要 --task "任务内容"');
    process.exitCode = 1;
    return;
  }
  const nova = buildStack();
  if (values.direct) {
    const out = await nova.gateway.chat({ model: values.model, messages: [{ role: 'user', content: task }] });
    console.log(out.message.content);
    return;
  }
  const result = await nova.agent.run({ task, model: values.model });
  console.log(result.answer);
  console.log(`\n── 步数 ${result.steps} · token ${result.usage.totalTokens} · ${result.ms}ms · ${result.finishReason}`);
}

async function cmdEval(values) {
  const { readFile } = await import('node:fs/promises');
  const file = values.file || new URL('../evals/golden.jsonl', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');
  const lines = (await readFile(file, 'utf8')).split('\n').filter((l) => l.trim());
  const cases = lines.map((l) => JSON.parse(l));

  let pass = 0;
  const rows = [];
  for (const c of cases) {
    // 每个用例独立栈，保证相互隔离与确定性
    const nova = buildStack();
    const r = await nova.agent.run({ task: c.task });
    const failures = [];
    const e = c.expect || {};
    if (e.answerContains && !String(r.answer).includes(e.answerContains)) failures.push(`应答应包含 "${e.answerContains}"`);
    if (e.finishReason && r.finishReason !== e.finishReason) failures.push(`finishReason 应为 ${e.finishReason}，实际 ${r.finishReason}`);
    if (e.toolsCalled) {
      const called = r.trace.events.filter((ev) => ev.type === 'tool_call').map((ev) => ev.payload.name);
      for (const t of e.toolsCalled) if (!called.includes(t)) failures.push(`应调用工具 ${t}（实际: ${called.join(',') || '无'}）`);
    }
    if (e.maxSteps && r.steps > e.maxSteps) failures.push(`步数 ${r.steps} > ${e.maxSteps}`);
    // 通用不变量：回放一致性
    try {
      const { result: replayed } = await (await import('./nova.js')).replayAgent(r.trace.toJSON(), {
        tools: nova.tools,
        budget: nova.agent.budget,
        systemPrompt: nova.agent.systemPrompt,
      });
      if (!(await import('./nova.js')).tracesEqual(r.trace, replayed.trace)) failures.push('回放轨迹不一致');
    } catch (err) {
      failures.push(`回放异常: ${err.message}`);
    }
    pass += failures.length === 0 ? 1 : 0;
    rows.push({ id: c.id, ok: failures.length === 0, detail: failures.join('；') || '通过', ms: r.ms });
  }

  console.log(`\n金标回归: ${pass}/${rows.length} 通过\n`);
  for (const r of rows) {
    console.log(`  ${r.ok ? '✅' : '❌'} ${r.id}${r.ok ? '' : ` — ${r.detail}`}`);
  }
  if (pass !== rows.length) process.exitCode = 1;
}

async function cmdDoctor(values) {
  const rows = [];
  const nodeMajor = Number(process.versions.node.split('.')[0]);
  rows.push(['Node 运行时', `v${process.versions.node}`, nodeMajor >= 20 ? '✅' : '❌ 需 ≥ 20']);
  rows.push(['平台', `${process.platform}/${process.arch}`, '✅']);
  const provider = process.env.NOVA_PROVIDER || 'nova-lite';
  rows.push(['提供商', provider, provider === 'nova-lite' ? '✅ 离线确定性' : process.env.NOVA_BASE_URL ? '✅' : '⚠️ 未设 NOVA_BASE_URL']);
  rows.push(['GPT 兼容端点', process.env.NOVA_BASE_URL || '（未配置）', process.env.NOVA_BASE_URL ? '✅' : '—']);
  const nova = buildStack();
  rows.push(['内置工具', nova.tools.list().map((t) => t.name).join(', '), '✅']);
  rows.push(['记忆', JSON.stringify(nova.memory.stats()), '✅']);
  rows.push(['微操作', `${nova.ops.size} 项`, '✅']);
  const probe = await nova.gateway.chat({ messages: [{ role: 'user', content: 'ping' }] });
  rows.push(['网关探测', `${nova.gateway.defaultProvider} 响应 ${probe.ms}ms`, '✅']);
  console.log('\nNovaFlagship 自检\n');
  for (const [k, v, s] of rows) console.log(`  ${s} ${k}: ${v}`);
}

async function cmdOps(values) {
  const nova = buildStack();
  const ops = nova.ops.list({ family: values.family });
  console.log(`\n微操作目录（${ops.length} 项）\n`);
  for (const o of ops) console.log(`  ${o.id}  ${o.name.padEnd(22)} ${o.desc}`);
}

main().catch((err) => {
  console.error(`✗ ${err?.stack || err}`);
  process.exitCode = 1;
});
