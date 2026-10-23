// 端到端：金标文件 + CLI 子进程 + 快速上手脚本
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createNova, replayAgent, tracesEqual } from '../src/nova.js';

const run = promisify(execFile);
const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');

test('golden.jsonl：每条用例期望全部满足 + 回放一致', async () => {
  const lines = (await readFile(path.join(ROOT, 'evals', 'golden.jsonl'), 'utf8')).split('\n').filter((l) => l.trim());
  assert.ok(lines.length >= 6);
  for (const line of lines) {
    const c = JSON.parse(line);
    const nova = createNova({ allowHttp: false, fsRoots: [] });
    const r = await nova.agent.run({ task: c.task });
    const e = c.expect || {};
    if (e.answerContains) assert.ok(String(r.answer).includes(e.answerContains), `${c.id}: 应答缺 ${e.answerContains}`);
    if (e.finishReason) assert.equal(r.finishReason, e.finishReason, c.id);
    if (e.maxSteps) assert.ok(r.steps <= e.maxSteps, `${c.id}: 步数超限`);
    if (e.toolsCalled) {
      const called = r.trace.events.filter((ev) => ev.type === 'tool_call').map((ev) => ev.payload.name);
      for (const t of e.toolsCalled) assert.ok(called.includes(t), `${c.id}: 未调用 ${t}`);
    }
    // 回放一致性（可复现性的硬断言）
    const { result } = await replayAgent(r.trace.toJSON(), {
      tools: nova.tools,
      budget: nova.agent.budget,
      systemPrompt: nova.agent.systemPrompt,
    });
    assert.ok(tracesEqual(r.trace, result.trace), `${c.id}: 回放不一致`);
  }
});

test('CLI：nova eval 子进程退出码 0', async () => {
  const { stdout } = await run(process.execPath, ['src/cli.js', 'eval'], { cwd: ROOT });
  assert.match(stdout, /金标回归: \d+\/\d+ 通过/);
  assert.ok(!/❌/.test(stdout), `存在失败用例:\n${stdout}`);
}, 60000);

test('CLI：nova version / ops', async () => {
  const v = await run(process.execPath, ['src/cli.js', 'version'], { cwd: ROOT });
  assert.match(v.stdout, /nova 1\.0\.0/);
  const ops = await run(process.execPath, ['src/cli.js', 'ops'], { cwd: ROOT });
  assert.match(ops.stdout, /RET-0001/);
  assert.match(ops.stdout, /微操作目录/);
}, 60000);

test('CLI：nova chat 单轮', async () => {
  const { stdout } = await run(process.execPath, ['src/cli.js', 'chat', '--task', '计算 (3+4)*2^10'], { cwd: ROOT });
  assert.match(stdout, /7168/);
  assert.match(stdout, /步数 \d+/);
}, 60000);

test('CLI：nova doctor 自检全绿', async () => {
  const { stdout } = await run(process.execPath, ['src/cli.js', 'doctor'], { cwd: ROOT });
  assert.match(stdout, /网关探测/);
  assert.ok(!/❌/.test(stdout), stdout);
}, 60000);

test('quickstart 示例可运行', async () => {
  const { stdout } = await run(process.execPath, ['examples/quickstart.mjs'], { cwd: ROOT });
  assert.match(stdout, /7168/);
  assert.match(stdout, /回放一致/);
}, 60000);
