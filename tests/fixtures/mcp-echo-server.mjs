// MCP Echo 测试服务器：按 JSON-RPC 2.0 over stdio 协议响应，用于验证 McpClient。
import { createInterface } from 'node:readline';

const TOOLS = [
  {
    name: 'echo',
    description: '原样回显输入',
    inputSchema: {
      type: 'object',
      properties: { text: { type: 'string' } },
      required: ['text'],
      additionalProperties: false,
    },
  },
  {
    name: 'add',
    description: '两数相加',
    inputSchema: {
      type: 'object',
      properties: { a: { type: 'number' }, b: { type: 'number' } },
      required: ['a', 'b'],
      additionalProperties: false,
    },
  },
];

const rl = createInterface({ input: process.stdin });
rl.on('line', (line) => {
  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    return;
  }
  if (msg.method === 'initialize') {
    respond(msg.id, {
      protocolVersion: '2024-11-05',
      capabilities: { tools: {} },
      serverInfo: { name: 'mcp-echo-fixture', version: '1.0.0' },
    });
    return;
  }
  if (msg.method === 'notifications/initialized') return;
  if (msg.method === 'tools/list') {
    respond(msg.id, { tools: TOOLS });
    return;
  }
  if (msg.method === 'tools/call') {
    const { name, arguments: args } = msg.params || {};
    if (name === 'echo') {
      respond(msg.id, { content: [{ type: 'text', text: JSON.stringify({ echo: args.text }) }] });
    } else if (name === 'add') {
      if (typeof args.a !== 'number' || typeof args.b !== 'number') {
        respond(msg.id, { isError: true, content: [{ type: 'text', text: 'add 需要数值参数 a/b' }] });
      } else {
        respond(msg.id, { content: [{ type: 'text', text: JSON.stringify({ sum: args.a + args.b }) }] });
      }
    } else {
      respond(msg.id, { isError: true, content: [{ type: 'text', text: `未知工具: ${name}` }] });
    }
    return;
  }
  if (msg.id !== undefined) {
    respond(msg.id, null, new Error(`未知方法: ${msg.method}`));
  }
});

function respond(id, result, error) {
  const payload = error
    ? { jsonrpc: '2.0', id, error: { code: -32601, message: error.message } }
    : { jsonrpc: '2.0', id, result };
  process.stdout.write(JSON.stringify(payload) + '\n');
}
