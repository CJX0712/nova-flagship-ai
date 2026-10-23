// L3 分词计量 + L2 Schema 校验
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { estimateTokens, estimateMessagesTokens, packToBudget, tokenizeWords, charNgrams } from '../src/core/tokenize.js';
import { validateSchema, validateOrThrow } from '../src/core/schema.js';

test('estimateTokens：启发式不变量', () => {
  assert.equal(estimateTokens(''), 0);
  assert.equal(estimateTokens(null), 0);
  assert.ok(estimateTokens('hello world') > 0);
  // 长度单调
  const short = estimateTokens('人工智能');
  const long = estimateTokens('人工智能真厉害');
  assert.ok(long >= short);
  // CJK ≈ 1 字符/token
  assert.equal(estimateTokens('旗舰'), 2);
});

test('estimateMessagesTokens：OpenAI 公开公式', () => {
  assert.equal(estimateMessagesTokens([]), 2); // 回复引导
  const one = estimateMessagesTokens([{ role: 'user', content: 'hi' }]);
  assert.equal(one, 2 + 4 + estimateTokens('hi')); // primer + perMessage + tokens('hi')
  assert.equal(estimateTokens('hi'), 1);
  const withTool = estimateMessagesTokens([
    { role: 'assistant', content: '', tool_calls: [{ function: { name: 'calc', arguments: '{}' } }] },
  ]);
  assert.ok(withTool > 4);
});

test('packToBudget：预算内贪心', () => {
  const items = [
    { id: 'a', score: 0.9, text: 'x'.repeat(40) },   // 10 tok
    { id: 'b', score: 0.5, text: 'y'.repeat(400) },  // 100 tok
    { id: 'c', score: 0.1, text: 'z'.repeat(40) },   // 10 tok
  ];
  const tokensOf = (it) => estimateTokens(it.text);
  const { picked, used } = packToBudget(items, 25, tokensOf);
  assert.deepEqual(picked.map((p) => p.id), ['a', 'c']);
  assert.ok(used <= 25);
  // 默认 tokensOf：走 item.text
  const d = packToBudget(items, 25);
  assert.deepEqual(d.picked.map((p) => p.id), ['a', 'c']);
  // 全部放不下时返回空
  assert.deepEqual(packToBudget([{ id: 'big', score: 1, text: 'x'.repeat(4000) }], 10, tokensOf).picked, []);
});

test('tokenizeWords / charNgrams', () => {
  assert.deepEqual(tokenizeWords('Hello 世界 world'), ['世', '界', 'hello', 'world']);
  assert.deepEqual(charNgrams('abc', 2), ['ab', 'bc']);
  assert.deepEqual(charNgrams('', 3), []);
});

test('schema：通过用例', () => {
  const schema = {
    type: 'object',
    properties: {
      name: { type: 'string', minLength: 1 },
      age: { type: 'integer', minimum: 0, maximum: 150 },
      tags: { type: 'array', items: { type: 'string' } },
      role: { enum: ['admin', 'user'] },
    },
    required: ['name'],
    additionalProperties: false,
  };
  assert.deepEqual(validateSchema({ name: 'x', age: 3, tags: ['a'], role: 'admin' }, schema), []);
});

test('schema：各类错误带路径', () => {
  const schema = {
    type: 'object',
    properties: {
      user: { type: 'object', properties: { email: { type: 'string', pattern: '^\\S+@\\S+$' } }, required: ['email'] },
      n: { type: 'number', minimum: 10 },
    },
    required: ['user'],
  };
  const errs = validateSchema({ n: 3, user: { email: 'bad' } }, schema);
  const paths = errs.map((e) => e.path);
  assert.ok(paths.includes('user.email'));
  assert.ok(paths.includes('n'));
  assert.throws(() => validateOrThrow({ n: 1, user: {} }, schema, 'input'), /input 校验失败/);
});

test('schema：嵌套数组项', () => {
  const schema = { type: 'array', items: { type: 'object', properties: { v: { type: 'integer' } }, required: ['v'] } };
  const errs = validateSchema([{ v: 1 }, { v: 'x' }], schema);
  assert.equal(errs.length, 1);
  assert.equal(errs[0].path, '[1].v');
});
