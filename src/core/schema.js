/**
 * NovaFlagship · L2 协议层：JSON Schema 子集校验器
 * ------------------------------------------------------------------
 * 覆盖 MCP / OpenAI function calling 实际用到的子集：
 *   type / enum / const / required / properties / items /
 *   minimum / maximum / minLength / maxLength / pattern /
 *   minItems / maxItems / additionalProperties
 * 错误带 JSON Path（如 "a.b[0].c"），与 AJV 的 error 风格对齐，便于迁移。
 */

const TYPES = {
  string: (v) => typeof v === 'string',
  number: (v) => typeof v === 'number' && Number.isFinite(v),
  integer: (v) => typeof v === 'number' && Number.isInteger(v),
  boolean: (v) => typeof v === 'boolean',
  array: (v) => Array.isArray(v),
  object: (v) => v !== null && typeof v === 'object' && !Array.isArray(v),
  null: (v) => v === null,
};

/** 校验 data 是否满足 schema。返回错误数组（空数组 = 通过）。 */
export function validateSchema(data, schema, path = '') {
  const errors = [];
  const at = (p) => (path ? `${path}.${p}` : p);

  if (schema === true || schema === undefined || schema === null) return errors;
  if (schema === false) {
    errors.push({ path, keyword: 'false', message: 'schema 为 false，任何值都不合法' });
    return errors;
  }

  if (schema.enum && !schema.enum.some((v) => deepEqual(v, data))) {
    errors.push({ path, keyword: 'enum', message: `取值必须属于 ${JSON.stringify(schema.enum)}` });
  }
  if ('const' in schema && !deepEqual(schema.const, data)) {
    errors.push({ path, keyword: 'const', message: `必须等于常量 ${JSON.stringify(schema.const)}` });
  }
  if (schema.type) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((t) => TYPES[t]?.(data))) {
      errors.push({ path, keyword: 'type', message: `类型应为 ${types.join(' | ')}，实际为 ${typeOf(data)}` });
      return errors; // 类型不符则后续检查无意义
    }
  }

  if (typeof data === 'string') {
    if (schema.minLength !== undefined && data.length < schema.minLength) {
      errors.push({ path, keyword: 'minLength', message: `长度 ${data.length} < ${schema.minLength}` });
    }
    if (schema.maxLength !== undefined && data.length > schema.maxLength) {
      errors.push({ path, keyword: 'maxLength', message: `长度 ${data.length} > ${schema.maxLength}` });
    }
    if (schema.pattern && !new RegExp(schema.pattern).test(data)) {
      errors.push({ path, keyword: 'pattern', message: `不匹配模式 ${schema.pattern}` });
    }
  }

  if (typeof data === 'number') {
    if (schema.minimum !== undefined && data < schema.minimum) {
      errors.push({ path, keyword: 'minimum', message: `${data} < ${schema.minimum}` });
    }
    if (schema.maximum !== undefined && data > schema.maximum) {
      errors.push({ path, keyword: 'maximum', message: `${data} > ${schema.maximum}` });
    }
  }

  if (Array.isArray(data)) {
    if (schema.minItems !== undefined && data.length < schema.minItems) {
      errors.push({ path, keyword: 'minItems', message: `元素数 ${data.length} < ${schema.minItems}` });
    }
    if (schema.maxItems !== undefined && data.length > schema.maxItems) {
      errors.push({ path, keyword: 'maxItems', message: `元素数 ${data.length} > ${schema.maxItems}` });
    }
    if (schema.items) {
      data.forEach((item, i) => {
        errors.push(...validateSchema(item, schema.items, `${path}[${i}]`));
      });
    }
  }

  if (TYPES.object(data)) {
    const keys = Object.keys(data);
    for (const req of schema.required || []) {
      if (!(req in data)) {
        errors.push({ path: at(req), keyword: 'required', message: `缺少必填字段 "${req}"` });
      }
    }
    if (schema.properties) {
      for (const [key, sub] of Object.entries(schema.properties)) {
        if (key in data) errors.push(...validateSchema(data[key], sub, at(key)));
      }
    }
    if (schema.additionalProperties === false && schema.properties) {
      for (const key of keys) {
        if (!(key in schema.properties)) {
          errors.push({ path: at(key), keyword: 'additionalProperties', message: `不允许的额外字段 "${key}"` });
        }
      }
    }
  }
  return errors;
}

/** 校验不通过则抛出带全部错误的异常。 */
export function validateOrThrow(data, schema, label = 'input') {
  const errors = validateSchema(data, schema);
  if (errors.length > 0) {
    const detail = errors.map((e) => `${e.path || '<root>'}: ${e.message}`).join('; ');
    throw new Error(`${label} 校验失败: ${detail}`);
  }
}

function typeOf(v) {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  return typeof v;
}

function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a !== typeof b || a === null || b === null) return false;
  if (typeof a !== 'object') return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => deepEqual(a[k], b[k]));
}
