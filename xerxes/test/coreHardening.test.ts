// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { LLMConfig, SecurityConfig, XerxesConfig } from '../src/core/config.js'
import {
  ClientError,
  ConfigurationError,
  FunctionExecutionError,
  XerxesError,
} from '../src/core/errors.js'
import { agentsSubdirFor, xerxesSubdirFor } from '../src/core/paths.js'
import { mergeChunk, mergeFields, safeJsonStringify, toJsonValue } from '../src/core/utils.js'

test('mergeFields and mergeChunk skip prototype-polluting keys from hostile deltas', () => {
  const polluted = JSON.parse('{"__proto__": {"polluted": "yes"}, "constructor": {"x": 1}, "prototype": {"y": 2}}')

  const target: Record<string, unknown> = { content: 'a', nested: {} }
  mergeFields(target, { content: 'b', nested: polluted })
  expect(target).toEqual({ content: 'ab', nested: {} })

  const response: Record<string, unknown> = { content: 'a' }
  mergeChunk(response, { content: 'b', ...polluted })
  expect(response).toEqual({ content: 'ab' })

  expect(({} as Record<string, unknown>).polluted).toBeUndefined()
  expect((Object.prototype as Record<string, unknown>).polluted).toBeUndefined()
  expect((Object.prototype as Record<string, unknown>).x).toBeUndefined()
})

test('toJsonValue and safeJsonStringify preserve __proto__ as data without prototype mutation', () => {
  const input = JSON.parse('{"__proto__": {"role": "data"}, "plain": 1}') as Record<string, unknown>

  const copied = toJsonValue(input) as Record<string, unknown>
  expect(Object.hasOwn(copied, '__proto__')).toBe(true)
  expect(Object.getPrototypeOf(copied)).toBe(Object.prototype)
  expect((copied.__proto__ as Record<string, unknown>).role).toBe('data')
  expect(copied.plain).toBe(1)

  const roundTripped = JSON.parse(safeJsonStringify(input))
  expect(Object.hasOwn(roundTripped, '__proto__')).toBe(true)
  expect(roundTripped).toEqual(input)
  expect(({} as Record<string, unknown>).role).toBeUndefined()
})

test('LLM and security secrets are redacted from toJSON and never reach persisted config files', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-core-secret-redaction-'))
  try {
    const llm = new LLMConfig({ api_key: 'explicit-secret' }, {})
    expect(llm.apiKey).toBe('explicit-secret')
    expect(llm.toJSON().api_key).toBeNull()
    expect(llm.toJSON().api_key_env_var).toBe('OPENAI_API_KEY')

    const security = new SecurityConfig({ api_key: 'security-secret' })
    expect(security.apiKey).toBe('security-secret')
    expect(security.toJSON().api_key).toBeNull()

    const config = new XerxesConfig(
      { llm: { api_key: 'llm-secret' }, security: { api_key: 'auth-secret' } },
      {},
    )
    const yamlPath = join(root, 'xerxes.yaml')
    const jsonPath = join(root, 'xerxes.json')
    config.toFile(yamlPath)
    config.toFile(jsonPath)
    for (const path of [yamlPath, jsonPath]) {
      const content = await readFile(path, 'utf8')
      expect(content).not.toContain('llm-secret')
      expect(content).not.toContain('auth-secret')
      expect(JSON.parse(JSON.stringify(config.toJSON())).llm.api_key).toBeNull()
    }

    const merged = config.merge(new XerxesConfig({}, {}))
    expect(merged.llm.apiKey).toBeUndefined()
    expect(merged.security.apiKey).toBeUndefined()
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('LLMConfig treats blank environment API keys as absent', () => {
  expect(new LLMConfig({}, { OPENAI_API_KEY: '' }).apiKey).toBeUndefined()
  expect(new LLMConfig({}, { OPENAI_API_KEY: '   ' }).apiKey).toBeUndefined()
  expect(new LLMConfig({}, { OPENAI_API_KEY: 'real-key' }).apiKey).toBe('real-key')
  expect(new LLMConfig({ api_key: 'explicit' }, { OPENAI_API_KEY: '   ' }).apiKey).toBe('explicit')
})

test('home subdir helpers reject traversal and absolute segments', () => {
  const environment = { XERXES_HOME: '/tmp/xerxes-core-traversal' }
  expect(xerxesSubdirFor(environment, 'daemon', 'logs')).toBe('/tmp/xerxes-core-traversal/daemon/logs')
  expect(xerxesSubdirFor(environment, 'a/b')).toBe('/tmp/xerxes-core-traversal/a/b')
  expect(() => xerxesSubdirFor(environment, '..')).toThrow(/unsafe path segment/)
  expect(() => xerxesSubdirFor(environment, 'a', '..')).toThrow(/unsafe path segment/)
  expect(() => xerxesSubdirFor(environment, 'a/../../b')).toThrow(/unsafe path segment/)
  expect(() => xerxesSubdirFor(environment, '/etc/passwd')).toThrow(/unsafe path segment/)

  expect(agentsSubdirFor('/tmp/xerxes-core-home', 'skills')).toBe('/tmp/xerxes-core-home/.agents/skills')
  expect(() => agentsSubdirFor('/tmp/xerxes-core-home', '..')).toThrow(/unsafe path segment/)
  expect(() => agentsSubdirFor('/tmp/xerxes-core-home', '/absolute')).toThrow(/unsafe path segment/)
})

test('readConfigFile wraps read failures and parse failures with the original cause', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-core-config-errors-'))
  try {
    const missingPath = join(root, 'missing.json')
    let readError: unknown
    try {
      XerxesConfig.fromFile(missingPath, { environment: {} })
    } catch (error) {
      readError = error
    }
    expect(readError).toBeInstanceOf(ConfigurationError)
    expect((readError as ConfigurationError).message).toContain('cannot be read')
    expect((readError as ConfigurationError).cause).toBeInstanceOf(Error)

    const invalidPath = join(root, 'invalid.json')
    await writeFile(invalidPath, '{ not json', 'utf8')
    let parseError: unknown
    try {
      XerxesConfig.fromFile(invalidPath, { environment: {} })
    } catch (error) {
      parseError = error
    }
    expect(parseError).toBeInstanceOf(ConfigurationError)
    expect((parseError as ConfigurationError).message).toContain('contains invalid JSON')
    expect((parseError as ConfigurationError).cause).toBeInstanceOf(Error)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('core errors forward causes through the native Error cause chain', () => {
  const original = new Error('root failure')

  const base = new XerxesError('base failed', {}, { cause: original })
  expect(base.cause).toBe(original)

  const functionError = new FunctionExecutionError('search', 'failed', original)
  expect(functionError.cause).toBe(original)
  expect(functionError).toBeInstanceOf(XerxesError)

  const clientError = new ClientError('anthropic', 'failed', original)
  expect(clientError.cause).toBe(original)

  const configError = new ConfigurationError('llm.apiKey', 'missing', {}, { cause: original })
  expect(configError.cause).toBe(original)

  const withoutCause = new FunctionExecutionError('noop', 'failed')
  expect(withoutCause.cause).toBeUndefined()
})
