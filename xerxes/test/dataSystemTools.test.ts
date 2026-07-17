// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { FunctionExecutionError } from '../src/core/errors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerDataTools } from '../src/tools/dataTools.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import { registerSystemTools } from '../src/tools/systemTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

function result(value: string): JsonObject {
  return JSON.parse(value) as JsonObject
}

async function inWorkspace(run: (workspace: string) => Promise<void>): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-bun-data-tools-'))
  try {
    await run(workspace)
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
}

test('JSONProcessor keeps JSON file operations inside the workspace and queries nested records', async () => {
  await inWorkspace(async workspace => {
    const registry = new ToolRegistry()
    registerDataTools(registry, new WorkspacePathResolver(workspace))
    const context = { metadata: {} }

    const saved = result(await registry.execute(call('JSONProcessor', {
      operation: 'save',
      file_path: 'state/settings.json',
      data: { users: [{ name: 'Ada', roles: ['admin'] }] },
    }), context))
    expect(saved.success).toBeTrue()
    expect(saved.file_path).toBe('state/settings.json')

    const loaded = result(await registry.execute(call('JSONProcessor', {
      operation: 'load',
      file_path: 'state/settings.json',
    }), context))
    expect(loaded.data).toEqual({ users: [{ name: 'Ada', roles: ['admin'] }] })
    const loadedData = loaded.data
    if (loadedData === undefined) throw new Error('JSONProcessor load did not return data')

    const queried = result(await registry.execute(call('JSONProcessor', {
      operation: 'query',
      data: loadedData,
      query: 'users[0].roles.0',
    }), context))
    expect(queried.result).toBe('admin')

    await expect(registry.execute(call('JSONProcessor', {
      operation: 'save',
      file_path: '../outside.json',
      data: { outside: true },
    }), context)).rejects.toBeInstanceOf(FunctionExecutionError)
  })
})

test('CSVProcessor handles quoted cells, read limits, conversion, and analysis', async () => {
  await inWorkspace(async workspace => {
    const registry = new ToolRegistry()
    registerDataTools(registry, new WorkspacePathResolver(workspace))
    const context = { metadata: {} }

    const written = result(await registry.execute(call('CSVProcessor', {
      operation: 'write',
      file_path: 'reports/people.csv',
      data: [
        { age: 30, name: 'Ada', note: 'comma, kept' },
        { age: 31, name: 'Grace', note: 'plain' },
      ],
    }), context))
    expect(written.rows_written).toBe(2)

    const read = result(await registry.execute(call('CSVProcessor', {
      operation: 'read',
      file_path: 'reports/people.csv',
      max_rows: 1,
    }), context))
    expect(read.count).toBe(1)
    expect(read.data).toEqual([{ age: '30', name: 'Ada', note: 'comma, kept' }])

    const converted = result(await registry.execute(call('CSVProcessor', {
      operation: 'convert',
      file_path: 'reports/people.csv',
    }), context))
    expect(converted.count).toBe(2)

    const analyzed = result(await registry.execute(call('CSVProcessor', {
      operation: 'analyze',
      file_path: 'reports/people.csv',
    }), context))
    expect(analyzed.headers).toEqual(['age', 'name', 'note'])
    expect(analyzed.total_rows).toBe(3)
  })
})

test('text, conversion, and date tools preserve practical dependency-free operations', async () => {
  await inWorkspace(async workspace => {
    const registry = new ToolRegistry()
    registerDataTools(registry, new WorkspacePathResolver(workspace))
    const context = { metadata: {} }

    const extracted = result(await registry.execute(call('TextProcessor', {
      operation: 'extract',
      text: 'Contact ada@example.com or visit https://example.com.',
      pattern: 'emails',
    }), context))
    expect(extracted.matches).toEqual(['ada@example.com'])

    const base64 = result(await registry.execute(call('DataConverter', {
      data: 'hello',
      from_format: 'text',
      to_format: 'base64',
    }), context))
    expect(base64.output).toBe('aGVsbG8=')

    const hashes = result(await registry.execute(call('DataConverter', {
      data: 'hello',
      from_format: 'text',
      to_format: 'hash',
    }), context))
    expect((hashes.output as JsonObject).sha256).toBe(
      '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824',
    )

    const formatted = result(await registry.execute(call('DateTimeProcessor', {
      operation: 'format',
      date_string: '2025-01-15T13:04:05Z',
      fmt: '%B %d, %Y',
      timezone: 'UTC',
    }), context))
    expect(formatted.formatted).toBe('January 15, 2025')

    const shifted = result(await registry.execute(call('DateTimeProcessor', {
      operation: 'delta',
      date_string: '2025-01-15T00:00:00Z',
      delta_days: 2,
      delta_minutes: 30,
      timezone: 'UTC',
    }), context))
    expect((shifted.delta as JsonObject).total_seconds).toBe(174_600)
    expect(shifted.new).toBe('2025-01-17T00:30:00.000Z')
  })
})

test('TextProcessor refuses model-supplied regexes on oversized subjects', async () => {
  await inWorkspace(async workspace => {
    const registry = new ToolRegistry()
    registerDataTools(registry, new WorkspacePathResolver(workspace))
    const context = { metadata: {} }
    const oversized = 'x'.repeat(1_000_001)

    await expect(registry.execute(call('TextProcessor', {
      operation: 'extract',
      pattern: 'x+',
      text: oversized,
    }), context)).rejects.toThrow('subject limit')
    await expect(registry.execute(call('TextProcessor', {
      operation: 'replace',
      pattern: 'x+',
      replacement: 'y',
      text: oversized,
    }), context)).rejects.toThrow('subject limit')

    // Pattern-free operations stay usable on the same subject.
    const stats = result(await registry.execute(call('TextProcessor', {
      operation: 'stats',
      text: oversized,
    }), context))
    expect(stats.length).toBe(1_000_001)
  })
})

test('SystemInfo uses Bun and Node primitives, while EnvironmentManager redacts sensitive values', async () => {
  const registry = new ToolRegistry()
  registerSystemTools(registry)
  const context = { metadata: {} }

  const system = result(await registry.execute(call('SystemInfo', { info_type: 'all' }), context))
  expect((system.os as JsonObject).system).toBeString()
  expect(typeof (system.memory as JsonObject).total).toBe('number')
  expect(typeof (system.disk as JsonObject).available).toBe('boolean')
  expect(Array.isArray((system.network as JsonObject).interfaces)).toBeTrue()

  const key = 'XERXES_TEST_API_TOKEN'
  const previous = process.env[key]
  process.env[key] = 'not-for-tool-output'
  try {
    const sensitive = result(await registry.execute(call('EnvironmentManager', {
      operation: 'get',
      key,
    }), context))
    // Redaction-list keys must be indistinguishable from unset keys (no existence oracle).
    expect(sensitive.exists).toBeFalse()
    expect(sensitive.redacted).toBeFalse()
    expect(sensitive.value).toBeNull()

    const unset = result(await registry.execute(call('EnvironmentManager', {
      operation: 'get',
      key: 'XERXES_TEST_DEFINITELY_UNSET',
    }), context))
    expect(unset).toEqual({ ...sensitive, key: 'XERXES_TEST_DEFINITELY_UNSET' })

    const listed = result(await registry.execute(call('EnvironmentManager', { operation: 'list' }), context))
    expect(typeof listed.count).toBe('number')
    expect((listed.environment as JsonObject)[key]).toBeUndefined()
  } finally {
    if (previous === undefined) delete process.env[key]
    else process.env[key] = previous
  }

  await expect(registry.execute(call('EnvironmentManager', {
    operation: 'set',
    key: 'NODE_ENV',
  }), context)).rejects.toBeInstanceOf(FunctionExecutionError)
})
