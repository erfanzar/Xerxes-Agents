// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ExecutionRegistry } from '../src/runtime/executionRegistry.js'
import { ToolPool, assembleToolPool } from '../src/runtime/toolPool.js'

test('tool pools are immutable snapshots and an absent registry returns the default empty pool', () => {
  const denied = new Set(['Write'])
  const categories = ['filesystem']
  const registry = new ExecutionRegistry()
  registry.registerTool('Read', undefined, {
    category: 'filesystem',
    description: 'Read workspace files',
    safe: true,
  })

  const empty = assembleToolPool(undefined, {
    categories,
    denyTools: denied,
    safeOnly: true,
  })
  expect(empty).toBeInstanceOf(ToolPool)
  expect(empty.toolCount).toBe(0)
  expect(empty.categories).toEqual([])
  expect([...empty.deniedTools]).toEqual([])
  expect(empty.safeOnly).toBe(false)

  const pool = assembleToolPool(registry, { categories, denyTools: denied })
  categories.push('mcp')
  denied.add('Read')
  registry.registerTool('Later', undefined, { category: 'filesystem' })

  expect(pool.toolNames).toEqual(['Read'])
  expect(pool.categories).toEqual(['filesystem'])
  expect([...pool.deniedTools]).toEqual(['Write'])
  expect(Object.isFrozen(pool)).toBe(true)
  expect(Object.isFrozen(pool.tools)).toBe(true)
  expect(Object.isFrozen(pool.categories)).toBe(true)
  expect(Object.isFrozen(pool.getTool('Read'))).toBe(true)
  expect((pool.deniedTools as ReadonlySet<string> & { add?: (name: string) => void }).add).toBeUndefined()
})

test('pool assembly applies safe, category, exact-name, prefix, and MCP filters in registry order', () => {
  const registry = new ExecutionRegistry()
  registry.registerTool('Read', undefined, {
    category: 'filesystem',
    description: 'Read workspace files',
    safe: true,
    sourceHint: 'builtin',
  })
  registry.registerTool('Write', undefined, {
    category: 'filesystem',
    description: 'Write workspace files',
    safe: false,
  })
  registry.registerTool('McpRemote', undefined, {
    category: 'filesystem',
    description: 'Remote read',
    safe: true,
    sourceHint: 'MCP:remote',
  })
  registry.registerTool('mcp__server__status', undefined, {
    category: 'filesystem',
    description: 'Server status',
    safe: true,
    sourceHint: 'plugin:status',
  })
  registry.registerTool('AgentInfo', undefined, {
    category: 'agent',
    description: 'Inspect agents',
    safe: true,
  })

  const pool = assembleToolPool(registry, {
    categories: ['filesystem'],
    denyTools: ['Write'],
    denyPrefixes: ['mcp__'],
    safeOnly: true,
    includeMcp: false,
  })

  expect(pool.toolNames).toEqual(['Read'])
  expect([...pool.deniedTools]).toEqual(['Write'])
  expect(pool.categories).toEqual(['filesystem'])
  expect(pool.safeOnly).toBe(true)
  expect(pool.getTool('Read')?.category).toBe('filesystem')
  expect(pool.getTool('write')).toBeUndefined()
})

test('pool schemas preserve explicit schemas, generate minimal fallbacks, and render source-compatible markdown', () => {
  const explicit = {
    name: 'Custom',
    input_schema: {
      type: 'object',
      properties: { path: { type: 'string' } },
    },
  }
  const registry = new ExecutionRegistry()
  registry.registerTool('Read', undefined, {
    category: 'filesystem',
    description: 'Read workspace files',
    safe: true,
  })
  registry.registerTool('Custom', undefined, {
    category: 'meta',
    description: 'Custom schema',
    schema: explicit,
  })
  registry.registerTool('Empty', undefined, {
    schema: {},
  })

  const pool = assembleToolPool(registry, { denyTools: ['blocked', 'zed'] })
  explicit.input_schema.properties.path.type = 'number'

  expect(pool.toSchemas()).toEqual([
    {
      name: 'Custom',
      input_schema: {
        type: 'object',
        properties: { path: { type: 'string' } },
      },
    },
    {
      name: 'Empty',
      description: 'Execute Empty',
      input_schema: { type: 'object', properties: {} },
    },
    {
      name: 'Read',
      description: 'Read workspace files',
      input_schema: { type: 'object', properties: {} },
    },
  ])
  expect(Object.isFrozen(pool.toSchemas())).toBe(true)
  expect(Object.isFrozen(pool.toSchemas()[0])).toBe(true)
  expect(pool.asMarkdown()).toBe(
    [
      '# Tool Pool',
      '',
      'Tools: 3',
      'Safe only: false',
      'Categories: all',
      'Denied: blocked, zed',
      '',
      '- **Custom** (meta) — Custom schema',
      '- **Empty** — ',
      '- **Read** [safe] (filesystem) — Read workspace files',
    ].join('\n'),
  )
})
