// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  creatorTraceValues,
  DeclarativeToolForge,
} from '../src/extensions/declarativeForge.js'
import { registerCreatorForgeTool } from '../src/tools/creatorForge.js'

const directories: string[] = []

afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, { force: true, recursive: true })
})

function forge(): DeclarativeToolForge {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-forge-'))
  directories.push(directory)
  return new DeclarativeToolForge(join(directory, 'forged-tools.json'))
}

test('declarative forge persists immutable versions and renders only declared scalar inputs', () => {
  const store = forge()
  const defined = store.define({
    name: 'release-note',
    version: '0.1.0',
    description: 'Format one verified release note.',
    parameters: [
      { name: 'scope', required: true },
      { name: 'summary', required: true },
      { name: 'prefix', default: 'Changed' },
    ],
    template: '{{prefix}} {{scope}}: {{summary}}',
  })
  expect(defined.name).toBe('release-note')
  expect(store.run('release-note', undefined, {
    scope: 'daemon',
    summary: 'added creator mode',
  })).toMatchObject({
    name: 'release-note',
    version: '0.1.0',
    output: 'Changed daemon: added creator mode',
  })

  const reopened = new DeclarativeToolForge(store.filePath)
  expect(reopened.list()).toHaveLength(1)
  expect(() => reopened.define({
    name: 'release-note',
    version: '0.1.0',
    description: 'replacement',
    template: 'replacement',
  })).toThrow('immutable')
  expect(() => reopened.run('release-note', '0.1.0', { typo: 'value' })).toThrow('not a declared parameter')
  expect(reopened.undefine('release-note', '0.1.0')).toBe(true)
  expect(reopened.list()).toEqual([])
})

test('declarative forge rejects executable-looking schema escapes instead of inventing a host capability', () => {
  const store = forge()
  expect(() => store.define({
    name: 'unsafe-tool',
    version: '0.1.0',
    description: 'Bad undeclared placeholder.',
    parameters: [],
    template: '{{command}}',
  })).toThrow('undeclared parameter command')
  expect(() => store.define({
    name: '../escape',
    version: '0.1.0',
    description: 'Bad name.',
    template: 'text',
  })).toThrow('lowercase letters')
})

test('CreatorForgeTool refuses durable catalog mutation in plan mode', async () => {
  const store = forge()
  const registry = new ToolRegistry()
  const metadata: Record<string, unknown> = { permission_mode: 'plan' }
  registerCreatorForgeTool(registry, store)

  await expect(registry.execute({
    id: 'forge-plan-1',
    type: 'function',
    function: {
      name: 'CreatorForgeTool',
      arguments: {
        action: 'define',
        name: 'briefing',
        version: '0.1.0',
        description: 'Format a briefing.',
        template: 'Briefing',
      },
    },
  }, { metadata, sessionId: 'session-plan' })).rejects.toThrow('disabled in plan mode')
  expect(store.list()).toEqual([])
  expect(creatorTraceValues(metadata)[0]).toMatchObject({ action: 'define', status: 'error' })
})

test('CreatorForgeTool is fail-closed and records an observable session trace', async () => {
  const store = forge()
  const registry = new ToolRegistry()
  const metadata: Record<string, unknown> = {}
  registerCreatorForgeTool(registry, store)

  expect(registry.capabilities('CreatorForgeTool')).toMatchObject({
    destructive: true,
    openWorld: false,
    readOnly: false,
  })
  const defined = JSON.parse(await registry.execute({
    id: 'forge-1',
    type: 'function',
    function: {
      name: 'CreatorForgeTool',
      arguments: {
        action: 'define',
        name: 'briefing',
        version: '0.1.0',
        description: 'Format a briefing.',
        parameters: [{ name: 'topic', required: true }],
        template: 'Briefing: {{topic}}',
      },
    },
  }, { metadata, sessionId: 'session-1' }))
  expect(defined.ok).toBe(true)

  const run = JSON.parse(await registry.execute({
    id: 'forge-2',
    type: 'function',
    function: {
      name: 'CreatorForgeTool',
      arguments: { action: 'run', name: 'briefing', input: { topic: 'telemetry' } },
    },
  }, { metadata, sessionId: 'session-1' }))
  expect(run.output).toBe('Briefing: telemetry')
  expect(creatorTraceValues(metadata).map(row => row.action)).toEqual(['define', 'run'])
})
