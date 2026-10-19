// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { AgentDefinition } from '../src/agents/definitions.js'
import { SUBAGENT_BLOCKED_TOOLS } from '../src/agents/subagentManager.js'
import { addPluginToolsToBuiltinAgents, toolListAdmits, withPluginTools } from '../src/agents/toolPatterns.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { ManagedPlugins } from '../src/extensions/managedPlugins.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { createAgentState } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'
import { permissionDisposition } from '../src/streaming/permissions.js'
import { registerPluginManageTool } from '../src/tools/pluginManage.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

const plugin = (name: string, word = name) => `export function register(r) {
  r.registerTool('${name}', { description: 'Say ${word}', parameters: { type: 'object', properties: {} }, execute: () => '${word}' }, { name: '${name}' })
}`

const tool = (name: string): ToolDefinition => ({ type: 'function', function: { name, description: name, parameters: { type: 'object' } } })

function builtIn(name: string, tools: string[], allowedTools: string[] | null = null): AgentDefinition {
  return { name, description: name, systemPrompt: '', model: '', tools, allowedTools, excludeTools: [], source: 'built-in', maxDepth: 5, isolation: '' }
}

test('tool lists admit plugin tools through a trailing-star pattern, and only those', () => {
  expect(toolListAdmits(['ReadFile', 'plugin_*'], 'plugin_forecast')).toBe(true)
  expect(toolListAdmits(['ReadFile', 'plugin_*'], 'plugins_other')).toBe(false)
  expect(toolListAdmits(['ReadFile'], 'plugin_forecast')).toBe(false)
  expect(toolListAdmits(['*'], 'anything')).toBe(false)
})

test('refreshing plugin tools keeps the same array when nothing changed, so the request stays cache-stable', () => {
  const current = [tool('ReadFile'), tool('plugin_a')]
  expect(withPluginTools(current, [tool('plugin_a'), tool('ReadFile')])).toBe(current)
  expect(withPluginTools(current, [tool('plugin_a'), tool('plugin_b')])?.map(entry => entry.function.name)).toEqual(['ReadFile', 'plugin_a', 'plugin_b'])
  expect(withPluginTools(current, [])?.map(entry => entry.function.name)).toEqual(['ReadFile'])
})

test('shipped full-access agents see plugin tools; only main-session profiles may manage them; read-only agents see neither', () => {
  const definitions = new Map<string, AgentDefinition>([
    ['default', builtIn('default', ['ReadFile'])],
    ['coder', builtIn('coder', [], ['ReadFile'])],
    ['researcher', builtIn('researcher', ['ReadFile'], ['ReadFile'])],
    ['mine', { ...builtIn('mine', ['ReadFile']), source: 'user' } as AgentDefinition],
  ])
  addPluginToolsToBuiltinAgents(definitions, { manage: true })
  expect(definitions.get('default')?.tools).toEqual(['ReadFile', 'plugin_*', 'PluginManageTool'])
  expect(definitions.get('coder')?.tools).toEqual([])
  expect(definitions.get('coder')?.allowedTools).toEqual(['ReadFile', 'plugin_*'])
  expect(definitions.get('researcher')?.tools).toEqual(['ReadFile'])
  expect(definitions.get('mine')?.tools).toEqual(['ReadFile'])
  expect(SUBAGENT_BLOCKED_TOOLS.has('PluginManageTool')).toBe(true)
})

test('plugin changes always ask for approval, even in accept-all; list and inspect do not', () => {
  const call = (args: Record<string, string>) => ({ function: { name: 'PluginManageTool', arguments: args } })
  for (const action of ['install', 'reload', 'enable', 'disable', 'uninstall']) {
    expect(permissionDisposition(call({ action }), 'accept-all')).toBe('prompt')
  }
  expect(permissionDisposition(call({ action: 'list' }), 'accept-all')).toBe('allow')
  expect(permissionDisposition(call({ action: 'inspect' }), 'accept-all')).toBe('allow')
})

test('temporary installs are never saved, reach registries filled earlier, and can be kept permanently', async () => {
  const root = await mkdtemp(join(tmpdir(), 'plugin-temporary-'))
  try {
    const manifest = join(root, 'plugins.json')
    await writeFile(join(root, 'hello.ts'), plugin('hello'))
    const manager = new ManagedPlugins(manifest)
    await manager.load()
    // Filled before the install, as a running turn's registry would be.
    const tools = new ToolRegistry()
    manager.registerTools(tools)
    let changes = 0
    manager.onChange(() => { changes += 1 })
    await expect(manager.change('install', join(root, 'hello.ts'), { temporary: true })).resolves.toContain('this run of Xerxes only')
    expect(changes).toBe(1)
    expect(tools.definitions().map(entry => entry.function.name)).toContain('plugin_hello')
    expect(await tools.execute({ id: '1', type: 'function', function: { name: 'plugin_hello', arguments: {} } }, { metadata: {} })).toBe('hello')
    expect(manager.inventory()[0]).toMatchObject({ name: 'hello', temporary: true })
    expect(JSON.parse(await readFile(manifest, 'utf8'))).toEqual([])
    // Another change re-reads the manifest; the temporary install survives it.
    await writeFile(join(root, 'other.ts'), plugin('other'))
    await manager.change('install', join(root, 'other.ts'))
    expect(manager.inventory().map(entry => entry.name).sort()).toEqual(['hello', 'other'])
    // A restart forgets it.
    const restarted = new ManagedPlugins(manifest)
    await restarted.load()
    expect(restarted.inventory().map(entry => entry.name)).toEqual(['other'])
    // Installing it again without the temporary option keeps it.
    await expect(manager.change('install', join(root, 'hello.ts'))).resolves.toContain('permanently')
    expect(JSON.parse(await readFile(manifest, 'utf8')).map((entry: { names: string[] }) => entry.names[0]).sort()).toEqual(['hello', 'other'])
    // Uninstalling withdraws the tool from the same live registry.
    await manager.change('uninstall', 'hello')
    expect(tools.definitions().map(entry => entry.function.name)).not.toContain('plugin_hello')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('within one turn the model installs a plugin with approval and calls its tool at the next step', async () => {
  const root = await mkdtemp(join(tmpdir(), 'plugin-same-turn-'))
  try {
    await writeFile(join(root, 'greet.ts'), plugin('greet', 'hi from the new plugin'))
    // A model install runs the plugin's tests first and refuses without them.
    await writeFile(join(root, 'greet.test.ts'), `import { expect, test } from 'bun:test'\ntest('greets', () => expect(1).toBe(1))`)
    const manager = new ManagedPlugins(join(root, 'plugins.json'))
    await manager.load()
    const tools = new ToolRegistry()
    manager.registerTools(tools)
    registerPluginManageTool(tools, manager, () => root)
    const offered: string[][] = []
    const approvals: string[] = []
    let round = 0
    const llm: LlmClient = {
      async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
        offered.push((request.tools ?? []).map(entry => entry.function.name))
        round += 1
        if (round === 1) {
          yield { toolCalls: [{ id: 'install', type: 'function', function: { name: 'PluginManageTool', arguments: { action: 'install', path: 'greet.ts' } } }], finishReason: 'tool_calls' }
        } else if (round === 2) {
          yield { toolCalls: [{ id: 'use', type: 'function', function: { name: 'plugin_greet', arguments: {} } }], finishReason: 'tool_calls' }
        } else {
          yield { content: 'done', finishReason: 'stop' }
        }
      },
    }
    const state = createAgentState()
    for await (const _event of runTurn({ model: 'm', state, userMessage: 'greet someone', tools: tools.definitions(), permissionMode: 'accept-all' }, {
      llm,
      toolExecutor: tools,
      permissionBroker: { request: async request => { approvals.push(request.toolCall.function.name); return 'approve' } },
      refreshTools: current => withPluginTools(current, tools.definitions()),
    })) { /* drain */ }
    expect(approvals).toEqual(['PluginManageTool'])
    expect(offered[0]).not.toContain('plugin_greet')
    expect(offered[1]).toContain('plugin_greet')
    const results = state.messages.filter(message => message.role === 'tool').map(message => String(message.content))
    expect(results.some(result => result.includes('hi from the new plugin'))).toBe(true)
    expect(manager.inventory()[0]).toMatchObject({ name: 'greet', temporary: true })
  } finally { await rm(root, { recursive: true, force: true }) }
})
