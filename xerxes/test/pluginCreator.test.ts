// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { AgentPresetRoster } from '../src/agents/presets.js'
import { loadBuiltinAgentDefinitions } from '../src/agents/definitions.js'
import { ManagedPlugins } from '../src/extensions/managedPlugins.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'

test('plugin creator is a resolvable preset with authoring instructions and tools', async () => {
  const home = await mkdtemp(join(tmpdir(), 'plugin-creator-'))
  try {
    const roster = new AgentPresetRoster({ home, projectDirectory: home })
    expect(roster.resolve('plugin-creator').trust).toBe('system')
    const definition = roster.definition('plugin-creator')!
    expect(definition.systemPrompt).toContain('authoring-tool-plugins')
    expect(definition.tools).toContain('SkillTool')
    expect(definition.tools).toContain('exec_command')
    expect(loadBuiltinAgentDefinitions(join(home, 'missing')).get('plugin-creator')?.systemPrompt).toContain('authoring-tool-plugins')
  } finally { await rm(home, { recursive: true, force: true }) }
})

test('documented example installs and executes through the managed tool boundary', async () => {
  const home = await mkdtemp(join(tmpdir(), 'plugin-example-'))
  try {
    const manager = new ManagedPlugins(join(home, 'plugins.json'))
    await manager.change('install', resolve(import.meta.dir, '../../examples/plugins/text-stats.ts'))
    const tools = new ToolRegistry()
    manager.registerTools(tools)
    const call = (args: (string | number)[]) => tools.execute({ id: 'example', type: 'function', function: { name: 'plugin_text_stats', arguments: { args } } }, { metadata: {} })
    expect(JSON.parse(String(await call(['Hello world\n🙂'])))).toEqual({ characters: 13, words: 3, lines: 2 })
    expect(JSON.parse(String(await call([''])))).toEqual({ characters: 0, words: 0, lines: 0 })
    await expect(call([42])).rejects.toThrow('exactly one string')
    await manager.change('disable', 'text-statistics')
    await expect(call(['hello'])).rejects.toThrow('not registered')
  } finally { await rm(home, { recursive: true, force: true }) }
})
