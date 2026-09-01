// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { AgentPresetRoster } from '../src/agents/presets.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  AGENT_PRESET_INSPECT_TOOL_NAME,
  AGENT_PRESET_TOOL_NAME,
  CREATOR_RUNTIME_TOOL_NAME,
  registerAgentPresetTools,
} from '../src/tools/agentPresets.js'

const roots: string[] = []
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

test('creator preset tools are creator-local and author validated user presets', async () => {
  const home = mkdtempSync(join(tmpdir(), 'xerxes-preset-tools-'))
  roots.push(home)
  const roster = new AgentPresetRoster({ home, projectDirectory: home })
  const tools = new ToolRegistry()
  let changes = 0
  registerAgentPresetTools(tools, roster, { onChanged: () => { changes += 1 } })

  expect(tools.get(AGENT_PRESET_INSPECT_TOOL_NAME, 'default')).toBeDefined()
  expect(tools.get(AGENT_PRESET_TOOL_NAME, 'creator')).toBeDefined()
  expect(tools.get(CREATOR_RUNTIME_TOOL_NAME, 'creator')).toBeDefined()

  const creatorContext = { agentId: 'creator', metadata: { project_root: home } }
  const inspect = tools.get(AGENT_PRESET_INSPECT_TOOL_NAME, 'creator')!
  const listed = await inspect({ action: 'list' }, creatorContext, new AbortController().signal) as { presets: Array<{ id: string }> }
  expect(listed.presets.some(row => row.id === 'creator')).toBe(true)
  await expect(Promise.resolve().then(() => inspect(
    { action: 'list' },
    { agentId: 'default', metadata: { project_root: home } },
    new AbortController().signal,
  ))).rejects.toThrow('does not declare')

  const author = tools.get(AGENT_PRESET_TOOL_NAME, 'creator')!
  const copied = await author(
    { action: 'copy', from: 'creator', id: 'custom-agent', name: 'Custom Agent' },
    creatorContext,
    new AbortController().signal,
  ) as { preset: { id: string; manageable: boolean } }
  expect(copied.preset).toMatchObject({ id: 'custom-agent', manageable: true })
  expect(changes).toBe(1)
  expect(roster.validate('custom-agent').broken).toBeUndefined()
  expect(roster.definition('custom-agent')?.tools).toContain(AGENT_PRESET_TOOL_NAME)
})
