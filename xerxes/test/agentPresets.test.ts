// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { AgentPresetRoster } from '../src/agents/presets.js'

const roots: string[] = []

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function roster(): AgentPresetRoster {
  const home = mkdtempSync(join(tmpdir(), 'xerxes-presets-'))
  roots.push(home)
  return new AgentPresetRoster({
    home,
    projectDirectory: home,
    userDirectory: join(home, 'agents'),
    settingsPath: join(home, 'agent-presets.json'),
  })
}

describe('AgentPresetRoster', () => {
  test('lists live built-ins with the deployment default', () => {
    const presets = roster().list()
    expect(presets.some(row => row.id === 'default' && row.trust === 'system' && row.isDefault)).toBe(true)
    expect(presets.some(row => row.id === 'objective' && row.trust === 'system')).toBe(true)
  })

  test('duplicates a preset as an editable self-contained user composition', () => {
    const presets = roster()
    const copy = presets.copy('default', 'my-agent', 'My Agent')
    expect(copy).toMatchObject({ id: 'my-agent', name: 'My Agent', trust: 'user', manageable: true })
    expect(copy.path).toEndWith('/agents/my-agent/agent.yaml')
    const read = presets.read('my-agent')
    expect(read.content).toContain('name: "my-agent"')
    expect(read.content).toContain('system_prompt: |-')
    expect(read.content).toContain('subagents:')
    expect(presets.validate('my-agent').broken).toBeUndefined()
  })

  test('persists the default and clears it when that user preset is removed', () => {
    const presets = roster()
    presets.copy('default', 'my-agent')
    presets.setDefault('my-agent')
    expect(presets.defaultId).toBe('my-agent')
    expect(presets.list().find(row => row.id === 'my-agent')?.isDefault).toBe(true)
    presets.remove('my-agent')
    expect(presets.defaultId).toBe('default')
    expect(presets.list().some(row => row.id === 'my-agent')).toBe(false)
  })

  test('validates writes before replacing the working composition', () => {
    const presets = roster()
    const copy = presets.copy('default', 'my-agent')
    const before = readFileSync(copy.path!, 'utf8')
    expect(() => presets.write('my-agent', 'version: 1\nagent:\n  name: other\n')).toThrow()
    expect(readFileSync(copy.path!, 'utf8')).toBe(before)
  })

  test('refuses to mutate shipped presets and duplicate identifiers', () => {
    const presets = roster()
    expect(() => presets.remove('default')).toThrow("not user-removable")
    presets.copy('default', 'my-agent')
    expect(() => presets.copy('default', 'my-agent')).toThrow('already exists')
    expect(() => presets.copy('default', '../escape')).toThrow('lowercase letters')
  })
})
