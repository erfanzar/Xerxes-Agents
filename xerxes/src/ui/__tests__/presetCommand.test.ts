// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { findSlashCommand } from '../app/slash/registry.js'

describe('/preset slash command', () => {
  it('lists the DSH-style roster and selects Creator mode through typed RPCs', async () => {
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const pages: string[] = []
    const messages: string[] = []
    const command = findSlashCommand('preset')
    const creatorCommand = findSlashCommand('creator')
    if (!command || !creatorCommand) throw new Error('preset commands missing')
    const ctx = {
      gateway: {
        gw: {},
        rpc: async (method: string, params: Record<string, unknown>) => {
          calls.push({ method, params })
          if (method === 'agentPreset.list') return {
            ok: true,
            presets: [
              { id: 'default', name: 'Default', trust: 'system', is_default: true, description: 'Coding agent' },
              { id: 'creator', name: 'Creator', trust: 'system', is_default: false, description: 'Authors presets' },
            ],
          }
          return { ok: true, agent_preset: String(params.agent_preset ?? '') }
        },
      },
      guarded: <T>(fn: (value: T) => void) => fn,
      guardedErr: () => undefined,
      session: {
        guardBusySessionSwitch: () => false,
        newSession: (message: string, title: string, agentPreset: string) => {
          messages.push(`${title}|${agentPreset}|${message}`)
        },
      },
      sid: 'session-1',
      stale: () => false,
      transcript: {
        config: (text: string) => messages.push(text),
        page: (text: string) => pages.push(text),
        sys: (text: string) => messages.push(text),
      },
      ui: {},
    } as never

    command.run('list', ctx, 'preset')
    command.run('creator', ctx, 'preset')
    creatorCommand.run('', ctx, 'creator')
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(calls).toEqual([
      { method: 'agentPreset.list', params: {} },
      { method: 'agentPreset.select', params: { agent_preset: 'creator' } },
    ])
    expect(pages.join('\n')).toContain('Creator')
    expect(messages.join('\n')).toContain('fixed after the first turn')
    expect(messages.join('\n')).toContain('Creator mode|creator|Creator mode started')
  })
})
