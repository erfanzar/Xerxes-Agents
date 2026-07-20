// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { findSlashCommand } from '../app/slash/registry.js'

describe('/ultra slash command', () => {
  it('is registered with set-on semantics documented', () => {
    const command = findSlashCommand('ultra')

    expect(command).toBeDefined()
    expect(command?.name).toBe('ultra')
    expect(command?.help).toContain('off')
  })

  it('forwards bare and off invocations to the daemon like /yolo', async () => {
    const sent: string[] = []
    const command = findSlashCommand('ultra')
    if (!command) throw new Error('ultra command missing')
    const ctx = {
      gateway: {
        gw: {
          request: async (_method: string, params: Record<string, unknown>) => {
            sent.push(String(params['command']))
            return { ok: true }
          },
        },
        rpc: async () => ({}),
      },
      guarded: <T>(fn: (value: T) => void) => fn,
      guardedErr: () => undefined,
      sid: 'test-session',
      stale: () => false,
      transcript: { page: () => undefined, sys: () => undefined },
      ui: {},
    } as never

    command.run('', ctx)
    command.run('off', ctx)
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(sent).toEqual(['ultra', 'ultra off'])
  })
})
