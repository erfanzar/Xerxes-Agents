// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { Store, type XerxesLike } from '../src/desktop/renderer/store.js'
import type { DaemonEvent } from '../src/desktop/renderer/types.js'

class Bridge implements XerxesLike {
  private handlers = new Set<(event: DaemonEvent) => void>()
  call(method: string): Promise<Record<string, unknown>> {
    if (method === 'initialize') return Promise.resolve({ session_id: 's1', session: { id: 's1', title: '', cwd: '/work' }, model: 'm', cwd: '/work', daemon_protocol: 35 })
    return Promise.resolve({ ok: true, sessions: [] })
  }
  push(type: string, payload: Record<string, unknown>): void { for (const handler of this.handlers) handler({ type, payload }) }
  subscribe(handler: (event: DaemonEvent) => void): () => void { this.handlers.add(handler); return () => this.handlers.delete(handler) }
}

test('the computer-use permissions notice waits until a task actually tries to use the computer', () => {
  const bridge = new Bridge()
  ;(globalThis as { window?: unknown }).window = { xerxes: { onEvent: (handler: (event: DaemonEvent) => void) => bridge.subscribe(handler) } }
  const store = new Store()
  store.start(bridge)
  expect(store.getSnapshot().computerUseWanted).toBe(false)
  bridge.push('tool_call', { id: 't1', name: 'ReadFile', arguments: '{"file_path":"a.csv"}' })
  expect(store.getSnapshot().computerUseWanted).toBe(false)
  bridge.push('tool_call', { id: 't2', name: 'computer_use', arguments: '{"action":"screenshot"}' })
  expect(store.getSnapshot().computerUseWanted).toBe(true)
})
