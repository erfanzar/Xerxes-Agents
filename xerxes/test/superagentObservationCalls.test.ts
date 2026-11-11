// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { WorkspaceObservationCalls } from '../src/daemon/observationCalls.js'

const input = { ability: 'github.read', view: 'issues' }

test('observation deadlines return without trusting a host that ignores abort, and keep its capacity occupied', async () => {
  let finish = (): void => {}; let calls = 0
  const pending = new Promise<void>(resolve => { finish = resolve })
  const service = new WorkspaceObservationCalls(() => ({ github: async () => { calls += 1; await pending; return { late: true } } }), 10)
  const connection = {}
  try {
    await expect(service.read(connection, '/unused', input)).rejects.toThrow('deadline')
    await expect(service.read(connection, '/unused', input)).rejects.toThrow('deadline')
    await expect(service.read(connection, '/unused', input)).rejects.toThrow('capacity')
    expect(calls).toBe(2)
    finish(); await Bun.sleep(1)
    expect(await service.read(connection, '/unused', input)).toEqual({ late: true })
  } finally { finish(); service.close() }
})

test('separate clients cannot exceed the shared sixteen-read admission bound', async () => {
  let finish = (): void => {}; let calls = 0
  const pending = new Promise<void>(resolve => { finish = resolve })
  const service = new WorkspaceObservationCalls(() => ({ github: async () => { calls += 1; await pending; return {} } }))
  const work = Array.from({ length: 16 }, () => service.read({}, '/unused', input).catch(() => ({})))
  try {
    await expect(service.read({}, '/unused', input)).rejects.toThrow('capacity')
    expect(calls).toBe(16)
  } finally { finish(); service.close(); await Promise.all(work) }
})

test('disconnect before queued admission cannot start a later host read', async () => {
  let calls = 0; const connection = {}
  const service = new WorkspaceObservationCalls(() => ({ github: async () => { calls += 1; return {} } }))
  service.cancel(connection)
  try { await expect(service.read(connection, '/unused', input)).rejects.toThrow('closed'); expect(calls).toBe(0) }
  finally { service.close() }
})

test('closing observation admission aborts active reads and refuses future ones', async () => {
  let ready = (): void => {}; let finish = (): void => {}; let signal: AbortSignal | undefined
  const started = new Promise<void>(resolve => { ready = resolve })
  const pending = new Promise<void>(resolve => { finish = resolve })
  const service = new WorkspaceObservationCalls(() => ({ github: async (_view, received) => { signal = received; ready(); await pending; return {} } }))
  const work = service.read({}, '/unused', input)
  try {
    await started; service.close(); await expect(work).rejects.toThrow('cancelled')
    expect(signal?.aborted).toBe(true)
    await expect(service.read({}, '/unused', input)).rejects.toThrow('closed')
  } finally { finish(); service.close() }
})
