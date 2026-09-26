// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { prepareManagedRuntime, type ManagedRuntimeClient } from '../src/ui/lib/managedRuntime.js'

function fixture(result: Record<string, unknown>, current = 'old') {
  const calls: string[] = []
  const create = (verify: boolean): ManagedRuntimeClient => ({
    async start() { calls.push(`start:${verify}`) },
    async request<T>(method: string) {
      calls.push(method)
      return (method === 'runtime.status' ? { pid: 42, daemon_build_id: verify ? 'new' : current } : result) as T
    },
    close() { calls.push('close') },
  })
  const wait = async (pid: number) => { expect(pid).toBe(42); calls.push('exited') }
  return { calls, create, wait }
}
test('managed release upgrades use atomic idle restart before opening the new build', async () => {
  const f = fixture({ ok: true })
  expect(await prepareManagedRuntime(f.create, 'new', f.wait)).toEqual({ busy: false })
  expect(f.calls).toEqual(['start:false', 'runtime.status', 'runtime.restart_if_idle', 'close', 'exited', 'start:true', 'runtime.status', 'close'])
})
test('busy remote work stays connected to its old daemon without a forced stop', async () => {
  const f = fixture({ ok: false, busy: true })
  expect(await prepareManagedRuntime(f.create, 'new', f.wait)).toEqual({ busy: true })
  expect(f.calls).not.toContain('exited')
  expect(f.calls).not.toContain('start:true')
})
test('current builds need no restart and unsupported updates remain errors', async () => {
  const f = fixture({}, 'new')
  await prepareManagedRuntime(f.create, 'new', f.wait)
  expect(f.calls).not.toContain('runtime.restart_if_idle')
  const rejected = fixture({ ok: false, error: 'Unknown method' })
  await expect(prepareManagedRuntime(rejected.create, 'new', rejected.wait)).rejects.toThrow('Unknown method')
  expect(rejected.calls.at(-1)).toBe('close')
})
test('cancelled startup closes its owned connection without requesting a stop', async () => {
  const f = fixture({ ok: true })
  await expect(prepareManagedRuntime(() => ({ ...f.create(false), async start() { throw new Error('cancelled') } }), 'new', f.wait)).rejects.toThrow('cancelled')
  expect(f.calls).toEqual(['close'])
})

function legacyFixture(sessions: unknown[], statusExtra: Record<string, unknown> = {}, pid: number | null = 42) {
  const calls: string[] = []
  const create = (verify: boolean): ManagedRuntimeClient => ({
    async start() { calls.push(`start:${verify}`) },
    async request<T>(method: string) {
      calls.push(method)
      if (method === 'runtime.status') return { ...(pid === null ? {} : { pid }), daemon_build_id: verify ? 'new' : 'old', active_subagents: 0, ...statusExtra } as T
      if (method === 'runtime.restart_if_idle') return { ok: false, error: 'Unknown method: runtime.restart_if_idle' } as T
      if (method === 'session.active_list') return { ok: true, sessions } as T
      if (method === 'terminal.list') return { ok: true, terminals: [] } as T
      if (method === 'monitor.list') return { ok: true, monitors: [] } as T
      if (method === 'shutdown') return { ok: true } as T
      return {} as T
    },
    close() { calls.push('close') },
  })
  const wait = async (exited: number) => { calls.push(`exited:${exited}`) }
  return { calls, create, wait }
}

test('an idle runtime too old for idle-restart is asked to shut down and replaced, never killed', async () => {
  const f = legacyFixture([{ key: 'k1', status: 'idle' }])
  expect(await prepareManagedRuntime(f.create, 'new', f.wait, {})).toEqual({ busy: false })
  expect(f.calls).toEqual(['start:false', 'runtime.status', 'runtime.restart_if_idle', 'session.active_list', 'terminal.list', 'monitor.list', 'shutdown', 'close', 'exited:42', 'start:true', 'runtime.status', 'close'])
})

test('an old runtime with any running work is left alone and reported busy', async () => {
  for (const f of [
    legacyFixture([{ key: 'k1', status: 'working', active_turn_id: 't1' }]),
    legacyFixture([{ key: 'k1', status: 'idle' }], { active_subagents: 2 }),
    legacyFixture([{ key: 'k1', status: 'idle' }], { channels_configured: true }),
  ]) {
    expect(await prepareManagedRuntime(f.create, 'new', f.wait, {})).toEqual({ busy: true })
    expect(f.calls).not.toContain('shutdown')
  }
})

test('an old runtime without a reported pid uses a confirmed pid file, or is left running', async () => {
  const found = legacyFixture([], {}, null)
  expect(await prepareManagedRuntime(found.create, 'new', found.wait, { pidFallback: async () => 77 })).toEqual({ busy: false })
  expect(found.calls).toContain('exited:77')
  const unconfirmed = legacyFixture([], {}, null)
  await expect(prepareManagedRuntime(unconfirmed.create, 'new', unconfirmed.wait, { pidFallback: async () => undefined })).rejects.toThrow('identity is unavailable')
  expect(unconfirmed.calls).not.toContain('shutdown')
})
