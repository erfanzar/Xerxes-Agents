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
