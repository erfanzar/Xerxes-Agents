// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, it, vi } from 'vitest'
import { findSlashCommand } from '../app/slash/registry.js'
import { getOverlayState } from '../app/overlayStore.js'

it('makes global restart explicit, rejects stale confirmation, and preserves busy work', async () => {
  const rpc = vi.fn().mockResolvedValueOnce({ ok: false, busy: true }).mockRejectedValueOnce(new Error('offline'))
  const sys = vi.fn(), error = vi.fn()
  let stale = false
  const ctx = { gateway: { rpc }, stale: () => stale, guarded: (fn: unknown) => fn, guardedErr: error, transcript: { sys } } as never
  findSlashCommand('restart')!.run('', ctx, 'restart')
  expect(rpc).not.toHaveBeenCalled()
  stale = true; getOverlayState().confirm!.onConfirm(); expect(rpc).not.toHaveBeenCalled()
  stale = false; getOverlayState().confirm!.onConfirm()
  await vi.waitFor(() => expect(sys).toHaveBeenCalledWith(expect.stringContaining('Nothing was stopped')))
  expect(rpc).toHaveBeenCalledWith('runtime.restart_if_idle', {})
  getOverlayState().confirm!.onConfirm()
  await vi.waitFor(() => expect(error).toHaveBeenCalled())
})

it('exposes daemon identity and confirms shutdown of all workspaces', async () => {
  const rpc = vi.fn().mockResolvedValueOnce({ ok: true, pid: 123, daemon_protocol: 35, daemon_build_id: 'test', runtime_ready: true, active_subagents: 8 }).mockResolvedValueOnce({ ok: true })
  const page = vi.fn(), sys = vi.fn(), die = vi.fn()
  const ctx = { gateway: { rpc }, session: { die }, stale: () => false, guarded: (fn: unknown) => fn, guardedErr: vi.fn(), transcript: { page, sys } } as never
  findSlashCommand('daemon')!.run('status', ctx, 'daemon')
  await vi.waitFor(() => expect(page).toHaveBeenCalledWith(expect.stringContaining('active subagents: 8'), 'Daemon'))
  findSlashCommand('daemon')!.run('stop', ctx, 'daemon')
  expect(rpc).toHaveBeenCalledTimes(1)
  expect(getOverlayState().confirm!.detail).toContain('every workspace')
  getOverlayState().confirm!.onConfirm()
  await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('shutdown', {}))
  await vi.waitFor(() => expect(die).toHaveBeenCalledOnce())
})
