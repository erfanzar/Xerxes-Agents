// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, it, vi } from 'vitest'
import { findSlashCommand } from '../app/slash/registry.js'
import { getOverlayState } from '../app/overlayStore.js'
import { patchTurnState, resetTurnState } from '../app/turnStore.js'

it('pages scoped search results and preserves index coverage warnings', async () => {
  const page = vi.fn(), error = vi.fn()
  const rpc = vi.fn().mockResolvedValueOnce({ ok: true, results: [{ session_id: 'one', message_index: 4, role: 'assistant', excerpt: 'found this', title: 'Saved work' }], stats: { unrecognized_messages: 2 } }).mockRejectedValueOnce(new Error('offline'))
  const ctx = { gateway: { rpc }, guarded: (fn: unknown) => fn, guardedErr: error, transcript: { page, sys: vi.fn() } } as never
  findSlashCommand('search')!.run('--session one --limit 3 found', ctx, 'search')
  await vi.waitFor(() => expect(page).toHaveBeenCalled())
  expect(rpc).toHaveBeenCalledWith('session.search', { query: 'found', limit: 3, session_id: 'one' })
  expect(page.mock.calls[0]![0]).toContain('/resume one')
  expect(page.mock.calls[0]![0]).toContain('2 messages could not be indexed')
  findSlashCommand('search')!.run('found', ctx, 'search')
  await vi.waitFor(() => expect(error).toHaveBeenCalled())
})

it('opens full prior-turn output independently of the current tool record', () => {
  resetTurnState()
  patchTurnState({ toolRecords: { current: { name: 'ReadFile', result: 'current result' } } })
  const page = vi.fn()
  const ctx = { local: { getHistoryItems: () => [{ toolRecords: { old: { name: 'Exec', result: 'first\n' + 'x'.repeat(1000) + '\nlast' } } }] }, transcript: { page, sys: vi.fn() } } as never
  findSlashCommand('tool-output')!.run('old', ctx, 'tool-output')
  expect(page.mock.calls[0]![0]).toContain('first\n' + 'x'.repeat(1000) + '\nlast')
  expect(page.mock.calls[0]![0]).not.toContain('current result')
  resetTurnState()
})

it('previews numbered multiline text, reports truncation, and propagates read errors', async () => {
  const page = vi.fn(), error = vi.fn()
  const rpc = vi.fn().mockResolvedValueOnce({ ok: true, path: 'deep/file.txt', content: 'one\n\nthree', truncated: true }).mockRejectedValueOnce(new Error('outside workspace'))
  const ctx = { gateway: { rpc }, guarded: (fn: unknown) => fn, guardedErr: error, transcript: { page, sys: vi.fn() } } as never
  findSlashCommand('file')!.run('deep/file.txt', ctx, 'file')
  await vi.waitFor(() => expect(page).toHaveBeenCalled())
  expect(page.mock.calls[0]![0]).toContain('1  one\n2  \n3  three')
  expect(page.mock.calls[0]![0]).toContain('128 KiB')
  findSlashCommand('file')!.run('../outside', ctx, 'file')
  await vi.waitFor(() => expect(error).toHaveBeenCalled())
  expect(page).toHaveBeenCalledTimes(1)
})

it('requires confirmation, refuses a stale session, and reports partial recorded-edit failures', async () => {
  const page = vi.fn(), rpc = vi.fn().mockResolvedValue({ ok: false, results: [{ path: '/a', ok: true, reverted: 2 }, { path: '/b', ok: false, error: 'file changed' }] })
  let stale = false
  const ctx = { sid: 'session-a', stale: () => stale, gateway: { rpc }, guarded: (fn: unknown) => fn, guardedErr: vi.fn(), transcript: { page, sys: vi.fn() } } as never
  findSlashCommand('undo-edits')!.run('--all', ctx, 'undo-edits')
  expect(rpc).not.toHaveBeenCalled()
  const confirm = getOverlayState().confirm!
  stale = true
  confirm.onConfirm()
  expect(rpc).not.toHaveBeenCalled()
  stale = false
  confirm.onConfirm()
  await vi.waitFor(() => expect(page).toHaveBeenCalled())
  expect(rpc).toHaveBeenCalledWith('changes.undo', { session_id: 'session-a', path: '' })
  expect(page.mock.calls[0]![0]).toContain('/a: 2 edits reversed')
  expect(page.mock.calls[0]![0]).toContain('/b: file changed')
})
