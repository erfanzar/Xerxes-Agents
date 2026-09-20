// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { workspaceFileDiff } from '../src/ui/lib/workspaceDiffPreview.js'
const overview = { kind: 'ok', diff: { files: 1, lines: [{ kind: 'file', text: 'old.ts' }], insertions: 0, deletions: 0, truncated: true, untracked: ['new.ts'], untrackedTruncated: false } }

test('older daemons can preview new files through their bounded file API', async () => {
  const calls: string[] = []
  const result = await workspaceFileDiff(async (method, params) => {
    calls.push(method); expect(params.path).toBe('new.ts')
    return method === 'workspace.diff' ? overview : { content: 'const added = 1\n', truncated: false }
  }, 'new.ts', true)
  expect(calls).toEqual(['workspace.diff', 'workspace.filePreview'])
  expect(result.kind === 'ok' && result.diff.lines).toContainEqual({ kind: 'add', text: '+const added = 1', newLine: 1 })
  expect(result.kind === 'ok' && result.diff.truncated).toBe(false)
})

test('a missing tracked diff never turns the whole file into fabricated additions', async () => {
  const calls: string[] = []
  const result = await workspaceFileDiff(async method => { calls.push(method); return overview }, 'tracked.ts', false)
  expect(result.kind).toBe('error')
  expect(calls).toEqual(['workspace.diff'])
})

test('preview failures, truncation and cancellation stay observable', async () => {
  const call = async (method: string) => method === 'workspace.diff' ? overview : { ok: false, error: 'File is outside the workspace' }
  expect(await workspaceFileDiff(call, 'new.ts', true)).toEqual({ kind: 'error', message: 'File is outside the workspace' })
  await expect(workspaceFileDiff(async () => { throw new Error('cancelled') }, 'new.ts', true)).rejects.toThrow('cancelled')
  const result = await workspaceFileDiff(async method => method === 'workspace.diff' ? overview : { content: 'x'.repeat(600) + '\n', truncated: true }, 'new.ts', true)
  expect(result.kind === 'ok' && result.diff.truncated).toBe(true)
  expect(result.kind === 'ok' && result.diff.lines.at(-1)?.text.length).toBe(512)
})
