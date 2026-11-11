// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { chmod, mkdir, mkdtemp, realpath, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { createBridge } from '../src/desktop/bridgeApi.js'
import { FILE_ACTION_CHANNEL, asFileActionResult, fileActionRequestOf } from '../src/desktop/fileActionShared.js'
import { isLaunchable, performFileAction } from '../src/desktop/main/fileActions.js'

function recorder(failure = '') {
  const calls: string[] = []
  return { calls, port: { openPath: async (path: string) => { calls.push(`open ${path}`); return failure }, showItemInFolder: (path: string) => { calls.push(`reveal ${path}`) } } }
}

test('open and reveal resolve inside the task folder, symlinks included', async () => {
  const root = await mkdtemp(join(tmpdir(), 'file-actions-'))
  const workspace = join(root, 'workspace')
  try {
    await mkdir(join(workspace, 'out'), { recursive: true })
    await Bun.write(join(workspace, 'out', 'report.xlsx'), 'x')
    await Bun.write(join(root, 'private.pdf'), 'x')
    await symlink(join(root, 'private.pdf'), join(workspace, 'link.pdf'))
    const { calls, port } = recorder()
    expect(await performFileAction({ action: 'open', path: 'out/report.xlsx', cwd: workspace }, port)).toEqual({ ok: true })
    expect(await performFileAction({ action: 'reveal', path: './out/report.xlsx', cwd: workspace }, port)).toEqual({ ok: true })
    const real = await realpath(join(workspace, 'out', 'report.xlsx'))
    expect(calls).toEqual([`open ${real}`, `reveal ${real}`])
    for (const path of ['../private.pdf', 'link.pdf', join(root, 'private.pdf')]) {
      expect(await performFileAction({ action: 'open', path, cwd: workspace }, port)).toMatchObject({ ok: false, error: expect.stringContaining('inside this task') })
    }
    expect(await performFileAction({ action: 'open', path: 'out', cwd: workspace }, port)).toMatchObject({ ok: false, error: expect.stringContaining('Only files') })
    expect(await performFileAction({ action: 'open', path: 'gone.pdf', cwd: workspace }, port)).toMatchObject({ ok: false, error: expect.stringContaining('no longer exists') })
    expect(calls.length).toBe(2)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('a file that would run as a program is never opened, only shown', async () => {
  const workspace = await mkdtemp(join(tmpdir(), 'file-actions-launch-'))
  try {
    await Bun.write(join(workspace, 'report.command'), 'echo hi')
    await Bun.write(join(workspace, 'run'), '#!/bin/sh\n')
    await chmod(join(workspace, 'run'), 0o755)
    // Copied from an exFAT drive: executable, but a document type opens by type.
    await Bun.write(join(workspace, 'summary.pdf'), '%PDF-')
    await chmod(join(workspace, 'summary.pdf'), 0o755)
    const { calls, port } = recorder()
    for (const path of ['report.command', 'run']) {
      expect(await performFileAction({ action: 'open', path, cwd: workspace }, port)).toMatchObject({ ok: false, error: expect.stringContaining('run as a program') })
      expect(await performFileAction({ action: 'reveal', path, cwd: workspace }, port)).toEqual({ ok: true })
    }
    expect(await performFileAction({ action: 'open', path: 'summary.pdf', cwd: workspace }, port)).toEqual({ ok: true })
    expect(calls.filter(call => call.startsWith('open')).length).toBe(1)
    expect(isLaunchable('/a/Setup.PKG', 0o644)).toBe(true)
    expect(isLaunchable('/a/notes.txt', 0o644)).toBe(false)
  } finally { await rm(workspace, { recursive: true, force: true }) }
})

test('a host failure to open is reported, not swallowed', async () => {
  const workspace = await mkdtemp(join(tmpdir(), 'file-actions-fail-'))
  try {
    await Bun.write(join(workspace, 'a.docx'), 'x')
    expect(await performFileAction({ action: 'open', path: 'a.docx', cwd: workspace }, recorder('No application knows how to open it').port))
      .toEqual({ ok: false, error: 'The file could not be opened: No application knows how to open it' })
  } finally { await rm(workspace, { recursive: true, force: true }) }
})

test('requests are validated on both sides of the bridge, and the bridge never rejects', async () => {
  expect(fileActionRequestOf({ action: 'open', path: 'a.pdf', cwd: '/w' })).toEqual({ action: 'open', path: 'a.pdf', cwd: '/w' })
  expect(fileActionRequestOf({ action: 'delete', path: 'a.pdf', cwd: '/w' })).toMatchObject({ error: expect.any(String) })
  expect(fileActionRequestOf({ action: 'open', path: 'a\npdf', cwd: '/w' })).toMatchObject({ error: expect.stringContaining('path') })
  expect(fileActionRequestOf({ action: 'open', path: 'a.pdf', cwd: 'relative' })).toMatchObject({ error: expect.stringContaining('absolute') })
  expect(asFileActionResult({ ok: true, extra: 1 })).toEqual({ ok: true })
  expect(asFileActionResult(null)).toMatchObject({ ok: false })
  const sent: unknown[][] = []
  const bridge = createBridge({
    invoke: async (channel, ...args) => { sent.push([channel, ...args]); if (args[0] && (args[0] as { path: string }).path === 'boom.pdf') throw new Error("Error invoking remote method 'x': Error: host died"); return { ok: true } },
    on: () => () => {},
    setZoomFactor: () => {},
  })
  expect(await bridge.fileAction('reveal', 'a.pdf', '/w')).toEqual({ ok: true })
  expect(sent).toEqual([[FILE_ACTION_CHANNEL, { action: 'reveal', path: 'a.pdf', cwd: '/w' }]])
  expect(await bridge.fileAction('open', 'boom.pdf', '/w')).toEqual({ ok: false, error: 'host died' })
  expect(await bridge.fileAction('open', '', '/w')).toMatchObject({ ok: false })
  expect(sent.length).toBe(2)
})
