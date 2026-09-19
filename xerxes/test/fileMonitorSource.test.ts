// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, symlink, unlink, writeFile, rm, rename } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { nativeFileMonitorSource } from '../src/runtime/fileMonitorSource.js'

async function waitForError(errors: unknown[], text: string): Promise<void> {
  const deadline = Date.now() + 2_000
  while (!errors.some(error => String(error).includes(text))) {
    if (Date.now() >= deadline) throw new Error('Expected file monitor error: ' + text)
    await Bun.sleep(10)
  }
}

async function fixture(): Promise<{ root: string; file: string; close: () => Promise<void> }> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-file-monitor-'))
  const file = join(root, 'watched.txt')
  await writeFile(file, 'a')
  return { root, file, close: () => rm(root, { recursive: true, force: true }) }
}

function nextEvent(events: { text: string; identity: string }[], timeoutMs = 1_000): Promise<{ text: string; identity: string }> {
  return new Promise((resolve, reject) => {
    let done = false
    const timer = setTimeout(() => { done = true; reject(new Error('timed out waiting for file event')) }, timeoutMs)
    const check = (): void => {
      if (done) return
      const event = events.shift()
      if (event) { done = true; clearTimeout(timer); resolve(event); return }
      setTimeout(check, 5)
    }
    check()
  })
}

test('file monitor reports changes, atomic replacement, deletion, and recreation without initial event', async () => {
  const f = await fixture()
  const events: { text: string; identity: string }[] = []
  const monitor = await nativeFileMonitorSource.open(f.root, 'watched.txt', event => events.push(event), error => { throw error })
  try {
    expect(events).toEqual([])
    await writeFile(f.file, 'b')
    const changed = await nextEvent(events)
    expect(JSON.parse(changed.text)).toMatchObject({ event: 'changed', path: 'watched.txt' })
    const replacement = join(f.root, 'replacement.txt')
    await writeFile(replacement, 'c')
    await rename(replacement, f.file)
    expect(JSON.parse((await nextEvent(events)).text)).toMatchObject({ event: 'changed', path: 'watched.txt' })
    await unlink(f.file)
    const deleted = await nextEvent(events)
    expect(JSON.parse(deleted.text)).toMatchObject({ event: 'deleted', path: 'watched.txt' })
    expect(deleted.identity).not.toBe(changed.identity)
    await writeFile(f.file, 'd')
    expect(JSON.parse((await nextEvent(events)).text)).toMatchObject({ event: 'recreated', path: 'watched.txt' })
  } finally { monitor.close(); await f.close() }
})

test('file monitor rejects lexical traversal and closes when parent directory is replaced', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-file-monitor-parent-'))
  const nested = join(root, 'nested')
  await mkdir(nested)
  const file = join(nested, 'watched.txt')
  await writeFile(file, 'a')
  const errors: unknown[] = []
  try {
    await expect(nativeFileMonitorSource.open(root, '../watched.txt', () => undefined, error => errors.push(error))).rejects.toThrow(/traversal|inside/)
    const monitor = await nativeFileMonitorSource.open(root, 'nested/watched.txt', () => undefined, error => errors.push(error))
    await rename(nested, join(root, 'old-nested'))
    await mkdir(nested)
    await writeFile(join(nested, 'watched.txt'), 'b')
    await waitForError(errors, 'parent directory was replaced')
    monitor.close()
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('file monitor rejects a symlink swap outside the workspace after opening', async () => {
  const f = await fixture()
  const outside = await mkdtemp(join(tmpdir(), 'xerxes-file-monitor-swap-'))
  const errors: unknown[] = []
  const events: { text: string; identity: string }[] = []
  try {
    const monitor = await nativeFileMonitorSource.open(f.root, f.file, event => events.push(event), error => errors.push(error))
    await unlink(f.file)
    await nextEvent(events)
    await symlink(join(outside, 'secret.txt'), f.file)
    await writeFile(join(outside, 'secret.txt'), 'secret')
    await waitForError(errors, 'outside')
    monitor.close()
  } finally { await f.close(); await rm(outside, { recursive: true, force: true }) }
})

test('file monitor allows A to B to A transitions', async () => {
  const f = await fixture()
  const events: { text: string; identity: string }[] = []
  const monitor = await nativeFileMonitorSource.open(f.root, f.file, event => events.push(event), error => { throw error })
  try {
    await writeFile(f.file, 'b'); const first = await nextEvent(events)
    await writeFile(f.file, 'a'); const second = await nextEvent(events)
    expect(first.identity).not.toBe(second.identity)
  } finally { monitor.close(); await f.close() }
})

test('file monitor resolves an internal symlink once and watches its canonical target', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-file-monitor-link-'))
  const targetDir = join(root, 'targets')
  const target = join(targetDir, 'watched.txt')
  const link = join(root, 'watched.txt')
  await mkdir(targetDir)
  await writeFile(target, 'a')
  await symlink(target, link)
  const events: { text: string; identity: string }[] = []
  try {
    const monitor = await nativeFileMonitorSource.open(root, link, event => events.push(event), error => { throw error })
    expect(monitor.path.endsWith('/targets/watched.txt')).toBe(true)
    await writeFile(target, 'b')
    expect(JSON.parse((await nextEvent(events)).text)).toMatchObject({ event: 'changed', path: 'targets/watched.txt' })
    monitor.close()
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('file monitor closes on abort, rejects symlink escapes, and reports watcher errors through cleanup', async () => {
  const f = await fixture()
  const outside = await mkdtemp(join(tmpdir(), 'xerxes-file-monitor-outside-'))
  try {
    const escape = join(f.root, 'escape.txt')
    await symlink(join(outside, 'secret.txt'), escape)
    await writeFile(join(outside, 'secret.txt'), 'secret')
    await expect(nativeFileMonitorSource.open(f.root, escape, () => undefined, () => undefined)).rejects.toThrow(/outside|regular/)
    const controller = new AbortController()
    const events: { text: string; identity: string }[] = []
    const monitor = await nativeFileMonitorSource.open(f.root, f.file, event => events.push(event), () => { throw new Error('unexpected watcher error') }, controller.signal)
    controller.abort()
    await writeFile(f.file, 'after abort')
    await new Promise(resolve => setTimeout(resolve, 150))
    expect(events).toEqual([])
    monitor.close()
  } finally { await f.close(); await rm(outside, { recursive: true, force: true }) }
})

test('file monitor rejects a signal already aborted before asynchronous setup', async () => {
  const f = await fixture()
  const controller = new AbortController()
  controller.abort(new Error('setup cancelled'))
  try {
    await expect(nativeFileMonitorSource.open(f.root, f.file, () => undefined, () => undefined, controller.signal)).rejects.toThrow('setup cancelled')
  } finally { await f.close() }
})
