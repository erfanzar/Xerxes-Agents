// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, spyOn, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime, type TurnRunner } from '../src/daemon/runtime.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'

async function fixture(runner?: TurnRunner) {
  const directory = await mkdtemp(join(tmpdir(), 'xr-permission-save-'))
  const options = { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions'), permissionMode: 'auto' }
  const store = new DaemonTranscriptStore({ directory: options.sessionDirectory, currentProjectDirectory: directory })
  const runtime = new InMemoryDaemonRuntime(runner, { ...options, transcriptStore: store })
  const session = await runtime.openSession('task')
  session.messages.push({role:'user',content:'Existing question'}, {role:'assistant',content:'Existing answer'})
  await runtime.flushSessions()
  return { runtime, session, store,
    async restored() {
      const restarted = new InMemoryDaemonRuntime(undefined, options)
      try { return (await restarted.openSession(session.id, undefined, {resume:true})).permissionMode ?? restarted.status().permission_mode }
      finally { await restarted.shutdown() }
    },
    async close() { await runtime.shutdown(); await rm(directory, { recursive:true, force:true }) },
  }
}

test('acknowledged manual permissions survive restart without a later flush', async () => {
  const f = await fixture()
  try {
    await f.runtime.setSessionPermissionMode('task', 'manual')
    expect(await f.restored()).toBe('manual')
  } finally { await f.close() }
})

test('failed permission save preserves the prior policy, pin, metadata and retry', async () => {
  const f = await fixture()
  try {
    await f.runtime.setSessionPermissionMode('task', 'manual')
    const before = structuredClone(f.session)
    const save = spyOn(f.store, 'save').mockRejectedValueOnce(new Error('private storage sentinel'))
    let error = ''
    try { await f.runtime.setSessionPermissionMode('task', 'accept-all') } catch (cause) { error = String(cause) }
    finally { save.mockRestore() }
    expect(f.session.permissionMode).toBe('manual')
    expect(f.session.permissionPinned).toBe(before.permissionPinned)
    expect(f.session.lastActive).toBe(before.lastActive)
    expect(f.session.metadata).toEqual(before.metadata)
    expect(error).toContain('unchanged')
    expect(error).not.toContain('private storage sentinel')
    await f.runtime.flushSessions()
    expect(await f.restored()).toBe('manual')
    await f.runtime.setSessionPermissionMode('task', 'plan')
    expect(await f.restored()).toBe('plan')
  } finally { await f.close() }
})

test('pending failed permission save cannot escape through another setting or flush', async () => {
  const f = await fixture(), started = Promise.withResolvers<void>(), finish = Promise.withResolvers<void>()
  let save: ReturnType<typeof spyOn> | undefined
  try {
    await f.runtime.setSessionPermissionMode('task', 'manual')
    save = spyOn(f.store, 'save').mockImplementationOnce(async () => { started.resolve(); await finish.promise; throw new Error('fixture failure') })
    const first = f.runtime.setSessionPermissionMode('task', 'accept-all').catch(error => String(error))
    // The pre-fix setter never called storage; avoid a hanging regression.
    await Promise.race([started.promise, first])
    const visible = f.session.permissionMode
    const next = f.runtime.setSessionPermissionMode('task', 'plan'), flush = f.runtime.flushSessions()
    finish.resolve()
    await Promise.allSettled([first, next, flush])
    save.mockRestore(); save = undefined
    expect(visible).toBe('manual')
    expect(await first).toContain('unchanged')
    expect(f.session.permissionMode).toBe('plan')
    expect(await f.restored()).toBe('plan')
    expect(JSON.stringify(f.session.metadata.context_deltas)).not.toContain('accept-all')
  } finally { finish.resolve(); save?.mockRestore(); await f.close() }
})

test('cancelling running work during a failed permission save retains the stricter policy', async () => {
  const running = Promise.withResolvers<void>(), saving = Promise.withResolvers<void>(), release = Promise.withResolvers<void>()
  let cancelled = false
  const f = await fixture({ async *run(_session, _text, signal) {
    running.resolve()
    await new Promise<void>(resolve => signal.addEventListener('abort', () => { cancelled = true; resolve() }, { once: true }))
    yield { type: 'text_part', payload: { text: 'Partial work retained.' } }
  } })
  let save: ReturnType<typeof spyOn> | undefined
  try {
    await f.runtime.setSessionPermissionMode('task', 'manual')
    const turn = f.runtime.submitTurn('task', 'Wait for cancellation.', () => {})
    await running.promise
    save = spyOn(f.store, 'save').mockImplementationOnce(async () => {
      saving.resolve(); await release.promise; throw new Error('private storage sentinel')
    })
    const selection = f.runtime.setSessionPermissionMode('task', 'accept-all').catch(error => String(error))
    await saving.promise
    expect(f.session.permissionMode).toBe('manual')
    expect(f.runtime.cancelTurn('task')).toBe(true)
    release.resolve()
    expect(await selection).toContain('unchanged')
    await turn
    save.mockRestore(); save = undefined
    expect(cancelled).toBe(true)
    expect(f.session.activeTurnId).toBe('')
    expect(await f.restored()).toBe('manual')
    expect(f.session.messages.some(message => message.content === 'Existing answer')).toBe(true)
  } finally { release.resolve(); save?.mockRestore(); f.runtime.cancelTurn('task'); await f.close() }
})
