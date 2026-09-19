// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, spyOn, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime, type DaemonSession, type TurnRunner } from '../src/daemon/runtime.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'

type Setting = 'model' | 'mode'
const state = (session: DaemonSession) => ({ model: session.model, modelPinned: session.modelPinned,
  mode: session.interactionMode, planMode: session.planMode, lastActive: session.lastActive,
  metadata: structuredClone(session.metadata) })

async function fixture(runner?: TurnRunner) {
  const directory = await mkdtemp(join(tmpdir(), 'xr-setting-persistence-'))
  const sessionDirectory = join(directory, 'sessions')
  const store = new DaemonTranscriptStore({ directory: sessionDirectory, currentProjectDirectory: directory })
  const changes: string[] = []
  const runtime = new InMemoryDaemonRuntime(runner, { model: 'initial-model', currentProjectDirectory: directory,
    transcriptStore: store, onSessionModeChange: (_id, mode) => { changes.push(mode) } })
  const session = await runtime.openSession('task')
  session.messages.push({ role: 'user', content: 'Existing task' }, { role: 'assistant', content: 'Existing reply' })
  session.metadata.provider_profile = 'initial-profile'
  await runtime.flushSessions()
  return { runtime, session, store, changes,
    select: (field: Setting, second = false) => field === 'model'
      ? runtime.setSessionModel('task', second ? 'retry-model' : 'rejected-model', second ? 'retry-profile' : 'rejected-profile')
      : runtime.setSessionMode('task', second ? 'researcher' : 'plan'),
    async resumed() {
      const restarted = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory })
      try { return state(await restarted.openSession(session.id, undefined, { resume: true })) }
      finally { await restarted.shutdown() }
    },
    async close() { await runtime.shutdown(); await rm(directory, { recursive: true, force: true }) },
  }
}

for (const field of ['model', 'mode'] as const) {
  test(`failed ${field} save preserves the live choice, pins and deltas; retry survives restart`, async () => {
    const f = await fixture()
    try {
      const before = state(f.session)
      const save = spyOn(f.store, 'save').mockRejectedValueOnce(new Error('private storage sentinel'))
      try { await expect(f.select(field)).rejects.toThrow('unchanged') }
      finally { save.mockRestore() }
      expect(state(f.session)).toEqual(before)
      expect(f.changes).toEqual([])
      await f.runtime.flushSessions()
      const old = await f.resumed()
      expect(old.model).toBe(before.model)
      expect(old.mode).toBe(before.mode)
      expect(old.planMode).toBe(before.planMode)
      expect(old.metadata.provider_profile).toBe('initial-profile')
      await f.select(field, true)
      const next = await f.resumed()
      expect(next.model).toBe(field === 'model' ? 'retry-model' : 'initial-model')
      expect(next.mode).toBe(field === 'mode' ? 'researcher' : 'code')
      expect(next.planMode).toBe(false)
      expect(next.metadata.provider_profile).toBe(field === 'model' ? 'retry-profile' : 'initial-profile')
      expect(f.changes).toEqual(field === 'mode' ? ['researcher'] : [])
    } finally { await f.close() }
  })

  test(`pending failed ${field} cannot escape through another setting or a concurrent flush`, async () => {
    const f = await fixture(), started = Promise.withResolvers<void>(), finish = Promise.withResolvers<void>()
    let save: ReturnType<typeof spyOn> | undefined
    try {
      const before = state(f.session)
      save = spyOn(f.store, 'save').mockImplementationOnce(async () => {
        started.resolve(); await finish.promise; throw new Error('private pending storage sentinel')
      })
      const first = f.select(field).catch(error => String(error))
      await started.promise
      const second = f.select(field, true), effort = f.runtime.setSessionReasoning('task', 'low'), flush = f.runtime.flushSessions()
      const pending = state(f.session)
      finish.resolve()
      const error = await first
      await second; await effort; await flush
      expect(error).toContain('unchanged')
      expect(error).not.toContain('private pending storage sentinel')
      save.mockRestore(); save = undefined
      expect(pending).toEqual(before)
      expect(JSON.stringify(f.session.metadata.context_deltas)).not.toContain(field === 'model' ? 'rejected-model' : 'plan')
      const resumed = await f.resumed()
      expect(resumed.model).toBe(field === 'model' ? 'retry-model' : 'initial-model')
      expect(resumed.mode).toBe(field === 'mode' ? 'researcher' : 'code')
      expect(resumed.metadata.reasoning_effort).toBe('low')
      expect(f.changes).toEqual(field === 'mode' ? ['researcher'] : [])
    } finally { finish.resolve(); save?.mockRestore(); await f.close() }
  })

  test(`cancellation while ${field} storage is pending retains the working choice and transcript`, async () => {
    const running = Promise.withResolvers<void>(), saving = Promise.withResolvers<void>(), release = Promise.withResolvers<void>()
    let cancelled = false
    const f = await fixture({ async *run(_session, _text, signal) {
      running.resolve()
      await new Promise<void>(resolve => signal.addEventListener('abort', () => { cancelled = true; resolve() }, { once: true }))
      yield { type: 'text_part', payload: { text: 'Partial work retained.' } }
    } })
    let save: ReturnType<typeof spyOn> | undefined
    try {
      const turn = f.runtime.submitTurn('task', 'Wait for cancellation.', () => {})
      await running.promise
      save = spyOn(f.store, 'save').mockImplementationOnce(async () => {
        saving.resolve(); await release.promise; throw new Error('isolated cancellation storage failure')
      })
      const selection = f.select(field).catch(error => String(error))
      await saving.promise
      expect(f.session.model).toBe('initial-model')
      expect(f.session.interactionMode).toBe('code')
      expect(f.runtime.cancelTurn('task')).toBe(true)
      release.resolve()
      const error = await selection
      await turn
      save.mockRestore(); save = undefined
      expect(cancelled).toBe(true)
      expect(error).toContain('unchanged')
      expect(f.session.activeTurnId).toBe('')
      const restored = await f.resumed()
      expect(restored.model).toBe('initial-model')
      expect(restored.mode).toBe('code')
      expect(f.session.messages.some(message => message.content === 'Existing reply')).toBe(true)
      expect(f.changes).toEqual([])
    } finally { release.resolve(); save?.mockRestore(); f.runtime.cancelTurn('task'); await f.close() }
  })
}
