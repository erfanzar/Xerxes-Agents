// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'

async function inRuntime(
  run: (runtime: InMemoryDaemonRuntime, directory: string) => Promise<void>,
  model = 'claude-code/default',
): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-session-model-'))
  try {
    await run(
      new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, model }),
      directory,
    )
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
}

test('two open sessions hold different models at the same time', async () => {
  await inRuntime(async runtime => {
    const first = await runtime.openSession('alpha')
    const second = await runtime.openSession('beta')

    await runtime.setSessionModel('alpha', 'codex/gpt-5.5')

    // The whole point of scoping: choosing in one session must not retarget
    // the other, which is what a daemon-wide model made unavoidable.
    expect(runtime.sessionStatus('alpha')?.model).toBe('codex/gpt-5.5')
    expect(runtime.sessionStatus('beta')?.model).toBe('claude-code/default')
    expect(first.id).not.toBe(second.id)
  })
})

test('a global reload leaves a session that picked its own model alone', async () => {
  await inRuntime(async runtime => {
    await runtime.openSession('pinned')
    await runtime.openSession('follower')
    await runtime.setSessionModel('pinned', 'codex/gpt-5.5')

    runtime.reload({ model: 'claude-sonnet-4-6' })

    // A reload is the daemon default moving, not an instruction to override a
    // session the user already aimed somewhere else.
    expect(runtime.sessionStatus('pinned')?.model).toBe('codex/gpt-5.5')
    expect(runtime.sessionStatus('follower')?.model).toBe('claude-sonnet-4-6')
  })
})

test('a session that never chose still follows the daemon default', async () => {
  await inRuntime(async runtime => {
    await runtime.openSession('drifter')

    runtime.reload({ model: 'claude-sonnet-4-6' })

    expect(runtime.sessionStatus('drifter')?.model).toBe('claude-sonnet-4-6')
  })
})

test('resuming history continues on the model that wrote it', async () => {
  await inRuntime(async (runtime, directory) => {
    const opened = await runtime.openSession('original')
    // An empty transcript is never persisted, so the session needs history for
    // there to be anything to resume.
    opened.messages.push({ role: 'user', content: 'hello' })
    await runtime.setSessionModel('original', 'codex/gpt-5.5')
    await runtime.flushSessions()

    // A fresh daemon whose default is a different provider entirely.
    const restarted = new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: 'claude-sonnet-4-6',
    })
    const resumed = await restarted.openSession(opened.id, undefined, { resume: true })

    // Resuming must not silently move a conversation onto another provider:
    // the transcript was produced by this model and continues on it.
    expect(resumed.model).toBe('codex/gpt-5.5')

    // And it stays put when the daemon default moves again.
    restarted.reload({ model: 'claude-opus-4-6' })
    expect(restarted.sessionStatus(opened.id)?.model).toBe('codex/gpt-5.5')
  })
})

test('an explicit model on reopen pins the session', async () => {
  await inRuntime(async runtime => {
    await runtime.openSession('reopened')
    await runtime.openSession('reopened', undefined, { model: 'codex/gpt-5.4' })

    runtime.reload({ model: 'claude-sonnet-4-6' })

    expect(runtime.sessionStatus('reopened')?.model).toBe('codex/gpt-5.4')
  })
})

test('setting an empty model leaves the session untouched', async () => {
  await inRuntime(async runtime => {
    await runtime.openSession('blank')
    await runtime.setSessionModel('blank', '   ')

    expect(runtime.sessionStatus('blank')?.model).toBe('claude-code/default')
  })
})
