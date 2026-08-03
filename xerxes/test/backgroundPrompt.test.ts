// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'

async function inRuntime(
  run: (runtime: InMemoryDaemonRuntime) => Promise<void>,
): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bg-'))
  try {
    await run(new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: 'claude-code/default',
    }))
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
}

test('a background session is distinct from the foreground one', async () => {
  await inRuntime(async runtime => {
    const foreground = await runtime.openSession('main')
    const background = await runtime.openSession('bg-abc123')

    // The point of backgrounding: the foreground session keeps its own
    // identity and history while the background prompt runs elsewhere.
    expect(background.id).not.toBe(foreground.id)
    expect(runtime.sessionStatus('main')?.id).toBe(foreground.id)
    expect(runtime.sessionStatus('bg-abc123')?.id).toBe(background.id)
  })
})

test('a background session inherits the model the user is working with', async () => {
  await inRuntime(async runtime => {
    await runtime.openSession('main')
    await runtime.setSessionModel('main', 'codex/gpt-5.5')
    const parent = runtime.sessionStatus('main')

    const background = await runtime.openSession('bg-inherit', undefined, {
      ...(parent?.model ? { model: parent.model } : {}),
    })

    // Answering a background prompt on the daemon default rather than the
    // model the user chose would silently bill a different provider.
    expect(background.model).toBe('codex/gpt-5.5')
  })
})

test('a background session keeps its inherited model when the default moves', async () => {
  await inRuntime(async runtime => {
    await runtime.openSession('main')
    await runtime.setSessionModel('main', 'codex/gpt-5.5')
    await runtime.openSession('bg-pinned', undefined, { model: 'codex/gpt-5.5' })

    runtime.reload({ model: 'claude-sonnet-4-6' })

    expect(runtime.sessionStatus('bg-pinned')?.model).toBe('codex/gpt-5.5')
    expect(runtime.sessionStatus('main')?.model).toBe('codex/gpt-5.5')
  })
})
