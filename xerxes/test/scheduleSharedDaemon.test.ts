// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, realpath, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { CronJob, JobStore } from '../src/cron/jobs.js'
import { DaemonServer } from '../src/daemon/server.js'
import { InMemoryDaemonRuntime, type DaemonEvent } from '../src/daemon/runtime.js'
import { requestDaemonControl } from '../src/daemon/controlClient.js'

async function waitFor(predicate: () => boolean, timeoutMs = 3_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error('condition was not met in time')
    await Bun.sleep(10)
  }
}

test('schedule RPCs from a session-less CLI connection manage the project they name', async () => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'schedule-shared-daemon-')))
  const launch = join(directory, 'launch'), other = join(directory, 'other')
  await mkdir(launch); await mkdir(other)
  const store = new JobStore(join(directory, 'jobs.json'))
  store.add(new CronJob({ id: 'in-launch', prompt: 'a', paused: true, intervalSeconds: 60, projectRoot: launch }))
  store.add(new CronJob({ id: 'in-other', prompt: 'b', paused: true, intervalSeconds: 60, projectRoot: other }))
  const socketPath = join(directory, 'daemon.sock')
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: launch, sessionDirectory: join(directory, 'sessions') })
  const server = new DaemonServer({ runtime, socketPath, projectDirectory: launch,
    cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') })
  await server.start()
  try {
    const listed = await requestDaemonControl(socketPath, 'schedule.list', { expected_project_directory: other })
    expect(listed.ok).toBe(true)
    expect((listed.jobs as { id: string }[]).map(job => job.id)).toEqual(['in-other'])
    const inspected = await requestDaemonControl(socketPath, 'schedule.inspect', { schedule_id: 'in-launch', expected_project_directory: other })
    expect(inspected.ok).toBe(false)
    const own = await requestDaemonControl(socketPath, 'schedule.list', { expected_project_directory: launch })
    expect((own.jobs as { id: string }[]).map(job => job.id)).toEqual(['in-launch'])
    expect(await requestDaemonControl(socketPath, 'schedule.list', { expected_project_directory: 42 })).toMatchObject({ ok: false })
  } finally { await server.stop(); await rm(directory, { recursive: true, force: true }) }
})

test('a follow-up queued behind its conversation turn runs instead of timing out unrun', async () => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'schedule-followup-queue-')))
  const prompts: string[] = []
  const runtime = new InMemoryDaemonRuntime({
    async *run(_session, text): AsyncGenerator<DaemonEvent> {
      prompts.push(text)
      yield { type: 'text_part', payload: { text: 'follow-up done' } }
    },
  }, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') })
  const chat = await runtime.openSession('chat')
  const store = new JobStore(join(directory, 'jobs.json'))
  store.add(new CronJob({ id: 'followup', prompt: 'check the build', oneshot: true, maxRuns: 2, targetSessionId: chat.id,
    nextRunAt: new Date(Date.now() - 1_000).toISOString() }))
  const server = new DaemonServer({ runtime, socketPath: join(directory, 'daemon.sock'), projectDirectory: directory,
    cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive'),
    cronJobTimeout: 100, cronPollInterval: 10 })
  // The person's own long turn holds the conversation for longer than the
  // follow-up's whole timeout.
  const humanTurn = (server as unknown as { withSessionOperation(key: string, operation: () => Promise<void>): Promise<void> })
    .withSessionOperation('chat', () => Bun.sleep(400))
  await server.start()
  try {
    await humanTurn
    await waitFor(() => store.get('followup') === undefined || typeof store.get('followup')?.metadata.last_error === 'string')
    expect(store.get('followup')?.metadata.last_error).toBeUndefined()
    expect(prompts.some(prompt => prompt.includes('check the build'))).toBe(true)
  } finally { await server.stop(); await rm(directory, { recursive: true, force: true }) }
})
