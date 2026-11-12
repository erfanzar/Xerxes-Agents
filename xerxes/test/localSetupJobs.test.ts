// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { createLocalSetupJobs, localStatusFromDoctor, type LocalCommandRunner, type LocalSetupJob } from '../src/daemon/localSetup.js'
import { NO_ENGINE_HELP } from '../src/local/command.js'

const llamaDoctor = {
  ok: true,
  engine: 'llama-server',
  ready: false,
  hardware: { platform: 'darwin', arch: 'arm64', total_mem_bytes: 24 * 2 ** 30, apple_silicon: true, gpu: { vendor: 'apple' } },
  selection: { ok: true, model: 'ternary-bonsai-2-27b' },
  llama_server: { found: true, path: '/opt/homebrew/bin/llama-server', build: 12110, version: '0.9.0', new_enough: true },
  model: { id: 'ternary-bonsai-2-27b', label: 'Ternary Bonsai 2 27B', present: false, verified: false, source_configured: true, bytes: 7_626_008_928 },
  server: { running: false, healthy: false, stale: false },
}

test('doctor output becomes one status, whichever engine it reported on', () => {
  expect(localStatusFromDoctor(llamaDoctor)).toEqual({
    ready: false,
    engine: { name: 'llama-server', label: 'llama.cpp', found: true, version: '0.9.0', newEnough: true },
    model: { id: 'ternary-bonsai-2-27b', label: 'Ternary Bonsai 2 27B', bytes: 7_626_008_928, downloaded: false },
    server: { running: false, port: null, contextLength: null },
    memoryBytes: 24 * 2 ** 30,
    blocker: null,
  })
  const mlx = localStatusFromDoctor({
    engine: 'mlx', ready: true, selection: { ok: true },
    mlx_server: { found: true, mlx_lm_version: '0.32.2', new_enough: true },
    model: { id: 'ternary-bonsai-2-27b', files_verified: 9, files_total: 9 },
    server: { running: true, port: 11581, context_length: 32768 },
  })
  expect(mlx).toMatchObject({ engine: { name: 'mlx', label: 'MLX', version: '0.32.2' }, model: { downloaded: true }, server: { running: true, port: 11581, contextLength: 32768 } })
})

test('a missing or outdated engine, or a machine too small, is what blocks setup', () => {
  expect(localStatusFromDoctor({ ...llamaDoctor, llama_server: { found: false } }).blocker).toBe(NO_ENGINE_HELP)
  expect(localStatusFromDoctor({ ...llamaDoctor, llama_server: { found: true, build: 9000, new_enough: false } }).blocker)
    .toContain('llama.cpp build 9000 is older than this model needs')
  expect(localStatusFromDoctor({ ...llamaDoctor, selection: { ok: false, reason: 'Needs 20 GiB of memory.' } }).blocker).toBe('Needs 20 GiB of memory.')
})

function harness(run: LocalCommandRunner, onSucceeded?: () => Promise<void>) {
  const published: LocalSetupJob[] = []
  let notify: () => void = () => {}
  const jobs = createLocalSetupJobs({
    run,
    publish: job => { published.push(job); notify() },
    ...(onSucceeded === undefined ? {} : { onSucceeded }),
    now: () => 1000,
  })
  const settled = () => new Promise<LocalSetupJob>(resolve => {
    const check = () => {
      const job = published.at(-1)
      if (job !== undefined && job.state !== 'running') resolve(job)
    }
    notify = check
    check()
  })
  return { jobs, published, settled }
}

test('setup progress is published line by line, and the result once the command succeeds', async () => {
  const applied: string[] = []
  const { jobs, published, settled } = harness(async (argv, io) => {
    expect(argv).toEqual(['setup', '--json'])
    io.stderr('Probing hardware…\nDownloading ternary-bonsai-2-27b: 45% (3.2 GiB / 7.1 GiB)\nDownload ver')
    io.stderr('ified (sha256 ok).\n')
    io.stdout(JSON.stringify({ ok: true, engine: 'llama-server', context_length: 32768 }) + '\n')
    return 0
  }, async () => { applied.push('mode') })
  const started = jobs.start()
  expect(started).toMatchObject({ ok: true, job: { state: 'running', step: 'Starting…' } })
  expect(jobs.start()).toMatchObject({ ok: false, code: 'busy' })
  const done = await settled()
  expect(done).toMatchObject({ state: 'done', result: { engine: 'llama-server', context_length: 32768 } })
  expect(applied).toEqual(['mode'])
  expect(published.filter(job => job.state === 'running').map(job => [job.step, job.state === 'running' ? job.percent : null])).toEqual([
    ['Starting…', null],
    ['Probing hardware…', null],
    ['Downloading ternary-bonsai-2-27b: 45% (3.2 GiB / 7.1 GiB)', 45],
    ['Download verified (sha256 ok).', 45],
  ])
  // A finished job can be followed by another.
  expect(jobs.start().ok).toBe(true)
})

test('a failed setup keeps the command\'s own error, and a failed mode change is reported as such', async () => {
  const failing = harness(async (_argv, io) => {
    io.stdout(JSON.stringify({ ok: false, error: { code: 'BINARY_NOT_FOUND', message: 'Install llama.cpp.' } }) + '\n')
    return 1
  })
  failing.jobs.start()
  expect(await failing.settled()).toMatchObject({ state: 'failed', error: { code: 'BINARY_NOT_FOUND', message: 'Install llama.cpp.' } })

  const unapplied = harness(async (_argv, io) => { io.stdout('{"ok":true}\n'); return 0 }, async () => { throw new Error('no window') })
  unapplied.jobs.start()
  expect(await unapplied.settled()).toMatchObject({ state: 'failed', error: { code: 'MODE_NOT_APPLIED', message: 'Bonsai is set up, but the mode was not applied: no window' } })
})

test('cancelling stops the command through its signal', async () => {
  const { jobs, settled } = harness((_argv, io) => new Promise((_resolve, reject) => {
    io.signal.addEventListener('abort', () => reject(new Error('aborted')))
  }))
  expect(jobs.cancel()).toBe(false)
  jobs.start()
  expect(jobs.cancel()).toBe(true)
  expect(await settled()).toMatchObject({ state: 'cancelled' })
})

test('status reports doctor\'s view and the last job', async () => {
  const { jobs } = harness(async (argv, io) => {
    expect(argv).toEqual(['doctor', '--json'])
    io.stdout(JSON.stringify(llamaDoctor) + '\n')
    return 1
  })
  expect(await jobs.status()).toMatchObject({ ok: true, status: { engine: { found: true } }, job: null })
  const broken = harness(async (_argv, io) => { io.stdout('not json\n'); return 1 })
  expect(await broken.jobs.status()).toMatchObject({ ok: false, code: 'BAD_OUTPUT' })
})
