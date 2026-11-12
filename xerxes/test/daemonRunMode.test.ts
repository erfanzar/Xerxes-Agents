// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { connect } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ProfileStore } from '../src/bridge/profiles.js'
import type { LocalCommandRunner } from '../src/daemon/localSetup.js'
import { runtimeSettingsFile } from '../src/daemon/runMode.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'

type Call = (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>

async function client(socketPath: string): Promise<{ call: Call; events: Array<{ type: string; payload: Record<string, unknown> }>; close: () => void }> {
  const socket = connect({ path: socketPath })
  await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) })
  socket.setEncoding('utf8')
  let buffer = '', id = 0
  const pending = new Map<number, (result: Record<string, unknown>) => void>()
  const events: Array<{ type: string; payload: Record<string, unknown> }> = []
  socket.on('data', chunk => {
    buffer += String(chunk)
    for (let newline = buffer.indexOf('\n'); newline >= 0; newline = buffer.indexOf('\n')) {
      const frame = JSON.parse(buffer.slice(0, newline)) as { id?: number; result?: Record<string, unknown>; error?: unknown; method?: string; params?: { type?: string; payload?: Record<string, unknown> } }
      buffer = buffer.slice(newline + 1)
      if (typeof frame.id === 'number') pending.get(frame.id)?.(frame.result ?? { error: frame.error })
      else if (frame.method === 'event' && typeof frame.params?.type === 'string') events.push({ type: frame.params.type, payload: frame.params.payload ?? {} })
    }
  })
  return {
    call: (method, params) => new Promise(resolve => { id += 1; pending.set(id, resolve); socket.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n') }),
    events,
    close: () => socket.destroy(),
  }
}

const bonsai = { name: 'bonsai-local', provider: 'custom', base_url: 'http://127.0.0.1:11581/v1', api_key: '', model: 'ternary-bonsai-2-27b', data_trust: 'local', model_capabilities: {}, sampling: {} }
const openrouter = { name: 'openrouter', provider: 'openrouter', base_url: 'https://openrouter.ai/api/v1', api_key: 'or-key', model: 'deepseek/deepseek-v4.1-flash', model_capabilities: {}, sampling: {} }

async function withDaemon(
  profiles: Record<string, unknown> | undefined,
  options: { localSetup?: LocalCommandRunner; loaded?: Record<string, unknown> },
  body: (call: Call, context: { configPath: string; store: ProfileStore; events: Array<{ type: string; payload: Record<string, unknown> }> }) => Promise<void>,
): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-run-mode-daemon-'))
  try {
    const profilesFile = join(root, 'profiles.json')
    if (profiles) await writeFile(profilesFile, JSON.stringify({ profiles }))
    const store = new ProfileStore(profilesFile)
    const configPath = join(root, 'daemon', 'config.json')
    const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
    const socketPath = join(root, 'daemon.sock')
    const server = new DaemonServer({
      profileStore: store, runtime, socketPath,
      modeSettings: { file: runtimeSettingsFile(configPath), loaded: options.loaded ?? {}, env: {} },
      ...(options.localSetup === undefined ? {} : { localSetup: options.localSetup }),
    })
    await server.start()
    const connection = await client(socketPath)
    try {
      await connection.call('initialize', { session_key: 'desktop-task' })
      await body(connection.call, { configPath, store, events: connection.events })
    } finally { connection.close(); await server.stop() }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

test('a fresh install has no mode, and the built-in Claude Code fallback is not reported as chosen', async () => {
  await withDaemon({ 'bonsai-local': bonsai }, {}, async call => {
    const listed = await call('provider_list', {})
    expect(listed.active_chosen).toBe(false)
    const rows = listed.profiles as Array<{ name: string; local: boolean }>
    expect(rows.find(row => row.name === 'bonsai-local')?.local).toBe(true)
    expect(rows.find(row => row.name === 'claude-code')?.local).toBe(false)
    expect(await call('run_mode.get', {})).toMatchObject({ ok: true, mode: null, worker: null, restart_pending: false })
  })
})

test('Hybrid needs Bonsai and a cloud lead; then the lead runs the task, the boundary is saved and Bonsai takes the subagent tiers', async () => {
  await withDaemon({ openrouter }, {}, async call => {
    expect(await call('run_mode.set', { mode: 'hybrid' })).toMatchObject({ ok: false, code: 'needs_local_setup' })
  })
  await withDaemon({ 'bonsai-local': bonsai, openrouter }, {}, async (call, { configPath, store }) => {
    store.setActive('bonsai-local')
    // Bonsai is chosen, so no cloud model is there to lead yet.
    expect(await call('run_mode.set', { mode: 'hybrid' })).toMatchObject({ ok: false, code: 'needs_cloud_profile' })
    expect(await call('run_mode.set', { mode: 'hybrid', worker_profile: 'bonsai-local' })).toMatchObject({ ok: false, code: 'wrong_location' })
    const set = await call('run_mode.set', { mode: 'hybrid', worker_profile: 'openrouter' })
    expect(set).toMatchObject({ ok: true, mode: 'hybrid', restart_required: true, worker: { profile: 'openrouter', local: false } })
    expect(store.chosenActive()).toBe('openrouter')
    // The task the request came from moves to the lead, as with provider_select.
    expect(((await call('session.status', {})).session as Record<string, unknown>).model).toBe('deepseek/deepseek-v4.1-flash')
    expect(JSON.parse(await readFile(configPath, 'utf8'))).toEqual({ runtime: { data_boundary: 'hybrid', boundary: { private_paths: ['**'], cloud_results: 'summary', approve_outbound: 'off' } } })
    const tiers = (await call('agent.settings.get', {})).settings as Record<string, unknown>
    expect(tiers).toMatchObject({
      default: 'balanced',
      balanced: { model: 'ternary-bonsai-2-27b', provider_profile: 'bonsai-local', reasoning_effort: 'off' },
      smart: { model: 'deepseek/deepseek-v4.1-flash', provider_profile: 'openrouter' },
    })
    expect((await call('provider_list', {})).active_chosen).toBe(true)
  })
})

test('a cloud lead that reports no model is refused with the reason, and nothing is saved', async () => {
  // A cloud profile whose endpoint refuses connections: it reports no models.
  const unreachable = { name: 'unreachable', provider: 'openai', base_url: 'http://127.0.0.1:9/v1', api_key: 'k', model: '', data_trust: 'cloud', model_capabilities: {}, sampling: {} }
  await withDaemon({ 'bonsai-local': bonsai, unreachable }, {}, async (call, { configPath, store }) => {
    const refused = await call('run_mode.set', { mode: 'hybrid', worker_profile: 'unreachable' })
    expect(refused).toMatchObject({ ok: false, code: 'worker_not_selected' })
    expect(String(refused.error)).toContain('Check its sign-in')
    expect(store.get('unreachable')?.model).toBe('')
    expect(await readFile(configPath, 'utf8').catch(() => 'absent')).toBe('absent')
  })
})

test('Private turns fully local mode on; Cloud needs a cloud model and leaves every tier on it', async () => {
  await withDaemon({ 'bonsai-local': bonsai, openrouter }, { loaded: { local_only: true } }, async (call, { configPath, store }) => {
    store.setActive('bonsai-local')
    expect(await call('run_mode.set', { mode: 'private' })).toMatchObject({ ok: true, mode: 'private', local_only: true, restart_required: false })
    expect((await call('agent.settings.get', {})).settings).toMatchObject({ smart: { provider_profile: 'bonsai-local' } })
    expect(await call('run_mode.set', { mode: 'cloud' })).toMatchObject({ ok: false, code: 'needs_cloud_profile' })
    expect(await call('run_mode.set', { mode: 'cloud', worker_profile: 'bonsai-local' })).toMatchObject({ ok: false, code: 'wrong_location' })
    expect(await call('run_mode.set', { mode: 'cloud', worker_profile: 'openrouter' })).toMatchObject({ ok: true, mode: 'cloud', local_only: false, restart_required: true })
    expect(JSON.parse(await readFile(configPath, 'utf8'))).toEqual({ runtime: {} })
    expect((await call('agent.settings.get', {})).settings).toEqual({})
    expect(await call('run_mode.set', { mode: 'remote' })).toMatchObject({ ok: false, code: 'invalid_request' })
  })
})

test('a local setup started from the app reports progress and applies the chosen mode when it succeeds', async () => {
  const run: LocalCommandRunner = async (argv, io) => {
    if (argv[0] === 'doctor') {
      io.stdout(JSON.stringify({ ok: true, engine: 'llama-server', ready: false, selection: { ok: true }, llama_server: { found: true, version: '0.9.0', new_enough: true }, model: { id: 'ternary-bonsai-2-27b', present: false, verified: false }, server: { running: false } }) + '\n')
      return 1
    }
    io.stderr('Downloading ternary-bonsai-2-27b: 50% (3.5 GiB / 7.1 GiB)\n')
    // What `xerxes local setup` saves before it prints its result.
    new ProfileStore(store!.filePath).save({ name: 'bonsai-local', baseUrl: bonsai.base_url, apiKey: '', model: bonsai.model, provider: 'custom', dataTrust: 'local', setActive: false })
    io.stdout(JSON.stringify({ ok: true, engine: 'llama-server', profile: 'bonsai-local', context_length: 32768 }) + '\n')
    return 0
  }
  let store: ProfileStore | undefined
  await withDaemon(undefined, { localSetup: run }, async (call, context) => {
    store = context.store
    expect(await call('local.status', {})).toMatchObject({ ok: true, status: { engine: { name: 'llama-server', found: true }, model: { downloaded: false } }, job: null })
    expect(await call('local.setup', { then_mode: 'cloud' })).toMatchObject({ ok: false, code: 'invalid_request' })
    expect(await call('local.setup', { then_mode: 'private' })).toMatchObject({ ok: true, job: { state: 'running' } })
    const deadline = Date.now() + 5000
    let job: Record<string, unknown> | undefined
    while (Date.now() < deadline) {
      job = (await call('local.status', {})).job as Record<string, unknown> | undefined
      if (job?.state !== 'running') break
      await Bun.sleep(10)
    }
    expect(job).toMatchObject({ state: 'done', result: { context_length: 32768 } })
    const steps = context.events.filter(event => event.type === 'local_setup').map(event => (event.payload.job as Record<string, unknown>).step)
    expect(steps).toContain('Downloading ternary-bonsai-2-27b: 50% (3.5 GiB / 7.1 GiB)')
    expect(await call('run_mode.get', {})).toMatchObject({ mode: 'private', worker: { profile: 'bonsai-local' } })
  })
})

test('a cloud lead picked for Hybrid before Bonsai was set up is applied when setup succeeds', async () => {
  let store: ProfileStore | undefined
  const run: LocalCommandRunner = async (argv, io) => {
    if (argv[0] === 'doctor') {
      io.stdout(JSON.stringify({ ok: true, engine: 'llama-server', ready: false, selection: { ok: true }, llama_server: { found: true, version: '0.9.0', new_enough: true }, model: { id: 'ternary-bonsai-2-27b', present: true, verified: true }, server: { running: false } }) + '\n')
      return 1
    }
    new ProfileStore(store!.filePath).save({ name: 'bonsai-local', baseUrl: bonsai.base_url, apiKey: '', model: bonsai.model, provider: 'custom', dataTrust: 'local', setActive: false })
    io.stdout(JSON.stringify({ ok: true, engine: 'llama-server', profile: 'bonsai-local', context_length: 32768 }) + '\n')
    return 0
  }
  await withDaemon({ openrouter }, { localSetup: run }, async (call, context) => {
    store = context.store
    // Picking the lead first is not an error: it waits for Bonsai.
    expect(await call('run_mode.set', { mode: 'hybrid', worker_profile: 'openrouter' })).toMatchObject({ ok: false, code: 'needs_local_setup' })
    // A lead only goes with Hybrid, must exist and must be a cloud provider.
    expect(await call('local.setup', { then_mode: 'private', then_worker_profile: 'openrouter' })).toMatchObject({ ok: false, code: 'invalid_request' })
    expect(await call('local.setup', { then_mode: 'hybrid', then_worker_profile: 42 })).toMatchObject({ ok: false, code: 'invalid_request' })
    expect(await call('local.setup', { then_mode: 'hybrid', then_worker_profile: 'nobody' })).toMatchObject({ ok: false, code: 'unknown_profile' })
    expect(await call('local.setup', { then_mode: 'hybrid', then_worker_profile: 'openrouter' })).toMatchObject({ ok: true, job: { state: 'running' } })
    const deadline = Date.now() + 5000
    let job: Record<string, unknown> | undefined
    while (Date.now() < deadline) {
      job = (await call('local.status', {})).job as Record<string, unknown> | undefined
      if (job?.state !== 'running') break
      await Bun.sleep(10)
    }
    expect(job).toMatchObject({ state: 'done' })
    expect(await call('run_mode.get', {})).toMatchObject({ mode: 'hybrid', worker: { profile: 'openrouter', local: false } })
  })
})

test('a daemon without the ports says so instead of pretending', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-run-mode-bare-'))
  try {
    const server = new DaemonServer({ profileStore: new ProfileStore(join(root, 'profiles.json')), runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, sessionDirectory: join(root, 's') }), socketPath: join(root, 'd.sock') })
    await server.start()
    const connection = await client(join(root, 'd.sock'))
    try {
      expect(await connection.call('run_mode.get', {})).toMatchObject({ ok: false, code: 'unsupported' })
      expect(await connection.call('local.status', {})).toMatchObject({ ok: false, code: 'unsupported' })
    } finally { connection.close(); await server.stop() }
  } finally { await rm(root, { recursive: true, force: true }) }
})
