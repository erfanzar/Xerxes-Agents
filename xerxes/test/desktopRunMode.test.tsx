// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { RunModeSection } from '../src/desktop/renderer/RunModeSection.js'
import { localSetupJobView, localStatusView, runModeView } from '../src/desktop/renderer/runModeView.js'
import { Store, type XerxesLike } from '../src/desktop/renderer/store.js'
import type { DaemonEvent } from '../src/desktop/renderer/types.js'

type Responder = (method: string, params: Record<string, unknown>) => Record<string, unknown>

class FakeBridge implements XerxesLike {
  readonly calls: Array<{ method: string; params: Record<string, unknown> }> = []
  private handlers = new Set<(event: DaemonEvent) => void>()
  constructor(public respond: Responder) {}
  call(method: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    this.calls.push({ method, params })
    return Promise.resolve(this.respond(method, params))
  }
  push(type: string, payload: Record<string, unknown>): void {
    for (const handler of this.handlers) handler({ type, payload })
  }
  subscribe(handler: (event: DaemonEvent) => void): () => void {
    this.handlers.add(handler)
    return () => this.handlers.delete(handler)
  }
}

const hybrid = {
  ok: true, mode: 'hybrid', worker: { profile: 'claude-code', model: 'claude-code/opus', local: false },
  local_only: false, restart_pending: false, locked_by: [],
}
const localStatus = {
  ready: true,
  engine: { name: 'llama-server', label: 'llama.cpp', found: true, version: '0.9.0', new_enough: true },
  model: { id: 'ternary-bonsai-2-27b', label: 'Ternary Bonsai 2 27B', bytes: 7_626_008_928, downloaded: false },
  server: { running: false, port: null, context_length: null },
  memory_bytes: 25_769_803_776,
  blocker: null,
}
const profiles = [
  { name: 'bonsai-local', label: 'bonsai-local', provider: 'custom', model: 'ternary-bonsai-2-27b', active: false, local: true },
  { name: 'claude-code', label: 'Claude Code', provider: 'claude-code', model: 'claude-code/opus', active: true, local: false },
]

function connectedStore(respond: Responder): { store: Store; bridge: FakeBridge } {
  const bridge = new FakeBridge((method, params) => {
    if (method === 'initialize') return { session_id: 's1', session: { id: 's1', title: '', cwd: '/work' }, model: 'ternary-bonsai-2-27b', cwd: '/work', daemon_protocol: 35 }
    if (method === 'session.list' || method === 'session.active_list') return { ok: true, sessions: [] }
    return respond(method, params)
  })
  ;(globalThis as { window?: unknown }).window = { xerxes: { onEvent: (handler: (event: DaemonEvent) => void) => bridge.subscribe(handler) } }
  const store = new Store()
  store.start(bridge)
  return { store, bridge }
}

test('run mode, local status and setup jobs are read from the wire without inventing values', () => {
  expect(runModeView(hybrid)).toEqual({
    mode: 'hybrid', worker: { profile: 'claude-code', model: 'claude-code/opus', local: false }, restartPending: false, lockedBy: [],
  })
  expect(runModeView({ ok: true, mode: 'remote', restart_required: true })).toMatchObject({ mode: null, worker: null, restartPending: true })
  expect(localStatusView(localStatus)).toMatchObject({ engine: { label: 'llama.cpp', newEnough: true }, model: { bytes: 7_626_008_928, downloaded: false } })
  expect(localStatusView(undefined)).toBeNull()
  expect(localSetupJobView({ id: 'j', state: 'running', step: 'Downloading: 5% (0.4 GiB / 7.1 GiB)', percent: 5 })).toEqual({ id: 'j', state: 'running', step: 'Downloading: 5% (0.4 GiB / 7.1 GiB)', percent: 5, error: null })
  expect(localSetupJobView({ id: 'j', state: 'failed', step: 'x', error: { code: 'BINARY_NOT_FOUND', message: 'Install llama.cpp.' } })).toMatchObject({ state: 'failed', percent: null, error: 'Install llama.cpp.' })
  expect(localSetupJobView({ state: 'done' })).toBeNull()
})

test('before anything is chosen, the built-in Claude Code fallback is listed as not in use, so it can be picked', async () => {
  const { store } = connectedStore(method => method === 'provider_list'
    ? { ok: true, active_chosen: false, profiles: [{ name: 'claude-code', provider: 'claude-code', model: '', active: true, local: false }] }
    : { ok: true })
  await store.loadProviders()
  expect(store.getSnapshot().providers[0]).toMatchObject({ name: 'claude-code', active: false, local: false })
})

test('choosing Hybrid without Bonsai asks for setup instead of reporting an error; a saved restart is applied', async () => {
  let mode: Record<string, unknown> = { ok: false, code: 'needs_local_setup', error: 'Set up Bonsai on this machine first.' }
  const { store, bridge } = connectedStore(method => method === 'run_mode.set' ? mode : method === 'desktop.restartRuntime' ? { ok: false, busy: true } : { ok: true })
  expect(await store.setRunMode('hybrid')).toEqual({ ok: false, code: 'needs_local_setup' })
  expect(store.getSnapshot().runModeError).toBe('')

  mode = { ...hybrid, mode: 'private', local_only: true, restart_required: true }
  expect(await store.setRunMode('private')).toEqual({ ok: true })
  expect(store.getSnapshot().runMode).toMatchObject({ mode: 'private', restartPending: true })
  await Bun.sleep(0)
  expect(bridge.calls.some(call => call.method === 'desktop.restartRuntime')).toBe(true)
  expect(store.getSnapshot().runtimeUpdateMessage).toBe('Mode saved. It applies when all running work finishes.')
})

test('setup progress events update the job, and a finished setup reloads the mode and providers', async () => {
  const { store, bridge } = connectedStore(method => method === 'run_mode.get' ? hybrid : method === 'local.status' ? { ok: true, status: localStatus, job: null } : method === 'provider_list' ? { ok: true, profiles } : { ok: true })
  bridge.push('local_setup', { job: { id: 'j1', state: 'running', step: 'Downloading ternary-bonsai-2-27b: 40% (2.8 GiB / 7.1 GiB)', percent: 40 } })
  expect(store.getSnapshot().localSetup.job).toMatchObject({ state: 'running', percent: 40 })
  bridge.calls.length = 0
  bridge.push('local_setup', { job: { id: 'j1', state: 'done', step: 'Download verified (sha256 ok).', result: { context_length: 32768 } } })
  await Bun.sleep(0)
  await Bun.sleep(0)
  expect(bridge.calls.map(call => call.method)).toEqual(expect.arrayContaining(['local.status', 'provider_list', 'run_mode.get']))
  expect(store.getSnapshot().runMode?.mode).toBe('hybrid')
})

test('the Models page leads with the three modes and says what Hybrid needs: Bonsai on this Mac and a cloud lead', async () => {
  const { store } = connectedStore(method => method === 'run_mode.get' ? hybrid : method === 'local.status' ? { ok: true, status: localStatus, job: null } : method === 'provider_list' ? { ok: true, profiles } : { ok: true })
  await Promise.all([store.loadRunMode(), store.loadLocalSetup(), store.loadProviders()])
  const html = renderToStaticMarkup(createElement(RunModeSection, { snap: store.getSnapshot() }))
  for (const words of ['Private', 'Hybrid', 'Cloud', 'Nothing leaves during a run.', 'Your files stay here.', 'On this Mac', 'does the file work', 'Ternary Bonsai 2 27B', 'llama.cpp 0.9.0', 'Download 7.6 GB and set up Bonsai', 'Cloud lead', 'cannot read your files', 'Claude Code']) {
    expect(html).toContain(words)
  }
  expect(html).toContain('aria-checked="true"')
  expect(html.match(/In use/g)).toHaveLength(1)

  const unknown = connectedStore(() => ({ ok: false, error: 'Unknown method: run_mode.get' })).store
  await unknown.loadRunMode()
  // An older runtime without modes shows nothing here rather than a broken choice.
  expect(renderToStaticMarkup(createElement(RunModeSection, { snap: unknown.getSnapshot() }))).toBe('')
})
