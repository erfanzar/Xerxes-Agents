// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtemp, rm, stat } from 'node:fs/promises'
import { createConnection, createServer, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { expect, it, vi } from 'vitest'
import { ProfileStore } from '../../bridge/profiles.js'
import { JobStore } from '../../cron/jobs.js'
import { DaemonServer } from '../../daemon/server.js'
import { InMemoryDaemonRuntime } from '../../daemon/runtime.js'
import { AgentTurnRunner } from '../../daemon/turnRunner.js'
import { RemoteProviderBindings } from '../../daemon/remoteProviderBindings.js'
import type { LlmClient } from '../../llms/client.js'
import { GatewayClient } from '../gatewayClient.js'
import { createLocalProviderBroker, requestLocalProviderBroker, type LocalProviderConsent } from '../lib/localProviderBroker.js'

const consent = (): LocalProviderConsent => ({ consent: true, destination: 'fixture-host', workspace: '/remote/project',
  profile: 'local', model: 'gpt-4o', expires_at: Date.now() + 60_000, max_requests: 10, max_output_tokens: 1024, max_concurrent: 2 })
const frame = (id = 'first') => ({ op: 'next', id, request: { model: 'gpt-4o', messages: [{ role: 'user', content: 'hello 🌍' }] } })
const storage = (dir: string) => ({ sessionArchiveDirectory: join(dir, 'sessions'), cronLeasePath: join(dir, 'cron.lease'),
  cronStoreFactory: () => new JobStore(join(dir, 'jobs.json')), cronArchiveDirectory: join(dir, 'cron-archive'), legacyScheduleDirectory: join(dir, 'scheduler') })

async function fixture(provider: LlmClient) {
  const dir = await mkdtemp(join(tmpdir(), 'xr-broker-test-'))
  const profiles = new ProfileStore(join(dir, 'profiles.json'))
  profiles.save({ name: 'local', provider: 'openai', baseUrl: 'https://provider.invalid/v1', apiKey: 'synthetic-private-key', model: 'gpt-4o' })
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: dir, sessionDirectory: join(dir, 'sessions') })
  const server = new DaemonServer({ ...storage(dir), runtime, projectDirectory: dir, socketPath: join(dir, 'rpc.sock'), profileStore: profiles, relayClientFactory: () => provider })
  const gateway = new GatewayClient({ externalSocketPath: join(dir, 'rpc.sock'), projectDir: dir })
  await server.start(); await gateway.start()
  const rpc = vi.fn((method: string, params: Record<string, unknown>) => gateway.request(method, params))
  return { gateway, rpc, async close() { gateway.close(); await server.stop(); await runtime.shutdown(); await rm(dir, { recursive: true, force: true }) } }
}

it('bridges actual daemon authority over a private local socket and removes it on idempotent close', async () => {
  const seen: unknown[] = []
  const f = await fixture({ async *stream(request) { seen.push(request); yield { content: 'Local reply 🌍' } } })
  const broker = await createLocalProviderBroker(f.rpc, consent())
  try {
    expect((await stat(dirname(broker.path))).mode & 0o777).toBe(0o700)
    expect((await stat(broker.path)).mode & 0o777).toBe(0o600)
    const reply = await requestLocalProviderBroker(broker.path, frame())
    expect(reply).toMatchObject({ deltas: [{ content: 'Local reply 🌍' }] })
    expect(seen).toHaveLength(1)
    expect(JSON.stringify(reply)).not.toContain('synthetic-private-key')
    expect(await requestLocalProviderBroker(broker.path, { ...frame('wrong'), request: { ...frame().request, model: 'unapproved' } })).toMatchObject({ error: 'route_mismatch' })
    await Promise.all([broker.close(), broker.close()])
    expect(f.rpc.mock.calls.filter(([method]) => method === 'provider.relay.revoke')).toHaveLength(1)
    await expect(stat(dirname(broker.path))).rejects.toThrow()
    expect(await requestLocalProviderBroker(broker.path, frame('closed'))).toEqual({ error: 'grant_unavailable' })
  } finally { await broker.close(); await f.close() }
})

it.each(['request', 'owner', 'broker'] as const)('cancels a waiting native provider when the %s disconnects', async boundary => {
  let started = false, aborted = false
  const f = await fixture({ async *stream(_request, signal) {
    started = true
    await new Promise<void>(resolve => signal!.addEventListener('abort', () => { aborted = true; resolve() }, { once: true }))
    yield { content: 'late result must not escape' }
  } })
  const controller = new AbortController()
  const broker = await createLocalProviderBroker(f.rpc, consent())
  try {
    const waiting = requestLocalProviderBroker(broker.path, frame(), controller.signal)
    await vi.waitFor(() => expect(started).toBe(true))
    if (boundary === 'request') controller.abort()
    else if (boundary === 'owner') (f.gateway as unknown as { socket: Socket }).socket.destroy()
    else await broker.close()
    expect(JSON.stringify(await waiting)).not.toContain('late result')
    await vi.waitFor(() => expect(aborted).toBe(true))
    if (boundary === 'owner') {
      await f.gateway.start()
      expect(await requestLocalProviderBroker(broker.path, frame('new-owner'))).toEqual({ error: 'grant_unavailable' })
    }
  } finally { await broker.close(); await f.close() }
})

it('expires and removes the local endpoint without persisting or renewing authority', async () => {
  const f = await fixture({ async *stream() { yield { content: 'unused' } } })
  const broker = await createLocalProviderBroker(f.rpc, { ...consent(), expires_at: Date.now() + 80 })
  try {
    await vi.waitFor(() => expect(f.rpc.mock.calls.some(([method]) => method === 'provider.relay.revoke')).toBe(true))
    expect(await requestLocalProviderBroker(broker.path, frame())).toEqual({ error: 'grant_unavailable' })
    await broker.close()
    expect(f.rpc.mock.calls.filter(([method]) => method === 'provider.relay.authorize')).toHaveLength(1)
    await expect(stat(broker.path)).rejects.toThrow()
  } finally { await broker.close(); await f.close() }
})

it('cancels a late authorization before opening a socket and revokes the resulting grant', async () => {
  const authorization = Promise.withResolvers<unknown>()
  const rpc = vi.fn(async (method: string) => method === 'provider.relay.authorize' ? authorization.promise : { ok: true })
  const controller = new AbortController()
  const opening = createLocalProviderBroker(rpc, consent(), controller.signal)
  const rejected = expect(opening).rejects.toThrow('Local provider bridge is unavailable')
  controller.abort()
  authorization.resolve({ ok: true, grant: { id: 'a'.repeat(32) } })
  await rejected
  expect(rpc).toHaveBeenCalledWith('provider.relay.revoke', { id: 'a'.repeat(32) })
  await expect(createLocalProviderBroker(rpc, consent(), controller.signal)).rejects.toThrow()
  expect(rpc.mock.calls.filter(([method]) => method === 'provider.relay.authorize')).toHaveLength(1)
})

it.each([
  {version:1,model:'wrong-model',reasoning:{shape:'inherent',efforts:[],canDisable:false,provenance:'bundled_catalog'}},
  {version:1,model:'gpt-4o',api_key:'synthetic-private-key',reasoning:{shape:'inherent',efforts:[],canDisable:false,provenance:'bundled_catalog'}},
])('rejects invalid local capability metadata and revokes the already-created authority',async capabilities=>{
  const rpc=vi.fn(async(method:string)=>method==='provider.relay.authorize'?{ok:true,grant:{id:'b'.repeat(32),capabilities}}:{ok:true})
  await expect(createLocalProviderBroker(rpc,consent())).rejects.toThrow('Local provider bridge is unavailable')
  expect(rpc).toHaveBeenCalledWith('provider.relay.revoke',{id:'b'.repeat(32)})
})

it('rejects malformed, oversized and pipelined frames without forwarding raw diagnostics', async () => {
  const rpc = vi.fn(async (method: string) => {
    if (method === 'provider.relay.authorize') return { ok: true, grant: { id: 'a'.repeat(32) } }
    if (method === 'provider.relay.next') throw new Error('synthetic-private-diagnostic')
    return { ok: true }
  })
  const broker = await createLocalProviderBroker(rpc, consent())
  const raw = (data: string) => new Promise<string>(resolve => {
    let result = ''
    const socket = createConnection(broker.path, () => socket.write(data))
    socket.on('data', chunk => { result += String(chunk) })
    socket.on('error', () => {})
    socket.on('close', () => resolve(result))
  })
  try {
    expect(await raw('{secret}\n')).toBe('')
    expect(await raw('{}\n{}\n')).toBe('')
    expect(await raw('x'.repeat(16 * 1024 * 1024 + 4097))).toBe('')
    expect(rpc.mock.calls.filter(([method]) => method === 'provider.relay.next')).toHaveLength(0)
    expect(await requestLocalProviderBroker(broker.path, frame())).toEqual({ error: 'grant_unavailable' })
    expect(await requestLocalProviderBroker(broker.path, { ...frame(), token: 'unwanted' })).toEqual({ error: 'invalid_request' })
  } finally { await broker.close() }
})

it('bounds stalled connections and still permits service after they close', async () => {
  const rpc = vi.fn(async (method: string) => method === 'provider.relay.authorize' ? { ok: true, grant: { id: 'a'.repeat(32) } } : { ok: true, reply: { done: true } })
  const broker = await createLocalProviderBroker(rpc, consent())
  const sockets: Socket[] = []
  try {
    for (let i = 0; i < 16; i++) await new Promise<void>(resolve => {
      const socket = createConnection(broker.path, resolve); sockets.push(socket)
    })
    expect(await requestLocalProviderBroker(broker.path, frame())).toEqual({ error: 'grant_unavailable' })
    for (const socket of sockets) socket.destroy()
    await vi.waitFor(async () => expect(await requestLocalProviderBroker(broker.path, frame())).toEqual({ done: true }))
  } finally { for (const socket of sockets) socket.destroy(); await broker.close() }
})

it('decodes split UTF-8 replies and rejects oversized responses without exposing their content', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xr-broker-peer-')), path = join(dir, 'test.sock')
  let oversized = false
  const server = createServer(socket => socket.once('data', () => {
    if (oversized) { socket.end('x'.repeat(1024 * 1024 + 4097)); return }
    const payload = Buffer.from(JSON.stringify({ done: true, deltas: [{ content: '🌍' }] }) + '\n')
    const split = payload.indexOf(Buffer.from('🌍')) + 1
    socket.write(payload.subarray(0, split))
    setTimeout(() => socket.end(payload.subarray(split)), 5)
  }))
  await new Promise<void>(resolve => server.listen(path, resolve))
  try {
    expect(await requestLocalProviderBroker(path, frame())).toMatchObject({ deltas: [{ content: '🌍' }] })
    oversized = true
    expect(await requestLocalProviderBroker(path, frame())).toEqual({ error: 'grant_unavailable' })
  } finally { await new Promise<void>(resolve => server.close(() => resolve())); await rm(dir, { recursive: true, force: true }) }
})

it('runs a remote native turn through two daemons and the local broker, then fails closed after revocation', async () => {
  const f = await fixture({ async *stream() { yield { content: 'Reply through the local broker.' } } })
  const dir = await mkdtemp(join(tmpdir(), 'xr-broker-remote-'))
  const broker = await createLocalProviderBroker(f.rpc, { ...consent(), workspace: dir })
  const bindings = new RemoteProviderBindings()
  const fallback = vi.fn()
  const runner = new AgentTurnRunner({ model: 'gpt-4o', tools: [], remoteProviderBindings: bindings,
    llm: { async *stream() { fallback(); yield { content: 'unapproved fallback' } } } })
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: dir, sessionDirectory: join(dir, 'sessions'), model: 'gpt-4o' })
  const server = new DaemonServer({ ...storage(dir), runtime, projectDirectory: dir, socketPath: join(dir, 'rpc.sock'),
    profileStore: new ProfileStore(join(dir, 'profiles.json')), remoteProviderBindings: bindings })
  let binding = ''
  const gateway = new GatewayClient({ externalSocketPath: join(dir, 'rpc.sock'), projectDir: dir,
    providerRelay: (id, value, signal) => id === binding ? requestLocalProviderBroker(broker.path, value, signal) : Promise.resolve({ error: 'grant_unavailable' }) })
  const events: unknown[] = []
  gateway.on('event', event => events.push(event))
  try {
    await server.start(); await gateway.start()
    await gateway.request('session.create')
    const bound = await gateway.request<{ binding: { binding: string } }>('provider.remote.bind', { consent: true, source: 'local fixture', profile: 'local', model: 'gpt-4o' })
    binding = bound.binding.binding
    await gateway.request('turn.submit', { text: 'Run with my approved local provider.' })
    await vi.waitFor(() => expect(JSON.stringify(runtime.listSessions())).toContain('Reply through the local broker.'))
    await vi.waitFor(() => expect(runtime.listSessions().every(session => !session.activeTurnId)).toBe(true))
    expect(JSON.stringify(events)).not.toContain('provider.remote.request')
    expect(JSON.stringify(events)).not.toContain('synthetic-private-key')
    await broker.close()
    await gateway.request('turn.submit', { text: 'After revocation.' })
    await vi.waitFor(() => expect(JSON.stringify(events)).toContain('Local provider grant is unavailable'))
    expect(fallback).not.toHaveBeenCalled()
  } finally {
    gateway.close(); await server.stop(); await runtime.shutdown(); await broker.close(); await f.close()
    await rm(dir, { recursive: true, force: true })
  }
})
