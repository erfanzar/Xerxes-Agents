// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { connect, type Socket } from 'node:net'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DaemonServer, type DaemonServerOptions } from '../src/daemon/server.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { ProfileStore } from '../src/bridge/profiles.js'
import type { CompletionRequest, LlmClient } from '../src/llms/client.js'
import { seedModelsDev } from './fixtures/modelsDev.js'

// Model capabilities come from models.dev at runtime; tests use its fixture.
seedModelsDev()

function record(value: unknown): value is Record<string, unknown> { return value !== null && typeof value === 'object' && !Array.isArray(value) }
function grantId(value: Record<string, unknown>): string {
  if (!record(value.grant) || typeof value.grant.id !== 'string') throw new Error('Fixture grant missing')
  return value.grant.id
}

class Client {
  private nextId = 0
  private buffer = ''
  private pending = new Map<number, { resolve: (value: Record<string, unknown>) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>()
  constructor(private readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', value => {
      this.buffer += value
      while (this.buffer.includes('\n')) {
        const end = this.buffer.indexOf('\n'), row: unknown = JSON.parse(this.buffer.slice(0, end)); this.buffer = this.buffer.slice(end + 1)
        if (!record(row) || typeof row.id !== 'number') continue
        const pending = this.pending.get(row.id); if (!pending) continue
        this.pending.delete(row.id); clearTimeout(pending.timer)
        if (row.error) pending.reject(new Error('RPC rejected fixture request')); else if (record(row.result)) pending.resolve(row.result); else pending.reject(new Error('Invalid fixture RPC result'))
      }
    })
    socket.on('close', () => { for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(new Error('Fixture connection closed')) }; this.pending.clear() })
  }
  request(method: string, params: object = {}): Promise<Record<string, unknown>> {
    const id = ++this.nextId
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error('Fixture request timed out')) }, 5000)
      this.pending.set(id, { resolve, reject, timer }); this.socket.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
    })
  }
  close() { this.socket.destroy() }
}
async function fixture(provider: LlmClient, codexModelCatalog: DaemonServerOptions['codexModelCatalog'] = async () => []) {
  const dir = await mkdtemp(join(tmpdir(), 'xr-daemon-relay-'))
  const profiles = new ProfileStore(join(dir, 'profiles.json'))
  profiles.save({ name: 'local', provider: 'openai', apiKey: 'synthetic-private-credential', baseUrl: 'https://provider.invalid/v1?key=synthetic-endpoint-secret', model: 'gpt-4o' })
  const server = new DaemonServer({ runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: dir, sessionDirectory: join(dir, 'sessions') }), projectDirectory: dir, socketPath: join(dir, 'daemon.sock'), profileStore: profiles, relayClientFactory: () => provider, codexModelCatalog })
  await server.start()
  const clients: Client[] = []
  const open = async () => { const socket = connect(join(dir, 'daemon.sock')); await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) }); const client = new Client(socket); clients.push(client); return client }
  return { open, profiles, async close() { clients.forEach(c => c.close()); await server.stop(); await rm(dir, { recursive: true, force: true }) } }
}
const consent = () => ({ consent: true, destination: 'verified-test-host', workspace: '/remote/project', profile: 'local', model: 'gpt-4o', expires_at: Date.now() + 60_000, max_requests: 10, max_output_tokens: 1024, max_concurrent: 2 })
const frame = { op: 'next', id: 'call-one', request: { model: 'gpt-4o', messages: [{ role: 'user', content: 'hello' }] } }

test('real daemon grants require consent, disclose no credentials, and remain owned by one connection', async () => {
  const host = await fixture({ async *stream() { yield { content: 'local provider reply' } } })
  try {
    const owner = await host.open(), other = await host.open()
    expect(await owner.request('provider.relay.authorize', { ...consent(), consent: false })).toMatchObject({ ok: false, code: 'invalid_request' })
    const inventory = await owner.request('provider.relay.inventory')
    expect(JSON.stringify(inventory)).not.toContain('synthetic-')
    const result = await owner.request('provider.relay.authorize', consent())
    expect(result).toMatchObject({ ok: true, grant: { profile: 'local', destination: 'verified-test-host', persistence: 'memory only' } })
    expect(result).toMatchObject({grant:{capabilities:{version:1,model:'gpt-4o',reasoning:{shape:'inherent',efforts:[],provenance:'bundled_catalog'}}}})
    expect(JSON.stringify(result)).not.toContain('synthetic-')
    expect(result.grant).not.toHaveProperty('token')
    const id = grantId(result)
    expect(await other.request('provider.relay.next', { id, frame })).toMatchObject({ ok: false, code: 'grant_unavailable' })
    expect(await owner.request('provider.relay.next', { id, frame })).toMatchObject({ ok: true, reply: { deltas: [{ content: 'local provider reply' }] } })
    expect(await owner.request('provider.relay.status', { id })).toMatchObject({ ok: true, grant: { requestsUsed: 1 } })
    owner.close(); await Bun.sleep(20)
    const replacement = await host.open()
    expect(await replacement.request('provider.relay.status', { id })).toMatchObject({ ok: false, code: 'grant_unavailable' })
  } finally { await host.close() }
})

test('authorized capability discovery runs against the exact local profile and exposes no configuration or catalog prose',async()=>{
  let lookups=0
  const host=await fixture({async *stream(){throw new Error('No provider request expected')}},async profile=>{
    lookups++;expect(profile.name).toBe('local-subscription')
    return [{id:'gpt-5.1',displayName:undefined,contextLimit:undefined,defaultReasoningLevel:'low',harnessCoupled:false,
      reasoningLevels:[{effort:'low',description:'synthetic-private-catalog-prose'},{effort:'ultra',description:undefined}]}]
  })
  try {
    host.profiles.save({name:'local-subscription',provider:'openai-codex',model:'gpt-5.1',baseUrl:'https://fixture.invalid',apiKey:'synthetic-profile-secret'})
    const owner=await host.open(),params={...consent(),profile:'local-subscription',model:'gpt-5.1',max_output_tokens:null,consent_provider_controlled_output:true}
    expect(await owner.request('provider.relay.authorize',{...params,consent:false})).toMatchObject({ok:false})
    expect(lookups).toBe(0)
    const granted=await owner.request('provider.relay.authorize',params)
    expect(granted).toMatchObject({ok:true,grant:{capabilities:{model:'gpt-5.1',reasoning:{efforts:['low','ultra'],provenance:'provider_reported'}}}})
    expect(lookups).toBe(1)
    expect(JSON.stringify(granted)).not.toContain('synthetic')
    expect(JSON.stringify(granted)).not.toContain('fixture.invalid')
    await owner.request('provider.relay.revoke',{id:grantId(granted)})
  } finally {await host.close()}
})

test('a grant that expires during local capability discovery cannot be returned as active',async()=>{
  const host=await fixture({async *stream(){throw new Error('Expired setup must not execute')}},async()=>{await Bun.sleep(100);return []})
  try {
    host.profiles.save({name:'local-subscription',provider:'openai-codex',model:'gpt-5',baseUrl:'https://fixture.invalid',apiKey:''})
    const owner=await host.open()
    expect(await owner.request('provider.relay.authorize',{...consent(),profile:'local-subscription',model:'gpt-5',expires_at:Date.now()+50,max_output_tokens:null,consent_provider_controlled_output:true})).toMatchObject({ok:false,code:'grant_expired'})
    expect(await owner.request('provider.relay.authorize',consent())).toMatchObject({ok:true})
  } finally {await host.close()}
})

test('revocation RPC is not blocked by a pending provider pull and aborts the actual client', async () => {
  const started = Promise.withResolvers<void>(); let aborted = false
  const host = await fixture({ async *stream(_request, signal) { started.resolve(); await new Promise<void>(resolve => signal!.addEventListener('abort', () => { aborted = true; resolve() }, { once: true })); yield { content: 'must not escape' } } })
  try {
    const owner = await host.open(), grant = await owner.request('provider.relay.authorize', consent()), id = grantId(grant)
    const waiting = owner.request('provider.relay.next', { id, frame })
    await started.promise
    expect(await owner.request('provider.relay.revoke', { id })).toEqual({ ok: true })
    expect(await waiting).toMatchObject({ ok: true, reply: { error: 'grant_revoked' } })
    expect(aborted).toBe(true)
    expect(await owner.request('provider.relay.status', { id })).toMatchObject({ grant: { status: 'revoked' } })
  } finally { await host.close() }
})

test('changing a consented route rejects the next request without calling the replacement provider', async () => {
  let calls = 0
  const host = await fixture({ async *stream() { calls++; yield { content: 'approved reply' } } })
  try {
    const owner = await host.open(), id = grantId(await owner.request('provider.relay.authorize', consent()))
    host.profiles.save({ name: 'local', provider: 'openai', apiKey: 'synthetic-replacement-key', baseUrl: 'https://replacement.invalid/v1?key=synthetic-new-secret', model: 'gpt-4o' })
    const result = await owner.request('provider.relay.next', { id, frame })
    expect(result).toMatchObject({ ok: true, reply: { error: 'route_changed' } })
    expect(JSON.stringify(result)).not.toContain('synthetic-')
    expect(calls).toBe(0)
  } finally { await host.close() }
})

test('disconnect aborts an active provider and reconnect cannot reuse its old grant', async () => {
  const started = Promise.withResolvers<void>(), aborted = Promise.withResolvers<void>()
  const host = await fixture({ async *stream(_request, signal) {
    started.resolve()
    await new Promise<void>(resolve => signal!.addEventListener('abort', () => { aborted.resolve(); resolve() }, { once: true }))
  } })
  try {
    const owner = await host.open(), id = grantId(await owner.request('provider.relay.authorize', consent()))
    const pending = owner.request('provider.relay.next', { id, frame }).catch(() => ({ closed: true }))
    await started.promise
    owner.close()
    await aborted.promise
    expect(await pending).toEqual({ closed: true })
    const replacement = await host.open()
    expect(await replacement.request('provider.relay.next', { id, frame })).toMatchObject({ ok: false, code: 'grant_unavailable' })
  } finally { await host.close() }
})

test('provider exception diagnostics never cross the daemon relay boundary', async () => {
  const host = await fixture({ async *stream() { throw new Error('Authorization: Bearer synthetic-secret-from-provider') } })
  try {
    const owner = await host.open(), id = grantId(await owner.request('provider.relay.authorize', consent()))
    const result = await owner.request('provider.relay.next', { id, frame })
    expect(result).toEqual({ ok: true, reply: { error: 'provider_failed' } })
    expect(JSON.stringify(result)).not.toContain('synthetic-')
  } finally { await host.close() }
})

test('idle restart cannot discard a live grant between remote provider requests', async () => {
  const host = await fixture({ async *stream() { yield { content: 'reply' } } })
  try {
    const owner = await host.open(), other = await host.open()
    await owner.request('provider.relay.authorize', consent())
    expect(await other.request('runtime.restart_if_idle')).toEqual({ ok: false, busy: true })
  } finally { await host.close() }
})

test('local profile defaults stay local, are snapshotted at consent, and allow bounded explicit overrides', async () => {
  const received: CompletionRequest[] = []
  const host = await fixture({ async *stream(request) { received.push(request); yield { content: 'reply' } } })
  try {
    host.profiles.save({ name: 'local', provider: 'openai', apiKey: 'synthetic-private-credential', baseUrl: 'https://provider.invalid/v1', model: 'gpt-4o', sampling: { max_tokens: 4096, temperature: 0.2, top_k: 20, top_p: 0.8, reasoning_effort: 'low', thinking_budget: 128, service_tier: 'flex', extraBody: { private: 'synthetic-secret' } } })
    const owner = await host.open(), id = grantId(await owner.request('provider.relay.authorize', consent()))
    host.profiles.save({ name: 'local', provider: 'openai', apiKey: 'synthetic-private-credential', baseUrl: 'https://provider.invalid/v1', model: 'gpt-4o', sampling: { temperature: 1.9, max_tokens: 5 } })
    const first = await owner.request('provider.relay.next', { id, frame })
    expect(first).toMatchObject({ ok: true, reply: { deltas: [{ content: 'reply' }] } })
    expect(received[0]).toMatchObject({ maxTokens: 1024, temperature: 0.2, topK: 20, topP: 0.8, thinking: { effort: 'low', budgetTokens: 128 }, serviceTier: 'flex' })
    expect(received[0]).not.toHaveProperty('extraBody')
    const override = { ...frame, id: 'explicit', request: { ...frame.request, maxTokens: 256, temperature: 0, topK: 0, topP: 0, thinking: { effort: 'none' }, serviceTier: 'default' } }
    await owner.request('provider.relay.next', { id, frame: override })
    expect(received[1]).toMatchObject({ maxTokens: 256, temperature: 0, topK: 0, topP: 0, thinking: { effort: 'none' }, serviceTier: 'default' })
    const refused = await owner.request('provider.relay.next', { id, frame: { ...override, id: 'oversized', request: { ...override.request, maxTokens: 1025 } } })
    expect(refused).toMatchObject({ ok: true, reply: { error: 'invalid_request' } })
    expect(received).toHaveLength(2)
    expect(JSON.stringify(await owner.request('provider.relay.status', { id }))).not.toContain('synthetic-')
  } finally { await host.close() }
})

test('invalid local sampling refuses consent without calling a provider', async () => {
  let calls = 0
  const host = await fixture({ async *stream() { calls++; } })
  try {
    const owner = await host.open()
    for (const sampling of [{ max_tokens: -1 }, { top_p: 2 }, { temperature: 'synthetic-secret' }]) {
      host.profiles.save({ name: 'local', provider: 'openai', apiKey: 'synthetic-private-credential', baseUrl: 'https://provider.invalid/v1', model: 'gpt-4o', sampling })
      expect(await owner.request('provider.relay.authorize', consent())).toMatchObject({ ok: false, code: 'invalid_request' })
    }
    expect(calls).toBe(0)
  } finally { await host.close() }
})

for (const provider of ['openai-codex', 'codex', 'openai_codex']) test(`${provider} provider-controlled output requires explicit consent and remains revocable`, async () => {
  let calls = 0
  const host = await fixture({ async *stream() { calls++; yield { content: 'subscription reply' } } })
  try {
    host.profiles.save({ name: 'subscription', provider, model: 'gpt-5', baseUrl: 'https://fixture.invalid', apiKey: '' })
    const owner = await host.open()
    const params = { ...consent(), profile: 'subscription', model: 'gpt-5' }
    const inventory = await owner.request('provider.relay.inventory')
    expect(inventory.profiles).toEqual(expect.arrayContaining([expect.objectContaining({ name: 'subscription', output_limit_mode: 'provider-controlled', setup: expect.stringContaining('Explicit consent') })]))
    expect(await owner.request('provider.relay.authorize', params)).toMatchObject({ ok: false, code: 'output_limit_unsupported' })
    for (const extra of [{}, { consent_provider_controlled_output: false }]) {
      expect(await owner.request('provider.relay.authorize', { ...params, max_output_tokens: null, ...extra })).toMatchObject({ ok: false, code: 'invalid_request' })
    }
    expect(calls).toBe(0)
    const grant = await owner.request('provider.relay.authorize', { ...params, max_output_tokens: null, consent_provider_controlled_output: true })
    expect(grant).toMatchObject({ ok: true, grant: { maxOutputTokens: null, outputLimitMode: 'provider-controlled', maxRequests: 10, maxConcurrent: 2 } })
    const id = grantId(grant), request = { ...frame, request: { ...frame.request, model: 'gpt-5' } }
    expect(await owner.request('provider.relay.next', { id, frame: request })).toMatchObject({ ok: true, reply: { deltas: [{ content: 'subscription reply' }] } })
    expect(calls).toBe(1)
    await owner.request('provider.relay.revoke', { id })
    expect(await owner.request('provider.relay.next', { id, frame: { ...request, id: 'after-revoke' } })).toMatchObject({ ok: true, reply: { error: 'grant_revoked' } })
    expect(calls).toBe(1)
    expect(await owner.request('provider.relay.authorize', { ...consent(), max_output_tokens: null, consent_provider_controlled_output: true })).toMatchObject({ ok: false, code: 'invalid_request' })
  } finally { await host.close() }
})

test('repeated revoked grants release authority and retain only bounded owner-scoped terminal status', async () => {
  let calls = 0
  const host = await fixture({async *stream(){calls++;yield {content:'active grant still works'}}})
  try {
    const owner=await host.open(),other=await host.open()
    const active=grantId(await owner.request('provider.relay.authorize',consent()))
    let first='',last=''
    for(let i=0;i<300;i++){
      last=grantId(await owner.request('provider.relay.authorize',consent()))
      if(!i)first=last
      expect(await owner.request('provider.relay.revoke',{id:last})).toEqual({ok:true})
      expect(await owner.request('provider.relay.status',{id:last})).toMatchObject({ok:true,grant:{status:'revoked',activeRequests:0}})
    }
    expect(await owner.request('provider.relay.status',{id:first})).toMatchObject({ok:false,code:'grant_unavailable'})
    expect(await other.request('provider.relay.status',{id:last})).toMatchObject({ok:false,code:'grant_unavailable'})
    expect(await owner.request('provider.relay.revoke',{id:last})).toEqual({ok:true})
    expect(await owner.request('provider.relay.next',{id:last,frame})).toMatchObject({ok:true,reply:{error:'grant_revoked'}})
    expect(await owner.request('provider.relay.next',{id:active,frame})).toMatchObject({ok:true,reply:{deltas:[{content:'active grant still works'}]}})
    expect(calls).toBe(1)
    owner.close();await Bun.sleep(10)
    expect(await other.request('provider.relay.status',{id:last})).toMatchObject({ok:false,code:'grant_unavailable'})
  }finally{await host.close()}
})
