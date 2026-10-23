// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, expect, test } from 'bun:test'
import { generateKeyPairSync, randomUUID, type KeyObject } from 'node:crypto'
import { mkdtemp, rm } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ProfileStore, type ProviderProfile } from '../src/bridge/profiles.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import type { CompletionRequest, LlmClient, LlmCompletion } from '../src/llms/client.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'
import { paramsHash, signHubAuth, type HubAuthClaims } from '../src/security/hubAuthEnvelope.js'
import { HubIdentityVerifier, enrollHub } from '../src/security/hubIdentityVerifier.js'
import { clearModelsDev, seedModelsDev } from './fixtures/modelsDev.js'

const API_KEY = 'sk-test-secret-key-1234567890abcdef'
const BASE_URL = 'https://api.openai.com/v1'

const directories: string[] = []
const nativeFetch = globalThis.fetch
let listing: 'ok' | 'fail' = 'ok'

beforeEach(() => {
  listing = 'ok'
  seedModelsDev()
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url.startsWith(BASE_URL) && url.includes('/models')) {
      if (listing === 'fail') throw new Error(`connect failed for ${BASE_URL}/models using key ${API_KEY}`)
      return new Response(JSON.stringify({ data: [{ id: 'gpt-5.1' }, { id: 'gpt-5' }] }))
    }
    return new Response('{}', { status: 404 })
  }) as typeof globalThis.fetch
})
afterEach(async () => {
  globalThis.fetch = nativeFetch
  clearModelsDev()
  for (const directory of directories.splice(0)) await rm(directory, { recursive: true, force: true })
})

interface Frame { id?: number; result?: JsonRpcPayload; error?: { code: number; message: string } }
class Client {
  readonly frames: Frame[] = []
  private buffer = ''
  private id = 0
  constructor(readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', chunk => {
      this.buffer += chunk
      let split: number
      while ((split = this.buffer.indexOf('\n')) >= 0) {
        this.frames.push(JSON.parse(this.buffer.slice(0, split)) as Frame)
        this.buffer = this.buffer.slice(split + 1)
      }
    })
  }
  static async open(path: string): Promise<Client> {
    const socket = connect(path)
    const client = new Client(socket)
    await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) })
    return client
  }
  async call(method: string, params: JsonRpcPayload = {}): Promise<JsonRpcPayload> {
    const id = ++this.id
    this.socket.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
    for (let attempt = 0; attempt < 2500; attempt++) {
      const frame = this.frames.find(candidate => candidate.id === id)
      if (frame) {
        if (frame.error) throw new Error(`${frame.error.code}: ${frame.error.message}`)
        return frame.result!
      }
      await Bun.sleep(2)
    }
    throw new Error(`Timed out waiting for ${method}`)
  }
}

async function until(check: () => boolean, label: string): Promise<void> {
  for (let attempt = 0; attempt < 2500; attempt++) {
    if (check()) return
    await Bun.sleep(2)
  }
  throw new Error(`Timed out waiting for ${label}`)
}

/** Fake provider: records requests, answers immediately unless gated, and honours abort. */
class FakeLlm implements LlmClient {
  readonly requests: CompletionRequest[] = []
  readonly signals: AbortSignal[] = []
  readonly gate = Promise.withResolvers<void>()
  gated = false
  hangs = false
  failure: Error | undefined
  closed = 0
  reply: LlmCompletion = { content: 'hello from the fake', toolCalls: [], usage: { inputTokens: 11, outputTokens: 7 } }
  async *stream(): AsyncGenerator<never> { /* complete() is used */ }
  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<LlmCompletion> {
    this.requests.push(request)
    if (signal) this.signals.push(signal)
    if (this.failure) throw this.failure
    if (this.gated || this.hangs) {
      const aborted = new Promise<never>((_, reject) => signal?.addEventListener('abort', () => reject(signal.reason), { once: true }))
      await Promise.race([this.hangs ? new Promise<never>(() => undefined) : this.gate.promise, aborted])
    }
    return this.reply
  }
  close(): void { this.closed += 1 }
}

interface Rig {
  readonly root: string
  readonly client: Client
  readonly fake: FakeLlm
  readonly profiles: ProfileStore
  readonly enrollments: string
  readonly privateKey: KeyObject
  socketPath: string
  sign(method: string, params: JsonRpcPayload, scopes: readonly string[], kind?: 'agent' | 'user'): JsonRpcPayload
  stop(): Promise<void>
}

async function rig(options: { timeoutMs?: number } = {}): Promise<Rig> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-hub-models-'))
  directories.push(root)
  const pair = generateKeyPairSync('ed25519')
  const enrollments = join(root, 'hub-enrollments.json')
  enrollHub(enrollments, { hub_id: 'hub-1', public_key_pem: pair.publicKey.export({ type: 'spki', format: 'pem' }).toString(), project_dirs: [root] })
  const profiles = new ProfileStore(join(root, 'profiles.json'))
  profiles.save({ apiKey: API_KEY, baseUrl: BASE_URL, model: 'gpt-5.1', name: 'work', provider: 'openai' })
  profiles.save({ apiKey: '', baseUrl: 'http://127.0.0.1:11434/v1', model: 'qwen', name: 'local', provider: 'ollama', setActive: false })
  const fake = new FakeLlm()
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, model: 'protocol-model', sessionDirectory: join(root, 'sessions') })
  const server = new DaemonServer({
    socketPath: join(root, 'daemon.sock'), projectDirectory: root, runtime, autoTitle: false, autoSnapshotTurns: false,
    eventJournal: { directory: join(root, 'events') }, hubEnrollmentPath: enrollments, profileStore: profiles,
    hubModelClientFactory: (_model: string, _profile: ProviderProfile) => fake,
    ...(options.timeoutMs === undefined ? {} : { hubModelTimeoutMs: options.timeoutMs }),
  })
  await server.start()
  const socketPath = join(root, 'daemon.sock')
  const client = await Client.open(socketPath)
  return {
    root, client, fake, profiles, enrollments, privateKey: pair.privateKey, socketPath,
    sign(method, params, scopes, kind = 'agent') {
      const claims: HubAuthClaims = {
        hub_id: 'hub-1', principal_id: 'xerxes', principal_kind: kind,
        ...(kind === 'agent' ? {} : { user_id: 'user-a', device_id: 'phone-a' }),
        display_name: 'XERXES', workspace_id: 'ws-1', method, params_hash: paramsHash(params),
        request_id: randomUUID(), nonce: randomUUID(), issued_at: Date.now(), expires_at: Date.now() + 60_000, scopes,
      }
      return { ...params, hub_auth: signHubAuth(claims, pair.privateKey) as unknown as JsonRpcPayload }
    },
    async stop() { client.socket.destroy(); await server.stop() },
  }
}

const completion = (over: JsonRpcPayload = {}): JsonRpcPayload => ({
  profile: 'work', model: 'gpt-5.1', system: 'You are XERXES.', messages: [{ role: 'user', content: 'Alice: what is running?' }], tools: [], ...over,
})

test('the catalog lists profiles without any credential, endpoint or token', async () => {
  const r = await rig()
  try {
    const result = await r.client.call('hub.models.catalog')
    expect(result.ok).toBe(true)
    expect(typeof result.hostname).toBe('string')
    const profiles = result.profiles as Record<string, unknown>[]
    const work = profiles.find(profile => profile.name === 'work')!
    expect(Object.keys(work).sort()).toEqual(['default_model', 'label', 'name', 'provider', 'signs_in'])
    expect(work).toMatchObject({ provider: 'openai', default_model: 'gpt-5.1', signs_in: expect.any(Boolean) })
    expect(profiles.some(profile => profile.name === 'local')).toBe(true)
    const wire = JSON.stringify(result)
    expect(wire).not.toContain(API_KEY)
    expect(wire).not.toContain('api.openai.com')
    expect(wire).not.toContain('11434')
  } finally { await r.stop() }
})

test('options lists models and efforts; a failing listing falls back to the configured model with a scrubbed note', async () => {
  const r = await rig()
  try {
    const ok = await r.client.call('hub.models.options', { profile: 'work' })
    expect(ok).toMatchObject({ ok: true, profile: 'work', provider: 'openai', model: 'gpt-5.1' })
    expect((ok.models as { id: string }[]).map(model => model.id)).toEqual(['gpt-5.1', 'gpt-5'])
    expect((ok.efforts as { effort: string }[]).map(entry => entry.effort)).toContain('high')
    expect('default_effort' in ok).toBe(true)
    const other = await r.client.call('hub.models.options', { profile: 'work', model: 'gpt-5' })
    expect(other.model).toBe('gpt-5')

    listing = 'fail'
    const fallback = await r.client.call('hub.models.options', { profile: 'work' })
    expect(fallback.ok).toBe(true)
    // The daemon's own fallback: the configured model first, then models cached on the profile from the earlier listing.
    expect((fallback.models as { id: string }[]).map(model => model.id)).toEqual(['gpt-5.1', 'gpt-5'])
    expect(typeof fallback.models_note).toBe('string')
    expect((fallback.models_note as string).length).toBeGreaterThan(0)
    const wire = JSON.stringify(fallback)
    expect(wire).not.toContain(API_KEY)
    expect(wire).not.toContain('api.openai.com')

    expect(await r.client.call('hub.models.options', { profile: 'nope' })).toMatchObject({ ok: false, code: 'unknown_profile' })
    expect(await r.client.call('hub.models.options', {})).toMatchObject({ ok: false, code: 'invalid_request' })
  } finally { await r.stop() }
})

test('complete runs one non-streaming completion and returns text, tool calls (JSON string arguments) and usage', async () => {
  const r = await rig()
  try {
    r.fake.reply = {
      content: 'on it',
      toolCalls: [{ id: 'call-1', type: 'function', function: { name: 'task_list', arguments: { limit: 3 } } }],
      usage: { inputTokens: 120, outputTokens: 30 },
    }
    const tools = [{ type: 'function', function: { name: 'task_list', description: 'List tasks', parameters: { type: 'object', properties: { limit: { type: 'number' } } } } }]
    const messages = [
      { role: 'user', content: 'Alice: list tasks' },
      { role: 'assistant', content: '', tool_calls: [{ id: 'c0', type: 'function', function: { name: 'task_list', arguments: { limit: 1 } } }] },
      { role: 'tool', name: 'task_list', tool_call_id: 'c0', content: '<untrusted>[]</untrusted>' },
    ]
    const result = await r.client.call('hub.model.complete', completion({ messages, tools }))
    expect(result).toEqual({
      ok: true, text: 'on it',
      tool_calls: [{ id: 'call-1', name: 'task_list', arguments: '{"limit":3}' }],
      usage: { input_tokens: 120, output_tokens: 30 },
    })
    const seen = r.fake.requests[0]!
    expect(seen.model).toBe('gpt-5.1')
    expect(seen.maxTokens).toBe(2048)
    expect(seen.messages[0]).toEqual({ role: 'system', content: 'You are XERXES.' })
    expect(seen.messages).toHaveLength(4)
    expect(seen.tools?.[0]?.function.name).toBe('task_list')
    expect(seen.thinking).toBeUndefined()
    expect(r.fake.closed).toBe(1)
  } finally { await r.stop() }
})

test('complete maps the thinking effort onto the request and refuses an unsupported one', async () => {
  const r = await rig()
  try {
    await r.client.call('hub.model.complete', completion({ effort: 'high', max_tokens: 500 }))
    expect(r.fake.requests[0]).toMatchObject({ maxTokens: 500, thinking: { effort: 'high' } })
    await r.client.call('hub.model.complete', completion({ effort: 'off' }))
    expect(r.fake.requests[1]!.thinking).toBeUndefined()
    const refused = await r.client.call('hub.model.complete', completion({ effort: 'ultra-mega' }))
    expect(refused).toMatchObject({ ok: false, code: 'invalid_request' })
    expect(r.fake.requests).toHaveLength(2)
  } finally { await r.stop() }
})

test('complete rejects unknown profiles, oversize and badly shaped requests without calling the provider', async () => {
  const r = await rig()
  try {
    expect(await r.client.call('hub.model.complete', completion({ profile: 'nope' }))).toMatchObject({ ok: false, code: 'unknown_profile' })
    expect(await r.client.call('hub.model.complete', completion({ system: 'x'.repeat(1024 * 1024 + 10) }))).toMatchObject({ ok: false, code: 'invalid_request' })
    const bad: JsonRpcPayload[] = [
      completion({ messages: 'hi' }),
      completion({ messages: [] }),
      completion({ messages: [{ role: 'wizard', content: 'x' }] }),
      completion({ messages: [{ role: 'tool', content: 'x' }] }),
      completion({ tools: [{ type: 'function', function: { name: 'bad name!', description: '', parameters: {} } }] }),
      completion({ tools: 'all' }),
      completion({ max_tokens: 9000 }),
      completion({ max_tokens: 0 }),
      completion({ system: 5 }),
      completion({ model: '' }),
    ]
    for (const params of bad) expect(await r.client.call('hub.model.complete', params)).toMatchObject({ ok: false, code: 'invalid_request' })
    expect(r.fake.requests).toHaveLength(0)
  } finally { await r.stop() }
})

test('provider failures and timeouts are typed and never echo the key or endpoint', async () => {
  const r = await rig({ timeoutMs: 80 })
  try {
    r.fake.failure = new Error(`401 from ${BASE_URL}/chat/completions: invalid key ${API_KEY}`)
    const failed = await r.client.call('hub.model.complete', completion())
    expect(failed).toMatchObject({ ok: false, code: 'provider_failed' })
    expect(JSON.stringify(failed)).not.toContain(API_KEY)
    expect(JSON.stringify(failed)).not.toContain('api.openai.com')

    r.fake.failure = undefined
    r.fake.hangs = true
    expect(await r.client.call('hub.model.complete', completion())).toMatchObject({ ok: false, code: 'timeout' })
  } finally { await r.stop() }
})

test('at most two completions run at once; the third is busy and capacity returns afterwards', async () => {
  const r = await rig()
  try {
    r.fake.gated = true
    const first = r.client.call('hub.model.complete', completion())
    const second = r.client.call('hub.model.complete', completion())
    await until(() => r.fake.requests.length === 2, 'two running completions')
    expect(await r.client.call('hub.model.complete', completion())).toMatchObject({ ok: false, code: 'busy' })
    r.fake.gate.resolve()
    expect(await first).toMatchObject({ ok: true })
    expect(await second).toMatchObject({ ok: true })
    r.fake.gated = false
    expect(await r.client.call('hub.model.complete', completion())).toMatchObject({ ok: true })
  } finally { r.fake.gate.resolve(); await r.stop() }
})

test('a completion is aborted when its connection closes, and frees its slot', async () => {
  const r = await rig()
  try {
    r.fake.gated = true
    const other = await Client.open(r.socketPath)
    void other.call('hub.model.complete', completion()).catch(() => undefined)
    await until(() => r.fake.signals.length === 1, 'the completion to start')
    expect(r.fake.signals[0]!.aborted).toBe(false)
    other.socket.destroy()
    await until(() => r.fake.signals[0]!.aborted, 'the abort after disconnect')
    r.fake.gated = false
    await until(() => r.fake.closed === 1, 'the client to be closed')
    expect(await r.client.call('hub.model.complete', completion())).toMatchObject({ ok: true })
  } finally { r.fake.gate.resolve(); await r.stop() }
})

test('runtime.status advertises hub_models_v1_supported', async () => {
  const r = await rig()
  try {
    expect(await r.client.call('runtime.status')).toMatchObject({ hub_models_v1_supported: true, principal_auth_v1_supported: true })
  } finally { await r.stop() }
})

test('a hub-signed call needs the supervisor.model scope; with it all three methods work', async () => {
  const r = await rig()
  try {
    for (const [method, params] of [
      ['hub.models.catalog', {}], ['hub.models.options', { profile: 'work' }], ['hub.model.complete', completion()],
    ] as [string, JsonRpcPayload][]) {
      await expect(r.client.call(method, r.sign(method, params, ['session.read', 'session.submit']))).rejects.toThrow('supervisor.model')
      const result = await r.client.call(method, r.sign(method, params, ['supervisor.model']))
      expect(result.ok).toBe(true)
    }
    // A human principal holding the scope (the Hub owner) is accepted too, but enrollment stays closed to every hub caller.
    expect((await r.client.call('hub.models.catalog', r.sign('hub.models.catalog', {}, ['supervisor.model'], 'user'))).ok).toBe(true)
    await expect(r.client.call('hub.enroll', r.sign('hub.enroll', { hub_id: 'x', public_key_pem: 'y' }, ['supervisor.model']))).rejects.toThrow('-32001')
  } finally { await r.stop() }
})

test('hub.enroll all_projects pins the machine: hub.list reports it and the key verifies for unrelated projects', async () => {
  const r = await rig()
  try {
    const machine = generateKeyPairSync('ed25519')
    const pem = machine.publicKey.export({ type: 'spki', format: 'pem' }).toString()
    const enrolled = await r.client.call('hub.enroll', { hub_id: 'hub-all', public_key_pem: pem, all_projects: true })
    expect(enrolled).toMatchObject({ ok: true, hub: { hub_id: 'hub-all', project_dirs: ['/'], all_projects: true } })
    expect(await r.client.call('hub.enroll', { hub_id: 'hub-bad', public_key_pem: pem, all_projects: 'yes' })).toMatchObject({ ok: false })

    const listed = (await r.client.call('hub.list')).hubs as { hub_id: string; all_projects: boolean }[]
    expect(listed.find(hub => hub.hub_id === 'hub-all')?.all_projects).toBe(true)
    expect(listed.find(hub => hub.hub_id === 'hub-1')?.all_projects).toBe(false)

    // Enrolling the same key again with explicit projects keeps the whole-machine pin (merge semantics).
    const again = await r.client.call('hub.enroll', { hub_id: 'hub-all', public_key_pem: pem, all_projects: true, project_dirs: [r.root] })
    expect((again.hub as { project_dirs: string[] }).project_dirs).toContain('/')

    const verifier = new HubIdentityVerifier({ path: r.enrollments })
    const dirA = await mkdtemp(join(tmpdir(), 'xh-all-a-'))
    const dirB = await mkdtemp(join(tmpdir(), 'xh-all-b-'))
    directories.push(dirA, dirB)
    const envelope = (method: string) => signHubAuth({
      hub_id: 'hub-all', principal_id: 'xerxes', principal_kind: 'agent', display_name: 'XERXES', workspace_id: 'ws', method,
      params_hash: paramsHash({}), request_id: randomUUID(), nonce: randomUUID(), issued_at: Date.now(), expires_at: Date.now() + 60_000, scopes: ['supervisor.model'],
    }, machine.privateKey)
    for (const projectDir of [dirA, dirB]) {
      expect(verifier.verify(envelope('hub.models.catalog'), { method: 'hub.models.catalog', params: {}, projectDir })).toMatchObject({ ok: true })
    }
    // The per-project hub is still limited to its own project.
    const pinned = signHubAuth({
      hub_id: 'hub-1', principal_id: 'xerxes', principal_kind: 'agent', display_name: 'XERXES', workspace_id: 'ws', method: 'hub.models.catalog',
      params_hash: paramsHash({}), request_id: randomUUID(), nonce: randomUUID(), issued_at: Date.now(), expires_at: Date.now() + 60_000, scopes: ['supervisor.model'],
    }, r.privateKey)
    expect(verifier.verify(pinned, { method: 'hub.models.catalog', params: {}, projectDir: dirA })).toMatchObject({ ok: false, reason: 'hub_not_enrolled_for_project' })
  } finally { await r.stop() }
})
