// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  BridgeServer,
  NdjsonBridgeOutput,
  serveBridgeNdjson,
  validateBridgeModelFetchUrl,
  type BridgeModelDiscoveryPort,
  type BridgeProfileStore,
  type BridgeRuntimeInfo,
  type BridgeRuntimePort,
  type BridgeRuntimeTurnInput,
  type BridgeServerOutput,
  type LegacyBridgeEvent,
} from '../src/bridge/server.js'
import { BridgeSession, type BridgeSessionStore } from '../src/bridge/session.js'
import type { BridgeWireFrame } from '../src/bridge/wireEvents.js'
import type { ProviderProfile, SaveProfileInput } from '../src/bridge/profiles.js'
import type { StreamEvent } from '../src/streaming/events.js'

class MemorySessionStore implements BridgeSessionStore {
  readonly records = new Map<string, unknown>()
  readonly writes: Array<{ readonly record: Readonly<Record<string, unknown>>; readonly sessionId: string }> = []

  read(sessionId: string): unknown | undefined {
    return this.records.get(sessionId)
  }

  write(sessionId: string, record: Readonly<Record<string, unknown>>): void {
    this.records.set(sessionId, record)
    this.writes.push({ sessionId, record })
  }
}

class RecordingOutput implements BridgeServerOutput {
  readonly legacy: LegacyBridgeEvent[] = []
  readonly wire: BridgeWireFrame[] = []

  writeLegacy(event: LegacyBridgeEvent): void {
    this.legacy.push(event)
  }

  writeWire(frame: BridgeWireFrame): void {
    this.wire.push(frame)
  }
}

class MemoryProfiles implements BridgeProfileStore {
  private activeName: string | undefined
  private readonly profiles = new Map<string, ProviderProfile>()

  constructor(initial: readonly ProviderProfile[] = [], activeName?: string) {
    for (const profile of initial) this.profiles.set(profile.name, copyProfile(profile))
    this.activeName = activeName
  }

  active(): ProviderProfile | undefined {
    const name = this.activeName
    const profile = name === undefined ? undefined : this.profiles.get(name)
    return profile === undefined ? undefined : copyProfile(profile)
  }

  delete(name: string): boolean {
    const removed = this.profiles.delete(name)
    if (removed && this.activeName === name) this.activeName = undefined
    return removed
  }

  list(): Array<ProviderProfile & { readonly active: boolean }> {
    return [...this.profiles.values()].map(profile => ({
      ...copyProfile(profile),
      active: profile.name === this.activeName,
    }))
  }

  save(input: SaveProfileInput): ProviderProfile {
    const profile: ProviderProfile = {
      name: input.name,
      api_key: input.apiKey,
      base_url: input.baseUrl.replace(/\/+$/u, ''),
      model: input.model,
      provider: input.provider ?? 'custom',
      sampling: { ...input.sampling },
    }
    this.profiles.set(profile.name, profile)
    if (input.setActive ?? true) this.activeName = profile.name
    return copyProfile(profile)
  }

  setActive(name: string): boolean {
    if (!this.profiles.has(name)) return false
    this.activeName = name
    return true
  }

  updateActiveModel(model: string): ProviderProfile | undefined {
    const profile = this.active()
    if (!profile) return undefined
    const updated = { ...profile, model }
    this.profiles.set(updated.name, updated)
    return copyProfile(updated)
  }

  updateSampling(name: string, updates: Record<string, unknown>): ProviderProfile | undefined {
    const profile = this.profiles.get(name)
    if (!profile) return undefined
    const sampling = { ...profile.sampling }
    for (const [key, value] of Object.entries(updates)) {
      if (value === null || value === undefined) delete sampling[key]
      else sampling[key] = value
    }
    const updated = { ...profile, sampling }
    this.profiles.set(name, updated)
    return copyProfile(updated)
  }
}

class RecordingRuntime implements BridgeRuntimePort {
  readonly configurations: Readonly<Record<string, unknown>>[] = []
  readonly permissionResponses: Array<{ readonly requestId: string; readonly response: string }> = []
  readonly questionResponses: Array<{
    readonly answers: Readonly<Record<string, string>>
    readonly requestId: string
  }> = []
  cancelCalls = 0
  cancelAllCount = 0
  info: BridgeRuntimeInfo = {
    agents: [{ name: 'coder', description: 'Writes code', source: 'built-in' }],
    skills: ['review'],
    tools: [{ name: 'Read', description: 'Read a file', safe: true }],
  }
  turn: (input: BridgeRuntimeTurnInput, signal: AbortSignal) => AsyncIterable<StreamEvent> = defaultTurn

  configure(config: Readonly<Record<string, unknown>>): void {
    this.configurations.push({ ...config })
  }

  initialize(): BridgeRuntimeInfo {
    return this.info
  }

  cancel(): void {
    this.cancelCalls += 1
  }

  cancelAll(): number {
    this.cancelAllCount += 1
    return 0
  }

  respondPermission(input: { readonly requestId: string; readonly response: string }): boolean {
    this.permissionResponses.push(input)
    return input.requestId === 'permission-1'
  }

  respondQuestion(input: { readonly answers: Readonly<Record<string, string>>; readonly requestId: string }): boolean {
    this.questionResponses.push(input)
    return input.requestId === 'question-1'
  }

  streamTurn(input: BridgeRuntimeTurnInput, signal: AbortSignal): AsyncIterable<StreamEvent> {
    return this.turn(input, signal)
  }
}

class RecordingDiscovery implements BridgeModelDiscoveryPort {
  readonly calls: Array<{ readonly apiKey: string; readonly baseUrl: string; readonly provider: string }> = []
  models: readonly string[] = ['gpt-4.1', 'gpt-4.1-mini']

  fetchModels(input: {
    readonly apiKey: string
    readonly baseUrl: string
    readonly provider: string
  }): readonly string[] {
    this.calls.push(input)
    return this.models
  }
}

function bridge(options: {
  readonly discovery?: BridgeModelDiscoveryPort
  readonly output?: RecordingOutput
  readonly profiles?: BridgeProfileStore
  readonly runtime?: RecordingRuntime
  readonly sessionStore?: MemorySessionStore
  readonly wireMode?: boolean
} = {}): {
  readonly output: RecordingOutput
  readonly profiles: BridgeProfileStore
  readonly runtime: RecordingRuntime
  readonly server: BridgeServer
  readonly sessionStore: MemorySessionStore
} {
  const output = options.output ?? new RecordingOutput()
  const profiles = options.profiles ?? new MemoryProfiles()
  const runtime = options.runtime ?? new RecordingRuntime()
  const sessionStore = options.sessionStore ?? new MemorySessionStore()
  const session = new BridgeSession({
    clock: () => new Date('2026-07-13T12:00:00.000Z'),
    cwd: '/initial',
    sessionId: 'feedface',
    store: sessionStore,
  })
  const server = new BridgeServer({
    idFactory: idFactory(),
    output,
    profileStore: profiles,
    runtime,
    session,
    ...(options.discovery === undefined ? {} : { modelDiscovery: options.discovery }),
    ...(options.wireMode === undefined ? {} : { wireMode: options.wireMode }),
  })
  return { server, output, profiles, runtime, sessionStore }
}

test('bridge initialization applies a real active profile and reports real runtime inventory', async () => {
  const profiles = new MemoryProfiles([profile('team', 'gpt-4.1')], 'team')
  const { server, output, runtime } = bridge({ profiles })

  expect(await server.dispatch({ method: 'init', params: { project_dir: '/workspace' } })).toEqual({ accepted: true })
  expect(server.configuration).toMatchObject({
    model: 'gpt-4.1',
    base_url: 'https://api.openai.com/v1',
    api_key: 'secret',
    provider: 'openai',
    mode: 'code',
    plan_mode: false,
  })
  expect(server.session.cwd).toBe('/workspace')
  expect(runtime.configurations).toHaveLength(1)
  expect(output.legacy).toEqual([{
    event: 'ready',
    data: {
      model: 'gpt-4.1',
      provider: 'openai',
      tools: 1,
      permission_mode: 'accept-all',
      has_profile: true,
      skills: ['review'],
      agents: ['coder'],
    },
  }])
})

test('wire mode maps the native stream, records usage, and persists the bridge session', async () => {
  const runtime = new RecordingRuntime()
  runtime.turn = async function* (input): AsyncGenerator<StreamEvent> {
    input.state.messages.push({ role: 'user', content: input.text })
    input.state.turnCount += 1
    yield { type: 'text', text: 'hello ' }
    yield { type: 'thinking', text: 'considering' }
    yield {
      type: 'tool_start',
      call: { id: 'call-read', type: 'function', function: { name: 'Read', arguments: { path: '/repo/a.ts' } } },
    }
    input.state.toolExecutions.push({
      durationMs: 12,
      inputs: { path: '/repo/a.ts' },
      name: 'Read',
      permitted: true,
      result: 'contents',
      toolCallId: 'call-read',
    })
    yield {
      type: 'tool_end',
      result: { durationMs: 12, name: 'Read', permitted: true, result: 'contents', toolCallId: 'call-read' },
    }
    input.state.messages.push({ role: 'assistant', content: 'hello' })
    input.state.totalInputTokens += 11
    input.state.totalOutputTokens += 4
    yield {
      type: 'turn_done',
      model: 'gpt-4.1',
      toolCallsCount: 1,
      usage: { inputTokens: 11, outputTokens: 4 },
    }
  }
  const { server, output, sessionStore } = bridge({ runtime, wireMode: true })

  await server.dispatch({ method: 'init', params: { model: 'gpt-4.1', project_dir: '/repo' } })
  output.wire.length = 0
  expect(await server.dispatch({ method: 'query', params: { text: 'read the file' } })).toEqual({ accepted: true })
  await server.waitForIdle()

  expect(output.wire.map(frame => frame.method === 'event' ? frame.params.type : frame.params.type)).toEqual([
    'TurnBegin', 'TextPart', 'ThinkPart', 'ToolCall', 'ToolResult', 'TurnEnd', 'StatusUpdate',
  ])
  const toolResult = output.wire.find(frame => frame.method === 'event' && frame.params.type === 'ToolResult')
  expect(toolResult).toMatchObject({
    params: { payload: { tool_call_id: 'call-read', return_value: 'contents', duration_ms: 12 } },
  })
  expect(sessionStore.writes).toHaveLength(1)
  expect(sessionStore.writes[0]?.record).toMatchObject({
    session_id: 'feedface',
    cwd: '/repo',
    turn_count: 1,
    total_input_tokens: 11,
    total_output_tokens: 4,
    messages: [
      { role: 'user', content: 'read the file' },
      { role: 'assistant', content: 'hello' },
    ],
  })
})

test('model discovery validates public URLs and invokes only the explicit discovery port', async () => {
  const discovery = new RecordingDiscovery()
  const { server, output } = bridge({ discovery })

  expect(validateBridgeModelFetchUrl('http://127.0.0.1:8000/v1')).toBe('Private IP addresses are not allowed')
  expect(validateBridgeModelFetchUrl('claude-code://local')).toBe('URL scheme must be http or https')
  expect(validateBridgeModelFetchUrl('https://api.example.com/v1')).toBeUndefined()

  expect(await server.dispatch({
    method: 'fetch_models',
    params: { base_url: 'https://api.example.com/v1', api_key: 'key', provider: 'custom' },
  })).toEqual({ accepted: true })
  expect(discovery.calls).toEqual([{
    baseUrl: 'https://api.example.com/v1',
    apiKey: 'key',
    provider: 'custom',
  }])
  expect(output.legacy.at(-1)).toEqual({
    event: 'models_list',
    data: { base_url: 'https://api.example.com/v1', models: ['gpt-4.1', 'gpt-4.1-mini'] },
  })

  expect(await server.dispatch({ method: 'fetch_models', params: { base_url: 'http://localhost:11434' } })).toEqual({
    accepted: false,
    error: 'fetch_models blocked: Private/localhost addresses are not allowed',
  })
  expect(discovery.calls).toHaveLength(1)
})

test('initialization adopts exactly one discovered model only through the supplied discovery port', async () => {
  const discovery = new RecordingDiscovery()
  discovery.models = ['remote-only']
  const { server, output } = bridge({ discovery })

  await server.dispatch({
    method: 'init',
    params: { base_url: 'https://models.example/v1', api_key: 'key', provider: 'custom' },
  })

  expect(server.configuration.model).toBe('remote-only')
  expect(discovery.calls).toEqual([{
    baseUrl: 'https://models.example/v1',
    apiKey: 'key',
    provider: 'custom',
  }])
  expect(output.legacy.at(-1)).toMatchObject({ event: 'ready', data: { model: 'remote-only' } })
})

test('provider mutation and interaction replies go through explicit persisted/runtime ports', async () => {
  const { server, output, runtime } = bridge()
  await server.dispatch({ method: 'init', params: { model: 'gpt-4.1' } })

  expect(await server.dispatch({
    method: 'provider_save',
    params: {
      name: 'staging',
      base_url: 'https://staging.example/v1',
      api_key: 'staging-key',
      model: 'gpt-4.1-mini',
      provider: 'custom',
    },
  })).toEqual({ accepted: true })
  expect(server.configuration).toMatchObject({ model: 'gpt-4.1-mini', provider: 'custom' })
  expect(output.legacy.at(-1)).toEqual({
    event: 'provider_saved',
    data: {
      profile: {
        name: 'staging',
        api_key: 'staging-key',
        base_url: 'https://staging.example/v1',
        model: 'gpt-4.1-mini',
        provider: 'custom',
        sampling: {},
      },
      message: "Profile 'staging' saved and activated. Model: gpt-4.1-mini",
    },
  })
  expect(await server.dispatch({
    method: 'permission_response', params: { request_id: 'permission-1', response: 'approve_for_session' },
  })).toEqual({ accepted: true })
  expect(await server.dispatch({
    method: 'question_response', params: { request_id: 'question-1', answers: { answer: 'yes' } },
  })).toEqual({ accepted: true })
  expect(runtime.permissionResponses).toEqual([{ requestId: 'permission-1', response: 'approve_for_session' }])
  expect(runtime.questionResponses).toEqual([{ requestId: 'question-1', answers: { answer: 'yes' } }])
})

test('cancel aborts the server-owned signal and NDJSON output remains transport-owned', async () => {
  const runtime = new RecordingRuntime()
  let observedAbort = false
  runtime.turn = async function* (_input, signal): AsyncGenerator<StreamEvent> {
    yield { type: 'text', text: 'started' }
    await new Promise<void>(resolve => signal.addEventListener('abort', () => {
      observedAbort = true
      resolve()
    }, { once: true }))
  }
  const lines: string[] = []
  const output = new NdjsonBridgeOutput({ write: line => lines.push(line) })
  const sessionStore = new MemorySessionStore()
  const server = new BridgeServer({
    idFactory: idFactory(),
    output,
    profileStore: new MemoryProfiles(),
    runtime,
    session: new BridgeSession({
      clock: () => new Date('2026-07-13T12:00:00.000Z'),
      cwd: '/repo',
      sessionId: 'deadbeef',
      store: sessionStore,
    }),
  })

  await serveBridgeNdjson(linesInput([
    '{"method":"init","params":{"model":"gpt-4.1"}}\n',
    '{"method":"query","params":{"text":"wait"}}\n',
  ]), server)
  expect(await server.dispatch({ method: 'cancel', params: {} })).toEqual({ accepted: true })
  await server.waitForIdle()

  expect(observedAbort).toBeTrue()
  expect(runtime.cancelCalls).toBe(1)
  expect(lines.map(line => JSON.parse(line) as { event: string }).map(item => item.event)).toEqual([
    'ready', 'text_chunk', 'query_done', 'state',
  ])
})

async function* defaultTurn(input: BridgeRuntimeTurnInput, _signal: AbortSignal): AsyncGenerator<StreamEvent> {
  input.state.messages.push({ role: 'user', content: input.text })
  input.state.messages.push({ role: 'assistant', content: 'ok' })
  input.state.turnCount += 1
  yield { type: 'text', text: 'ok' }
  yield {
    type: 'turn_done',
    model: String(input.config.model ?? ''),
    toolCallsCount: 0,
    usage: { inputTokens: 1, outputTokens: 1 },
  }
}

function profile(name: string, model: string): ProviderProfile {
  return {
    name,
    api_key: 'secret',
    base_url: 'https://api.openai.com/v1',
    model,
    provider: 'openai',
    sampling: {},
  }
}

function copyProfile(value: ProviderProfile): ProviderProfile {
  return { ...value, sampling: { ...value.sampling } }
}

function idFactory(): () => string {
  let value = 0
  return () => `id-${++value}`
}

async function* linesInput(lines: readonly string[]): AsyncGenerator<string> {
  yield* lines
}
