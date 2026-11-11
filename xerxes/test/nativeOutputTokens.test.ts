// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { connect } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { LLMConfig } from '../src/core/config.js'
import { ProfileStore } from '../src/bridge/profiles.js'
import type { DaemonConfig } from '../src/daemon/config.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { runtimeConnection } from '../src/daemon/runtimeConnection.js'
import { DaemonServer } from '../src/daemon/server.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { OpenAiCompatibleClient } from '../src/llms/client.js'
import { createAgentState } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'

function config(runtime: Record<string, unknown>): DaemonConfig {
  return { runtime, channels: {}, control: {}, workspace: {}, maxConcurrentTurns: 8, projectDirectory: '.' }
}

test('OpenAI config defaults to 32768 but other native provider defaults stay unchanged', () => {
  expect(new LLMConfig({}, {}).maxTokens).toBe(32_768)
  expect(new LLMConfig({ provider: 'anthropic' }, {}).maxTokens).toBe(2_048)
})

test('OpenAI config accepts -1, preserving it when serialized, and rejects other nonpositive caps', () => {
  expect(new LLMConfig({ max_tokens: -1 }, {}).toJSON().max_tokens).toBe(-1)
  for (const value of [0, -2, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
    expect(() => new LLMConfig({ max_tokens: value }, {})).toThrow()
  }
  expect(() => new LLMConfig({ provider: 'anthropic', max_tokens: -1 }, {})).toThrow()
})

test('unconfigured connections leave the default to the OpenAI adapter and cannot carry -1 to other providers', () => {
  for (const provider of ['openai', 'custom', 'lmstudio', 'ollama']) {
    expect(runtimeConnection(config({ provider, model: 'glm-test' }), undefined)?.maxTokens).toBeUndefined()
    expect(runtimeConnection(config({ provider, model: 'glm-test', max_tokens: -1 }), undefined)?.maxTokens).toBe(-1)
  }
  for (const provider of ['openai-codex', 'anthropic', 'gemini', 'groq']) {
    expect(runtimeConnection(config({ provider, model: 'native-test' }), undefined)?.maxTokens).toBeUndefined()
    expect(runtimeConnection(config({ provider, model: 'native-test', max_tokens: -1 }), undefined)?.maxTokens).toBeUndefined()
    expect(runtimeConnection(config({ provider, model: 'native-test', max_tokens: 512 }), undefined)?.maxTokens).toBe(512)
  }
})

test('a pinned native provider never reads an OpenAI live budget or inherits its sentinel', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-native-output-scope-'))
  try {
    for (const provider of ['anthropic', 'gemini', 'openai-codex']) for (const oldCap of [undefined, -1]) {
      let reads = 0
      const sent: Array<number | undefined> = []
      const llm = { async *stream(request: import('../src/llms/client.js').CompletionRequest) { sent.push(request.maxTokens); yield { content: 'done' } } }
      const runner = new AgentTurnRunner({ model: 'native-test', llm, ...(oldCap === undefined ? {} : { maxTokens: oldCap }),
        maxTokensProvider: () => { reads += 1; return undefined },
        resolveSessionProvider: () => ({ llm, providerOverrides: { provider }, maxOutputTokens: () => 64_000 }),
      })
      const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, model: 'native-test', sessionDirectory: join(directory, 'sessions') })
      try {
        const session = await runtime.openSession(`${provider}-${oldCap}`)
        await runtime.submitTurn(session.sessionKey, 'hello', () => {})
        expect(reads).toBe(0)
        expect(sent).toEqual([64_000])
      } finally { await runtime.shutdown() }
    }
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('a streaming no-cap request is bounded by context room without ever sending a negative wire cap', async () => {
  const requests: Array<number | undefined> = []
  for await (const _event of runTurn({ model: 'test-model', maxTokens: -1, state: createAgentState([]), userMessage: 'hello' }, {
    roundOutputRoom: () => 8_192,
    llm: { async *stream(request) { requests.push(request.maxTokens); yield { content: 'done' } } },
  })) { /* consume */ }
  expect(requests).toEqual([8_192])
})

async function socketClient(path: string) {
  const socket = connect({ path })
  await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) })
  socket.setEncoding('utf8')
  let sequence = 0, buffer = ''
  const pending = new Map<number, ReturnType<typeof Promise.withResolvers<unknown>>>()
  socket.on('data', chunk => {
    buffer += chunk
    for (let newline = buffer.indexOf('\n'); newline >= 0; newline = buffer.indexOf('\n')) {
      const line = buffer.slice(0, newline); buffer = buffer.slice(newline + 1)
      if (!line.trim()) continue
      const frame: unknown = JSON.parse(line)
      if (!frame || typeof frame !== 'object' || !('id' in frame) || typeof frame.id !== 'number') continue
      const waiter = pending.get(frame.id)
      pending.delete(frame.id)
      if (frame && 'error' in frame) waiter?.reject(new Error(JSON.stringify(frame.error)))
      else waiter?.resolve('result' in frame ? frame.result : undefined)
    }
  })
  return { socket, rpc(method: string, params: Record<string, unknown>) {
    const id = ++sequence, result = Promise.withResolvers<unknown>()
    pending.set(id, result)
    socket.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
    return result.promise
  } }
}

test('/sampling validates and saves the selected session profile, not a different global Codex profile', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-pinned-output-setting-'))
  const profiles = new ProfileStore(join(directory, 'profiles.json'))
  profiles.save({ name: 'pinned-openai', provider: 'openai', model: 'glm-pinned', baseUrl: 'https://offline.invalid/v1', apiKey: '' })
  profiles.save({ name: 'global-codex', provider: 'openai-codex', model: 'gpt-global', baseUrl: 'https://codex-offline.invalid', apiKey: '' })
  profiles.setActive('global-codex')
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions'), runtimeSettings: { provider: 'openai-codex', model: 'gpt-global' } })
  const server = new DaemonServer({ runtime, socketPath: join(directory, 'daemon.sock'), projectDirectory: directory, profileStore: profiles, codexModelCatalog: async () => [] })
  await server.start()
  const client = await socketClient(join(directory, 'daemon.sock'))
  try {
    await client.rpc('initialize', { session_key: 'pinned-output', project_dir: directory })
    const session = await runtime.openSession('pinned-output', directory)
    session.model = 'glm-pinned'; session.modelPinned = true; session.metadata.provider_profile = 'pinned-openai'
    expect(await client.rpc('slash', { command: '/sampling max_tokens -1' })).toMatchObject({ ok: true, sampling: { max_tokens: -1 } })
    expect(profiles.get('pinned-openai')?.sampling.max_tokens).toBe(-1)
    expect(profiles.get('global-codex')?.sampling.max_tokens).toBeUndefined()
    session.model = 'gpt-global'; session.metadata.provider_profile = 'global-codex'
    expect(await client.rpc('slash', { command: '/sampling max_tokens -1' })).toMatchObject({ ok: false })
  } finally {
    client.socket.destroy(); await server.stop(); await runtime.shutdown(); await rm(directory, { recursive: true, force: true })
  }
})

test('/sampling updates the actual next OpenAI request even while the existing turn is waiting on a tool', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-live-output-'))
  const started = Promise.withResolvers<void>(), release = Promise.withResolvers<void>()
  const registry = new ToolRegistry()
  registry.register({ type: 'function', function: { name: 'WaitForSampling', description: 'Wait in an offline test.', parameters: { type: 'object', properties: {} } } }, async () => {
    started.resolve(); await release.promise; return 'ready'
  })
  const payloads: Record<string, unknown>[] = []
  const llm = new OpenAiCompatibleClient({ providerName: 'openai', apiKey: 'offline-fixture', baseUrl: 'http://offline.invalid/v1', fetchImplementation: async (_url, init) => {
    const body: unknown = JSON.parse(String(init?.body))
    if (!body || typeof body !== 'object' || Array.isArray(body)) throw new Error('Expected request object')
    payloads.push(Object.fromEntries(Object.entries(body)))
    const first = payloads.length === 1
    const delta = first ? { tool_calls: [{ index: 0, id: 'sampling-wait', type: 'function', function: { name: 'WaitForSampling', arguments: '{}' } }] } : { content: 'done' }
    return new Response(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: first ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'Content-Type': 'text/event-stream' } })
  } })
  let runtime: InMemoryDaemonRuntime
  runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions'), runtimeSettings: { model: 'glm-test', provider: 'openai', max_tokens: 512, permission_mode: 'accept-all' }, turnRunnerFactory: settings => new AgentTurnRunner({
    model: 'glm-test', llm, providerOverrides: { provider: 'openai' }, ...(typeof settings.max_tokens === 'number' ? { maxTokens: settings.max_tokens } : {}),
    maxTokensProvider: () => { const value = runtime.settings().max_tokens; return typeof value === 'number' ? value : undefined },
    permissionMode: 'accept-all', tools: registry.definitions(), toolExecutor: registry,
  }) })
  const server = new DaemonServer({ runtime, socketPath: join(directory, 'daemon.sock'), projectDirectory: directory, profileStore: new ProfileStore(join(directory, 'profiles.json')) })
  await server.start()
  const client = await socketClient(join(directory, 'daemon.sock'))
  let turn: Promise<void> | undefined
  try {
    await client.rpc('initialize', { session_key: 'sampling-owner', project_dir: directory })
    const session = await runtime.openSession('sampling-owner', directory)
    turn = runtime.submitTurn(session.sessionKey, 'Call WaitForSampling once, then finish.', () => {})
    await started.promise
    expect(payloads[0]?.max_tokens).toBe(512)
    expect(await client.rpc('slash', { command: '/sampling max_tokens 2000' })).toMatchObject({ ok: true, sampling: { max_tokens: 2_000 } })
    release.resolve(); await turn
    expect(payloads.map(payload => payload.max_tokens)).toEqual([512, 2_000])
    expect(await client.rpc('slash', { command: '/sampling max_tokens -1' })).toMatchObject({ ok: true, sampling: { max_tokens: -1 } })
    await runtime.submitTurn(session.sessionKey, 'finish', () => {})
    expect(payloads.at(-1)).not.toHaveProperty('max_tokens')
    expect(await client.rpc('slash', { command: '/sampling max_tokens 0' })).toMatchObject({ ok: false })
    expect(await client.rpc('slash', { command: '/sampling reset' })).toMatchObject({ ok: true, sampling: { max_tokens: 32_768 } })
    expect(runtime.settings()).not.toHaveProperty('max_tokens')
    await runtime.submitTurn(session.sessionKey, 'finish', () => {})
    expect(payloads.at(-1)?.max_tokens).toBe(32_768)
  } finally {
    release.resolve(); await turn?.catch(() => undefined)
    client.socket.destroy(); await server.stop(); await runtime.shutdown()
    await rm(directory, { recursive: true, force: true })
  }
}, 15_000)
