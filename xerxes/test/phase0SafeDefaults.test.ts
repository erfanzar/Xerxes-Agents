// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { AcpAgentRunner } from '../src/acp/runner.js'
import { AcpServer } from '../src/acp/server.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { HookRunner } from '../src/extensions/hooks.js'
import {
  createLlmClient,
  OpenAiCompatibleClient,
  ProviderConnectTimeoutError,
  type CompletionRequest,
  type LlmClient,
  type LlmDelta,
} from '../src/llms/client.js'
import {
  ambientApiKeyAllowed,
  defaultMaxNetworkRetries,
  getApiKey,
  isLoopbackBaseUrl,
  parseProviderNetworkConfig,
} from '../src/llms/providerRegistry.js'
import { classifyError, ErrorKind, isRecoverableNetworkError } from '../src/runtime/errorClassifier.js'
import {
  extractNoSaveOption,
  newOneShotSessionId,
  OneShotSessionRecorder,
  oneShotTranscriptStore,
  parseOneShotSaveConfig,
  userHookRunnerForCwd,
} from '../src/runtime/oneShotSession.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'
import { Xerxes } from '../src/xerxes.js'

const cleanups: (() => Promise<void> | void)[] = []
afterEach(async () => {
  while (cleanups.length) await cleanups.pop()!()
})

async function collect(events: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const result: StreamEvent[] = []
  for await (const event of events) result.push(event)
  return result
}

async function closedLoopbackPort(): Promise<number> {
  const server = Bun.serve({ port: 0, hostname: '127.0.0.1', fetch: () => new Response('') })
  const port = server.port!
  await server.stop(true)
  return port
}

function hungServer(): number {
  const server = Bun.serve({ port: 0, hostname: '127.0.0.1', fetch: () => new Promise<Response>(() => {}) })
  cleanups.push(() => server.stop(true))
  return server.port!
}

async function tempDir(prefix: string): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), prefix))
  cleanups.push(() => rm(directory, { recursive: true, force: true }))
  return directory
}

function withEnv(name: string, value: string | undefined): void {
  const previous = process.env[name]
  if (value === undefined) delete process.env[name]
  else process.env[name] = value
  cleanups.push(() => {
    if (previous === undefined) delete process.env[name]
    else process.env[name] = previous
  })
}

// --- 1. dead endpoint fails fast -------------------------------------------

test('Bun refused connections and connect timeouts classify as network errors', () => {
  const refused = Object.assign(new Error('Unable to connect. Is the computer able to access the url?'), { code: 'ConnectionRefused' })
  expect(classifyError(refused).kind).toBe(ErrorKind.PROVIDER_DOWN)
  expect(isRecoverableNetworkError(refused)).toBe(true)
  const timeout = new ProviderConnectTimeoutError('127.0.0.1:9', 15_000)
  expect(timeout.message).toBe('cannot reach 127.0.0.1:9 within 15s')
  expect(isRecoverableNetworkError(timeout)).toBe(true)
  expect(classifyError(timeout).retryable).toBe(true)
})

test('network retry defaults: 3 for loopback, 6 otherwise; config parser validates', () => {
  for (const url of ['http://localhost:8080/v1', 'http://127.0.0.5:1/v1', 'http://[::1]:11434/v1', 'unix:/tmp/x.sock']) {
    expect(isLoopbackBaseUrl(url)).toBe(true)
    expect(defaultMaxNetworkRetries(url)).toBe(3)
  }
  expect(defaultMaxNetworkRetries('https://api.openai.com/v1')).toBe(6)
  expect(defaultMaxNetworkRetries('http://10.0.0.2:8000/v1')).toBe(6)
  expect(defaultMaxNetworkRetries()).toBe(6)
  expect(parseProviderNetworkConfig({}, {})).toEqual({ connectTimeoutMs: 15_000 })
  expect(parseProviderNetworkConfig({ provider: { connect_timeout_ms: 2000, max_network_retries: 1 } }, {}))
    .toEqual({ connectTimeoutMs: 2000, maxNetworkRetries: 1 })
  expect(parseProviderNetworkConfig({ provider: { connect_timeout_ms: 2000 } }, { XERXES_PROVIDER_CONNECT_TIMEOUT_MS: '500' }).connectTimeoutMs).toBe(500)
  expect(() => parseProviderNetworkConfig({ provider: { connect_timeout_ms: -1 } }, {})).toThrow(/provider\.connect_timeout_ms/)
  expect(() => parseProviderNetworkConfig({}, { XERXES_PROVIDER_CONNECT_TIMEOUT_MS: 'soon' })).toThrow(/XERXES_PROVIDER_CONNECT_TIMEOUT_MS/)
  // Existing configs use `runtime.provider` as the provider NAME; that must never break startup.
  expect(parseProviderNetworkConfig({ provider: 'openrouter' }, {})).toEqual({ connectTimeoutMs: 15_000 })
  expect(parseProviderNetworkConfig({ provider: 'openrouter', provider_network: { connect_timeout_ms: 3000, max_network_retries: 2 } }, {}))
    .toEqual({ connectTimeoutMs: 3000, maxNetworkRetries: 2 })
  expect(() => parseProviderNetworkConfig({ provider_network: 'fast' }, {})).toThrow(/provider_network/)
})

test('a refused loopback port ends the turn after the network retry cap', async () => {
  const port = await closedLoopbackPort()
  const llm = new OpenAiCompatibleClient({ providerName: 'custom', apiKey: 'k', baseUrl: `http://127.0.0.1:${port}/v1` })
  const delays: number[] = []
  const started = performance.now()
  const events = await collect(runTurn({ model: 'local-model', state: createAgentState(), userMessage: 'hi' }, {
    llm,
    maxNetworkRetries: defaultMaxNetworkRetries(`http://127.0.0.1:${port}/v1`),
    delay: async ms => { delays.push(ms) },
  }))
  expect(performance.now() - started).toBeLessThan(5_000)
  const retries = events.filter(event => event.type === 'provider_retry')
  expect(retries).toHaveLength(4)
  expect(retries.slice(0, 3).every(event => event.type === 'provider_retry' && event.maxAttempts === 0 && !event.final)).toBe(true)
  expect(retries.at(-1)).toMatchObject({ final: true })
  expect(delays).toHaveLength(3)
  const text = events.filter(event => event.type === 'text').map(event => event.type === 'text' ? event.text : '').join('')
  expect(text).toContain('gave up after 3 network retries')
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'provider_failed' })
})

test('a server that accepts but never answers fails within the connect timeout', async () => {
  const port = hungServer()
  const llm = new OpenAiCompatibleClient({ providerName: 'custom', apiKey: 'k', baseUrl: `http://127.0.0.1:${port}/v1`, connectTimeoutMs: 150 })
  const started = performance.now()
  const events = await collect(runTurn({ model: 'local-model', state: createAgentState(), userMessage: 'hi' }, {
    llm,
    maxNetworkRetries: 1,
    delay: async () => {},
  }))
  expect(performance.now() - started).toBeLessThan(3_000)
  const final = events.filter(event => event.type === 'provider_retry').at(-1)
  expect(final).toMatchObject({ final: true, error: `cannot reach 127.0.0.1:${port} within 0.15s` })
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'provider_failed' })
})

test('the caller abort wins over the connect timeout and is reported as cancellation', async () => {
  const port = hungServer()
  const llm = new OpenAiCompatibleClient({ providerName: 'custom', apiKey: 'k', baseUrl: `http://127.0.0.1:${port}/v1`, connectTimeoutMs: 5_000 })
  const controller = new AbortController()
  setTimeout(() => controller.abort(new DOMException('stop', 'AbortError')), 50)
  const started = performance.now()
  const events = await collect(runTurn({ model: 'local-model', state: createAgentState(), userMessage: 'hi' }, {
    llm,
    delay: async () => {},
  }, controller.signal))
  expect(performance.now() - started).toBeLessThan(2_000)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'aborted' })
  expect(events.some(event => event.type === 'text')).toBe(false)
})

test('non-streaming completions are not bounded by the connect timeout', async () => {
  const fetches: RequestInit[] = []
  const llm = new OpenAiCompatibleClient({
    providerName: 'custom', apiKey: 'k', baseUrl: 'http://127.0.0.1:1/v1', connectTimeoutMs: 20,
    fetchImplementation: async (_input, init) => {
      fetches.push(init ?? {})
      await Bun.sleep(60)
      return Response.json({ choices: [{ message: { content: 'slow but fine' }, finish_reason: 'stop' }] })
    },
  })
  const result = await llm.complete({ model: 'm', messages: [{ role: 'user', content: 'hi' }] })
  expect(result.content).toBe('slow but fine')
})

// --- 2. no ambient key to custom hosts --------------------------------------

test('environment keys go only to their provider\'s own host', () => {
  const env = {
    OPENAI_API_KEY: 'sk-openai', OPENROUTER_API_KEY: 'sk-or', ANTHROPIC_API_KEY: 'sk-ant',
    KIMI_CODE_API_KEY: 'sk-kimi', ZHIPU_API_KEY: 'sk-zai', CUSTOM_API_KEY: 'sk-custom',
  }
  // No override: today's behavior.
  expect(getApiKey('openai', {}, env)).toBe('sk-openai')
  // Own hosts keep their keys.
  expect(getApiKey('openai', {}, env, 'https://api.openai.com/v1')).toBe('sk-openai')
  expect(getApiKey('openrouter', {}, env, 'https://openrouter.ai/api/v1')).toBe('sk-or')
  expect(getApiKey('anthropic', {}, env, 'https://api.anthropic.com')).toBe('sk-ant')
  expect(getApiKey('kimi-code', {}, env, 'https://api.kimi.com/coding/v1')).toBe('sk-kimi')
  expect(getApiKey('zhipu', {}, env, 'https://api.z.ai/api/coding/paas/v4')).toBe('sk-zai')
  // A custom host that fell back to the openai route gets nothing.
  expect(getApiKey('openai', {}, env, 'http://my-box.lan:8000/v1')).toBe('')
  expect(getApiKey('anthropic', {}, env, 'https://proxy.example.com')).toBe('')
  // Providers without a registry default send their own variable.
  expect(getApiKey('custom', {}, env, 'http://my-box.lan:8000/v1')).toBe('sk-custom')
  // Explicit configuration and the escape hatch.
  expect(getApiKey('openai', { openai_api_key: 'explicit' }, env, 'http://my-box.lan:8000/v1')).toBe('explicit')
  expect(getApiKey('openai', {}, { ...env, XERXES_ALLOW_AMBIENT_KEY: '1' }, 'http://my-box.lan:8000/v1')).toBe('sk-openai')
  expect(ambientApiKeyAllowed('ollama', 'http://localhost:11434/v1', {})).toBe(true)
})

test('a custom base_url never receives OPENAI_API_KEY and a 401 explains why', async () => {
  withEnv('OPENAI_API_KEY', 'sk-ambient-secret')
  withEnv('XERXES_ALLOW_AMBIENT_KEY', undefined)
  const seen: (string | null)[] = []
  const fetchImplementation = async (_input: RequestInfo | URL, init?: RequestInit) => {
    seen.push(new Headers(init?.headers).get('authorization'))
    return new Response('{"error":"unauthorized"}', { status: 401 })
  }
  const llm = createLlmClient('some-model', { provider: 'openai', base_url: 'http://my-box.lan:8000/v1' }, { fetchImplementation })
  const failure = await (async () => {
    for await (const _delta of llm.stream({ model: 'some-model', messages: [{ role: 'user', content: 'hi' }] })) { /* drain */ }
  })().catch((error: unknown) => error)
  expect(seen).toEqual([null])
  expect(String((failure as Error).message)).toContain('withheld the ambient OPENAI_API_KEY from my-box.lan:8000')
  expect(String((failure as Error).message)).toContain('XERXES_ALLOW_AMBIENT_KEY=1')

  const explicit = createLlmClient('some-model', { provider: 'openai', base_url: 'http://my-box.lan:8000/v1', api_key: 'sk-profile' }, { fetchImplementation })
  await (async () => {
    for await (const _delta of explicit.stream({ model: 'some-model', messages: [{ role: 'user', content: 'hi' }] })) { /* drain */ }
  })().catch(() => undefined)
  expect(seen.at(-1)).toBe('Bearer sk-profile')

  const own = createLlmClient('some-model', { provider: 'openai' }, { fetchImplementation })
  await (async () => {
    for await (const _delta of own.stream({ model: 'some-model', messages: [{ role: 'user', content: 'hi' }] })) { /* drain */ }
  })().catch(() => undefined)
  expect(seen.at(-1)).toBe('Bearer sk-ambient-secret')
})

// --- 3. hooks on every entry point -----------------------------------------

const WRITE_FILE: ToolDefinition = {
  type: 'function',
  function: { name: 'WriteFile', description: 'Write a file.', parameters: { type: 'object', required: ['path', 'content'] } },
}

class WriteThenTextClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    if (request.messages.some(message => message.role === 'tool')) {
      yield { content: 'Done.', usage: { inputTokens: 1, outputTokens: 1 } }
      return
    }
    yield {
      toolCalls: [{ id: 'call-write', type: 'function', function: { name: 'WriteFile', arguments: { path: 'note.txt', content: 'hello' } } }],
      usage: { inputTokens: 1, outputTokens: 1 },
    }
  }
}

test('the user\'s configured before_tool_call shell hook fires in a one-shot turn', async () => {
  const home = await tempDir('xerxes-hooks-home-')
  const workspace = await tempDir('xerxes-hooks-ws-')
  const log = join(home, 'hook.json')
  await writeFile(join(home, 'config.yaml'), `hooks:\n  before_tool_call:\n    - command: "cat > '${log}'"\n`)
  const hookRunner = userHookRunnerForCwd({ home, environment: {} })(workspace)
  const registry = new ToolRegistry()
  registry.register(WRITE_FILE, inputs => `wrote ${inputs.path}`)
  const events = await collect(runTurn({
    model: 'gpt-4o', state: createAgentState(), userMessage: 'write', permissionMode: 'accept-all', tools: [WRITE_FILE],
  }, { llm: new WriteThenTextClient(), hookRunner, toolExecutor: registry }))
  expect(events.at(-1)?.type).toBe('turn_done')
  const payload = JSON.parse(await readFile(log, 'utf8')) as Record<string, unknown>
  expect(payload).toMatchObject({ hook_point: 'before_tool_call', name: 'WriteFile', toolCallId: 'call-write' })
})

test('ACP runs the session\'s hooks before tool calls', async () => {
  const seen: { cwd: string; name: unknown }[] = []
  const registry = new ToolRegistry()
  registry.register(WRITE_FILE, inputs => `wrote ${inputs.path}`)
  const runner = new AcpAgentRunner({
    llm: new WriteThenTextClient(),
    model: 'gpt-4o',
    defaultPermissionMode: 'accept-all',
    tools: [WRITE_FILE],
    toolExecutor: registry,
    hookRunnerForCwd: cwd => {
      const hooks = new HookRunner()
      hooks.register('before_tool_call', payload => { seen.push({ cwd, name: payload.name }) })
      return hooks
    },
  })
  const server = new AcpServer({ runner })
  const sessionId = String(server.openSession('/workspace').session_id)
  const session = server.sessions.get(sessionId)!
  const result = await runner.runPrompt({ session, text: 'write the note' })
  expect(result).toMatchObject({ ok: true })
  expect(seen).toEqual([{ cwd: '/workspace', name: 'WriteFile' }])
})

test('the SDK facade runs an explicitly supplied hook runner and none otherwise', async () => {
  const seen: unknown[] = []
  const hookRunner = new HookRunner()
  hookRunner.register('before_tool_call', payload => { seen.push(payload.name) })
  const registry = new ToolRegistry()
  registry.register(WRITE_FILE, inputs => `wrote ${inputs.path}`)
  const xerxes = new Xerxes({
    llm: new WriteThenTextClient(), model: 'gpt-4o', coreTools: false, toolRegistry: registry,
    permissionMode: 'accept-all', hookRunner,
  })
  await xerxes.run('write the note')
  expect(seen).toEqual(['WriteFile'])
})

// --- 4. one-shot runs are saved --------------------------------------------

test('one-shot save settings: --no-save, runtime.oneshot.save and env', () => {
  expect(extractNoSaveOption(['--no-save', 'hello'])).toEqual({ noSave: true, rest: ['hello'] })
  expect(extractNoSaveOption(['hello', '--', '--no-save'])).toEqual({ noSave: false, rest: ['hello', '--', '--no-save'] })
  expect(parseOneShotSaveConfig({}, {})).toBe(true)
  expect(parseOneShotSaveConfig({ oneshot: { save: false } }, {})).toBe(false)
  expect(parseOneShotSaveConfig({ oneshot: { save: false } }, { XERXES_ONESHOT_SAVE: '1' })).toBe(true)
  expect(() => parseOneShotSaveConfig({ oneshot: { save: 'no' } }, {})).toThrow(/oneshot\.save/)
  expect(newOneShotSessionId()).toMatch(/^[0-9a-f]{12}$/)
})

test('a saved one-shot run loads as a resumable daemon session', async () => {
  const home = await tempDir('xerxes-oneshot-home-')
  const cwd = await tempDir('xerxes-oneshot-cwd-')
  const sessionId = newOneShotSessionId()
  const recorder = new OneShotSessionRecorder({
    store: oneShotTranscriptStore(home, cwd),
    sessionId, cwd, agentId: 'default', model: 'gpt-4o', agentsRoot: join(home, 'agents'),
    permissionMode: 'accept-all', systemPrompt: 'SYSTEM PROMPT',
  })
  const state = createAgentState()
  const llm: LlmClient = { async *stream() { yield { content: 'One-shot answer.', usage: { inputTokens: 2, outputTokens: 3 } } } }
  for await (const event of runTurn({ model: 'gpt-4o', state, userMessage: 'question', systemPrompt: 'SYSTEM PROMPT' }, { llm })) {
    if (event.type === 'turn_done') recorder.schedule(state)
  }
  await recorder.flush(state)

  const runtime = new InMemoryDaemonRuntime(undefined, {
    model: 'gpt-4o', currentProjectDirectory: cwd, sessionDirectory: join(home, 'sessions'), workspaceRoot: join(home, 'agents'),
  })
  cleanups.push(() => runtime.shutdown())
  const resumed = await runtime.openSession(sessionId, undefined, { resume: true, cwd })
  expect(resumed.id).toBe(sessionId)
  expect(resumed.messages.map(message => [message.role, message.content])).toEqual([
    ['user', 'question'],
    ['assistant', 'One-shot answer.'],
  ])
  expect(resumed.turnCount).toBe(1)
})

test('an interrupted one-shot run is saved with its prompt', async () => {
  const home = await tempDir('xerxes-oneshot-int-')
  const cwd = await tempDir('xerxes-oneshot-intcwd-')
  const sessionId = newOneShotSessionId()
  const store = oneShotTranscriptStore(home, cwd)
  const recorder = new OneShotSessionRecorder({ store, sessionId, cwd, agentId: 'default', model: 'gpt-4o', agentsRoot: join(home, 'agents') })
  const state = createAgentState()
  const controller = new AbortController()
  const llm: LlmClient = { async *stream(_request, signal) {
    yield { content: 'partial ' }
    controller.abort()
    signal?.throwIfAborted()
  } }
  await collect(runTurn({ model: 'gpt-4o', state, userMessage: 'long task' }, { llm }, controller.signal))
  await recorder.flush(state)
  const saved = await store.load(sessionId, { currentProjectDirectory: cwd })
  expect(saved?.messages[0]).toMatchObject({ role: 'user', content: 'long task' })
  expect(saved?.metadata.session_source).toBe('oneshot')
})
