// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { DaemonSession } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { LlmClient, LlmDelta } from '../src/llms/client.js'
import { engineRecoveryFor, isEngineFailure, type RecoverableEngine } from '../src/local/engineRecovery.js'
import { createLmStudioEngine, type LmStudioPorts, type LmsRunResult } from '../src/local/lmstudio.js'
import { createLocalServer, type ServerPorts } from '../src/local/server.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'

test('an engine failure is a dropped or refused connection, or LM Studio saying the model is gone', () => {
  expect(isEngineFailure(Object.assign(new Error('Unable to connect. Is the computer able to access the url?'), { code: 'ConnectionRefused' }))).toBe(true)
  expect(isEngineFailure(new Error('The model has crashed without additional information. (Exit code: null)'))).toBe(true)
  expect(isEngineFailure(new Error('No models loaded. Please load a model in the developer page or use the `lms load` command.'))).toBe(true)
  expect(isEngineFailure(new Error('Model ternary-bonsai-2-27b-mlx is not loaded'))).toBe(true)
  expect(isEngineFailure(Object.assign(new Error('invalid api key'), { status: 401 }))).toBe(false)
  expect(isEngineFailure(new Error('maximum context length exceeded'))).toBe(false)
})

test('only a loopback route gets a recovery hook, and it asks each engine for that port and model', async () => {
  const asked: unknown[] = []
  const engine: RecoverableEngine = {
    async recover(target) {
      asked.push(target)
      return { recovered: true, message: 'reloaded' }
    },
  }
  expect(engineRecoveryFor([engine], { baseUrl: 'https://api.openai.com/v1', model: 'gpt-4o' })).toBeUndefined()
  expect(engineRecoveryFor([engine], { baseUrl: undefined, model: 'm' })).toBeUndefined()
  const hook = engineRecoveryFor([engine], { baseUrl: 'http://127.0.0.1:1234/v1', model: 'bonsai' })
  expect(await hook?.()).toBe('reloaded')
  expect(asked).toEqual([{ port: 1234, model: 'bonsai' }])
})

test('LM Studio recovery reloads a crashed model Xerxes loaded, with its recorded window, and keeps ownership', async () => {
  const stateDir = await mkdtemp(join(tmpdir(), 'xerxes-lms-recover-'))
  try {
    const lms = fakeLmStudio()
    const engine = createLmStudioEngine(lms.ports, { lmsPath: '/bin/lms', stateDir, pollMs: 1 })
    await engine.start({ modelKey: 'bonsai-key', identifier: 'bonsai', contextLength: 16_384 })
    expect(lms.commands.filter(argv => argv[1] === 'load')).toHaveLength(1)

    // Another port or model is not ours; a loaded model did not crash.
    expect(await engine.recover({ port: 9999, model: 'bonsai' })).toEqual({ recovered: false, reason: 'other-endpoint' })
    expect(await engine.recover({ port: 1234, model: 'other' })).toEqual({ recovered: false, reason: 'other-endpoint' })
    expect(await engine.recover({ port: 1234, model: 'bonsai' })).toEqual({ recovered: false, reason: 'still-running' })

    lms.loaded.clear() // the model ran out of memory and LM Studio unloaded it
    const result = await engine.recover({ port: 1234, model: 'bonsai' })
    expect(result.recovered).toBe(true)
    expect(result.recovered && result.message).toContain('16384-token window')
    const loads = lms.commands.filter(argv => argv[1] === 'load')
    expect(loads).toHaveLength(2)
    expect(loads[1]).toEqual(expect.arrayContaining(['bonsai-key', '--context-length', '16384', '--identifier', 'bonsai']))
    const state = JSON.parse(await readFile(join(stateDir, 'lmstudio.json'), 'utf8')) as Record<string, unknown>
    // Xerxes started the server the first time, so `stop` must still stop it.
    expect(state).toMatchObject({ startedServer: true, loadedModel: true, contextLength: 16_384 })
  } finally {
    await rm(stateDir, { recursive: true, force: true })
  }
})

test('LM Studio recovery leaves a model the user loaded alone', async () => {
  const stateDir = await mkdtemp(join(tmpdir(), 'xerxes-lms-recover-'))
  try {
    const lms = fakeLmStudio()
    const engine = createLmStudioEngine(lms.ports, { lmsPath: '/bin/lms', stateDir, pollMs: 1 })
    expect(await engine.recover({ port: 1234, model: 'bonsai' })).toEqual({ recovered: false, reason: 'not-managed' })
    expect(lms.commands).toEqual([])
  } finally {
    await rm(stateDir, { recursive: true, force: true })
  }
})

test('llama-server recovery restarts a server that exited, with the same model, port and window', async () => {
  const stateDir = await mkdtemp(join(tmpdir(), 'xerxes-llama-recover-'))
  try {
    const server = fakeLlamaServer()
    const engine = createLocalServer(server.ports, { llamaServerPath: '/bin/llama-server', stateDir, pollMs: 1, graceMs: 1 })
    await engine.start({ modelPath: '/models/bonsai.gguf', port: 11581, ctxSize: 32_768 })
    expect(await engine.recover({ port: 11581, model: 'bonsai' })).toEqual({ recovered: false, reason: 'still-running' })
    expect(await engine.recover({ port: 8080, model: 'bonsai' })).toEqual({ recovered: false, reason: 'other-endpoint' })

    server.kill()
    const result = await engine.recover({ port: 11581, model: 'bonsai' })
    expect(result.recovered).toBe(true)
    expect(result.recovered && result.message).toContain('32768-token window')
    expect(server.spawned).toHaveLength(2)
    expect(server.spawned[1]).toEqual(expect.arrayContaining(['-m', '/models/bonsai.gguf', '--port', '11581', '--ctx-size', '32768']))
  } finally {
    await rm(stateDir, { recursive: true, force: true })
  }
})

test('llama-server recovery does not guess a window it never recorded', async () => {
  const stateDir = await mkdtemp(join(tmpdir(), 'xerxes-llama-recover-'))
  try {
    const server = fakeLlamaServer()
    // A state file written before Xerxes recorded the window.
    await writeFile(join(stateDir, 'server.json'), JSON.stringify({
      pid: 4242, port: 11581, model: '/models/bonsai.gguf', startedAt: '2026-11-01T00:00:00.000Z', logPath: join(stateDir, 'log'), bin: '/bin/llama-server',
    }))
    const engine = createLocalServer(server.ports, { llamaServerPath: '/bin/llama-server', stateDir, pollMs: 1, graceMs: 1 })
    expect(await engine.recover({ port: 11581, model: 'bonsai' })).toEqual({ recovered: false, reason: 'not-managed' })
    expect(server.spawned).toEqual([])
  } finally {
    await rm(stateDir, { recursive: true, force: true })
  }
})

test('the loop reloads a crashed engine once, retries the step, and says so', async () => {
  let calls = 0
  const llm: LlmClient = {
    async *stream(): AsyncGenerator<LlmDelta> {
      calls += 1
      if (calls === 1) throw new Error('The model has crashed without additional information. (Exit code: null)')
      yield { content: 'answer' }
    },
  }
  let recoveries = 0
  const events: StreamEvent[] = []
  for await (const event of runTurn(
    { model: 'bonsai', state: createAgentState([]), userMessage: 'go' },
    {
      llm,
      recoverEngine: async () => {
        recoveries += 1
        return 'Reloaded bonsai with its 16384-token window.'
      },
      delay: async () => undefined,
    },
  )) events.push(event)

  expect(recoveries).toBe(1)
  expect(calls).toBe(2)
  const retry = events.find(event => event.type === 'provider_retry')
  expect(retry).toMatchObject({ final: false, delay: 0 })
  expect(retry?.type === 'provider_retry' && retry.error).toContain('Reloaded bonsai with its 16384-token window.')
  expect(events.filter(event => event.type === 'text').map(event => event.type === 'text' ? event.text : '').join('')).toBe('answer')
})

test('the loop recovers at most once per turn and reports a failed reload', async () => {
  const llm: LlmClient = {
    // eslint-disable-next-line require-yield
    async *stream(): AsyncGenerator<LlmDelta> {
      throw new Error('No models loaded. Please load a model in the developer page or use the `lms load` command.')
    },
  }
  let recoveries = 0
  const events: StreamEvent[] = []
  for await (const event of runTurn(
    { model: 'bonsai', state: createAgentState([]), userMessage: 'go' },
    {
      llm,
      recoverEngine: async () => {
        recoveries += 1
        throw new Error('`lms load` exited with code 1.')
      },
      retryDelays: [],
      delay: async () => undefined,
    },
  )) events.push(event)

  expect(recoveries).toBe(1)
  const retries = events.filter(event => event.type === 'provider_retry')
  expect(retries.at(-1)).toMatchObject({ final: true })
  expect(JSON.stringify(retries)).toContain('could not be reloaded: `lms load` exited with code 1.')
})

test('a daemon turn hands the route to the recovery hook and finishes after a reload', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-recover-turn-'))
  try {
    let calls = 0
    const llm: LlmClient = {
      async *stream(): AsyncGenerator<LlmDelta> {
        calls += 1
        if (calls === 1) throw new Error('Model bonsai is not loaded')
        yield { content: 'done' }
      },
    }
    const routes: unknown[] = []
    const runner = new AgentTurnRunner({
      llm,
      model: 'bonsai',
      providerOverrides: { provider: 'custom', base_url: 'http://127.0.0.1:1234/v1' },
      recoverEngineFor: route => {
        routes.push(route)
        return async () => 'Reloaded bonsai.'
      },
    })
    const session: DaemonSession = {
      activeTurnId: '', agentId: 'default', cancelRequested: false, cwd: root, extra: {}, id: 'recover',
      interactionMode: 'code', sessionKey: 'recover', lastActive: 0, messages: [], metadata: {},
      model: 'bonsai', planMode: false, status: 'working', thinkingContent: [],
      toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0, workspace: root,
    }
    const events: unknown[] = []
    for await (const event of runner.run(session, 'go', new AbortController().signal)) events.push(event)
    expect(routes).toEqual([{ baseUrl: 'http://127.0.0.1:1234/v1', model: 'bonsai' }])
    expect(calls).toBe(2)
    expect(JSON.stringify(events)).toContain('Reloaded bonsai.')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

function fakeLmStudio(): { ports: LmStudioPorts; commands: string[][]; loaded: Map<string, number> } {
  const commands: string[][] = []
  const loaded = new Map<string, number>()
  let serverRunning = false
  const ok = (stdout = ''): LmsRunResult => ({ exitCode: 0, stdout, stderr: '' })
  const ports: LmStudioPorts = {
    which: () => undefined,
    exists: async () => true,
    async run(argv) {
      const args = argv.slice(1)
      if (args[0] === 'server' && args[1] === 'status') return ok(JSON.stringify({ running: serverRunning, port: 1234 }))
      commands.push([...argv])
      if (args[0] === 'server' && args[1] === 'start') serverRunning = true
      if (args[0] === 'load') {
        const identifier = args[args.indexOf('--identifier') + 1]!
        loaded.set(identifier, Number(args[args.indexOf('--context-length') + 1]))
      }
      return ok()
    },
    async fetch() {
      const data = [...loaded].map(([id, window]) => ({ id, state: 'loaded', loaded_context_length: window }))
      return new Response(JSON.stringify({ data }), { headers: { 'Content-Type': 'application/json' } })
    },
    sleep: async () => undefined,
    now: () => Date.now(),
  }
  return { ports, commands, loaded }
}

function fakeLlamaServer(): { ports: ServerPorts; spawned: string[][]; kill(): void } {
  const spawned: string[][] = []
  let alive: number | undefined
  let nextPid = 100
  const ports: ServerPorts = {
    which: () => undefined,
    exists: async () => true,
    spawn(argv) {
      spawned.push([...argv])
      alive = nextPid++
      return { pid: alive, exited: new Promise<number>(() => undefined) }
    },
    fetch: async () => ({ status: alive === undefined ? 503 : 200 }),
    isPidAlive: pid => pid === alive,
    commandOf: async pid => pid === alive ? spawned.at(-1)?.join(' ') : undefined,
    signal: () => { alive = undefined },
    portInUse: async () => false,
    sleep: async () => undefined,
    now: () => Date.now(),
  }
  return { ports, spawned, kill: () => { alive = undefined } }
}
