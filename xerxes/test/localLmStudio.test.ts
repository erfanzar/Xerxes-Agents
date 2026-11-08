// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  createLmStudioEngine,
  parseLmsModelList,
  parseLmsServerStatus,
  type LmsRunResult,
  type LmStudioPorts,
} from '../src/local/lmstudio.js'
import { ServerError } from '../src/local/server.js'

let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'xerxes-lmstudio-'))
})
afterEach(async () => {
  await rm(dir, { recursive: true, force: true })
})

const LMS = '/Users/me/.lmstudio/bin/lms'
const KEY = 'ternary-bonsai-2-27b-mlx'

interface Loaded {
  window: number
}

/** A fake LM Studio: `lms` argv in, server and loaded models as state, `/api/v0/models` out. */
interface FakeLmStudio {
  readonly ports: LmStudioPorts
  readonly runs: string[][]
  readonly progress: string[]
  server: { running: boolean; port: number }
  loaded: Map<string, Loaded>
  library: string[]
  lmsOnPath: boolean
  bundled: boolean
  /** Window LM Studio actually loads; defaults to the requested one. */
  loadWindow?: number
  loadResult: LmsRunResult
}

function fakeLmStudio(): FakeLmStudio {
  let clock = 1_000_000
  const f: FakeLmStudio = {
    runs: [],
    progress: [],
    server: { running: false, port: 1234 },
    loaded: new Map(),
    library: [KEY, 'qwen/qwen3-8b'],
    lmsOnPath: false,
    bundled: true,
    loadResult: { exitCode: 0, stdout: 'Model loaded successfully', stderr: '' },
    ports: undefined as unknown as LmStudioPorts,
  }
  const ok = (stdout = ''): LmsRunResult => ({ exitCode: 0, stdout, stderr: '' })
  const ports: LmStudioPorts = {
    which: (name) => (name === 'lms' && f.lmsOnPath ? '/usr/local/bin/lms' : undefined),
    exists: async (path) => path === LMS && f.bundled,
    async run(argv) {
      const args = argv.slice(1)
      f.runs.push(args)
      const [command, sub] = args
      if (command === 'server' && sub === 'status') return ok(JSON.stringify(f.server))
      if (command === 'server' && sub === 'start') {
        const at = args.indexOf('--port')
        f.server = { running: true, port: at < 0 ? f.server.port : Number(args[at + 1]) }
        return ok(`Success! Server is now running on port ${f.server.port}`)
      }
      if (command === 'server' && sub === 'stop') {
        f.server = { ...f.server, running: false }
        return ok('Stopped the server')
      }
      if (command === 'ls') {
        return ok(JSON.stringify(f.library.map((modelKey) => ({ type: 'llm', modelKey, maxContextLength: 262144 }))))
      }
      if (command === 'load') {
        if (f.loadResult.exitCode !== 0) return f.loadResult
        const identifier = args[args.indexOf('--identifier') + 1] ?? ''
        const ctxAt = args.indexOf('--context-length')
        f.loaded.set(identifier, { window: f.loadWindow ?? (ctxAt < 0 ? 4096 : Number(args[ctxAt + 1])) })
        return f.loadResult
      }
      if (command === 'unload') {
        f.loaded.delete(args[1] ?? '')
        return ok('Unloaded')
      }
      return { exitCode: 2, stdout: '', stderr: `unexpected lms ${args.join(' ')}` }
    },
    async fetch(url) {
      const parsed = new URL(url)
      if (!f.server.running || Number(parsed.port) !== f.server.port) throw new Error('connect ECONNREFUSED')
      if (parsed.pathname !== '/api/v0/models') return new Response('not found', { status: 404 })
      const data = [
        ...f.library.filter((key) => !f.loaded.has(key)).map((id) => ({ id, state: 'not-loaded', max_context_length: 262144 })),
        ...[...f.loaded].map(([id, { window }]) => ({ id, state: 'loaded', max_context_length: 262144, loaded_context_length: window })),
      ]
      return new Response(JSON.stringify({ object: 'list', data }))
    },
    sleep: async (ms) => {
      clock += ms
    },
    now: () => clock,
  }
  return Object.assign(f, { ports })
}

const engineFor = (f: FakeLmStudio) => createLmStudioEngine(f.ports, { homeDir: '/Users/me', stateDir: dir, pollMs: 10 })

describe('LM Studio engine', () => {
  test('finds lms on PATH, then in ~/.lmstudio/bin, and honours a configured path', async () => {
    const f = fakeLmStudio()
    expect(await engineFor(f).findBinary()).toBe(LMS)
    f.lmsOnPath = true
    expect(await engineFor(f).findBinary()).toBe('/usr/local/bin/lms')
    const configured = createLmStudioEngine(f.ports, { lmsPath: '/nowhere/lms', stateDir: dir })
    expect(await configured.findBinary()).toBeUndefined()
    f.bundled = false
    f.lmsOnPath = false
    const missing = await engineFor(f).start({ modelKey: KEY, identifier: KEY }).catch((error: unknown) => error)
    expect(missing).toBeInstanceOf(ServerError)
    expect((missing as ServerError).code).toBe('BINARY_NOT_FOUND')
    expect((missing as ServerError).message).toContain('lmstudio.ai')
  })

  test('start: starts the server, loads one slot with the chosen window, records it; stop undoes both', async () => {
    const f = fakeLmStudio()
    const engine = engineFor(f)
    const status = await engine.start({ modelKey: KEY, identifier: KEY, contextLength: 16384, onProgress: (t) => f.progress.push(t) })
    expect(status).toMatchObject({ running: true, healthy: true, port: 1234, model: KEY, contextLength: 16384 })
    // No --port: LM Studio keeps the port it last served on.
    expect(f.runs).toContainEqual(['server', 'start'])
    expect(f.runs).toContainEqual(['load', KEY, '--context-length', '16384', '--parallel', '1', '--gpu', 'max', '--identifier', KEY, '-y'])
    const state = JSON.parse(await readFile(join(dir, 'lmstudio.json'), 'utf8')) as Record<string, unknown>
    expect(state).toMatchObject({ engine: 'lmstudio', port: 1234, identifier: KEY, contextLength: 16384, startedServer: true, loadedModel: true })
    expect(await engine.status()).toMatchObject({ running: true, contextLength: 16384 })

    expect(await engine.stop()).toEqual({ stopped: true, model: KEY, unloaded: true, serverStopped: true })
    expect(f.loaded.size).toBe(0)
    expect(f.server.running).toBe(false)
    expect(await engine.status()).toEqual({ running: false, healthy: false, stale: false })
  })

  test("start: uses the user's running server (its port wins) and stop leaves that server up", async () => {
    const f = fakeLmStudio()
    f.server = { running: true, port: 1235 }
    const engine = engineFor(f)
    const status = await engine.start({ modelKey: KEY, identifier: KEY, contextLength: 16384, port: 11581, onProgress: (t) => f.progress.push(t) })
    expect(status.port).toBe(1235)
    expect(f.runs.some((args) => args[0] === 'server' && args[1] === 'start')).toBe(false)
    expect(f.progress.join('\n')).toContain('already running on port 1235')
    expect(await engine.stop()).toEqual({ stopped: true, model: KEY, unloaded: true, serverStopped: false })
    expect(f.server.running).toBe(true)
  })

  test('an explicit port is passed to `lms server start`', async () => {
    const f = fakeLmStudio()
    expect((await engineFor(f).start({ modelKey: KEY, identifier: KEY, port: 11581 })).port).toBe(11581)
    expect(f.runs).toContainEqual(['server', 'start', '--port', '11581'])
  })

  test('the window LM Studio reports is what is recorded, even when it differs from the request', async () => {
    const f = fakeLmStudio()
    f.loadWindow = 42752
    const status = await engineFor(f).start({ modelKey: KEY, identifier: KEY, contextLength: 65536, onProgress: (t) => f.progress.push(t) })
    expect(status.contextLength).toBe(42752)
    expect(f.progress.join('\n')).toContain('loaded a 42752-token window (asked for 65536)')
  })

  test('an identical load already in LM Studio is adopted, and stop leaves it loaded', async () => {
    const f = fakeLmStudio()
    f.server = { running: true, port: 1234 }
    f.loaded.set(KEY, { window: 16384 })
    const engine = engineFor(f)
    expect(await engine.start({ modelKey: KEY, identifier: KEY, contextLength: 16384 })).toMatchObject({ contextLength: 16384 })
    expect(f.runs.some((args) => args[0] === 'load')).toBe(false)
    expect(await engine.stop()).toEqual({ stopped: true, model: KEY, unloaded: false, serverStopped: false })
    expect(f.loaded.has(KEY)).toBe(true)
  })

  test('a load with another window is refused rather than silently reloaded', async () => {
    const f = fakeLmStudio()
    f.server = { running: true, port: 1234 }
    f.loaded.set(KEY, { window: 4096 })
    const error = await engineFor(f).start({ modelKey: KEY, identifier: KEY, contextLength: 16384 }).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ServerError)
    expect((error as ServerError).code).toBe('ALREADY_RUNNING')
    expect((error as ServerError).message).toContain(`lms unload ${KEY}`)
    expect(f.loaded.get(KEY)?.window).toBe(4096)
  })

  test("a failed load reports lms' output, stops the server Xerxes started, and records nothing", async () => {
    const f = fakeLmStudio()
    f.loadResult = { exitCode: 1, stdout: '', stderr: 'Error: Model loading aborted due to insufficient system resources.' }
    const error = await engineFor(f).start({ modelKey: KEY, identifier: KEY, contextLength: 262144 }).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ServerError)
    expect((error as ServerError).code).toBe('LOAD_FAILED')
    expect((error as ServerError).message).toContain('insufficient system resources')
    expect(f.server.running).toBe(false)
    expect(await readFile(join(dir, 'lmstudio.json'), 'utf8').catch(() => 'absent')).toBe('absent')
  })

  test('a model unloaded outside Xerxes reads as stale, and stop only clears the record', async () => {
    const f = fakeLmStudio()
    const engine = engineFor(f)
    await engine.start({ modelKey: KEY, identifier: KEY, contextLength: 16384 })
    f.loaded.clear()
    expect(await engine.status()).toMatchObject({ running: false, stale: true, model: KEY })
    const runsBefore = f.runs.length
    expect(await engine.stop()).toEqual({ stopped: false, reason: 'stale' })
    expect(f.runs.length).toBe(runsBefore)
    expect(await engine.status()).toEqual({ running: false, healthy: false, stale: false })
  })

  test('a second start while Xerxes has a live load is refused', async () => {
    const f = fakeLmStudio()
    const engine = engineFor(f)
    await engine.start({ modelKey: KEY, identifier: KEY, contextLength: 16384 })
    const error = await engine.start({ modelKey: KEY, identifier: KEY, contextLength: 16384 }).catch((e: unknown) => e)
    expect((error as ServerError).code).toBe('ALREADY_RUNNING')
  })

  test("stop keeps a server Xerxes started when someone else's model is loaded on it", async () => {
    const f = fakeLmStudio()
    const engine = engineFor(f)
    await engine.start({ modelKey: KEY, identifier: KEY, contextLength: 16384 })
    f.loaded.set('qwen/qwen3-8b', { window: 8192 })
    expect(await engine.stop()).toEqual({ stopped: true, model: KEY, unloaded: true, serverStopped: false })
    expect(f.server.running).toBe(true)
  })
})

test('lms JSON output parsers validate what they read', () => {
  expect(parseLmsServerStatus('{"running":false,"port":1235}')).toEqual({ running: false, port: 1235 })
  expect(() => parseLmsServerStatus('Waking up LM Studio service...')).toThrow('did not print JSON')
  expect(() => parseLmsServerStatus('{"port":1}')).toThrow('unexpected object')
  expect(
    parseLmsModelList(
      JSON.stringify([
        { type: 'llm', modelKey: KEY, format: 'safetensors', displayName: 'Ternary Bonsai 2 27B', maxContextLength: 262144 },
        { type: 'embedding', modelKey: 'nomic-embed' },
        { modelKey: '' },
      ]),
    ),
  ).toEqual([{ modelKey: KEY, displayName: 'Ternary Bonsai 2 27B', format: 'safetensors', maxContextLength: 262144 }])
  expect(() => parseLmsModelList('{}')).toThrow('did not print a list')
})
