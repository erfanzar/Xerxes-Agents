// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { appendFileSync } from 'node:fs'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { createLocalServer, ServerError, type ServerPorts } from '../src/local/server.js'

let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'xerxes-local-srv-'))
})
afterEach(async () => {
  await rm(dir, { recursive: true, force: true })
})

interface Fake {
  readonly ports: ServerPorts
  readonly spawned: { argv: string[]; logPath: string }[]
  readonly signals: { pid: number; signal: string }[]
  readonly alive: Set<number>
  readonly commands: Map<number, string>
  health: 'ok' | 'down'
  portBusy: boolean
  binary: string | undefined
  ignoreSigterm: boolean
  exitImmediately: number | undefined
  logLine: string
}

function fake(): Fake {
  let clock = 1_000_000
  const state: Fake = {
    spawned: [],
    signals: [],
    alive: new Set(),
    commands: new Map(),
    health: 'ok',
    portBusy: false,
    binary: '/opt/bin/llama-server',
    ignoreSigterm: false,
    exitImmediately: undefined,
    logLine: 'loading model...',
    ports: undefined as unknown as ServerPorts,
  }
  const ports: ServerPorts = {
    which: () => state.binary,
    exists: async (path) => path === state.binary,
    spawn(argv, { logPath }) {
      state.spawned.push({ argv, logPath })
      appendFileSync(logPath, state.logLine + '\n')
      const pid = 4242
      if (state.exitImmediately !== undefined) {
        return { pid, exited: Promise.resolve(state.exitImmediately) }
      }
      state.alive.add(pid)
      state.commands.set(pid, argv.join(' '))
      return { pid, exited: new Promise<number>(() => undefined) }
    },
    fetch: async () => {
      if (state.health === 'down') throw new Error('ECONNREFUSED')
      return { status: 200 }
    },
    isPidAlive: (pid) => state.alive.has(pid),
    commandOf: async (pid) => state.commands.get(pid),
    signal(pid, signal) {
      state.signals.push({ pid, signal })
      if (signal === 'SIGKILL' || !state.ignoreSigterm) state.alive.delete(pid)
    },
    portInUse: async () => state.portBusy,
    sleep: async (ms) => {
      clock += ms
    },
    now: () => clock,
  }
  return Object.assign(state, { ports })
}

const start = { modelPath: '/models/bonsai-9b.gguf', port: 11581, ctxSize: 8192 }

describe('createLocalServer', () => {
  test('spawns the exact argv, writes state, and returns once /health is 200', async () => {
    const f = fake()
    const server = createLocalServer(f.ports, { stateDir: dir })
    const status = await server.start(start)
    expect(f.spawned).toHaveLength(1)
    expect(f.spawned[0]?.argv).toEqual([
      '/opt/bin/llama-server',
      '-m',
      '/models/bonsai-9b.gguf',
      '--host',
      '127.0.0.1',
      '--port',
      '11581',
      '--ctx-size',
      '8192',
      '--jinja',
    ])
    expect(status).toMatchObject({ running: true, healthy: true, pid: 4242, port: 11581 })
    const state = JSON.parse(await readFile(join(dir, 'server.json'), 'utf8'))
    expect(state).toMatchObject({ pid: 4242, port: 11581, model: '/models/bonsai-9b.gguf', logPath: join(dir, 'llama-server.log') })
    expect(typeof state.startedAt).toBe('string')
  })

  test('a configured llama_server_path is used, and a missing binary is an actionable error', async () => {
    const f = fake()
    f.binary = undefined
    const server = createLocalServer(f.ports, { stateDir: dir })
    const error = await server.start(start).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ServerError)
    expect((error as ServerError).code).toBe('BINARY_NOT_FOUND')
    expect((error as ServerError).message).toContain('brew install llama.cpp')
    expect((error as ServerError).message).toContain('Prism')
    expect(f.spawned).toEqual([])

    const configured = createLocalServer(f.ports, { stateDir: dir, llamaServerPath: '/nowhere/llama-server' })
    const missing = await configured.start(start).catch((e: unknown) => e)
    expect((missing as ServerError).message).toContain('/nowhere/llama-server')
  })

  test('a port in use is refused before spawning', async () => {
    const f = fake()
    f.portBusy = true
    const error = await createLocalServer(f.ports, { stateDir: dir }).start(start).catch((e: unknown) => e)
    expect((error as ServerError).code).toBe('PORT_IN_USE')
    expect((error as ServerError).message).toContain('11581')
    expect(f.spawned).toEqual([])
  })

  test('health timeout kills the child, removes state and reports the log tail', async () => {
    const f = fake()
    f.health = 'down'
    f.logLine = 'error: failed to load model'
    const server = createLocalServer(f.ports, { stateDir: dir, pollMs: 100, graceMs: 1000 })
    const error = await server.start({ ...start, timeoutMs: 1000 }).catch((e: unknown) => e)
    expect((error as ServerError).code).toBe('START_TIMEOUT')
    expect((error as ServerError).message).toContain('failed to load model')
    expect(f.signals).toEqual([{ pid: 4242, signal: 'SIGTERM' }])
    expect(f.alive.has(4242)).toBe(false)
    await expect(stat(join(dir, 'server.json'))).rejects.toThrow()
  })

  test('a child that ignores SIGTERM is SIGKILLed after the grace period', async () => {
    const f = fake()
    f.health = 'down'
    f.ignoreSigterm = true
    const server = createLocalServer(f.ports, { stateDir: dir, pollMs: 100, graceMs: 500 })
    await server.start({ ...start, timeoutMs: 300 }).catch(() => undefined)
    expect(f.signals.map((s) => s.signal)).toEqual(['SIGTERM', 'SIGKILL'])
  })

  test('a child that exits before becoming healthy fails fast with its log', async () => {
    const f = fake()
    f.health = 'down'
    f.exitImmediately = 3
    f.logLine = 'unknown model architecture'
    const server = createLocalServer(f.ports, { stateDir: dir, pollMs: 50 })
    const error = await server.start({ ...start, timeoutMs: 60_000 }).catch((e: unknown) => e)
    expect((error as ServerError).code).toBe('START_FAILED')
    expect((error as ServerError).message).toContain('code 3')
    expect((error as ServerError).message).toContain('unknown model architecture')
    await expect(stat(join(dir, 'server.json'))).rejects.toThrow()
  })

  test('status with a stale pid reports stale and not running', async () => {
    const f = fake()
    await writeFile(
      join(dir, 'server.json'),
      JSON.stringify({ pid: 999, port: 11581, model: '/m.gguf', startedAt: 'x', logPath: '/l', bin: '/opt/bin/llama-server' }),
    )
    const status = await createLocalServer(f.ports, { stateDir: dir }).status()
    expect(status).toMatchObject({ running: false, healthy: false, stale: true, pid: 999 })
  })

  test('status with no state file is simply not running', async () => {
    const status = await createLocalServer(fake().ports, { stateDir: dir }).status()
    expect(status).toEqual({ running: false, healthy: false, stale: false })
  })

  test('status reflects /health of a running server', async () => {
    const f = fake()
    const server = createLocalServer(f.ports, { stateDir: dir })
    await server.start(start)
    f.health = 'down'
    expect(await server.status()).toMatchObject({ running: true, healthy: false })
  })

  test('stop sends SIGTERM to our server and removes state', async () => {
    const f = fake()
    const server = createLocalServer(f.ports, { stateDir: dir, graceMs: 500 })
    await server.start(start)
    const result = await server.stop()
    expect(result).toEqual({ stopped: true, pid: 4242, forced: false })
    expect(f.signals).toEqual([{ pid: 4242, signal: 'SIGTERM' }])
    await expect(stat(join(dir, 'server.json'))).rejects.toThrow()
  })

  test('stop never signals a recycled pid whose command does not match', async () => {
    const f = fake()
    f.alive.add(777)
    f.commands.set(777, '/usr/bin/some-other-program --serve')
    await writeFile(
      join(dir, 'server.json'),
      JSON.stringify({ pid: 777, port: 11581, model: '/m.gguf', startedAt: 'x', logPath: '/l', bin: '/opt/bin/llama-server' }),
    )
    const result = await createLocalServer(f.ports, { stateDir: dir }).stop()
    expect(result).toEqual({ stopped: false, reason: 'foreign-process', pid: 777 })
    expect(f.signals).toEqual([])
    expect(f.alive.has(777)).toBe(true)
  })

  test('stop with a dead recorded pid cleans up without signalling', async () => {
    const f = fake()
    await writeFile(
      join(dir, 'server.json'),
      JSON.stringify({ pid: 5, port: 1, model: '/m.gguf', startedAt: 'x', logPath: '/l', bin: '/b/llama-server' }),
    )
    const server = createLocalServer(f.ports, { stateDir: dir })
    expect(await server.stop()).toEqual({ stopped: false, reason: 'stale', pid: 5 })
    expect(f.signals).toEqual([])
    expect(await server.stop()).toEqual({ stopped: false, reason: 'not-running' })
  })

  test('starting while one is already running is refused', async () => {
    const f = fake()
    const server = createLocalServer(f.ports, { stateDir: dir })
    await server.start(start)
    const error = await server.start(start).catch((e: unknown) => e)
    expect((error as ServerError).code).toBe('ALREADY_RUNNING')
    expect(f.spawned).toHaveLength(1)
  })
})
