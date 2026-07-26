// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { existsSync } from 'node:fs'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { daemonPaths } from '../../daemon/paths.js'
import { DaemonServer } from '../../daemon/server.js'
import { GatewayClient } from '../gatewayClient.js'

const ENV_KEYS = ['XERXES_HOME', 'XERXES_DAEMON_TRANSPORT'] as const
const savedEnvironment = new Map<string, string | undefined>()

afterEach(() => {
  for (const key of ENV_KEYS) {
    const value = savedEnvironment.get(key)
    if (value === undefined) delete process.env[key]
    else process.env[key] = value
  }
  savedEnvironment.clear()
})

const setEnvironment = (values: Record<string, string>): void => {
  for (const [key, value] of Object.entries(values)) {
    if (!savedEnvironment.has(key)) savedEnvironment.set(key, process.env[key])
    process.env[key] = value
  }
}

interface RunningDaemon {
  readonly client: GatewayClient
  readonly directory: string
  readonly endpointPath: string
  readonly server: DaemonServer
}

const startWebSocketDaemon = async (): Promise<RunningDaemon> => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-ws-daemon-'))
  const home = join(directory, 'home')
  const project = join(directory, 'project')
  await Bun.write(join(project, 'marker.txt'), '')
  setEnvironment({ XERXES_HOME: home, XERXES_DAEMON_TRANSPORT: 'websocket' })
  const paths = daemonPaths(project, process.env)
  const server = new DaemonServer({
    socketPath: paths.socketPath,
    endpointPath: paths.endpointPath,
    pidPath: paths.pidPath,
    transport: 'websocket',
    websocket: { port: 0 }
  })
  await server.start()
  return { client: new GatewayClient({ projectDir: project }), directory, endpointPath: paths.endpointPath, server }
}

describe('daemon websocket control transport', () => {
  it('publishes a loopback endpoint file and serves JSON-RPC to the TUI client', async () => {
    const running = await startWebSocketDaemon()
    try {
      expect(existsSync(running.endpointPath)).toBe(true)
      const published = JSON.parse(await readFile(running.endpointPath, 'utf8'))
      expect(published.transport).toBe('ws')
      expect(published.protocol).toBe(35)
      expect(published.pid).toBe(process.pid)
      expect(typeof published.token).toBe('string')
      expect(published.token.length).toBeGreaterThanOrEqual(16)
      const url = new URL(published.url)
      expect(url.protocol).toBe('ws:')
      expect(url.hostname).toBe('127.0.0.1')
      expect(Number(url.port)).toBeGreaterThan(0)

      // The TUI client discovers the endpoint file and completes a round trip.
      await running.client.start()
      const status = await running.client.request<Record<string, unknown>>('runtime.status', {})
      expect(status.daemon_protocol).toBe(35)
      expect(status.runtime).toBe('bun-typescript')
    } finally {
      running.client.close()
      await running.server.stop()
      await rm(running.directory, { force: true, recursive: true })
    }
    // stop() removes the published endpoint so stale tokens cannot be reused.
    expect(existsSync(running.endpointPath)).toBe(false)
  }, 15_000)

  it('rejects clients without the endpoint token', async () => {
    const running = await startWebSocketDaemon()
    try {
      const published = JSON.parse(await readFile(running.endpointPath, 'utf8'))
      const bad = new URL(published.url)
      bad.searchParams.set('token', 'definitely-wrong')
      const opened = await new Promise<boolean>(resolve => {
        const ws = new WebSocket(bad)
        const timer = setTimeout(() => resolve(false), 5_000)
        ws.addEventListener('open', () => {
          clearTimeout(timer)
          ws.close()
          resolve(true)
        })
        ws.addEventListener('error', () => {
          clearTimeout(timer)
          resolve(false)
        })
      })
      expect(opened).toBe(false)
    } finally {
      await running.server.stop()
      await rm(running.directory, { force: true, recursive: true })
    }
  }, 15_000)

  it('honors an explicit unix transport selection on every host', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'xerxes-ws-explicit-'))
    setEnvironment({ XERXES_HOME: join(directory, 'home'), XERXES_DAEMON_TRANSPORT: 'websocket' })
    const project = join(directory, 'project')
    await Bun.write(join(project, 'marker.txt'), '')
    const paths = daemonPaths(project, process.env)
    // Explicit unix selection must be honored even on Windows (libuv maps the
    // path into the named-pipe namespace there), never silently swapped.
    const server = new DaemonServer({
      socketPath: paths.socketPath,
      endpointPath: paths.endpointPath,
      transport: 'unix',
      websocket: { port: 0 }
    })
    try {
      await server.start()
      if (process.platform !== 'win32') {
        expect(existsSync(paths.socketPath)).toBe(true)
      }
      // The endpoint file belongs to the websocket transport only.
      expect(existsSync(paths.endpointPath)).toBe(false)
    } finally {
      await server.stop().catch(() => undefined)
      await rm(directory, { force: true, recursive: true })
    }
  }, 15_000)
})
