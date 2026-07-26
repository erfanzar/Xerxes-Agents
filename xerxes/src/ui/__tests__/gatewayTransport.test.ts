// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { daemonPaths, readDaemonEndpoint, resolveDaemonTransport } from '../gatewayClient.js'

const directories: string[] = []

afterEach(() => {
  for (const directory of directories.splice(0)) {
    rmSync(directory, { force: true, recursive: true })
  }
})

describe('resolveDaemonTransport', () => {
  it('defaults to unix on POSIX and websocket on win32', () => {
    expect(resolveDaemonTransport({}, 'linux')).toBe('unix')
    expect(resolveDaemonTransport({}, 'darwin')).toBe('unix')
    expect(resolveDaemonTransport({}, 'win32')).toBe('websocket')
  })

  it('honors XERXES_DAEMON_TRANSPORT over the platform default', () => {
    expect(resolveDaemonTransport({ XERXES_DAEMON_TRANSPORT: 'websocket' }, 'linux')).toBe('websocket')
    expect(resolveDaemonTransport({ XERXES_DAEMON_TRANSPORT: 'unix' }, 'win32')).toBe('unix')
  })
})

describe('readDaemonEndpoint', () => {
  it('parses a valid endpoint file and rejects malformed ones', () => {
    const directory = mkdtempSync(join(tmpdir(), 'xerxes-gateway-transport-'))
    directories.push(directory)
    const endpointPath = join(directory, 'endpoint.json')

    expect(readDaemonEndpoint(endpointPath)).toBeNull()

    writeFileSync(
      endpointPath,
      JSON.stringify({ transport: 'ws', url: 'ws://127.0.0.1:9999/', token: 'abc', pid: 1, protocol: 35 })
    )
    expect(readDaemonEndpoint(endpointPath)?.token).toBe('abc')

    writeFileSync(
      endpointPath,
      JSON.stringify({ transport: 'ws', url: 'ws://127.0.0.1:9999/', token: 'abc', pid: 1, protocol: 34 })
    )
    expect(readDaemonEndpoint(endpointPath)).toBeNull()
  })

  it('derives endpoint paths beside the per-project socket and pid paths', () => {
    const paths = daemonPaths(join(tmpdir(), 'xerxes-transport-project'))
    expect(paths.endpointPath.endsWith('.endpoint.json')).toBe(true)
    expect(paths.endpointPath.slice(0, -'.endpoint.json'.length)).toBe(paths.socketPath.slice(0, -'.sock'.length))
  })
})
