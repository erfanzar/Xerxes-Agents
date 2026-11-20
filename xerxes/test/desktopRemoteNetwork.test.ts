// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { EventEmitter } from 'node:events'
import type { ChildProcess } from 'node:child_process'

import { expect, test } from 'bun:test'

import { browserLoadError, browserUrl, forwardArgs, MAX_PORT_FORWARDS, RemoteNetwork, validRemotePort } from '../src/desktop/main/remoteNetwork.js'

const machine = { alias: 'tpu', target: 'erfan@tpu.example.test', workspacePath: '/home/erfan/proj' }

function fakeChild(): ChildProcess & { killed: boolean; signals: string[] } {
  const child = new EventEmitter() as EventEmitter & { stderr: EventEmitter; killed: boolean; signals: string[]; kill(signal?: string): boolean }
  child.stderr = new EventEmitter()
  child.killed = false
  child.signals = []
  child.kill = (signal = 'SIGTERM') => { child.killed = true; child.signals.push(signal); return true }
  return child as unknown as ChildProcess & { killed: boolean; signals: string[] }
}

function fixture(options: { failReady?: boolean } = {}) {
  const spawned: { args: readonly string[]; child: ReturnType<typeof fakeChild> }[] = []
  let next = 40000
  const network = new RemoteNetwork(machine, {
    spawn: args => { const child = fakeChild(); spawned.push({ args, child }); return child },
    freePort: async preferred => preferred && preferred >= 1024 ? preferred : next++,
    waitReady: async () => { if (options.failReady) throw new Error('not ready') },
  })
  return { network, spawned }
}

test('forward argv binds loopback only, runs no remote command and fails fast', () => {
  const socks = forwardArgs(machine, { kind: 'socks', local: 41000 })
  expect(socks).toContain('-N')
  expect(socks).toContain('ExitOnForwardFailure=yes')
  expect(socks).toContain('StrictHostKeyChecking=yes')
  expect(socks).toContain('ForwardAgent=no')
  expect(socks.slice(socks.indexOf('-D'), socks.indexOf('-D') + 2)).toEqual(['-D', '127.0.0.1:41000'])
  expect(socks.at(-1)).toBe('erfan@tpu.example.test')
  expect(socks.at(-2)).toBe('--')
  const port = forwardArgs(machine, { kind: 'port', local: 8080, remote: 8080 })
  expect(port.slice(port.indexOf('-L'), port.indexOf('-L') + 2)).toEqual(['-L', '127.0.0.1:8080:127.0.0.1:8080'])
  expect(port).not.toContain('-R')
})

test('port validation and the address the tab loads', () => {
  expect(validRemotePort('8080')).toBe(8080)
  for (const bad of [0, 65536, -1, 'x', 1.5]) expect(() => validRemotePort(bad)).toThrow('1 to 65535')
  expect(browserUrl('8080')).toBe('http://localhost:8080/')
  expect(browserUrl('localhost:8080/docs')).toBe('http://localhost:8080/docs')
  expect(browserUrl('https://example.com')).toBe('https://example.com/')
  expect(browserUrl('10.0.0.4:6006')).toBe('http://10.0.0.4:6006/')
  expect(() => browserUrl('file:///etc/passwd')).toThrow('http')
  expect(() => browserUrl('javascript:alert(1)')).toThrow()
  expect(() => browserUrl('  ')).toThrow()
})

test('a refused address over SSH says the destination refused, not just the opaque SOCKS code', () => {
  const refused = browserLoadError(-120, 'ERR_SOCKS_CONNECTION_FAILED', 'http://10.164.2.31:9090/', 'v5p-128-develop')
  expect(refused).toContain('v5p-128-develop could not reach 10.164.2.31:9090')
  expect(refused).toContain('service may be stopped')
  expect(refused).toEndWith('ERR_SOCKS_CONNECTION_FAILED (-120)')
  expect(browserLoadError(-130, 'ERR_PROXY_CONNECTION_FAILED', 'http://localhost:8080/', 'tpu')).toContain('connection to tpu is down')
  expect(browserLoadError(-102, 'ERR_CONNECTION_REFUSED', 'http://localhost:3000/', null)).toBe('Could not reach localhost:3000. ERR_CONNECTION_REFUSED (-102)')
  expect(browserLoadError(-3, 'ERR_ABORTED', 'not a url', null)).toBe('ERR_ABORTED (-3)')
})

test('forwards reuse one process per port, list their state and stop on request', async () => {
  const { network, spawned } = fixture()
  expect(await network.forward(8080)).toBe(8080)
  expect(await network.forward('8080')).toBe(8080)
  expect(spawned).toHaveLength(1)
  expect(network.list()).toEqual([{ remote: 8080, local: 8080, state: 'ready' }])
  network.stop(8080)
  expect(spawned[0]!.child.killed).toBe(true)
  expect(network.list()).toEqual([])
})

test('the SOCKS listener starts once and everything closes with the workspace', async () => {
  const { network, spawned } = fixture()
  const port = await network.socksPort()
  expect(await network.socksPort()).toBe(port)
  await network.forward(6006)
  expect(spawned).toHaveLength(2)
  network.close()
  expect(spawned.every(row => row.child.killed)).toBe(true)
  await expect(network.socksPort()).rejects.toThrow('closed')
  await expect(network.forward(9000)).rejects.toThrow('closed')
})

test('an ssh exit is reported as a failed forward, and a retry starts a new process', async () => {
  const { network, spawned } = fixture()
  const ready = network.forward(5000)
  await Promise.resolve()
  await ready
  spawned[0]!.child.stderr?.emit('data', 'bind [127.0.0.1]:5000: Address already in use\n')
  spawned[0]!.child.emit('exit', 255)
  const [row] = network.list()
  expect(row?.state).toBe('failed')
  await network.forward(5000)
  expect(spawned).toHaveLength(2)
})

test('a forward that never becomes ready is killed and reported', async () => {
  const { network, spawned } = fixture({ failReady: true })
  await expect(network.forward(7000)).rejects.toThrow('not ready')
  expect(spawned[0]!.child.killed).toBe(true)
  expect(network.list()[0]?.state).toBe('failed')
})

test('closing a workspace while allocating its SOCKS port never starts ssh', async () => {
  const port = Promise.withResolvers<number>()
  const network = new RemoteNetwork(machine, {
    freePort: () => port.promise,
    spawn: () => { throw new Error('ssh must not start after close') },
  })
  const ready = network.socksPort()
  network.close()
  port.resolve(41000)
  await expect(ready).rejects.toThrow('workspace is closed')
})

test('at most MAX_PORT_FORWARDS ports per workspace', async () => {
  const { network } = fixture()
  for (let port = 0; port < MAX_PORT_FORWARDS; port++) await network.forward(20000 + port)
  await expect(network.forward(30000)).rejects.toThrow(`At most ${MAX_PORT_FORWARDS}`)
})
