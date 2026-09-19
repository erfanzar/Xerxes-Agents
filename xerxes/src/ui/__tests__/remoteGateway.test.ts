// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { createServer, type Socket } from 'node:net'
import { once } from 'node:events'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { expect, it } from 'vitest'
import { GatewayClient } from '../gatewayClient.js'

it('external gateways verify protocol without replacing the remote daemon for a different local build', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xr-gw-'))
  const path = join(dir, 'rpc.sock')
  const server = createServer(socket => {
    socket.setEncoding('utf8')
    let buffer = ''
    socket.on('data', data => {
      buffer += data
      while (buffer.includes('\n')) {
        const end = buffer.indexOf('\n'), frame = JSON.parse(buffer.slice(0, end)); buffer = buffer.slice(end + 1)
        socket.write(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: { runtime: 'bun-typescript', daemon_protocol: 35, daemon_build_id: 'remote-build' } }) + '\n')
      }
    })
  })
  await new Promise<void>(resolve => server.listen(path, resolve))
  const client = new GatewayClient({ externalSocketPath: path, projectDir: '/remote/project', expectedDaemonBuildId: 'different-local-build' })
  try {
    await client.start()
    expect(client.didSpawnDaemon).toBe(false)
    expect(await client.request('runtime.status', {})).toMatchObject({ daemon_build_id: 'remote-build' })
  } finally { client.close(); await new Promise<void>(resolve => server.close(() => resolve())); await rm(dir, { recursive: true, force: true }) }
})

it('a missing external tunnel never falls back to starting a local daemon', async () => {
  const client = new GatewayClient({ externalSocketPath: '/tmp/xerxes-does-not-exist/rpc.sock', projectDir: '/remote/project' })
  try { await expect(client.start()).rejects.toThrow('Remote daemon tunnel is unavailable'); expect(client.didSpawnDaemon).toBe(false) }
  finally { client.close() }
})

it('waits for a replacement external socket and reconnects the same gateway', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xr-reconnect-'))
  const path = join(dir, 'rpc.sock')
  const sockets = new Set<Socket>()
  const serve = () => createServer(socket => {
    sockets.add(socket)
    socket.on('close', () => sockets.delete(socket))
    let buffer = ''
    socket.on('data', data => {
      buffer += String(data)
      while (buffer.includes('\n')) {
        const end = buffer.indexOf('\n'), frame = JSON.parse(buffer.slice(0, end)); buffer = buffer.slice(end + 1)
        socket.write(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result: { runtime: 'bun-typescript', daemon_protocol: 35 } }) + '\n')
      }
    })
  })
  const initial = serve()
  let replacement: ReturnType<typeof serve> | undefined
  const client = new GatewayClient({ externalSocketPath: path, projectDir: '/remote/project' })
  await new Promise<void>(resolve => initial.listen(path, resolve))
  try {
    await client.start()
    const lost = once(client, 'close')
    sockets.forEach(socket => socket.destroy())
    await new Promise<void>(resolve => initial.close(() => resolve()))
    await lost
    let recovered = false
    const pending = client.start().then(() => { recovered = true })
    await new Promise(resolve => setTimeout(resolve, 120))
    expect(recovered).toBe(false)
    replacement = serve()
    await new Promise<void>(resolve => replacement!.listen(path, resolve))
    await pending
    expect(client.connected).toBe(true)
    expect(client.didSpawnDaemon).toBe(false)

    const lostAgain = once(client, 'close')
    sockets.forEach(socket => socket.destroy())
    await new Promise<void>(resolve => replacement!.close(() => resolve()))
    replacement = undefined
    await lostAgain
    const cancelled = client.start()
    client.close()
    await expect(cancelled).rejects.toThrow('cancelled')
    expect(client.connected).toBe(false)
  } finally {
    client.close()
    sockets.forEach(socket => socket.destroy())
    if (initial.listening) await new Promise<void>(resolve => initial.close(() => resolve()))
    if (replacement?.listening) await new Promise<void>(resolve => replacement!.close(() => resolve()))
    await rm(dir, { recursive: true, force: true })
  }
})
