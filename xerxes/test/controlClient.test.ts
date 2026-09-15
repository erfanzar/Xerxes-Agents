// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createServer, type Server } from 'node:net'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { requestDaemonControl } from '../src/daemon/controlClient.js'

async function fixture(handler: (socket: import('node:net').Socket) => void): Promise<{ path: string; server: Server; close: () => Promise<void> }> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-control-client-'))
  const path = join(directory, 'daemon.sock')
  const server = createServer(handler)
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(path, resolve)
  })
  return {
    path,
    server,
    close: async () => { await new Promise<void>(resolve => server.close(() => resolve())); await rm(directory, { recursive: true, force: true }) },
  }
}

test('control client sends one NDJSON request and ignores daemon events', async () => {
  const f = await fixture(socket => {
    socket.setEncoding('utf8')
    socket.on('data', raw => {
      const request = JSON.parse(String(raw))
      socket.write(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'status', payload: {} } }) + '\n')
      socket.write(JSON.stringify({ jsonrpc: '2.0', id: request.id, result: { ok: true, value: 3 } }) + '\n')
    })
  })
  try { await expect(requestDaemonControl(f.path, 'runtime.status', { detail: true })).resolves.toEqual({ ok: true, value: 3 }) }
  finally { await f.close() }
})

test('control client handles fragmented UTF-8 responses and rejects oversized frames', async () => {
  const fragmented = await fixture(socket => socket.on('data', raw => {
    const request = JSON.parse(String(raw))
    const frame = Buffer.from(JSON.stringify({ jsonrpc: '2.0', id: request.id, result: { text: 'café 🚀' } }) + '\n')
    const marker = frame.indexOf(Buffer.from('é')) + 1
    socket.write(frame.subarray(0, marker))
    setTimeout(() => socket.write(frame.subarray(marker)), 1)
  }))
  try { await expect(requestDaemonControl(fragmented.path, 'status', {})).resolves.toEqual({ text: 'café 🚀' }) }
  finally { await fragmented.close() }
  const oversized = await fixture(socket => socket.on('data', raw => {
    const request = JSON.parse(String(raw))
    socket.write(JSON.stringify({ jsonrpc: '2.0', id: request.id, result: { output: 'x'.repeat(16 * 1024 * 1024) } }) + '\n')
  }))
  try { await expect(requestDaemonControl(oversized.path, 'status', {})).rejects.toThrow('exceeds the socket frame limit') }
  finally { await oversized.close() }
})

test('control client rejects daemon JSON-RPC failures and malformed responses', async () => {
  const failed = await fixture(socket => socket.on('data', raw => {
    const request = JSON.parse(String(raw))
    socket.write(JSON.stringify({ jsonrpc: '2.0', id: request.id, error: { code: -32000, message: 'denied' } }) + '\n')
  }))
  try { await expect(requestDaemonControl(failed.path, 'session.delete', {})).rejects.toThrow('denied') }
  finally { await failed.close() }
  const malformed = await fixture(socket => socket.on('data', () => socket.write('{broken\n')))
  try { await expect(requestDaemonControl(malformed.path, 'status', {})).rejects.toThrow('malformed') }
  finally { await malformed.close() }
})

test('control client retains an unfinished response after a complete event frame', async () => {
  const f = await fixture(socket => socket.on('data', raw => {
    const request = JSON.parse(String(raw))
    const response = JSON.stringify({ jsonrpc: '2.0', id: request.id, result: { text: 'retained tail' } }) + '\n'
    socket.write(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: {} }) + '\n' + response.slice(0, 20))
    setTimeout(() => socket.write(response.slice(20)), 5)
  }))
  try { await expect(requestDaemonControl(f.path, 'status', {})).resolves.toEqual({ text: 'retained tail' }) }
  finally { await f.close() }
})

test('control client rejects close, abort, timeout, and absent daemon', async () => {
  const closed = await fixture(socket => socket.on('data', () => socket.destroy()))
  try { await expect(requestDaemonControl(closed.path, 'status', {})).rejects.toThrow('closed') }
  finally { await closed.close() }
  const controller = new AbortController()
  const held = await fixture(() => undefined)
  try {
    const pending = requestDaemonControl(held.path, 'schedule.run', {}, { signal: controller.signal, timeoutMs: 500 })
    controller.abort(new Error('cancelled by test'))
    await expect(pending).rejects.toThrow('cancelled by test')
    await expect(requestDaemonControl(held.path, 'schedule.run', {}, { timeoutMs: 5 })).rejects.toThrow('timed out')
  } finally { await held.close() }
  await expect(requestDaemonControl('/tmp/xerxes-no-such-daemon.sock', 'status', {}, { timeoutMs: 100 })).rejects.toThrow(/not running|connection failed|ENOENT|closed/)
})
