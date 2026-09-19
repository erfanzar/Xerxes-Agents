// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmod, mkdtemp, rm } from 'node:fs/promises'
import { createConnection, createServer, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { parseLocalProviderCapabilities, type LocalProviderCapabilities } from '../../protocol/localProviderCapabilities.js'

/** This bridge transports only the documented private RPC envelope. Provider
 * configuration, clients and credentials remain in the local daemon. */
type Rpc = (method: string, params: Record<string, unknown>) => Promise<unknown>
export interface LocalProviderConsent {
  consent: true
  destination: string
  workspace: string
  profile: string
  model: string
  expires_at: number
  max_requests: number
  max_output_tokens: number | null
  max_concurrent: number
  consent_provider_controlled_output?: true
}
export interface LocalProviderBroker {
  readonly path: string
  readonly capabilities?: LocalProviderCapabilities
  close(): Promise<void>
}

const REQUEST_LIMIT = 16 * 1024 * 1024 + 4096
const REPLY_LIMIT = 1024 * 1024 + 4096
const DEADLINE_MS = 60_000
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value)
const unavailable = () => ({ error: 'grant_unavailable' })
const failure = () => new Error('Local provider bridge is unavailable. Review local setup before reconnecting.')

/** One newline frame per socket, bounded before parsing (including incomplete
 * frames). Do not decode chunks independently: UTF-8 may straddle reads. */
function readFrame(socket: Socket, limit: number, receive: (value: unknown) => void, invalid: () => void): void {
  const chunks: Buffer[] = []
  let bytes = 0, ended = false
  socket.on('data', (chunk: Buffer) => {
    if (ended) { invalid(); return }
    bytes += chunk.length
    if (bytes > limit) { ended = true; invalid(); return }
    const newline = chunk.indexOf(10)
    if (newline < 0) { chunks.push(chunk); return }
    ended = true
    if (newline !== chunk.length - 1) { invalid(); return }
    chunks.push(chunk.subarray(0, newline))
    let value: unknown
    try { value = JSON.parse(Buffer.concat(chunks).toString('utf8')) }
    catch { invalid(); return }
    receive(value)
  })
}

/** Call only after the user has reviewed and consented to this exact scope.
 * The owning gateway must remain connected for the lifetime of the broker.
 * Neither the grant id nor a daemon credential is given to the child renderer. */
export async function createLocalProviderBroker(rpc: Rpc, consent: LocalProviderConsent, signal?: AbortSignal): Promise<LocalProviderBroker> {
  if (signal?.aborted) throw failure()
  // Snapshot before the await: a caller cannot mutate approved scope in flight.
  const scope = { ...consent }
  let authorized: unknown
  try { authorized = await rpc('provider.relay.authorize', { ...scope }) }
  catch { throw failure() }
  if (!record(authorized) || authorized.ok !== true || !record(authorized.grant) ||
    typeof authorized.grant.id !== 'string' || !/^[a-f0-9]{32}$/.test(authorized.grant.id)) throw failure()
  const id = authorized.grant.id
  let directory: string | undefined
  let closed = false
  let closing: Promise<void> | undefined
  let expiry: ReturnType<typeof setTimeout> | undefined
  const sockets = new Set<Socket>()
  const server = createServer(socket => {
    socket.on('error', () => {}) // Never echo filesystem paths or RPC diagnostics.
    if (closed || sockets.size >= 16) { socket.destroy(); return }
    sockets.add(socket)
    let frameId: string | undefined, completed = false
    const cancel = () => {
      if (!completed && frameId) {
        const cancelled = frameId; frameId = undefined
        void rpc('provider.relay.next', { id, frame: { op: 'cancel', id: cancelled } }).catch(() => {})
      }
    }
    const deadline = setTimeout(() => { cancel(); socket.destroy() }, DEADLINE_MS)
    deadline.unref?.()
    socket.once('close', () => { clearTimeout(deadline); sockets.delete(socket); cancel() })
    const answer = (value: unknown) => {
      if (closed || socket.destroyed) return
      let line: string
      try { line = JSON.stringify(value) + '\n' } catch { line = JSON.stringify(unavailable()) + '\n' }
      if (Buffer.byteLength(line) > REPLY_LIMIT) { cancel(); line = JSON.stringify(unavailable()) + '\n' }
      completed = true
      socket.end(line)
    }
    readFrame(socket, REQUEST_LIMIT, value => {
      if (!record(value) || Object.keys(value).some(key => !['op', 'id', 'request'].includes(key)) ||
        !['next', 'cancel'].includes(String(value.op)) || typeof value.id !== 'string' || !/^[a-zA-Z0-9_-]{1,64}$/.test(value.id)) {
        answer({ error: 'invalid_request' }); return
      }
      frameId = value.id
      void rpc('provider.relay.next', { id, frame: value }).then(result => {
        if (!record(result) || result.ok !== true || !record(result.reply)) { cancel(); answer(unavailable()); return }
        answer(result.reply)
      }, () => { cancel(); answer(unavailable()) })
    }, () => { cancel(); socket.destroy() })
  })
  const close = (): Promise<void> => {
    if (closing) return closing
    closed = true
    clearTimeout(expiry)
    signal?.removeEventListener('abort', abort)
    for (const socket of sockets) socket.destroy()
    closing = (async () => {
      await new Promise<void>(resolve => server.close(() => resolve()))
      // Revocation is idempotent at this owner, including startup failure.
      // A lost gateway also revokes at the daemon's disconnect boundary.
      try { await rpc('provider.relay.revoke', { id }) } catch { /* Owner disconnect revokes. */ }
      if (directory) await rm(directory, { recursive: true, force: true })
    })()
    return closing
  }
  const abort = () => { void close().catch(() => {}) }
  try {
    if (signal?.aborted) throw failure()
    const capabilities = parseLocalProviderCapabilities(authorized.grant.capabilities, scope.model)
    directory = await mkdtemp(join(tmpdir(), 'xr-p-'))
    await chmod(directory, 0o700)
    const path = join(directory, 'relay.sock')
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(path, () => { server.removeListener('error', reject); resolve() })
    })
    server.on('error', abort)
    await chmod(path, 0o600)
    if (signal?.aborted || Date.now() >= scope.expires_at) throw failure()
    signal?.addEventListener('abort', abort, { once: true })
    expiry = setTimeout(abort, Math.max(1, scope.expires_at - Date.now()))
    expiry.unref?.()
    return { path, close, ...(capabilities ? {capabilities} : {}) }
  } catch {
    await close()
    throw failure()
  }
}

/** Child renderer transport. No retry: replaying a failed provider request could
 * spend authority twice. Gateway/session recovery must explicitly rebind. */
export function requestLocalProviderBroker(path: string, frame: Readonly<Record<string, unknown>>, signal?: AbortSignal): Promise<unknown> {
  if (signal?.aborted) return Promise.resolve({ error: 'cancelled' })
  let line: string
  try { line = JSON.stringify(frame) + '\n' } catch { return Promise.resolve({ error: 'invalid_request' }) }
  if (Buffer.byteLength(line) > REQUEST_LIMIT) return Promise.resolve({ error: 'invalid_request' })
  return new Promise(resolve => {
    let done = false
    const socket = createConnection({ path })
    const finish = (value: unknown) => {
      if (done) return
      done = true
      clearTimeout(deadline)
      signal?.removeEventListener('abort', abort)
      socket.destroy()
      resolve(value)
    }
    const abort = () => finish({ error: 'cancelled' })
    const deadline = setTimeout(() => finish(unavailable()), DEADLINE_MS)
    deadline.unref?.()
    socket.once('error', () => finish(unavailable()))
    socket.once('end', () => finish(unavailable()))
    socket.once('close', () => finish(unavailable()))
    socket.once('connect', () => { if (!done) socket.write(line) })
    readFrame(socket, REPLY_LIMIT, value => finish(record(value) ? value : unavailable()), () => finish(unavailable()))
    signal?.addEventListener('abort', abort, { once: true })
    if (signal?.aborted) abort()
  })
}
