// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createConnection, type Socket } from 'node:net'
import type { JsonRpcPayload } from '../protocol/jsonRpc.js'

const DEFAULT_TIMEOUT_MS = 30_000
const MAX_FRAME_BYTES = 16 * 1024 * 1024
const MAX_TIMEOUT_MS = 2_147_483_647

export interface DaemonControlRequestOptions {
  readonly signal?: AbortSignal
  readonly timeoutMs?: number
}

/** Send exactly one newline-delimited JSON-RPC request to an existing daemon. */
export function requestDaemonControl(
  socketPath: string,
  method: string,
  params: JsonRpcPayload,
  options: DaemonControlRequestOptions = {},
): Promise<JsonRpcPayload> {
  const path = socketPath.trim()
  const name = method.trim()
  if (!path) return Promise.reject(new TypeError('daemon socket path is required'))
  if (!name) return Promise.reject(new TypeError('daemon control method is required'))
  if (path.includes('\u0000') || path.includes('\n') || path.includes('\r')) return Promise.reject(new TypeError('daemon socket path contains forbidden control characters'))
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || timeoutMs > MAX_TIMEOUT_MS) return Promise.reject(new TypeError(`daemon control timeout must be between 1 and ${MAX_TIMEOUT_MS}ms`))
  if (options.signal?.aborted) return Promise.reject(options.signal.reason ?? new Error('daemon control request aborted'))

  const id = `control-${crypto.randomUUID()}`
  const frame = `${JSON.stringify({ jsonrpc: '2.0', id, method: name, params })}\n`
  if (Buffer.byteLength(frame, 'utf8') > MAX_FRAME_BYTES) return Promise.reject(new Error('daemon control request exceeds the socket frame limit'))

  return new Promise<JsonRpcPayload>((resolve, reject) => {
    let socket: Socket | undefined
    let chunks: string[] = []
    let bufferedBytes = 0
    let settled = false
    let sent = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const abort = (): void => finish(options.signal?.reason ?? new Error('daemon control request aborted'))
    const cleanup = (): void => {
      if (timer !== undefined) clearTimeout(timer)
      options.signal?.removeEventListener('abort', abort)
      socket?.removeAllListeners()
      if (socket && !socket.destroyed) socket.destroy()
    }
    const finish = (error?: unknown, result?: JsonRpcPayload): void => {
      if (settled) return
      settled = true
      cleanup()
      if (error !== undefined) reject(error instanceof Error ? error : new Error(String(error)))
      else resolve(result ?? {})
    }
    const onData = (chunk: Buffer | string): void => {
      const text = typeof chunk === 'string' ? chunk : chunk.toString('utf8')
      chunks.push(text)
      bufferedBytes += Buffer.byteLength(text, 'utf8')
      if (bufferedBytes > MAX_FRAME_BYTES) {
        finish(new Error('daemon control response exceeds the socket frame limit'))
        return
      }
      // Scan each new chunk once; joining and rescanning the whole unfinished
      // frame on every read makes large responses quadratic.
      if (!text.includes('\n')) return
      let buffer = chunks.join('')
      chunks = []
      bufferedBytes = 0
      let newline = buffer.indexOf('\n')
      while (newline >= 0 && !settled) {
        const line = buffer.slice(0, newline)
        buffer = buffer.slice(newline + 1)
        newline = buffer.indexOf('\n')
        if (!line.trim()) continue
        let value: unknown
        try { value = JSON.parse(line) } catch { finish(new Error('daemon returned malformed JSON-RPC response')); return }
        if (!value || typeof value !== 'object' || Array.isArray(value)) { finish(new Error('daemon returned malformed JSON-RPC response')); return }
        const response = value as Record<string, unknown>
        if (response.jsonrpc !== '2.0') { finish(new Error('daemon returned a non-JSON-RPC response')); return }
        if (response.method === 'event') continue
        if (response.id !== id) { finish(new Error('daemon returned a response for an unexpected request id')); return }
        if (response.error && typeof response.error === 'object' && !Array.isArray(response.error)) {
          const error = response.error as Record<string, unknown>
          finish(new Error(`daemon control ${name} failed${typeof error.message === 'string' ? `: ${error.message}` : ''}`))
          return
        }
        if (!response.result || typeof response.result !== 'object' || Array.isArray(response.result)) {
          finish(new Error('daemon returned a malformed JSON-RPC result'))
          return
        }
        finish(undefined, response.result as JsonRpcPayload)
      }
      if (!settled && buffer) {
        chunks = [buffer]
        bufferedBytes = Buffer.byteLength(buffer, 'utf8')
      }
    }
    const onError = (error: Error & { readonly code?: string }): void => {
      if (!sent && (error.code === 'ENOENT' || error.code === 'ECONNREFUSED')) {
        finish(new Error(`Xerxes daemon is not running or its socket is unavailable at ${path}; start xerxes daemon --project-dir <project> or open the TUI, and verify the socket path`))
        return
      }
      finish(new Error(sent
        ? `daemon control connection failed after the request was sent; outcome is unknown, inspect the daemon before repeating the mutation: ${error.message}`
        : `daemon control connection failed: ${error.message}`))
    }
    const onClose = (): void => finish(new Error(sent
      ? 'daemon control connection closed after the request was sent; outcome is unknown, inspect the daemon before repeating the mutation'
      : 'daemon control connection closed before a response'))
    socket = createConnection(path)
    socket.setEncoding('utf8')
    socket.on('data', onData)
    socket.once('error', onError)
    socket.once('close', onClose)
    socket.once('connect', () => {
      if (!settled) { sent = true; socket?.write(frame) }
    })
    options.signal?.addEventListener('abort', abort, { once: true })
    timer = setTimeout(() => finish(new Error(sent
      ? `daemon control request timed out after the request was sent; outcome is unknown, inspect the daemon before repeating the mutation`
      : `daemon control request timed out after ${timeoutMs}ms`)), timeoutMs)
  })
}
