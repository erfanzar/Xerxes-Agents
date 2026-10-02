// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { createBoundedMonitorSocket, type MonitorSocket } from './websocketMonitorSocket.js'
import { WebSocketMonitorProtocolError } from './websocketMonitorFrames.js'

const MAX_URL_CHARS = 4_096
const MAX_FRAME_BYTES = 64 * 1024
const INITIAL_TIMEOUT_MS = 5_000
const RETRY_DELAYS_MS = [250, 1_000, 4_000] as const

export interface WebSocketMonitorEvent {
  readonly kind: 'message' | 'gap'
  readonly text: string
  readonly identity: string
}

export interface WebSocketMonitorSource {
  open(
    url: string,
    onEvent: (event: WebSocketMonitorEvent) => void,
    onStatus: (status: string) => void,
    onError: (error: unknown) => void,
    signal?: AbortSignal,
  ): Promise<{ readonly url: string; readonly close: () => void }>
}

const nativeWebSocketMonitorSource: WebSocketMonitorSource = {
  async open(url, onEvent, onStatus, onError, signal) {
    const address = validateUrl(url)
    if (signal?.aborted) throw signal.reason ?? new Error('WebSocket monitor was aborted before opening')
    let socket: MonitorSocket | undefined
    let timer: ReturnType<typeof setTimeout> | undefined
    let closed = false
    let everOpened = false
    let initialSettled = false
    let retries = 0
    let generation = 0
    let resolveInitial: (() => void) | undefined
    let rejectInitial: ((error: unknown) => void) | undefined
    const initial = new Promise<void>((resolve, reject) => { resolveInitial = resolve; rejectInitial = reject })

    const safeStatus = (status: string): void => {
      try { onStatus(status) } catch { /* observers cannot break cleanup */ }
    }
    const safeError = (error: unknown): void => {
      try { onError(error) } catch { /* observers cannot create an async failure */ }
    }
    const clearTimer = (): void => {
      if (timer !== undefined) clearTimeout(timer)
      timer = undefined
    }
    const terminal = (error: unknown): void => {
      if (closed) return
      closed = true
      generation += 1
      clearTimer()
      const active = socket
      socket = undefined
      try { active?.close() } catch { /* best effort */ }
      signal?.removeEventListener('abort', abort)
      if (!initialSettled) {
        initialSettled = true
        rejectInitial?.(error)
        resolveInitial = undefined
        rejectInitial = undefined
      } else safeError(error)
    }
    const close = (): void => {
      if (closed) return
      closed = true
      generation += 1
      clearTimer()
      const active = socket
      socket = undefined
      try { active?.close() } catch { /* best effort */ }
      signal?.removeEventListener('abort', abort)
      if (!initialSettled) {
        initialSettled = true
        rejectInitial?.(signal?.reason ?? new Error('WebSocket monitor closed before connection'))
        resolveInitial = undefined
        rejectInitial = undefined
      }
    }
    const failAttempt = (error: unknown, opened: boolean, active: MonitorSocket | undefined, activeGeneration: number, done: { value: boolean }): void => {
      if (closed || done.value || activeGeneration !== generation) return
      done.value = true
      clearTimer()
      generation += 1
      if (socket === active) socket = undefined
      try { active?.close() } catch { /* best effort */ }
      if (!everOpened) {
        terminal(error)
        return
      }
      if (opened) {
        const gap = JSON.stringify({ event: 'gap', url: address, message: 'WebSocket disconnected; messages may have been missed' })
        try { onEvent({ kind: 'gap', text: gap, identity: `gap-${activeGeneration}-${Date.now()}` }) } catch (callbackError) { terminal(callbackError); return }
        if (closed) return
      }
      if (retries >= RETRY_DELAYS_MS.length) {
        terminal(new Error(`WebSocket monitor disconnected and exhausted ${RETRY_DELAYS_MS.length} reconnect attempts: ${String(error)}`))
        return
      }
      safeStatus('reconnecting')
      const delay = RETRY_DELAYS_MS[retries]
      retries += 1
      timer = setTimeout(() => { timer = undefined; connect() }, delay)
    }
    const handleMessage = (value: unknown, activeGeneration: number): void => {
      if (closed || activeGeneration !== generation) return
      if (typeof value !== 'string') {
        terminal(new Error('WebSocket monitor received a binary frame; only text frames are supported'))
        return
      }
      if (Buffer.byteLength(value, 'utf8') > MAX_FRAME_BYTES) {
        terminal(new Error(`WebSocket monitor text frame exceeds ${MAX_FRAME_BYTES} bytes`))
        return
      }
      const identity = createHash('sha256').update(value, 'utf8').digest('hex')
      try { onEvent({ kind: 'message', text: value, identity }) } catch (error) { terminal(error) }
    }
    const connect = (): void => {
      if (closed) return
      const activeGeneration = ++generation
      const done = { value: false }
      let opened = false
      let active: MonitorSocket | undefined
      try { active = createBoundedMonitorSocket(address) } catch (error) {
        failAttempt(error, false, undefined, activeGeneration, done)
        return
      }
      socket = active
      timer = setTimeout(() => {
        if (closed || done.value || activeGeneration !== generation || opened) return
        failAttempt(new Error(`WebSocket monitor connection timed out after ${INITIAL_TIMEOUT_MS}ms`), false, active, activeGeneration, done)
      }, INITIAL_TIMEOUT_MS)
      active.onopen = () => {
        if (closed || done.value || activeGeneration !== generation) return
        opened = true
        clearTimer()
        everOpened = true
        // The budget bounds consecutive failed reconnects. Without this reset a
        // healthy long watch failed on its fourth disconnect of all time.
        retries = 0
        if (!initialSettled) {
          initialSettled = true
          resolveInitial?.()
          resolveInitial = undefined
          rejectInitial = undefined
        }
        safeStatus('connected')
      }
      active.onmessage = event => handleMessage(event.data, activeGeneration)
      active.onerror = error => {
        if (closed || done.value || activeGeneration !== generation) return
        if (error instanceof WebSocketMonitorProtocolError) { terminal(error); return }
        failAttempt(error, opened, active, activeGeneration, done)
      }
      active.onclose = event => {
        if (closed || done.value || activeGeneration !== generation) return
        const reason = event.reason || `close code ${event.code}`
        failAttempt(new Error(`WebSocket monitor closed: ${reason}`), opened, active, activeGeneration, done)
      }
    }
    const abort = (): void => close()
    signal?.addEventListener('abort', abort, { once: true })
    connect()
    await initial
    if (closed) throw signal?.reason ?? new Error('WebSocket monitor closed before connection')
    return Object.freeze({ url: address, close })
  },
}

export { nativeWebSocketMonitorSource }

function validateUrl(value: string): string {
  if (typeof value !== 'string' || !value || value.length > MAX_URL_CHARS || value.includes('\u0000') || value.includes('\n') || value.includes('\r')) {
    throw new Error('WebSocket monitor URL must be a bounded URL without control characters')
  }
  let parsed: URL
  try { parsed = new URL(value) } catch { throw new Error('WebSocket monitor URL is invalid') }
  if (parsed.protocol !== 'wss:' && parsed.protocol !== 'ws:') throw new Error('WebSocket monitor URL must use ws:// or wss://')
  if (parsed.username || parsed.password) throw new Error('WebSocket monitor URL must not contain credentials')
  if (parsed.search || parsed.hash) throw new Error('WebSocket monitor URL must not contain query parameters or fragments')
  if (parsed.protocol === 'ws:' && !isLoopback(parsed.hostname)) throw new Error('Unencrypted ws:// monitors are limited to loopback hosts')
  return parsed.toString()
}

function isLoopback(hostname: string): boolean {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, '')
  return normalized === 'localhost' || normalized === '127.0.0.1' || normalized === '::1'
}
