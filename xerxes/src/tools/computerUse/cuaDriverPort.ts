// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ComputerUseUnavailableError } from './backend.js'
import type { CuaDriverMcpPort } from './cuaBackend.js'
import type { MCPClient } from '../../mcp/client.js'
import type { MCPToolCallResult } from '../../mcp/types.js'
import type { JsonObject } from '../../types/toolCalls.js'

export interface MCPClientCuaDriverPortOptions {
  readonly isProvisioned?: (signal?: AbortSignal) => boolean | Promise<boolean>
  readonly timeoutMs?: number
}

const ENDED_SESSION = /session '([^'\s]{1,200})' has ended; tool call '[^']*' was rejected/

/** The session id CuaDriver 0.28 names when it refused a call because that session ended, else undefined. */
export function endedSessionId(result: MCPToolCallResult): string | undefined {
  if (result.isError !== true) return undefined
  for (const content of result.content) {
    if (content.type !== 'text') continue
    const match = ENDED_SESSION.exec(content.text)
    if (match) return match[1]
  }
  return undefined
}

/**
 * One lazy, runtime-owned stdio MCP connection. No retries of uncertain actions; the one retry is a
 * call the driver explicitly rejected unexecuted because its session had ended (revived first).
 */
export class MCPClientCuaDriverPort implements CuaDriverMcpPort {
  private connecting: Promise<void> | undefined
  private connected = false
  private disposed = false
  private readonly closed = new AbortController()
  private closing: Promise<void> | undefined
  private readonly timeoutMs: number

  constructor(
    private readonly client: Pick<MCPClient, 'callTool' | 'connect' | 'disconnect'>,
    private readonly options: MCPClientCuaDriverPortOptions = {},
  ) {
    this.timeoutMs = options.timeoutMs ?? 30_000
    if (!Number.isSafeInteger(this.timeoutMs) || this.timeoutMs < 1 || this.timeoutMs > 120_000) {
      throw new Error('CuaDriver timeoutMs must be an integer between 1 and 120000')
    }
  }

  async isAvailable(signal?: AbortSignal): Promise<boolean> {
    this.check(signal)
    return this.options.isProvisioned === undefined || Boolean(await this.options.isProvisioned(signal))
  }

  async start(signal?: AbortSignal): Promise<void> {
    this.check(signal)
    if (this.connected) return
    if (!await this.isAvailable(signal)) throw new ComputerUseUnavailableError('CuaDriver binary is missing; set computer_use_cua_driver / XERXES_CUA_DRIVER to its absolute path')
    this.check(signal)
    this.connecting ??= this.bounded(() => this.client.connect(), signal).then(() => {
      this.check(signal)
      this.connected = true
    })
    // Each waiter keeps its own cancellation signal, even when another caller
    // initiated the shared handshake. Cancellation closes, never retries it.
    await this.bounded(async () => { await this.connecting }, signal)
  }

  async stop(): Promise<void> {
    this.disposed = true
    this.connected = false
    this.closed.abort()
    this.closing ??= this.client.disconnect()
    await this.closing
  }

  async callTool(name: string, arguments_: JsonObject, signal?: AbortSignal): Promise<MCPToolCallResult> {
    await this.start(signal)
    this.check(signal)
    const result = await this.bounded(callSignal => this.client.callTool(name, arguments_, { signal: callSignal }), signal)
    const ended = endedSessionId(result)
    if (ended === undefined || name === 'start_session' || name === 'end_session') return result
    // CuaDriver ends idle sessions and then rejects every call *before running it*, naming the session to
    // revive. Reviving and repeating the call once is therefore safe: the rejected action never happened.
    const revived = await this.bounded(callSignal => this.client.callTool('start_session', { session: ended }, { signal: callSignal }), signal)
    if (revived.isError) return result
    this.check(signal)
    return this.bounded(callSignal => this.client.callTool(name, arguments_, { signal: callSignal }), signal)
  }

  private check(signal?: AbortSignal): void {
    if (signal?.aborted) throw new ComputerUseUnavailableError('CuaDriver call cancelled; no action was retried')
    if (this.disposed) throw new ComputerUseUnavailableError('CuaDriver transport is closed; no action was retried')
  }

  private async bounded<T>(invoke: (signal: AbortSignal) => Promise<T>, signal?: AbortSignal): Promise<T> {
    this.check(signal)
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    let detach: (() => void) | undefined
    try {
      return await Promise.race([
        Promise.resolve().then(() => { this.check(signal); return invoke(controller.signal) }),
        new Promise<never>((_, reject) => {
          const fail = (message: string): void => {
            reject(new ComputerUseUnavailableError(message + '; outcome may be uncertain, capture again; no action was retried'))
            controller.abort()
          }
          timer = setTimeout(() => fail(`CuaDriver timed out after ${this.timeoutMs}ms`), this.timeoutMs)
          const abort = (): void => fail('CuaDriver call cancelled')
          const close = (): void => fail('CuaDriver transport is closed')
          signal?.addEventListener('abort', abort, { once: true })
          this.closed.signal.addEventListener('abort', close, { once: true })
          detach = () => {
            signal?.removeEventListener('abort', abort)
            this.closed.signal.removeEventListener('abort', close)
          }
          if (signal?.aborted) abort()
          if (this.closed.signal.aborted) close()
        }),
      ])
    } catch (error) {
      await this.stop()
      if (error instanceof ComputerUseUnavailableError) throw error
      throw new ComputerUseUnavailableError('CuaDriver is unavailable: ' + (error instanceof Error ? error.message : String(error)), error)
    } finally {
      clearTimeout(timer)
      detach?.()
    }
  }
}
