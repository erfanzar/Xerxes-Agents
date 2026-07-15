// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { StreamEvent } from '../streaming/events.js'
import type { ToolDefinition } from '../types/toolCalls.js'
import { toAcpEvent } from './events.js'
import { AcpPermissionBoard, routePermission } from './permissions.js'
import { AcpSessionStore } from './session.js'
import type {
  AcpModelInfo,
  AcpPromptHandler,
  AcpPromptRunner,
  AcpWireEvent,
} from './types.js'

/** Capability advertisement returned by ACP's `initialize` method. */
export class ServerCapabilities {
  readonly fork: boolean
  readonly permissions: boolean
  readonly protocolVersion: string
  readonly streaming: boolean
  readonly tools: boolean

  constructor(options: Partial<ServerCapabilitiesOptions> = {}) {
    this.protocolVersion = options.protocolVersion ?? '0.9'
    this.streaming = options.streaming ?? true
    this.tools = options.tools ?? true
    this.permissions = options.permissions ?? true
    this.fork = options.fork ?? true
  }

  toWire(): Record<string, boolean | string> {
    return {
      protocol_version: this.protocolVersion,
      streaming: this.streaming,
      tools: this.tools,
      permissions: this.permissions,
      fork: this.fork,
    }
  }
}

export interface ServerCapabilitiesOptions {
  readonly fork: boolean
  readonly permissions: boolean
  readonly protocolVersion: string
  readonly streaming: boolean
  readonly tools: boolean
}

export interface AcpServerOptions {
  readonly capabilities?: ServerCapabilities
  readonly modelListProvider?: () => readonly AcpModelInfo[]
  readonly onSessionClose?: (sessionId: string) => void
  readonly promptHandler?: AcpPromptHandler
  readonly runner?: AcpPromptRunner
  readonly toolListProvider?: () => readonly ToolDefinition[]
}

/**
 * Transport-independent ACP session and prompt facade.
 *
 * It deliberately does not own sockets or streams. `StdioJsonRpcServer` maps
 * JSON-RPC method names onto this class, while embedders can call the same API
 * directly for an HTTP/WebSocket transport.
 */
export class AcpServer {
  readonly capabilities: ServerCapabilities
  readonly permissions = new AcpPermissionBoard()
  readonly sessions = new AcpSessionStore()

  private readonly modelListProvider: () => readonly AcpModelInfo[]
  private readonly onSessionClose: ((sessionId: string) => void) | undefined
  private readonly releasedSessions = new Set<string>()
  private readonly promptHandler: AcpPromptHandler
  private readonly runner: AcpPromptRunner | undefined
  private readonly toolListProvider: () => readonly ToolDefinition[]

  constructor(options: AcpServerOptions) {
    if (!options.promptHandler && !options.runner) {
      throw new Error('AcpServer requires either promptHandler or runner')
    }
    this.capabilities = options.capabilities ?? new ServerCapabilities()
    this.runner = options.runner
    this.promptHandler = options.promptHandler ?? bindRunner(options.runner)
    this.toolListProvider = options.toolListProvider
      ?? (options.runner?.listTools ? () => options.runner?.listTools?.() ?? [] : () => [])
    this.modelListProvider = options.modelListProvider
      ?? (options.runner?.listModels ? () => options.runner?.listModels?.() ?? [] : () => [])
    this.onSessionClose = options.onSessionClose
    options.runner?.setPermissionBoard?.(this.permissions)
  }

  initialize(_clientInfo?: Record<string, unknown>): Record<string, unknown> {
    return {
      server_name: 'xerxes',
      capabilities: this.capabilities.toWire(),
    }
  }

  listTools(): readonly ToolDefinition[] {
    return [...this.toolListProvider()]
  }

  listModels(): readonly AcpModelInfo[] {
    return [...this.modelListProvider()]
  }

  openSession(cwd: string, options: { readonly model?: string | null; readonly title?: string } = {}): Record<string, unknown> {
    const session = this.sessions.create(cwd, options)
    return {
      session_id: session.sessionId,
      cwd: session.cwd,
      model: session.modelOverride,
    }
  }

  listSessions(): readonly Record<string, unknown>[] {
    return this.sessions.list().map(session => ({
      session_id: session.sessionId,
      cwd: session.cwd,
      model: session.modelOverride,
      title: session.title,
      cancelled: session.cancelled,
    }))
  }

  setModel(sessionId: string, model: string | null): Record<string, boolean> {
    return { ok: this.sessions.setModel(sessionId, model) }
  }

  cancel(sessionId: string): Record<string, boolean> {
    const cancelled = this.sessions.cancel(sessionId)
    if (cancelled) {
      this.runner?.cancel(sessionId)
      this.releaseSession(sessionId)
    }
    return { ok: cancelled }
  }

  closeSession(sessionId: string): Record<string, boolean> {
    const closed = this.sessions.drop(sessionId)
    if (closed) {
      this.runner?.cancel(sessionId)
      this.runner?.resetSession?.(sessionId)
      this.releaseSession(sessionId)
    }
    return { ok: closed }
  }

  private releaseSession(sessionId: string): void {
    if (this.releasedSessions.has(sessionId)) return
    this.onSessionClose?.(sessionId)
    this.releasedSessions.add(sessionId)
  }

  async prompt(sessionId: string, text: string, emit?: (event: AcpWireEvent) => void | Promise<void>): Promise<unknown> {
    const session = this.sessions.get(sessionId)
    if (!session) {
      return { error: `unknown session: ${sessionId}` }
    }
    return emit === undefined
      ? this.promptHandler({ session, text })
      : this.promptHandler({ session, text, emit })
  }

  streamEvent(event: StreamEvent | unknown): AcpWireEvent {
    return toAcpEvent(event).toWire()
  }

  requestPermission(options: {
    readonly description: string
    readonly inputs: import('../types/toolCalls.js').JsonObject
    readonly sessionId: string
    readonly toolName: string
  }): Record<string, string> {
    const request = routePermission(options)
    this.permissions.submit(request)
    return { permission_id: request.id }
  }

  respondPermission(permissionId: string, allow: boolean): Record<string, boolean> {
    return { ok: this.permissions.resolve(permissionId, allow) }
  }

  pendingPermissions(): readonly Record<string, unknown>[] {
    return this.permissions.snapshotPending().map(request => ({
      id: request.id,
      session_id: request.sessionId,
      tool_name: request.toolName,
      description: request.description,
      inputs: request.inputs,
    }))
  }

  respondQuestion(inputId: string, answer: string): Record<string, unknown> {
    return this.runner?.respondQuestion?.(inputId, answer) ?? { ok: false }
  }

  pendingQuestions(): readonly Record<string, unknown>[] {
    return this.runner?.pendingQuestions?.() ?? []
  }

  /** Abort every active prompt so transport shutdown cannot wait forever. */
  shutdown(): number {
    let cancelled = 0
    for (const session of this.sessions.list()) {
      this.sessions.cancel(session.sessionId)
      if (this.runner?.cancel(session.sessionId)) cancelled += 1
      this.releaseSession(session.sessionId)
    }
    return cancelled
  }
}

function bindRunner(runner: AcpPromptRunner | undefined): AcpPromptHandler {
  if (!runner) {
    throw new Error('AcpServer runner was not configured')
  }
  return request => runner.runPrompt(request)
}
