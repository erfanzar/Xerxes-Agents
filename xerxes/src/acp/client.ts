// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { XERXES_VERSION } from '../core/version.js'
import { errorMessage } from '../executors/toolRegistry.js'
import { MAX_ACP_FRAME_BYTES } from './transport.js'
import {
  ACP_JSON_RPC_ERRORS,
  acpJsonRpcFailure,
  acpJsonRpcSuccess,
  isAcpJsonRpcId,
  isAcpRecord,
  type AcpJsonRpcId,
} from './types.js'

/** Normalized plan entry emitted by an ACP `plan` session update. */
export interface AcpPlanEntry {
  readonly content: string
  readonly priority?: string
  readonly status?: string
}

/** Structured file diff emitted by ACP tool calls, updates, or permission requests. */
export interface AcpToolDiff {
  readonly newText?: string
  readonly oldText?: string
  readonly path: string
}

/** File location hint emitted by ACP tool calls or permission requests. */
export interface AcpToolLocation {
  readonly line?: number
  readonly path: string
}

/** Slash command or skill advertised by an ACP agent via `available_commands_update`. */
export interface AcpAvailableCommand {
  readonly description?: string
  readonly input_hint?: string
  readonly kind?: 'command' | 'skill'
  readonly name: string
  readonly source?: string
}

/** Auth method advertised by an ACP agent during `initialize`. */
export interface AcpAuthMethod {
  readonly description?: string
  readonly id: string
  readonly name: string
}

/** Prompt input capabilities advertised by an ACP agent during `initialize`. */
export interface AcpPromptCapabilities {
  readonly audio: boolean
  readonly embeddedContext: boolean
  readonly image: boolean
}

/** Session lifecycle capabilities advertised by an ACP agent during `initialize`. */
export interface AcpSessionCapabilities {
  readonly additionalDirectories: boolean
  readonly close: boolean
  readonly list: boolean
  readonly load: boolean
  readonly resume: boolean
}

/** MCP transport capabilities advertised by an ACP agent during `initialize`. */
export interface AcpMcpCapabilities {
  readonly http: boolean
  readonly sse: boolean
}

/** Content block sent inside an ACP `session/prompt` request. */
export type AcpPromptContentBlock =
  | { readonly text: string; readonly type: 'text' }
  | { readonly data: string; readonly mimeType: string; readonly type: 'image' }
  | {
    readonly resource: {
      readonly mimeType?: string
      readonly text: string
      readonly uri: string
    }
    readonly type: 'resource'
  }

/** Token usage reported on an ACP `session/prompt` result. */
export interface AcpPromptUsage {
  readonly cachedReadTokens?: number
  readonly cachedWriteTokens?: number
  readonly inputTokens?: number
  readonly outputTokens?: number
  readonly thoughtTokens?: number
  readonly totalTokens?: number
}

/** Normalized stream events emitted by {@link AcpStdioClient}. */
export type AcpClientEvent =
  | { readonly sessionId: string; readonly text: string; readonly type: 'text_delta' }
  | { readonly sessionId: string; readonly text: string; readonly type: 'thought_delta' }
  | {
    readonly diff?: AcpToolDiff
    readonly kind?: string
    readonly locations?: readonly AcpToolLocation[]
    readonly name: string
    readonly rawInput?: unknown
    readonly sessionId: string
    readonly status?: string
    readonly title?: string
    readonly toolCallId: string
    readonly type: 'tool_call'
  }
  | {
    readonly content?: string
    readonly diff?: AcpToolDiff
    readonly isError?: boolean
    readonly locations?: readonly AcpToolLocation[]
    readonly name?: string
    readonly permissionDenied?: boolean
    readonly rawInput?: unknown
    readonly rawOutput?: unknown
    readonly sessionId: string
    readonly status?: string
    readonly title?: string
    readonly toolCallId: string
    readonly type: 'tool_call_update'
  }
  | { readonly entries: readonly AcpPlanEntry[]; readonly sessionId: string; readonly type: 'plan' }
  | {
    readonly modeId: string
    readonly modeName?: string
    readonly sessionId: string
    readonly type: 'mode_update'
  }
  | {
    readonly modelId: string
    readonly modelName?: string
    readonly sessionId: string
    readonly type: 'model_update'
  }
  | {
    readonly configId?: string
    readonly configOptions?: readonly unknown[]
    readonly rawUpdate: Readonly<Record<string, unknown>>
    readonly sessionId: string
    readonly type: 'config_update'
    readonly value?: string
  }
  | {
    readonly cachedReadTokens?: number
    readonly cachedWriteTokens?: number
    readonly contextSize?: number
    readonly contextUsed?: number
    readonly inputTokens?: number
    readonly outputTokens?: number
    readonly sessionId: string
    readonly thoughtTokens?: number
    readonly totalTokens?: number
    readonly type: 'usage_update'
  }
  | {
    readonly sessionId: string
    readonly title?: string
    readonly type: 'session_info'
    readonly updatedAt?: string
  }
  | {
    readonly commands: readonly AcpAvailableCommand[]
    readonly sessionId: string
    readonly type: 'available_commands'
  }
  | {
    readonly rawUpdate: Readonly<Record<string, unknown>>
    readonly sessionId: string
    readonly text?: string
    readonly type: 'continuation'
  }
  | { readonly text: string; readonly type: 'stderr' }
  | { readonly message: string; readonly type: 'error' }
  | { readonly code: number | null; readonly signal: string | null; readonly type: 'exit' }

export interface AcpPermissionOption {
  readonly kind?: string
  readonly name: string
  readonly optionId: string
}

export interface AcpClientPermissionRequest {
  readonly diff?: AcpToolDiff
  readonly kind?: string
  readonly locations?: readonly AcpToolLocation[]
  readonly options: readonly AcpPermissionOption[]
  readonly rawInput?: unknown
  readonly sessionId: string
  readonly title: string
  readonly toolCallId: string
  readonly toolName: string
}

export interface AcpClientPermissionResponse {
  readonly optionId?: string
  readonly outcome: 'cancelled' | 'selected'
}

export interface AcpReadTextFileRequest {
  readonly limit?: number
  readonly line?: number
  readonly path: string
  readonly sessionId: string
}

export interface AcpWriteTextFileRequest {
  readonly content: string
  readonly path: string
  readonly sessionId: string
}

export type AcpElicitationFieldType = 'string' | 'number' | 'integer' | 'boolean' | 'array'

export interface AcpElicitationOption {
  readonly const: string
  readonly title: string
}

export interface AcpElicitationField {
  readonly default?: unknown
  readonly description?: string
  readonly enum?: readonly string[]
  readonly enumNames?: readonly string[]
  readonly items?: {
    readonly anyOf?: readonly AcpElicitationOption[]
    readonly enum?: readonly string[]
    readonly oneOf?: readonly AcpElicitationOption[]
    readonly type?: string
  }
  readonly key: string
  readonly oneOf?: readonly AcpElicitationOption[]
  readonly required: boolean
  readonly title?: string
  readonly type: AcpElicitationFieldType
}

export interface AcpElicitationRequest {
  readonly fields: readonly AcpElicitationField[]
  readonly message: string
  readonly mode: 'form' | 'url' | string
  readonly rawParams: Readonly<Record<string, unknown>>
  readonly requestedSchema?: Readonly<Record<string, unknown>>
  readonly sessionId: string
  readonly toolCallId?: string
  readonly url?: string
}

export interface AcpElicitationResponse {
  readonly action: 'accept' | 'decline' | 'cancel'
  readonly content?: Readonly<Record<string, unknown>>
}

/** Host callbacks answering agent-to-client JSON-RPC requests. */
export interface AcpClientHostHandlers {
  readonly createElicitation?: (
    request: AcpElicitationRequest,
  ) => Promise<AcpElicitationResponse> | AcpElicitationResponse
  readonly readTextFile?: (
    request: AcpReadTextFileRequest,
  ) => Promise<{ readonly content: string }> | { readonly content: string }
  readonly requestPermission?: (
    request: AcpClientPermissionRequest,
  ) => Promise<AcpClientPermissionResponse> | AcpClientPermissionResponse
  readonly writeTextFile?: (
    request: AcpWriteTextFileRequest,
  ) => Promise<void> | void
}

/** Injectable process/stream seam for deterministic offline unit tests. */
export interface AcpSpawnedProcess {
  readonly exited: Promise<number | null>
  readonly kill: () => void
  readonly stderr?: ReadableStream<Uint8Array> | null
  readonly stdout: ReadableStream<Uint8Array>
  readonly writeLine: (line: string) => void | Promise<void>
}

export type AcpProcessSpawner = (
  command: string,
  args: readonly string[],
  options: {
    readonly cwd?: string
    readonly env?: Readonly<Record<string, string | undefined>>
  },
) => AcpSpawnedProcess

export interface AcpInitializeParams {
  readonly clientCapabilities?: Readonly<Record<string, unknown>>
  readonly clientInfo?: {
    readonly name: string
    readonly version: string
  }
  readonly protocolVersion?: number
}

export interface AcpInitializeResult {
  readonly agentCapabilities: Readonly<Record<string, unknown>>
  readonly agentInfo?: {
    readonly name?: string
    readonly title?: string
    readonly version?: string
  }
  readonly authMethods: readonly AcpAuthMethod[]
  readonly capabilitiesMeta?: Readonly<Record<string, unknown>>
  readonly mcpCapabilities: AcpMcpCapabilities
  readonly promptCapabilities: AcpPromptCapabilities
  readonly protocolVersion: number | string
  readonly raw: Readonly<Record<string, unknown>>
  readonly sessionCapabilities: AcpSessionCapabilities
  readonly supportsLoadSession: boolean
}

export interface AcpNewSessionParams {
  readonly cwd: string
  readonly mcpServers?: readonly Readonly<Record<string, unknown>>[]
}

export interface AcpNewSessionResult {
  readonly configOptions?: readonly unknown[]
  readonly models?: unknown
  readonly modes?: unknown
  readonly raw: Readonly<Record<string, unknown>>
  readonly sessionId: string
}

export interface AcpLoadSessionParams {
  readonly cwd: string
  readonly mcpServers?: readonly Readonly<Record<string, unknown>>[]
  readonly sessionId: string
}

export interface AcpLoadSessionResult {
  readonly configOptions?: readonly unknown[]
  readonly models?: unknown
  readonly modes?: unknown
  readonly raw: Readonly<Record<string, unknown>>
  readonly sessionId: string
}

export interface AcpResumeSessionParams {
  readonly cwd: string
  readonly mcpServers?: readonly Readonly<Record<string, unknown>>[]
  readonly sessionId: string
}

export interface AcpResumeSessionResult {
  readonly configOptions?: readonly unknown[]
  readonly models?: unknown
  readonly modes?: unknown
  readonly raw: Readonly<Record<string, unknown>>
  readonly sessionId: string
}

export interface AcpSetModelParams {
  readonly configId?: string
  readonly modelId: string
  readonly sessionId: string
}

export interface AcpSetModeParams {
  readonly configId?: string
  readonly modeId: string
  readonly sessionId: string
}

export interface AcpSetConfigOptionParams {
  readonly configId: string
  readonly sessionId: string
  readonly value: string
}

/** Typed JSON-RPC 2.0 error returned by an external ACP agent. */
export class AcpRpcError extends Error {
  readonly code: number
  readonly method: string

  constructor(method: string, code: number, message: string) {
    super(`ACP RPC "${method}" error (${code}): ${message}`)
    this.name = 'AcpRpcError'
    this.code = code
    this.method = method
  }
}

export interface AcpPromptParams {
  readonly cancelGraceMs?: number
  readonly content?: readonly AcpPromptContentBlock[]
  readonly onEvent?: (event: AcpClientEvent) => void
  readonly sessionId: string
  readonly signal?: AbortSignal
  readonly text: string
}

export interface AcpPromptResult {
  readonly raw: Readonly<Record<string, unknown>>
  readonly stopReason: string
  readonly usage?: AcpPromptUsage
}

export interface AcpStdioClientOptions {
  readonly args?: readonly string[]
  readonly command: string
  readonly cwd?: string
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly handlers?: AcpClientHostHandlers
  readonly onEvent?: (event: AcpClientEvent) => void
  readonly requestTimeoutMs?: number
  readonly spawn?: AcpProcessSpawner
  readonly transport?: AcpSpawnedProcess
}

interface PendingRpc {
  readonly method: string
  readonly reject: (error: Error) => void
  readonly resolve: (result: unknown) => void
  readonly timer: ReturnType<typeof setTimeout> | undefined
}

const DEFAULT_RPC_TIMEOUT_MS = 120_000
const DISPOSE_EXIT_WAIT_MS = 1_500
const DEFAULT_CANCEL_GRACE_MS = 500
const STDERR_EXIT_DRAIN_WAIT_MS = 50

/**
 * Generic bidirectional ACP stdio JSON-RPC 2.0 client.
 *
 * Spawns an external ACP agent subprocess (or attaches to an injected stream
 * transport) over newline-delimited JSON-RPC 2.0, handles client-to-agent RPCs
 * (`initialize`, `session/new`, `session/load`, `session/prompt`,
 * `session/cancel`), normalizes incoming `session/update` notifications
 * (including post-settle continuation frames), and answers incoming
 * agent-to-client requests (`session/request_permission`, `elicitation/create`,
 * `fs/read_text_file`, `fs/write_text_file`) through injectable host callbacks.
 */
export class AcpStdioClient {
  private readonly command: string
  private readonly args: readonly string[]
  private readonly cwd: string | undefined
  private readonly env: Readonly<Record<string, string | undefined>> | undefined
  private readonly requestTimeoutMs: number
  private readonly spawner: AcpProcessSpawner | undefined
  private readonly injectedTransport: AcpSpawnedProcess | undefined
  private handlers: AcpClientHostHandlers
  private readonly listeners = new Set<(event: AcpClientEvent) => void>()
  private readonly promptListeners = new Map<string, (event: AcpClientEvent) => void>()
  private readonly pending = new Map<number, PendingRpc>()
  private readonly settledSessions = new Set<string>()
  private readonly stderrTail: string[] = []
  private proc: AcpSpawnedProcess | undefined
  private stdoutReader: ReturnType<ReadableStream<Uint8Array>['getReader']> | undefined
  private stderrReader: ReturnType<ReadableStream<Uint8Array>['getReader']> | undefined
  private stdoutTask: Promise<void> | undefined
  private stderrTask: Promise<void> | undefined
  private writeQueue: Promise<void> = Promise.resolve()
  private nextId = 1
  private disposed = false
  private started = false
  private exited = false
  private lastInitializeResult: AcpInitializeResult | undefined

  constructor(options: AcpStdioClientOptions) {
    this.command = options.command
    this.args = [...(options.args ?? [])]
    this.cwd = options.cwd
    this.env = options.env
    this.handlers = options.handlers ?? {}
    this.requestTimeoutMs = options.requestTimeoutMs ?? DEFAULT_RPC_TIMEOUT_MS
    this.spawner = options.spawn
    this.injectedTransport = options.transport
    if (options.onEvent) {
      this.listeners.add(options.onEvent)
    }
  }

  /** Replace or update host callbacks (e.g., per-turn approval forwarding). */
  setHandlers(handlers: AcpClientHostHandlers): void {
    this.handlers = handlers
  }

  /** Subscribe to all normalized ACP client events. */
  onEvent(listener: (event: AcpClientEvent) => void): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  /** Recent stderr lines captured from the child process. */
  get recentStderr(): string {
    return this.stderrTail.join('\n').trim()
  }

  get isDisposed(): boolean {
    return this.disposed
  }

  get hasExited(): boolean {
    return this.exited
  }

  get initializeResult(): AcpInitializeResult | undefined {
    return this.lastInitializeResult
  }

  /** Start the underlying subprocess/transport if not already running. */
  start(): void {
    if (this.disposed) {
      throw new Error('ACP client has been disposed')
    }
    if (this.started) {
      return
    }
    this.started = true
    const proc = this.injectedTransport
      ?? (this.spawner
        ? this.spawner(this.command, this.args, {
            ...(this.cwd !== undefined ? { cwd: this.cwd } : {}),
            ...(this.env !== undefined ? { env: this.env } : {}),
          })
        : spawnBunAcpProcess(this.command, this.args, {
            ...(this.cwd !== undefined ? { cwd: this.cwd } : {}),
            ...(this.env !== undefined ? { env: this.env } : {}),
          }))
    this.proc = proc
    this.stdoutTask = this.readStdout(proc.stdout)
    if (proc.stderr) {
      this.stderrTask = this.readStderr(proc.stderr)
    }
    void proc.exited.then(
      async code => {
        this.exited = true
        if (!this.disposed && this.stderrTask) {
          await raceWithUnrefTimeout(this.stderrTask.catch(() => undefined), STDERR_EXIT_DRAIN_WAIT_MS)
        }
        this.emit({ type: 'exit', code, signal: null })
        if (!this.disposed && this.pending.size > 0) {
          const detail = this.recentStderr ? `: ${this.recentStderr}` : ''
          this.rejectAllPending(
            new Error(`ACP agent "${this.command}" exited unexpectedly (code ${code ?? 'unknown'})${detail}`),
          )
        }
      },
      async error => {
        this.exited = true
        if (!this.disposed && this.stderrTask) {
          await raceWithUnrefTimeout(this.stderrTask.catch(() => undefined), STDERR_EXIT_DRAIN_WAIT_MS)
        }
        if (!this.disposed) {
          this.emit({ type: 'error', message: errorMessage(error) })
          this.rejectAllPending(new Error(`ACP agent "${this.command}" failed: ${errorMessage(error)}`))
        }
      },
    )
  }

  async initialize(params: AcpInitializeParams = {}): Promise<AcpInitializeResult> {
    this.start()
    const payload = {
      protocolVersion: params.protocolVersion ?? 1,
      clientInfo: params.clientInfo ?? {
        name: 'xerxes-agents',
        version: XERXES_VERSION,
      },
      clientCapabilities: params.clientCapabilities ?? {
        fs: {
          readTextFile: true,
          writeTextFile: true,
        },
        terminal: false,
        ...(this.handlers.createElicitation ? { elicitation: { form: {} } } : {}),
      },
    }
    const rawResult = await this.call('initialize', payload)
    const raw = isAcpRecord(rawResult) ? rawResult : {}
    const agentCapabilities = isAcpRecord(raw.agentCapabilities)
      ? raw.agentCapabilities
      : isAcpRecord(raw.capabilities)
        ? raw.capabilities
        : {}
    const rawAgentInfo = isAcpRecord(raw.agentInfo) ? raw.agentInfo : undefined
    const agentInfo = rawAgentInfo
      ? {
          ...(typeof rawAgentInfo.name === 'string' ? { name: rawAgentInfo.name } : {}),
          ...(typeof rawAgentInfo.title === 'string' ? { title: rawAgentInfo.title } : {}),
          ...(typeof rawAgentInfo.version === 'string' ? { version: rawAgentInfo.version } : {}),
        }
      : typeof raw.server_name === 'string'
        ? { name: raw.server_name }
        : undefined
    const protocolVersion =
      typeof raw.protocolVersion === 'number' || typeof raw.protocolVersion === 'string'
        ? raw.protocolVersion
        : typeof agentCapabilities.protocol_version === 'string'
          ? agentCapabilities.protocol_version
          : 1
    const supportsLoadSession =
      agentCapabilities.loadSession === true || agentCapabilities.load_session === true
    const rawPromptCaps = isAcpRecord(agentCapabilities.promptCapabilities)
      ? agentCapabilities.promptCapabilities
      : isAcpRecord(agentCapabilities.prompt_capabilities)
        ? agentCapabilities.prompt_capabilities
        : {}
    const promptCapabilities: AcpPromptCapabilities = {
      image: rawPromptCaps.image === true,
      audio: rawPromptCaps.audio === true,
      embeddedContext:
        rawPromptCaps.embeddedContext === true || rawPromptCaps.embedded_context === true,
    }
    const rawSessionCaps = isAcpRecord(agentCapabilities.sessionCapabilities)
      ? agentCapabilities.sessionCapabilities
      : isAcpRecord(agentCapabilities.session_capabilities)
        ? agentCapabilities.session_capabilities
        : {}
    const hasSessionCap = (value: unknown): boolean =>
      value !== undefined && value !== false && value !== null
    const sessionCapabilities: AcpSessionCapabilities = {
      resume: hasSessionCap(rawSessionCaps.resume),
      close: hasSessionCap(rawSessionCaps.close),
      load: supportsLoadSession || hasSessionCap(rawSessionCaps.load),
      list: hasSessionCap(rawSessionCaps.list),
      additionalDirectories:
        hasSessionCap(rawSessionCaps.additionalDirectories) ||
        hasSessionCap(rawSessionCaps.additional_directories),
    }
    const rawMcpCaps = isAcpRecord(agentCapabilities.mcpCapabilities)
      ? agentCapabilities.mcpCapabilities
      : isAcpRecord(agentCapabilities.mcp_capabilities)
        ? agentCapabilities.mcp_capabilities
        : {}
    const mcpCapabilities: AcpMcpCapabilities = {
      http: rawMcpCaps.http === true,
      sse: rawMcpCaps.sse === true,
    }
    const capabilitiesMeta = isAcpRecord(agentCapabilities._meta)
      ? agentCapabilities._meta
      : isAcpRecord(agentCapabilities.meta)
        ? agentCapabilities.meta
        : undefined
    const authMethods = parseAuthMethods(raw.authMethods ?? raw.auth_methods)
    const initResult: AcpInitializeResult = {
      protocolVersion,
      agentCapabilities,
      ...(agentInfo ? { agentInfo } : {}),
      authMethods,
      ...(capabilitiesMeta ? { capabilitiesMeta } : {}),
      mcpCapabilities,
      promptCapabilities,
      sessionCapabilities,
      supportsLoadSession,
      raw,
    }
    this.lastInitializeResult = initResult
    return initResult
  }

  async newSession(params: AcpNewSessionParams): Promise<AcpNewSessionResult> {
    this.start()
    const rawResult = await this.call('session/new', {
      cwd: params.cwd,
      mcpServers: params.mcpServers ?? [],
    })
    const raw = isAcpRecord(rawResult) ? rawResult : {}
    const sessionId = readStringField(raw, 'sessionId', 'session_id')
    if (!sessionId) {
      throw new Error('ACP agent session/new response did not include a sessionId')
    }
    const configOptions = Array.isArray(raw.configOptions)
      ? raw.configOptions
      : Array.isArray(raw.config_options)
        ? raw.config_options
        : undefined
    return {
      sessionId,
      ...(configOptions !== undefined ? { configOptions } : {}),
      ...(raw.models !== undefined ? { models: raw.models } : {}),
      ...(raw.modes !== undefined ? { modes: raw.modes } : {}),
      raw,
    }
  }

  async loadSession(params: AcpLoadSessionParams): Promise<AcpLoadSessionResult> {
    this.start()
    const rawResult = await this.call('session/load', {
      sessionId: params.sessionId,
      cwd: params.cwd,
      mcpServers: params.mcpServers ?? [],
    })
    const raw = isAcpRecord(rawResult) ? rawResult : {}
    const sessionId = readStringField(raw, 'sessionId', 'session_id') || params.sessionId
    const configOptions = Array.isArray(raw.configOptions)
      ? raw.configOptions
      : Array.isArray(raw.config_options)
        ? raw.config_options
        : undefined
    return {
      sessionId,
      ...(configOptions !== undefined ? { configOptions } : {}),
      ...(raw.models !== undefined ? { models: raw.models } : {}),
      ...(raw.modes !== undefined ? { modes: raw.modes } : {}),
      raw,
    }
  }

  async resumeSession(params: AcpResumeSessionParams): Promise<AcpResumeSessionResult> {
    this.start()
    const rawResult = await this.call('session/resume', {
      sessionId: params.sessionId,
      cwd: params.cwd,
      mcpServers: params.mcpServers ?? [],
    })
    const raw = isAcpRecord(rawResult) ? rawResult : {}
    const sessionId = readStringField(raw, 'sessionId', 'session_id') || params.sessionId
    const configOptions = Array.isArray(raw.configOptions)
      ? raw.configOptions
      : Array.isArray(raw.config_options)
        ? raw.config_options
        : undefined
    return {
      sessionId,
      ...(configOptions !== undefined ? { configOptions } : {}),
      ...(raw.models !== undefined ? { models: raw.models } : {}),
      ...(raw.modes !== undefined ? { modes: raw.modes } : {}),
      raw,
    }
  }

  async closeSession(sessionId: string, timeoutMs = 500): Promise<void> {
    if (this.disposed || !this.started || this.exited) {
      return
    }
    await this.call('session/close', { sessionId }, timeoutMs)
  }

  async setConfigOption(params: AcpSetConfigOptionParams): Promise<Record<string, unknown>> {
    this.start()
    const rawResult = await this.call('session/set_config_option', {
      sessionId: params.sessionId,
      configId: params.configId,
      value: params.value,
    })
    return isAcpRecord(rawResult) ? rawResult : {}
  }

  async setModel(params: AcpSetModelParams): Promise<Record<string, unknown>> {
    this.start()
    const configId = params.configId?.trim()
    let rawResult: unknown
    if (configId) {
      try {
        rawResult = await this.call('session/set_config_option', {
          sessionId: params.sessionId,
          configId,
          value: params.modelId,
        })
      } catch (error) {
        if (!isMethodNotFoundError(error)) {
          throw error
        }
        rawResult = await this.call('session/set_model', {
          sessionId: params.sessionId,
          modelId: params.modelId,
        })
      }
    } else {
      try {
        rawResult = await this.call('session/set_model', {
          sessionId: params.sessionId,
          modelId: params.modelId,
        })
      } catch (error) {
        if (!isMethodNotFoundError(error)) {
          throw error
        }
        rawResult = await this.call('session/set_config_option', {
          sessionId: params.sessionId,
          configId: 'model',
          value: params.modelId,
        })
      }
    }
    return isAcpRecord(rawResult) ? rawResult : {}
  }

  async setMode(params: AcpSetModeParams): Promise<Record<string, unknown>> {
    this.start()
    const configId = params.configId?.trim()
    let rawResult: unknown
    if (configId) {
      try {
        rawResult = await this.call('session/set_config_option', {
          sessionId: params.sessionId,
          configId,
          value: params.modeId,
        })
      } catch (error) {
        if (!isMethodNotFoundError(error)) {
          throw error
        }
        rawResult = await this.call('session/set_mode', {
          sessionId: params.sessionId,
          modeId: params.modeId,
        })
      }
    } else {
      try {
        rawResult = await this.call('session/set_mode', {
          sessionId: params.sessionId,
          modeId: params.modeId,
        })
      } catch (error) {
        if (!isMethodNotFoundError(error)) {
          throw error
        }
        rawResult = await this.call('session/set_config_option', {
          sessionId: params.sessionId,
          configId: 'mode',
          value: params.modeId,
        })
      }
    }
    return isAcpRecord(rawResult) ? rawResult : {}
  }

  async prompt(params: AcpPromptParams): Promise<AcpPromptResult> {
    this.start()
    if (params.signal?.aborted) {
      throw new Error('ACP prompt aborted before start')
    }
    this.settledSessions.delete(params.sessionId)
    if (params.onEvent) {
      this.promptListeners.set(params.sessionId, params.onEvent)
    }

    const promptBlocks: readonly AcpPromptContentBlock[] =
      params.content && params.content.length > 0
        ? params.content
        : [{ type: 'text', text: params.text }]

    const { id: rpcId, promise: rpcPromise } = this.callWithId(
      'session/prompt',
      {
        sessionId: params.sessionId,
        session_id: params.sessionId,
        prompt: promptBlocks,
        text: params.text,
      },
      // Prompts can run for many minutes while tools execute; rely on explicit cancel/abort.
      0,
    )

    const cancelGraceMs = Math.max(0, params.cancelGraceMs ?? DEFAULT_CANCEL_GRACE_MS)
    let onAbort: (() => void) | undefined
    let graceTimer: ReturnType<typeof setTimeout> | undefined
    const abortPromise = params.signal
      ? new Promise<never>((_resolve, reject) => {
          onAbort = () => {
            void this.cancel(params.sessionId).catch(() => undefined)
            const forceCancel = () => {
              const pending = this.pending.get(rpcId)
              if (pending) {
                this.pending.delete(rpcId)
                if (pending.timer) clearTimeout(pending.timer)
              }
              reject(new Error('ACP prompt cancelled'))
            }
            if (cancelGraceMs === 0) {
              forceCancel()
              return
            }
            graceTimer = setTimeout(forceCancel, cancelGraceMs)
            graceTimer.unref?.()
          }
          params.signal!.addEventListener('abort', onAbort, { once: true })
        })
      : undefined

    try {
      const rawResult = abortPromise
        ? await Promise.race([rpcPromise, abortPromise])
        : await rpcPromise
      const raw = isAcpRecord(rawResult) ? rawResult : {}
      const stopReason = readStringField(raw, 'stopReason', 'stop_reason') || 'end_turn'
      const usage = parsePromptUsage(raw.usage)
      return {
        stopReason,
        raw,
        ...(usage ? { usage } : {}),
      }
    } finally {
      if (graceTimer !== undefined) {
        clearTimeout(graceTimer)
      }
      this.settledSessions.add(params.sessionId)
      if (params.onEvent && this.promptListeners.get(params.sessionId) === params.onEvent) {
        this.promptListeners.delete(params.sessionId)
      }
      if (params.signal && onAbort) {
        params.signal.removeEventListener('abort', onAbort)
      }
    }
  }

  async cancel(sessionId: string): Promise<void> {
    if (this.disposed || !this.started) {
      return
    }
    await this.sendFrame({
      jsonrpc: '2.0',
      method: 'session/cancel',
      params: {
        sessionId,
        session_id: sessionId,
      },
    })
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return
    }
    this.disposed = true
    this.promptListeners.clear()
    this.rejectAllPending(new Error('ACP client disposed'))
    const proc = this.proc
    this.proc = undefined
    if (proc) {
      try {
        proc.kill()
      } catch {
        // Ignore kill failures on already-exited processes.
      }
    }
    if (this.stdoutReader) {
      await this.stdoutReader.cancel().catch(() => undefined)
    }
    if (this.stderrReader) {
      await this.stderrReader.cancel().catch(() => undefined)
    }
    if (proc && !this.exited) {
      await raceWithUnrefTimeout(proc.exited.catch(() => null), DISPOSE_EXIT_WAIT_MS)
    }
    await Promise.allSettled([
      this.stdoutTask ?? Promise.resolve(),
      this.stderrTask ?? Promise.resolve(),
      this.writeQueue,
    ])
  }

  private call(
    method: string,
    params: Readonly<Record<string, unknown>>,
    timeoutMs: number = this.requestTimeoutMs,
  ): Promise<unknown> {
    return this.callWithId(method, params, timeoutMs).promise
  }

  private callWithId(
    method: string,
    params: Readonly<Record<string, unknown>>,
    timeoutMs: number = this.requestTimeoutMs,
  ): { readonly id: number; readonly promise: Promise<unknown> } {
    const id = this.nextId++
    if (this.disposed) {
      return { id, promise: Promise.reject(new Error('ACP client disposed')) }
    }
    if (!this.proc) {
      return { id, promise: Promise.reject(new Error('ACP client transport not started')) }
    }
    const promise = new Promise<unknown>((resolve, reject) => {
      const timer = timeoutMs > 0
        ? setTimeout(() => {
            if (this.pending.delete(id)) {
              reject(new Error(`ACP RPC "${method}" timed out after ${timeoutMs}ms`))
            }
          }, timeoutMs)
        : undefined
      timer?.unref?.()
      this.pending.set(id, { method, resolve, reject, timer })
      void this.sendFrame({
        jsonrpc: '2.0',
        id,
        method,
        params,
      }).catch(error => {
        const entry = this.pending.get(id)
        if (entry) {
          this.pending.delete(id)
          if (entry.timer) clearTimeout(entry.timer)
          reject(error instanceof Error ? error : new Error(errorMessage(error)))
        }
      })
    })
    return { id, promise }
  }

  private sendFrame(frame: unknown): Promise<void> {
    const proc = this.proc
    if (!proc || this.disposed) {
      return Promise.reject(new Error('ACP client is not connected'))
    }
    const line = `${JSON.stringify(frame)}\n`
    const sent = this.writeQueue.then(() => proc.writeLine(line)).then(() => undefined)
    this.writeQueue = sent.catch(() => undefined)
    return sent
  }

  private rejectAllPending(error: Error): void {
    for (const [id, entry] of this.pending) {
      this.pending.delete(id)
      if (entry.timer) clearTimeout(entry.timer)
      entry.reject(error)
    }
  }

  private async readStdout(stream: ReadableStream<Uint8Array>): Promise<void> {
    const reader = stream.getReader()
    this.stdoutReader = reader
    const decoder = new TextDecoder()
    let buffer = ''
    try {
      while (!this.disposed) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }
        buffer += decoder.decode(value, { stream: true })
        let newline = buffer.indexOf('\n')
        while (newline >= 0) {
          const rawLine = buffer.slice(0, newline)
          buffer = buffer.slice(newline + 1)
          if (Buffer.byteLength(rawLine, 'utf8') > MAX_ACP_FRAME_BYTES) {
            this.emit({
              type: 'error',
              message: `ACP frame exceeded maximum size of ${MAX_ACP_FRAME_BYTES} bytes`,
            })
            continue
          }
          const trimmed = rawLine.trim()
          if (trimmed) {
            this.handleInboundLine(trimmed)
          }
          newline = buffer.indexOf('\n')
        }
      }
      buffer += decoder.decode()
      const tail = buffer.trim()
      if (!this.disposed && tail) {
        this.handleInboundLine(tail)
      }
    } catch (error) {
      if (!this.disposed) {
        this.emit({ type: 'error', message: errorMessage(error) })
      }
    } finally {
      if (this.stdoutReader === reader) {
        this.stdoutReader = undefined
      }
      try {
        reader.releaseLock()
      } catch {
        // Stream already cancelled/closed.
      }
    }
  }

  private async readStderr(stream: ReadableStream<Uint8Array>): Promise<void> {
    const reader = stream.getReader()
    this.stderrReader = reader
    const decoder = new TextDecoder()
    let buffer = ''
    try {
      while (!this.disposed) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }
        buffer += decoder.decode(value, { stream: true })
        let newline = buffer.indexOf('\n')
        while (newline >= 0) {
          const line = buffer.slice(0, newline).trim()
          buffer = buffer.slice(newline + 1)
          if (line) {
            this.recordStderr(line)
          }
          newline = buffer.indexOf('\n')
        }
      }
      buffer += decoder.decode()
      const tail = buffer.trim()
      if (!this.disposed && tail) {
        this.recordStderr(tail)
      }
    } catch {
      // Ignore stderr stream cancellation during disposal.
    } finally {
      if (this.stderrReader === reader) {
        this.stderrReader = undefined
      }
      try {
        reader.releaseLock()
      } catch {
        // Stream already closed.
      }
    }
  }

  private recordStderr(line: string): void {
    this.stderrTail.push(line)
    if (this.stderrTail.length > 50) {
      this.stderrTail.shift()
    }
    this.emit({ type: 'stderr', text: line })
  }

  private handleInboundLine(line: string): void {
    let parsed: unknown
    try {
      parsed = JSON.parse(line) as unknown
    } catch {
      // Some external agents print non-JSON diagnostic lines on stdout; ignore safely.
      return
    }
    if (!isAcpRecord(parsed)) {
      return
    }

    const hasMethod = typeof parsed.method === 'string'
    const hasId = Object.hasOwn(parsed, 'id') && parsed.id !== undefined

    // 1. Response to one of our outgoing requests.
    if (!hasMethod && hasId && typeof parsed.id === 'number') {
      const pending = this.pending.get(parsed.id)
      if (!pending) {
        return
      }
      this.pending.delete(parsed.id)
      if (pending.timer) clearTimeout(pending.timer)
      if (isAcpRecord(parsed.error)) {
        const message = typeof parsed.error.message === 'string'
          ? parsed.error.message
          : `ACP RPC "${pending.method}" failed`
        const code = typeof parsed.error.code === 'number' ? parsed.error.code : -32000
        pending.reject(new AcpRpcError(pending.method, code, message))
      } else {
        pending.resolve(parsed.result)
      }
      return
    }

    // 2. Agent-to-client request (has both `method` and `id`).
    if (hasMethod && hasId && isAcpJsonRpcId(parsed.id)) {
      const params = isAcpRecord(parsed.params) ? parsed.params : {}
      void this.handleAgentRequest(parsed.id, parsed.method as string, params)
      return
    }

    // 3. Agent-to-client notification (has `method`, no `id`).
    if (hasMethod) {
      const params = isAcpRecord(parsed.params) ? parsed.params : {}
      this.handleAgentNotification(parsed.method as string, params)
    }
  }

  private async handleAgentRequest(
    id: AcpJsonRpcId,
    method: string,
    params: Readonly<Record<string, unknown>>,
  ): Promise<void> {
    try {
      switch (method) {
        case 'session/request_permission': {
          const request = parsePermissionRequest(params)
          const response = await this.resolvePermission(request)
          await this.sendFrame(acpJsonRpcSuccess(id, formatPermissionRpcResult(response)))
          return
        }
        case 'elicitation/create': {
          if (!this.handlers.createElicitation) {
            await this.sendFrame(
              acpJsonRpcFailure(
                id,
                ACP_JSON_RPC_ERRORS.methodNotFound,
                'elicitation/create handler is not configured on this ACP client',
              ),
            )
            return
          }
          const request = parseElicitationRequest(params)
          const response = await this.resolveElicitation(request)
          await this.sendFrame(acpJsonRpcSuccess(id, formatElicitationRpcResult(response)))
          return
        }
        case 'fs/read_text_file': {
          if (!this.handlers.readTextFile) {
            await this.sendFrame(
              acpJsonRpcFailure(
                id,
                ACP_JSON_RPC_ERRORS.methodNotFound,
                'fs/read_text_file handler is not configured on this ACP client',
              ),
            )
            return
          }
          const req: AcpReadTextFileRequest = {
            sessionId: readStringField(params, 'sessionId', 'session_id'),
            path: readStringField(params, 'path'),
            ...(typeof params.line === 'number' ? { line: params.line } : {}),
            ...(typeof params.limit === 'number' ? { limit: params.limit } : {}),
          }
          const result = await this.handlers.readTextFile(req)
          await this.sendFrame(acpJsonRpcSuccess(id, { content: result.content }))
          return
        }
        case 'fs/write_text_file': {
          if (!this.handlers.writeTextFile) {
            await this.sendFrame(
              acpJsonRpcFailure(
                id,
                ACP_JSON_RPC_ERRORS.methodNotFound,
                'fs/write_text_file handler is not configured on this ACP client',
              ),
            )
            return
          }
          const req: AcpWriteTextFileRequest = {
            sessionId: readStringField(params, 'sessionId', 'session_id'),
            path: readStringField(params, 'path'),
            content: typeof params.content === 'string' ? params.content : '',
          }
          await this.handlers.writeTextFile(req)
          await this.sendFrame(acpJsonRpcSuccess(id, {}))
          return
        }
        default: {
          await this.sendFrame(
            acpJsonRpcFailure(
              id,
              ACP_JSON_RPC_ERRORS.methodNotFound,
              `ACP client does not implement method "${method}"`,
            ),
          )
        }
      }
    } catch (error) {
      await this.sendFrame(
        acpJsonRpcFailure(id, ACP_JSON_RPC_ERRORS.internalError, errorMessage(error)),
      ).catch(() => undefined)
    }
  }

  private async resolvePermission(request: AcpClientPermissionRequest): Promise<AcpClientPermissionResponse> {
    if (!this.handlers.requestPermission) {
      return defaultDenyPermission(request)
    }
    try {
      const decision = await this.handlers.requestPermission(request)
      if (decision.outcome === 'selected' && decision.optionId) {
        return decision
      }
      return defaultDenyPermission(request)
    } catch {
      return defaultDenyPermission(request)
    }
  }

  private async resolveElicitation(request: AcpElicitationRequest): Promise<AcpElicitationResponse> {
    if (!this.handlers.createElicitation) {
      return { action: 'cancel' }
    }
    try {
      const response = await this.handlers.createElicitation(request)
      if (response.action === 'accept' || response.action === 'decline' || response.action === 'cancel') {
        return response
      }
      return { action: 'cancel' }
    } catch {
      return { action: 'cancel' }
    }
  }

  private handleAgentNotification(method: string, params: Readonly<Record<string, unknown>>): void {
    if (method !== 'session/update' && method !== 'session_update') {
      return
    }
    const sessionId = readStringField(params, 'sessionId', 'session_id')
    const rawUpdate = isAcpRecord(params.update)
      ? params.update
      : isAcpRecord(params.event)
        ? params.event
        : params
    const isPostSettle = sessionId !== '' && this.settledSessions.has(sessionId)
    const normalizedEvents = normalizeAcpSessionUpdate(sessionId, rawUpdate, isPostSettle)
    for (const event of normalizedEvents) {
      this.emit(event, sessionId)
    }
  }

  private emit(event: AcpClientEvent, sessionId?: string): void {
    if (sessionId) {
      const promptListener = this.promptListeners.get(sessionId)
      if (promptListener) {
        try {
          promptListener(event)
        } catch {
          // Keep stream healthy if a subscriber throws.
        }
      }
    }
    for (const listener of this.listeners) {
      try {
        listener(event)
      } catch {
        // Keep stream healthy if a subscriber throws.
      }
    }
  }
}

/**
 * Normalize an incoming ACP `session/update` payload into zero or more
 * {@link AcpClientEvent} values. Handles standard ACP spec shapes, Xerxes ACP
 * wire events, and Antigravity-style post-settle continuation frames.
 */
export function normalizeAcpSessionUpdate(
  sessionId: string,
  update: Readonly<Record<string, unknown>>,
  isPostSettle = false,
): AcpClientEvent[] {
  const events: AcpClientEvent[] = []
  const kind = readStringField(update, 'sessionUpdate', 'type', 'kind')

  switch (kind) {
    case 'agent_message_chunk':
    case 'text_delta':
    case 'text': {
      const text = extractAcpContentText(update.content) || readStringField(update, 'delta', 'text')
      if (text) {
        events.push({ type: 'text_delta', sessionId, text })
        if (isPostSettle) {
          events.push({ type: 'continuation', sessionId, text, rawUpdate: update })
        }
      }
      return events
    }

    case 'agent_thought_chunk':
    case 'thinking_delta':
    case 'thinking': {
      const text = extractAcpContentText(update.content) || readStringField(update, 'delta', 'text', 'think')
      if (text) {
        events.push({ type: 'thought_delta', sessionId, text })
      }
      return events
    }

    case 'tool_call':
    case 'tool_start': {
      const toolCallId = readStringField(update, 'toolCallId', 'tool_call_id', 'id') || crypto.randomUUID()
      const title = readStringField(update, 'title') || undefined
      const toolKind = readStringField(update, 'kind') || undefined
      const name = readStringField(update, 'name', 'toolName', 'tool_name') || title || toolKind || 'tool'
      const status = readStringField(update, 'status') || 'in_progress'
      const rawInput = update.rawInput ?? update.inputs ?? update.arguments
      const diff = extractAcpToolDiff(update.content)
      const locations = extractAcpToolLocations(update.locations)
      events.push({
        type: 'tool_call',
        sessionId,
        toolCallId,
        name,
        ...(title ? { title } : {}),
        ...(toolKind ? { kind: toolKind } : {}),
        ...(status ? { status } : {}),
        ...(rawInput !== undefined ? { rawInput } : {}),
        ...(diff ? { diff } : {}),
        ...(locations && locations.length > 0 ? { locations } : {}),
      })
      return events
    }

    case 'tool_call_update':
    case 'tool_end': {
      const toolCallId = readStringField(update, 'toolCallId', 'tool_call_id', 'id') || 'tool'
      const title = readStringField(update, 'title') || undefined
      const name = readStringField(update, 'name', 'toolName', 'tool_name') || title || undefined
      const status = readStringField(update, 'status') || (kind === 'tool_end' ? 'completed' : undefined)
      const rawInput = update.rawInput ?? update.inputs
      const rawOutput = update.rawOutput ?? update.result
      const diff = extractAcpToolDiff(update.content)
      const locations = extractAcpToolLocations(update.locations)
      const textContent = extractAcpToolContent(update.content)
      const content = textContent || stringifyOutput(rawOutput) || undefined
      const normalizedStatus = status?.toLowerCase()
      const isError =
        normalizedStatus === 'failed' ||
        normalizedStatus === 'error' ||
        normalizedStatus === 'rejected' ||
        normalizedStatus === 'denied' ||
        update.isError === true ||
        update.is_error === true ||
        update.permitted === false
      const permissionDenied =
        update.permitted === false ||
        normalizedStatus === 'rejected' ||
        normalizedStatus === 'denied' ||
        (isError && /\b(?:permission\s+denied|denied\s+by\s+user|user\s+rejected|rejected\s+by\s+user|not\s+permitted)\b/i.test(content ?? ''))
      events.push({
        type: 'tool_call_update',
        sessionId,
        toolCallId,
        ...(name ? { name } : {}),
        ...(title ? { title } : {}),
        ...(status ? { status } : {}),
        ...(rawInput !== undefined ? { rawInput } : {}),
        ...(rawOutput !== undefined ? { rawOutput } : {}),
        ...(content !== undefined ? { content } : {}),
        ...(diff ? { diff } : {}),
        ...(locations && locations.length > 0 ? { locations } : {}),
        ...(isError ? { isError: true } : {}),
        ...(permissionDenied ? { permissionDenied: true } : {}),
      })
      return events
    }

    case 'plan': {
      const rawEntries = Array.isArray(update.entries) ? update.entries : []
      const entries: AcpPlanEntry[] = []
      for (const item of rawEntries) {
        if (!isAcpRecord(item)) continue
        const content = readStringField(item, 'content', 'title', 'text')
        if (!content) continue
        const status = readStringField(item, 'status') || undefined
        const priority = readStringField(item, 'priority') || undefined
        entries.push({
          content,
          ...(status ? { status } : {}),
          ...(priority ? { priority } : {}),
        })
      }
      events.push({ type: 'plan', sessionId, entries })
      return events
    }

    case 'current_mode_update':
    case 'mode_update': {
      const modeId = readStringField(update, 'currentModeId', 'current_mode_id', 'modeId', 'mode_id', 'mode')
      const modeName = readStringField(update, 'currentModeName', 'current_mode_name', 'modeName', 'name') || undefined
      if (modeId) {
        events.push({
          type: 'mode_update',
          sessionId,
          modeId,
          ...(modeName ? { modeName } : {}),
        })
      }
      return events
    }

    case 'current_model_update':
    case 'model_update': {
      const modelId = readStringField(update, 'currentModelId', 'current_model_id', 'modelId', 'model_id', 'model')
      const modelName = readStringField(update, 'currentModelName', 'current_model_name', 'modelName', 'name') || undefined
      if (modelId) {
        events.push({
          type: 'model_update',
          sessionId,
          modelId,
          ...(modelName ? { modelName } : {}),
        })
      }
      return events
    }

    case 'config_option_update':
    case 'config_options_update':
    case 'session_config_update': {
      const configOptions = Array.isArray(update.configOptions)
        ? update.configOptions
        : Array.isArray(update.config_options)
          ? update.config_options
          : undefined
      const configId = readStringField(update, 'configId', 'config_id', 'id') || undefined
      const value = readStringField(update, 'value', 'currentValue', 'current_value') || undefined
      if (configOptions || (configId && value)) {
        events.push({
          type: 'config_update',
          sessionId,
          ...(configOptions ? { configOptions } : {}),
          ...(configId ? { configId } : {}),
          ...(value ? { value } : {}),
          rawUpdate: update,
        })
      }
      return events
    }

    case 'usage_update': {
      const contextUsed = readNumberField(update, 'used', 'contextUsed', 'context_used', 'contextTokens', 'context_tokens')
      const contextSize = readNumberField(update, 'size', 'contextSize', 'context_size', 'maxContext', 'max_context', 'contextLimit', 'context_limit')
      const nestedUsage = isAcpRecord(update.usage) ? update.usage : update
      const inputTokens = readNumberField(nestedUsage, 'inputTokens', 'input_tokens')
      const outputTokens = readNumberField(nestedUsage, 'outputTokens', 'output_tokens')
      const cachedReadTokens = readNumberField(nestedUsage, 'cachedReadTokens', 'cached_read_tokens', 'cacheReadTokens', 'cache_read_tokens')
      const cachedWriteTokens = readNumberField(nestedUsage, 'cachedWriteTokens', 'cached_write_tokens', 'cacheCreationTokens', 'cache_creation_tokens')
      const thoughtTokens = readNumberField(nestedUsage, 'thoughtTokens', 'thought_tokens', 'reasoningTokens', 'reasoning_tokens')
      const totalTokens = readNumberField(nestedUsage, 'totalTokens', 'total_tokens')
      if (
        contextUsed !== undefined ||
        contextSize !== undefined ||
        inputTokens !== undefined ||
        outputTokens !== undefined ||
        cachedReadTokens !== undefined ||
        cachedWriteTokens !== undefined ||
        thoughtTokens !== undefined ||
        totalTokens !== undefined
      ) {
        events.push({
          type: 'usage_update',
          sessionId,
          ...(contextUsed !== undefined ? { contextUsed } : {}),
          ...(contextSize !== undefined ? { contextSize } : {}),
          ...(inputTokens !== undefined ? { inputTokens } : {}),
          ...(outputTokens !== undefined ? { outputTokens } : {}),
          ...(cachedReadTokens !== undefined ? { cachedReadTokens } : {}),
          ...(cachedWriteTokens !== undefined ? { cachedWriteTokens } : {}),
          ...(thoughtTokens !== undefined ? { thoughtTokens } : {}),
          ...(totalTokens !== undefined ? { totalTokens } : {}),
        })
      }
      return events
    }

    case 'session_info_update':
    case 'session_info': {
      const title = readStringField(update, 'title') || undefined
      const updatedAt = readStringField(update, 'updatedAt', 'updated_at') || undefined
      if (title || updatedAt) {
        events.push({
          type: 'session_info',
          sessionId,
          ...(title ? { title } : {}),
          ...(updatedAt ? { updatedAt } : {}),
        })
      }
      return events
    }

    case 'available_commands_update': {
      const rawCommands = Array.isArray(update.availableCommands)
        ? update.availableCommands
        : Array.isArray(update.available_commands)
          ? update.available_commands
          : []
      const commands: AcpAvailableCommand[] = []
      const seen = new Set<string>()
      for (const item of rawCommands) {
        if (!isAcpRecord(item)) continue
        const rawName = readStringField(item, 'name').trim().replace(/^\/+/, '')
        const dedupeKey = rawName.toLowerCase()
        if (!rawName || seen.has(dedupeKey)) continue
        seen.add(dedupeKey)
        const description = readStringField(item, 'description') || undefined
        const inputRecord = isAcpRecord(item.input) ? item.input : undefined
        const inputHint =
          (inputRecord ? readStringField(inputRecord, 'hint', 'placeholder', 'description') : '') ||
          readStringField(item, 'inputHint', 'input_hint') ||
          undefined
        const meta = isAcpRecord(item._meta)
          ? item._meta
          : isAcpRecord(item.meta)
            ? item.meta
            : undefined
        const rawKind = (
          readStringField(item, 'kind', 'type', 'category') ||
          (meta ? readStringField(meta, 'kind', 'type', 'category') : '')
        ).toLowerCase()
        const metaPath = meta ? readStringField(meta, 'path', 'skillPath', 'skill_path') : ''
        const kind: 'command' | 'skill' | undefined =
          rawKind === 'skill' ||
          rawKind === 'skills' ||
          /(?:^|[/\\])skill\.md$/i.test(metaPath)
            ? 'skill'
            : rawKind === 'command' || rawKind === 'commands'
              ? 'command'
              : undefined
        const source =
          readStringField(item, 'source') ||
          (meta ? readStringField(meta, 'source', 'origin') : '') ||
          undefined
        commands.push({
          name: rawName,
          ...(description ? { description } : {}),
          ...(inputHint ? { input_hint: inputHint } : {}),
          ...(kind ? { kind } : {}),
          ...(source ? { source } : {}),
        })
      }
      events.push({ type: 'available_commands', sessionId, commands })
      return events
    }

    case 'user_message_chunk': {
      // Replay echo during session/load; never treat user echoes as assistant deltas.
      return events
    }

    case 'continuation':
    case 'post_settle':
    case 'agent_continuation':
    case 'continuation_chunk': {
      const nestedText = extractAcpContentText(update.content) || readStringField(update, 'delta', 'text')
      if (nestedText) {
        events.push({ type: 'text_delta', sessionId, text: nestedText })
      }
      events.push({
        type: 'continuation',
        sessionId,
        ...(nestedText ? { text: nestedText } : {}),
        rawUpdate: update,
      })
      return events
    }

    default: {
      // Preserve Antigravity-style continuation or vendor-specific post-settle updates cleanly.
      const fallbackText = extractAcpContentText(update.content) || readStringField(update, 'delta', 'text')
      if (fallbackText) {
        events.push({ type: 'text_delta', sessionId, text: fallbackText })
      }
      if (isPostSettle || fallbackText || kind) {
        events.push({
          type: 'continuation',
          sessionId,
          ...(fallbackText ? { text: fallbackText } : {}),
          rawUpdate: update,
        })
      }
      return events
    }
  }
}

/** True when an ACP JSON-RPC error indicates the external agent requires user login. */
export function isAcpAuthRequiredError(error: unknown): boolean {
  if (error instanceof AcpRpcError) {
    if (
      (error.code === -32000 || error.code === 401) &&
      /auth(?:entication)?\s+required|not\s+(?:signed|logged)\s+in|unauthenticated|unauthorized|login\s+required/i.test(
        error.message,
      )
    ) {
      return true
    }
  }
  const msg = errorMessage(error)
  return /\bauthentication\s+required\b|\bunauthenticated\b|\blogin\s+required\b/i.test(msg)
}

function parsePermissionRequest(params: Readonly<Record<string, unknown>>): AcpClientPermissionRequest {
  const sessionId = readStringField(params, 'sessionId', 'session_id')
  const toolCall = isAcpRecord(params.toolCall) ? params.toolCall : {}
  const toolCallId =
    readStringField(toolCall, 'toolCallId', 'tool_call_id', 'id') ||
    readStringField(params, 'toolCallId', 'tool_call_id') ||
    crypto.randomUUID()
  const title = readStringField(toolCall, 'title') || readStringField(params, 'title', 'description')
  const toolKind = readStringField(toolCall, 'kind') || undefined
  const toolName =
    readStringField(toolCall, 'name', 'toolName', 'tool_name') ||
    readStringField(params, 'toolName', 'tool_name') ||
    title ||
    toolKind ||
    'tool'
  const rawInput = toolCall.rawInput ?? params.rawInput ?? params.inputs
  const diff = extractAcpToolDiff(toolCall.content ?? params.content)
  const locations = extractAcpToolLocations(toolCall.locations ?? params.locations)
  const rawOptions = Array.isArray(params.options) ? params.options : []
  const options: AcpPermissionOption[] = []
  for (const item of rawOptions) {
    if (!isAcpRecord(item)) continue
    const optionId = readStringField(item, 'optionId', 'id')
    if (!optionId) continue
    const name = readStringField(item, 'name', 'label') || optionId
    const kind = readStringField(item, 'kind') || undefined
    options.push({ optionId, name, ...(kind ? { kind } : {}) })
  }
  return {
    sessionId,
    toolCallId,
    title: title || toolName,
    toolName,
    ...(toolKind ? { kind: toolKind } : {}),
    ...(rawInput !== undefined ? { rawInput } : {}),
    ...(diff ? { diff } : {}),
    ...(locations && locations.length > 0 ? { locations } : {}),
    options,
  }
}

function defaultDenyPermission(request: AcpClientPermissionRequest): AcpClientPermissionResponse {
  const rejectOption =
    request.options.find(opt => opt.kind === 'reject_once') ??
    request.options.find(opt => opt.kind === 'reject_always') ??
    request.options.find(opt => /\b(?:reject|deny|cancel)\b/i.test(opt.name))
  if (rejectOption) {
    return { outcome: 'selected', optionId: rejectOption.optionId }
  }
  return { outcome: 'cancelled' }
}

function formatPermissionRpcResult(response: AcpClientPermissionResponse): Record<string, unknown> {
  if (response.outcome === 'selected' && response.optionId) {
    return {
      outcome: {
        outcome: 'selected',
        optionId: response.optionId,
      },
    }
  }
  return {
    outcome: {
      outcome: 'cancelled',
    },
  }
}

export function parseElicitationRequest(params: Readonly<Record<string, unknown>>): AcpElicitationRequest {
  const sessionId = readStringField(params, 'sessionId', 'session_id')
  const toolCallId = readStringField(params, 'toolCallId', 'tool_call_id') || undefined
  const url = readStringField(params, 'url') || undefined
  const mode = readStringField(params, 'mode') || (url ? 'url' : 'form')
  const message = readStringField(params, 'message', 'title') || 'Input required'
  const requestedSchema = isAcpRecord(params.requestedSchema)
    ? params.requestedSchema
    : isAcpRecord(params.requested_schema)
      ? params.requested_schema
      : undefined
  const fields = requestedSchema ? parseElicitationFields(requestedSchema) : []
  return {
    sessionId,
    ...(toolCallId ? { toolCallId } : {}),
    mode,
    message,
    ...(url ? { url } : {}),
    ...(requestedSchema ? { requestedSchema } : {}),
    fields,
    rawParams: params,
  }
}

function parseElicitationFields(schema: Readonly<Record<string, unknown>>): AcpElicitationField[] {
  const properties = isAcpRecord(schema.properties) ? schema.properties : undefined
  if (!properties) {
    return []
  }
  const requiredKeys = new Set<string>(
    Array.isArray(schema.required)
      ? schema.required.filter((item): item is string => typeof item === 'string')
      : [],
  )
  const fields: AcpElicitationField[] = []
  for (const [key, rawField] of Object.entries(properties)) {
    if (!isAcpRecord(rawField)) continue
    const rawType = readStringField(rawField, 'type').toLowerCase()
    const fieldType: AcpElicitationFieldType =
      rawType === 'array' || rawType === 'boolean' || rawType === 'number' || rawType === 'integer'
        ? rawType
        : 'string'
    const title = readStringField(rawField, 'title') || undefined
    const description = readStringField(rawField, 'description') || undefined
    const oneOf = parseElicitationOptions(rawField.oneOf ?? rawField.anyOf)
    const enumValues = Array.isArray(rawField.enum)
      ? rawField.enum.filter((item): item is string => typeof item === 'string')
      : undefined
    const enumNames = Array.isArray(rawField.enumNames)
      ? rawField.enumNames.filter((item): item is string => typeof item === 'string')
      : undefined
    const rawItems = isAcpRecord(rawField.items) ? rawField.items : undefined
    const items = rawItems
      ? (() => {
          const itemAnyOf = parseElicitationOptions(rawItems.anyOf)
          const itemOneOf = parseElicitationOptions(rawItems.oneOf)
          const itemEnum = Array.isArray(rawItems.enum)
            ? rawItems.enum.filter((item): item is string => typeof item === 'string')
            : undefined
          const itemType = readStringField(rawItems, 'type') || undefined
          return {
            ...(itemType ? { type: itemType } : {}),
            ...(itemAnyOf && itemAnyOf.length > 0 ? { anyOf: itemAnyOf } : {}),
            ...(itemOneOf && itemOneOf.length > 0 ? { oneOf: itemOneOf } : {}),
            ...(itemEnum && itemEnum.length > 0 ? { enum: itemEnum } : {}),
          }
        })()
      : undefined
    fields.push({
      key,
      type: fieldType,
      required: requiredKeys.has(key),
      ...(title ? { title } : {}),
      ...(description ? { description } : {}),
      ...(rawField.default !== undefined ? { default: rawField.default } : {}),
      ...(oneOf && oneOf.length > 0 ? { oneOf } : {}),
      ...(enumValues && enumValues.length > 0 ? { enum: enumValues } : {}),
      ...(enumNames && enumNames.length > 0 ? { enumNames } : {}),
      ...(items ? { items } : {}),
    })
  }
  return fields
}

function parseElicitationOptions(raw: unknown): readonly AcpElicitationOption[] | undefined {
  if (!Array.isArray(raw)) {
    return undefined
  }
  const options: AcpElicitationOption[] = []
  for (const item of raw) {
    if (!isAcpRecord(item)) continue
    const constVal =
      typeof item.const === 'string'
        ? item.const
        : typeof item.const === 'number' || typeof item.const === 'boolean'
          ? String(item.const)
          : readStringField(item, 'value', 'id')
    if (!constVal) continue
    const title = readStringField(item, 'title', 'label', 'name') || constVal
    options.push({ const: constVal, title })
  }
  return options.length > 0 ? options : undefined
}

function formatElicitationRpcResult(response: AcpElicitationResponse): Record<string, unknown> {
  if (response.action === 'accept') {
    return {
      action: 'accept',
      content: response.content ?? {},
    }
  }
  if (response.action === 'decline') {
    return { action: 'decline' }
  }
  return { action: 'cancel' }
}

function extractAcpContentText(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }
  if (Array.isArray(content)) {
    return content.map(item => extractAcpContentText(item)).filter(Boolean).join('')
  }
  if (!isAcpRecord(content)) {
    return ''
  }
  if (typeof content.text === 'string') {
    return content.text
  }
  if (content.content !== undefined) {
    return extractAcpContentText(content.content)
  }
  return ''
}

function extractAcpToolDiff(content: unknown): AcpToolDiff | undefined {
  const entries = Array.isArray(content) ? content : content !== undefined ? [content] : []
  for (const entry of entries) {
    if (!isAcpRecord(entry)) continue
    if (entry.type !== 'diff') continue
    const path = readStringField(entry, 'path')
    const oldText = typeof entry.oldText === 'string'
      ? entry.oldText
      : typeof entry.old_text === 'string'
        ? entry.old_text
        : undefined
    const newText = typeof entry.newText === 'string'
      ? entry.newText
      : typeof entry.new_text === 'string'
        ? entry.new_text
        : undefined
    if (path || oldText !== undefined || newText !== undefined) {
      return {
        path,
        ...(oldText !== undefined ? { oldText } : {}),
        ...(newText !== undefined ? { newText } : {}),
      }
    }
  }
  return undefined
}

function extractAcpToolLocations(raw: unknown): readonly AcpToolLocation[] | undefined {
  if (!Array.isArray(raw)) {
    return undefined
  }
  const locations: AcpToolLocation[] = []
  for (const item of raw) {
    if (!isAcpRecord(item)) continue
    const path = readStringField(item, 'path')
    if (!path) continue
    const line = typeof item.line === 'number' && Number.isFinite(item.line) ? Math.trunc(item.line) : undefined
    locations.push({
      path,
      ...(line !== undefined ? { line } : {}),
    })
  }
  return locations.length > 0 ? locations : undefined
}

function extractAcpToolContent(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }
  if (Array.isArray(content)) {
    const parts: string[] = []
    for (const entry of content) {
      if (typeof entry === 'string') {
        parts.push(entry)
        continue
      }
      if (!isAcpRecord(entry)) continue
      if (entry.type === 'diff') {
        // Structured diff blocks are preserved intact on `event.diff` rather than flattened.
        continue
      }
      const text = extractAcpContentText(entry)
      if (text) parts.push(text)
    }
    return parts.join('\n')
  }
  if (isAcpRecord(content) && content.type === 'diff') {
    return ''
  }
  return extractAcpContentText(content)
}

function parseAuthMethods(raw: unknown): readonly AcpAuthMethod[] {
  if (!Array.isArray(raw)) {
    return []
  }
  const methods: AcpAuthMethod[] = []
  const seen = new Set<string>()
  for (const item of raw) {
    if (!isAcpRecord(item)) continue
    const id = readStringField(item, 'id')
    if (!id || seen.has(id)) continue
    seen.add(id)
    const name = readStringField(item, 'name', 'label') || id
    const description = readStringField(item, 'description') || undefined
    methods.push({
      id,
      name,
      ...(description ? { description } : {}),
    })
  }
  return methods
}

function parsePromptUsage(raw: unknown): AcpPromptUsage | undefined {
  if (!isAcpRecord(raw)) {
    return undefined
  }
  const inputTokens = readNumberField(raw, 'inputTokens', 'input_tokens')
  const outputTokens = readNumberField(raw, 'outputTokens', 'output_tokens')
  const cachedReadTokens = readNumberField(raw, 'cachedReadTokens', 'cached_read_tokens', 'cacheReadTokens', 'cache_read_tokens')
  const cachedWriteTokens = readNumberField(raw, 'cachedWriteTokens', 'cached_write_tokens', 'cacheCreationTokens', 'cache_creation_tokens')
  const thoughtTokens = readNumberField(raw, 'thoughtTokens', 'thought_tokens', 'reasoningTokens', 'reasoning_tokens')
  const totalTokens = readNumberField(raw, 'totalTokens', 'total_tokens')
  if (
    inputTokens === undefined &&
    outputTokens === undefined &&
    cachedReadTokens === undefined &&
    cachedWriteTokens === undefined &&
    thoughtTokens === undefined &&
    totalTokens === undefined
  ) {
    return undefined
  }
  return {
    ...(inputTokens !== undefined ? { inputTokens } : {}),
    ...(outputTokens !== undefined ? { outputTokens } : {}),
    ...(cachedReadTokens !== undefined ? { cachedReadTokens } : {}),
    ...(cachedWriteTokens !== undefined ? { cachedWriteTokens } : {}),
    ...(thoughtTokens !== undefined ? { thoughtTokens } : {}),
    ...(totalTokens !== undefined ? { totalTokens } : {}),
  }
}

function stringifyOutput(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  if (value === undefined || value === null) {
    return ''
  }
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function isMethodNotFoundError(error: unknown): boolean {
  if (error instanceof AcpRpcError && error.code === ACP_JSON_RPC_ERRORS.methodNotFound) {
    return true
  }
  const msg = errorMessage(error)
  return /\(-32601\)|method not found|unknown method|not implemented|unsupported method/i.test(msg)
}

function readStringField(record: Readonly<Record<string, unknown>>, ...keys: readonly string[]): string {
  for (const key of keys) {
    const value = record[key]
    if (typeof value === 'string' && value.length > 0) {
      return value
    }
  }
  return ''
}

function readNumberField(record: Readonly<Record<string, unknown>>, ...keys: readonly string[]): number | undefined {
  for (const key of keys) {
    const value = record[key]
    if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
      return value
    }
  }
  return undefined
}

function spawnBunAcpProcess(
  command: string,
  args: readonly string[],
  options: {
    readonly cwd?: string
    readonly env?: Readonly<Record<string, string | undefined>>
  },
): AcpSpawnedProcess {
  const mergedEnv: Record<string, string> = {}
  for (const [key, value] of Object.entries({ ...process.env, ...(options.env ?? {}) })) {
    if (typeof value === 'string') {
      mergedEnv[key] = value
    }
  }
  const child = Bun.spawn([command, ...args], {
    ...(options.cwd ? { cwd: options.cwd } : {}),
    env: mergedEnv,
    stdin: 'pipe',
    stdout: 'pipe',
    stderr: 'pipe',
  })

  return {
    stdout: child.stdout,
    stderr: child.stderr,
    exited: child.exited.then(code => (typeof code === 'number' ? code : null)),
    writeLine: (line: string) => {
      if (child.stdin && typeof child.stdin !== 'number') {
        child.stdin.write(line)
        void child.stdin.flush?.()
      }
    },
    kill: () => {
      try {
        if (child.stdin && typeof child.stdin !== 'number') {
          void child.stdin.end?.()
        }
      } catch {
        // Ignore already-closed stdin.
      }
      try {
        child.kill()
      } catch {
        // Ignore already-exited subprocess.
      }
    },
  }
}

async function raceWithUnrefTimeout(work: Promise<unknown>, timeoutMs: number): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    await Promise.race([
      work,
      new Promise<void>(resolve => {
        timer = setTimeout(resolve, timeoutMs)
        timer.unref?.()
      }),
    ])
  } finally {
    if (timer !== undefined) clearTimeout(timer)
  }
}
