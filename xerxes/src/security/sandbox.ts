// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { XerxesError } from '../core/errors.js'
import type { ToolExecutionContext, ToolExecutor } from '../executors/toolRegistry.js'
import type { ToolCall } from '../types/toolCalls.js'
import { SubprocessSandboxBackend, SubprocessSandboxConfigurationError } from './subprocessSandbox.js'

export const SandboxMode = {
  OFF: 'off',
  WARN: 'warn',
  STRICT: 'strict',
} as const

export type SandboxMode = (typeof SandboxMode)[keyof typeof SandboxMode]

export const ExecutionContext = {
  HOST: 'host',
  SANDBOX: 'sandbox',
} as const

export type ExecutionContext = (typeof ExecutionContext)[keyof typeof ExecutionContext]

export interface SandboxBackendConfig {
  readonly envVars?: Readonly<Record<string, string>>
  /** Subprocess backend accepts `allowedCommands: string[]` and optional `maxOutputChars: number`. */
  readonly extraArgs?: Readonly<Record<string, unknown>>
  readonly image?: string
  readonly mountPaths?: Readonly<Record<string, string>>
  readonly mountReadonly?: boolean
}

export interface SandboxConfig {
  readonly backendConfig?: SandboxBackendConfig
  readonly backendType?: string
  readonly elevatedTools?: Iterable<string>
  readonly mode?: SandboxMode
  readonly sandboxMemoryLimitMb?: number
  readonly sandboxNetworkAccess?: boolean
  readonly sandboxTimeout?: number
  readonly sandboxedTools?: Iterable<string>
  readonly workingDirectory?: string
}

export interface ExecutionDecision {
  readonly context: ExecutionContext
  readonly reason: string
  readonly toolName: string
}

export interface SandboxExecutionRequest {
  readonly arguments: ToolCall['function']['arguments']
  readonly context: ToolExecutionContext
  readonly signal?: AbortSignal
  readonly toolName: string
}

/**
 * A remote/container backend owns real isolation and must expose only a
 * serializable tool request boundary. JavaScript closures are never shipped
 * into a sandbox.
 */
export interface SandboxBackend {
  execute(request: SandboxExecutionRequest): Promise<string>
  getCapabilities?(): Readonly<Record<string, unknown>>
  isAvailable?(): Promise<boolean> | boolean
}

export interface SandboxRouterOptions {
  readonly backend?: SandboxBackend
  readonly config?: SandboxConfig
  readonly onWarning?: (decision: ExecutionDecision) => void
}

export class SandboxExecutionUnavailableError extends XerxesError {
  readonly toolName: string

  constructor(toolName: string) {
    super("Tool '" + toolName + "' requires sandbox execution, but no sandbox backend is configured")
    this.toolName = toolName
  }
}

/**
 * Resolve host versus sandbox execution under the Xerxes sandbox policy.
 *
 * This port intentionally keeps routing independent from a specific Docker,
 * subprocess, or cloud runtime. A concrete backend receives typed data, not
 * an executable closure, so a host never accidentally runs sandbox-marked
 * code while serializing it.
 */
export class SandboxRouter {
  readonly backend: SandboxBackend | undefined
  readonly config: Readonly<RequiredSandboxConfig>
  private readonly onWarning: (decision: ExecutionDecision) => void

  constructor(options: SandboxRouterOptions = {}) {
    this.config = normalizeConfig(options.config)
    this.backend = options.backend ?? backendFromConfig(this.config)
    this.onWarning = options.onWarning ?? (() => undefined)
  }

  decide(toolName: string): ExecutionDecision {
    if (this.config.elevatedTools.has(toolName)) {
      return { context: ExecutionContext.HOST, toolName, reason: 'Tool is marked as elevated' }
    }
    if (this.config.mode === SandboxMode.OFF) {
      return { context: ExecutionContext.HOST, toolName, reason: 'Sandbox mode is off' }
    }
    if (this.config.sandboxedTools.has(toolName)) {
      if (this.config.mode === SandboxMode.WARN) {
        const decision = {
          context: ExecutionContext.HOST,
          toolName,
          reason: 'Warn mode advisory: tool would run in sandbox, executing on host',
        } as const
        this.onWarning(decision)
        return decision
      }
      return { context: ExecutionContext.SANDBOX, toolName, reason: 'Strict sandbox enforcement' }
    }
    return { context: ExecutionContext.HOST, toolName, reason: 'Tool not designated for sandbox' }
  }

  /** Execute a serializable request through a configured strict backend. */
  async executeInSandbox(request: SandboxExecutionRequest): Promise<string> {
    if (!this.backend) {
      throw new SandboxExecutionUnavailableError(request.toolName)
    }
    if (this.backend.isAvailable && !(await this.backend.isAvailable())) {
      throw new SandboxExecutionUnavailableError(request.toolName)
    }
    return this.backend.execute(request)
  }

  /** Route a complete tool call, using the host executor only when policy permits it. */
  async execute(
    call: ToolCall,
    context: ToolExecutionContext,
    executeHost: () => Promise<string>,
    signal?: AbortSignal,
  ): Promise<string> {
    const decision = this.decide(call.function.name)
    if (decision.context === ExecutionContext.HOST) {
      return executeHost()
    }
    return this.executeInSandbox({
      toolName: call.function.name,
      arguments: call.function.arguments,
      context,
      ...(signal ? { signal } : {}),
    })
  }
}

/** Wrap an existing executor without changing its agent-aware tool lookup semantics. */
export class SandboxedToolExecutor implements ToolExecutor {
  constructor(
    private readonly host: ToolExecutor,
    readonly router: SandboxRouter,
  ) {}

  execute(call: ToolCall, context: ToolExecutionContext, signal?: AbortSignal): Promise<string> {
    return this.router.execute(call, context, () => this.host.execute(call, context, signal), signal)
  }
}

interface RequiredSandboxConfig {
  readonly backendConfig: Readonly<{
    readonly envVars: Readonly<Record<string, string>>
    readonly extraArgs: Readonly<Record<string, unknown>>
    readonly image: string
    readonly mountPaths: Readonly<Record<string, string>>
    readonly mountReadonly: boolean
  }>
  readonly backendType: string | undefined
  readonly elevatedTools: ReadonlySet<string>
  readonly mode: SandboxMode
  readonly sandboxMemoryLimitMb: number
  readonly sandboxNetworkAccess: boolean
  readonly sandboxTimeout: number
  readonly sandboxedTools: ReadonlySet<string>
  readonly workingDirectory: string | undefined
}

function normalizeConfig(config: SandboxConfig | undefined): RequiredSandboxConfig {
  const source = config ?? {}
  const timeout = source.sandboxTimeout ?? 30
  const memory = source.sandboxMemoryLimitMb ?? 512
  if (!Number.isFinite(timeout) || timeout <= 0) {
    throw new RangeError('sandboxTimeout must be a positive number')
  }
  if (!Number.isInteger(memory) || memory < 16) {
    throw new RangeError('sandboxMemoryLimitMb must be an integer of at least 16')
  }
  const mode = source.mode ?? SandboxMode.OFF
  if (!Object.values(SandboxMode).includes(mode)) {
    throw new RangeError('Unknown sandbox mode: ' + mode)
  }
  const backend = source.backendConfig ?? {}
  return Object.freeze({
    mode,
    sandboxedTools: new Set(source.sandboxedTools ?? []),
    elevatedTools: new Set(source.elevatedTools ?? []),
    sandboxTimeout: timeout,
    sandboxMemoryLimitMb: memory,
    sandboxNetworkAccess: source.sandboxNetworkAccess ?? false,
    workingDirectory: source.workingDirectory,
    backendType: source.backendType,
    backendConfig: Object.freeze({
      image: backend.image ?? 'node:22-slim',
      mountPaths: Object.freeze({ ...(backend.mountPaths ?? {}) }),
      mountReadonly: backend.mountReadonly ?? true,
      envVars: Object.freeze({ ...(backend.envVars ?? {}) }),
      extraArgs: Object.freeze({ ...(backend.extraArgs ?? {}) }),
    }),
  })
}

function backendFromConfig(config: RequiredSandboxConfig): SandboxBackend | undefined {
  if (config.backendType === undefined) {
    return undefined
  }
  if (config.backendType !== 'subprocess') {
    throw new SubprocessSandboxConfigurationError(
      `backendType ${JSON.stringify(config.backendType)} is not available in the Bun runtime; pass an explicit backend`,
    )
  }
  const allowedCommands = requiredStringArray(config.backendConfig.extraArgs, 'allowedCommands')
  const maxOutputChars = optionalPositiveInteger(config.backendConfig.extraArgs, 'maxOutputChars')
  return new SubprocessSandboxBackend({
    allowedCommands,
    allowedTools: config.sandboxedTools,
    environment: config.backendConfig.envVars,
    maxTimeoutMs: Math.ceil(config.sandboxTimeout * 1_000),
    memoryLimitMb: config.sandboxMemoryLimitMb,
    networkAccessRequested: config.sandboxNetworkAccess,
    ...(config.workingDirectory === undefined ? {} : { workingDirectory: config.workingDirectory }),
    ...(maxOutputChars === undefined ? {} : { maxOutputChars }),
  })
}

function requiredStringArray(values: Readonly<Record<string, unknown>>, name: string): string[] {
  const value = values[name]
  if (!Array.isArray(value) || value.length === 0) {
    throw new SubprocessSandboxConfigurationError(`backendConfig.extraArgs.${name} must be a non-empty string array`)
  }
  const strings: string[] = []
  for (const item of value) {
    if (typeof item !== 'string') {
      throw new SubprocessSandboxConfigurationError(`backendConfig.extraArgs.${name} must be a non-empty string array`)
    }
    strings.push(item)
  }
  return strings
}

function optionalPositiveInteger(values: Readonly<Record<string, unknown>>, name: string): number | undefined {
  const value = values[name]
  if (value === undefined) {
    return undefined
  }
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) {
    throw new SubprocessSandboxConfigurationError(`backendConfig.extraArgs.${name} must be a positive integer`)
  }
  return value
}
