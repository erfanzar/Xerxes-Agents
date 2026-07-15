// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  SandboxBackendAdapterConfigurationError,
  SandboxBackendAdapterRequestError,
  hostAdapterIsAvailable,
  normalizeSandboxCommandPolicy,
  positiveSafeInteger,
  prepareSandboxCommand,
  requiredHostMethod,
  serializeSandboxCommandResult,
} from './contracts.js'
import type {
  SandboxBackendAdapter,
  SandboxCommand,
  SandboxCommandPolicy,
  SandboxCommandPolicyOptions,
  SandboxCommandResult,
  SandboxHostProbe,
} from './contracts.js'

const UNSAFE_EXECUTABLE = /[\s;&|`$<>\0]/

/** Request passed to a caller-owned local-process sandbox implementation. */
export interface SubprocessSandboxHostRequest {
  readonly command: SandboxCommand
  /** Request-only resource limits; the host must report its own enforcement capabilities. */
  readonly memoryLimitMb: number
  readonly networkAccessRequested: boolean
  readonly signal?: AbortSignal
  readonly toolName: string
}

/** External subprocess boundary. This adapter deliberately never calls Bun.spawn. */
export interface SubprocessSandboxHost extends SandboxHostProbe {
  executeSubprocess(request: SubprocessSandboxHostRequest): Promise<SandboxCommandResult>
}

/** Host-process resource policy. `allowedCommands` is mandatory and fail-closed. */
export interface SubprocessSandboxConfig extends SandboxCommandPolicyOptions {
  readonly allowedCommands: Iterable<string>
  readonly memoryLimitMb?: number
  readonly networkAccessRequested?: boolean
}

export interface SubprocessSandboxAdapterOptions {
  readonly config: SubprocessSandboxConfig
  readonly host?: SubprocessSandboxHost
}

interface ResolvedSubprocessSandboxConfig {
  readonly allowedCommands: ReadonlySet<string>
  readonly commandPolicy: SandboxCommandPolicy
  readonly memoryLimitMb: number
  readonly networkAccessRequested: boolean
}

/**
 * Explicit-host subprocess adapter.
 *
 * The existing `security/subprocessSandbox.ts` remains the Bun-owned local
 * implementation. This port represents the Python backend family at an
 * injectable boundary so embedded callers can choose their actual process
 * controller without the runtime silently spawning anything.
 */
export class SubprocessSandboxAdapter implements SandboxBackendAdapter {
  readonly name = 'subprocess' as const
  readonly #config: ResolvedSubprocessSandboxConfig
  readonly #host: SubprocessSandboxHost | undefined

  constructor(options: SubprocessSandboxAdapterOptions) {
    this.#config = normalizeConfig(options.config)
    this.#host = options.host
  }

  async execute(request: Parameters<SandboxBackendAdapter['execute']>[0]): Promise<string> {
    const host = requiredHostMethod(this.name, this.#host, 'executing a subprocess sandbox command', 'executeSubprocess')
    const prepared = prepareSandboxCommand(this.name, request, this.#config.commandPolicy)
    const executable = prepared.command.argv[0]
    if (executable === undefined || !this.#config.allowedCommands.has(executable)) {
      throw new SandboxBackendAdapterRequestError(
        this.name,
        prepared.toolName,
        `command ${JSON.stringify(executable)} is not in the subprocess allow-list`,
        executable,
      )
    }
    const result = await host.executeSubprocess({
      command: prepared.command,
      memoryLimitMb: this.#config.memoryLimitMb,
      networkAccessRequested: this.#config.networkAccessRequested,
      toolName: prepared.toolName,
      ...(prepared.signal === undefined ? {} : { signal: prepared.signal }),
    })
    return serializeSandboxCommandResult(this.name, prepared.command, result)
  }

  isAvailable(): Promise<boolean> {
    return hostAdapterIsAvailable(this.#host)
  }

  getCapabilities(): Readonly<Record<string, unknown>> {
    return Object.freeze({
      backend: this.name,
      hostConfigured: this.#host !== undefined,
      allowedCommands: [...this.#config.allowedCommands].sort(),
      memoryLimitMb: this.#config.memoryLimitMb,
      networkAccessRequested: this.#config.networkAccessRequested,
      commandTransport: 'direct_argv_host_port',
      isolationLevel: 'host_defined',
      filesystemIsolation: 'host_defined',
      networkIsolation: 'host_defined',
    })
  }
}

function normalizeConfig(config: SubprocessSandboxConfig): ResolvedSubprocessSandboxConfig {
  if (config === null || typeof config !== 'object') {
    throw new SandboxBackendAdapterConfigurationError('subprocess', 'config with allowedCommands is required')
  }
  return Object.freeze({
    allowedCommands: normalizeAllowedCommands(config.allowedCommands),
    commandPolicy: normalizeSandboxCommandPolicy('subprocess', config),
    memoryLimitMb: positiveSafeInteger('subprocess', config.memoryLimitMb ?? 512, 'memoryLimitMb'),
    networkAccessRequested: config.networkAccessRequested ?? false,
  })
}

function normalizeAllowedCommands(values: Iterable<string>): ReadonlySet<string> {
  const commands = new Set<string>()
  for (const value of values) {
    if (typeof value !== 'string' || !value || value.trim() !== value || UNSAFE_EXECUTABLE.test(value)) {
      throw new SandboxBackendAdapterConfigurationError(
        'subprocess',
        'allowedCommands must contain direct executable names or paths without shell syntax',
      )
    }
    commands.add(value)
  }
  if (!commands.size) {
    throw new SandboxBackendAdapterConfigurationError('subprocess', 'allowedCommands must not be empty')
  }
  return commands
}
