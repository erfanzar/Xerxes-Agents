// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  SandboxBackendProtocolError,
  hostAdapterIsAvailable,
  normalizeSandboxCommandPolicy,
  positiveSafeInteger,
  prepareSandboxCommand,
  requiredHostMethod,
  requiredText,
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

const DEFAULT_MODAL_IMAGE = 'oven/bun:1.3'
const DEFAULT_MODAL_CPU = 1
const DEFAULT_MODAL_MEMORY_MB = 1_024

/** Fully resolved request a caller-owned Modal bridge receives. */
export interface ModalSandboxCreateRequest {
  readonly command: SandboxCommand
  readonly cpu: number
  readonly image: string
  readonly memoryLimitMb: number
  readonly signal?: AbortSignal
  readonly toolName: string
}

/** Opaque Modal sandbox lifecycle supplied by the host. */
export interface ModalSandbox {
  readonly id: string
  close(): Promise<void>
  wait(signal?: AbortSignal): Promise<SandboxCommandResult>
}

/** External Modal boundary. No Modal package or credential lookup occurs in this adapter. */
export interface ModalSandboxHost extends SandboxHostProbe {
  createSandbox(request: ModalSandboxCreateRequest): Promise<ModalSandbox>
}

/** Modal resource configuration and generic direct-argv execution policy. */
export interface ModalSandboxConfig extends SandboxCommandPolicyOptions {
  readonly cpu?: number
  readonly image?: string
  readonly memoryLimitMb?: number
}

export interface ModalSandboxAdapterOptions {
  readonly config?: ModalSandboxConfig
  readonly host?: ModalSandboxHost
}

interface ResolvedModalSandboxConfig {
  readonly commandPolicy: SandboxCommandPolicy
  readonly cpu: number
  readonly image: string
  readonly memoryLimitMb: number
}

/**
 * Creates one Modal sandbox through an injected host, waits for a confirmed
 * result, and closes the remote resource before returning it to the router.
 */
export class ModalSandboxAdapter implements SandboxBackendAdapter {
  readonly name = 'modal' as const
  readonly #config: ResolvedModalSandboxConfig
  readonly #host: ModalSandboxHost | undefined

  constructor(options: ModalSandboxAdapterOptions = {}) {
    this.#config = normalizeConfig(options.config)
    this.#host = options.host
  }

  async execute(request: Parameters<SandboxBackendAdapter['execute']>[0]): Promise<string> {
    const host = requiredHostMethod(this.name, this.#host, 'creating a Modal sandbox', 'createSandbox')
    const prepared = prepareSandboxCommand(this.name, request, this.#config.commandPolicy)
    const sandbox = await host.createSandbox({
      command: prepared.command,
      cpu: this.#config.cpu,
      image: this.#config.image,
      memoryLimitMb: this.#config.memoryLimitMb,
      toolName: prepared.toolName,
      ...(prepared.signal === undefined ? {} : { signal: prepared.signal }),
    })
    assertSandbox(sandbox)
    const result = await waitAndCloseSandbox(this.name, sandbox, prepared.signal)
    const withSandboxId = result.resourceId === undefined ? { ...result, resourceId: sandbox.id } : result
    return serializeSandboxCommandResult(this.name, prepared.command, withSandboxId)
  }

  isAvailable(): Promise<boolean> {
    return hostAdapterIsAvailable(this.#host)
  }

  getCapabilities(): Readonly<Record<string, unknown>> {
    return Object.freeze({
      backend: this.name,
      hostConfigured: this.#host !== undefined,
      image: this.#config.image,
      cpu: this.#config.cpu,
      memoryLimitMb: this.#config.memoryLimitMb,
      lifecycle: 'create_wait_close',
      commandTransport: 'direct_argv_host_port',
      filesystemIsolation: 'host_defined',
      networkIsolation: 'host_defined',
    })
  }
}

function normalizeConfig(config: ModalSandboxConfig | undefined): ResolvedModalSandboxConfig {
  const source = config ?? {}
  const cpu = source.cpu ?? DEFAULT_MODAL_CPU
  if (typeof cpu !== 'number' || !Number.isFinite(cpu) || cpu <= 0) {
    throw new TypeError('modal cpu must be a positive finite number')
  }
  return Object.freeze({
    commandPolicy: normalizeSandboxCommandPolicy('modal', source),
    cpu,
    image: requiredText('modal', source.image ?? DEFAULT_MODAL_IMAGE, 'image'),
    memoryLimitMb: positiveSafeInteger('modal', source.memoryLimitMb ?? DEFAULT_MODAL_MEMORY_MB, 'memoryLimitMb'),
  })
}

function assertSandbox(sandbox: ModalSandbox): void {
  if (
    sandbox === null
    || typeof sandbox !== 'object'
    || typeof sandbox.id !== 'string'
    || !sandbox.id.trim()
    || typeof sandbox.wait !== 'function'
    || typeof sandbox.close !== 'function'
  ) {
    throw new SandboxBackendProtocolError('modal', 'host must return a sandbox with id, wait(), and close()')
  }
}

async function waitAndCloseSandbox(
  backend: 'modal',
  sandbox: ModalSandbox,
  signal: AbortSignal | undefined,
): Promise<SandboxCommandResult> {
  let executionFailed = false
  let executionError: unknown
  let result: SandboxCommandResult | undefined
  try {
    result = await sandbox.wait(signal)
  } catch (error) {
    executionFailed = true
    executionError = error
  }

  let cleanupFailed = false
  let cleanupError: unknown
  try {
    await sandbox.close()
  } catch (error) {
    cleanupFailed = true
    cleanupError = error
  }

  if (executionFailed && cleanupFailed) {
    throw new AggregateError(
      [executionError, cleanupError],
      'Modal sandbox execution and cleanup both failed',
    )
  }
  if (executionFailed) throw executionError
  if (cleanupFailed) throw cleanupError
  if (result === undefined) {
    throw new SandboxBackendProtocolError(backend, 'sandbox wait completed without a result')
  }
  return result
}
