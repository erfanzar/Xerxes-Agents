// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  SandboxBackendProtocolError,
  hostAdapterIsAvailable,
  normalizeSandboxCommandPolicy,
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
  SandboxHostExecutionRequest,
  SandboxHostProbe,
} from './contracts.js'

const DEFAULT_DAYTONA_IMAGE = 'oven/bun:1.3'
const DEFAULT_DAYTONA_REGION = 'us-east-1'

/** Inputs needed to provision one remote Daytona workspace. */
export interface DaytonaWorkspaceCreateRequest {
  readonly image: string
  readonly region: string
}

/** Host-owned Daytona workspace lifecycle. */
export interface DaytonaWorkspace {
  readonly id: string
  delete(): Promise<void>
  execute(request: SandboxHostExecutionRequest): Promise<SandboxCommandResult>
}

/** External Daytona boundary. The caller chooses and authenticates the SDK/API implementation. */
export interface DaytonaSandboxHost extends SandboxHostProbe {
  createWorkspace(request: DaytonaWorkspaceCreateRequest): Promise<DaytonaWorkspace>
}

/** Daytona provisioning and direct-argv policy. */
export interface DaytonaSandboxConfig extends SandboxCommandPolicyOptions {
  readonly region?: string
  readonly workspaceImage?: string
}

export interface DaytonaSandboxAdapterOptions {
  readonly config?: DaytonaSandboxConfig
  readonly host?: DaytonaSandboxHost
}

interface ResolvedDaytonaSandboxConfig {
  readonly commandPolicy: SandboxCommandPolicy
  readonly region: string
  readonly workspaceImage: string
}

/**
 * Provision one Daytona workspace per request, execute through its host-owned
 * API, and always request deletion before publishing a result.
 */
export class DaytonaSandboxAdapter implements SandboxBackendAdapter {
  readonly name = 'daytona' as const
  readonly #config: ResolvedDaytonaSandboxConfig
  readonly #host: DaytonaSandboxHost | undefined

  constructor(options: DaytonaSandboxAdapterOptions = {}) {
    this.#config = normalizeConfig(options.config)
    this.#host = options.host
  }

  async execute(request: Parameters<SandboxBackendAdapter['execute']>[0]): Promise<string> {
    const host = requiredHostMethod(this.name, this.#host, 'creating a Daytona workspace', 'createWorkspace')
    const prepared = prepareSandboxCommand(this.name, request, this.#config.commandPolicy)
    const workspace = await host.createWorkspace({
      image: this.#config.workspaceImage,
      region: this.#config.region,
    })
    assertWorkspace(workspace)
    const result = await executeAndDeleteWorkspace(this.name, workspace, prepared)
    const withWorkspaceId = result.resourceId === undefined ? { ...result, resourceId: workspace.id } : result
    return serializeSandboxCommandResult(this.name, prepared.command, withWorkspaceId)
  }

  isAvailable(): Promise<boolean> {
    return hostAdapterIsAvailable(this.#host)
  }

  getCapabilities(): Readonly<Record<string, unknown>> {
    return Object.freeze({
      backend: this.name,
      hostConfigured: this.#host !== undefined,
      workspaceImage: this.#config.workspaceImage,
      region: this.#config.region,
      lifecycle: 'create_execute_delete',
      commandTransport: 'direct_argv_host_port',
      filesystemIsolation: 'host_defined',
      networkIsolation: 'host_defined',
    })
  }
}

function normalizeConfig(config: DaytonaSandboxConfig | undefined): ResolvedDaytonaSandboxConfig {
  const source = config ?? {}
  return Object.freeze({
    commandPolicy: normalizeSandboxCommandPolicy('daytona', source),
    region: requiredText('daytona', source.region ?? DEFAULT_DAYTONA_REGION, 'region'),
    workspaceImage: requiredText('daytona', source.workspaceImage ?? DEFAULT_DAYTONA_IMAGE, 'workspaceImage'),
  })
}

function assertWorkspace(workspace: DaytonaWorkspace): void {
  if (
    workspace === null
    || typeof workspace !== 'object'
    || typeof workspace.id !== 'string'
    || !workspace.id.trim()
    || typeof workspace.execute !== 'function'
    || typeof workspace.delete !== 'function'
  ) {
    throw new SandboxBackendProtocolError('daytona', 'host must return a workspace with id, execute(), and delete()')
  }
}

async function executeAndDeleteWorkspace(
  backend: 'daytona',
  workspace: DaytonaWorkspace,
  request: SandboxHostExecutionRequest,
): Promise<SandboxCommandResult> {
  let executionFailed = false
  let executionError: unknown
  let result: SandboxCommandResult | undefined
  try {
    result = await workspace.execute(request)
  } catch (error) {
    executionFailed = true
    executionError = error
  }

  let cleanupFailed = false
  let cleanupError: unknown
  try {
    await workspace.delete()
  } catch (error) {
    cleanupFailed = true
    cleanupError = error
  }

  if (executionFailed && cleanupFailed) {
    throw new AggregateError(
      [executionError, cleanupError],
      'Daytona sandbox execution and workspace cleanup both failed',
    )
  }
  if (executionFailed) throw executionError
  if (cleanupFailed) throw cleanupError
  if (result === undefined) {
    throw new SandboxBackendProtocolError(backend, 'workspace execution completed without a result')
  }
  return result
}
