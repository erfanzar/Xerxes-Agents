// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
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

const DEFAULT_DOCKER_IMAGE = 'oven/bun:1.3'
const DEFAULT_MEMORY_LIMIT_MB = 512

/** One bind mount selected by the caller and passed as data to a Docker host port. */
export interface DockerSandboxMount {
  readonly containerPath: string
  readonly hostPath: string
  readonly readOnly: boolean
}

/** Configuration retained by the Docker adapter; no Docker client is selected here. */
export interface DockerSandboxConfig extends SandboxCommandPolicyOptions {
  readonly image?: string
  readonly memoryLimitMb?: number
  readonly mountPaths?: Readonly<Record<string, string>>
  readonly mountReadonly?: boolean
  readonly networkAccess?: boolean
}

/** Fully resolved container execution delivered to an injected Docker implementation. */
export interface DockerSandboxHostRequest {
  readonly command: SandboxCommand
  readonly image: string
  readonly memoryLimitMb: number
  readonly mounts: readonly DockerSandboxMount[]
  readonly networkAccess: boolean
  readonly signal?: AbortSignal
  readonly toolName: string
}

/** External Docker boundary. Hosts may use an SDK, API, or daemon bridge outside this module. */
export interface DockerSandboxHost extends SandboxHostProbe {
  runContainer(request: DockerSandboxHostRequest): Promise<SandboxCommandResult>
}

export interface DockerSandboxAdapterOptions {
  readonly config?: DockerSandboxConfig
  readonly host?: DockerSandboxHost
}

interface ResolvedDockerSandboxConfig {
  readonly commandPolicy: SandboxCommandPolicy
  readonly image: string
  readonly memoryLimitMb: number
  readonly mounts: readonly DockerSandboxMount[]
  readonly networkAccess: boolean
}

/**
 * Docker sandbox adapter with no ambient Docker discovery or process launch.
 *
 * A host owns the actual daemon/API transport. This adapter only validates the
 * execution contract and translates an existing `SandboxExecutionRequest` into
 * an immutable host request.
 */
export class DockerSandboxAdapter implements SandboxBackendAdapter {
  readonly name = 'docker' as const
  readonly #config: ResolvedDockerSandboxConfig
  readonly #host: DockerSandboxHost | undefined

  constructor(options: DockerSandboxAdapterOptions = {}) {
    this.#config = normalizeConfig(options.config)
    this.#host = options.host
  }

  async execute(request: Parameters<SandboxBackendAdapter['execute']>[0]): Promise<string> {
    const host = requiredHostMethod(this.name, this.#host, 'executing a Docker sandbox command', 'runContainer')
    const prepared = prepareSandboxCommand(this.name, request, this.#config.commandPolicy)
    const result = await host.runContainer({
      command: prepared.command,
      image: this.#config.image,
      memoryLimitMb: this.#config.memoryLimitMb,
      mounts: this.#config.mounts,
      networkAccess: this.#config.networkAccess,
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
      image: this.#config.image,
      memoryLimitMb: this.#config.memoryLimitMb,
      networkAccess: this.#config.networkAccess,
      mountCount: this.#config.mounts.length,
      commandTransport: 'direct_argv_host_port',
      filesystemIsolation: 'host_defined',
      networkIsolation: 'host_defined',
    })
  }
}

function normalizeConfig(config: DockerSandboxConfig | undefined): ResolvedDockerSandboxConfig {
  const source = config ?? {}
  const image = requiredText('docker', source.image ?? DEFAULT_DOCKER_IMAGE, 'image')
  const memoryLimitMb = positiveSafeInteger('docker', source.memoryLimitMb ?? DEFAULT_MEMORY_LIMIT_MB, 'memoryLimitMb')
  return Object.freeze({
    commandPolicy: normalizeSandboxCommandPolicy('docker', source),
    image,
    memoryLimitMb,
    mounts: normalizeMounts(source.mountPaths, source.mountReadonly ?? true),
    networkAccess: source.networkAccess ?? false,
  })
}

function normalizeMounts(
  paths: Readonly<Record<string, string>> | undefined,
  readOnly: boolean,
): readonly DockerSandboxMount[] {
  if (typeof readOnly !== 'boolean') {
    throw new TypeError('docker mountReadonly must be a boolean')
  }
  const mounts: DockerSandboxMount[] = []
  for (const [hostPath, containerPath] of Object.entries(paths ?? {}).sort(([left], [right]) => left.localeCompare(right))) {
    mounts.push(Object.freeze({
      hostPath: requiredText('docker', hostPath, 'mountPaths host path'),
      containerPath: requiredText('docker', containerPath, 'mountPaths container path'),
      readOnly,
    }))
  }
  return Object.freeze(mounts)
}
