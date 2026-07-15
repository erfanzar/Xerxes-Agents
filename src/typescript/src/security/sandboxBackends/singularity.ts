// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  SandboxBackendUnavailableError,
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
  SandboxHostProbe,
} from './contracts.js'

const DEFAULT_SINGULARITY_IMAGE = 'docker://oven/bun:1.3'

/** Container runtime names supported by an injected HPC host. */
export type SingularityRuntime = 'apptainer' | 'singularity'

/** Host execution request after the adapter has selected an available runtime. */
export interface SingularitySandboxHostRequest {
  readonly command: SandboxCommand
  readonly image: string
  readonly runtime: SingularityRuntime
  readonly signal?: AbortSignal
  readonly toolName: string
}

/** External HPC container boundary. This module never runs `singularity` or `apptainer` itself. */
export interface SingularitySandboxHost extends SandboxHostProbe {
  executeContainer(request: SingularitySandboxHostRequest): Promise<SandboxCommandResult>
  resolveRuntime(): Promise<SingularityRuntime | undefined>
}

/** Singularity/Apptainer image selection plus generic direct-argv policy. */
export interface SingularitySandboxConfig extends SandboxCommandPolicyOptions {
  readonly image?: string
}

export interface SingularitySandboxAdapterOptions {
  readonly config?: SingularitySandboxConfig
  readonly host?: SingularitySandboxHost
}

interface ResolvedSingularitySandboxConfig {
  readonly commandPolicy: SandboxCommandPolicy
  readonly image: string
}

/**
 * Uses an injected HPC host to resolve and invoke Singularity or Apptainer.
 * A missing runtime is a visible unavailable error; there is no process or
 * Docker fallback hidden in the Bun runtime.
 */
export class SingularitySandboxAdapter implements SandboxBackendAdapter {
  readonly name = 'singularity' as const
  readonly #config: ResolvedSingularitySandboxConfig
  readonly #host: SingularitySandboxHost | undefined

  constructor(options: SingularitySandboxAdapterOptions = {}) {
    this.#config = normalizeConfig(options.config)
    this.#host = options.host
  }

  async execute(request: Parameters<SandboxBackendAdapter['execute']>[0]): Promise<string> {
    const host = requiredHostMethod(this.name, this.#host, 'resolving an HPC container runtime', 'resolveRuntime')
    requiredHostMethod(this.name, host, 'executing an HPC container command', 'executeContainer')
    const prepared = prepareSandboxCommand(this.name, request, this.#config.commandPolicy)
    const runtime = await host.resolveRuntime()
    if (runtime === undefined) {
      throw new SandboxBackendUnavailableError(this.name, 'neither singularity nor apptainer is available from the host')
    }
    const result = await host.executeContainer({
      command: prepared.command,
      image: this.#config.image,
      runtime,
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
      supportedRuntimes: ['singularity', 'apptainer'],
      commandTransport: 'direct_argv_host_port',
      filesystemIsolation: 'host_defined',
      networkIsolation: 'host_defined',
    })
  }
}

function normalizeConfig(config: SingularitySandboxConfig | undefined): ResolvedSingularitySandboxConfig {
  const source = config ?? {}
  return Object.freeze({
    commandPolicy: normalizeSandboxCommandPolicy('singularity', source),
    image: requiredText('singularity', source.image ?? DEFAULT_SINGULARITY_IMAGE, 'image'),
  })
}
