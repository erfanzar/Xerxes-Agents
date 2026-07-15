// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  SandboxBackendAdapterConfigurationError,
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

/** Explicit SSH endpoint configuration. Credentials and environment variables are never discovered here. */
export interface SshConnectionConfig {
  readonly host: string
  /** Opaque key/path reference interpreted only by a caller-owned SSH host. */
  readonly identityFile?: string
  readonly port?: number
  readonly user?: string
}

/** Direct-argv request delivered to an injected SSH transport. */
export interface SshSandboxHostRequest {
  readonly command: SandboxCommand
  readonly connection: SshConnectionConfig
  readonly signal?: AbortSignal
  readonly toolName: string
}

/** External SSH boundary. It may use OpenSSH, an agent, or a library selected by the caller. */
export interface SshSandboxHost extends SandboxHostProbe {
  executeRemote(request: SshSandboxHostRequest): Promise<SandboxCommandResult>
}

/** SSH endpoint selection plus generic direct-argv execution policy. */
export interface SshSandboxConfig extends SandboxCommandPolicyOptions, SshConnectionConfig {}

export interface SshSandboxAdapterOptions {
  readonly config: SshSandboxConfig
  readonly host?: SshSandboxHost
}

interface ResolvedSshSandboxConfig {
  readonly commandPolicy: SandboxCommandPolicy
  readonly connection: SshConnectionConfig
}

/**
 * SSH sandbox adapter that requires an explicit endpoint and caller-owned
 * transport. It does not consult `XERXES_SSH_HOST`, spawn `ssh`, or compose a
 * shell command from untrusted inputs.
 */
export class SshSandboxAdapter implements SandboxBackendAdapter {
  readonly name = 'ssh' as const
  readonly #config: ResolvedSshSandboxConfig
  readonly #host: SshSandboxHost | undefined

  constructor(options: SshSandboxAdapterOptions) {
    this.#config = normalizeConfig(options.config)
    this.#host = options.host
  }

  async execute(request: Parameters<SandboxBackendAdapter['execute']>[0]): Promise<string> {
    const host = requiredHostMethod(this.name, this.#host, 'executing an SSH sandbox command', 'executeRemote')
    const prepared = prepareSandboxCommand(this.name, request, this.#config.commandPolicy)
    const result = await host.executeRemote({
      command: prepared.command,
      connection: this.#config.connection,
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
      endpointConfigured: true,
      port: this.#config.connection.port,
      userConfigured: this.#config.connection.user !== undefined,
      identityConfigured: this.#config.connection.identityFile !== undefined,
      commandTransport: 'direct_argv_host_port',
      filesystemIsolation: 'remote_host_defined',
      networkIsolation: 'remote_host_defined',
    })
  }
}

function normalizeConfig(config: SshSandboxConfig): ResolvedSshSandboxConfig {
  if (config === null || typeof config !== 'object') {
    throw new SandboxBackendAdapterConfigurationError('ssh', 'config with an explicit host is required')
  }
  const port = config.port === undefined ? undefined : positiveSafeInteger('ssh', config.port, 'port')
  if (port !== undefined && port > 65_535) {
    throw new SandboxBackendAdapterConfigurationError('ssh', 'port must not exceed 65535', { port })
  }
  const user = config.user === undefined ? undefined : requiredText('ssh', config.user, 'user')
  const identityFile = config.identityFile === undefined
    ? undefined
    : requiredText('ssh', config.identityFile, 'identityFile')
  return Object.freeze({
    commandPolicy: normalizeSandboxCommandPolicy('ssh', config),
    connection: Object.freeze({
      host: requiredText('ssh', config.host, 'host'),
      ...(user === undefined ? {} : { user }),
      ...(port === undefined ? {} : { port }),
      ...(identityFile === undefined ? {} : { identityFile }),
    }),
  })
}
