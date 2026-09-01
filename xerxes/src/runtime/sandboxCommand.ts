// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { defaultLocalSandboxHost } from '../security/defaultLocalSandboxHost.js'
import { LocalSandboxBackend, type LocalSandboxPolicy } from '../security/localSandbox.js'

export type SandboxCommandAction = 'status'

export interface SandboxCommandOptions {
  readonly action: SandboxCommandAction
}

export interface SandboxCommandResult {
  readonly ok: boolean
  readonly message?: string
  readonly error?: string
}

export async function runSandboxCommand(options: SandboxCommandOptions): Promise<SandboxCommandResult> {
  switch (options.action) {
    case 'status': {
      const policy: LocalSandboxPolicy = {
        environment: {},
        memoryLimitMb: 128,
        mountPaths: {},
        mountReadonly: true,
        networkAccess: false,
        timeoutMs: 5_000,
      }
      const backend = new LocalSandboxBackend(defaultLocalSandboxHost, policy)
      const capabilities = backend.getCapabilities()
      const limitations = capabilities.limitations as readonly string[]
      const lines = [
        `platform: ${capabilities.platform}`,
        `available: ${capabilities.available}`,
        `runner: ${String(capabilities.runner ?? 'none')}`,
        `enforcement: ${capabilities.enforcement}`,
        `filesystemIsolation: ${capabilities.filesystemIsolation}`,
        `networkIsolation: ${capabilities.networkIsolation}`,
        `processIsolation: ${capabilities.processIsolation}`,
        `failClosed: ${capabilities.failClosed}`,
        `limitations: ${limitations.join(', ') || 'none'}`,
      ]
      return { ok: true, message: lines.join('\n') }
    }
  }
}
