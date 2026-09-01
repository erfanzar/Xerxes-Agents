// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { type LocalSandboxHost, type LocalSandboxProbeResult, type LocalSandboxRunner } from './localSandbox.js'
import { type SandboxExecutionRequest } from './sandbox.js'

function executable(name: string): string | null {
  return Bun.which(name)
}

export const defaultLocalSandboxHost: LocalSandboxHost = {
  platform: process.platform,

  async probe(runner: LocalSandboxRunner): Promise<LocalSandboxProbeResult> {
    switch (runner) {
      case 'bubblewrap': {
        const found = executable('bwrap') !== null
        return {
          available: found,
          enforcement: found ? 'full' : 'none',
          filesystemIsolation: found,
          networkIsolation: found,
          processIsolation: found,
          limitations: found ? [] : ['bubblewrap (bwrap) is not installed'],
        }
      }
      case 'landlock': {
        // Landlock is a Linux kernel feature; detection requires unshare or a test binary.
        return {
          available: false,
          enforcement: 'none',
          filesystemIsolation: false,
          networkIsolation: false,
          processIsolation: false,
          limitations: ['Landlock probing requires a compiled helper or kernel feature test'],
        }
      }
      case 'seatbelt': {
        const found = executable('sandbox-exec') !== null
        return {
          available: found,
          enforcement: found ? 'full' : 'none',
          filesystemIsolation: found,
          networkIsolation: found,
          processIsolation: found,
          limitations: found ? [] : ['sandbox-exec is not available on this macOS host'],
        }
      }
      case 'windows-acl': {
        const found = process.platform === 'win32' && executable('powershell.exe') !== null
        return {
          available: found,
          enforcement: found ? 'partial' : 'none',
          filesystemIsolation: found,
          networkIsolation: false,
          processIsolation: found,
          limitations: found
            ? ['network isolation requires Windows Firewall or AppContainer rules']
            : ['windows-acl requires Windows PowerShell'],
        }
      }
    }
  },

  async execute(
    runner: LocalSandboxRunner,
    _request: SandboxExecutionRequest,
    _policy: unknown,
  ): Promise<string> {
    throw new Error(`Default host does not execute commands; use an injected host for ${runner}`)
  },
}
