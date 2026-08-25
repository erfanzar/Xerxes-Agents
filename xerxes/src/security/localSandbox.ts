// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { XerxesError } from '../core/errors.js'
import type { SandboxBackend, SandboxExecutionRequest } from './sandbox.js'

export const LocalSandboxRunner = {
  BUBBLEWRAP: 'bubblewrap',
  LANDLOCK: 'landlock',
  SEATBELT: 'seatbelt',
  WINDOWS_ACL: 'windows-acl',
} as const

export type LocalSandboxRunner = (typeof LocalSandboxRunner)[keyof typeof LocalSandboxRunner]
export type LocalSandboxEnforcement = 'full' | 'partial' | 'none'

export interface LocalSandboxProbeResult {
  readonly available: boolean
  readonly enforcement: LocalSandboxEnforcement
  readonly filesystemIsolation: boolean
  readonly limitations: readonly string[]
  readonly networkIsolation: boolean
  readonly processIsolation: boolean
}

export interface LocalSandboxPolicy {
  readonly environment: Readonly<Record<string, string>>
  readonly memoryLimitMb: number
  readonly mountPaths: Readonly<Record<string, string>>
  readonly mountReadonly: boolean
  readonly networkAccess: boolean
  readonly timeoutMs: number
  readonly workingDirectory?: string
}

export interface LocalSandboxHost {
  readonly platform: NodeJS.Platform
  probe(runner: LocalSandboxRunner): Promise<LocalSandboxProbeResult>
  execute(
    runner: LocalSandboxRunner,
    request: SandboxExecutionRequest,
    policy: LocalSandboxPolicy,
  ): Promise<string>
}

export class LocalSandboxUnavailableError extends XerxesError {
  readonly code = 'SANDBOX_UNAVAILABLE' as const
  readonly platform: NodeJS.Platform

  constructor(platform: NodeJS.Platform, limitations: readonly string[] = []) {
    super(`SANDBOX_UNAVAILABLE: no enforceable local sandbox runner is available on ${platform}`, {
      code: 'SANDBOX_UNAVAILABLE',
      platform,
      limitations: [...limitations],
    })
    this.platform = platform
  }
}

interface SelectedLocalSandboxRunner {
  readonly runner: LocalSandboxRunner
  readonly probe: LocalSandboxProbeResult
}

/**
 * Selects an OS isolation mechanism through a caller-owned host boundary.
 *
 * The backend never substitutes the process-only subprocess backend. If every
 * platform runner fails its functional probe, execution fails closed with
 * `SANDBOX_UNAVAILABLE`.
 */
export class LocalSandboxBackend implements SandboxBackend {
  readonly #host: LocalSandboxHost
  readonly #policy: LocalSandboxPolicy
  #selection: Promise<SelectedLocalSandboxRunner | undefined> | undefined
  #lastLimitations: readonly string[] = []
  #selected: SelectedLocalSandboxRunner | undefined

  constructor(host: LocalSandboxHost, policy: LocalSandboxPolicy) {
    this.#host = host
    this.#policy = Object.freeze({
      ...policy,
      environment: Object.freeze({ ...policy.environment }),
      mountPaths: Object.freeze({ ...policy.mountPaths }),
    })
  }

  async execute(request: SandboxExecutionRequest): Promise<string> {
    const selected = await this.#select()
    if (selected === undefined) {
      throw new LocalSandboxUnavailableError(this.#host.platform, this.#lastLimitations)
    }
    return this.#host.execute(selected.runner, request, this.#policy)
  }

  async isAvailable(): Promise<boolean> {
    return (await this.#select()) !== undefined
  }

  getUnavailableError(): LocalSandboxUnavailableError {
    return new LocalSandboxUnavailableError(this.#host.platform, this.#lastLimitations)
  }

  getCapabilities(): Readonly<Record<string, unknown>> {
    return Object.freeze({
      backend: 'local',
      platform: this.#host.platform,
      available: this.#selected !== undefined,
      runner: this.#selected?.runner,
      enforcement: this.#selected?.probe.enforcement ?? 'none',
      limitations: [...(this.#selected?.probe.limitations ?? this.#lastLimitations)],
      failClosed: true,
      filesystemIsolation: this.#selected?.probe.filesystemIsolation ?? false,
      networkIsolation: this.#selected?.probe.networkIsolation ?? false,
      processIsolation: this.#selected?.probe.processIsolation ?? false,
    })
  }

  async #select(): Promise<SelectedLocalSandboxRunner | undefined> {
    this.#selection ??= this.#probeRunners()
    return this.#selection
  }

  async #probeRunners(): Promise<SelectedLocalSandboxRunner | undefined> {
    const limitations: string[] = []
    for (const runner of runnersForPlatform(this.#host.platform)) {
      let probe: LocalSandboxProbeResult
      try {
        probe = await this.#host.probe(runner)
      } catch (error) {
        limitations.push(`${runner} probe failed: ${errorMessage(error)}`)
        continue
      }
      limitations.push(...probe.limitations)
      if (probe.available && probe.enforcement !== 'none' && satisfiesPolicy(probe, this.#policy)) {
        const selected = Object.freeze({
          runner,
          probe: Object.freeze({ ...probe, limitations: Object.freeze([...probe.limitations]) }),
        })
        this.#selected = selected
        this.#lastLimitations = selected.probe.limitations
        return selected
      }
    }
    this.#lastLimitations = Object.freeze(limitations)
    return undefined
  }
}

function satisfiesPolicy(probe: LocalSandboxProbeResult, policy: LocalSandboxPolicy): boolean {
  if (!probe.filesystemIsolation) return false
  if (!policy.networkAccess && !probe.networkIsolation) return false
  return true
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function runnersForPlatform(platform: NodeJS.Platform): readonly LocalSandboxRunner[] {
  switch (platform) {
    case 'linux':
      return [LocalSandboxRunner.BUBBLEWRAP, LocalSandboxRunner.LANDLOCK]
    case 'darwin':
      return [LocalSandboxRunner.SEATBELT]
    case 'win32':
      return [LocalSandboxRunner.WINDOWS_ACL]
    default:
      return []
  }
}
