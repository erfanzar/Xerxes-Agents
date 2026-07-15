// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

type MaybePromise<Value> = Promise<Value> | Value

export interface WandBRun {
  readonly url?: string
  finish(): MaybePromise<void>
  log(metrics: Readonly<Record<string, unknown>>): MaybePromise<void>
}

export interface WandBStartInput {
  readonly config: Readonly<Record<string, unknown>>
  readonly entity?: string
  readonly name?: string
  readonly project: string
}

/** Explicit telemetry boundary implemented by a host-owned WandB SDK adapter. */
export interface WandBTelemetryPort {
  startRun(input: WandBStartInput): MaybePromise<WandBRun>
}

export interface WandBHookOptions {
  readonly entity?: string
  readonly project?: string
  readonly telemetry?: WandBTelemetryPort
}

export interface WandBStartOptions {
  readonly name?: string
}

/**
 * Optional WandB metrics hook with no SDK import, credential lookup, or global singleton.
 *
 * When a host does not supply a telemetry port, start/log/finish are harmless
 * no-ops just like the optional Python integration; once supplied, transport
 * errors propagate so callers never mistake a failed remote write for success.
 */
export class WandBHook {
  readonly entity: string | undefined
  readonly project: string
  readonly telemetry: WandBTelemetryPort | undefined
  private activeRun: WandBRun | undefined

  constructor(options: WandBHookOptions = {}) {
    this.project = requiredText(options.project ?? 'xerxes-rl', 'project')
    this.entity = options.entity === undefined ? undefined : requiredText(options.entity, 'entity')
    this.telemetry = options.telemetry
  }

  /** Whether a host supplied a real telemetry adapter for this hook. */
  isAvailable(): boolean {
    return this.telemetry !== undefined
  }

  /** Start one telemetry run and return its dashboard URL, or an empty string when disabled. */
  async start(config: Readonly<Record<string, unknown>>, options: WandBStartOptions = {}): Promise<string> {
    const telemetry = this.telemetry
    if (telemetry === undefined) return ''
    if (this.activeRun !== undefined) await this.finish()
    const name = options.name === undefined ? undefined : requiredText(options.name, 'name')
    const run = await telemetry.startRun({
      project: this.project,
      config: Object.freeze({ ...config }),
      ...(this.entity === undefined ? {} : { entity: this.entity }),
      ...(name === undefined ? {} : { name }),
    })
    this.activeRun = run
    return typeof run.url === 'string' ? run.url : ''
  }

  /** Forward metrics to the active host telemetry run, if there is one. */
  async log(metrics: Readonly<Record<string, unknown>>): Promise<void> {
    const run = this.activeRun
    if (run === undefined) return
    await run.log(Object.freeze({ ...metrics }))
  }

  /** Close the active run and clear it even when the host SDK rejects the finish call. */
  async finish(): Promise<void> {
    const run = this.activeRun
    if (run === undefined) return
    try {
      await run.finish()
    } finally {
      this.activeRun = undefined
    }
  }
}

function requiredText(value: string, name: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${name} must be non-empty`)
  return value.trim()
}
