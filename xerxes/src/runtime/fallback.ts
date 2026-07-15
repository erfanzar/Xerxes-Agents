// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Ordered provider/tool choices for a capability. */
export class FallbackChain {
  readonly alternatives: readonly string[]
  readonly capability: string
  readonly preferred: string

  constructor(capability: string, preferred: string, alternatives: Iterable<string> = []) {
    this.capability = capability
    this.preferred = preferred
    this.alternatives = [...alternatives]
  }

  /** Return the preferred choice followed by unique, non-empty alternatives. */
  order(): string[] {
    const seen = new Set<string>()
    const result: string[] = []
    for (const candidate of [this.preferred, ...this.alternatives]) {
      if (!candidate || seen.has(candidate)) continue
      seen.add(candidate)
      result.push(candidate)
    }
    return result
  }
}

/** Capability-keyed fallback routing registry. */
export class FallbackRegistry {
  private readonly chains = new Map<string, FallbackChain>()

  set(capability: string, preferred: string, alternatives: Iterable<string> = []): void {
    this.chains.set(capability, new FallbackChain(capability, preferred, alternatives))
  }

  get(capability: string): FallbackChain | undefined {
    return this.chains.get(capability)
  }

  orderFor(capability: string): string[] {
    return this.chains.get(capability)?.order() ?? []
  }

  nextAfter(capability: string, current: string): string | undefined {
    const order = this.orderFor(capability)
    const index = order.indexOf(current)
    return index >= 0 ? order[index + 1] : undefined
  }

  remove(capability: string): boolean {
    return this.chains.delete(capability)
  }

  /** Return a shallow copy so callers cannot mutate routing registrations. */
  all(): Map<string, FallbackChain> {
    return new Map(this.chains)
  }
}

export interface HealthSnapshotOptions {
  readonly lastChecked?: number
  readonly latencyMs?: number
  readonly message?: string
  readonly name: string
  readonly status?: string
}

/** Last observation of a probed tool's health. */
export class HealthSnapshot {
  readonly lastChecked: number
  readonly latencyMs: number
  readonly message: string
  readonly name: string
  readonly status: string

  constructor(options: HealthSnapshotOptions) {
    this.name = options.name
    this.status = options.status ?? 'unknown'
    this.latencyMs = options.latencyMs ?? 0
    this.lastChecked = options.lastChecked ?? 0
    this.message = options.message ?? ''
  }
}

export type ProbeFn = () => unknown

export interface ToolHealthProberOptions {
  /** Wall-clock seconds used for snapshots and scheduled due times. */
  readonly now?: () => number
  /** Monotonic seconds used only for probe duration measurement. */
  readonly monotonicNow?: () => number
}

export interface ProbeRegistrationOptions {
  readonly intervalSeconds?: number
}

interface RegisteredProbe {
  readonly intervalSeconds: number
  readonly probe: ProbeFn
}

/** Periodically runs synchronous health probes and retains the latest result per tool. */
export class ToolHealthProber {
  private readonly clock: () => number
  private readonly monotonicClock: () => number
  private readonly nextDue = new Map<string, number>()
  private readonly probes = new Map<string, RegisteredProbe>()
  private readonly latestSnapshots = new Map<string, HealthSnapshot>()

  constructor(options: ToolHealthProberOptions = {}) {
    this.clock = options.now ?? (() => Date.now() / 1_000)
    this.monotonicClock = options.monotonicNow ?? (() => performance.now() / 1_000)
  }

  register(name: string, probe: ProbeFn, options: ProbeRegistrationOptions = {}): void {
    const intervalSeconds = options.intervalSeconds ?? 60
    if (!Number.isFinite(intervalSeconds) || intervalSeconds < 0) {
      throw new RangeError('intervalSeconds must be a non-negative finite number')
    }
    this.probes.set(name, { probe, intervalSeconds })
    this.nextDue.set(name, 0)
    if (!this.latestSnapshots.has(name)) this.latestSnapshots.set(name, emptySnapshot(name))
  }

  unregister(name: string): void {
    this.probes.delete(name)
    this.nextDue.delete(name)
    this.latestSnapshots.delete(name)
  }

  /** Run one registered probe immediately, regardless of its scheduled interval. */
  runOne(name: string, now = this.clock()): HealthSnapshot {
    const registered = this.probes.get(name)
    if (!registered) {
      const snapshot = new HealthSnapshot({ name, lastChecked: now, message: 'not registered' })
      this.latestSnapshots.set(name, snapshot)
      return snapshot
    }

    const started = this.monotonicClock()
    let snapshot: HealthSnapshot
    try {
      const raw = registered.probe()
      const latencyMs = (this.monotonicClock() - started) * 1_000
      snapshot = snapshotFromProbe(name, raw, now, latencyMs)
    } catch (error) {
      const latencyMs = (this.monotonicClock() - started) * 1_000
      snapshot = new HealthSnapshot({
        name,
        status: 'down',
        latencyMs,
        lastChecked: now,
        message: `${errorName(error)}: ${errorMessage(error)}`,
      })
    }
    this.latestSnapshots.set(name, snapshot)
    this.nextDue.set(name, now + registered.intervalSeconds)
    return snapshot
  }

  /** Run each currently due probe and return its fresh snapshots. */
  runDue(now = this.clock()): HealthSnapshot[] {
    const due = [...this.nextDue].flatMap(([name, dueAt]) => dueAt <= now ? [name] : [])
    return due.map(name => this.runOne(name, now))
  }

  snapshot(name: string): HealthSnapshot | undefined {
    return this.latestSnapshots.get(name)
  }

  snapshots(): Map<string, HealthSnapshot> {
    return new Map(this.latestSnapshots)
  }

  healthy(name: string): boolean {
    return this.snapshot(name)?.status === 'ok'
  }
}

function emptySnapshot(name: string): HealthSnapshot {
  return new HealthSnapshot({ name })
}

function snapshotFromProbe(name: string, raw: unknown, now: number, latencyMs: number): HealthSnapshot {
  if (isSnapshotLike(raw)) {
    return new HealthSnapshot({
      name: raw.name || name,
      status: raw.status || 'ok',
      latencyMs: raw.latencyMs || latencyMs,
      lastChecked: now,
      message: raw.message ?? '',
    })
  }
  if (raw === true) return new HealthSnapshot({ name, status: 'ok', latencyMs, lastChecked: now })
  if (raw === false || raw === null || raw === undefined) {
    return new HealthSnapshot({ name, status: 'down', latencyMs, lastChecked: now })
  }
  return new HealthSnapshot({ name, status: 'ok', latencyMs, lastChecked: now, message: String(raw).slice(0, 120) })
}

interface SnapshotLike {
  readonly name: string
  readonly lastChecked?: number
  readonly latencyMs?: number
  readonly message?: string
  readonly status?: string
}

function isSnapshotLike(value: unknown): value is SnapshotLike {
  if (value === null || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return typeof record.name === 'string'
}

function errorName(error: unknown): string {
  return error instanceof Error && error.name ? error.name : 'Error'
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
