// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type ProcessSignal = number | NodeJS.Signals

/**
 * Minimal Bun subprocess boundary retained by the background-process registry.
 *
 * Bun.Subprocess satisfies this shape directly. Keeping the boundary small
 * also makes lifecycle behavior deterministic in tests without spawning a
 * real child process.
 */
export interface BunSubprocessLike {
  readonly exited: Promise<number>
  readonly exitCode: number | null
  readonly pid: number
  kill(signal?: ProcessSignal): void
}

/** Immutable descriptive record for one registered background process. */
export interface ProcessRecord {
  readonly command: string
  readonly cwd: string | null
  readonly metadata: Readonly<Record<string, unknown>>
  readonly name: string
  readonly pid: number
  readonly procId: string
  /** Epoch seconds, matching the persisted Python process-registry shape. */
  readonly startedAt: number
}

export interface ProcessRegistrationOptions {
  readonly command?: string
  readonly cwd?: string | null
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly name?: string
}

export interface ProcessRegistryOptions {
  /** Injectable deterministic ID source; values must be non-empty and unique while registered. */
  readonly idFactory?: () => string
  /** Epoch-seconds clock used only when a process is registered. */
  readonly now?: () => number
}

const MAX_ID_ATTEMPTS = 100

/**
 * In-memory registry for process handles started by background execution tools.
 *
 * The JavaScript runtime serializes synchronous map updates naturally; async
 * wait operations hold no locks, so a slow child process cannot block listing,
 * removal, or signals for other tracked processes.
 */
export class ProcessRegistry {
  private readonly handles = new Map<string, BunSubprocessLike>()
  private readonly idFactory: () => string
  private readonly now: () => number
  private readonly records = new Map<string, ProcessRecord>()

  constructor(options: ProcessRegistryOptions = {}) {
    this.idFactory = options.idFactory ?? defaultProcessId
    this.now = options.now ?? (() => Date.now() / 1_000)
  }

  get size(): number {
    return this.records.size
  }

  /** Register one live Bun subprocess and return its stable process identifier. */
  register(process: BunSubprocessLike, options: ProcessRegistrationOptions = {}): string {
    const pid = validatePid(process.pid)
    const procId = this.nextId()
    const startedAt = validateTimestamp(this.now())
    const record = freezeRecord({
      procId,
      pid,
      name: options.name?.trim() || 'pid-' + pid,
      command: options.command ?? '',
      cwd: options.cwd ?? null,
      metadata: { ...(options.metadata ?? {}) },
      startedAt,
    })
    this.handles.set(procId, process)
    this.records.set(procId, record)
    return procId
  }

  /** Return immutable snapshots in registration order, including exited handles until removal. */
  list(): ProcessRecord[] {
    return [...this.records.values()].map(copyRecord)
  }

  /** Return the original live handle for direct advanced inspection, if it is still registered. */
  get(procId: string): BunSubprocessLike | undefined {
    return this.handles.get(procId)
  }

  /** Return an immutable record snapshot, or undefined when the identifier is unknown. */
  record(procId: string): ProcessRecord | undefined {
    const record = this.records.get(procId)
    return record === undefined ? undefined : copyRecord(record)
  }

  /**
   * Return a process exit code, null while the process is running, or
   * undefined when no process is registered under the supplied identifier.
   */
  poll(procId: string): number | null | undefined {
    const process = this.handles.get(procId)
    return process === undefined ? undefined : normalizedExitCode(process.exitCode)
  }

  /**
   * Wait asynchronously for a process to exit.
   *
   * The timeout is in seconds, matching the Python API. It returns null when
   * the timeout expires or the subprocess reports a rejected exit promise,
   * and undefined only when the process identifier is unknown.
   */
  async wait(procId: string, timeout?: number): Promise<number | null | undefined> {
    const process = this.handles.get(procId)
    if (process === undefined) {
      return undefined
    }
    const current = normalizedExitCode(process.exitCode)
    if (current !== null) {
      return current
    }
    const timeoutMilliseconds = timeout === undefined ? undefined : timeoutToMilliseconds(timeout)
    try {
      if (timeoutMilliseconds === undefined) {
        return normalizedExitCode(await process.exited)
      }
      return await waitWithTimeout(process.exited, timeoutMilliseconds)
    } catch {
      return null
    }
  }

  /** Deliver SIGTERM to a still-running registered process. */
  terminate(procId: string): boolean {
    return this.sendSignal(procId, 'SIGTERM')
  }

  /** Deliver SIGKILL to a still-running registered process. */
  kill(procId: string): boolean {
    return this.sendSignal(procId, 'SIGKILL')
  }

  /** Deliver a concrete signal to a still-running registered process. */
  signal(procId: string, signal: ProcessSignal): boolean {
    return this.sendSignal(procId, signal)
  }

  /** Forget one process without sending it a signal. */
  remove(procId: string): boolean {
    const present = this.handles.delete(procId)
    this.records.delete(procId)
    return present
  }

  /** Forget every registered process without sending any signals. */
  clear(): number {
    const count = this.records.size
    this.handles.clear()
    this.records.clear()
    return count
  }

  private nextId(): string {
    for (let attempt = 0; attempt < MAX_ID_ATTEMPTS; attempt += 1) {
      const candidate = this.idFactory().trim()
      if (!candidate) {
        throw new TypeError('process id factory returned an empty identifier')
      }
      if (!this.handles.has(candidate)) {
        return candidate
      }
    }
    throw new Error('process id factory produced too many duplicate identifiers')
  }

  private sendSignal(procId: string, signal: ProcessSignal): boolean {
    const process = this.handles.get(procId)
    if (process === undefined || normalizedExitCode(process.exitCode) !== null) {
      return false
    }
    try {
      process.kill(signal)
      return true
    } catch {
      return false
    }
  }
}

let defaultRegistry: ProcessRegistry | undefined

/** Return the lazily-created process-wide registry for non-injected background tools. */
export function getDefaultProcessRegistry(): ProcessRegistry {
  defaultRegistry ??= new ProcessRegistry()
  return defaultRegistry
}

/** Python-compatible singleton accessor name. */
export const getDefaultRegistry = getDefaultProcessRegistry

function defaultProcessId(): string {
  return crypto.randomUUID().replaceAll('-', '').slice(0, 12)
}

function validatePid(pid: number): number {
  if (!Number.isInteger(pid) || pid < 1) {
    throw new TypeError('process.pid must be a positive integer')
  }
  return pid
}

function validateTimestamp(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError('process registry clock must return a non-negative finite epoch timestamp')
  }
  return value
}

function timeoutToMilliseconds(timeout: number): number {
  if (!Number.isFinite(timeout) || timeout < 0) {
    return 0
  }
  return Math.floor(timeout * 1_000)
}

function normalizedExitCode(value: number | null): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

async function waitWithTimeout(exited: Promise<number>, timeoutMilliseconds: number): Promise<number | null> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<null>(resolveTimeout => {
    timer = setTimeout(() => resolveTimeout(null), timeoutMilliseconds)
  })
  try {
    return await Promise.race([
      exited.then(normalizedExitCode, () => null),
      timeout,
    ])
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer)
    }
  }
}

function freezeRecord(record: ProcessRecord): ProcessRecord {
  return Object.freeze({
    ...record,
    metadata: Object.freeze({ ...record.metadata }),
  })
}

function copyRecord(record: ProcessRecord): ProcessRecord {
  return freezeRecord(record)
}
