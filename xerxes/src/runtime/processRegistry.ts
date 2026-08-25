// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type ProcessSignal = number | NodeJS.Signals

/** Process groups exist on POSIX only; Windows falls back to direct child kills. */
const PROCESS_GROUPS_AVAILABLE = process.platform !== 'win32'

/**
 * Grace period between SIGTERM and SIGKILL escalation when terminating a
 * process tree, so a well-behaved child can flush and exit on its own.
 */
export const PROCESS_TREE_GRACE_MS = 2_000

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
  /**
   * Signal name when the process died from a signal, null on normal exit.
   *
   * Optional because Bun.Subprocess satisfies it directly while test doubles
   * may omit it. Without consulting this, a signal-killed child polls as
   * "running" forever: Bun leaves `exitCode` null and reports the death only
   * here.
   */
  readonly signalCode?: string | null
  kill(signal?: ProcessSignal): void
}

/** Shell-conventional exit codes for the common fatal signals (128 + n). */
const SIGNAL_EXIT_CODES: Readonly<Record<string, number>> = {
  SIGHUP: 129,
  SIGINT: 130,
  SIGQUIT: 131,
  SIGABRT: 134,
  SIGKILL: 137,
  SIGTERM: 143,
  SIGSTOP: 159,
  SIGTSTP: 148,
  SIGPIPE: 141,
  SIGALRM: 142,
  SIGUSR1: 138,
  SIGUSR2: 140,
}

/**
 * Exit status of a subprocess that has finished, null while still running.
 *
 * Unlike a bare `exitCode` read, this treats a set `signalCode` as terminal —
 * Bun reports signal deaths only there — mapping them to the conventional
 * 128+n codes (unknown signals to -1) so pollers see an honest terminal value
 * instead of polling forever.
 */
export function terminalExitCode(child: Pick<BunSubprocessLike, 'exitCode' | 'signalCode'>): number | null {
  const exit = normalizedExitCode(child.exitCode)
  if (exit !== null) {
    return exit
  }
  const signal = child.signalCode
  if (typeof signal === 'string' && signal !== '') {
    return SIGNAL_EXIT_CODES[signal] ?? -1
  }
  return null
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
  /**
   * The child was spawned detached and therefore leads its own POSIX process
   * group, so signals can be delivered to the whole subtree via `-pid`.
   * Leave unset for handles whose PID is not a group leader: a group signal to
   * an unrelated live group would be catastrophic.
   */
  readonly processGroupLeader?: boolean
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
  private readonly groupLeaders = new Set<string>()
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
    if (options.processGroupLeader === true) this.groupLeaders.add(procId)
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
    return process === undefined ? undefined : terminalExitCode(process)
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
    const current = terminalExitCode(process)
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
    this.groupLeaders.delete(procId)
    this.records.delete(procId)
    return present
  }

  /** Forget every registered process without sending any signals. */
  clear(): number {
    const count = this.records.size
    this.handles.clear()
    this.groupLeaders.clear()
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
    if (process === undefined || terminalExitCode(process) !== null) {
      return false
    }
    return deliverProcessSignal(process, signal, this.groupLeaders.has(procId))
  }
}

/**
 * Deliver a signal to a subprocess, reaching the whole process group when the
 * child was spawned detached.
 *
 * A direct `child.kill()` reaches only the direct child, which is how a timed
 * out or cancelled command left grandchildren alive: a shell's backgrounded
 * `sleep` kept running — and kept the output pipes open — after its parent was
 * reported dead. Signalling `-pid` reaches every member of the group the
 * detached child leads. When the group delivery fails (the handle is a test
 * double, the group is already gone, or the platform has none) the direct child
 * receives the signal instead, preserving the historical behavior.
 */
export function deliverProcessSignal(
  child: BunSubprocessLike,
  signal: ProcessSignal,
  processGroupLeader: boolean,
): boolean {
  if (processGroupLeader && PROCESS_GROUPS_AVAILABLE) {
    try {
      // Negative PID targets the process GROUP the detached child leads. This
      // is the global kill, not the handle's own `kill(signal)` API.
      process.kill(-child.pid, signal)
      return true
    } catch {
      // ESRCH or equivalent: no live group under this PID. Fall through to the
      // direct child so the caller still gets an honest answer.
    }
  }
  try {
    child.kill(signal)
    return true
  } catch {
    return false
  }
}

/**
 * Terminate one detached subprocess and everything in its process group.
 *
 * SIGTERM first so the child can flush and shut down cleanly; anything still
 * alive after {@link PROCESS_TREE_GRACE_MS} gets SIGKILL. After the leader is
 * gone, {@link sweepProcessGroupAfterExit} re-signals the group once more: a
 * helper forked DURING the kill window (a trap handler spawning a sleeper just
 * before the parent dies) never saw any of the signals above, and watching only
 * the direct child cannot see it either.
 */
export async function terminateProcessSubtree(
  child: BunSubprocessLike,
  options: {
    readonly initialSignal?: ProcessSignal
    readonly processGroupLeader?: boolean
  } = {},
): Promise<void> {
  if (terminalExitCode(child) !== null) {
    return
  }
  const groupLeader = options.processGroupLeader ?? false
  const initialSignal = options.initialSignal ?? 'SIGTERM'
  deliverProcessSignal(child, initialSignal, groupLeader)
  if (initialSignal !== 'SIGKILL' && !(await exitedWithin(child, PROCESS_TREE_GRACE_MS))) {
    deliverProcessSignal(child, 'SIGKILL', groupLeader)
  }
  await sweepProcessGroupAfterExit(child)
}

/**
 * SIGKILL whatever remains of a terminated child's process group, once.
 *
 * Called after the direct child is (or is about to be) dead, this catches
 * members that were forked during the termination window and therefore received
 * none of the earlier signals — the "TERM-trapping shell forks a sleeper, then
 * dies" orphan that survived every prior pass. Harmless when the whole group is
 * already gone: the group-directed kill misses with ESRCH and is swallowed.
 */
export async function sweepProcessGroupAfterExit(child: BunSubprocessLike): Promise<void> {
  if (!PROCESS_GROUPS_AVAILABLE) {
    return
  }
  // The sweep must not race the leader's own death: wait for it (bounded — the
  // caller has already delivered SIGTERM/SIGKILL, so this resolves quickly)
  // before signalling the group one final time. A recycled PID becoming a new
  // group leader within this sub-second window is not a realistic hazard, while
  // skipping the sweep demonstrably orphans late-forked helpers.
  await exitedWithin(child, PROCESS_TREE_GRACE_MS)
  try {
    process.kill(-child.pid, 'SIGKILL')
  } catch {
    // ESRCH: nothing left under this group. Done.
  }
}

/** Resolve whether the subprocess exited within `graceMs`, polling its own exit promise. */
async function exitedWithin(child: BunSubprocessLike, graceMs: number): Promise<boolean> {
  if (terminalExitCode(child) !== null) {
    return true
  }
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race([
      child.exited.then(() => true, () => true),
      new Promise<boolean>(resolve => {
        timer = setTimeout(() => resolve(false), graceMs)
      }),
    ])
  } finally {
    if (timer !== undefined) clearTimeout(timer)
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
