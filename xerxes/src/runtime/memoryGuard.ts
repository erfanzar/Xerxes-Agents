// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Memory guard for the commands agents run.
 *
 * Every command an agent starts — a foreground tool call, a background job,
 * anything those start in turn — is a descendant of the daemon, and the
 * operating system bills all of it to Xerxes. A test suite that leaked GPU
 * memory grew past 17 GB on a 16 GB Mac and froze it. Every couple of seconds
 * the guard sums each of the daemon's child process trees by real footprint
 * (on macOS this includes GPU memory, which `ps` does not show) and stops a
 * tree that passes the limit. The tool that started it reports why, so the
 * agent can run less at once instead of retrying the same thing.
 */

import { dlopen, FFIType, ptr } from 'bun:ffi'
import { readFileSync } from 'node:fs'
import { readFile, writeFile } from 'node:fs/promises'
import { totalmem } from 'node:os'

export interface ProcessRow {
  readonly pid: number
  readonly ppid: number
  readonly command: string
}

/** The host capabilities the guard needs; tests inject their own. */
export interface MemoryGuardPorts {
  listProcesses(): Promise<readonly ProcessRow[]>
  /** Physical footprint in bytes, or undefined when it cannot be read. */
  footprintBytes(pid: number): number | undefined
  kill(pid: number, signal: 'SIGTERM' | 'SIGKILL'): void
}

export interface MemoryGuardStop {
  readonly rootPid: number
  readonly pids: readonly number[]
  readonly command: string
  readonly bytes: number
  readonly limitBytes: number
  readonly at: number
}

export interface MemoryGuardOptions {
  /** The daemon: its child trees are what the guard watches. */
  readonly rootPid: number
  /** Current limit in bytes; 0 or less turns the guard off. */
  readonly limitBytes: () => number
  readonly ports: MemoryGuardPorts
  readonly onStop?: (stop: MemoryGuardStop) => void
  /** Pids never stopped: a person's own terminal, for example. */
  readonly exclude?: () => ReadonlySet<number>
  /** SIGTERM first; whatever is still alive this long after gets SIGKILL. */
  readonly graceMs?: number
}

const DEFAULT_GRACE_MS = 3_000
const DEFAULT_INTERVAL_MS = 2_000
const MAX_NOTES = 512

/** Why a process was stopped, by pid, for the tool that started it. */
const notes = new Map<number, string>()

/** The guard's note for a process it stopped (or one in a tree it stopped). */
export function memoryGuardNote(pid: number | undefined): string | undefined {
  return pid === undefined ? undefined : notes.get(pid)
}

function rememberNote(pids: readonly number[], note: string): void {
  for (const pid of pids) {
    notes.delete(pid)
    notes.set(pid, note)
  }
  while (notes.size > MAX_NOTES) notes.delete(notes.keys().next().value!)
}

export function formatBytes(bytes: number): string {
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`
}

export function memoryGuardStopNote(stop: Pick<MemoryGuardStop, 'bytes' | 'limitBytes'>): string {
  return `[memory guard] Stopped: this command and the processes it started used ${formatBytes(stop.bytes)} of memory, `
    + `over the ${formatBytes(stop.limitBytes)} limit, so it was ended to keep the computer responsive. Do not rerun it `
    + 'unchanged: run less at once (one test file or batch per command), or find what keeps growing. The limit is '
    + 'set in Settings → General.'
}

export class MemoryGuard {
  private readonly stopping = new Set<number>()
  private timer: ReturnType<typeof setInterval> | undefined
  private running = false

  constructor(private readonly options: MemoryGuardOptions) {}

  /** One pass: measure every child tree of the root and stop those over the limit. */
  async tick(): Promise<readonly MemoryGuardStop[]> {
    const limitBytes = this.options.limitBytes()
    if (!(limitBytes > 0)) return []
    const rows = await this.options.ports.listProcesses()
    const children = new Map<number, number[]>()
    const commands = new Map<number, string>()
    for (const row of rows) {
      commands.set(row.pid, row.command)
      const siblings = children.get(row.ppid)
      if (siblings) siblings.push(row.pid)
      else children.set(row.ppid, [row.pid])
    }
    const excluded = this.options.exclude?.() ?? new Set<number>()
    const stops: MemoryGuardStop[] = []
    for (const rootPid of children.get(this.options.rootPid) ?? []) {
      if (excluded.has(rootPid) || this.stopping.has(rootPid)) continue
      const pids = subtree(rootPid, children)
      const sizes = new Map(pids.map(pid => [pid, this.options.ports.footprintBytes(pid) ?? 0]))
      let bytes = 0
      for (const size of sizes.values()) bytes += size
      if (bytes <= limitBytes) continue
      // Name the work, not the shell around it: the largest process is what grew.
      const heaviest = pids.reduce((best, pid) => sizes.get(pid)! > sizes.get(best)! ? pid : best, rootPid)
      const stop: MemoryGuardStop = { rootPid, pids, command: commands.get(heaviest) ?? commands.get(rootPid) ?? '', bytes, limitBytes, at: Date.now() }
      this.stopTree(stop)
      stops.push(stop)
    }
    return stops
  }

  start(intervalMs = DEFAULT_INTERVAL_MS): () => void {
    if (this.timer) return () => this.stop()
    this.timer = setInterval(() => {
      if (this.running) return
      this.running = true
      void this.tick().catch(() => { /* A failed pass is retried on the next tick. */ }).finally(() => { this.running = false })
    }, intervalMs)
    this.timer.unref?.()
    return () => this.stop()
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer)
    this.timer = undefined
  }

  private stopTree(stop: MemoryGuardStop): void {
    this.stopping.add(stop.rootPid)
    rememberNote(stop.pids, memoryGuardStopNote(stop))
    // Deepest first, so a parent cannot respawn a child it just lost.
    for (const pid of [...stop.pids].reverse()) this.options.ports.kill(pid, 'SIGTERM')
    const escalate = setTimeout(() => {
      for (const pid of [...stop.pids].reverse()) this.options.ports.kill(pid, 'SIGKILL')
      this.stopping.delete(stop.rootPid)
    }, this.options.graceMs ?? DEFAULT_GRACE_MS)
    escalate.unref?.()
    this.options.onStop?.(stop)
  }
}

function subtree(rootPid: number, children: ReadonlyMap<number, readonly number[]>): number[] {
  const pids: number[] = []
  const queue = [rootPid]
  const seen = new Set<number>()
  while (queue.length) {
    const pid = queue.shift()!
    if (seen.has(pid)) continue
    seen.add(pid)
    pids.push(pid)
    queue.push(...(children.get(pid) ?? []))
  }
  return pids
}

/** Half the computer's memory: what one runaway command may take before it is stopped. */
export function defaultMemoryLimitBytes(totalBytes: number = totalmem()): number {
  return Math.floor(totalBytes / 2)
}

export interface MemoryGuardSettings {
  /** Megabytes; 0 is off; undefined is the default (half the computer's memory). */
  readonly limitMb?: number
}

/** Environment override first (`XERXES_COMMAND_MEMORY_LIMIT_MB`, 0 = off), then the saved setting, then the default. */
export function resolveMemoryLimitBytes(
  settings: MemoryGuardSettings,
  environment: Readonly<Record<string, string | undefined>> = process.env,
  totalBytes: number = totalmem(),
): number {
  const fromEnvironment = environment.XERXES_COMMAND_MEMORY_LIMIT_MB?.trim()
  if (fromEnvironment !== undefined && /^\d+$/.test(fromEnvironment)) return Number(fromEnvironment) * 1024 * 1024
  if (settings.limitMb !== undefined) return settings.limitMb * 1024 * 1024
  return defaultMemoryLimitBytes(totalBytes)
}

export async function readMemoryGuardSettings(file: string): Promise<MemoryGuardSettings> {
  try {
    const value: unknown = JSON.parse(await readFile(file, 'utf8'))
    const limit = value && typeof value === 'object' ? (value as Record<string, unknown>).limit_mb : undefined
    return typeof limit === 'number' && Number.isSafeInteger(limit) && limit >= 0 ? { limitMb: limit } : {}
  } catch {
    return {}
  }
}

/** Save the limit in megabytes (0 = off), or undefined to return to the default. */
export async function saveMemoryGuardSettings(file: string, limitMb: number | undefined): Promise<void> {
  if (limitMb !== undefined && (!Number.isSafeInteger(limitMb) || limitMb < 0)) throw new Error('limit_mb must be a whole number of megabytes, 0 or more')
  await writeFile(file, JSON.stringify(limitMb === undefined ? {} : { limit_mb: limitMb }) + '\n', 'utf8')
}

/** Whether this computer's footprints can be read. */
export function memoryGuardSupported(platform: NodeJS.Platform = process.platform): boolean {
  return platform === 'darwin' || platform === 'linux'
}

/** The real host: `ps` for the tree, and the kernel's own footprint count. */
export function hostMemoryGuardPorts(platform: NodeJS.Platform = process.platform): MemoryGuardPorts {
  return {
    async listProcesses() {
      const child = Bun.spawn(['ps', '-A', '-o', 'pid=,ppid=,command='], { stdout: 'pipe', stderr: 'ignore', stdin: 'ignore' })
      const text = await new Response(child.stdout).text()
      await child.exited
      const rows: ProcessRow[] = []
      for (const line of text.split('\n')) {
        const match = /^\s*(\d+)\s+(\d+)\s+(.*)$/.exec(line)
        if (match) rows.push({ pid: Number(match[1]), ppid: Number(match[2]), command: match[3]!.trim() })
      }
      return rows
    },
    footprintBytes: platform === 'darwin' ? darwinFootprint() : platform === 'linux' ? linuxFootprint : () => undefined,
    kill(pid, signal) {
      try { process.kill(pid, signal) } catch { /* Already gone. */ }
    },
  }
}

/**
 * macOS: `proc_pid_rusage` (RUSAGE_INFO_V4) `ri_phys_footprint` — the figure
 * Activity Monitor and Force Quit show, GPU memory included.
 */
function darwinFootprint(): (pid: number) => number | undefined {
  try {
    // The symbol lives in libSystem.
    const library = dlopen('/usr/lib/libSystem.B.dylib', {
      proc_pid_rusage: { args: [FFIType.i32, FFIType.i32, FFIType.ptr], returns: FFIType.i32 },
    })
    const RUSAGE_INFO_V4 = 4
    // ri_uuid[16] is two uint64 slots; ri_phys_footprint is the eighth uint64 after it.
    const PHYS_FOOTPRINT_SLOT = 2 + 7
    const buffer = new BigUint64Array(64)
    return pid => {
      if (library.symbols.proc_pid_rusage(pid, RUSAGE_INFO_V4, ptr(buffer)) !== 0) return undefined
      return Number(buffer[PHYS_FOOTPRINT_SLOT])
    }
  } catch {
    return () => undefined
  }
}

/** Linux: resident memory from /proc (GPU memory is not counted there). */
function linuxFootprint(pid: number): number | undefined {
  try {
    const match = /^VmRSS:\s+(\d+)\s+kB/m.exec(readFileSync(`/proc/${pid}/status`, 'utf8'))
    return match ? Number(match[1]) * 1024 : undefined
  } catch {
    return undefined
  }
}
