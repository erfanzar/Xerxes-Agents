// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Long-running commands the model starts and then checks on, rather than waits
// out or abandons.
//
// Without this there were only two options for a build, a server, or a training
// run, and both were bad. Waiting meant blocking the turn on a foreground call
// until it hit the 120s ceiling and got killed mid-work. Escaping that meant
// `nohup … &`, which is fire-and-forget: the output goes nowhere the model can
// read, nothing records that the process exists, and the model cannot tell
// success from silent failure — so it guesses, or polls with `ps` and greps for
// a pid it never reliably learned.
//
// A background command here keeps its identity (a `proc_id`), its output (a
// bounded tail, drained continuously so the child never blocks on a full pipe),
// and its exit status. The model starts it, does something else, and polls.

import { ValidationError } from '../core/errors.js'
import { ProcessRegistry, terminalExitCode, type ProcessRecord } from '../runtime/processRegistry.js'
import type { TerminalHandle, TerminalRegistry } from '../runtime/terminalRegistry.js'

import { BoundedOutputBuffer, capOutput, drainStream, type StreamDrain } from './processOutput.js'

/** Output retained per stream per process before the oldest is dropped. */
const OUTPUT_CAPACITY_CHARS = 1_000_000

/** Ceiling on a single `check_command` block, so a poll can never become a wait. */
export const MAX_CHECK_WAIT_MS = 60_000

/** Grace between SIGTERM and the SIGKILL escalation in {@link BackgroundCommandManager.killForOwner}. */
const KILL_GRACE_MS = 2_000

/** Process groups (and therefore whole-tree kills) exist on POSIX only. */
const PROCESS_GROUPS_AVAILABLE = process.platform !== 'win32'

export interface BackgroundStartOptions {
  readonly args?: readonly string[]
  readonly command: string
  readonly cwd: string
  readonly name?: string
}

/**
 * Explicit scope for callers that use the manager directly rather than through
 * an authenticated tool execution context. Symbol identity prevents any real
 * session ID from colliding with this compatibility scope.
 */
export const LEGACY_PRIVATE_BACKGROUND_SCOPE: unique symbol = Symbol('legacy-private-background-commands')

export type BackgroundCommandOwner = string | typeof LEGACY_PRIVATE_BACKGROUND_SCOPE

export interface BackgroundStartResult {
  readonly command: readonly string[]
  readonly cwd: string
  readonly pid: number
  readonly procId: string
  readonly running: true
}

export interface BackgroundCheckResult {
  readonly command: readonly string[]
  /** Set when output was discarded to stay within the retained tail. */
  readonly droppedOutput?: true
  readonly exitCode: number | null
  readonly procId: string
  readonly running: boolean
  readonly stderr: string
  readonly stdout: string
  /** More output is buffered than this response carried; poll again. */
  readonly truncated: boolean
}

interface BackgroundEntry {
  readonly command: readonly string[]
  readonly drains: readonly StreamDrain[]
  readonly owner: BackgroundCommandOwner
  readonly process: Bun.Subprocess
  readonly stderr: BoundedOutputBuffer
  readonly stdout: BoundedOutputBuffer
  readonly terminal?: TerminalHandle
}

/**
 * Owns background child processes across isolated session scopes.
 *
 * Registration goes through {@link ProcessRegistry}, which already had the
 * lifecycle vocabulary — poll, wait, terminate, kill — and no caller at all.
 */
export class BackgroundCommandManager {
  private readonly entries = new Map<string, BackgroundEntry>()
  private readonly terminals: TerminalRegistry | undefined

  constructor(
    private readonly registry: ProcessRegistry = new ProcessRegistry(),
    terminals?: TerminalRegistry,
  ) {
    this.terminals = terminals
  }

  /**
   * Spawn a detached command and return its handle immediately.
   *
   * `stdin` is ignored rather than piped: nothing can answer a prompt for a
   * process the model is not attached to, and leaving it open lets an
   * interactive command wait forever for input that will never come.
   */
  start(options: BackgroundStartOptions): BackgroundStartResult {
    return this.startForOwner(LEGACY_PRIVATE_BACKGROUND_SCOPE, options)
  }

  /** Spawn a command owned by an authenticated session scope. */
  startForOwner(owner: BackgroundCommandOwner, options: BackgroundStartOptions): BackgroundStartResult {
    const argv = [options.command, ...(options.args ?? [])]
    // Detached on POSIX so the child leads its own process group; kills then
    // reach the whole subtree (`cmd &` grandchildren included) instead of just
    // the direct child. The cost is that the child would outlive this process —
    // which is why every teardown path here goes through the registry's
    // group-aware signaling and `exited`-based reaping.
    const child = Bun.spawn(argv, {
      cwd: options.cwd,
      stdin: 'ignore',
      stdout: 'pipe',
      stderr: 'pipe',
      ...(PROCESS_GROUPS_AVAILABLE ? { detached: true } : {}),
    })
    const stdout = new BoundedOutputBuffer(OUTPUT_CAPACITY_CHARS)
    const stderr = new BoundedOutputBuffer(OUTPUT_CAPACITY_CHARS)
    const procId = this.registry.register(child, {
      command: argv.join(' '),
      cwd: options.cwd,
      ...(options.name ? { name: options.name } : {}),
      ...(PROCESS_GROUPS_AVAILABLE ? { processGroupLeader: true } : {}),
    })
    const terminalOwnerSessionId = typeof owner === 'string' ? owner : 'legacy-private-background-commands'
    let terminal: TerminalHandle | undefined
    try {
      terminal = this.terminals?.open({
        id: procId,
        kind: 'background',
        ownerSessionId: terminalOwnerSessionId,
        command: argv.join(' '),
        cwd: options.cwd,
        pid: child.pid,
        ...(options.name ? { label: options.name } : {}),
        control: { kill: async signal => void (await this.killForOwner(owner, procId, signal)) },
      })
    } catch (error) {
      // The child is already live and detached — it would outlive us with no
      // registry entry and no mirror if registration failed here. Kill the
      // whole group, drop the registration, and surface the failure.
      try {
        this.registry.signal(procId, 'SIGKILL')
      } catch {
        // Nothing left to signal; fall through to removal either way.
      }
      this.registry.remove(procId)
      throw error
    }
    // Drain continuously from the moment it starts. A process whose output is
    // only read when polled fills its pipe buffer and blocks — so a build left
    // unpolled for a minute would stall on its own logging.
    const drains = [
      drainStream(child.stdout as ReadableStream<Uint8Array>, stdout, text => terminal?.append(text)),
      drainStream(child.stderr as ReadableStream<Uint8Array>, stderr, text => terminal?.append(text)),
    ]
    this.entries.set(procId, {
      command: argv,
      drains,
      owner,
      process: child,
      stdout,
      stderr,
      ...(terminal ? { terminal } : {}),
    })
    // Close the mirror on natural exit too, not only on an explicit kill: a
    // build that finishes on its own must stop being listed as running.
    void child.exited.then(code => terminal?.close(typeof code === 'number' ? code : null)).catch(() => {})
    return { procId, pid: child.pid, running: true, command: argv, cwd: options.cwd }
  }

  /**
   * Report status and hand back output that has not been read yet.
   *
   * Output is consumed, so successive polls show progress rather than repeating
   * from the beginning. `waitMs` lets a caller give a short-lived command a
   * chance to finish instead of returning "running" and being asked again
   * immediately; it is bounded so a poll cannot silently become a blocking wait.
   */
  check(procId: string, maxOutputChars: number, waitMs = 0): Promise<BackgroundCheckResult> {
    return this.checkForOwner(LEGACY_PRIVATE_BACKGROUND_SCOPE, procId, maxOutputChars, waitMs)
  }

  /** Read a command only when it belongs to the requested owner. */
  async checkForOwner(
    owner: BackgroundCommandOwner,
    procId: string,
    maxOutputChars: number,
    waitMs = 0,
  ): Promise<BackgroundCheckResult> {
    const entry = this.require(procId, owner)
    if (waitMs > 0) {
      await raceBounded(
        entry.process.exited.then(() => undefined, () => undefined),
        Math.min(waitMs, MAX_CHECK_WAIT_MS),
      )
    }
    // terminalExitCode, not a bare exitCode read: a signal-killed child keeps
    // exitCode null forever in Bun and would poll as running for its whole
    // (already over) lifetime.
    const exitCode = terminalExitCode(entry.process)
    const running = exitCode === null
    // A process that has exited may still have output in flight in the pipe;
    // without this the final lines of a completed command are reported as empty.
    if (!running) {
      await raceBounded(
        Promise.all(entry.drains.map(drain => drain.done)).then(() => undefined),
        50,
      )
    }
    const outText = entry.stdout.take(maxOutputChars)
    const errText = entry.stderr.take(maxOutputChars)
    const dropped = entry.stdout.dropped || entry.stderr.dropped
    return {
      procId,
      command: entry.command,
      running,
      exitCode,
      stdout: capOutput(outText.text, maxOutputChars).text,
      stderr: capOutput(errText.text, maxOutputChars).text,
      truncated: outText.truncated || errText.truncated,
      ...(dropped ? { droppedOutput: true as const } : {}),
    }
  }

  /**
   * Signal a background process and stop tracking it.
   *
   * SIGTERM first so the child can shut down cleanly; a caller that needs it
   * gone regardless passes SIGKILL. The signal reaches the child's whole
   * process group (POSIX), and a SIGTERM the child ignores is escalated to
   * SIGKILL after {@link KILL_GRACE_MS}. Reported honestly: `signalled` is
   * false when the process had already exited on its own, because claiming to
   * have killed something that was already dead would misrepresent what
   * happened.
   */
  kill(procId: string, signal: 'SIGKILL' | 'SIGTERM' = 'SIGTERM'): Promise<{
    readonly exitCode: number | null
    readonly procId: string
    readonly signalled: boolean
  }> {
    return this.killForOwner(LEGACY_PRIVATE_BACKGROUND_SCOPE, procId, signal)
  }

  /** Signal and release a command only when it belongs to the requested owner. */
  async killForOwner(
    owner: BackgroundCommandOwner,
    procId: string,
    signal: 'SIGKILL' | 'SIGTERM' = 'SIGTERM',
  ): Promise<{
    readonly exitCode: number | null
    readonly procId: string
    readonly signalled: boolean
  }> {
    const entry = this.require(procId, owner)
    const signalled = this.registry.signal(procId, signal)
    if (signalled) {
      if (signal === 'SIGTERM') {
        // The group was asked to terminate; escalate when it ignored that ask,
        // so "killed" never quietly means "still running with a new signal
        // budget".
        const finished = await this.exitedWithin(entry, KILL_GRACE_MS)
        if (!finished) {
          this.registry.signal(procId, 'SIGKILL')
          await this.exitedWithin(entry, KILL_GRACE_MS)
        }
      } else {
        await this.exitedWithin(entry, KILL_GRACE_MS)
      }
    }
    this.release(procId)
    return { procId, signalled, exitCode: terminalExitCode(entry.process) }
  }

  /** Bounded wait for one entry's process to exit; never rejects, never waits past the grace. */
  private async exitedWithin(entry: BackgroundEntry, graceMs: number): Promise<boolean> {
    return (await raceBounded(entry.process.exited.then(() => true, () => true), graceMs)) === true
  }

  /** Every direct-API process in the private compatibility scope. */
  list(): readonly ProcessRecord[] {
    return this.listForOwner(LEGACY_PRIVATE_BACKGROUND_SCOPE)
  }

  /** Every process in one owner scope, running or exited but not yet reaped. */
  listForOwner(owner: BackgroundCommandOwner): readonly ProcessRecord[] {
    return this.registry.list().filter(record => this.entries.get(record.procId)?.owner === owner)
  }

  /** Terminate and release only commands owned by one session. */
  async disposeOwner(owner: BackgroundCommandOwner): Promise<void> {
    await Promise.all([...this.entries.entries()]
      .filter(([, entry]) => entry.owner === owner)
      .map(async ([procId]) => {
        try {
          await this.killForOwner(owner, procId, 'SIGKILL')
        } catch {
          // Already gone; teardown must not fail on a race with natural exit.
        }
      }))
  }

  /**
   * Terminate everything still running.
   *
   * Called on session teardown: a background process outliving the session that
   * started it has no one left to read its output or notice it failed.
   */
  async disposeAll(): Promise<void> {
    await Promise.all([...this.entries.keys()].map(async procId => {
      try {
        const owner = this.entries.get(procId)?.owner
        if (owner !== undefined) await this.killForOwner(owner, procId, 'SIGKILL')
      } catch {
        // Already gone; teardown must not fail on a race with natural exit.
      }
    }))
  }

  private require(procId: string, owner: BackgroundCommandOwner): BackgroundEntry {
    const entry = this.entries.get(procId)
    if (entry === undefined || entry.owner !== owner) {
      // Deliberately identical for a missing ID and a different owner: proc_id
      // possession is not authorization and must not become an existence oracle.
      throw new ValidationError('proc_id', 'is not a known background command', procId)
    }
    return entry
  }

  private release(procId: string): void {
    const entry = this.entries.get(procId)
    for (const drain of entry?.drains ?? []) drain.cancel()
    entry?.terminal?.close(terminalExitCode(entry.process))
    this.entries.delete(procId)
    this.registry.remove(procId)
  }
}

/**
 * Promise.race against a timeout whose timer clears itself.
 *
 * Without the clear, every losing timeout held the event loop open for its full
 * duration — one leaked 50ms or waitMs timer per poll, forever accumulating.
 */
async function raceBounded<T>(promise: Promise<T>, timeoutMs: number): Promise<T | null> {
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race([
      promise,
      new Promise<null>(resolve => {
        timer = setTimeout(() => resolve(null), timeoutMs)
      }),
    ])
  } finally {
    if (timer !== undefined) clearTimeout(timer)
  }
}
