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
import { ProcessRegistry, type ProcessRecord } from '../runtime/processRegistry.js'

import { BoundedOutputBuffer, capOutput, drainStream, type StreamDrain } from './processOutput.js'

/** Output retained per stream per process before the oldest is dropped. */
const OUTPUT_CAPACITY_CHARS = 1_000_000

/** Ceiling on a single `check_command` block, so a poll can never become a wait. */
export const MAX_CHECK_WAIT_MS = 60_000

export interface BackgroundStartOptions {
  readonly args?: readonly string[]
  readonly command: string
  readonly cwd: string
  readonly name?: string
}

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
  readonly process: Bun.Subprocess
  readonly stderr: BoundedOutputBuffer
  readonly stdout: BoundedOutputBuffer
}

/**
 * Owns background child processes for one session.
 *
 * Registration goes through {@link ProcessRegistry}, which already had the
 * lifecycle vocabulary — poll, wait, terminate, kill — and no caller at all.
 */
export class BackgroundCommandManager {
  private readonly entries = new Map<string, BackgroundEntry>()

  constructor(private readonly registry: ProcessRegistry = new ProcessRegistry()) {}

  /**
   * Spawn a detached command and return its handle immediately.
   *
   * `stdin` is ignored rather than piped: nothing can answer a prompt for a
   * process the model is not attached to, and leaving it open lets an
   * interactive command wait forever for input that will never come.
   */
  start(options: BackgroundStartOptions): BackgroundStartResult {
    const argv = [options.command, ...(options.args ?? [])]
    const child = Bun.spawn(argv, {
      cwd: options.cwd,
      stdin: 'ignore',
      stdout: 'pipe',
      stderr: 'pipe',
    })
    const stdout = new BoundedOutputBuffer(OUTPUT_CAPACITY_CHARS)
    const stderr = new BoundedOutputBuffer(OUTPUT_CAPACITY_CHARS)
    // Drain continuously from the moment it starts. A process whose output is
    // only read when polled fills its pipe buffer and blocks — so a build left
    // unpolled for a minute would stall on its own logging.
    const drains = [drainStream(child.stdout as ReadableStream<Uint8Array>, stdout),
      drainStream(child.stderr as ReadableStream<Uint8Array>, stderr)]
    const procId = this.registry.register(child, {
      command: argv.join(' '),
      cwd: options.cwd,
      ...(options.name ? { name: options.name } : {}),
    })
    this.entries.set(procId, { command: argv, drains, process: child, stdout, stderr })
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
  async check(procId: string, maxOutputChars: number, waitMs = 0): Promise<BackgroundCheckResult> {
    const entry = this.require(procId)
    if (waitMs > 0) {
      await Promise.race([
        entry.process.exited,
        new Promise<void>(resolve => setTimeout(resolve, Math.min(waitMs, MAX_CHECK_WAIT_MS))),
      ])
    }
    const exitCode = normalizedExit(entry.process.exitCode)
    const running = exitCode === null
    // A process that has exited may still have output in flight in the pipe;
    // without this the final lines of a completed command are reported as empty.
    if (!running) {
      await Promise.race([
        Promise.all(entry.drains.map(drain => drain.done)),
        new Promise<void>(resolve => setTimeout(resolve, 50)),
      ])
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
   * gone regardless passes SIGKILL. Reported honestly: `signalled` is false when
   * the process had already exited, because claiming to have killed something
   * that was already dead would misrepresent what happened.
   */
  async kill(procId: string, signal: 'SIGKILL' | 'SIGTERM' = 'SIGTERM'): Promise<{
    readonly exitCode: number | null
    readonly procId: string
    readonly signalled: boolean
  }> {
    const entry = this.require(procId)
    const signalled = this.registry.signal(procId, signal)
    if (signalled) {
      await Promise.race([
        entry.process.exited,
        new Promise<void>(resolve => setTimeout(resolve, 2_000)),
      ])
    }
    this.release(procId)
    return { procId, signalled, exitCode: normalizedExit(entry.process.exitCode) }
  }

  /** Every tracked process, running or exited but not yet reaped. */
  list(): readonly ProcessRecord[] {
    return this.registry.list().filter(record => this.entries.has(record.procId))
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
        await this.kill(procId, 'SIGKILL')
      } catch {
        // Already gone; teardown must not fail on a race with natural exit.
      }
    }))
  }

  private require(procId: string): BackgroundEntry {
    const entry = this.entries.get(procId)
    if (entry === undefined) {
      throw new ValidationError('proc_id', 'is not a known background command', procId)
    }
    return entry
  }

  private release(procId: string): void {
    const entry = this.entries.get(procId)
    for (const drain of entry?.drains ?? []) drain.cancel()
    this.entries.delete(procId)
    this.registry.remove(procId)
  }
}

function normalizedExit(value: number | null): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}
