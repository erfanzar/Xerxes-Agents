// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { stat } from 'node:fs/promises'
import { basename, resolve } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { WorkspacePathResolver } from '../tools/pathSafety.js'

const DEFAULT_MAX_PENDING_OUTPUT_CHARS = 1_000_000
const DEFAULT_MAX_OUTPUT_CHARS = 4_000
const DEFAULT_YIELD_MS = 1_000
const OUTPUT_SETTLE_MS = 50

export interface PtySessionSummary {
  readonly command: string
  readonly exitCode: number | null
  readonly running: boolean
  readonly sessionId: string
  readonly workdir: string
}

export interface PtyOutput extends PtySessionSummary {
  readonly maxOutputChars: number
  readonly outputTruncated: boolean
  readonly stdout: string
  readonly yieldTimeMs: number
  readonly note?: string
}

export interface PtySessionManagerOptions {
  /** Bounds unresolved output retained in memory for one terminal session. */
  readonly maxPendingOutputChars?: number
  /** Restrict `workdir` to this root, including existing symlinks. */
  readonly workspaceRoot?: string
}

export interface CreatePtySessionOptions {
  readonly cols?: number
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly login?: boolean
  readonly maxOutputChars?: number
  readonly rows?: number
  readonly shell?: string
  readonly workdir?: string
  readonly yieldTimeMs?: number
}

export interface WritePtySessionOptions {
  readonly chars?: string
  readonly closeStdin?: boolean
  readonly interrupt?: boolean
  readonly maxOutputChars?: number
  readonly yieldTimeMs?: number
}

interface PtySession {
  readonly command: string
  readonly decoder: TextDecoder
  readonly id: string
  readonly output: OutputBuffer
  readonly process: Bun.Subprocess
  readonly terminal: Bun.Terminal
  readonly waiters: Set<() => void>
  readonly workdir: string
}

/**
 * Owns persistent, interactive Bun PTYs scoped to one Xerxes session.
 *
 * Bun's terminal callback consumes output immediately, so this manager retains
 * unread output itself. A capped response does not throw away the remainder;
 * the next `write` call can drain it.
 */
export class PtySessionManager {
  private readonly maxPendingOutputChars: number
  private readonly paths: WorkspacePathResolver | undefined
  private readonly sessions = new Map<string, PtySession>()

  constructor(options: PtySessionManagerOptions = {}) {
    this.maxPendingOutputChars = requirePositiveInteger(
      options.maxPendingOutputChars ?? DEFAULT_MAX_PENDING_OUTPUT_CHARS,
      'maxPendingOutputChars',
    )
    this.paths = options.workspaceRoot === undefined ? undefined : new WorkspacePathResolver(options.workspaceRoot)
  }

  async createSession(command: string, options: CreatePtySessionOptions = {}): Promise<PtyOutput> {
    const workdir = await this.resolveWorkdir(options.workdir)
    const shell = options.shell ?? process.env.SHELL ?? '/bin/sh'
    const args = shellArguments(shell, command, options.login ?? true)
    const id = `pty_${crypto.randomUUID().replaceAll('-', '').slice(0, 10)}`
    const output = new OutputBuffer(this.maxPendingOutputChars)
    const waiters = new Set<() => void>()
    const decoder = new TextDecoder()
    const terminal = new Bun.Terminal({
      cols: options.cols ?? 80,
      rows: options.rows ?? 24,
      data: (_terminal, bytes) => {
        output.append(decoder.decode(bytes, { stream: true }))
        resolveWaiters(waiters)
      },
      exit: () => resolveWaiters(waiters),
    })
    let childProcess: Bun.Subprocess
    try {
      childProcess = Bun.spawn(args, {
        cwd: workdir,
        detached: true,
        env: { ...process.env, ...options.env },
        terminal,
      })
    } catch (error) {
      terminal.close()
      throw error
    }
    const session: PtySession = { id, command, workdir, process: childProcess, terminal, output, waiters, decoder }
    this.sessions.set(id, session)
    void childProcess.exited.then(() => resolveWaiters(waiters))
    return this.read(session, options.yieldTimeMs, options.maxOutputChars)
  }

  async write(sessionId: string, options: WritePtySessionOptions = {}): Promise<PtyOutput> {
    const session = this.requireSession(sessionId)
    const yieldTimeMs = options.yieldTimeMs ?? DEFAULT_YIELD_MS
    requireNonnegativeInteger(yieldTimeMs, 'yieldTimeMs')
    if (options.interrupt && session.process.exitCode === null) {
      session.process.kill('SIGINT')
    }
    if (options.chars) session.terminal.write(options.chars)
    if (options.closeStdin) {
      session.terminal.write('\u0004')
      // A terminal echoes the typed input before the child reacts to EOF. Give
      // a short-lived command the requested window to flush its final output
      // so one write_stdin call observes the complete request/response pair.
      if (session.process.exitCode === null && yieldTimeMs > 0) {
        await waitForExit(session.process, yieldTimeMs)
      }
    }
    return this.read(session, options.yieldTimeMs, options.maxOutputChars)
  }

  async close(sessionId: string): Promise<{ readonly closed: true; readonly exitCode: number | null; readonly sessionId: string }> {
    const session = this.requireSession(sessionId)
    if (session.process.exitCode === null) {
      session.process.kill('SIGTERM')
      await waitForExit(session.process, 2_000)
      if (session.process.exitCode === null) {
        session.process.kill('SIGKILL')
        await session.process.exited
      }
    }
    if (!session.terminal.closed) session.terminal.close()
    this.sessions.delete(sessionId)
    return { sessionId, closed: true, exitCode: session.process.exitCode }
  }

  listSessions(): PtySessionSummary[] {
    return [...this.sessions.values()]
      .sort((left, right) => left.id.localeCompare(right.id))
      .map(session => this.summary(session))
  }

  async closeAll(): Promise<void> {
    await Promise.all([...this.sessions.keys()].map(sessionId => this.close(sessionId)))
  }

  private async read(
    session: PtySession,
    yieldTimeMs = DEFAULT_YIELD_MS,
    maxOutputChars = DEFAULT_MAX_OUTPUT_CHARS,
  ): Promise<PtyOutput> {
    const normalizedYield = requireNonnegativeInteger(yieldTimeMs, 'yieldTimeMs')
    const normalizedMax = requireNonnegativeInteger(maxOutputChars, 'maxOutputChars')
    if (!session.output.hasData() && session.process.exitCode === null && normalizedYield > 0) {
      await waitForSessionActivity(session, normalizedYield)
    }
    // A short settle period lets a one-shot shell command reach its exit event
    // after its first stdout chunk, while long-running sessions still return
    // promptly with their initial output and a pollable running state.
    if (session.output.hasData() && session.process.exitCode === null && normalizedYield > 0) {
      await waitForExit(session.process, Math.min(normalizedYield, OUTPUT_SETTLE_MS))
    }
    const drained = session.output.take(normalizedMax)
    const summary = this.summary(session)
    return Object.freeze({
      ...summary,
      stdout: drained.text,
      outputTruncated: drained.truncated,
      yieldTimeMs: normalizedYield,
      maxOutputChars: normalizedMax,
      ...(summary.running ? { note: `Process is still running; poll with write_stdin(session_id='${session.id}', chars='').` } : {}),
    })
  }

  private summary(session: PtySession): PtySessionSummary {
    return Object.freeze({
      sessionId: session.id,
      command: session.command,
      workdir: session.workdir,
      running: session.process.exitCode === null,
      exitCode: session.process.exitCode,
    })
  }

  private requireSession(sessionId: string): PtySession {
    const session = this.sessions.get(sessionId)
    if (session === undefined) throw new ValidationError('session_id', 'PTY session not found', sessionId)
    return session
  }

  private async resolveWorkdir(candidate: string | undefined): Promise<string> {
    const workdir = candidate?.trim() || '.'
    const resolved = this.paths === undefined ? resolve(workdir) : await this.paths.resolve(workdir)
    let metadata
    try {
      metadata = await stat(resolved)
    } catch (error) {
      throw new ValidationError('workdir', 'must refer to an existing directory', workdir, { cause: errorMessage(error) })
    }
    if (!metadata.isDirectory()) {
      throw new ValidationError('workdir', 'must refer to an existing directory', workdir)
    }
    return resolved
  }
}

class OutputBuffer {
  private dropped = false
  private readonly chunks: string[] = []
  private length = 0

  constructor(private readonly limit: number) {}

  append(value: string): void {
    if (!value) return
    this.chunks.push(value)
    this.length += value.length
    while (this.length > this.limit && this.chunks.length) {
      const first = this.chunks[0]
      if (first === undefined) break
      const excess = this.length - this.limit
      if (first.length <= excess) {
        this.chunks.shift()
        this.length -= first.length
      } else {
        this.chunks[0] = first.slice(excess)
        this.length -= excess
      }
      this.dropped = true
    }
  }

  hasData(): boolean {
    return this.length > 0 || this.dropped
  }

  take(maxChars: number): { readonly text: string; readonly truncated: boolean } {
    const prefix = this.dropped ? '[Earlier terminal output was discarded due to the session output limit.]\n' : ''
    this.dropped = false
    if (maxChars === 0) {
      return { text: prefix, truncated: this.length > 0 }
    }
    const budget = Math.max(maxChars - prefix.length, 0)
    let remaining = budget
    const values: string[] = [prefix]
    while (remaining > 0 && this.chunks.length) {
      const current = this.chunks[0]
      if (current === undefined) break
      if (current.length <= remaining) {
        this.chunks.shift()
        this.length -= current.length
        values.push(current)
        remaining -= current.length
      } else {
        values.push(current.slice(0, remaining))
        this.chunks[0] = current.slice(remaining)
        this.length -= remaining
        remaining = 0
      }
    }
    return { text: values.join(''), truncated: this.length > 0 }
  }
}

function shellArguments(shell: string, command: string, login: boolean): string[] {
  const name = basename(shell)
  const supportsLogin = login && (name.endsWith('bash') || name.endsWith('zsh'))
  if (!command.trim()) return [shell, ...(supportsLogin ? ['-l'] : [])]
  return [shell, ...(supportsLogin ? ['-l'] : []), '-c', command]
}

function resolveWaiters(waiters: Set<() => void>): void {
  for (const resolve of waiters) resolve()
  waiters.clear()
}

function waitForSessionActivity(session: PtySession, timeoutMs: number): Promise<void> {
  return new Promise(resolve => {
    const timer = setTimeout(() => {
      session.waiters.delete(wake)
      resolve()
    }, timeoutMs)
    const wake = () => {
      clearTimeout(timer)
      resolve()
    }
    session.waiters.add(wake)
    if (session.output.hasData() || session.process.exitCode !== null) wake()
  })
}

async function waitForExit(process: Bun.Subprocess, timeoutMs: number): Promise<void> {
  await Promise.race([
    process.exited.then(() => undefined),
    new Promise<void>(resolve => setTimeout(resolve, timeoutMs)),
  ])
}

function requirePositiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) throw new ValidationError(name, 'must be a positive integer', value)
  return value
}

function requireNonnegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) throw new ValidationError(name, 'must be a non-negative integer', value)
  return value
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
