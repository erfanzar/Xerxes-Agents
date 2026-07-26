// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { stat } from 'node:fs/promises'
import { resolve } from 'node:path'

import { exitShellInput, interruptTerminalInput, resolveDefaultShell, shellInvocation } from '../core/shell.js'
import { ValidationError } from '../core/errors.js'
import { WorkspacePathResolver } from '../tools/pathSafety.js'
import type { IPty } from 'bun-pty'

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
  /** Platform override for tests; defaults to `process.platform`. */
  readonly platform?: NodeJS.Platform
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
  readonly driver: PtyDriver
  readonly id: string
  readonly output: OutputBuffer
  readonly waiters: Set<() => void>
  readonly workdir: string
}

interface PtyDriverOptions {
  readonly cwd: string
  readonly env: Record<string, string>
  readonly cols: number
  readonly rows: number
  readonly onData: (text: string) => void
  readonly onExit: () => void
}

/**
 * The slice of a pseudo-terminal the session manager needs. POSIX sessions run
 * on `Bun.Terminal`; native Windows sessions run on ConPTY through `bun-pty`,
 * because `Bun.Terminal` has no Windows support (oven-sh/bun#25565).
 */
interface PtyDriver {
  readonly exitCode: number | null
  readonly exited: Promise<number>
  write(data: string): void
  interrupt(): void
  terminate(): void
  forceKill(): void
  sendEof(): void
  closeTerminal(): void
}

/** POSIX driver: current `Bun.Terminal` + `Bun.spawn` behavior, unchanged. */
class BunTerminalPtyDriver implements PtyDriver {
  private readonly decoder = new TextDecoder()
  private readonly process: Bun.Subprocess
  private readonly terminal: Bun.Terminal

  constructor(args: string[], options: PtyDriverOptions) {
    this.terminal = new Bun.Terminal({
      cols: options.cols,
      rows: options.rows,
      data: (_terminal, bytes) => {
        options.onData(this.decoder.decode(bytes, { stream: true }))
      },
      exit: () => options.onExit(),
    })
    try {
      this.process = Bun.spawn(args, {
        cwd: options.cwd,
        detached: true,
        env: options.env,
        terminal: this.terminal,
      })
    } catch (error) {
      this.terminal.close()
      throw error
    }
  }

  get exitCode(): number | null {
    return this.process.exitCode
  }

  get exited(): Promise<number> {
    return this.process.exited
  }

  write(data: string): void {
    this.terminal.write(data)
  }

  interrupt(): void {
    if (this.process.exitCode === null) this.process.kill('SIGINT')
  }

  terminate(): void {
    if (this.process.exitCode === null) this.process.kill('SIGTERM')
  }

  forceKill(): void {
    if (this.process.exitCode === null) this.process.kill('SIGKILL')
  }

  sendEof(): void {
    this.terminal.write(exitShellInput('linux'))
  }

  closeTerminal(): void {
    if (!this.terminal.closed) this.terminal.close()
  }
}

/** Native Windows driver: ConPTY via the `bun-pty` (Rust portable-pty) FFI. */
class ConPtyDriver implements PtyDriver {
  private exitCodeValue: number | null = null
  private readonly pty: IPty
  readonly exited: Promise<number>

  constructor(spawnPty: typeof import('bun-pty').spawn, args: string[], options: PtyDriverOptions) {
    const file = args[0]
    if (!file) throw new ValidationError('command', 'PTY argv must not be empty', args.join(' '))
    this.pty = spawnPty(file, args.slice(1), {
      name: 'xterm-256color',
      cols: options.cols,
      rows: options.rows,
      cwd: options.cwd,
      env: options.env,
    })
    this.pty.onData((data) => options.onData(data))
    this.exited = new Promise<number>((resolveExit) => {
      this.pty.onExit(({ exitCode }) => {
        this.exitCodeValue = exitCode
        options.onExit()
        resolveExit(exitCode)
      })
    })
  }

  get exitCode(): number | null {
    return this.exitCodeValue
  }

  write(data: string): void {
    this.pty.write(data)
  }

  interrupt(): void {
    // ConPTY delivers the Ctrl+C control character to the console's foreground
    // process group; Windows has no cross-process SIGINT.
    if (this.exitCodeValue === null) this.pty.write(interruptTerminalInput('win32'))
  }

  terminate(): void {
    if (this.exitCodeValue === null) this.pty.kill('SIGTERM')
  }

  forceKill(): void {
    if (this.exitCodeValue === null) this.pty.kill('SIGKILL')
  }

  sendEof(): void {
    this.write(exitShellInput('win32'))
  }

  closeTerminal(): void {
    // Killing the child tears the ConPTY session down with it.
    if (this.exitCodeValue === null) this.pty.kill()
  }
}

async function createPtyDriver(
  platform: NodeJS.Platform,
  args: string[],
  options: PtyDriverOptions,
): Promise<PtyDriver> {
  if (platform === 'win32') {
    // Lazy: bun-pty loads its native Rust library through Bun FFI, which is a
    // genuine startup cost and only resolvable where prebuilt binaries exist.
    const { spawn: spawnPty } = await import('bun-pty')
    return new ConPtyDriver(spawnPty, args, options)
  }
  return new BunTerminalPtyDriver(args, options)
}

/**
 * Owns persistent, interactive PTYs scoped to one Xerxes session.
 *
 * The terminal driver consumes output immediately, so this manager retains
 * unread output itself. A capped response does not throw away the remainder;
 * the next `write` call can drain it.
 */
export class PtySessionManager {
  private readonly maxPendingOutputChars: number
  private readonly paths: WorkspacePathResolver | undefined
  private readonly platform: NodeJS.Platform
  private readonly sessions = new Map<string, PtySession>()

  constructor(options: PtySessionManagerOptions = {}) {
    this.maxPendingOutputChars = requirePositiveInteger(
      options.maxPendingOutputChars ?? DEFAULT_MAX_PENDING_OUTPUT_CHARS,
      'maxPendingOutputChars',
    )
    this.paths = options.workspaceRoot === undefined ? undefined : new WorkspacePathResolver(options.workspaceRoot)
    this.platform = options.platform ?? process.platform
  }

  async createSession(command: string, options: CreatePtySessionOptions = {}): Promise<PtyOutput> {
    const workdir = await this.resolveWorkdir(options.workdir)
    const shell = options.shell ?? resolveDefaultShell(process.env, this.platform)
    const argv = shellInvocation(shell, command, options.login ?? true, this.platform)
    const id = `pty_${crypto.randomUUID().replaceAll('-', '').slice(0, 10)}`
    const output = new OutputBuffer(this.maxPendingOutputChars)
    const waiters = new Set<() => void>()
    const env: Record<string, string> = {}
    for (const [key, value] of Object.entries(process.env)) {
      if (value !== undefined) env[key] = value
    }
    for (const [key, value] of Object.entries(options.env ?? {})) {
      if (value === undefined) delete env[key]
      else env[key] = value
    }
    const driver = await createPtyDriver(this.platform, argv, {
      cwd: workdir,
      env,
      cols: options.cols ?? 80,
      rows: options.rows ?? 24,
      onData: (text) => {
        output.append(text)
        resolveWaiters(waiters)
      },
      onExit: () => resolveWaiters(waiters),
    })
    const session: PtySession = { id, command, workdir, driver, output, waiters }
    this.sessions.set(id, session)
    void driver.exited.then(() => resolveWaiters(waiters))
    return this.read(session, options.yieldTimeMs, options.maxOutputChars)
  }

  async write(sessionId: string, options: WritePtySessionOptions = {}): Promise<PtyOutput> {
    const session = this.requireSession(sessionId)
    const yieldTimeMs = options.yieldTimeMs ?? DEFAULT_YIELD_MS
    requireNonnegativeInteger(yieldTimeMs, 'yieldTimeMs')
    if (options.interrupt && session.driver.exitCode === null) {
      session.driver.interrupt()
    }
    if (options.chars) session.driver.write(options.chars)
    if (options.closeStdin) {
      session.driver.sendEof()
      // A terminal echoes the typed input before the child reacts to EOF. Give
      // a short-lived command the requested window to flush its final output
      // so one write_stdin call observes the complete request/response pair.
      if (session.driver.exitCode === null && yieldTimeMs > 0) {
        await waitForExit(session.driver, yieldTimeMs)
      }
    }
    return this.read(session, options.yieldTimeMs, options.maxOutputChars)
  }

  async close(sessionId: string): Promise<{ readonly closed: true; readonly exitCode: number | null; readonly sessionId: string }> {
    const session = this.requireSession(sessionId)
    if (session.driver.exitCode === null) {
      session.driver.terminate()
      await waitForExit(session.driver, 2_000)
      if (session.driver.exitCode === null) {
        session.driver.forceKill()
        await session.driver.exited
      }
    }
    session.driver.closeTerminal()
    this.sessions.delete(sessionId)
    return { sessionId, closed: true, exitCode: session.driver.exitCode }
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
    if (!session.output.hasData() && session.driver.exitCode === null && normalizedYield > 0) {
      await waitForSessionActivity(session, normalizedYield)
    }
    // A short settle period lets a one-shot shell command reach its exit event
    // after its first stdout chunk, while long-running sessions still return
    // promptly with their initial output and a pollable running state.
    if (session.output.hasData() && session.driver.exitCode === null && normalizedYield > 0) {
      await waitForExit(session.driver, Math.min(normalizedYield, OUTPUT_SETTLE_MS))
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
      running: session.driver.exitCode === null,
      exitCode: session.driver.exitCode,
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
    if (session.output.hasData() || session.driver.exitCode !== null) wake()
  })
}

async function waitForExit(driver: PtyDriver, timeoutMs: number): Promise<void> {
  await Promise.race([
    driver.exited.then(() => undefined),
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
