// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync } from 'node:fs'
import { stat } from 'node:fs/promises'
import { basename, resolve } from 'node:path'

import { ValidationError } from '../core/errors.js'
import {
  CTRL_C,
  defaultInteractiveShell,
  interruptViaTerminalWrite,
  isWindows,
  shellCommandArgv,
} from '../core/hostPlatform.js'
import type { TerminalHandle, TerminalRegistry } from '../runtime/terminalRegistry.js'
import { WorkspacePathResolver } from '../tools/pathSafety.js'

/**
 * TIOCSCTTY per platform. Bun's PTY support starts the child on the terminal
 * but never makes it the *controlling* terminal, so the kernel has no
 * foreground process group to signal: a resize never reached the shell as
 * SIGWINCH (zsh kept drawing for the old width) and Ctrl-C never became
 * SIGINT (a running command could not be interrupted).
 */
const TIOCSCTTY: Partial<Record<NodeJS.Platform, number>> = {
  darwin: 0x20007461,
  freebsd: 0x20007461,
  openbsd: 0x20007461,
  linux: 0x540e,
}

let perlPath: string | null | undefined

/** A perl that can claim the terminal before exec'ing the real program, if one exists. */
function controllingTerminalHelper(): string | null {
  if (perlPath !== undefined) return perlPath
  perlPath = ['/usr/bin/perl', '/bin/perl', '/usr/local/bin/perl', '/opt/homebrew/bin/perl'].find(path => existsSync(path)) ?? null
  return perlPath
}

/**
 * Wrap argv so the child claims the PTY as its controlling terminal
 * (ioctl TIOCSCTTY — it is already a session leader via `detached`) and then
 * execs the real program in place, keeping the same pid. Returns argv
 * unchanged where that is not possible (Windows, no perl, unknown platform).
 */
export function withControllingTerminal(argv: readonly string[], platform: NodeJS.Platform = process.platform): { argv: string[]; claimed: boolean } {
  const request = TIOCSCTTY[platform]
  const perl = platform === 'win32' || request === undefined ? null : controllingTerminalHelper()
  if (!perl || argv.length === 0) return { argv: [...argv], claimed: false }
  const script = `ioctl(STDIN, ${request}, 0); exec { $ARGV[0] } @ARGV or die "xerxes: cannot start $ARGV[0]: $!\\n"`
  return { argv: [perl, '-e', script, ...argv], claimed: true }
}

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
  /** Mirror sessions here so the TUI terminal panel can watch them live. */
  readonly terminals?: TerminalRegistry
  /** Restrict `workdir` to this root, including existing symlinks. */
  readonly workspaceRoot?: string
  readonly activeWorkspaceRoot?: () => string | undefined
}

export interface CreatePtySessionOptions {
  readonly cols?: number
  readonly env?: Readonly<Record<string, string | undefined>>
  /** Display label in the terminals list; defaults to the command's first words. */
  readonly label?: string
  readonly ownerSessionId?: string
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
  readonly owner?: string
  readonly decoder: TextDecoder
  readonly id: string
  readonly mirror?: TerminalHandle
  readonly output: OutputBuffer
  readonly process: Bun.Subprocess
  readonly terminal: Bun.Terminal
  /** Whether the child owns its terminal; if not, resizes are signalled by hand. */
  readonly ownsTerminal: boolean
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
  private readonly terminals: TerminalRegistry | undefined

  constructor(options: PtySessionManagerOptions = {}) {
    this.maxPendingOutputChars = requirePositiveInteger(
      options.maxPendingOutputChars ?? DEFAULT_MAX_PENDING_OUTPUT_CHARS,
      'maxPendingOutputChars',
    )
    this.paths = options.workspaceRoot === undefined ? undefined : new WorkspacePathResolver(options.workspaceRoot, options.activeWorkspaceRoot)
    this.terminals = options.terminals
  }

  async createSession(command: string, options: CreatePtySessionOptions = {}): Promise<PtyOutput> {
    const workdir = await this.resolveWorkdir(options.workdir)
    const shell = options.shell ?? defaultInteractiveShell()
    const wrapped = withControllingTerminal(shellCommandArgv(shell, command, options.login ?? true))
    const args = wrapped.argv
    const id = `pty_${crypto.randomUUID().replaceAll('-', '').slice(0, 10)}`
    const output = new OutputBuffer(this.maxPendingOutputChars)
    const waiters = new Set<() => void>()
    const decoder = new TextDecoder()
    // Opened before the terminal so the `data` callback can mirror from the
    // very first byte. The controls resolve the session by id at call time,
    // which is what lets them exist before the session does.
    const mirror = options.ownerSessionId === undefined ? undefined : this.terminals?.open({
      id,
      kind: 'pty',
      ownerSessionId: options.ownerSessionId,
      // A bare interactive shell has no command line; name it after the shell
      // (run history refuses an empty title).
      command: command.trim() || basename(shell),
      ...(options.label ? { label: options.label } : {}),
      cwd: workdir,
      control: {
        write: async chars => {
          this.requireSession(id).terminal.write(chars)
        },
        interrupt: async () => {
          this.sendInterrupt(this.requireSession(id))
        },
        kill: async () => void (await this.close(id)),
      },
    })
    let terminal: Bun.Terminal
    try {
      terminal = new Bun.Terminal({
      cols: options.cols ?? 80,
      rows: options.rows ?? 24,
      data: (_terminal, bytes) => {
        const text = decoder.decode(bytes, { stream: true })
        output.append(text)
        mirror?.append(text)
        resolveWaiters(waiters)
      },
      exit: () => resolveWaiters(waiters),
      })
    } catch (error) {
      mirror?.close(null)
      if (error instanceof Error && error.message.includes('PTY not supported')) {
        throw new Error('Interactive PTY sessions are unavailable in this Bun runtime on this platform. Use a non-interactive command, or run Xerxes in WSL2 or on a supported Linux/macOS host.')
      }
      throw error
    }
    let childProcess: Bun.Subprocess
    try {
      childProcess = Bun.spawn(args, {
        cwd: workdir,
        // `detached` is POSIX session leadership (setsid), which a PTY child
        // needs to own its controlling terminal. On Windows it maps to
        // DETACHED_PROCESS — "no console" — which contradicts the pseudoconsole
        // the terminal option attaches, and the child dies instantly with exit
        // code 1 and no output. ConPTY already isolates the session there.
        detached: !isWindows(),
        env: { ...process.env, ...options.env },
        terminal,
      })
    } catch (error) {
      terminal.close()
      mirror?.close(null)
      throw error
    }
    const session: PtySession = {
      id,
      command,
      ...(options.ownerSessionId === undefined ? {} : { owner: options.ownerSessionId }),
      workdir,
      process: childProcess,
      terminal,
      ownsTerminal: wrapped.claimed,
      output,
      waiters,
      decoder,
      ...(mirror ? { mirror } : {}),
    }
    this.sessions.set(id, session)
    void childProcess.exited.then(code => {
      resolveWaiters(waiters)
      mirror?.close(typeof code === 'number' ? code : null)
    })
    return this.read(session, options.yieldTimeMs, options.maxOutputChars)
  }

  async write(sessionId: string, options: WritePtySessionOptions = {}): Promise<PtyOutput> {
    const session = this.requireSession(sessionId)
    const yieldTimeMs = options.yieldTimeMs ?? DEFAULT_YIELD_MS
    requireNonnegativeInteger(yieldTimeMs, 'yieldTimeMs')
    if (options.interrupt) this.sendInterrupt(session)
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
      // Hang up first — what closing a terminal window does. Interactive
      // shells ignore SIGTERM (so close() used to sit out the whole grace
      // period) but exit on SIGHUP and pass it on to their jobs.
      if (!isWindows()) {
        session.process.kill('SIGHUP')
        await waitForExit(session.process, 500)
      }
      if (session.process.exitCode === null) {
        session.process.kill('SIGTERM')
        await waitForExit(session.process, 1_500)
      }
      if (session.process.exitCode === null) {
        session.process.kill('SIGKILL')
        await session.process.exited
      }
    }
    if (!session.terminal.closed) session.terminal.close()
    session.mirror?.close(session.process.exitCode)
    this.sessions.delete(sessionId)
    return { sessionId, closed: true, exitCode: session.process.exitCode }
  }

  /**
   * Write only when the caller owns the session. A missing id and a foreign
   * id fail identically — session-id possession is not authorization and must
   * not become an existence oracle.
   */
  /**
   * True when the shell itself holds its terminal's foreground — waiting at a
   * prompt — rather than a command it started (a build, a REPL, `top`). Only
   * answerable when the child owns its terminal; otherwise (Windows, no perl)
   * it is reported busy so nothing interrupts work we cannot see.
   */
  isAtPrompt(sessionId: string): boolean {
    const session = this.sessions.get(sessionId)
    if (!session || session.process.exitCode !== null) return true
    if (!session.ownsTerminal || isWindows()) return false
    const probe = Bun.spawnSync(['ps', '-o', 'tpgid=,pgid=', '-p', String(session.process.pid)], { stdout: 'pipe', stderr: 'ignore' })
    const [foreground, own] = probe.stdout.toString().trim().split(/\s+/).map(Number)
    return foreground !== undefined && Number.isInteger(foreground) && foreground > 0 && foreground === own
  }

  /** Follow the viewer's size; a full-screen program redraws on SIGWINCH. */
  resizeForOwner(owner: string, sessionId: string, cols: number, rows: number): void {
    const session = this.requireOwned(owner, sessionId)
    const clamp = (value: number, max: number): number => Math.max(1, Math.min(max, Math.trunc(value)))
    if (session.terminal.closed) return
    session.terminal.resize(clamp(cols, 1000), clamp(rows, 500))
    // A child that owns its terminal gets SIGWINCH from the kernel. One that
    // could not claim it is told by hand: its process group (= its pid, it is
    // a session leader) covers the shell and whatever it is running.
    if (!session.ownsTerminal && !isWindows() && session.process.exitCode === null) {
      try { process.kill(-session.process.pid, 'SIGWINCH') } catch { /* already gone */ }
    }
  }

  async writeForOwner(owner: string, sessionId: string, options: WritePtySessionOptions = {}): Promise<PtyOutput> {
    return this.write(this.requireOwned(owner, sessionId).id, options)
  }

  /** Close only when the caller owns the session; same no-oracle rule. */
  async closeForOwner(
    owner: string,
    sessionId: string,
  ): Promise<{ readonly closed: true; readonly exitCode: number | null; readonly sessionId: string }> {
    return this.close(this.requireOwned(owner, sessionId).id)
  }

  /** Sessions owned by one Xerxes session, in stable id order. */
  listForOwner(owner: string): PtySessionSummary[] {
    return this.listSessions().filter(summary => this.sessions.get(summary.sessionId)?.owner === owner)
  }

  /** Daemon teardown hook: close every session one owner opened. */
  async disposeOwner(owner: string): Promise<void> {
    await Promise.all(
      [...this.sessions.values()]
        .filter(session => session.owner === owner)
        .map(async session => {
          try {
            await this.close(session.id)
          } catch {
            // Already exited; teardown must not fail on a race with natural exit.
          }
        }),
    )
  }

  /** Daemon teardown hook: close everything still open. */
  async disposeAll(): Promise<void> {
    await this.closeAll()
  }

  listSessions(): PtySessionSummary[] {
    return [...this.sessions.values()]
      .sort((left, right) => left.id.localeCompare(right.id))
      .map(session => this.summary(session))
  }

  async closeAll(): Promise<void> {
    await Promise.all([...this.sessions.keys()].map(sessionId => this.close(sessionId)))
  }

  /**
   * Deliver Ctrl+C to whatever the session is running, without killing it.
   *
   * Windows has no SIGINT delivery to another process: Node/Bun map every
   * signal except 0 onto TerminateProcess, so `kill('SIGINT')` would kill the
   * shell itself and end the session instead of interrupting the command
   * running inside it. Writing the Ctrl+C control character into the terminal
   * is the equivalent that the console driver turns into a real interrupt and
   * that leaves the shell alive for the next call.
   */
  private sendInterrupt(session: PtySession): void {
    if (session.process.exitCode !== null) return
    if (interruptViaTerminalWrite()) {
      session.terminal.write(CTRL_C)
    } else {
      session.process.kill('SIGINT')
    }
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

  private requireOwned(owner: string, sessionId: string): PtySession {
    const session = this.sessions.get(sessionId)
    if (session === undefined || session.owner !== owner) {
      throw new ValidationError('session_id', 'PTY session not found', sessionId)
    }
    return session
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
