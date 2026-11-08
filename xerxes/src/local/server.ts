// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Supervisor for a harness-managed engine process: llama.cpp's `llama-server`
 * or mlx-lm's `mlx_lm.server`. Both serve an OpenAI-compatible API on
 * 127.0.0.1 and answer `/health`. Nothing here runs until the user calls
 * `xerxes local start` / `setup`. The child is detached from
 * the daemon lifecycle, logs to a file, and is recorded in a state file so a
 * later `stop` / `status` can find it. Every privileged action (spawn, signals,
 * process inspection, network) goes through `ServerPorts`, so tests use fakes.
 *
 * Safety rule: a pid is only ever signalled after its command line is checked
 * against the recorded binary and model, so a recycled pid is never killed.
 */
import { spawn as spawnChild } from 'node:child_process'
import { closeSync, openSync } from 'node:fs'
import { mkdir, open, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:net'
import { basename, dirname, join } from 'node:path'

import { XerxesError } from '../core/errors.js'
import { xerxesHome } from '../daemon/paths.js'
import type { EngineRecovery, EngineTarget } from './engineRecovery.js'

export const DEFAULT_HEALTH_TIMEOUT_MS = 180_000
const VERSION_TIMEOUT_MS = 10_000
const DEFAULT_POLL_MS = 500
const DEFAULT_GRACE_MS = 10_000
const LOG_TAIL_BYTES = 4096
const PS_TIMEOUT_MS = 5000

export type ServerErrorCode =
  | 'BINARY_NOT_FOUND'
  | 'PORT_IN_USE'
  | 'ALREADY_RUNNING'
  | 'START_FAILED'
  | 'START_TIMEOUT'
  | 'LOAD_FAILED'
  | 'STOP_FAILED'

export class ServerError extends XerxesError {
  readonly code: ServerErrorCode

  constructor(code: ServerErrorCode, message: string, details: Record<string, unknown> = {}) {
    super(message, { code, ...details })
    this.code = code
  }
}

export interface SpawnedProcess {
  readonly pid: number
  /** Resolves with the exit code when the child exits (never rejects). */
  readonly exited: Promise<number>
}

export interface ServerPorts {
  which(name: string): string | undefined
  exists(path: string): Promise<boolean>
  /** Start a detached child writing stdout+stderr to `logPath`. Throws if it cannot start. */
  spawn(argv: string[], options: { readonly logPath: string }): SpawnedProcess
  fetch(url: string, init: { readonly signal?: AbortSignal }): Promise<{ readonly status: number }>
  isPidAlive(pid: number): boolean
  /** Full command line of a pid, or undefined when it cannot be read. */
  commandOf(pid: number): Promise<string | undefined>
  signal(pid: number, signal: 'SIGTERM' | 'SIGKILL'): void
  portInUse(port: number): Promise<boolean>
  /** Run a short command and return its stdout and stderr together; rejects when it cannot run. */
  run(argv: readonly string[], timeoutMs: number): Promise<{ readonly exitCode: number; readonly output: string }>
  sleep(ms: number): Promise<void>
  now(): number
}

/** The engine processes this supervisor can run. */
export type ManagedEngine = 'llama-server' | 'mlx'

export interface LocalServerState {
  readonly pid: number
  readonly port: number
  readonly model: string
  readonly startedAt: string
  readonly logPath: string
  /** Resolved engine binary path; used to verify the pid before signalling. */
  readonly bin: string
  /**
   * The window it was started with (llama-server's --ctx-size; for mlx-lm, the
   * window recorded for the profile). Absent in older state files.
   */
  readonly ctxSize?: number
}

export interface ServerStatus {
  readonly running: boolean
  readonly healthy: boolean
  /** A state file exists but its process is gone or is not our engine. */
  readonly stale: boolean
  readonly pid?: number
  readonly port?: number
  readonly model?: string
  readonly startedAt?: string
  readonly logPath?: string
  /** The window the engine reports it loaded (LM Studio's `loaded_context_length`). */
  readonly contextLength?: number
}

export interface StartOptions {
  /** llama-server: the .gguf file. mlx: the model folder. */
  readonly modelPath: string
  readonly port: number
  readonly ctxSize: number
  readonly timeoutMs?: number
}

export type StopResult =
  | { readonly stopped: true; readonly pid: number; readonly forced: boolean }
  | { readonly stopped: false; readonly reason: 'not-running' | 'stale' | 'foreign-process'; readonly pid?: number }

export interface LocalServer {
  readonly engine: ManagedEngine
  findBinary(): Promise<string | undefined>
  /**
   * What the engine says its version is (`llama-server --version`, or
   * `mlx_lm --version` next to `mlx_lm.server`), or undefined when it cannot be read.
   */
  version(): Promise<string | undefined>
  start(options: StartOptions): Promise<ServerStatus>
  stop(): Promise<StopResult>
  status(): Promise<ServerStatus>
  /**
   * Restart the engine Xerxes started when it has exited, with the same
   * model, port and window. One that is still alive (even if busy or
   * unhealthy), one on another port, and one whose window was never recorded
   * are left alone. Rejects when the restart fails.
   */
  recover(target: EngineTarget): Promise<EngineRecovery>
}

export interface LocalServerOptions {
  /** Default llama-server. */
  readonly engine?: ManagedEngine
  /** The engine binary; found on PATH when absent. */
  readonly binaryPath?: string
  readonly stateDir?: string
  readonly pollMs?: number
  readonly graceMs?: number
}

export const BINARY_INSTALL_HELP =
  'llama-server (llama.cpp) was not found. Install it with `brew install llama.cpp` (macOS/Linux) or from ' +
  'https://github.com/ggml-org/llama.cpp/releases, then put `llama-server` on PATH or set ' +
  '`runtime.local.llama_server_path` in ~/.xerxes/daemon/config.json (or XERXES_LOCAL_LLAMA_SERVER).'

export const MLX_INSTALL_HELP =
  'mlx_lm.server (mlx-lm) was not found. Install it on Apple silicon with `uv tool install mlx-lm` ' +
  '(or `pip install mlx-lm`), then put `mlx_lm.server` on PATH or set `runtime.local.mlx_server_path` ' +
  'in ~/.xerxes/daemon/config.json (or XERXES_LOCAL_MLX_SERVER).'

interface EngineSpec {
  readonly label: string
  readonly binary: string
  readonly stateFile: string
  readonly logFile: string
  readonly installHelp: string
  argv(bin: string, start: StartOptions): string[]
  versionArgv(bin: string): string[]
}

const ENGINES: Readonly<Record<ManagedEngine, EngineSpec>> = {
  'llama-server': {
    label: 'llama-server',
    binary: 'llama-server',
    stateFile: 'server.json',
    logFile: 'llama-server.log',
    installHelp: BINARY_INSTALL_HELP,
    argv: (bin, start) => [
      bin, '-m', start.modelPath, '--host', '127.0.0.1', '--port', String(start.port), '--ctx-size', String(start.ctxSize), '--jinja',
    ],
    versionArgv: bin => [bin, '--version'],
  },
  mlx: {
    label: 'mlx_lm.server',
    binary: 'mlx_lm.server',
    stateFile: 'mlx-server.json',
    logFile: 'mlx-server.log',
    installHelp: MLX_INSTALL_HELP,
    // One request at a time, as llama-server and LM Studio run (one slot), and
    // two prompt caches (the task and one helper call) instead of ten: each
    // holds a whole context's cache in the same memory as the weights.
    argv: (bin, start) => [
      bin, '--model', start.modelPath, '--host', '127.0.0.1', '--port', String(start.port),
      '--decode-concurrency', '1', '--prompt-concurrency', '1', '--prompt-cache-size', '2',
    ],
    // mlx_lm.server has no --version; the mlx_lm command installed beside it does.
    versionArgv: bin => [join(dirname(bin), 'mlx_lm'), '--version'],
  },
}

function isState(value: unknown): value is LocalServerState {
  if (typeof value !== 'object' || value === null) return false
  const v = value as Record<string, unknown>
  return (
    typeof v.pid === 'number' &&
    Number.isInteger(v.pid) &&
    v.pid > 0 &&
    typeof v.port === 'number' &&
    typeof v.model === 'string' &&
    typeof v.startedAt === 'string' &&
    typeof v.logPath === 'string' &&
    typeof v.bin === 'string' &&
    (v.ctxSize === undefined || (typeof v.ctxSize === 'number' && Number.isInteger(v.ctxSize) && v.ctxSize > 0))
  )
}

/** Last bytes of a log file; empty string when the log is missing. */
export async function readLogTail(path: string, bytes = LOG_TAIL_BYTES): Promise<string> {
  let handle
  try {
    handle = await open(path, 'r')
  } catch {
    return ''
  }
  try {
    const { size } = await handle.stat()
    const length = Math.min(size, bytes)
    const buffer = Buffer.alloc(length)
    await handle.read(buffer, 0, length, size - length)
    return buffer.toString('utf8').trim()
  } finally {
    await handle.close()
  }
}

export function createLocalServer(ports: ServerPorts, options: LocalServerOptions = {}): LocalServer {
  const engine = options.engine ?? 'llama-server'
  const spec = ENGINES[engine]
  const stateDir = options.stateDir ?? join(xerxesHome(), 'local')
  const statePath = join(stateDir, spec.stateFile)
  const logPath = join(stateDir, spec.logFile)
  const pollMs = options.pollMs ?? DEFAULT_POLL_MS
  const graceMs = options.graceMs ?? DEFAULT_GRACE_MS

  async function findBinary(): Promise<string | undefined> {
    if (options.binaryPath !== undefined) {
      return (await ports.exists(options.binaryPath)) ? options.binaryPath : undefined
    }
    return ports.which(spec.binary)
  }

  async function version(): Promise<string | undefined> {
    const bin = await findBinary()
    if (bin === undefined) return undefined
    const argv = spec.versionArgv(bin)
    if (argv[0] !== bin && !(await ports.exists(argv[0]!))) return undefined
    try {
      const result = await ports.run(argv, VERSION_TIMEOUT_MS)
      const output = result.output.trim()
      return output === '' ? undefined : output
    } catch {
      return undefined
    }
  }

  async function readState(): Promise<LocalServerState | undefined> {
    let text: string
    try {
      text = await readFile(statePath, 'utf8')
    } catch {
      return undefined
    }
    try {
      const parsed: unknown = JSON.parse(text)
      return isState(parsed) ? parsed : undefined
    } catch {
      return undefined
    }
  }

  async function writeState(state: LocalServerState): Promise<void> {
    await mkdir(stateDir, { recursive: true, mode: 0o700 })
    await writeFile(statePath, JSON.stringify(state, null, 2) + '\n', { mode: 0o600 })
  }

  const removeState = (): Promise<void> => rm(statePath, { force: true })

  /** `running` only when the pid is alive AND its command line is our engine on our model. */
  async function ownership(state: LocalServerState): Promise<'running' | 'dead' | 'foreign'> {
    if (!ports.isPidAlive(state.pid)) return 'dead'
    const command = await ports.commandOf(state.pid)
    if (command !== undefined && command.includes(basename(state.bin)) && command.includes(state.model)) {
      return 'running'
    }
    return 'foreign'
  }

  async function healthy(port: number): Promise<boolean> {
    try {
      const response = await ports.fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(2000) })
      return response.status === 200
    } catch {
      return false
    }
  }

  async function waitForExit(pid: number, withinMs: number): Promise<boolean> {
    const deadline = ports.now() + withinMs
    while (ports.isPidAlive(pid)) {
      if (ports.now() >= deadline) return false
      await ports.sleep(Math.min(pollMs, 200))
    }
    return true
  }

  /** Terminate a child this supervisor itself just spawned (no ownership check needed). */
  async function terminateOwnChild(pid: number): Promise<void> {
    try {
      ports.signal(pid, 'SIGTERM')
    } catch {
      return
    }
    if (await waitForExit(pid, graceMs)) return
    try {
      ports.signal(pid, 'SIGKILL')
    } catch {
      // already gone
    }
  }

  async function status(): Promise<ServerStatus> {
    const state = await readState()
    if (state === undefined) return { running: false, healthy: false, stale: false }
    const own = await ownership(state)
    if (own !== 'running') {
      return { running: false, healthy: false, stale: true, pid: state.pid, port: state.port, model: state.model }
    }
    return {
      running: true,
      healthy: await healthy(state.port),
      stale: false,
      pid: state.pid,
      port: state.port,
      model: state.model,
      startedAt: state.startedAt,
      logPath: state.logPath,
    }
  }

  async function start(start: StartOptions): Promise<ServerStatus> {
    const existing = await readState()
    if (existing !== undefined) {
      if ((await ownership(existing)) === 'running') {
        throw new ServerError(
          'ALREADY_RUNNING',
          `A local server is already running (pid ${existing.pid}, port ${existing.port}). Run \`xerxes local stop\` first.`,
          { pid: existing.pid, port: existing.port },
        )
      }
      await removeState()
    }

    const bin = await findBinary()
    if (bin === undefined) {
      const configured = options.binaryPath
      throw new ServerError(
        'BINARY_NOT_FOUND',
        configured === undefined ? spec.installHelp : `${spec.label} not found at the configured path ${configured}. ${spec.installHelp}`,
      )
    }
    if (await ports.portInUse(start.port)) {
      throw new ServerError(
        'PORT_IN_USE',
        `Port ${start.port} is already in use. Stop whatever is listening on it or choose another with \`runtime.local.port\` (or XERXES_LOCAL_PORT).`,
        { port: start.port },
      )
    }

    await mkdir(stateDir, { recursive: true, mode: 0o700 })
    await rm(logPath, { force: true })
    const argv = spec.argv(bin, start)
    let child: SpawnedProcess
    try {
      child = ports.spawn(argv, { logPath })
    } catch (error) {
      throw new ServerError(
        'START_FAILED',
        `Could not start ${bin}: ${error instanceof Error ? error.message : String(error)}`,
      )
    }
    const startedAt = new Date(ports.now()).toISOString()
    await writeState({ pid: child.pid, port: start.port, model: start.modelPath, startedAt, logPath, bin, ctxSize: start.ctxSize })

    let exitCode: number | undefined
    void child.exited.then((code) => {
      exitCode = code
    })
    const timeoutMs = start.timeoutMs ?? DEFAULT_HEALTH_TIMEOUT_MS
    const deadline = ports.now() + timeoutMs
    for (;;) {
      if (exitCode !== undefined) {
        await removeState()
        const tail = await readLogTail(logPath)
        throw new ServerError(
          'START_FAILED',
          `${spec.label} exited with code ${exitCode} before it became healthy.${tail === '' ? '' : `\nLog tail (${logPath}):\n${tail}`}`,
          { exitCode, logPath },
        )
      }
      if (await healthy(start.port)) break
      if (ports.now() >= deadline) {
        await terminateOwnChild(child.pid)
        await removeState()
        const tail = await readLogTail(logPath)
        throw new ServerError(
          'START_TIMEOUT',
          `${spec.label} did not report healthy on port ${start.port} within ${Math.round(timeoutMs / 1000)}s and was stopped.${tail === '' ? '' : `\nLog tail (${logPath}):\n${tail}`}`,
          { timeoutMs, logPath },
        )
      }
      await ports.sleep(pollMs)
    }
    return {
      running: true,
      healthy: true,
      stale: false,
      pid: child.pid,
      port: start.port,
      model: start.modelPath,
      startedAt,
      logPath,
    }
  }

  async function stop(): Promise<StopResult> {
    const state = await readState()
    if (state === undefined) return { stopped: false, reason: 'not-running' }
    const own = await ownership(state)
    if (own === 'dead') {
      await removeState()
      return { stopped: false, reason: 'stale', pid: state.pid }
    }
    if (own === 'foreign') {
      // The recorded pid now belongs to something else (or cannot be inspected): never signal it.
      await removeState()
      return { stopped: false, reason: 'foreign-process', pid: state.pid }
    }
    ports.signal(state.pid, 'SIGTERM')
    let forced = false
    if (!(await waitForExit(state.pid, graceMs))) {
      if ((await ownership(state)) === 'running') {
        ports.signal(state.pid, 'SIGKILL')
        forced = true
        await waitForExit(state.pid, graceMs)
      }
    }
    await removeState()
    return { stopped: true, pid: state.pid, forced }
  }

  async function recover(target: EngineTarget): Promise<EngineRecovery> {
    const state = await readState()
    if (state === undefined || state.ctxSize === undefined) return { recovered: false, reason: 'not-managed' }
    if (state.port !== target.port) return { recovered: false, reason: 'other-endpoint' }
    // A live process may be loading or busy; it is never killed to recover.
    if ((await ownership(state)) === 'running') return { recovered: false, reason: 'still-running' }
    await start({ modelPath: state.model, port: state.port, ctxSize: state.ctxSize })
    return {
      recovered: true,
      message: `${spec.label} had stopped (it may have run out of memory). Xerxes restarted it with its ${state.ctxSize}-token window and is retrying the step once.`,
    }
  }

  return { engine, findBinary, version, start, stop, status, recover }
}

async function psCommand(pid: number): Promise<string | undefined> {
  if (process.platform === 'win32') return undefined
  let proc: ReturnType<typeof Bun.spawn>
  try {
    proc = Bun.spawn({ cmd: ['ps', '-o', 'command=', '-p', String(pid)], stdin: 'ignore', stdout: 'pipe', stderr: 'ignore' })
  } catch {
    return undefined
  }
  const timer = setTimeout(() => proc.kill(), PS_TIMEOUT_MS)
  try {
    const stdout = proc.stdout
    const text = stdout instanceof ReadableStream ? await new Response(stdout).text() : ''
    const code = await proc.exited
    const command = text.trim()
    return code === 0 && command !== '' ? command : undefined
  } finally {
    clearTimeout(timer)
  }
}

function portIsInUse(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const probe = createServer()
    probe.once('error', (error: NodeJS.ErrnoException) => resolve(error.code === 'EADDRINUSE'))
    probe.once('listening', () => probe.close(() => resolve(false)))
    probe.listen(port, '127.0.0.1')
  })
}

/** Real ports: detached child with stdout/stderr appended to the log file, argv-only (no shell). */
export function systemServerPorts(): ServerPorts {
  return {
    which: (name) => Bun.which(name) ?? undefined,
    exists: (path) => Bun.file(path).exists(),
    spawn(argv, { logPath }) {
      const [command, ...args] = argv
      if (command === undefined) throw new Error('empty argv')
      const fd = openSync(logPath, 'a', 0o600)
      try {
        const child = spawnChild(command, args, { detached: true, stdio: ['ignore', fd, fd] })
        const exited = new Promise<number>((resolve) => {
          child.once('exit', (code) => resolve(code ?? 1))
          child.once('error', () => resolve(127))
        })
        child.unref()
        if (child.pid === undefined) throw new Error(`${command} could not be started (is it executable?)`)
        return { pid: child.pid, exited }
      } finally {
        closeSync(fd)
      }
    },
    fetch: (url, init) => fetch(url, init),
    isPidAlive(pid) {
      try {
        process.kill(pid, 0)
        return true
      } catch (error) {
        return (error as NodeJS.ErrnoException).code === 'EPERM'
      }
    },
    commandOf: psCommand,
    signal: (pid, signal) => process.kill(pid, signal),
    portInUse: portIsInUse,
    async run(argv, timeoutMs) {
      const proc = Bun.spawn({ cmd: [...argv], stdin: 'ignore', stdout: 'pipe', stderr: 'pipe' })
      const timer = setTimeout(() => proc.kill(), timeoutMs)
      try {
        const [stdout, stderr] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text()])
        return { exitCode: await proc.exited, output: `${stdout}\n${stderr}` }
      } finally {
        clearTimeout(timer)
      }
    },
    sleep: (ms) => Bun.sleep(ms),
    now: () => Date.now(),
  }
}
