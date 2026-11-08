// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * LM Studio as a managed local engine, driven through its `lms` CLI. LM Studio
 * owns the inference process; Xerxes only asks it to serve on 127.0.0.1 and to
 * load one model with one slot (`--parallel 1`) and a chosen window, then
 * talks to it over the OpenAI-compatible HTTP API like any other provider.
 *
 * On Apple silicon this is the engine for MLX packs such as Ternary Bonsai 2
 * 27B, whose Hadamard transforms stock MLX loaders do not apply.
 *
 * A state file records what Xerxes itself did (started the server, loaded the
 * model), so `stop` undoes only that and never unloads a model or stops a
 * server the user started. Every side effect (running `lms`, HTTP, files'
 * existence, time) goes through `LmStudioPorts`, so tests use fakes.
 */
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'
import { parseLmStudioModels, systemWindowFetch, type LmStudioModelEntry, type WindowFetch } from './loadedWindow.js'
import { ServerError, type ServerStatus } from './server.js'

export const DEFAULT_LOAD_TIMEOUT_MS = 300_000
const SERVER_START_TIMEOUT_MS = 60_000
const COMMAND_TIMEOUT_MS = 30_000
const HTTP_TIMEOUT_MS = 3000
const DEFAULT_POLL_MS = 500
const OUTPUT_TAIL_CHARS = 2000

export const LMS_INSTALL_HELP =
  "LM Studio's `lms` CLI was not found. Install LM Studio from https://lmstudio.ai and open it once " +
  '(it installs `lms` into ~/.lmstudio/bin), or set `runtime.local.lms_path` in ~/.xerxes/daemon/config.json ' +
  '(or XERXES_LOCAL_LMS).'

export interface LmsRunResult {
  readonly exitCode: number
  readonly stdout: string
  readonly stderr: string
}

export interface LmStudioPorts {
  which(name: string): string | undefined
  exists(path: string): Promise<boolean>
  /**
   * Run `lms` with an argv (never a shell). Resolves with its exit code and
   * output; a run past `timeoutMs` is killed and rejects.
   */
  run(argv: readonly string[], options: { readonly timeoutMs: number }): Promise<LmsRunResult>
  readonly fetch: WindowFetch
  sleep(ms: number): Promise<void>
  now(): number
}

export interface LmStudioOptions {
  /** Configured `lms` path (runtime.local.lms_path / XERXES_LOCAL_LMS). */
  readonly lmsPath?: string
  /** Home directory used to find ~/.lmstudio/bin/lms. */
  readonly homeDir?: string
  readonly stateDir?: string
  readonly pollMs?: number
}

export interface LmStudioModel {
  readonly modelKey: string
  readonly displayName?: string
  readonly format?: string
  readonly maxContextLength?: number
}

export interface LmStudioServerInfo {
  readonly running: boolean
  readonly port?: number
}

export interface LmStudioStartOptions {
  /** LM Studio model key, as `lms ls` prints it. */
  readonly modelKey: string
  /** API identifier; the profile's model id. */
  readonly identifier: string
  /** `--context-length`. Undefined lets LM Studio choose; the loaded window is reported either way. */
  readonly contextLength?: number
  /** Port for `lms server start` when the server is not running. Undefined keeps LM Studio's last port. */
  readonly port?: number
  readonly timeoutMs?: number
  readonly onProgress?: (text: string) => void
}

export type LmStudioStopResult =
  | {
      readonly stopped: true
      readonly model: string
      /** Xerxes loaded the model and unloaded it now. */
      readonly unloaded: boolean
      /** Xerxes started LM Studio's server and stopped it now. */
      readonly serverStopped: boolean
    }
  | { readonly stopped: false; readonly reason: 'not-running' | 'stale' }

export interface LmStudioEngine {
  findBinary(): Promise<string | undefined>
  /** LLMs in LM Studio's library (`lms ls --llm --json`). */
  listModels(): Promise<LmStudioModel[]>
  /** LM Studio's HTTP server (`lms server status --json`). */
  serverInfo(): Promise<LmStudioServerInfo>
  start(options: LmStudioStartOptions): Promise<ServerStatus>
  stop(): Promise<LmStudioStopResult>
  status(): Promise<ServerStatus>
}

interface LmStudioState {
  readonly engine: 'lmstudio'
  readonly port: number
  readonly modelKey: string
  readonly identifier: string
  readonly contextLength: number
  readonly startedAt: string
  /** Xerxes ran `lms server start`, so `stop` may stop the server again. */
  readonly startedServer: boolean
  /** Xerxes ran `lms load`; false when an identical load was adopted, which `stop` leaves alone. */
  readonly loadedModel: boolean
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isState(value: unknown): value is LmStudioState {
  if (!isRecord(value)) return false
  return (
    value.engine === 'lmstudio' &&
    typeof value.port === 'number' &&
    Number.isInteger(value.port) &&
    typeof value.modelKey === 'string' &&
    typeof value.identifier === 'string' &&
    typeof value.contextLength === 'number' &&
    typeof value.startedAt === 'string' &&
    typeof value.startedServer === 'boolean' &&
    typeof value.loadedModel === 'boolean'
  )
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : undefined
}

/** Parse `lms ls --json`. Throws when the output is not that shape. */
export function parseLmsModelList(text: string): LmStudioModel[] {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    throw new Error('`lms ls --json` did not print JSON')
  }
  if (!Array.isArray(parsed)) throw new Error('`lms ls --json` did not print a list')
  const models: LmStudioModel[] = []
  for (const item of parsed) {
    if (!isRecord(item) || typeof item.modelKey !== 'string' || item.modelKey.trim() === '') continue
    if (item.type !== undefined && item.type !== 'llm') continue
    const max = positiveInteger(item.maxContextLength)
    models.push({
      modelKey: item.modelKey,
      ...(typeof item.displayName === 'string' ? { displayName: item.displayName } : {}),
      ...(typeof item.format === 'string' ? { format: item.format } : {}),
      ...(max === undefined ? {} : { maxContextLength: max }),
    })
  }
  return models
}

/** Parse `lms server status --json` (`{"running":true,"port":1234}`). */
export function parseLmsServerStatus(text: string): LmStudioServerInfo {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    throw new Error('`lms server status --json` did not print JSON')
  }
  if (!isRecord(parsed) || typeof parsed.running !== 'boolean') {
    throw new Error('`lms server status --json` printed an unexpected object')
  }
  const port = positiveInteger(parsed.port)
  return { running: parsed.running, ...(port === undefined ? {} : { port }) }
}

function tail(text: string): string {
  const trimmed = text.trim()
  return trimmed.length > OUTPUT_TAIL_CHARS ? `…${trimmed.slice(-OUTPUT_TAIL_CHARS)}` : trimmed
}

function outputOf(result: LmsRunResult): string {
  const text = tail([result.stderr, result.stdout].filter((part) => part.trim() !== '').join('\n'))
  return text === '' ? '' : `\n${text}`
}

export function createLmStudioEngine(ports: LmStudioPorts, options: LmStudioOptions = {}): LmStudioEngine {
  const stateDir = options.stateDir ?? join(xerxesHome(), 'local')
  const statePath = join(stateDir, 'lmstudio.json')
  const pollMs = options.pollMs ?? DEFAULT_POLL_MS

  async function findBinary(): Promise<string | undefined> {
    if (options.lmsPath !== undefined) return (await ports.exists(options.lmsPath)) ? options.lmsPath : undefined
    const onPath = ports.which('lms')
    if (onPath !== undefined) return onPath
    const bundled = join(options.homeDir ?? homedir(), '.lmstudio', 'bin', 'lms')
    return (await ports.exists(bundled)) ? bundled : undefined
  }

  async function requireBinary(): Promise<string> {
    const lms = await findBinary()
    if (lms !== undefined) return lms
    throw new ServerError(
      'BINARY_NOT_FOUND',
      options.lmsPath === undefined ? LMS_INSTALL_HELP : `lms not found at the configured path ${options.lmsPath}. ${LMS_INSTALL_HELP}`,
    )
  }

  async function lms(args: readonly string[], timeoutMs: number, code: 'START_FAILED' | 'LOAD_FAILED' | 'STOP_FAILED'): Promise<LmsRunResult> {
    const bin = await requireBinary()
    let result: LmsRunResult
    try {
      result = await ports.run([bin, ...args], { timeoutMs })
    } catch (error) {
      throw new ServerError(code, `\`lms ${args.join(' ')}\` failed: ${error instanceof Error ? error.message : String(error)}`)
    }
    if (result.exitCode !== 0) {
      throw new ServerError(code, `\`lms ${args.join(' ')}\` exited with code ${result.exitCode}.${outputOf(result)}`, {
        exitCode: result.exitCode,
      })
    }
    return result
  }

  async function readState(): Promise<LmStudioState | undefined> {
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

  async function writeState(state: LmStudioState): Promise<void> {
    await mkdir(stateDir, { recursive: true, mode: 0o700 })
    await writeFile(statePath, JSON.stringify(state, null, 2) + '\n', { mode: 0o600 })
  }

  const removeState = (): Promise<void> => rm(statePath, { force: true })

  /** LM Studio's model list on a port, or undefined when nothing answers there as LM Studio. */
  async function loadedModels(port: number): Promise<LmStudioModelEntry[] | undefined> {
    try {
      const response = await ports.fetch(`http://127.0.0.1:${port}/api/v0/models`, {
        signal: AbortSignal.timeout(HTTP_TIMEOUT_MS),
        headers: { Accept: 'application/json' },
      })
      if (!response.ok) {
        await response.body?.cancel().catch(() => undefined)
        return undefined
      }
      return parseLmStudioModels(await response.json())
    } catch {
      return undefined
    }
  }

  async function listModels(): Promise<LmStudioModel[]> {
    const result = await lms(['ls', '--llm', '--json'], COMMAND_TIMEOUT_MS, 'START_FAILED')
    try {
      return parseLmsModelList(result.stdout)
    } catch (error) {
      throw new ServerError('START_FAILED', `Could not read LM Studio's model list: ${(error as Error).message}`)
    }
  }

  async function serverInfo(): Promise<LmStudioServerInfo> {
    const result = await lms(['server', 'status', '--json'], COMMAND_TIMEOUT_MS, 'START_FAILED')
    try {
      return parseLmsServerStatus(result.stdout)
    } catch (error) {
      throw new ServerError('START_FAILED', `Could not read LM Studio's server status: ${(error as Error).message}`)
    }
  }

  async function waitFor<T>(withinMs: number, probe: () => Promise<T | undefined>): Promise<T | undefined> {
    const deadline = ports.now() + withinMs
    for (;;) {
      const value = await probe()
      if (value !== undefined) return value
      if (ports.now() >= deadline) return undefined
      await ports.sleep(pollMs)
    }
  }

  async function status(): Promise<ServerStatus> {
    const state = await readState()
    if (state === undefined) return { running: false, healthy: false, stale: false }
    const entries = await loadedModels(state.port)
    const entry = entries?.find((candidate) => candidate.id === state.identifier && candidate.state === 'loaded')
    if (entry === undefined) {
      return { running: false, healthy: false, stale: true, port: state.port, model: state.identifier }
    }
    return {
      running: true,
      healthy: true,
      stale: false,
      port: state.port,
      model: state.identifier,
      startedAt: state.startedAt,
      contextLength: entry.loadedContextLength ?? state.contextLength,
    }
  }

  async function start(start: LmStudioStartOptions): Promise<ServerStatus> {
    const progress = start.onProgress ?? (() => undefined)
    const existing = await readState()
    if (existing !== undefined) {
      const current = await status()
      if (current.running) {
        throw new ServerError(
          'ALREADY_RUNNING',
          `Xerxes already has ${existing.identifier} loaded in LM Studio (port ${existing.port}). Run \`xerxes local stop --engine lmstudio\` first.`,
          { port: existing.port },
        )
      }
      await removeState()
    }
    await requireBinary()

    // LM Studio runs one HTTP server. Use it when it is up; start it otherwise.
    const before = await serverInfo()
    let port: number
    let startedServer = false
    if (before.running && before.port !== undefined) {
      port = before.port
      if (start.port !== undefined && start.port !== port) {
        progress(`LM Studio's server is already running on port ${port}; using it instead of port ${start.port}.`)
      }
    } else {
      progress(`Starting LM Studio's server${start.port === undefined ? '' : ` on 127.0.0.1:${start.port}`}…`)
      await lms(['server', 'start', ...(start.port === undefined ? [] : ['--port', String(start.port)])], SERVER_START_TIMEOUT_MS, 'START_FAILED')
      startedServer = true
      const after = await serverInfo()
      const resolved = after.port ?? start.port
      if (!after.running || resolved === undefined) {
        throw new ServerError('START_FAILED', '`lms server start` returned, but LM Studio reports its server is not running.')
      }
      port = resolved
    }

    const stopOwnServer = async (): Promise<void> => {
      if (!startedServer) return
      await ports.run([await requireBinary(), 'server', 'stop'], { timeoutMs: COMMAND_TIMEOUT_MS }).catch(() => undefined)
    }

    try {
      const listed = await waitFor(SERVER_START_TIMEOUT_MS, () => loadedModels(port))
      if (listed === undefined) {
        throw new ServerError('START_FAILED', `LM Studio's server did not answer on http://127.0.0.1:${port}/api/v0/models.`, { port })
      }
      const present = listed.find((entry) => entry.id === start.identifier && entry.state === 'loaded')
      let loadedModel = false
      if (present !== undefined) {
        if (start.contextLength !== undefined && present.loadedContextLength !== start.contextLength) {
          throw new ServerError(
            'ALREADY_RUNNING',
            `LM Studio already has ${start.identifier} loaded with a ${present.loadedContextLength ?? 'unknown'}-token window, not ${start.contextLength}. ` +
              `Unload it with \`lms unload ${start.identifier}\`, or set runtime.local.ctx_size to ${present.loadedContextLength ?? 'its window'}.`,
            { port },
          )
        }
        progress(`LM Studio already has ${start.identifier} loaded; using it.`)
      } else {
        const args = [
          'load',
          start.modelKey,
          ...(start.contextLength === undefined ? [] : ['--context-length', String(start.contextLength)]),
          '--parallel',
          '1',
          '--gpu',
          'max',
          '--identifier',
          start.identifier,
          '-y',
        ]
        progress(`Loading ${start.modelKey} (${start.contextLength === undefined ? "LM Studio's default window" : `${start.contextLength}-token window`}, one slot)…`)
        await lms(args, start.timeoutMs ?? DEFAULT_LOAD_TIMEOUT_MS, 'LOAD_FAILED')
        loadedModel = true
      }

      const loaded = await waitFor(10_000, async () => {
        const entries = await loadedModels(port)
        const entry = entries?.find((candidate) => candidate.id === start.identifier)
        return entry?.state === 'loaded' && entry.loadedContextLength !== undefined ? entry : undefined
      })
      if (loaded?.loadedContextLength === undefined) {
        if (loadedModel) {
          await ports.run([await requireBinary(), 'unload', start.identifier], { timeoutMs: COMMAND_TIMEOUT_MS }).catch(() => undefined)
        }
        throw new ServerError('LOAD_FAILED', `\`lms load\` finished, but LM Studio does not report ${start.identifier} as loaded with a window.`)
      }
      if (start.contextLength !== undefined && loaded.loadedContextLength !== start.contextLength) {
        progress(`LM Studio loaded a ${loaded.loadedContextLength}-token window (asked for ${start.contextLength}).`)
      }
      const startedAt = new Date(ports.now()).toISOString()
      await writeState({
        engine: 'lmstudio',
        port,
        modelKey: start.modelKey,
        identifier: start.identifier,
        contextLength: loaded.loadedContextLength,
        startedAt,
        startedServer,
        loadedModel,
      })
      return {
        running: true,
        healthy: true,
        stale: false,
        port,
        model: start.identifier,
        startedAt,
        contextLength: loaded.loadedContextLength,
      }
    } catch (error) {
      await stopOwnServer()
      throw error
    }
  }

  async function stop(): Promise<LmStudioStopResult> {
    const state = await readState()
    if (state === undefined) return { stopped: false, reason: 'not-running' }
    const entries = await loadedModels(state.port)
    const ours = entries?.find((entry) => entry.id === state.identifier && entry.state === 'loaded')
    if (entries === undefined || ours === undefined) {
      await removeState()
      return { stopped: false, reason: 'stale' }
    }
    let unloaded = false
    if (state.loadedModel) {
      await lms(['unload', state.identifier], COMMAND_TIMEOUT_MS, 'STOP_FAILED')
      unloaded = true
    }
    // Stop the server only when Xerxes started it and nothing else is loaded on it.
    let serverStopped = false
    if (state.startedServer) {
      const remaining = (await loadedModels(state.port))?.filter((entry) => entry.state === 'loaded') ?? []
      if (remaining.length === 0) {
        await lms(['server', 'stop'], COMMAND_TIMEOUT_MS, 'STOP_FAILED')
        serverStopped = true
      }
    }
    await removeState()
    return { stopped: true, model: state.identifier, unloaded, serverStopped }
  }

  return { findBinary, listModels, serverInfo, start, stop, status }
}

/** Real ports: `lms` via Bun.spawn (argv only), the platform fetch, real time. */
export function systemLmStudioPorts(): LmStudioPorts {
  return {
    which: (name) => Bun.which(name) ?? undefined,
    exists: (path) => Bun.file(path).exists(),
    async run(argv, { timeoutMs }) {
      const proc = Bun.spawn({ cmd: [...argv], stdin: 'ignore', stdout: 'pipe', stderr: 'pipe' })
      let timedOut = false
      const timer = setTimeout(() => {
        timedOut = true
        proc.kill()
      }, timeoutMs)
      try {
        const [stdout, stderr, exitCode] = await Promise.all([
          new Response(proc.stdout).text(),
          new Response(proc.stderr).text(),
          proc.exited,
        ])
        if (timedOut) throw new Error(`timed out after ${Math.round(timeoutMs / 1000)}s`)
        return { exitCode, stdout, stderr }
      } finally {
        clearTimeout(timer)
      }
    },
    fetch: systemWindowFetch,
    sleep: (ms) => Bun.sleep(ms),
    now: () => Date.now(),
  }
}
