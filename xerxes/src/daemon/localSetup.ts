// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * `xerxes local setup` and `doctor` for the desktop app, on the same code path
 * as the CLI. A setup downloads gigabytes, far longer than one RPC may take,
 * so it runs as a job: `local.setup` starts it and returns, each progress line
 * the command prints is published as a `local_setup` event, and `local.status`
 * reports the job too, so a reloaded window picks it up again.
 */
import { randomUUID } from 'node:crypto'

import { NO_ENGINE_HELP } from '../local/command.js'

/** Run `xerxes local <argv>` with these sinks (production: `runLocalCommand` with the system ports). */
export type LocalCommandRunner = (
  argv: readonly string[],
  io: { stdout(text: string): void; stderr(text: string): void; readonly signal: AbortSignal },
) => Promise<number>

export type LocalEngine = 'llama-server' | 'mlx' | 'lmstudio'

interface JobBase {
  readonly id: string
  readonly startedAt: number
  /** The latest line the command printed. */
  readonly step: string
}

export type LocalSetupJob =
  | (JobBase & { readonly state: 'running'; readonly percent: number | null })
  | (JobBase & { readonly state: 'done'; readonly finishedAt: number; readonly result: Readonly<Record<string, unknown>> })
  | (JobBase & { readonly state: 'failed'; readonly finishedAt: number; readonly error: { readonly code: string; readonly message: string } })
  | (JobBase & { readonly state: 'cancelled'; readonly finishedAt: number })

export interface LocalStatus {
  readonly ready: boolean
  readonly engine: {
    readonly name: LocalEngine
    readonly label: string
    readonly found: boolean
    readonly version: string | null
    /** False when the installed engine is older than the model needs; null when unknown. */
    readonly newEnough: boolean | null
  }
  readonly model: { readonly id: string | null; readonly label: string | null; readonly bytes: number | null; readonly downloaded: boolean }
  readonly server: { readonly running: boolean; readonly port: number | null; readonly contextLength: number | null }
  readonly memoryBytes: number | null
  /** What to do before setup can succeed (install or update the engine, free memory); null when setup can run. */
  readonly blocker: string | null
}

const ENGINE_LABELS: Record<LocalEngine, string> = { 'llama-server': 'llama.cpp', mlx: 'MLX', lmstudio: 'LM Studio' }

function record(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function text(value: unknown): string | null {
  return typeof value === 'string' && value.trim() !== '' ? value : null
}

function count(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
}

/** The single JSON object a `--json` local command prints, or a typed failure. */
export function parseCommandJson(stdout: string): { ok: true; payload: Record<string, unknown> } | { ok: false; code: string; message: string } {
  const line = stdout.trim().split('\n').filter(Boolean).at(-1)
  if (line === undefined) return { ok: false, code: 'NO_OUTPUT', message: 'The local command printed no result.' }
  let parsed: unknown
  try {
    parsed = JSON.parse(line)
  } catch {
    return { ok: false, code: 'BAD_OUTPUT', message: 'The local command printed a result that is not JSON.' }
  }
  const payload = record(parsed)
  if (payload.ok === true) return { ok: true, payload }
  const error = record(payload.error)
  return { ok: false, code: text(error.code) ?? 'FAILED', message: text(error.message) ?? 'The local command failed.' }
}

/** `xerxes local doctor --json`, whichever engine it reported on, as one shape. */
export function localStatusFromDoctor(payload: Readonly<Record<string, unknown>>): LocalStatus {
  const name: LocalEngine = payload.engine === 'mlx' || payload.engine === 'lmstudio' ? payload.engine : 'llama-server'
  const binary = record(name === 'mlx' ? payload.mlx_server : name === 'lmstudio' ? payload.lms : payload.llama_server)
  const found = binary.found === true
  const version = name === 'mlx' ? text(binary.mlx_lm_version)
    : text(binary.version) ?? (count(binary.build) === null ? null : `build ${String(binary.build)}`)
  const newEnough = typeof binary.new_enough === 'boolean' ? binary.new_enough : null
  const model = record(payload.model)
  const downloaded = name === 'mlx'
    ? count(model.files_total) !== null && model.files_total !== 0 && model.files_verified === model.files_total
    : name === 'lmstudio' ? model.present === true : model.verified === true || (model.verified === null && model.present === true)
  const server = record(payload.server)
  const selection = record(payload.selection)
  const hardware = record(payload.hardware)
  const label = ENGINE_LABELS[name]
  const blocker = !found ? NO_ENGINE_HELP
    : newEnough === false ? `${label} ${version ?? ''} is older than this model needs. Update it (\`brew upgrade llama.cpp\` or \`uv tool upgrade mlx-lm\`), then check again.`.replace('  ', ' ')
      : selection.ok === false ? text(selection.reason) ?? 'No model fits this machine.'
        : null
  return {
    ready: payload.ready === true,
    engine: { name, label, found, version, newEnough },
    model: { id: text(model.id), label: text(model.label), bytes: count(model.bytes), downloaded },
    server: { running: server.running === true, port: count(server.port), contextLength: count(server.context_length) },
    memoryBytes: count(hardware.total_mem_bytes),
    blocker,
  }
}

const PERCENT = /:\s(\d{1,3})% \(/

export interface LocalSetupJobsOptions {
  readonly run: LocalCommandRunner
  /** Every change to the job: started, each progress line, and the end. */
  readonly publish: (job: LocalSetupJob) => void
  /** After a setup succeeds and before `done` is published (the daemon applies the chosen mode here). */
  readonly onSucceeded?: (result: Readonly<Record<string, unknown>>) => Promise<void>
  readonly now?: () => number
}

export interface LocalSetupJobs {
  /** `local doctor --json`, plus the current or last job. */
  status(): Promise<{ ok: true; status: LocalStatus; job: LocalSetupJob | null } | { ok: false; code: string; error: string; job: LocalSetupJob | null }>
  /** Start `local setup --json`; refused while one is running. */
  start(): { ok: true; job: LocalSetupJob } | { ok: false; code: 'busy'; job: LocalSetupJob }
  /** Stop the running setup; a partial download is kept and resumed next time. */
  cancel(): boolean
  current(): LocalSetupJob | null
}

export function createLocalSetupJobs(options: LocalSetupJobsOptions): LocalSetupJobs {
  const now = options.now ?? Date.now
  let job: LocalSetupJob | null = null
  let controller: AbortController | null = null

  const set = (next: LocalSetupJob): void => {
    job = next
    options.publish(next)
  }

  async function runSetup(id: string, startedAt: number, signal: AbortSignal): Promise<void> {
    let stdout = ''
    let step = 'Starting…'
    let percent: number | null = null
    let pending = ''
    const onLine = (line: string): void => {
      const trimmed = line.trim()
      if (trimmed === '' || job?.id !== id || job.state !== 'running') return
      step = trimmed
      const match = PERCENT.exec(trimmed)
      percent = match === null ? (trimmed.startsWith('Download') ? percent : null) : Math.min(100, Number(match[1]))
      set({ id, startedAt, state: 'running', step, percent })
    }
    let exit: number
    try {
      exit = await options.run(['setup', '--json'], {
        stdout: chunk => { stdout += chunk },
        stderr: chunk => {
          pending += chunk
          for (let newline = pending.indexOf('\n'); newline >= 0; newline = pending.indexOf('\n')) {
            onLine(pending.slice(0, newline))
            pending = pending.slice(newline + 1)
          }
        },
        signal,
      })
    } catch (error) {
      if (signal.aborted) {
        set({ id, startedAt, state: 'cancelled', step, finishedAt: now() })
        return
      }
      set({ id, startedAt, state: 'failed', step, finishedAt: now(), error: { code: 'FAILED', message: error instanceof Error ? error.message : String(error) } })
      return
    }
    if (pending.trim() !== '') onLine(pending)
    if (signal.aborted) {
      set({ id, startedAt, state: 'cancelled', step, finishedAt: now() })
      return
    }
    const parsed = parseCommandJson(stdout)
    if (!parsed.ok || exit !== 0) {
      const error = parsed.ok ? { code: 'FAILED', message: `Setup exited with status ${exit}.` } : { code: parsed.code, message: parsed.message }
      set({ id, startedAt, state: 'failed', step, finishedAt: now(), error })
      return
    }
    try {
      await options.onSucceeded?.(parsed.payload)
    } catch (error) {
      set({ id, startedAt, state: 'failed', step, finishedAt: now(), error: { code: 'MODE_NOT_APPLIED', message: `Bonsai is set up, but the mode was not applied: ${error instanceof Error ? error.message : String(error)}` } })
      return
    }
    set({ id, startedAt, state: 'done', step, finishedAt: now(), result: parsed.payload })
  }

  return {
    async status() {
      let stdout = ''
      const controller = new AbortController()
      try {
        await options.run(['doctor', '--json'], { stdout: chunk => { stdout += chunk }, stderr: () => {}, signal: controller.signal })
      } catch (error) {
        return { ok: false, code: 'FAILED', error: error instanceof Error ? error.message : String(error), job }
      }
      const parsed = parseCommandJson(stdout)
      if (!parsed.ok) return { ok: false, code: parsed.code, error: parsed.message, job }
      return { ok: true, status: localStatusFromDoctor(parsed.payload), job }
    },
    start() {
      if (job?.state === 'running') return { ok: false, code: 'busy', job }
      const id = randomUUID()
      const startedAt = now()
      const running = new AbortController()
      controller = running
      set({ id, startedAt, state: 'running', step: 'Starting…', percent: null })
      const started = job!
      void runSetup(id, startedAt, running.signal).finally(() => {
        if (controller === running) controller = null
      })
      return { ok: true, job: started }
    },
    cancel() {
      if (job?.state !== 'running' || controller === null) return false
      controller.abort()
      return true
    },
    current() {
      return job
    },
  }
}

/** The wire form of a job (snake_case, like every other daemon payload). */
export function localSetupJobPayload(job: LocalSetupJob | null): Record<string, unknown> | null {
  if (job === null) return null
  return {
    id: job.id,
    state: job.state,
    step: job.step,
    started_at: job.startedAt,
    ...(job.state === 'running' ? { percent: job.percent } : { finished_at: job.finishedAt }),
    ...(job.state === 'done' ? { result: job.result } : {}),
    ...(job.state === 'failed' ? { error: job.error } : {}),
  }
}

export function localStatusPayload(status: LocalStatus): Record<string, unknown> {
  return {
    ready: status.ready,
    engine: { name: status.engine.name, label: status.engine.label, found: status.engine.found, version: status.engine.version, new_enough: status.engine.newEnough },
    model: { id: status.model.id, label: status.model.label, bytes: status.model.bytes, downloaded: status.model.downloaded },
    server: { running: status.server.running, port: status.server.port, context_length: status.server.contextLength },
    memory_bytes: status.memoryBytes,
    blocker: status.blocker,
  }
}
