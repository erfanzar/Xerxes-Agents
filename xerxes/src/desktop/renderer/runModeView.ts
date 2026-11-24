// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { LocalSetupJobView, LocalStatusView, RunModeName, RunModeView } from './types.js'

/** True when running on macOS; falls back to false on Windows/Linux. */
export function isMacPlatform(): boolean {
  if (typeof navigator !== 'undefined') {
    const info = `${navigator.platform ?? ''} ${navigator.userAgent ?? ''}`.trim()
    if (info) return /Mac|iPhone|iPad|iPod/i.test(info)
  }
  return typeof process !== 'undefined' && process.platform === 'darwin'
}

/** Platform-aware label for the local device ("this Mac" on macOS, "this computer" elsewhere). */
export function localMachineLabel(capitalized = false): string {
  const mac = isMacPlatform()
  if (capitalized) return mac ? 'This Mac' : 'This computer'
  return mac ? 'this Mac' : 'this computer'
}

/** Short noun for the local device ("Mac" on macOS, "Computer" elsewhere). */
export function localMachineNoun(capitalized = false): string {
  const mac = isMacPlatform()
  if (capitalized) return mac ? 'Mac' : 'Computer'
  return mac ? 'Mac' : 'computer'
}

export interface RunModeChoice {
  readonly mode: RunModeName
  readonly label: string
  readonly summary: string
  readonly boundary: string
}

/** What each mode means, in plain words: who does the work and what may leave. */
export function runModeChoices(machine = localMachineLabel()): ReadonlyArray<RunModeChoice> {
  return [
    { mode: 'private', label: 'Private', summary: `A local model on ${machine} does all the work.`, boundary: 'Nothing leaves during a run.' },
    { mode: 'hybrid', label: 'Hybrid', summary: `A cloud model plans and checks; a local model does the file work on ${machine}.`, boundary: 'Your files stay here. Plans and short summaries go to the cloud.' },
    { mode: 'cloud', label: 'Cloud', summary: 'A cloud model does the work.', boundary: 'Task context may leave.' },
  ]
}

/** Default run-mode choices evaluated for the current platform. */
export const RUN_MODE_CHOICES: ReadonlyArray<RunModeChoice> = runModeChoices()

function record(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function text(value: unknown): string | null {
  return typeof value === 'string' && value.trim() !== '' ? value : null
}

function count(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
}

export function runModeView(result: Readonly<Record<string, unknown>>): RunModeView {
  const mode = result.mode === 'private' || result.mode === 'hybrid' || result.mode === 'cloud' ? result.mode : null
  const worker = record(result.worker)
  return {
    mode,
    worker: text(worker.profile) === null ? null : { profile: text(worker.profile)!, model: text(worker.model) ?? '', local: worker.local === true },
    restartPending: result.restart_pending === true || result.restart_required === true,
    lockedBy: Array.isArray(result.locked_by) ? result.locked_by.filter((name): name is string => typeof name === 'string') : [],
  }
}

export function localStatusView(raw: unknown): LocalStatusView | null {
  const status = record(raw)
  if (Object.keys(status).length === 0) return null
  const engine = record(status.engine)
  const model = record(status.model)
  const server = record(status.server)
  return {
    ready: status.ready === true,
    engine: {
      name: text(engine.name) ?? 'llama-server',
      label: text(engine.label) ?? 'llama.cpp',
      found: engine.found === true,
      version: text(engine.version),
      newEnough: typeof engine.new_enough === 'boolean' ? engine.new_enough : null,
    },
    model: { id: text(model.id), label: text(model.label), bytes: count(model.bytes), downloaded: model.downloaded === true },
    server: { running: server.running === true, port: count(server.port), contextLength: count(server.context_length) },
    memoryBytes: count(status.memory_bytes),
    blocker: text(status.blocker),
  }
}

export function localSetupJobView(raw: unknown): LocalSetupJobView | null {
  const job = record(raw)
  const id = text(job.id)
  const state = job.state === 'running' || job.state === 'done' || job.state === 'failed' || job.state === 'cancelled' ? job.state : null
  if (id === null || state === null) return null
  const error = record(job.error)
  return { id, state, step: text(job.step) ?? '', percent: state === 'running' ? count(job.percent) : null, error: text(error.message) }
}

/** "24 GB": memory as Macs are sold, in binary gigabytes. */
export function memoryGigabytes(bytes: number): string {
  return `${Math.round(bytes / 2 ** 30)} GB`
}

/** "7.6 GB": download sizes as people read them on a Mac (decimal gigabytes). */
export function gigabytes(bytes: number): string {
  return `${(bytes / 1e9).toFixed(bytes >= 1e10 ? 0 : 1)} GB`
}
