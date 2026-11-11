// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Private, Hybrid and Cloud: the product's three ways to run a task, read from
 * and written to the settings that already govern them.
 *
 *   private  Bonsai (a local profile) does the work and `runtime.local_only`
 *            is on: nothing but this machine is reachable during a run
 *   hybrid   a cloud model leads: it plans, splits the work and checks it,
 *            while the data boundary keeps every workspace file away from it
 *            (`data_boundary: hybrid`, every path private) and Bonsai agents
 *            do the file work on this machine, their replies reaching the
 *            lead as capped summaries
 *   cloud    a cloud model does the work
 *
 * The main agent is the active provider profile. Each mode also places
 * subagents (the Agent intelligence tiers): all on Bonsai in Private; light and
 * balanced on Bonsai with thinking off and smart on the lead in Hybrid; all on
 * the cloud model in Cloud. Fully local mode and the boundary are read when the
 * daemon starts, so changing them needs a restart; the state says when one is
 * pending.
 */
import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

import type { AgentIntelligenceConfig } from '../agents/intelligence.js'
import { ValidationError } from '../core/errors.js'
import { parseAdvisorConfig } from '../runtime/advisorBudget.js'
import { LOCAL_ONLY_ENV, parseLocalOnlyConfig } from '../runtime/localOnly.js'
import type { DataTrust } from '../security/boundaryTypes.js'
import { deriveDataTrust, parseDataBoundaryConfig } from '../security/dataBoundary.js'

export type RunMode = 'private' | 'hybrid' | 'cloud'
export const RUN_MODES: readonly RunMode[] = ['private', 'hybrid', 'cloud']

/** What Hybrid sets in `runtime.boundary`: every path stays local, local agents' replies reach the lead capped. */
export const HYBRID_BOUNDARY: Readonly<Record<string, unknown>> = Object.freeze({
  private_paths: ['**'],
  cloud_results: 'summary',
  approve_outbound: 'off',
})
const DATA_BOUNDARY_ENV = 'XERXES_DATA_BOUNDARY'

type Env = Readonly<Record<string, string | undefined>>
type Runtime = Readonly<Record<string, unknown>>

export interface RunModeProfile {
  readonly name: string
  readonly model: string
  readonly base_url?: string
  readonly data_trust?: DataTrust
}

export interface RunModeState {
  /** Null until a main profile is chosen, or when the settings match no mode. */
  readonly mode: RunMode | null
  /** The main agent: Bonsai in Private, the cloud lead in Hybrid, the cloud model in Cloud. */
  readonly worker: { readonly profile: string; readonly model: string; readonly local: boolean } | null
  readonly localOnly: boolean
  /** The saved settings differ from the ones this daemon started with. */
  readonly restartPending: boolean
  /** Environment variables that override the saved settings. */
  readonly lockedBy: readonly string[]
}

export function isLocalProfile(profile: Pick<RunModeProfile, 'base_url' | 'data_trust'>): boolean {
  return deriveDataTrust(profile) === 'local'
}

function boundaryMode(runtime: Runtime, env: Env): string {
  try {
    return parseDataBoundaryConfig({ ...runtime }, { ...env }).mode
  } catch {
    // A setting the runtime would refuse is no mode at all.
    return 'invalid'
  }
}

/** The mode the saved settings and the chosen main profile describe. */
export function readRunMode(input: {
  readonly saved: Runtime
  /** The runtime settings this daemon process started with. */
  readonly loaded: Runtime
  readonly env: Env
  /** The explicitly chosen active profile; undefined while only the built-in fallback is active. */
  readonly worker: RunModeProfile | undefined
}): RunModeState {
  const localOnly = parseLocalOnlyConfig(input.saved, input.env)
  const hybridBoundary = boundaryMode(input.saved, input.env) === 'hybrid'
  const worker = input.worker === undefined ? null : { profile: input.worker.name, model: input.worker.model, local: isLocalProfile(input.worker) }
  // A local main agent without fully local mode (the old advisor setup) is no
  // longer one of the modes: it shows as unset so the person picks again.
  const mode: RunMode | null = worker === null ? null
    : worker.local ? (localOnly ? 'private' : null)
    : hybridBoundary ? 'hybrid' : 'cloud'
  const boundaryKey = (runtime: Runtime) => JSON.stringify([boundaryMode(runtime, input.env), runtime.boundary ?? null])
  const restartPending = localOnly !== parseLocalOnlyConfig(input.loaded, input.env)
    || boundaryKey(input.saved) !== boundaryKey(input.loaded)
    || parseAdvisorConfig({ ...input.saved }, { ...input.env }).profile !== parseAdvisorConfig({ ...input.loaded }, { ...input.env }).profile
  const lockedBy = [LOCAL_ONLY_ENV, DATA_BOUNDARY_ENV].filter(name => (input.env[name]?.trim() ?? '') !== '')
  return { mode, worker, localOnly, restartPending, lockedBy }
}

/**
 * The runtime settings after a change of mode. Private turns fully local mode
 * on; Hybrid turns the data boundary on in its own shape; leaving Hybrid takes
 * that boundary off again. The advisor is no longer part of a mode, so its
 * profile is cleared (fully local mode refused it, and a cloud lead has no use
 * for it). Everything else in `runtime` is kept.
 */
export function applyRunMode(saved: Runtime, mode: RunMode): Record<string, unknown> {
  const next: Record<string, unknown> = { ...saved }
  delete next.local_only
  if (next.data_boundary === 'hybrid') {
    delete next.data_boundary
    delete next.boundary
  }
  if (isRecord(next.advisor)) {
    const advisor = { ...next.advisor }
    delete advisor.profile
    delete advisor.model
    if (Object.keys(advisor).length === 0) delete next.advisor
    else next.advisor = advisor
  }
  if (mode === 'private') next.local_only = true
  if (mode === 'hybrid') {
    next.data_boundary = 'hybrid'
    next.boundary = { ...(isRecord(saved.boundary) ? saved.boundary : {}), ...HYBRID_BOUNDARY }
  }
  // The same checks the runtime applies on start, so a bad value is refused here.
  parseLocalOnlyConfig(next, {})
  parseAdvisorConfig(next, {})
  parseDataBoundaryConfig(next, {})
  return next
}

/**
 * Where subagents run in a mode (the Agent intelligence tiers). Bonsai answers
 * directly (thinking off) for light and balanced work, where a brief says
 * exactly what to do; smart keeps thinking. Cloud leaves every tier on the
 * main model.
 */
export function runModeTiers(mode: RunMode, local: RunModeProfile | undefined, main: RunModeProfile | undefined): AgentIntelligenceConfig {
  if (mode === 'cloud') return {}
  if (!local) throw new ValidationError('mode', `${mode} needs Bonsai set up on this machine`)
  const fast = { model: local.model, provider_profile: local.name, reasoning_effort: 'off' }
  const smart = mode === 'hybrid' && main ? { model: main.model, provider_profile: main.name } : { model: local.model, provider_profile: local.name }
  return { default: 'balanced', light: fast, balanced: fast, smart }
}

export function parseRunMode(value: unknown): RunMode {
  if (typeof value === 'string' && (RUN_MODES as readonly string[]).includes(value)) return value as RunMode
  throw new ValidationError('mode', `must be one of ${RUN_MODES.join(', ')}`, value)
}

/** Read and write the `runtime` section of the daemon's config.json. */
export interface RuntimeSettingsFile {
  read(): Record<string, unknown>
  write(runtime: Readonly<Record<string, unknown>>): void
}

/**
 * The daemon config file. Only the `runtime` section is replaced; other
 * sections and unknown keys are kept. Written to a temporary file and renamed,
 * owner-only, since the file can hold API keys.
 */
export function runtimeSettingsFile(path: string): RuntimeSettingsFile {
  const document = (): Record<string, unknown> => {
    let raw: string
    try {
      raw = readFileSync(path, 'utf8')
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return {}
      throw error
    }
    if (raw.trim() === '') return {}
    const parsed: unknown = JSON.parse(raw)
    if (!isRecord(parsed)) throw new ValidationError('config.json', 'must be a JSON object', path)
    return parsed
  }
  return {
    read() {
      const runtime = document().runtime
      return isRecord(runtime) ? { ...runtime } : {}
    },
    write(runtime) {
      const next = { ...document(), runtime }
      mkdirSync(dirname(path), { recursive: true })
      const temporary = `${path}.${process.pid}.tmp`
      writeFileSync(temporary, JSON.stringify(next, null, 2) + '\n', { mode: 0o600 })
      renameSync(temporary, path)
    },
  }
}

export function runModePayload(state: RunModeState): Record<string, unknown> {
  return {
    mode: state.mode,
    worker: state.worker,
    local_only: state.localOnly,
    restart_pending: state.restartPending,
    locked_by: [...state.lockedBy],
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
