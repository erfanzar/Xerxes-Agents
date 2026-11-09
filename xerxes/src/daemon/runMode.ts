// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Private, Hybrid and Cloud: the product's three ways to run a task, read from
 * and written to the settings that already govern them.
 *
 *   private  the worker is a local profile and `runtime.local_only` is on, so
 *            nothing but this machine is reachable during a run
 *   hybrid   the worker is a local profile; `runtime.advisor.profile` names the
 *            cloud advisor, and `runtime.escalation.max_usd` caps cloud spend
 *   cloud    the worker is a cloud profile
 *
 * The worker is the active provider profile. Fully local mode and the advisor
 * are read when the daemon starts, so changing either needs a restart; the
 * state says when one is pending.
 */
import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { ADVISOR_PROFILE_ENV, parseAdvisorConfig } from '../runtime/advisorBudget.js'
import { parseEscalationConfig } from '../runtime/escalationPolicy.js'
import { LOCAL_ONLY_ENV, parseLocalOnlyConfig } from '../runtime/localOnly.js'
import type { DataTrust } from '../security/boundaryTypes.js'
import { deriveDataTrust } from '../security/dataBoundary.js'

export type RunMode = 'private' | 'hybrid' | 'cloud'
export const RUN_MODES: readonly RunMode[] = ['private', 'hybrid', 'cloud']

type Env = Readonly<Record<string, string | undefined>>
type Runtime = Readonly<Record<string, unknown>>

export interface RunModeProfile {
  readonly name: string
  readonly model: string
  readonly base_url?: string
  readonly data_trust?: DataTrust
}

export interface RunModeState {
  /** Null until a worker profile has been chosen. */
  readonly mode: RunMode | null
  readonly worker: { readonly profile: string; readonly model: string; readonly local: boolean } | null
  readonly advisor: { readonly profile: string; readonly model: string | null; readonly found: boolean } | null
  readonly localOnly: boolean
  readonly budgetUsd: number
  /** The saved settings differ from the ones this daemon started with. */
  readonly restartPending: boolean
  /** Environment variables that override the saved settings. */
  readonly lockedBy: readonly string[]
}

export function isLocalProfile(profile: Pick<RunModeProfile, 'base_url' | 'data_trust'>): boolean {
  return deriveDataTrust(profile) === 'local'
}

/** The mode the saved settings and the chosen worker describe. */
export function readRunMode(input: {
  readonly saved: Runtime
  /** The runtime settings this daemon process started with. */
  readonly loaded: Runtime
  readonly env: Env
  /** The explicitly chosen active profile; undefined while only the built-in fallback is active. */
  readonly worker: RunModeProfile | undefined
  readonly profile: (name: string) => RunModeProfile | undefined
}): RunModeState {
  const localOnly = parseLocalOnlyConfig(input.saved, input.env)
  const advisorConfig = parseAdvisorConfig({ ...input.saved }, { ...input.env })
  const loadedAdvisor = parseAdvisorConfig({ ...input.loaded }, { ...input.env })
  const budgetUsd = parseEscalationConfig({ ...input.saved }, { ...input.env }).maxUsd
  const worker = input.worker === undefined ? null : { profile: input.worker.name, model: input.worker.model, local: isLocalProfile(input.worker) }
  const advisorProfile = advisorConfig.profile === undefined ? undefined : input.profile(advisorConfig.profile)
  const advisor = advisorConfig.profile === undefined ? null : {
    profile: advisorConfig.profile,
    model: advisorConfig.model ?? advisorProfile?.model ?? null,
    found: advisorProfile !== undefined,
  }
  const mode: RunMode | null = worker === null ? null : !worker.local ? 'cloud' : localOnly ? 'private' : 'hybrid'
  const restartPending = localOnly !== parseLocalOnlyConfig(input.loaded, input.env)
    || advisorConfig.profile !== loadedAdvisor.profile
    || advisorConfig.model !== loadedAdvisor.model
  const lockedBy = [LOCAL_ONLY_ENV, ADVISOR_PROFILE_ENV].filter(name => (input.env[name]?.trim() ?? '') !== '')
  return { mode, worker, advisor, localOnly, budgetUsd, restartPending, lockedBy }
}

export interface RunModeChange {
  readonly mode: RunMode
  /** A cloud profile name; null removes the advisor; absent keeps the saved one. */
  readonly advisorProfile?: string | null
  readonly budgetUsd?: number
}

/**
 * The runtime settings after a change of mode. Leaving Private turns fully
 * local mode off; the advisor stays saved in Private (fully local mode refuses
 * it anyway) so Hybrid gets it back.
 */
export function applyRunMode(saved: Runtime, change: RunModeChange): Record<string, unknown> {
  const next: Record<string, unknown> = { ...saved }
  if (change.mode === 'private') next.local_only = true
  else delete next.local_only
  if (change.advisorProfile !== undefined) {
    const advisor: Record<string, unknown> = isRecord(saved.advisor) ? { ...saved.advisor } : {}
    if (change.advisorProfile === null) delete advisor.profile
    else if (advisor.profile !== change.advisorProfile) {
      advisor.profile = change.advisorProfile
      // A model chosen for the previous advisor belongs to that provider.
      delete advisor.model
    }
    if (Object.keys(advisor).length === 0) delete next.advisor
    else next.advisor = advisor
  }
  if (change.budgetUsd !== undefined) {
    next.escalation = { ...(isRecord(saved.escalation) ? saved.escalation : {}), max_usd: change.budgetUsd }
  }
  // The same checks the runtime applies on start, so a bad value is refused here.
  parseLocalOnlyConfig(next, {})
  parseAdvisorConfig(next, {})
  parseEscalationConfig(next, {})
  return next
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
    advisor: state.advisor,
    local_only: state.localOnly,
    budget_usd: state.budgetUsd,
    restart_pending: state.restartPending,
    locked_by: [...state.lockedBy],
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
