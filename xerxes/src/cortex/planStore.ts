// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Durable store for reviewed hybrid plans (M3.1) so a plan can re-run on new
 * inputs. Files live at `<dir>/<id>.json` (mode 0600, directory 0700). Only a
 * plan saved with `reviewed: true` can be prepared for a re-run. Everything
 * read back is validated; a malformed or unknown-version file is refused, never
 * repaired or guessed at.
 */

import { chmod, mkdir, readdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { xerxesHome } from '../daemon/paths.js'
import {
  externalInputRefs,
  parseHybridPlan,
  rebindHybridPlan,
  validateHybridPlan,
  type HybridPlan,
  type HybridRef,
} from './hybridPlan.js'

export const PLAN_ID_PATTERN = /^[a-z0-9-]{1,64}$/
const MAX_NAME = 200

export interface PlanStoreOptions {
  /** Default: `<xerxesHome()>/plans`. */
  readonly dir?: string
  readonly now?: () => Date
}

export interface SavePlanOptions {
  readonly reviewed: boolean
  readonly name?: string
}

export interface StoredPlan {
  readonly id: string
  readonly name?: string
  readonly reviewed: boolean
  readonly savedAt: string
  readonly plan: HybridPlan
}

export interface PlanSummary {
  readonly id: string
  readonly name?: string
  readonly objective: string
  readonly reviewed: boolean
  readonly savedAt: string
  readonly steps: number
}

export type RerunErrorCode = 'not_found' | 'not_reviewed' | 'missing_input' | 'unknown_input' | 'invalid'

export type RerunResult =
  | { readonly ok: true; readonly plan: HybridPlan; readonly rebound: Readonly<Record<string, string>> }
  | { readonly ok: false; readonly code: RerunErrorCode; readonly error: string; readonly refs?: readonly string[] }

interface ScanResult {
  readonly plans: StoredPlan[]
  readonly invalid: { readonly id: string; readonly error: string }[]
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isMissingFile(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function assertPlanId(id: string): void {
  if (typeof id !== 'string' || !PLAN_ID_PATTERN.test(id)) {
    throw new ValidationError('id', 'plan id must match [a-z0-9-]{1,64}', id)
  }
}

function parseRecord(raw: unknown, expectedId: string): StoredPlan {
  if (!isRecord(raw)) throw new ValidationError('record', 'must be an object')
  if (raw.version !== 1) throw new ValidationError('record.version', 'unsupported plan file version', raw.version)
  if (raw.id !== expectedId) throw new ValidationError('record.id', 'does not match the file name', raw.id)
  if (typeof raw.reviewed !== 'boolean') throw new ValidationError('record.reviewed', 'must be true or false', raw.reviewed)
  if (typeof raw.savedAt !== 'string' || Number.isNaN(Date.parse(raw.savedAt))) {
    throw new ValidationError('record.savedAt', 'must be an ISO timestamp', raw.savedAt)
  }
  let name: string | undefined
  if (raw.name !== undefined) {
    if (typeof raw.name !== 'string' || raw.name.length > MAX_NAME) {
      throw new ValidationError('record.name', `must be a string of at most ${MAX_NAME} characters`, raw.name)
    }
    name = raw.name
  }
  const plan = parseHybridPlan(raw.plan)
  return { id: expectedId, reviewed: raw.reviewed, savedAt: raw.savedAt, plan, ...(name !== undefined ? { name } : {}) }
}

export class PlanStore {
  readonly dir: string
  private readonly now: () => Date

  constructor(options: PlanStoreOptions = {}) {
    this.dir = options.dir ?? join(xerxesHome(), 'plans')
    this.now = options.now ?? (() => new Date())
  }

  private pathFor(id: string): string {
    assertPlanId(id)
    return join(this.dir, `${id}.json`)
  }

  /** Validate and write a plan under its own id (replacing any earlier save). Returns the stored record. */
  async save(plan: HybridPlan, options: SavePlanOptions): Promise<StoredPlan> {
    const validated = parseHybridPlan(JSON.parse(JSON.stringify(plan)))
    assertPlanId(validated.id)
    if (typeof options.reviewed !== 'boolean') throw new ValidationError('reviewed', 'must be true or false', options.reviewed)
    const name = options.name?.trim()
    if (name !== undefined && (name.length === 0 || name.length > MAX_NAME)) {
      throw new ValidationError('name', `must be 1-${MAX_NAME} characters`, options.name)
    }
    const stored: StoredPlan = {
      id: validated.id,
      reviewed: options.reviewed,
      savedAt: this.now().toISOString(),
      plan: validated,
      ...(name !== undefined ? { name } : {}),
    }
    await mkdir(this.dir, { recursive: true, mode: 0o700 })
    await chmod(this.dir, 0o700)
    const path = this.pathFor(stored.id)
    const temporary = `${path}.${process.pid}.${Date.now()}.tmp`
    const body = { version: 1, id: stored.id, ...(stored.name !== undefined ? { name: stored.name } : {}), reviewed: stored.reviewed, savedAt: stored.savedAt, plan: stored.plan }
    try {
      await writeFile(temporary, `${JSON.stringify(body, null, 2)}\n`, { mode: 0o600 })
      await chmod(temporary, 0o600)
      await rename(temporary, path)
    } catch (error) {
      await rm(temporary, { force: true })
      throw error
    }
    return stored
  }

  /** The stored plan, `undefined` when none exists. A malformed file throws a ValidationError. */
  async load(id: string): Promise<StoredPlan | undefined> {
    const path = this.pathFor(id)
    let text: string
    try {
      text = await readFile(path, 'utf8')
    } catch (error) {
      if (isMissingFile(error)) return undefined
      throw error
    }
    let raw: unknown
    try {
      raw = JSON.parse(text)
    } catch (error) {
      throw new ValidationError('record', `plan file ${id} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`)
    }
    return parseRecord(raw, id)
  }

  private async scan(): Promise<ScanResult> {
    let names: string[]
    try {
      names = await readdir(this.dir)
    } catch (error) {
      if (isMissingFile(error)) return { plans: [], invalid: [] }
      throw error
    }
    const plans: StoredPlan[] = []
    const invalid: { id: string; error: string }[] = []
    for (const name of names.sort()) {
      if (!name.endsWith('.json')) continue
      const id = name.slice(0, -'.json'.length)
      try {
        const stored = await this.load(id)
        if (stored) plans.push(stored)
      } catch (error) {
        if (!(error instanceof ValidationError)) throw error
        invalid.push({ id, error: error.message })
      }
    }
    return { plans, invalid }
  }

  /** Valid saved plans, oldest id first. Files that fail validation are reported by `listInvalid`. */
  async list(): Promise<PlanSummary[]> {
    const { plans } = await this.scan()
    return plans.map(stored => ({
      id: stored.id,
      objective: stored.plan.objective,
      reviewed: stored.reviewed,
      savedAt: stored.savedAt,
      steps: stored.plan.steps.length,
      ...(stored.name !== undefined ? { name: stored.name } : {}),
    }))
  }

  /** Files in the store that were refused, so a bad file is visible rather than silently dropped. */
  async listInvalid(): Promise<{ readonly id: string; readonly error: string }[]> {
    return (await this.scan()).invalid
  }

  /** Delete a saved plan. Returns whether one existed. */
  async remove(id: string): Promise<boolean> {
    const path = this.pathFor(id)
    try {
      await readFile(path)
    } catch (error) {
      if (isMissingFile(error)) return false
      throw error
    }
    await rm(path)
    return true
  }

  /**
   * Re-run a reviewed plan on new inputs. `newInputs` maps each of the plan's
   * external input references to its replacement reference. Every external
   * input must be supplied. Sensitivity never loosens: a replacement keeps the
   * reference's declared class, and the result is revalidated.
   */
  async prepareRerun(id: string, newInputs: Readonly<Record<string, string>>): Promise<RerunResult> {
    let stored: StoredPlan | undefined
    try {
      stored = await this.load(id)
    } catch (error) {
      if (error instanceof ValidationError) return { ok: false, code: 'invalid', error: error.message }
      throw error
    }
    if (!stored) return { ok: false, code: 'not_found', error: `No saved plan ${id}` }
    if (!stored.reviewed) {
      return { ok: false, code: 'not_reviewed', error: `Plan ${id} has not been reviewed; review it before re-running.` }
    }

    const required = externalInputRefs(stored.plan)
    const missing = [...required.keys()].filter(ref => !Object.hasOwn(newInputs, ref))
    if (missing.length > 0) {
      return { ok: false, code: 'missing_input', error: `Missing input(s): ${missing.join(', ')}`, refs: missing }
    }
    const unknown = Object.keys(newInputs).filter(ref => !required.has(ref))
    if (unknown.length > 0) {
      return { ok: false, code: 'unknown_input', error: `Plan ${id} has no input(s): ${unknown.join(', ')}`, refs: unknown }
    }
    const bindings = new Map<string, HybridRef>()
    for (const [ref, sensitivity] of required) {
      const replacement = newInputs[ref]
      if (typeof replacement !== 'string' || !replacement.trim()) {
        return { ok: false, code: 'invalid', error: `Input ${ref} needs a non-empty replacement` }
      }
      bindings.set(ref, { ref: replacement, sensitivity })
    }

    const plan = rebindHybridPlan(stored.plan, bindings)
    const verdict = validateHybridPlan(plan, 'hybrid')
    if (!verdict.ok) return { ok: false, code: 'invalid', error: verdict.errors.join('; ') }
    const rebound: Record<string, string> = {}
    for (const [ref, binding] of bindings) rebound[ref] = binding.ref
    return { ok: true, plan, rebound }
  }
}
