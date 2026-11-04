// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Hybrid planning (M3.1): split work into dependent steps, give each step an
 * allowed execution location, and run the steps while guaranteeing that a
 * private value is never handed to a cloud execution. OFF by default
 * (`hybrid_plans`: off|on, env `XERXES_HYBRID_PLANS`).
 *
 * Privacy decides which locations are allowed; the model/router only chooses
 * among them. Sensitivity propagates through dependencies: a step's output is
 * private when any input it consumes (declared, or produced by a dependency)
 * is private. A task whose strongest-model steps all need private context
 * cannot be split; the caller finishes it locally or explains the limitation.
 */

import { ValidationError } from '../core/errors.js'
import type { Placement, Sensitivity, StepKind } from '../routing/types.js'
import { DataBoundaryError, type DataBoundaryMode } from '../security/boundaryTypes.js'
import { planDependencyLayers, validateExecutionPlan, type ExecutionPlan, type PlanComplexity, type PlanStep } from './planner.js'

type Environment = Readonly<Record<string, string | undefined>>

/** One named artifact a step consumes or produces, with its data class. */
export interface HybridRef {
  readonly ref: string
  readonly sensitivity: Sensitivity
}

export type HybridStep = PlanStep & {
  readonly placement: Placement
  readonly kind: StepKind
  readonly inputs: readonly HybridRef[]
  readonly produces: HybridRef
  readonly check?: string
}

export type HybridPlan = Omit<ExecutionPlan, 'steps'> & {
  readonly steps: readonly HybridStep[]
  readonly version: 1
}

/** Why a task that needs private context for its strongest-model work is not split. */
export const HYBRID_UNSPLITTABLE_REASON =
  'Cannot split without losing necessary private context: every step that needs the strongest model depends on private data. Finish locally or explain the limitation.'

const STRONG_KINDS: ReadonlySet<StepKind> = new Set<StepKind>(['reasoning', 'review', 'plan'])

/** The artifact reference a step produces by default. */
export function stepOutputRef(stepId: string): string {
  return `out:${stepId}`
}

/**
 * Whether hybrid plans are on: `XERXES_HYBRID_PLANS` wins, then
 * `runtime.hybrid_plans` ("on"/"off"), default off. Bad values throw.
 */
export function parseHybridPlansConfig(runtime: Readonly<Record<string, unknown>>, env: Environment = process.env): boolean {
  const fromEnv = env.XERXES_HYBRID_PLANS?.trim()
  if (fromEnv) {
    if (/^(1|true|yes|on)$/i.test(fromEnv)) return true
    if (/^(0|false|no|off)$/i.test(fromEnv)) return false
    throw new ValidationError('XERXES_HYBRID_PLANS', 'must be on or off', fromEnv)
  }
  const value = runtime.hybrid_plans
  if (value === undefined || value === null) return false
  if (value === true) return true
  if (value === false) return false
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase()
    if (normalized === 'on') return true
    if (normalized === 'off') return false
  }
  throw new ValidationError('hybrid_plans', 'must be "on" or "off"', value)
}

function stricter(a: Sensitivity, b: Sensitivity): Sensitivity {
  return a === 'private' || b === 'private' ? 'private' : 'shareable'
}

function defaultKindOf(step: PlanStep): StepKind {
  const text = step.action.toLowerCase()
  if (/review|verify|audit/.test(text)) return 'review'
  if (/plan|decompose/.test(text)) return 'plan'
  if (/reason|decide|derive|prove/.test(text)) return 'reasoning'
  if (/summar/.test(text)) return 'summarize'
  if (/research|search|lookup/.test(text)) return 'research'
  if (/tool|exec|read|write|run|edit|fetch|file|shell/.test(text)) return 'tool_use'
  return 'analysis'
}

/** Default external inputs: the comma-separated `inputs` argument of a step. */
function defaultInputsOf(step: PlanStep): readonly string[] {
  return (step.arguments.inputs ?? '').split(',').map(part => part.trim()).filter(part => part.length > 0)
}

export interface AssignPlacementsContext {
  readonly boundaryMode: DataBoundaryMode
  /** Data class of an external input reference (file, URL, user text). */
  readonly inputSensitivity: (ref: string) => Sensitivity
  /** Default: derived from the step action text. */
  readonly kindOf?: (step: PlanStep) => StepKind
  /** External input refs a step consumes. Default: the comma-separated `inputs` argument. */
  readonly inputsOf?: (step: PlanStep) => readonly string[]
}

/**
 * Turn a plain plan into a hybrid plan. Private data stays local: any step
 * with a private input (direct or inherited from a dependency) is placed
 * `local` and its output is private. Under boundary mode `private` every step
 * is local. Tool steps run local; reasoning, review and plan steps over only
 * shareable inputs are `either`.
 */
export function assignPlacements(plan: ExecutionPlan, ctx: AssignPlacementsContext): { plan: HybridPlan; notes: string[] } {
  validateExecutionPlan(plan)
  const kindOf = ctx.kindOf ?? defaultKindOf
  const inputsOf = ctx.inputsOf ?? defaultInputsOf
  const notes: string[] = []
  const outputSensitivity = new Map<string, Sensitivity>()
  const built = new Map<string, HybridStep>()

  if (ctx.boundaryMode === 'private') notes.push('Boundary mode is private: every step runs locally.')

  for (const layer of planDependencyLayers(plan)) {
    for (const step of layer) {
      const merged = new Map<string, Sensitivity>()
      const add = (ref: string, sensitivity: Sensitivity): void => {
        if (!ref.trim()) throw new ValidationError(`steps.${step.id}.inputs`, 'input reference cannot be empty', ref)
        merged.set(ref, stricter(merged.get(ref) ?? 'shareable', sensitivity))
      }
      for (const ref of inputsOf(step)) add(ref, ctx.inputSensitivity(ref))
      for (const dependency of step.dependencies) add(stepOutputRef(dependency), outputSensitivity.get(dependency) ?? 'private')

      const privateRefs = [...merged].filter(([, sensitivity]) => sensitivity === 'private').map(([ref]) => ref)
      const sensitivity: Sensitivity = privateRefs.length > 0 ? 'private' : 'shareable'
      const kind = kindOf(step)
      let placement: Placement
      if (privateRefs.length > 0) {
        placement = 'local'
        notes.push(`Step ${step.id} is local and its output is private: it consumes private ${privateRefs.join(', ')}.`)
      } else if (ctx.boundaryMode === 'private') {
        placement = 'local'
      } else if (kind === 'tool_use') {
        placement = 'local'
        notes.push(`Step ${step.id} is local: tool steps act on the local machine.`)
      } else {
        placement = 'either'
      }
      outputSensitivity.set(step.id, sensitivity)
      const check = step.arguments.check?.trim()
      built.set(step.id, {
        ...step,
        placement,
        kind,
        inputs: [...merged].map(([ref, itemSensitivity]) => ({ ref, sensitivity: itemSensitivity })),
        produces: { ref: stepOutputRef(step.id), sensitivity },
        ...(check ? { check } : {}),
      })
    }
  }

  const steps = plan.steps.map(step => {
    const hybrid = built.get(step.id)
    if (!hybrid) throw new Error(`Hybrid plan lost step ${step.id}`)
    return hybrid
  })
  return { plan: { ...plan, steps, version: 1 }, notes }
}

/**
 * Whether the plan can use a cloud model at all. Unsplittable when every step
 * that needs the strongest model (reasoning, review, plan) depends on private
 * data. `cloudSteps` lists steps that may run off the machine.
 */
export function analyzeSplit(plan: HybridPlan): { splittable: boolean; cloudSteps: string[]; reason: string } {
  const cloudSteps = plan.steps.filter(step => step.placement !== 'local').map(step => step.id)
  const strong = plan.steps.filter(step => STRONG_KINDS.has(step.kind))
  const dependsOnPrivate = (step: HybridStep): boolean =>
    step.produces.sensitivity === 'private' || step.inputs.some(input => input.sensitivity === 'private')
  if (strong.length > 0 && strong.every(dependsOnPrivate)) {
    return { splittable: false, cloudSteps: [], reason: HYBRID_UNSPLITTABLE_REASON }
  }
  if (strong.length === 0) {
    return { splittable: true, cloudSteps, reason: 'No step needs the strongest model; the plan can finish locally.' }
  }
  if (cloudSteps.length === 0) {
    return { splittable: true, cloudSteps, reason: 'No step is placed for cloud execution under the current boundary.' }
  }
  return { splittable: true, cloudSteps, reason: `Steps ${cloudSteps.join(', ')} can use a stronger model without private context.` }
}

interface PlanProblems {
  readonly structural: string[]
  readonly boundary: string[]
}

function inspectHybridPlan(plan: HybridPlan, mode: DataBoundaryMode): PlanProblems {
  const structural: string[] = []
  const boundary: string[] = []
  try {
    validateExecutionPlan(plan)
  } catch (error) {
    structural.push(error instanceof Error ? error.message : String(error))
    return { structural, boundary }
  }

  const byId = new Map(plan.steps.map(step => [step.id, step] as const))
  const producers = new Map<string, HybridStep>()
  for (const step of plan.steps) {
    const ref = step.produces.ref
    if (!ref.trim()) {
      structural.push(`step ${step.id} has an empty output reference`)
    } else if (producers.has(ref)) {
      structural.push(`output reference ${ref} is produced by more than one step`)
    } else {
      producers.set(ref, step)
    }
  }

  for (const step of plan.steps) {
    const effective = new Map<string, Sensitivity>()
    for (const input of step.inputs) {
      const producer = producers.get(input.ref)
      if (producer !== undefined) {
        if (!step.dependencies.includes(producer.id)) {
          structural.push(`step ${step.id} consumes ${input.ref}, produced by step ${producer.id}, which is not one of its dependencies`)
        }
        if (producer.produces.sensitivity === 'private' && input.sensitivity !== 'private') {
          boundary.push(`step ${step.id} declares ${input.ref} shareable but step ${producer.id} produces it as private`)
        }
      }
      effective.set(input.ref, stricter(effective.get(input.ref) ?? 'shareable', input.sensitivity))
    }
    for (const dependencyId of step.dependencies) {
      const dependency = byId.get(dependencyId)
      if (dependency?.produces.sensitivity === 'private') effective.set(dependency.produces.ref, 'private')
    }
    const privateRefs = [...effective].filter(([, sensitivity]) => sensitivity === 'private').map(([ref]) => ref)
    if (privateRefs.length > 0 && step.produces.sensitivity !== 'private') {
      boundary.push(`step ${step.id} consumes private ${privateRefs.join(', ')} but declares its output shareable`)
    }
    if (step.placement !== 'local') {
      if (privateRefs.length > 0) {
        boundary.push(`step ${step.id} is placed ${step.placement} but consumes private ${privateRefs.join(', ')}`)
      }
      if (step.produces.sensitivity === 'private') {
        boundary.push(`step ${step.id} is placed ${step.placement} but produces private output`)
      }
      if (mode === 'private') {
        boundary.push(`step ${step.id} is placed ${step.placement} but boundary mode private allows only local steps`)
      }
    }
  }
  return { structural, boundary }
}

/** Check a hybrid plan fails closed: graph validity plus the privacy rules above. */
export function validateHybridPlan(plan: HybridPlan, boundaryMode: DataBoundaryMode): { ok: true } | { ok: false; errors: string[] } {
  const problems = inspectHybridPlan(plan, boundaryMode)
  const errors = [...problems.structural, ...problems.boundary]
  return errors.length === 0 ? { ok: true } : { ok: false, errors }
}

export interface HybridExecution {
  readonly output: string
  readonly accepted: boolean
  readonly detail?: string
}

export interface HybridPlanPorts {
  /** Run one step at `location` with only its declared inputs. */
  execute(step: HybridStep, location: 'local' | 'cloud', inputs: ReadonlyMap<string, string>, signal?: AbortSignal): Promise<HybridExecution>
  /** Resolve an `either` step. Default: local. Anything but `'cloud'` is treated as local. */
  chooseLocation?(step: HybridStep): 'local' | 'cloud'
  /** Values of the plan's external input references. A missing one fails its step. */
  readonly externalInputs?: Readonly<Record<string, string>>
}

export type HybridStepStatus = 'succeeded' | 'rejected' | 'failed' | 'skipped'

export interface HybridStepResult {
  readonly stepId: string
  readonly status: HybridStepStatus
  readonly location?: 'local' | 'cloud'
  readonly output?: string
  readonly detail?: string
}

export interface HybridRunResult {
  /** Every step ran and was accepted, and the run was not aborted. */
  readonly accepted: boolean
  readonly aborted: boolean
  /** One result per step, in plan order. */
  readonly results: readonly HybridStepResult[]
}

/**
 * Run a hybrid plan layer by layer (steps within a layer in parallel). Only a
 * step's declared inputs reach `execute`. A private value is never given to a
 * cloud execution: a plan that would do so is refused with
 * `DATA_BOUNDARY_DENIED` before anything runs. Dependents of a failed or
 * unaccepted step are skipped. An aborted signal stops further steps.
 */
export async function runHybridPlan(plan: HybridPlan, ports: HybridPlanPorts, signal?: AbortSignal): Promise<HybridRunResult> {
  const problems = inspectHybridPlan(plan, 'hybrid')
  if (problems.boundary.length > 0) {
    throw new DataBoundaryError('DATA_BOUNDARY_DENIED', `Hybrid plan refused: ${problems.boundary.join('; ')}`)
  }
  if (problems.structural.length > 0) {
    throw new ValidationError('plan', problems.structural.join('; '), plan.id)
  }

  const byId = new Map(plan.steps.map(step => [step.id, step] as const))
  const results = new Map<string, HybridStepResult>()
  const outputs = new Map<string, { readonly value: string; readonly sensitivity: Sensitivity }>()
  let aborted = false

  for (const layer of planDependencyLayers(plan)) {
    const steps = layer.map(item => {
      const step = byId.get(item.id)
      if (!step) throw new Error(`Hybrid plan lost step ${item.id}`)
      return step
    })
    if (signal?.aborted) {
      aborted = true
      for (const step of steps) results.set(step.id, { stepId: step.id, status: 'skipped', detail: 'aborted' })
      continue
    }

    const runnable: { step: HybridStep; location: 'local' | 'cloud'; inputs: Map<string, string> }[] = []
    for (const step of steps) {
      const blocker = step.dependencies.find(dependency => results.get(dependency)?.status !== 'succeeded')
      if (blocker !== undefined) {
        results.set(step.id, {
          stepId: step.id,
          status: 'skipped',
          detail: `dependency ${blocker} did not succeed (${results.get(blocker)?.status ?? 'unknown'})`,
        })
        continue
      }
      const location = resolveLocation(step, ports)
      const privateNow = step.produces.sensitivity === 'private'
        || step.inputs.some(input => input.sensitivity === 'private')
        || step.dependencies.some(dependency => byId.get(dependency)?.produces.sensitivity === 'private')
      if (location === 'cloud' && privateNow) {
        throw new DataBoundaryError('DATA_BOUNDARY_DENIED', `Step ${step.id} would run in the cloud with private data; refused.`)
      }
      const inputs = new Map<string, string>()
      let missing: string | undefined
      for (const input of step.inputs) {
        const produced = outputs.get(input.ref)
        const value = produced !== undefined ? produced.value : ports.externalInputs?.[input.ref]
        if (value === undefined) {
          missing = input.ref
          break
        }
        inputs.set(input.ref, value)
      }
      if (missing !== undefined) {
        results.set(step.id, { stepId: step.id, status: 'failed', detail: `missing input ${missing}` })
        continue
      }
      runnable.push({ step, location, inputs })
    }

    await Promise.all(runnable.map(async ({ step, location, inputs }) => {
      try {
        const execution = await ports.execute(step, location, inputs, signal)
        if (execution.accepted) {
          outputs.set(step.produces.ref, { value: execution.output, sensitivity: step.produces.sensitivity })
        }
        results.set(step.id, {
          stepId: step.id,
          status: execution.accepted ? 'succeeded' : 'rejected',
          location,
          output: execution.output,
          ...(execution.detail !== undefined ? { detail: execution.detail } : {}),
        })
      } catch (error) {
        if (error instanceof DataBoundaryError) throw error
        results.set(step.id, {
          stepId: step.id,
          status: 'failed',
          location,
          detail: signal?.aborted ? 'aborted' : error instanceof Error ? error.message : String(error),
        })
      }
    }))
    if (signal?.aborted) aborted = true
  }

  const ordered = plan.steps.map(step => results.get(step.id) ?? { stepId: step.id, status: 'skipped' as const, detail: 'not run' })
  return {
    accepted: !aborted && ordered.every(result => result.status === 'succeeded'),
    aborted,
    results: ordered,
  }
}

function resolveLocation(step: HybridStep, ports: HybridPlanPorts): 'local' | 'cloud' {
  if (step.placement === 'local') return 'local'
  if (step.placement === 'cloud') return 'cloud'
  return ports.chooseLocation?.(step) === 'cloud' ? 'cloud' : 'local'
}

/** External input references of a plan (not produced by any step) with their strictest declared sensitivity. */
export function externalInputRefs(plan: HybridPlan): Map<string, Sensitivity> {
  const produced = new Set(plan.steps.map(step => step.produces.ref))
  const refs = new Map<string, Sensitivity>()
  for (const step of plan.steps) {
    for (const input of step.inputs) {
      if (produced.has(input.ref)) continue
      refs.set(input.ref, stricter(refs.get(input.ref) ?? 'shareable', input.sensitivity))
    }
  }
  return refs
}

/**
 * Point a plan's external inputs at new references. Tighten only: sensitivity
 * and placement can become stricter (a step that now consumes private data
 * becomes local and its output private) but are never loosened.
 */
export function rebindHybridPlan(
  plan: HybridPlan,
  bindings: ReadonlyMap<string, HybridRef>,
): HybridPlan {
  const producerOf = new Map(plan.steps.map(step => [step.produces.ref, step.id] as const))
  const outputSensitivity = new Map<string, Sensitivity>()
  const rebuilt = new Map<string, HybridStep>()
  for (const layer of planDependencyLayers(plan)) {
    for (const item of layer) {
      const step = plan.steps.find(candidate => candidate.id === item.id)
      if (!step) throw new Error(`Hybrid plan lost step ${item.id}`)
      const merged = new Map<string, Sensitivity>()
      const add = (ref: string, sensitivity: Sensitivity): void => {
        merged.set(ref, stricter(merged.get(ref) ?? 'shareable', sensitivity))
      }
      let anyPrivate = false
      for (const input of step.inputs) {
        const producerId = producerOf.get(input.ref)
        if (producerId !== undefined) {
          add(input.ref, stricter(input.sensitivity, outputSensitivity.get(producerId) ?? 'private'))
        } else {
          const binding = bindings.get(input.ref)
          if (binding) add(binding.ref, stricter(input.sensitivity, binding.sensitivity))
          else add(input.ref, input.sensitivity)
        }
      }
      for (const dependency of step.dependencies) {
        if (outputSensitivity.get(dependency) === 'private') anyPrivate = true
      }
      for (const sensitivity of merged.values()) if (sensitivity === 'private') anyPrivate = true
      const sensitivity: Sensitivity = anyPrivate || step.produces.sensitivity === 'private' ? 'private' : 'shareable'
      outputSensitivity.set(step.id, sensitivity)
      rebuilt.set(step.id, {
        ...step,
        placement: anyPrivate ? 'local' : step.placement,
        inputs: [...merged].map(([ref, itemSensitivity]) => ({ ref, sensitivity: itemSensitivity })),
        produces: { ref: step.produces.ref, sensitivity },
      })
    }
  }
  return {
    ...plan,
    steps: plan.steps.map(step => {
      const next = rebuilt.get(step.id)
      if (!next) throw new Error(`Hybrid plan lost step ${step.id}`)
      return next
    }),
  }
}

const PLACEMENTS: ReadonlySet<string> = new Set(['local', 'cloud', 'either'])
const KINDS: ReadonlySet<string> = new Set(['plan', 'tool_use', 'analysis', 'reasoning', 'review', 'summarize', 'research'])
const SENSITIVITIES: ReadonlySet<string> = new Set(['private', 'shareable'])
const COMPLEXITIES: ReadonlySet<string> = new Set(['low', 'medium', 'high'])
const MAX_STEPS = 256
const MAX_INPUTS = 64
const MAX_TEXT = 20_000

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function text(value: unknown, path: string, allowEmpty = false): string {
  if (typeof value !== 'string') throw new ValidationError(path, 'must be a string')
  if (value.length > MAX_TEXT) throw new ValidationError(path, `must be at most ${MAX_TEXT} characters`)
  if (!allowEmpty && !value.trim()) throw new ValidationError(path, 'cannot be empty')
  return value
}

function sensitivityOf(value: unknown, path: string): Sensitivity {
  if (typeof value !== 'string' || !SENSITIVITIES.has(value)) throw new ValidationError(path, 'must be "private" or "shareable"', value)
  return value as Sensitivity
}

function refOf(value: unknown, path: string): HybridRef {
  if (!isRecord(value)) throw new ValidationError(path, 'must be an object with ref and sensitivity')
  return { ref: text(value.ref, `${path}.ref`), sensitivity: sensitivityOf(value.sensitivity, `${path}.sensitivity`) }
}

/** Strictly parse untrusted JSON into a validated hybrid plan; throws a ValidationError naming the field. */
export function parseHybridPlan(value: unknown): HybridPlan {
  if (!isRecord(value)) throw new ValidationError('plan', 'must be an object')
  if (value.version !== 1) throw new ValidationError('plan.version', 'unsupported hybrid plan version', value.version)
  const complexity = value.complexity
  if (typeof complexity !== 'string' || !COMPLEXITIES.has(complexity)) {
    throw new ValidationError('plan.complexity', 'must be low, medium or high', complexity)
  }
  const minutes = value.estimatedMinutes
  if (typeof minutes !== 'number' || !Number.isFinite(minutes) || minutes < 0) {
    throw new ValidationError('plan.estimatedMinutes', 'must be a non-negative number', minutes)
  }
  if (!Array.isArray(value.steps) || value.steps.length === 0 || value.steps.length > MAX_STEPS) {
    throw new ValidationError('plan.steps', `must be an array of 1-${MAX_STEPS} steps`)
  }
  const steps: HybridStep[] = value.steps.map((raw: unknown, index: number): HybridStep => {
    const path = `plan.steps[${index}]`
    if (!isRecord(raw)) throw new ValidationError(path, 'must be an object')
    if (typeof raw.placement !== 'string' || !PLACEMENTS.has(raw.placement)) {
      throw new ValidationError(`${path}.placement`, 'must be local, cloud or either', raw.placement)
    }
    if (typeof raw.kind !== 'string' || !KINDS.has(raw.kind)) throw new ValidationError(`${path}.kind`, 'unknown step kind', raw.kind)
    if (!isRecord(raw.arguments)) throw new ValidationError(`${path}.arguments`, 'must be an object of strings')
    const args: Record<string, string> = {}
    for (const [key, argument] of Object.entries(raw.arguments)) args[key] = text(argument, `${path}.arguments.${key}`, true)
    if (!Array.isArray(raw.dependencies)) throw new ValidationError(`${path}.dependencies`, 'must be an array of step ids')
    const dependencies = raw.dependencies.map((dependency: unknown, at: number) => text(dependency, `${path}.dependencies[${at}]`))
    if (!Array.isArray(raw.inputs) || raw.inputs.length > MAX_INPUTS) {
      throw new ValidationError(`${path}.inputs`, `must be an array of at most ${MAX_INPUTS} references`)
    }
    const inputs = raw.inputs.map((input: unknown, at: number) => refOf(input, `${path}.inputs[${at}]`))
    const agentId = raw.agentId === undefined ? undefined : text(raw.agentId, `${path}.agentId`)
    const check = raw.check === undefined ? undefined : text(raw.check, `${path}.check`)
    return {
      id: text(raw.id, `${path}.id`),
      action: text(raw.action, `${path}.action`),
      description: text(raw.description, `${path}.description`, true),
      arguments: args,
      dependencies,
      placement: raw.placement as Placement,
      kind: raw.kind as StepKind,
      inputs,
      produces: refOf(raw.produces, `${path}.produces`),
      ...(agentId !== undefined ? { agentId } : {}),
      ...(check !== undefined ? { check } : {}),
    }
  })
  const plan: HybridPlan = {
    id: text(value.id, 'plan.id'),
    objective: text(value.objective, 'plan.objective'),
    complexity: complexity as PlanComplexity,
    estimatedMinutes: minutes,
    steps,
    version: 1,
  }
  const problems = inspectHybridPlan(plan, 'hybrid')
  const errors = [...problems.structural, ...problems.boundary]
  if (errors.length > 0) throw new ValidationError('plan', errors.join('; '))
  return plan
}
