// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SkillCandidate, ToolArguments, ToolCallEvent } from './model.js'

export type VerificationStepKind = 'args_subset' | 'call_count' | 'sequence_position' | 'status_success' | 'tool_called'

/** One transparent assertion in a repeatable skill verification recipe. */
export interface VerificationStep {
  readonly argsRequired?: ToolArguments
  readonly expectedCount?: number
  readonly kind: VerificationStepKind | string
  readonly message: string
  readonly position?: number
  readonly toolName?: string
}

export interface VerificationFailure {
  readonly index: number
  readonly reason: string
}

export interface VerificationResult {
  readonly failedSteps: readonly VerificationFailure[]
  readonly passed: boolean
  readonly passedSteps: readonly number[]
}

export interface SkillVerifierOptions {
  /** Preserve Python-compatible key-only argument checks unless hosts explicitly opt into value equality. */
  readonly compareArgumentValues?: boolean
}

/** Builds and evaluates deterministic recipes from observed successful calls. */
export class SkillVerifier {
  private readonly compareArgumentValues: boolean

  constructor(options: SkillVerifierOptions = {}) {
    this.compareArgumentValues = options.compareArgumentValues ?? false
  }

  generate(candidate: SkillCandidate): VerificationStep[] {
    const successful = candidate.successfulEvents
    const steps: VerificationStep[] = [{
      kind: 'call_count',
      expectedCount: successful.length,
      message: 'expects ' + successful.length + ' successful tool calls',
    }]
    for (let position = 0; position < successful.length; position += 1) {
      const event = successful[position]
      if (!event) {
        continue
      }
      steps.push({
        kind: 'sequence_position',
        toolName: event.toolName,
        position,
        message: 'position ' + position + ' should call ' + event.toolName,
      })
      const arguments_ = firstArguments(event.arguments, 3)
      if (Object.keys(arguments_).length) {
        steps.push({
          kind: 'args_subset',
          toolName: event.toolName,
          position,
          argsRequired: arguments_,
          message: event.toolName + ' expects keys ' + Object.keys(arguments_).join(', '),
        })
      }
    }
    return steps
  }

  verify(steps: readonly VerificationStep[], candidate: SkillCandidate): VerificationResult {
    const observed = candidate.successfulEvents
    const passedSteps: number[] = []
    const failedSteps: VerificationFailure[] = []
    for (let index = 0; index < steps.length; index += 1) {
      const step = steps[index]
      if (!step) {
        continue
      }
      const result = this.evaluateStep(step, observed)
      if (result.ok) {
        passedSteps.push(index)
      } else {
        failedSteps.push({ index, reason: result.reason })
      }
    }
    return { passed: failedSteps.length === 0, passedSteps, failedSteps }
  }

  private evaluateStep(
    step: VerificationStep,
    observed: readonly ToolCallEvent[],
  ): { readonly ok: boolean; readonly reason: string } {
    switch (step.kind) {
      case 'call_count':
        if (step.expectedCount !== undefined && observed.length !== step.expectedCount) {
          return failure('expected ' + step.expectedCount + ' successful calls, got ' + observed.length)
        }
        return success()
      case 'tool_called':
        if (!step.toolName || !observed.some(event => event.toolName === step.toolName)) {
          return failure('tool ' + String(step.toolName) + ' was never called')
        }
        return success()
      case 'sequence_position': {
        const event = eventAt(step.position, observed)
        if (!event) {
          return failure('position ' + String(step.position) + ' not in observed sequence')
        }
        if (event.toolName !== step.toolName) {
          return failure('expected ' + String(step.toolName) + ' at pos ' + step.position + ', got ' + event.toolName)
        }
        return success()
      }
      case 'args_subset': {
        const event = eventAt(step.position, observed)
        if (!event) {
          return failure('position out of range')
        }
        const required = step.argsRequired ?? {}
        for (const [key, value] of Object.entries(required)) {
          if (!Object.hasOwn(event.arguments, key)) {
            return failure('missing required arg key ' + key)
          }
          if (this.compareArgumentValues && !valuesEqual(event.arguments[key], value)) {
            return failure('argument value did not match for key ' + key)
          }
        }
        return success()
      }
      case 'status_success': {
        const event = eventAt(step.position, observed)
        if (!event) {
          return failure('position out of range')
        }
        return event.status === 'success' ? success() : failure('non-success status')
      }
      default:
        return failure('unknown step kind ' + step.kind)
    }
  }
}

function eventAt(position: number | undefined, events: readonly ToolCallEvent[]): ToolCallEvent | undefined {
  if (position === undefined || !Number.isInteger(position) || position < 0) {
    return undefined
  }
  return events[position]
}

function firstArguments(arguments_: ToolArguments, count: number): ToolArguments {
  const result: Record<string, unknown> = {}
  for (const key of Object.keys(arguments_).slice(0, count)) {
    result[key] = arguments_[key]
  }
  return result
}

function valuesEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) {
    return true
  }
  if (Array.isArray(left) && Array.isArray(right)) {
    return left.length === right.length && left.every((value, index) => valuesEqual(value, right[index]))
  }
  if (isRecord(left) && isRecord(right)) {
    const leftKeys = Object.keys(left).sort()
    const rightKeys = Object.keys(right).sort()
    return leftKeys.length === rightKeys.length
      && leftKeys.every((key, index) => key === rightKeys[index] && valuesEqual(left[key], right[key]))
  }
  return false
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function failure(reason: string): { readonly ok: false; readonly reason: string } {
  return { ok: false, reason }
}

function success(): { readonly ok: true; readonly reason: string } {
  return { ok: true, reason: '' }
}
