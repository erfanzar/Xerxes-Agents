// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import {
  EscalationController,
  parseEscalationConfig,
  type EscalationConfig,
  type EscalationOptions,
} from '../src/runtime/escalationPolicy.js'
import type { CheckFinding } from '../src/runtime/resultChecks.js'

function errors(count: number): CheckFinding[] {
  return Array.from({ length: count }, (_, index) => ({
    code: 'csv_ragged' as const,
    severity: 'error' as const,
    path: `out${index}.csv`,
    message: `row ${index} is ragged`,
  }))
}

function warnings(count: number): CheckFinding[] {
  return Array.from({ length: count }, (_, index) => ({
    code: 'unverified_claim' as const,
    severity: 'warn' as const,
    path: `w${index}`,
    message: 'claim not verified',
  }))
}

const ON: EscalationConfig = {
  mode: 'on',
  maxLocalRetries: 2,
  maxModelChanges: 1,
  maxAdvisorCalls: 2,
  maxUsd: 1,
}

function controller(overrides: Partial<EscalationOptions> = {}, config: Partial<EscalationConfig> = {}): EscalationController {
  return new EscalationController({
    config: { ...ON, ...config },
    boundaryMode: 'off',
    touchesPrivate: false,
    advisorAvailable: true,
    alternateModelAvailable: true,
    ...overrides,
  })
}

describe('parseEscalationConfig', () => {
  test('defaults to off with the documented limits', () => {
    expect(parseEscalationConfig({}, {})).toEqual({
      mode: 'off',
      maxLocalRetries: 2,
      maxModelChanges: 1,
      maxAdvisorCalls: 2,
      maxUsd: 1,
    })
  })

  test('reads the file and lets the env var win for mode', () => {
    const runtime = { escalation: { mode: 'on', max_local_retries: 4, max_usd: 2.5 } }
    expect(parseEscalationConfig(runtime, {})).toMatchObject({ mode: 'on', maxLocalRetries: 4, maxUsd: 2.5 })
    expect(parseEscalationConfig(runtime, { XERXES_ESCALATION: 'off' }).mode).toBe('off')
    expect(parseEscalationConfig({}, { XERXES_ESCALATION: 'ON' }).mode).toBe('on')
  })

  test('rejects bad values and unknown keys, naming the key', () => {
    const fail = (runtime: Record<string, unknown>, env: Record<string, string> = {}): ValidationError => {
      try {
        parseEscalationConfig(runtime, env)
      } catch (error) {
        expect(error).toBeInstanceOf(ValidationError)
        return error as ValidationError
      }
      throw new Error('expected a ValidationError')
    }
    expect(fail({ escalation: { mode: 'maybe' } }).field).toBe('escalation.mode')
    expect(fail({}, { XERXES_ESCALATION: 'yes' }).field).toBe('XERXES_ESCALATION')
    expect(fail({ escalation: { max_local_retries: 1.5 } }).field).toBe('escalation.max_local_retries')
    expect(fail({ escalation: { max_model_changes: -1 } }).field).toBe('escalation.max_model_changes')
    expect(fail({ escalation: { max_usd: 0 } }).field).toBe('escalation.max_usd')
    expect(fail({ escalation: { max_usd: '1' } }).field).toBe('escalation.max_usd')
    expect(fail({ escalation: { retries: 3 } }).field).toBe('escalation.retries')
    expect(fail({ escalation: 'on' }).field).toBe('escalation')
  })
})

describe('EscalationController', () => {
  test('off mode never escalates: accept when clean, stop when not', () => {
    const clean = controller({}, { mode: 'off' })
    expect(clean.next({ findings: [] })).toEqual({ action: 'accept' })

    const failing = controller({}, { mode: 'off' })
    const step = failing.next({ findings: errors(1) })
    expect(step.action).toBe('stop')
    if (step.action === 'stop') {
      expect(step.reason).toBe('escalation off')
      expect(step.findings).toHaveLength(1)
    }
    expect(failing.summary()).toMatchObject({ localRetries: 0, modelChanges: 0, advisorCalls: 0 })
  })

  test('accepts clean findings', () => {
    const c = controller()
    expect(c.next({ findings: [], costUsd: 0.01 })).toEqual({ action: 'accept' })
    expect(c.summary()).toMatchObject({ attempts: 1, lastAction: 'accept', spentUsd: 0.01 })
  })

  test('warn-only findings are accepted, never escalated', () => {
    const c = controller()
    expect(c.next({ findings: warnings(3) })).toEqual({ action: 'accept' })
  })

  test('walks the full ladder: retry_local, change_model, ask_advisor, stop', () => {
    const c = controller()
    const actions: string[] = []
    // Each attempt strictly improves so the no-improvement rule never fires.
    for (const count of [6, 5, 4, 3, 2, 1]) actions.push(c.next({ findings: errors(count) }).action)
    expect(actions).toEqual(['retry_local', 'retry_local', 'change_model', 'ask_advisor', 'ask_advisor', 'stop'])
    expect(c.summary()).toMatchObject({
      attempts: 6,
      localRetries: 2,
      modelChanges: 1,
      advisorCalls: 2,
      lastAction: 'stop',
    })
  })

  test('feedback names exactly the failed checks and the advisor request carries them', () => {
    const c = controller({ taskSummary: 'Reconcile the ledgers.' }, { maxLocalRetries: 1, maxModelChanges: 1 })
    const retry = c.next({ findings: errors(3) })
    expect(retry.action).toBe('retry_local')
    if (retry.action === 'retry_local') {
      expect(retry.feedback).toContain('csv_ragged')
      expect(retry.feedback).toContain('out0.csv')
      expect(retry.feedback).toContain('out2.csv')
    }
    const change = c.next({ findings: errors(2) })
    expect(change.action).toBe('change_model')
    if (change.action === 'change_model') expect(change.feedback).toContain('out1.csv')
    const ask = c.next({ findings: errors(1) })
    expect(ask.action).toBe('ask_advisor')
    if (ask.action === 'ask_advisor') {
      expect(ask.question.length).toBeGreaterThan(0)
      expect(ask.context).toContain('Reconcile the ledgers.')
      expect(ask.context).toContain('out0.csv')
    }
  })

  test('skips change_model when no alternate model is available', () => {
    const c = controller({ alternateModelAvailable: false }, { maxLocalRetries: 1 })
    expect(c.next({ findings: errors(3) }).action).toBe('retry_local')
    expect(c.next({ findings: errors(2) }).action).toBe('ask_advisor')
  })

  test('stops once cumulative spend reaches the limit', () => {
    const c = controller()
    expect(c.next({ findings: errors(3), costUsd: 0.6 }).action).toBe('retry_local')
    const step = c.next({ findings: errors(2), costUsd: 0.6 })
    expect(step.action).toBe('stop')
    if (step.action === 'stop') expect(step.reason).toContain('spend limit')
    expect(c.summary().spentUsd).toBeCloseTo(1.2)
    // A stop is final.
    expect(c.next({ findings: [] }).action).toBe('stop')
  })

  test('unknown cost counts as zero and is reported', () => {
    const c = controller()
    c.next({ findings: errors(3) })
    c.next({ findings: errors(2), costUsd: 0.25 })
    expect(c.summary()).toMatchObject({ spentUsd: 0.25, unknownCostAttempts: 1 })
  })

  test('rejects a negative or non-finite cost', () => {
    expect(() => controller().next({ findings: errors(1), costUsd: -1 })).toThrow(ValidationError)
    expect(() => controller().next({ findings: errors(1), costUsd: Number.NaN })).toThrow(ValidationError)
  })

  test('private boundary skips the advisor and the reason says why', () => {
    const c = controller({ boundaryMode: 'private' }, { maxLocalRetries: 1, maxModelChanges: 0 })
    expect(c.next({ findings: errors(3) }).action).toBe('retry_local')
    const step = c.next({ findings: errors(2) })
    expect(step.action).toBe('stop')
    if (step.action === 'stop') expect(step.reason).toContain('data boundary is private')
    expect(c.summary().advisorCalls).toBe(0)
  })

  test('private data under hybrid skips the advisor; with the boundary off it is allowed', () => {
    const blocked = controller({ boundaryMode: 'hybrid', touchesPrivate: true }, { maxLocalRetries: 0, maxModelChanges: 0 })
    const step = blocked.next({ findings: errors(1) })
    expect(step.action).toBe('stop')
    if (step.action === 'stop') expect(step.reason).toContain('touches private data')

    const allowed = controller({ boundaryMode: 'off', touchesPrivate: true }, { maxLocalRetries: 0, maxModelChanges: 0 })
    expect(allowed.next({ findings: errors(1) }).action).toBe('ask_advisor')

    const shareable = controller({ boundaryMode: 'hybrid', touchesPrivate: false }, { maxLocalRetries: 0, maxModelChanges: 0 })
    expect(shareable.next({ findings: errors(1) }).action).toBe('ask_advisor')
  })

  test('no advisor configured stops with that reason', () => {
    const c = controller({ advisorAvailable: false }, { maxLocalRetries: 0, maxModelChanges: 0 })
    const step = c.next({ findings: errors(1) })
    expect(step.action).toBe('stop')
    if (step.action === 'stop') expect(step.reason).toContain('no advisor is configured')
  })

  test('stops after two consecutive attempts that do not improve', () => {
    const c = controller({}, { maxLocalRetries: 5 })
    expect(c.next({ findings: errors(2) }).action).toBe('retry_local')
    expect(c.next({ findings: errors(2) }).action).toBe('retry_local')
    const step = c.next({ findings: errors(2) })
    expect(step.action).toBe('stop')
    if (step.action === 'stop') {
      expect(step.reason).toBe('no improvement after 3 attempts')
      expect(step.findings).toHaveLength(2)
    }
  })

  test('an improvement resets the no-improvement streak', () => {
    const c = controller({}, { maxLocalRetries: 5 })
    expect(c.next({ findings: errors(3) }).action).toBe('retry_local')
    expect(c.next({ findings: errors(3) }).action).toBe('retry_local')
    expect(c.next({ findings: errors(2) }).action).toBe('retry_local')
    expect(c.next({ findings: errors(2) }).action).toBe('retry_local')
    expect(c.next({ findings: errors(1) }).action).toBe('retry_local')
  })

  test('fewer warnings at the same error count counts as improvement', () => {
    const c = controller({}, { maxLocalRetries: 5 })
    expect(c.next({ findings: [...errors(1), ...warnings(3)] }).action).toBe('retry_local')
    expect(c.next({ findings: [...errors(1), ...warnings(2)] }).action).toBe('retry_local')
    expect(c.next({ findings: [...errors(1), ...warnings(1)] }).action).toBe('retry_local')
  })
})
