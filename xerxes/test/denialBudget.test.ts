// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  DEFAULT_MAX_CONSECUTIVE_DENIALS,
  DenialBudget,
  denialBudgetFromConfig,
  denialBudgetStopText,
} from '../src/runtime/denialBudget.js'

test('an omitted maximum takes the default rather than running unbounded', () => {
  const budget = new DenialBudget()
  expect(budget.maxDenials).toBe(DEFAULT_MAX_CONSECUTIVE_DENIALS)
  expect(budget.exhausted).toBeFalse()
})

test('refusals accumulate until the budget is exhausted', () => {
  const budget = new DenialBudget(3)
  expect(budget.record('policy_denied', 'Bash')).toBe(1)
  expect(budget.remaining).toBe(2)
  budget.record('permission_rejected', 'Bash')
  expect(budget.exhausted).toBeFalse()
  budget.record('cancelled', 'Bash')
  expect(budget.exhausted).toBeTrue()
  expect(budget.remaining).toBe(0)
})

test('a successful tool clears the streak so intermittent denials never exhaust it', () => {
  const budget = new DenialBudget(2)
  budget.record('policy_denied', 'Bash')
  budget.reset()
  budget.record('policy_denied', 'Bash')
  expect(budget.exhausted).toBeFalse()
  expect(budget.used).toBe(1)
})

test('a non-positive maximum is an explicit opt-out', () => {
  const disabled = new DenialBudget(0)
  expect(disabled.maxDenials).toBeUndefined()
  expect(disabled.remaining).toBeUndefined()
  for (let index = 0; index < 100; index += 1) disabled.record('policy_denied', 'Bash')
  expect(disabled.exhausted).toBeFalse()
  expect(new DenialBudget(null).maxDenials).toBeUndefined()
})

test('a non-integer maximum is rejected instead of silently rounded', () => {
  expect(() => new DenialBudget(2.5)).toThrow(RangeError)
})

test('the stop text names the last refusal and never asks for approval', () => {
  const budget = new DenialBudget(2)
  budget.record('policy_denied', 'Bash')
  budget.record('permission_rejected', 'WriteFile')

  const text = denialBudgetStopText(budget)
  expect(text).toContain('2 consecutive tool calls were refused')
  expect(text).toContain('a rejected permission prompt on WriteFile')
  expect(text.toLowerCase()).not.toContain('approve')
  expect(text.toLowerCase()).not.toContain('confirm')
})

test('the stop text stays coherent when nothing was recorded', () => {
  expect(denialBudgetStopText(new DenialBudget(1))).toContain('0 consecutive tool calls')
})

test('the last denial is forgotten together with the streak', () => {
  const budget = new DenialBudget(4)
  budget.record('cancelled', 'Bash')
  expect(budget.lastDenial).toEqual({ kind: 'cancelled', toolName: 'Bash' })
  budget.reset()
  expect(budget.lastDenial).toBeUndefined()
})

test('config wins over the environment and both beat the default', () => {
  const environment = { XERXES_MAX_CONSECUTIVE_DENIALS: '7' }
  expect(denialBudgetFromConfig({ max_consecutive_denials: 4 }, { environment }).maxDenials).toBe(4)
  expect(denialBudgetFromConfig({}, { environment }).maxDenials).toBe(7)
  expect(denialBudgetFromConfig({}, { environment: {} }).maxDenials)
    .toBe(DEFAULT_MAX_CONSECUTIVE_DENIALS)
})

test('a typo in the configured ceiling is a configuration error, not unbounded', () => {
  expect(() => denialBudgetFromConfig({ max_consecutive_denials: 'lots' })).toThrow(RangeError)
  expect(() => denialBudgetFromConfig({ max_consecutive_denials: -3 })).toThrow(RangeError)
})
