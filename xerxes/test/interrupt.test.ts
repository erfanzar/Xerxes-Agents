// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  AsyncInterruptScope,
  InterruptRequestedError,
  InterruptToken,
  clearCurrentToken,
  currentToken,
  interruptScope,
  setCurrentToken,
} from '../src/runtime/interrupt.js'

test('interrupt tokens are reusable, cooperative, and expose an AbortSignal generation', async () => {
  const token = new InterruptToken()
  expect(token.isSet()).toBe(false)
  expect(token.signal.aborted).toBe(false)

  const waiting = token.wait(100)
  token.set()
  expect(await waiting).toBe(true)
  expect(token.isSet()).toBe(true)
  expect(token.signal.aborted).toBe(true)
  expect(() => token.throwIfSet()).toThrow(InterruptRequestedError)

  const interruptedSignal = token.signal
  token.clear()
  expect(token.isSet()).toBe(false)
  expect(interruptedSignal.aborted).toBe(true)
  expect(token.signal).not.toBe(interruptedSignal)
  expect(token.signal.aborted).toBe(false)
  expect(await token.wait(1)).toBe(false)
})

test('current-token helpers bind only the current async chain and can be cleared', () => {
  clearCurrentToken()
  try {
    expect(currentToken()).toBeUndefined()
    const token = new InterruptToken()
    setCurrentToken(token)
    expect(currentToken()).toBe(token)
  } finally {
    clearCurrentToken()
  }
  expect(currentToken()).toBeUndefined()
})

test('interrupt scopes propagate across awaits, nest correctly, and isolate concurrent work', async () => {
  clearCurrentToken()
  const outer = new InterruptToken()
  try {
    await interruptScope(async installed => {
      expect(installed).toBe(outer)
      expect(currentToken()).toBe(outer)
      await Promise.resolve()
      expect(currentToken()).toBe(outer)

      const inner = new InterruptToken()
      await interruptScope(async nested => {
        expect(nested).toBe(inner)
        await Promise.resolve()
        expect(currentToken()).toBe(inner)
      }, inner)
      expect(currentToken()).toBe(outer)
    }, outer)
    expect(currentToken()).toBeUndefined()
  } finally {
    clearCurrentToken()
  }

  const scope = new AsyncInterruptScope()
  const first = new InterruptToken()
  const second = new InterruptToken()
  const [firstSeen, secondSeen] = await Promise.all([
    scope.run(first, async () => {
      await Promise.resolve()
      return scope.current()
    }),
    scope.run(second, async () => {
      await Promise.resolve()
      return scope.current()
    }),
  ])
  expect(firstSeen).toBe(first)
  expect(secondSeen).toBe(second)
  expect(scope.current()).toBeUndefined()
})
