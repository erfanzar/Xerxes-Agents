// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, it } from 'vitest'

import {
  COMPOSER_DOUBLE_SPACE_MS,
  focusComposer,
  isComposerFocused,
  refocusComposerOnDoubleSpace,
  registerComposerFocusTarget,
  resetComposerFocusTracking,
} from '../app/composerFocus.js'

function fakeTarget(initialFocused = false) {
  const state = { focusCalls: 0, focused: initialFocused }

  return {
    state,
    get focused() {
      return state.focused
    },
    focus() {
      state.focusCalls += 1
      state.focused = true
    },
  }
}

afterEach(() => {
  registerComposerFocusTarget(null)
  resetComposerFocusTracking()
})

describe('composer double-space refocus', () => {
  it('stays inert on a single unfocused space and refocuses on the second within the window', () => {
    const target = fakeTarget(false)
    registerComposerFocusTarget(target)

    expect(refocusComposerOnDoubleSpace(1_000)).toBe(false)
    expect(target.state.focusCalls).toBe(0)
    expect(isComposerFocused()).toBe(false)

    expect(refocusComposerOnDoubleSpace(1_000 + COMPOSER_DOUBLE_SPACE_MS)).toBe(true)
    expect(target.state.focusCalls).toBe(1)
    expect(isComposerFocused()).toBe(true)
  })

  it('does not refocus when the two presses land outside the window', () => {
    const target = fakeTarget(false)
    registerComposerFocusTarget(target)

    expect(refocusComposerOnDoubleSpace(1_000)).toBe(false)
    expect(refocusComposerOnDoubleSpace(1_000 + COMPOSER_DOUBLE_SPACE_MS + 1)).toBe(false)
    expect(target.state.focusCalls).toBe(0)
  })

  it('stays inert while the composer is already focused and re-arms after a blur', () => {
    const target = fakeTarget(true)
    registerComposerFocusTarget(target)

    expect(refocusComposerOnDoubleSpace(1_000)).toBe(false)
    expect(refocusComposerOnDoubleSpace(1_050)).toBe(false)
    expect(target.state.focusCalls).toBe(0)

    // A focused composer keeps resetting the clock, so a blur followed by one
    // space must not complete a stale pair.
    target.state.focused = false
    expect(refocusComposerOnDoubleSpace(1_100)).toBe(false)
    expect(refocusComposerOnDoubleSpace(1_150)).toBe(true)
    expect(target.state.focusCalls).toBe(1)
  })

  it('is a no-op when no composer is registered', () => {
    expect(refocusComposerOnDoubleSpace(1_000)).toBe(false)
    expect(refocusComposerOnDoubleSpace(1_050)).toBe(false)
    expect(focusComposer()).toBe(false)
  })

  it('focusComposer refuses to steal focus from an already focused composer', () => {
    const target = fakeTarget(true)
    registerComposerFocusTarget(target)

    expect(focusComposer()).toBe(false)
    expect(target.state.focusCalls).toBe(0)
  })
})
