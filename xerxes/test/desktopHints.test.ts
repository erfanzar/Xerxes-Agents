// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { applyCompletion, wantsHints } from '../src/desktop/renderer/hints.js'

describe('composer hints', () => {
  test('a single slash token hints; prose and finished args never do', () => {
    expect(wantsHints('/')).toBe(true)
    expect(wantsHints('/com')).toBe(true)
    expect(wantsHints('/skill ')).toBe(true)
    expect(wantsHints('/skill rev')).toBe(true)
    expect(wantsHints('/skill review:sec')).toBe(true)
    expect(wantsHints('hello')).toBe(false)
    expect(wantsHints('/compact now please')).toBe(false)
    expect(wantsHints('/skill rev extra words')).toBe(false)
    expect(wantsHints('')).toBe(false)
    expect(wantsHints('plan /com inside prose')).toBe(false)
  })

  test('a picked hint becomes the draft with one trailing space', () => {
    expect(applyCompletion('/compact')).toBe('/compact ')
    expect(applyCompletion('/skill review ')).toBe('/skill review ')
    expect(applyCompletion('/skill review:security ')).toBe('/skill review:security ')
  })
})
