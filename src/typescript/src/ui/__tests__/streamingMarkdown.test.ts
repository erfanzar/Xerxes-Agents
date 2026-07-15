// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { fenceOpenAt, findStableBoundary, splitStreaming } from '../lib/streamingMarkdown.js'

describe('fenceOpenAt', () => {
  it('is false outside any fence', () => {
    expect(fenceOpenAt('a\n\nb', 4)).toBe(false)
  })
  it('is true between an opening fence and its close', () => {
    const s = '```ts\ncode here\n'
    expect(fenceOpenAt(s, s.length)).toBe(true)
  })
  it('is false once the fence closes', () => {
    const s = '```\ncode\n```\n'
    expect(fenceOpenAt(s, s.length)).toBe(false)
  })
})

describe('findStableBoundary', () => {
  it('returns -1 when there is no blank-line boundary yet', () => {
    expect(findStableBoundary('still typing the first paragraph')).toBe(-1)
  })
  it('finds the last blank-line boundary', () => {
    const text = 'para one\n\npara two in progress'
    expect(findStableBoundary(text)).toBe('para one\n\n'.length)
  })
  it('refuses a boundary that lands inside an open code fence', () => {
    // blank line is inside the fence; no safe boundary → -1
    const text = '```\nline one\n\nline two'
    expect(findStableBoundary(text)).toBe(-1)
  })
  it('allows a boundary after a closed fence', () => {
    const text = '```\ncode\n```\n\nnext para'
    expect(findStableBoundary(text)).toBe('```\ncode\n```\n\n'.length)
  })
})

describe('splitStreaming (monotonic)', () => {
  it('keeps everything unstable until the first boundary', () => {
    const s = splitStreaming('typing...', '')
    expect(s.stablePrefix).toBe('')
    expect(s.unstableSuffix).toBe('typing...')
  })

  it('freezes the prefix once a boundary appears', () => {
    const text = 'first block\n\nsecond'
    const s = splitStreaming(text, '')
    expect(s.stablePrefix).toBe('first block\n\n')
    expect(s.unstableSuffix).toBe('second')
  })

  it('only advances the prefix, never retreats', () => {
    const t1 = 'a\n\nb' // boundary after 'a\n\n'
    const s1 = splitStreaming(t1, '')
    expect(s1.stablePrefix).toBe('a\n\n')

    // next delta extends b, still no new boundary → prefix unchanged
    const t2 = 'a\n\nb more text'
    const s2 = splitStreaming(t2, s1.stablePrefix)
    expect(s2.stablePrefix).toBe('a\n\n')
    expect(s2.unstableSuffix).toBe('b more text')

    // a new boundary appears → prefix grows
    const t3 = 'a\n\nb more text\n\nc'
    const s3 = splitStreaming(t3, s2.stablePrefix)
    expect(s3.stablePrefix).toBe('a\n\nb more text\n\n')
    expect(s3.unstableSuffix).toBe('c')
  })

  it('resets when the text no longer starts with the previous prefix', () => {
    const s = splitStreaming('brand new turn', 'old prefix\n\n')
    expect(s.stablePrefix).toBe('')
    expect(s.unstableSuffix).toBe('brand new turn')
  })
})
