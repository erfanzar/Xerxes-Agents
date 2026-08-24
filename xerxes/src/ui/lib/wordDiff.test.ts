// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Word-level diff math for the F7 viewer: del/add run pairing, common
// prefix/suffix range trimming, and the color blend behind word spans.
import { describe, expect, it } from 'vitest'

import type { DiffLine } from './gitDiff.js'
import { blendColors, intraLineWordRanges, WORD_TINT_ADD } from './wordDiff.js'

const row = (kind: DiffLine['kind'], text: string): DiffLine => ({ kind, text })

const codeRange = (lines: readonly DiffLine[], index: number): { end: number; start: number } | undefined => {
  const range = intraLineWordRanges(lines).get(index)

  return range ? { end: range.end, start: range.start } : undefined
}

describe('intraLineWordRanges', () => {
  it('highlights only the changed middle of a paired rewrite', () => {
    const lines = [
      row('hunk', '@@ -1,2 +1,2 @@'),
      row('del', '-const token = raw.split(" ")[1]'),
      row('add', '+const token = parseTokenHeader(raw)')
    ]

    // Ranges are relative to the code text (the +/- marker excluded):
    // both sides share "const token = " (14 chars), then del keeps
    // "raw.split(\" \")[1]" (17) and add keeps "parseTokenHeader(raw)" (21).
    expect(codeRange(lines, 1)).toEqual({ end: 31, start: 14 })
    expect(codeRange(lines, 2)).toEqual({ end: 35, start: 14 })
  })

  it('pairs a contiguous del-run with the following add-run in order', () => {
    const lines = [
      row('del', '-alpha one'),
      row('del', '-beta one'),
      row('add', '+alpha two'),
      row('add', '+beta two'),
      row('add', '+gamma two')
    ]
    const ranges = intraLineWordRanges(lines)

    // one ↔ one, two ↔ two; the surplus third add has no old side.
    expect(ranges.get(0)).toBeDefined()
    expect(ranges.get(1)).toBeDefined()
    expect(ranges.get(2)).toBeDefined()
    expect(ranges.get(3)).toBeDefined()
    expect(ranges.get(4)).toBeUndefined()
  })

  it('skips pure insertion runs with nothing to pair against', () => {
    const lines = [row('hunk', '@@ -1,0 +1,2 @@'), row('add', '+fresh'), row('add', '+lines')]

    expect(intraLineWordRanges(lines).size).toBe(0)
  })

  it('never pairs across context or hunk boundaries', () => {
    const lines = [
      row('del', '-before hunk'),
      row('context', ' separator'),
      row('add', '+after context'),
      row('del', '-before next'),
      row('hunk', '@@ -5,1 +5,1 @@'),
      row('add', '+after hunk')
    ]

    expect(intraLineWordRanges(lines).size).toBe(0)
  })

  it('re-pairs when a new del-run starts right after an add-run', () => {
    const lines = [
      row('del', '-first old'),
      row('add', '+first new'),
      row('del', '-second old'),
      row('add', '+second new')
    ]
    const ranges = intraLineWordRanges(lines)

    expect(codeRange(lines, 0)).toBeDefined()
    expect(codeRange(lines, 1)).toBeDefined()
    expect(codeRange(lines, 2)).toBeDefined()
    expect(codeRange(lines, 3)).toBeDefined()
  })

  it('reports no ranges for identical paired lines', () => {
    const lines = [row('del', '-same'), row('add', '+same')]

    expect(intraLineWordRanges(lines).size).toBe(0)
  })

  it('finds a change at line start when the suffix is shared', () => {
    const lines = [row('del', '-old docs line'), row('add', '+new docs line')]

    expect(codeRange(lines, 0)).toEqual({ end: 3, start: 0 })
    expect(codeRange(lines, 1)).toEqual({ end: 3, start: 0 })
  })

  it('handles an append where the old side is fully kept', () => {
    const lines = [row('del', '-abc'), row('add', '+abcdef')]

    expect(codeRange(lines, 0)).toEqual({ end: 3, start: 3 })
    expect(codeRange(lines, 1)).toEqual({ end: 6, start: 3 })
  })

  it('is empty for an empty diff', () => {
    expect(intraLineWordRanges([]).size).toBe(0)
  })
})

describe('blendColors', () => {
  it('blends hex colors channel-wise at the requested ratio', () => {
    expect(blendColors('#000000', '#ffffff', 0.5)).toBe('#808080')
    // Mockup 07 w-add: rgba(word, .3) composited over the add-row tint.
    expect(blendColors('#14251b', 'rgb(131,201,157)', WORD_TINT_ADD)).toBe('#355642')
  })

  it('clamps ratios and returns the nearer color', () => {
    expect(blendColors('#112233', '#445566', -1)).toBe('#112233')
    expect(blendColors('#112233', '#445566', 0)).toBe('#112233')
    expect(blendColors('#112233', '#445566', 1)).toBe('#445566')
    expect(blendColors('#112233', '#445566', 2)).toBe('#445566')
  })

  it('degrades to the base color for unparseable operands', () => {
    expect(blendColors('#14251b', 'ansi256(72)', 0.5)).toBe('#14251b')
    expect(blendColors('not-a-color', '#ffffff', 0.5)).toBe('not-a-color')
  })

  it('accepts rgb() theme roles as either operand', () => {
    expect(blendColors('rgb(20, 37, 27)', '#83c99d', 1)).toBe('#83c99d')
  })
})
