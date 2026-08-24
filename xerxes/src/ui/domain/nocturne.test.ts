// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// The design system's own rules, as assertions. Each of these is a sentence
// from the canvas that a future edit could quietly break.
import { describe, expect, it } from 'vitest'

import { NOCTURNE_DARK } from '../theme.js'

import {
  ATTENTION_ORDER,
  attentionRank,
  byAttention,
  clipPath,
  densityFor,
  FULL_DENSITY,
  GLYPH,
  leaderRun,
  type NocturneState,
  stateSkin,
  wrapWithContinuation
} from './nocturne.js'

describe('the glyph table', () => {
  it('gives every mark exactly one job', () => {
    const glyphs = Object.values(GLYPH)

    expect(new Set(glyphs).size).toBe(glyphs.length)
  })
})

describe('attention order', () => {
  it('ranks by what you have to do, not by lifecycle', () => {
    expect(ATTENTION_ORDER).toEqual(['needsInput', 'working', 'done', 'failed'])
  })

  it('sorts a mixed list so nothing blocked sits under something finished', () => {
    const rows: { state: NocturneState }[] = [
      { state: 'done' },
      { state: 'failed' },
      { state: 'needsInput' },
      { state: 'working' }
    ]

    expect([...rows].sort(byAttention(row => row.state)).map(row => row.state)).toEqual([
      'needsInput',
      'working',
      'done',
      'failed'
    ])
  })

  it('puts an unknown state last rather than first', () => {
    expect(attentionRank('nonsense' as NocturneState)).toBe(ATTENTION_ORDER.length)
  })
})

describe('state skins', () => {
  it('maps each state onto its one voice colour', () => {
    expect(stateSkin('needsInput', NOCTURNE_DARK).dot).toBe(NOCTURNE_DARK.needsInput)
    expect(stateSkin('working', NOCTURNE_DARK).dot).toBe(NOCTURNE_DARK.working)
    expect(stateSkin('done', NOCTURNE_DARK).dot).toBe(NOCTURNE_DARK.done)
    expect(stateSkin('failed', NOCTURNE_DARK).dot).toBe(NOCTURNE_DARK.failed)
  })

  it('gives the working state the plain hairline, so a busy board is not four tinted frames', () => {
    expect(stateSkin('working', NOCTURNE_DARK).border).toBe(NOCTURNE_DARK.hairline)
    expect(stateSkin('needsInput', NOCTURNE_DARK).border).toBe(NOCTURNE_DARK.needsInputCardBorder)
  })

  it('softens prose that has to carry a state colour itself', () => {
    // The voice colours are tuned for marks and chrome, not for paragraphs.
    expect(stateSkin('needsInput', NOCTURNE_DARK).text).toBe(NOCTURNE_DARK.needsInputText)
    expect(stateSkin('failed', NOCTURNE_DARK).text).toBe(NOCTURNE_DARK.failedText)
  })
})

describe('leaderRun', () => {
  it('fills the gap between a label and its right-aligned quantity', () => {
    expect(leaderRun(40, 10, 5)).toBe(` ${'·'.repeat(20)}`)
  })

  it('prints nothing rather than a stub — one or two dots read as a fault', () => {
    expect(leaderRun(20, 10, 4)).toBe('')
    expect(leaderRun(10, 40, 5)).toBe('')
  })

  it('stops short of the edge so an ambiguous-width glyph cannot push the quantity off', () => {
    // 5 columns of cushion: the leading separator plus a 4-column margin,
    // so the run plus the quantity always land inside the column.
    const run = leaderRun(30, 10, 5)

    expect(run.length).toBe(1 + (30 - 10 - 5 - 5))
    expect(10 + run.length + 5).toBeLessThan(30)
  })
})

describe('densityFor', () => {
  it('gives things up in the canvas order as the terminal narrows', () => {
    const wide = densityFor(150)
    expect(wide).toEqual(FULL_DENSITY)

    // Side panels go first…
    expect(densityFor(95).sidePanels).toBe(false)
    expect(densityFor(95).goals).toBe(true)
    // …then card goals…
    expect(densityFor(80).goals).toBe(false)
    expect(densityFor(80).captionSource).toBe(true)
    // …then the caption's source label…
    expect(densityFor(70).captionSource).toBe(false)
    expect(densityFor(70).cardBudget).toBe(true)
    // …and the second half of a card's budget last.
    expect(densityFor(50).cardBudget).toBe(false)
  })

  it('never gives up in a different order, however narrow', () => {
    for (let cols = 20; cols <= 160; cols += 1) {
      const density = densityFor(cols)

      // Anything kept implies everything cheaper than it is kept too.
      expect(!density.sidePanels || density.goals).toBe(true)
      expect(!density.goals || density.captionSource).toBe(true)
      expect(!density.captionSource || density.cardBudget).toBe(true)
    }
  })
})

describe('clipPath', () => {
  it('clips from the left — the filename is the part you are looking for', () => {
    expect(clipPath('packages/runtime/scheduler.ts', 16)).toBe('…me/scheduler.ts')
  })

  it('leaves a path that already fits alone', () => {
    expect(clipPath('scheduler.ts', 16)).toBe('scheduler.ts')
  })
})

describe('wrapWithContinuation', () => {
  it('marks every continuation so a wrap is never read as a second command', () => {
    expect(wrapWithContinuation('rm -rf node_modules && bun install --frozen-lockfile', 26)).toEqual([
      'rm -rf node_modules && bun',
      '↳ install ',
      '↳ --frozen-lockfile'
    ])
  })

  it('leaves a command that already fits on one line alone', () => {
    expect(wrapWithContinuation('bun test', 40)).toEqual(['bun test'])
  })

  it('breaks mid-token only when one token is wider than the column', () => {
    expect(wrapWithContinuation('/very/long/path/that/never/breaks', 12)).toEqual([
      '/very/long/p',
      '↳ ath/that/n',
      '↳ ever/break',
      '↳ s'
    ])
  })

  it('gives up rather than mangling a column too narrow for the glyph', () => {
    expect(wrapWithContinuation('bun test', 3)).toEqual(['bun test'])
  })
})
