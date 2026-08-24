// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import {
  DERAFSH_ANIMATION_FRAME_MS,
  DERAFSH_ANIMATION_FRAME_COUNT,
  DERAFSH_KAVIANI_ART,
  DERAFSH_KAVIANI_COMPACT_ART,
  DERAFSH_KAVIANI_GLYPH,
  DERAFSH_KAVIANI_WIDTH,
  compactBrailleRows,
  derafshAnimationEnabled,
  derafshCompactGradientFrame,
  derafshGradientFrame,
  derafshGradientPalette,
  derafshGradientRamp,
  derafshKaviani,
  derafshWavePosition
} from '../banner.js'
import { DARK_THEME } from '../theme.js'

describe('Derafsh Kaviani terminal mark', () => {
  it('preserves the supplied Xerxes Braille-pixel Derafsh asset', () => {
    const lines = derafshKaviani(DARK_THEME.color)

    expect(DERAFSH_KAVIANI_GLYPH).toBe('✦')
    expect(DERAFSH_KAVIANI_ART).toHaveLength(20)
    expect(DERAFSH_KAVIANI_WIDTH).toBeGreaterThan(20)
    expect(lines.map(([, text]) => text)).toEqual(DERAFSH_KAVIANI_ART)
    expect(DERAFSH_KAVIANI_ART[0]).toBe('⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⣿⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀')
    expect(DERAFSH_KAVIANI_ART[13]).toBe('⣠⣼⠷⠾⣿⡿⠿⠿⠿⠷⢾⣿⡷⠾⢿⡿⠿⠿⠿⠷⢾⣧⣄')
    expect(DERAFSH_KAVIANI_ART[19]).toBe('⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠛⠀⠀⠀⠀⠀⠀⠀')
    expect(lines.map(([color]) => color)).toEqual(Array.from({ length: 20 }, () => DARK_THEME.color.warn))
  })

  it('keeps skin-provided mark art on the existing markup path', () => {
    expect(derafshKaviani(DARK_THEME.color, '[#abcdef]custom mark[/]')).toEqual([['#abcdef', 'custom mark']])
  })

  it('OR-downsamples pairs of Braille rows without changing the mark width', () => {
    expect(compactBrailleRows(['⣿', '⣿'])).toEqual(['⣿'])
    expect(compactBrailleRows(['⣿', '⠀'])).toEqual(['⠛'])
    expect(compactBrailleRows(['⠀', '⣿'])).toEqual(['⣤'])
    expect(DERAFSH_KAVIANI_COMPACT_ART).toHaveLength(10)
    expect(DERAFSH_KAVIANI_COMPACT_ART.every(line => line.length === DERAFSH_KAVIANI_WIDTH)).toBe(true)
    expect(DERAFSH_KAVIANI_COMPACT_ART.every(line => [...line].every(glyph => /[\u2800-\u28ff]/u.test(glyph)))).toBe(
      true
    )
  })

  it('preserves every Derafsh glyph while shifting a cyclic gradient', () => {
    const first = derafshGradientFrame(DARK_THEME.color, 0)
    const next = derafshGradientFrame(DARK_THEME.color, 1)
    const wrapped = derafshGradientFrame(DARK_THEME.color, DERAFSH_ANIMATION_FRAME_COUNT)
    const negative = derafshGradientFrame(DARK_THEME.color, -1)
    const last = derafshGradientFrame(DARK_THEME.color, DERAFSH_ANIMATION_FRAME_COUNT - 1)

    expect(first.map(([, text]) => text)).toEqual(DERAFSH_KAVIANI_ART)
    expect(next.map(([, text]) => text)).toEqual(DERAFSH_KAVIANI_ART)
    expect(new Set(first.map(([color]) => color)).size).toBeGreaterThan(3)
    expect(next.map(([color]) => color)).not.toEqual(first.map(([color]) => color))
    expect(wrapped).toEqual(first)
    expect(negative).toEqual(last)
  })

  it('moves a fast, non-linear colour wave without moving the art', () => {
    const rows = DERAFSH_KAVIANI_ART.length
    const first = Array.from({ length: rows }, (_, row) => derafshWavePosition(row, rows, 0))
    const next = Array.from({ length: rows }, (_, row) => derafshWavePosition(row, rows, 1))
    const linearStep = 1 / (rows - 1)
    const spatialSteps = first.slice(1).map((position, row) => position - first[row]!)

    expect(DERAFSH_ANIMATION_FRAME_MS).toBeLessThan(100)
    expect(next).not.toEqual(first)
    expect(spatialSteps.some(step => Math.abs(step - linearStep) > 0.01)).toBe(true)
    expect(
      Array.from({ length: rows }, (_, row) => derafshWavePosition(row, rows, DERAFSH_ANIMATION_FRAME_COUNT))
    ).toEqual(first)
    expect(derafshWavePosition(-10, rows, 0)).toBe(derafshWavePosition(0, rows, 0))
    expect(derafshWavePosition(rows + 10, rows, 0)).toBe(derafshWavePosition(rows - 1, rows, 0))
  })

  it('keeps the compact Derafsh on the same animated gradient cycle', () => {
    const first = derafshCompactGradientFrame(DARK_THEME.color, 0)
    const next = derafshCompactGradientFrame(DARK_THEME.color, 1)

    expect(first.map(([, text]) => text)).toEqual(DERAFSH_KAVIANI_COMPACT_ART)
    expect(new Set(first.map(([color]) => color)).size).toBeGreaterThan(3)
    expect(next.map(([color]) => color)).not.toEqual(first.map(([color]) => color))
    expect(derafshCompactGradientFrame(DARK_THEME.color, DERAFSH_ANIMATION_FRAME_COUNT)).toEqual(first)
  })

  it('anchors the gradient in lapis blue, royal purple, and deep azure', () => {
    const palette = derafshGradientPalette(DARK_THEME.color)

    expect(palette).toContain('#6ea8fe')
    // The third stop follows the brand token, not `warn` — amber is a
    // semantic warning colour and must not tint the emblem (v3 pivot).
    expect(palette).toContain(DARK_THEME.color.brandGold)
    expect(palette).toContain(DARK_THEME.color.system)
  })

  it('samples a static wordmark ramp across the palette without repeating hues', () => {
    const ramp = derafshGradientRamp(DARK_THEME.color, 6)

    expect(ramp).toHaveLength(6)
    expect(new Set(ramp).size).toBe(6)
    expect(ramp[0]).toBe('#6ea8fe')
    // Deterministic: same theme, same ramp — a per-letter wordmark must not flicker.
    expect(derafshGradientRamp(DARK_THEME.color, 6)).toEqual(ramp)
  })

  it('handles degenerate wordmark lengths', () => {
    expect(derafshGradientRamp(DARK_THEME.color, 0)).toEqual([])
    expect(derafshGradientRamp(DARK_THEME.color, -3)).toEqual([])
    expect(derafshGradientRamp(DARK_THEME.color, 1)).toEqual(['#6ea8fe'])
  })

  it('animates only in motion-capable interactive terminals', () => {
    expect(derafshAnimationEnabled({}, true)).toBe(true)
    expect(derafshAnimationEnabled({}, false)).toBe(false)
    expect(derafshAnimationEnabled({ CI: '1' }, true)).toBe(false)
    expect(derafshAnimationEnabled({ TERM: 'dumb' }, true)).toBe(false)
    expect(derafshAnimationEnabled({ NO_COLOR: '' }, true)).toBe(true)
    expect(derafshAnimationEnabled({ XERXES_TUI_ANIMATIONS: '0' }, true)).toBe(false)
    expect(derafshAnimationEnabled({ CI: '1', XERXES_TUI_ANIMATIONS: '1' }, true)).toBe(true)
  })
})
