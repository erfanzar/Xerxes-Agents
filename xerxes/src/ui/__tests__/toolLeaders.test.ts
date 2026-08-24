// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
import { describe, expect, it } from 'vitest'

import { contentColumnWidth } from '../domain/startupLayout.js'
import { stringWidth } from '../lib/terminalRuntime.opentui.js'
import { isQuietToolName, toolLeaderDots } from '../opentui/messageLine.js'

// Mockup row shape (anatomy element ④):
//   ⏺ Bash bun test test/auth.test.ts ✓ ···· 1.8s
// The glyph leads and carries the outcome tint; the ✓/✗ mark sits right after
// the summary text, BEFORE the dotted leader, which still lands the duration
// at the reading-column edge.
const parts = {
  args: 'test/auth.test.ts',
  duration: '1.8s',
  glyph: '⏺',
  mark: '✓',
  name: 'Bash'
}

describe('tool leader dots', () => {
  it('fills exactly to the reading column edge', () => {
    for (const cols of [80, 100, 120, 140, 220]) {
      const dots = toolLeaderDots(parts, cols)
      expect(dots.length).toBeGreaterThan(0)
      // Same segment order the renderer paints: glyph+name, args, mark,
      // leader, duration. The width math is order-independent, so this is
      // also the painted row's total footprint.
      const total =
        stringWidth(`${parts.glyph} ${parts.name}`) +
        stringWidth(`  ${parts.args}`) +
        stringWidth(` ${parts.mark}`) +
        stringWidth(dots) +
        stringWidth(`  ${parts.duration}`)
      // The row is drawn inside the trail body column: reading column minus
      // the rail gutter (1) and trail padding (2), then a 4-column safety
      // margin for ambiguous-width glyphs and scrollbar gutters. The leader
      // must land the duration NEAR the edge, never past it.
      expect(total).toBe(contentColumnWidth(cols) - 7)
    }
  })

  it('counts the mark wherever it sits between summary and leader', () => {
    // Reordering the mark ahead of the dots (mockup change) must not change
    // the fill: both orders occupy the same cells.
    const withMark = toolLeaderDots(parts, 120)
    const withoutMark = toolLeaderDots({ ...parts, mark: '' }, 120)

    expect(withoutMark.length - withMark.length).toBe(stringWidth(` ${parts.mark}`))
  })

  it('omits dots when the terminal width is unknown', () => {
    // Direct MessageLine consumers (stream segments, tests) pass no cols; the
    // row must keep its pre-leader shape rather than guess a width.
    expect(toolLeaderDots(parts, undefined)).toBe('')
  })

  it('omits dots when the arguments already fill the line', () => {
    const long = { ...parts, args: 'x'.repeat(200) }
    expect(toolLeaderDots(long, 120)).toBe('')
  })
})

describe('quiet read-only calls', () => {
  it('tints display names of read-only tools faint', () => {
    for (const name of ['Read File', 'Glob', 'Grep', 'List', 'View']) {
      expect(isQuietToolName(name)).toBe(true)
    }
  })

  it('leaves mutating and network calls at full outcome colour', () => {
    for (const name of ['Bash', 'Edit', 'Write', 'WebFetch', 'Browser Click']) {
      expect(isQuietToolName(name)).toBe(false)
    }
  })
})
