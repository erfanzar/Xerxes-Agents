// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import {
  OVERLAY_PANEL_SPECS,
  overlayPanelSize,
  overlayPanelWidth,
  responsivePanelWidth
} from '../opentui/overlayLayout.js'

describe('overlayPanelSize', () => {
  it('gives a tall terminal most of its rows, capped by the spec', () => {
    // The F6 panel used to compute its own margin and land on 20 rows of a
    // 42-row terminal — a mostly empty box for a long list.
    expect(overlayPanelSize({ height: 42, width: 140 }, OVERLAY_PANEL_SPECS.agents).height).toBe(38)
    expect(overlayPanelSize({ height: 60, width: 140 }, OVERLAY_PANEL_SPECS.agents).height).toBe(44)
  })

  it('keeps a gutter rather than filling the terminal edge to edge', () => {
    expect(overlayPanelSize({ height: 30, width: 140 }, OVERLAY_PANEL_SPECS.diff).height).toBe(26)
  })

  it('treats the minimum height as aspirational on a short terminal', () => {
    const { height } = overlayPanelSize({ height: 6, width: 140 }, OVERLAY_PANEL_SPECS.diff)

    // The terminal always wins: a panel taller than the screen would put its
    // footer — and every key hint on it — out of reach.
    expect(height).toBeLessThanOrEqual(6)
    expect(height).toBeGreaterThan(0)
  })

  it('shrinks to its content instead of rendering a large empty box', () => {
    const empty = overlayPanelSize(
      { height: 42, width: 140 },
      { ...OVERLAY_PANEL_SPECS.agents, desiredHeight: 8 }
    )
    const busy = overlayPanelSize(
      { height: 42, width: 140 },
      { ...OVERLAY_PANEL_SPECS.agents, desiredHeight: 8 + 40 }
    )

    // Simply making the panel bigger made the empty case look worse, not
    // better: the complaint was the void, not the row count.
    expect(empty.height).toBeLessThan(busy.height)
    expect(busy.height).toBe(38)
  })

  it('never shrinks below the aspirational minimum', () => {
    const tiny = overlayPanelSize(
      { height: 42, width: 140 },
      { ...OVERLAY_PANEL_SPECS.agents, desiredHeight: 1 }
    )

    expect(tiny.height).toBe(12)
  })

  it('matches the width primitive for the same spec', () => {
    const spec = OVERLAY_PANEL_SPECS.terminals

    expect(overlayPanelSize({ height: 40, width: 200 }, spec).width).toBe(
      responsivePanelWidth(200, { max: spec.maxWidth, min: spec.minWidth })
    )
    expect(overlayPanelWidth(200, spec)).toBe(overlayPanelSize({ height: 40, width: 200 }, spec).width)
  })
})

describe('responsive overlay layout', () => {
  it('honors preferred and maximum widths when space is available', () => {
    expect(responsivePanelWidth(60, { max: 84, min: 34 })).toBe(56)
    expect(responsivePanelWidth(120, { max: 84, min: 34 })).toBe(84)
  })

  it('treats the minimum as aspirational on very narrow terminals', () => {
    expect(responsivePanelWidth(30, { max: 84, min: 34 })).toBe(28)
    expect(responsivePanelWidth(10, { max: 84, min: 34 })).toBe(8)
    expect(responsivePanelWidth(1, { max: 84, min: 34 })).toBe(1)
  })
})
