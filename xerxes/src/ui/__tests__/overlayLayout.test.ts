// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { responsivePanelWidth } from '../opentui/overlayLayout.js'

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
