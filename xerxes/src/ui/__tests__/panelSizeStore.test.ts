// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it } from 'vitest'

import { adjustPanelWidth, resetPanelWidth, withPanelWidthDelta } from '../app/panelSizeStore.js'

describe('resizable panel width', () => {
  afterEach(resetPanelWidth)

  it('never exceeds a terminal narrower than the normal minimum', () => {
    expect(withPanelWidthDelta(80, 20)).toBe(18)
    expect(withPanelWidthDelta(80, 2)).toBe(1)
  })

  it('retains the usable minimum when the terminal has room', () => {
    adjustPanelWidth(-60)
    expect(withPanelWidthDelta(40, 100)).toBe(24)
  })
})
