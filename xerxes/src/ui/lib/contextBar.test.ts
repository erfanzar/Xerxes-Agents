// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { EMPTY_BLOCK, FILLED_BLOCK, HALF_BLOCK, contextBar, contextBarWithPct } from './contextBar.js'

describe('contextBar', () => {
  it('renders empty, full, and exact half utilization', () => {
    expect(contextBar({ used: 0, window: 200_000, width: 10 })).toBe(EMPTY_BLOCK.repeat(10))
    expect(contextBar({ used: 200_000, window: 200_000, width: 10 })).toBe(FILLED_BLOCK.repeat(10))
    expect(contextBar({ used: 100_000, window: 200_000, width: 10 })).toBe(
      FILLED_BLOCK.repeat(5) + EMPTY_BLOCK.repeat(5)
    )
  })

  it('uses a half block for partial cells and clamps overfull context', () => {
    expect(contextBar({ used: 5, window: 100, width: 10 })).toBe(HALF_BLOCK + EMPTY_BLOCK.repeat(9))
    expect(contextBar({ used: 2, window: 1, width: 3, filled: '#', empty: '.' })).toBe('###')
  })

  it('returns an empty meter for non-positive widths', () => {
    expect(contextBar({ used: 1, window: 2, width: 0 })).toBe('')
    expect(contextBar({ used: 1, window: 2, width: -2 })).toBe('')
  })
})

describe('contextBarWithPct', () => {
  it('keeps the raw percentage visible and can suppress it', () => {
    const options = { used: 50_000, window: 200_000, width: 20 }

    expect(contextBarWithPct(options)).toContain('25.0%')
    expect(contextBarWithPct({ ...options, showPct: false })).toBe(contextBar(options))
  })
})
