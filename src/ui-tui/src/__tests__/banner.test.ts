// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { logo } from '../banner.js'
import { DARK_THEME } from '../theme.js'

describe('banner logo', () => {
  it('uses the blue to purple wordmark palette instead of the UI accent color', () => {
    const colors = logo(DARK_THEME.color).map(([color]) => color)

    expect(colors).toEqual(['#61a7ff', '#5b8dff', '#6575ee', '#665fd6', '#6258b8', '#544c96'])
    expect(colors).not.toContain(DARK_THEME.color.accent)
  })
})
