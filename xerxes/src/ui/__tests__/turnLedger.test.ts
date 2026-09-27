// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { ledgerDuration } from '../opentui/messageLine.js'

describe('turn receipt', () => {
  it('reads seconds, then minutes, then hours', () => {
    expect(ledgerDuration(4.24)).toBe('4.2s')
    expect(ledgerDuration(83.4)).toBe('1m 23s')
    expect(ledgerDuration(7500)).toBe('2h 5m')
  })

})
