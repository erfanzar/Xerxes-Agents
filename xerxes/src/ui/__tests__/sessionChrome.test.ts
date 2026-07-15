// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { sessionDisplayTitle, statusIdentity } from '../domain/statusFormat.js'
import { displayModeLabel } from '../opentui/appChrome.js'

describe('session chrome', () => {
  it('keeps model and mode as the compact status identity', () => {
    expect(statusIdentity('anthropic/claude-sonnet-4-5', 'plan', 'high', true)).toBe('sonnet 4.5 high fast · plan')
    expect(statusIdentity('', undefined)).toBe('model unset · code')
  })

  it('uses Grok-style title case for mode labels in both prompt and session chrome', () => {
    expect(displayModeLabel('code')).toBe('Code')
    expect(displayModeLabel('researcher')).toBe('Researcher')
    expect(displayModeLabel('')).toBe('Code')
  })

  it('replaces transient TUI transport titles with the first user request', () => {
    expect(sessionDisplayTitle('tui:400b8d876331', '  Explain\nthis repository.  ')).toBe('Explain this repository.')
    expect(sessionDisplayTitle('Release audit', 'fallback')).toBe('Release audit')
    expect(sessionDisplayTitle('tui:abc', 'x'.repeat(80), 12)).toBe('xxxxxxxxxxx…')
  })
})
