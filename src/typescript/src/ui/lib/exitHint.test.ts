// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { formatExitHint } from './exitHint.js'

describe('formatExitHint', () => {
  it('prints a resume command when a session id exists', () => {
    expect(formatExitHint({ sessionId: 'abc123' })).toBe(
      'Goodbye.\nResume this session: xerxes -r abc123\nor: bun run xerxes -r abc123'
    )
  })

  it('falls back to a start command before a session exists', () => {
    expect(formatExitHint({ sessionId: null })).toBe(
      'Goodbye.\nResume a saved session: xerxes -r <session-id>\nor start again: bun run xerxes'
    )
  })

  it('supports the caller command label', () => {
    expect(formatExitHint({ aliasCommand: 'x', bunCommand: 'bun run x', sessionId: 's1' })).toBe(
      'Goodbye.\nResume this session: x -r s1\nor: bun run x -r s1'
    )
  })
})
