// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { ctxMeterBar, sessionDisplayTitle, statusIdentity } from '../domain/statusFormat.js'
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

  it('leaves an unnamed session blank rather than inventing a title', () => {
    // The blank is load bearing: SessionHeader renders just the mode label
    // for it. Substituting a placeholder here made every chat read as
    // "Untitled chat" and made that branch unreachable.
    expect(sessionDisplayTitle('tui:400b8d876331')).toBe('')
    expect(sessionDisplayTitle('')).toBe('')
    expect(sessionDisplayTitle(null)).toBe('')
  })

  it('still shows and clamps a real title', () => {
    expect(sessionDisplayTitle('Release audit')).toBe('Release audit')
    expect(sessionDisplayTitle('Release audit for the daemon', 12)).toBe('Release aud…')
  })

  it('renders the five-cell context meter bar across the pressure range', () => {
    expect(ctxMeterBar(0)).toBe('▱▱▱▱▱')
    expect(ctxMeterBar(38)).toBe('▰▰▱▱▱')
    expect(ctxMeterBar(50)).toBe('▰▰▰▱▱')
    expect(ctxMeterBar(100)).toBe('▰▰▰▰▰')
    expect(ctxMeterBar(120)).toBe('▰▰▰▰▰')
  })

  it('keeps the meter honest at the edges', () => {
    // Unknown pressure stays an all-empty bar rather than guessing.
    expect(ctxMeterBar(undefined)).toBe('▱▱▱▱▱')
    expect(ctxMeterBar(Number.NaN)).toBe('▱▱▱▱▱')
    // Any non-zero usage earns a visible cell.
    expect(ctxMeterBar(4)).toBe('▰▱▱▱▱')
    expect(ctxMeterBar(-10)).toBe('▱▱▱▱▱')
  })
})
