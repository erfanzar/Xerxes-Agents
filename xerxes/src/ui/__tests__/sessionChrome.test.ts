// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { ctxMeterBar, sessionDisplayTitle, sessionTelemetryLine, statusIdentity, telemetryDuration } from '../domain/statusFormat.js'
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

  it('hides the telemetry line for a fresh session', () => {
    // A bar of zeros is noise; the row mounts only once a turn has run.
    expect(sessionTelemetryLine({ calls: 0, input: 0, output: 0, total: 0 })).toBe('')
    expect(sessionTelemetryLine({ calls: 1, input: 10, output: 5, total: 15, turns: 0 })).toBe('')
  })

  it('renders the desktop-parity cumulative stats line', () => {
    const line = sessionTelemetryLine({
      calls: 64,
      input: 592_000_000,
      output: 931_000,
      total: 592_931_000,
      turns: 64,
      llm_steps: 1_200,
      tool_steps: 906,
      llm_ms: 885 * 60_000 + 11_000,
      tool_ms: 194 * 60_000 + 25_000,
      ttft_avg_ms: 8_800,
      tok_per_sec: 44,
      cache_hit_rate: 0.98
    })
    expect(line).toBe(
      '64 turns · 2.1k steps · LLM 885m11s · tools 194m25s · TTFT 8.8s · 44 tok/s · cache 98% · 592m in · 931k out'
    )
  })

  it('omits unknown telemetry instead of faking it', () => {
    // No cache telemetry from the provider → no cache figure at all, and no
    // TTFT/throughput before their first samples.
    const line = sessionTelemetryLine({ calls: 1, input: 100, output: 20, total: 120, turns: 1, llm_ms: 45 * 60_000 })
    expect(line).toBe('1 turn · LLM 45m00s · 100 in · 20 out')
  })

  it('keeps the desktop duration format: minutes run past 60', () => {
    expect(telemetryDuration(885 * 60_000 + 11_000)).toBe('885m11s')
    expect(telemetryDuration(14 * 3_600_000 + 45 * 60_000)).toBe('885m00s')
    expect(telemetryDuration(9_400)).toBe('9s')
    expect(telemetryDuration(0)).toBe('0s')
  })
})
