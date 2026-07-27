// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it } from 'vitest'

import {
  $turnLive,
  beginTurnPulse,
  endTurnPulse,
  getTurnPulse,
  patchTurnState,
  resetTurnState
} from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import {
  describeLiveness,
  LIVENESS_PULSE_MS,
  LIVENESS_STALL_MS,
  LIVENESS_TOKENS_AFTER_MS,
  LIVENESS_VERB_MS,
  livenessGlyph,
  livenessLabel,
  livenessTokens,
  livenessVerb
} from '../lib/liveness.js'
import { DEFAULT_BRANDING, Skin } from '../lib/skinEngine.js'

const at = (startedAt: number, now: number, extra: Partial<{ lastDeltaAt: number; toolCount: number }> = {}) =>
  describeLiveness({ lastDeltaAt: extra.lastDeltaAt ?? 0, now, startedAt, toolCount: extra.toolCount ?? 0 })

describe('liveness phases', () => {
  it('reports streaming while deltas keep arriving', () => {
    const liveness = at(1_000, 9_000, { lastDeltaAt: 8_500 })

    expect(liveness.phase).toBe('streaming')
    expect(liveness.elapsedMs).toBe(8_000)
    expect(liveness.stallMs).toBe(500)
  })

  it('reports a running tool even when the text stream has been quiet for minutes', () => {
    const liveness = at(1_000, 300_000, { lastDeltaAt: 2_000, toolCount: 1 })

    expect(liveness.phase).toBe('tool')
    expect(liveness.stallMs).toBe(298_000)
  })

  it('falls to stalled once nothing runs and no delta lands for the stall window', () => {
    const lastDeltaAt = 5_000

    expect(at(1_000, lastDeltaAt + LIVENESS_STALL_MS - 1, { lastDeltaAt }).phase).toBe('streaming')
    expect(at(1_000, lastDeltaAt + LIVENESS_STALL_MS, { lastDeltaAt }).phase).toBe('stalled')
  })

  it('measures the first-token wait from the turn start, so a slow open still stalls', () => {
    const liveness = at(1_000, 1_000 + LIVENESS_STALL_MS)

    expect(liveness.stallMs).toBe(LIVENESS_STALL_MS)
    expect(liveness.phase).toBe('stalled')
  })

  it('stays at zero before a turn has started', () => {
    const liveness = at(0, 50_000)

    expect(liveness).toMatchObject({ elapsedMs: 0, phase: 'streaming', stallMs: 0 })
  })
})

describe('liveness intensity', () => {
  it('breathes up and back down across one pulse period', () => {
    const dim = at(1, 1, { lastDeltaAt: 1 })
    const bright = at(1, 1 + LIVENESS_PULSE_MS / 2, { lastDeltaAt: 1 })
    const dimAgain = at(1, 1 + LIVENESS_PULSE_MS, { lastDeltaAt: 1 })

    expect(dim.intensity).toBeCloseTo(0)
    expect(bright.intensity).toBeCloseTo(1)
    expect(dimAgain.intensity).toBeCloseTo(0)
  })

  it('ramps on the clock, so a dropped frame resumes where time went, not where the frames stopped', () => {
    const quarter = at(1, 1 + LIVENESS_PULSE_MS / 4, { lastDeltaAt: 1 })
    const lateNow = 1 + LIVENESS_PULSE_MS * 8.25
    const skipped = at(1, lateNow, { lastDeltaAt: lateNow })

    expect(skipped.intensity).toBeCloseTo(quarter.intensity)
  })

  it('damps a stalled turn to a faint heartbeat instead of a full beat', () => {
    const stalled = at(1, 1 + LIVENESS_PULSE_MS / 2 + LIVENESS_PULSE_MS * 10)

    expect(stalled.phase).toBe('stalled')
    expect(stalled.intensity).toBeGreaterThan(0)
    expect(stalled.intensity).toBeLessThan(0.5)
  })
})

describe('liveness glyph and verb', () => {
  it('walks the pulse frames with intensity and freezes when stalled', () => {
    expect(livenessGlyph('streaming', 0)).toBe('·')
    expect(livenessGlyph('streaming', 1)).toBe('◆')
    expect(livenessGlyph('tool', 0.5)).toBe('◈')
    expect(livenessGlyph('stalled', 1)).toBe('◌')
  })

  it('rotates verbs on the clock and wraps around the list', () => {
    const verbs = ['inscribing', 'consulting', 'surveying']

    expect(livenessVerb(verbs, 0)).toBe('inscribing')
    expect(livenessVerb(verbs, LIVENESS_VERB_MS)).toBe('consulting')
    expect(livenessVerb(verbs, LIVENESS_VERB_MS * 3)).toBe('inscribing')
  })

  it('never renders an empty verb when a skin ships a blank list', () => {
    expect(livenessVerb([], 0)).toBe('working')
    expect(livenessVerb(['  ', ''], 12_345)).toBe('working')
  })

  it('sources its verbs from the skin branding rather than a hard-coded list', () => {
    const skin = new Skin({ name: 'test', branding: { spinner_verbs: 'charging, striking' } })
    const shipped = new Skin({ name: 'default' }).spinnerVerbs()

    expect(skin.spinnerVerbs()).toEqual(['charging', 'striking'])
    expect(livenessVerb(skin.spinnerVerbs(), LIVENESS_VERB_MS)).toBe('striking')
    expect(shipped).toEqual(DEFAULT_BRANDING.spinner_verbs!.split(','))
  })
})

describe('liveness label', () => {
  const verbs = ['inscribing']

  const base = 1_000

  it('shows the verb and an elapsed clock', () => {
    const liveness = at(base, base + 72_000, { lastDeltaAt: base + 71_900 })

    expect(livenessLabel(liveness, { tokens: 0, verbs: ['gilding'] })).toBe('gilding… 01:12')
  })

  it('keeps short turns quiet by withholding token counts', () => {
    const early = at(base, base + LIVENESS_TOKENS_AFTER_MS - 1, { lastDeltaAt: base + LIVENESS_TOKENS_AFTER_MS - 1 })
    const later = at(base, base + LIVENESS_TOKENS_AFTER_MS, { lastDeltaAt: base + LIVENESS_TOKENS_AFTER_MS })

    expect(livenessLabel(early, { tokens: 4_200, verbs })).not.toContain('tokens')
    expect(livenessLabel(later, { tokens: 4_200, verbs })).toContain('4.2K tokens')
  })

  it('tells the user Esc interrupts once the turn looks wedged', () => {
    const stalled = at(base, base + LIVENESS_STALL_MS)
    const streaming = at(base, base + LIVENESS_STALL_MS - 1, { lastDeltaAt: base + LIVENESS_STALL_MS - 1 })

    expect(livenessLabel(stalled, { tokens: 0, verbs })).toContain('Esc to interrupt')
    expect(livenessLabel(streaming, { tokens: 0, verbs })).not.toContain('Esc')
  })

  it('counts reasoning, tool, and streamed text towards the turn token volume', () => {
    expect(livenessTokens({ reasoningTokens: 10, streaming: '12345678', toolTokens: 5 })).toBe(17)
  })
})

describe('turn pulse stamping', () => {
  afterEach(() => {
    resetTurnState()
    resetUiState()
  })

  it('stamps a delta whenever streamed or reasoning text grows', () => {
    beginTurnPulse(1_000)
    expect(getTurnPulse().lastDeltaAt).toBe(0)

    patchTurnState({ streaming: 'hel' })
    const afterStream = getTurnPulse().lastDeltaAt
    expect(afterStream).toBeGreaterThan(0)

    patchTurnState({ reasoning: 'because' })
    expect(getTurnPulse().lastDeltaAt).toBeGreaterThanOrEqual(afterStream)
  })

  it('does not treat the end-of-turn blanking of the stream as a delta', () => {
    beginTurnPulse(1_000)
    patchTurnState({ streaming: 'hello' })
    const stamped = getTurnPulse().lastDeltaAt

    patchTurnState({ streaming: '' })
    expect(getTurnPulse().lastDeltaAt).toBe(stamped)
  })

  it('keeps the original start when a turn already in flight re-opens the window', () => {
    beginTurnPulse(1_000)
    beginTurnPulse(9_000)

    expect(getTurnPulse().startedAt).toBe(1_000)

    endTurnPulse()
    expect(beginTurnPulse(9_000).startedAt).toBe(9_000)
  })

  it('clears the pulse when the turn store resets', () => {
    beginTurnPulse(1_000)
    resetTurnState()

    expect(getTurnPulse()).toEqual({ lastDeltaAt: 0, startedAt: 0 })
  })
})

describe('coarse liveness gate', () => {
  afterEach(() => {
    resetTurnState()
    resetUiState()
  })

  it('is live while the turn is busy or a tool is still running', () => {
    expect($turnLive.get()).toBe(false)

    patchUiState({ busy: true })
    expect($turnLive.get()).toBe(true)

    patchUiState({ busy: false })
    patchTurnState({ tools: [{ id: 't1', name: 'Read', context: '' }] })
    expect($turnLive.get()).toBe(true)
  })

  it('opens and closes the pulse window itself, so a remounted indicator cannot leak a stale start', () => {
    expect(getTurnPulse().startedAt).toBe(0)

    patchUiState({ busy: true })
    expect(getTurnPulse().startedAt).toBeGreaterThan(0)

    patchTurnState({ streaming: 'partial' })
    patchUiState({ busy: false })

    expect(getTurnPulse()).toEqual({ lastDeltaAt: 0, startedAt: 0 })
  })

  it('does not re-notify subscribers on every delta', () => {
    let notifications = 0
    const unlisten = $turnLive.listen(() => notifications++)

    patchUiState({ busy: true })
    for (let index = 0; index < 20; index += 1) {
      patchTurnState({ streaming: 'x'.repeat(index + 1) })
    }

    expect(notifications).toBe(1)
    unlisten()
  })
})
