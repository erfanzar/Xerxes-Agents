// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/**
 * Pure liveness model behind the TUI's live turn indicator.
 *
 * Everything here is a function of wall-clock stamps, so the indicator can be
 * exercised in a unit test with no terminal, no renderer, and no timers.
 */

import { compactStatusNumber, formatStatusDuration } from './statusSnapshot.js'
import { estimateTokensRough } from './text.js'

/** Silence this long with nothing running means the turn looks wedged. */
export const LIVENESS_STALL_MS = 3_000
/** One full dim → bright → dim breath of the indicator glyph. */
export const LIVENESS_PULSE_MS = 1_200
/** How long a spinner verb holds before the next one takes over. */
export const LIVENESS_VERB_MS = 4_000
/** Token counts are noise on a short turn; they only earn their space later. */
export const LIVENESS_TOKENS_AFTER_MS = 30_000

/** A stalled turn keeps a faint heartbeat, so it reads as waiting, not dead. */
const STALLED_INTENSITY_CEILING = 0.35
/** Increasing visual weight; the triangle ramp walks up and back down them. */
const PULSE_FRAMES = ['·', '◇', '◈', '◆'] as const
const STALLED_GLYPH = '◌'
const FALLBACK_VERB = 'working'

export type LivenessPhase = 'stalled' | 'streaming' | 'tool'

export interface LivenessInput {
  /** Wall clock of the most recent streaming delta; 0 before the first one. */
  lastDeltaAt: number
  now: number
  /** Wall clock the current busy stretch began; 0 while idle. */
  startedAt: number
  /** Tools the runtime is executing right now. */
  toolCount: number
}

export interface Liveness {
  elapsedMs: number
  intensity: number
  phase: LivenessPhase
  stallMs: number
}

/**
 * Ramped from the clock rather than from a frame counter: when the terminal
 * drops frames — a huge paste, a laggy SSH link — a counter-driven ramp slows
 * to a crawl and the indicator ends up looking as stuck as the turn it is
 * supposed to vouch for. Sampling the clock just skips ahead instead.
 */
const pulseIntensity = (elapsedMs: number, phase: LivenessPhase): number => {
  const fraction = (Math.max(0, elapsedMs) % LIVENESS_PULSE_MS) / LIVENESS_PULSE_MS
  const triangle = 1 - Math.abs(1 - 2 * fraction)

  return phase === 'stalled' ? triangle * STALLED_INTENSITY_CEILING : triangle
}

/** Classify what the current turn is doing and how hard the glyph should beat. */
export const describeLiveness = ({ lastDeltaAt, now, startedAt, toolCount }: LivenessInput): Liveness => {
  const elapsedMs = startedAt > 0 ? Math.max(0, now - startedAt) : 0
  // Before the first delta the turn start is the baseline: a long pre-token
  // wait is exactly when the user needs to hear that Esc still works, so it
  // must not masquerade as healthy streaming.
  const since = lastDeltaAt > 0 ? lastDeltaAt : startedAt
  const stallMs = since > 0 ? Math.max(0, now - since) : 0
  const phase: LivenessPhase = toolCount > 0 ? 'tool' : stallMs >= LIVENESS_STALL_MS ? 'stalled' : 'streaming'

  return { elapsedMs, intensity: pulseIntensity(elapsedMs, phase), phase, stallMs }
}

/** Pick the verb on screen, rotating on the clock so dropped frames don't stick. */
export const livenessVerb = (verbs: readonly string[], elapsedMs: number): string => {
  const usable = verbs.map(verb => verb.trim()).filter(Boolean)

  if (!usable.length) {
    return FALLBACK_VERB
  }

  return usable[Math.floor(Math.max(0, elapsedMs) / LIVENESS_VERB_MS) % usable.length]!
}

/** The one-character pulse; a stalled turn freezes on a hollow ring. */
export const livenessGlyph = (phase: LivenessPhase, intensity: number): string => {
  if (phase === 'stalled') {
    return STALLED_GLYPH
  }

  const index = Math.floor(Math.max(0, Math.min(1, intensity)) * PULSE_FRAMES.length)

  return PULSE_FRAMES[Math.min(PULSE_FRAMES.length - 1, index)]!
}

/** Rough assistant-side token volume for the turn so far. */
export const livenessTokens = (turn: { reasoningTokens: number; streaming: string; toolTokens: number }): number =>
  turn.reasoningTokens + turn.toolTokens + estimateTokensRough(turn.streaming)

export interface LivenessLabelInput {
  tokens: number
  verbs: readonly string[]
}

/** Render the indicator's text: verb, elapsed clock, and the earned extras. */
export const livenessLabel = (liveness: Liveness, { tokens, verbs }: LivenessLabelInput): string => {
  const parts = [`${livenessVerb(verbs, liveness.elapsedMs)}… ${formatStatusDuration(liveness.elapsedMs / 1000)}`]

  if (liveness.elapsedMs >= LIVENESS_TOKENS_AFTER_MS && tokens > 0) {
    parts.push(`${compactStatusNumber(tokens)} tokens`)
  }

  if (liveness.phase === 'stalled') {
    parts.push('Esc to interrupt')
  }

  return parts.join(' · ')
}
