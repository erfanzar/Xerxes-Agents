// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// What the composer promises while a turn is running.
//
// The composer used to say "Queue a follow-up…" and "Enter queue" no matter what
// `display.busy_input_mode` was set to, while the default mode is `steer` — so
// pressing Enter injected the message into the running turn and the interface
// said it had been held for later. Typing something to redirect the agent and
// being told it was queued is indistinguishable from steering not working, and it
// also hid the real fallback: when a steer is rejected the text genuinely is
// queued, and nothing in the label changed to say so.

import type { BusyInputMode } from '../app/interfaces.js'

export interface BusyInputLabels {
  /** What Enter does, for the footer hint. */
  readonly enter: string
  /** What Escape does when nothing is queued yet. */
  readonly escape: string
  readonly placeholder: string
}

/**
 * Labels for the active busy-input mode.
 *
 * `queuedCount` matters only for Escape: with text already held, Escape clears
 * that text rather than interrupting the turn, and saying "interrupt" there would
 * be the same class of lie.
 */
export function busyInputLabels(mode: BusyInputMode, queuedCount = 0): BusyInputLabels {
  const escape = queuedCount > 0 ? 'clear queue' : 'interrupt'
  if (mode === 'steer') {
    return {
      placeholder: 'Steer the running turn… (esc to interrupt)',
      enter: 'steer',
      escape,
    }
  }
  if (mode === 'interrupt') {
    return {
      placeholder: 'Interrupt with a new message… (esc to interrupt)',
      enter: 'interrupt',
      escape,
    }
  }
  return {
    placeholder: 'Queue a follow-up… (esc to interrupt)',
    enter: 'queue',
    escape,
  }
}
