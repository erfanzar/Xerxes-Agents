// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// The home screen's START WITH chips.
//
// One rule from the design canvas governs this whole file: "chips carry their
// own consequence — counts, ages, file totals — so the choice is informed
// before the keypress. A chip with nothing true to say is not shown."
//
// So every chip below is built from a signal the product actually has. The
// canvas also draws a "fix failing tests · 3 red in packages/runtime" chip;
// there is no test-status signal on the wire yet, and inventing the caption
// would break the rule the chip exists to demonstrate, so it is absent rather
// than decorative. Add the chip when the signal lands, not before.
import { diffLabel, type RepoPulse } from '../lib/repoPulse.js'
import type { NocturneState } from './nocturne.js'

export interface StartChip {
  /** The slash command this chip runs, if it is one. Rendered in the accent. */
  command?: string
  /** The consequence: what is true right now that makes this worth pressing. */
  consequence: string
  /** Stable identity, so the digit a chip answers to does not shuffle. */
  id: string
  /** What the chip says it does, in plain words. */
  label: string
  /** Text dropped into the composer. Never submitted — you still press ⏎. */
  prompt: string
  /** Which voice the chip's mark wears. */
  tone: NocturneState
}

export interface StartWithInput {
  /** Agents in a state that needs a human. */
  agentsNeedingInput: number
  /** Agents currently running. */
  agentsWorking: number
  /** Whether a model is configured at all. */
  hasModel: boolean
  pulse: RepoPulse
}

/**
 * Build the chip list, most-consequential first.
 *
 * Order is by what you have to do, the same rule the agents screen groups by:
 * a blocked setup step outranks a dirty tree, which outranks an invitation to
 * read the repo.
 */
export function startWithChips({
  agentsNeedingInput,
  agentsWorking,
  hasModel,
  pulse
}: StartWithInput): StartChip[] {
  const chips: StartChip[] = []

  if (!hasModel) {
    chips.push({
      consequence: 'nothing is configured yet',
      id: 'provider',
      command: '/provider',
      label: 'choose a model',
      prompt: '/provider',
      tone: 'needsInput'
    })
  }

  if (agentsNeedingInput > 0 || agentsWorking > 0) {
    const parts = [
      agentsWorking ? `${agentsWorking} working` : '',
      agentsNeedingInput ? `${agentsNeedingInput} need you` : ''
    ].filter(Boolean)

    chips.push({
      consequence: parts.join(' · '),
      id: 'agents',
      command: '/agents',
      label: 'check the fleet',
      prompt: '/agents',
      tone: agentsNeedingInput > 0 ? 'needsInput' : 'working'
    })
  }

  const diff = diffLabel(pulse)

  if (diff) {
    chips.push({
      consequence: diff,
      id: 'diff',
      command: '/diff',
      label: 'review the working tree',
      prompt: '/diff',
      tone: 'working'
    })
  } else if (pulse.ahead > 0) {
    // A clean tree that is ahead of upstream still owes you a read-through
    // before it becomes someone else's problem.
    chips.push({
      consequence: `${pulse.ahead} commit${pulse.ahead === 1 ? '' : 's'} ahead of upstream`,
      id: 'unpushed',
      label: 'summarise what is unpushed',
      prompt: 'Summarise the commits on this branch that are not yet on upstream, grouped by intent.',
      tone: 'working'
    })
  }

  // The one chip that is always available. Its caption is a description of
  // what it will produce rather than live state — the canvas draws it the
  // same way, because an empty screen still has to offer a way in.
  chips.push({
    consequence: 'entry points, hot paths, dead code',
    id: 'map',
    label: 'map this repo',
    prompt: 'Map this repository: entry points, the hot paths through it, and anything that looks dead.',
    tone: 'working'
  })

  return chips
}

/** The digit that activates a chip, or '' past the ninth. */
export const chipKey = (index: number): string => (index < 9 ? String(index + 1) : '')
