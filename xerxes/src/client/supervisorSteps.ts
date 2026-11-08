// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SupervisorMessage, SupervisorStep } from './hubParsers.js'

/**
 * Live steps of a XERXES turn, shared by the desktop and the phone. The Hub
 * streams steps without storing them; a turn's steps are finished once a
 * XERXES reply (or the Hub's "XERXES stopped" notice) lands at or after the
 * last step. Notices posted mid-turn ("XERXES did: …") do not finish it.
 */

/** A message that ends a turn: XERXES's reply or the Hub's stop notice. */
export function endsTurn(message: Pick<SupervisorMessage, 'role' | 'text'>): boolean {
  return message.role === 'xerxes' || (message.role === 'system' && message.text.startsWith('XERXES stopped'))
}

export function stepsFinished(steps: readonly SupervisorStep[], messages: readonly SupervisorMessage[]): boolean {
  const last = steps.at(-1)
  if (last === undefined) return true
  const at = Date.parse(last.at)
  return messages.some(message => endsTurn(message) && Date.parse(message.at) >= at)
}

/** The steps of the turn still running, or none once it has finished. */
export function liveSteps(steps: readonly SupervisorStep[] | undefined, messages: readonly SupervisorMessage[]): readonly SupervisorStep[] {
  return steps !== undefined && steps.length > 0 && !stepsFinished(steps, messages) ? steps : []
}

/** What XERXES is doing right now, for the live status line; undefined before the first step. */
export function stepPhrase(steps: readonly SupervisorStep[]): string | undefined {
  const running = [...steps].reverse().find(step => step.kind === 'tool' && step.state === 'running')
  if (running !== undefined) return `${running.text}…`
  const last = steps.at(-1)
  if (last === undefined) return undefined
  return last.kind === 'say' ? last.text : 'Working…'
}
