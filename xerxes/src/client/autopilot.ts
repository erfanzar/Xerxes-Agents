// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { AutopilotRow } from './hubTypes.js'

/**
 * Shared Autopilot wording and choices for the desktop and the phone, so both
 * say the same thing. Autopilot answers a task's questions and steers, stops or
 * messages tasks on its own; starting a task, setting a goal, changing a model
 * and every permission request (commands, edits) stay with a person.
 */

export type AutopilotExpiryChoice = 'keep' | 'none' | '1h' | '8h' | '24h' | '7d'

export interface AutopilotExpiryOption {
  readonly value: AutopilotExpiryChoice
  readonly label: string
}

const HOUR = 3_600_000

const DURATIONS: Readonly<Record<Exclude<AutopilotExpiryChoice, 'keep' | 'none'>, number>> = {
  '1h': HOUR,
  '8h': 8 * HOUR,
  '24h': 24 * HOUR,
  '7d': 7 * 24 * HOUR,
}

/** True when an expiry exists and has already passed. */
export function isExpiryLapsed(expiresAt: string | null, nowMs: number): boolean {
  return expiresAt !== null && Date.parse(expiresAt) <= nowMs
}

/**
 * The choices offered for one row. `keep` exists only while the row has an
 * expiry that is still in the future: the Hub refuses to enable Autopilot on a
 * kept expiry that has passed.
 */
export function autopilotExpiryOptions(row: Pick<AutopilotRow, 'expiresAt'>, nowMs: number): readonly AutopilotExpiryOption[] {
  const canKeep = row.expiresAt !== null && !isExpiryLapsed(row.expiresAt, nowMs)
  return [
    ...(canKeep ? [{ value: 'keep' as const, label: 'Keep current expiry' }] : []),
    { value: 'none', label: 'No expiry' },
    { value: '1h', label: '1 hour' },
    { value: '8h', label: '8 hours' },
    { value: '24h', label: '24 hours' },
    { value: '7d', label: '7 days' },
  ]
}

/**
 * The choice a row starts on: a live expiry is kept, no expiry stays none, and
 * a lapsed expiry starts on a short new one rather than silently on "no expiry".
 */
export function defaultAutopilotExpiry(row: Pick<AutopilotRow, 'expiresAt'>, nowMs: number): AutopilotExpiryChoice {
  if (row.expiresAt === null) return 'none'
  return isExpiryLapsed(row.expiresAt, nowMs) ? '24h' : 'keep'
}

/** A chosen `keep` that is no longer on offer (no expiry, or it passed) falls back to the default. */
export function effectiveExpiryChoice(choice: AutopilotExpiryChoice, row: Pick<AutopilotRow, 'expiresAt'>, nowMs: number): AutopilotExpiryChoice {
  const canKeep = row.expiresAt !== null && !isExpiryLapsed(row.expiresAt, nowMs)
  return choice === 'keep' && !canKeep ? defaultAutopilotExpiry(row, nowMs) : choice
}

/**
 * The `expires_at` argument for a choice: `undefined` keeps what the Hub has,
 * `null` clears it, a string is an ISO time in the future.
 */
export function autopilotExpiresAt(choice: AutopilotExpiryChoice, nowMs: number): string | null | undefined {
  if (choice === 'keep') return undefined
  if (choice === 'none') return null
  return new Date(nowMs + DURATIONS[choice]).toISOString()
}

/** "Autopilot on · 2 workspaces", or undefined when no workspace is active. */
export function autopilotChipText(rows: readonly AutopilotRow[]): string | undefined {
  const active = rows.filter(row => row.active).length
  if (active === 0) return undefined
  return `Autopilot on · ${active} workspace${active === 1 ? '' : 's'}`
}

export const AUTOPILOT_EXPLANATION =
  'XERXES watches these workspaces and wakes by itself: it answers a task\u2019s questions and permission requests, and starts, steers, stops or messages tasks for you without a Confirm. Everything it does is posted to the thread.'

export const AUTOPILOT_READ_ONLY_NOTE = 'Only the Hub owner can change Autopilot. This is how it is set now.'

export const AUTOPILOT_INSTRUCTION_PLACEHOLDER = 'Keep going until the goal is done; take the recommended option'

/** XERXES acts for the owner without a Confirm, except while another person is in the thread. */
export const AUTOPILOT_COMPOSER_HINT = 'XERXES acts for you at once. If another person wrote in this thread recently, its actions wait for Confirm.'

export const AUTOPILOT_EXPIRY_LAPSED_NOTE = 'The previous expiry has passed. Choose a new one before you save.'

/** "Done automatically on behalf of Ann" for a card Autopilot created. */
export function autoCardNote(confirmedBy: string | undefined): string {
  return confirmedBy === undefined ? 'Done automatically by Autopilot.' : `Done automatically on behalf of ${confirmedBy}.`
}

// ── the owner's Autopilot thread ──────────────────────────────────────

/** The Hub's private thread for the owner's Autopilot wake turns is `autopilot-<ownerUserId>`. */
export const AUTOPILOT_THREAD_PREFIX = 'autopilot-'

export function autopilotThreadId(ownerUserId: string): string {
  return `${AUTOPILOT_THREAD_PREFIX}${ownerUserId}`
}

export const AUTOPILOT_THREAD_NOTE =
  'Private to you. Autopilot\u2019s wake-ups, its replies and the actions waiting for your Confirm appear here.'

export function pendingBadgeLabel(count: number): string {
  return `${count} waiting for you`
}
