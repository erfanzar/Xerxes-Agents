// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SupervisorConfirmation, SupervisorMessage } from './hubParsers.js'
import { clip } from './parse.js'

/**
 * XERXES action cards in the conversation timeline, shared by the desktop and
 * the phone. A card that still waits for a person stays a card below the
 * conversation; a settled one (done, cancelled, expired, failed) becomes one
 * quiet event line placed at the time it was made, so records never pile up
 * under newer replies.
 */

/** A card that still waits for a person. */
export function isOpenCard(card: Pick<SupervisorConfirmation, 'state'>): boolean {
  return card.state === 'pending' || card.state === 'confirming'
}

/** A settled card as one event line. */
export function recordLine(card: SupervisorConfirmation, fallbackAt: string): SupervisorMessage {
  const what = clip(`${card.tool}: ${card.summary}`, 220)
  const text = card.state === 'failed' ? `XERXES failed: ${what}${card.message ? ` (${clip(card.message, 160)})` : ''}`
    : card.state === 'cancelled' ? `Cancelled: ${what}`
    : card.state === 'expired' ? `Expired: ${what}`
    : card.auto === true ? `XERXES did: ${what}` : `Confirmed: ${what}`
  return { id: `card:${card.id}`, role: 'system', text, at: card.createdAt ?? fallbackAt, source: 'text' }
}

/** Messages with settled cards placed at their own time; a card from an older Hub (no time) goes after the newest message. */
export function withRecords(messages: readonly SupervisorMessage[], records: readonly SupervisorConfirmation[]): readonly SupervisorMessage[] {
  if (records.length === 0) return messages
  const fallback = messages.at(-1)?.at ?? new Date(0).toISOString()
  return [...messages, ...records.map(card => recordLine(card, fallback))]
    .map((message, index) => ({ message, index }))
    .sort((a, b) => (Date.parse(a.message.at) - Date.parse(b.message.at)) || a.index - b.index)
    .map(entry => entry.message)
}

/** Split a thread's cards: open ones stay cards, the rest are timeline records. */
export function splitCards(cards: readonly SupervisorConfirmation[]): { readonly open: readonly SupervisorConfirmation[]; readonly records: readonly SupervisorConfirmation[] } {
  return { open: cards.filter(isOpenCard), records: cards.filter(card => !isOpenCard(card)) }
}
