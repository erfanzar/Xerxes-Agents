// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { VoiceError } from './protocol.js'

/**
 * Spend control for voice. Spend is reserved before a provider is called and
 * settled afterwards. An unknown price is never treated as zero: when a limit
 * is configured, a call whose cost cannot be estimated is refused; without a
 * limit it is allowed but the usage report says the total is incomplete.
 */

export interface VoicePricing {
  readonly sttUsdPerMinute?: number
  readonly ttsUsdPerMillionChars?: number
  readonly realtimeUsdPerMinute?: number
}

export type VoiceMeter = 'stt' | 'tts' | 'realtime'

/** Estimated USD for `quantity` (minutes for stt/realtime, characters for tts); undefined when the price is unknown. */
export function estimateUsd(pricing: VoicePricing, meter: VoiceMeter, quantity: number): number | undefined {
  if (!Number.isFinite(quantity) || quantity < 0) return undefined
  switch (meter) {
    case 'stt': return pricing.sttUsdPerMinute === undefined ? undefined : pricing.sttUsdPerMinute * quantity
    case 'realtime': return pricing.realtimeUsdPerMinute === undefined ? undefined : pricing.realtimeUsdPerMinute * quantity
    case 'tts': return pricing.ttsUsdPerMillionChars === undefined ? undefined : (pricing.ttsUsdPerMillionChars * quantity) / 1_000_000
  }
}

export interface BudgetLimits {
  readonly maxSessionUsd?: number
  readonly maxDailyUserUsd?: number
}

export interface BudgetUsage {
  readonly sessionUsd: number
  readonly dailyUsd: number
  /** True when some spend could not be priced, so the totals are lower bounds. */
  readonly unknownCost: boolean
  readonly maxSessionUsd?: number
  readonly maxDailyUsd?: number
}

export interface Reservation {
  readonly id: number
  readonly userId: string
  readonly sessionId: string
  readonly meter: VoiceMeter
  readonly estimateUsd: number | undefined
}

interface Ledger {
  spent: number
  reserved: number
  unknown: boolean
}

interface DailyLedger extends Ledger {
  day: string
}

export interface VoiceBudgetOptions {
  readonly limits?: BudgetLimits
  readonly clock?: () => number
  /** Users tracked at once; stale days are evicted first. */
  readonly maxUsers?: number
}

export class VoiceBudget {
  private readonly limits: BudgetLimits
  private readonly clock: () => number
  private readonly maxUsers: number
  private readonly sessions = new Map<string, Ledger>()
  private readonly daily = new Map<string, DailyLedger>()
  private readonly open = new Map<number, Reservation>()
  private sequence = 0

  constructor(options: VoiceBudgetOptions = {}) {
    this.limits = options.limits ?? {}
    this.clock = options.clock ?? Date.now
    this.maxUsers = options.maxUsers ?? 10_000
    for (const [name, value] of Object.entries(this.limits)) {
      if (value !== undefined && (!Number.isFinite(value) || value < 0)) throw new RangeError(`${name} must be a non-negative number`)
    }
  }

  get limited(): boolean {
    return this.limits.maxSessionUsd !== undefined || this.limits.maxDailyUserUsd !== undefined
  }

  /** Reserve before calling a provider. Throws VOICE_BUDGET when the call would exceed a limit or cannot be priced under a limit. */
  reserve(userId: string, sessionId: string, meter: VoiceMeter, estimate: number | undefined): Reservation {
    const session = this.session(sessionId)
    const day = this.day(userId)
    if (estimate === undefined) {
      if (this.limited) throw new VoiceError('VOICE_BUDGET', `the ${meter} price is unknown, so the call cannot be checked against the spend limit`)
      session.unknown = true
      day.unknown = true
    } else {
      const { maxSessionUsd, maxDailyUserUsd } = this.limits
      if (maxSessionUsd !== undefined && session.spent + session.reserved + estimate > maxSessionUsd) {
        throw new VoiceError('VOICE_BUDGET', 'this voice session reached its spend limit')
      }
      if (maxDailyUserUsd !== undefined && day.spent + day.reserved + estimate > maxDailyUserUsd) {
        throw new VoiceError('VOICE_BUDGET', 'your daily voice spend limit was reached')
      }
      session.reserved += estimate
      day.reserved += estimate
    }
    const reservation: Reservation = { id: (this.sequence += 1), userId, sessionId, meter, estimateUsd: estimate }
    this.open.set(reservation.id, reservation)
    return reservation
  }

  /** Replace a reservation with the actual cost (or the estimate when the provider reports none). */
  settle(reservation: Reservation, actualUsd?: number): void {
    if (!this.open.delete(reservation.id)) return
    const session = this.session(reservation.sessionId)
    const day = this.day(reservation.userId)
    const reserved = reservation.estimateUsd ?? 0
    session.reserved = Math.max(0, session.reserved - reserved)
    day.reserved = Math.max(0, day.reserved - reserved)
    const final = actualUsd ?? reservation.estimateUsd
    if (final === undefined) {
      session.unknown = true
      day.unknown = true
      return
    }
    session.spent += final
    day.spent += final
  }

  /** Drop a reservation for a call that never reached the provider. */
  release(reservation: Reservation): void {
    if (!this.open.delete(reservation.id)) return
    const reserved = reservation.estimateUsd ?? 0
    const session = this.session(reservation.sessionId)
    const day = this.day(reservation.userId)
    session.reserved = Math.max(0, session.reserved - reserved)
    day.reserved = Math.max(0, day.reserved - reserved)
  }

  /** Record supervisor or agent spend that happened elsewhere; undefined marks the totals incomplete. */
  addExternalSpend(userId: string, sessionId: string, usd: number | undefined): void {
    const session = this.session(sessionId)
    const day = this.day(userId)
    if (usd === undefined) {
      session.unknown = true
      day.unknown = true
      return
    }
    session.spent += usd
    day.spent += usd
  }

  usage(userId: string, sessionId: string): BudgetUsage {
    const session = this.session(sessionId)
    const day = this.day(userId)
    return {
      sessionUsd: round(session.spent),
      dailyUsd: round(day.spent),
      unknownCost: session.unknown || day.unknown,
      ...(this.limits.maxSessionUsd === undefined ? {} : { maxSessionUsd: this.limits.maxSessionUsd }),
      ...(this.limits.maxDailyUserUsd === undefined ? {} : { maxDailyUsd: this.limits.maxDailyUserUsd }),
    }
  }

  endSession(sessionId: string): void {
    this.sessions.delete(sessionId)
    for (const [id, reservation] of this.open) if (reservation.sessionId === sessionId) this.open.delete(id)
  }

  private session(sessionId: string): Ledger {
    let ledger = this.sessions.get(sessionId)
    if (ledger === undefined) {
      ledger = { spent: 0, reserved: 0, unknown: false }
      this.sessions.set(sessionId, ledger)
    }
    return ledger
  }

  private day(userId: string): DailyLedger {
    const today = new Date(this.clock()).toISOString().slice(0, 10)
    let ledger = this.daily.get(userId)
    if (ledger === undefined || ledger.day !== today) {
      // Reserved amounts belong to calls still in flight; they carry over a midnight rollover.
      ledger = { day: today, spent: 0, reserved: ledger?.reserved ?? 0, unknown: false }
      this.daily.set(userId, ledger)
      if (this.daily.size > this.maxUsers) {
        for (const [id, row] of this.daily) {
          if (row.day !== today && row.reserved === 0) this.daily.delete(id)
          if (this.daily.size <= this.maxUsers) break
        }
      }
    }
    return ledger
  }
}

function round(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000
}
