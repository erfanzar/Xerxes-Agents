// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { MemoryStorage } from './storage.js'

export const PROFILE_KEY_PREFIX = '_profile_'

export interface ConfidentValueOptions {
  readonly confidence?: number
  readonly evidenceCount?: number
  readonly lastUpdated?: Date
}

export interface ConfidentValueRecord extends Record<string, unknown> {
  readonly confidence: number
  readonly evidence_count: number
  readonly last_updated: string
  readonly value: unknown
}

/** A profile fact paired with evidence count, confidence, and recency. */
export class ConfidentValue {
  confidence: number
  evidenceCount: number
  lastUpdated: Date

  constructor(
    readonly value: unknown,
    options: ConfidentValueOptions = {},
  ) {
    this.confidence = clampConfidence(options.confidence ?? 0)
    this.evidenceCount = integer(options.evidenceCount ?? 0)
    this.lastUpdated = options.lastUpdated ?? new Date()
  }

  demote(weight = 0.5): void {
    this.confidence = Math.max(0, this.confidence - weight)
    this.lastUpdated = new Date()
  }

  reinforce(weight = 1): void {
    this.confidence = Math.min(1, this.confidence + weight)
    this.evidenceCount += 1
    this.lastUpdated = new Date()
  }

  toRecord(): ConfidentValueRecord {
    return {
      value: this.value,
      confidence: this.confidence,
      last_updated: this.lastUpdated.toISOString(),
      evidence_count: this.evidenceCount,
    }
  }

  static fromRecord(data: Record<string, unknown>): ConfidentValue {
    return new ConfidentValue(data.value, {
      confidence: number(data.confidence, 0),
      lastUpdated: parseDate(data.last_updated),
      evidenceCount: integer(data.evidence_count),
    })
  }
}

export interface ProfileFeedback extends Record<string, unknown> {
  readonly delta: number
  readonly signal: string
  readonly target: string
  readonly timestamp: string
}

export interface UserProfileOptions {
  readonly domains?: readonly string[]
  readonly explicitPreferences?: ReadonlyMap<string, ConfidentValue>
  readonly expertise?: ReadonlyMap<string, ConfidentValue>
  readonly feedbackHistory?: readonly ProfileFeedback[]
  readonly implicitPreferences?: ReadonlyMap<string, ConfidentValue>
  readonly lastSeen?: Date
  readonly notes?: readonly string[]
  readonly recurringGoals?: readonly string[]
  readonly tone?: ConfidentValue
  readonly userId: string
}

export interface UserProfileRenderOptions {
  readonly maxLines?: number
  readonly minConfidence?: number
}

export interface UserProfileRecord extends Record<string, unknown> {
  readonly domains: readonly string[]
  readonly explicit_preferences: Record<string, ConfidentValueRecord>
  readonly expertise: Record<string, ConfidentValueRecord>
  readonly feedback_history: readonly ProfileFeedback[]
  readonly implicit_preferences: Record<string, ConfidentValueRecord>
  readonly last_seen: string
  readonly notes: readonly string[]
  readonly recurring_goals: readonly string[]
  readonly tone: ConfidentValueRecord | null
  readonly user_id: string
}

/** Long-lived user facts and preferences with confidence-weighted rendering. */
export class UserProfile {
  readonly domains: string[]
  readonly explicitPreferences: Map<string, ConfidentValue>
  readonly expertise: Map<string, ConfidentValue>
  readonly feedbackHistory: ProfileFeedback[]
  readonly implicitPreferences: Map<string, ConfidentValue>
  lastSeen: Date
  readonly notes: string[]
  readonly recurringGoals: string[]
  tone: ConfidentValue | undefined
  readonly userId: string

  constructor(options: UserProfileOptions) {
    this.userId = options.userId
    this.expertise = new Map(options.expertise)
    this.domains = [...(options.domains ?? [])]
    this.tone = options.tone
    this.recurringGoals = [...(options.recurringGoals ?? [])]
    this.explicitPreferences = new Map(options.explicitPreferences)
    this.implicitPreferences = new Map(options.implicitPreferences)
    this.notes = [...(options.notes ?? [])]
    this.lastSeen = options.lastSeen ?? new Date()
    this.feedbackHistory = [...(options.feedbackHistory ?? [])]
  }

  recordFeedback(signal: string, options: { readonly delta?: number; readonly target?: string } = {}): void {
    this.feedbackHistory.push({
      signal,
      target: options.target ?? '',
      delta: options.delta ?? 1,
      timestamp: new Date().toISOString(),
    })
    if (this.feedbackHistory.length > 256) {
      this.feedbackHistory.splice(0, this.feedbackHistory.length - 128)
    }
  }

  render(options: UserProfileRenderOptions = {}): string {
    const maxLines = Math.max(0, integer(options.maxLines ?? 12))
    const minConfidence = options.minConfidence ?? 0.3
    const lines: string[] = []
    const add = (line: string): void => {
      if (lines.length < maxLines) lines.push(line)
    }

    if (this.domains.length > 0) add(`- Active domains: ${this.domains.slice(0, 5).join(', ')}`)
    if (this.tone && this.tone.confidence >= minConfidence) {
      add(`- Preferred tone: ${displayValue(this.tone.value)} (confidence ${this.tone.confidence.toFixed(2)})`)
    }
    for (const [topic, value] of this.expertise) {
      if (value.confidence >= minConfidence) {
        add(`- Expertise in ${topic}: ${displayValue(value.value)} (confidence ${value.confidence.toFixed(2)})`)
      }
    }
    for (const [preference, value] of this.explicitPreferences) {
      if (value.confidence >= minConfidence) add(`- Prefers ${preference}: ${displayValue(value.value)}`)
    }
    for (const [preference, value] of this.implicitPreferences) {
      if (value.confidence >= minConfidence) {
        add(`- Likely prefers ${preference}: ${displayValue(value.value)} (inferred)`)
      }
    }
    if (this.recurringGoals.length > 0) add(`- Recurring goals: ${this.recurringGoals.slice(0, 3).join('; ')}`)
    for (const note of this.notes) add(`- Note: ${note}`)
    return lines.join('\n')
  }

  toRecord(): UserProfileRecord {
    return {
      user_id: this.userId,
      domains: [...this.domains],
      recurring_goals: [...this.recurringGoals],
      notes: [...this.notes],
      last_seen: this.lastSeen.toISOString(),
      feedback_history: this.feedbackHistory.map(feedback => ({ ...feedback })),
      tone: this.tone?.toRecord() ?? null,
      expertise: confidentValuesRecord(this.expertise),
      explicit_preferences: confidentValuesRecord(this.explicitPreferences),
      implicit_preferences: confidentValuesRecord(this.implicitPreferences),
    }
  }

  static fromRecord(data: Record<string, unknown>): UserProfile {
    const userId = typeof data.user_id === 'string' && data.user_id ? data.user_id : undefined
    if (!userId) throw new Error('UserProfile record requires a user_id')
    const tone = isRecord(data.tone) ? ConfidentValue.fromRecord(data.tone) : undefined
    return new UserProfile({
      userId,
      domains: strings(data.domains),
      recurringGoals: strings(data.recurring_goals),
      notes: strings(data.notes),
      lastSeen: parseDate(data.last_seen),
      feedbackHistory: profileFeedback(data.feedback_history),
      ...(tone ? { tone } : {}),
      expertise: confidentValues(data.expertise),
      explicitPreferences: confidentValues(data.explicit_preferences),
      implicitPreferences: confidentValues(data.implicit_preferences),
    })
  }
}

export interface ProfileDecayOptions {
  readonly halfLifeDays?: number
  readonly pruneThreshold?: number
}

/** In-process profile registry backed by an optional MemoryStorage. */
export class UserProfileStore {
  private readonly profiles = new Map<string, UserProfile>()

  constructor(readonly storage?: MemoryStorage) {
    this.hydrate()
  }

  allUserIds(): string[] {
    return [...this.profiles.keys()]
  }

  decayAll(options: ProfileDecayOptions = {}): Record<string, number> {
    const now = new Date()
    const halfLifeDays = options.halfLifeDays ?? 30
    const pruneThreshold = options.pruneThreshold ?? 0.05
    const prunes: Record<string, number> = {}
    for (const [userId, profile] of this.profiles) {
      const pruned = decayProfile(profile, now, halfLifeDays, pruneThreshold)
      prunes[userId] = pruned
      if (pruned > 0) this.save(profile)
    }
    return prunes
  }

  delete(userId: string): boolean {
    const removed = this.profiles.delete(userId)
    if (!this.storage) return removed
    try {
      this.storage.delete(`${PROFILE_KEY_PREFIX}${userId}`)
    } catch {
      // Persistence is intentionally best-effort so an unavailable store cannot keep stale profiles live.
    }
    return removed
  }

  get(userId: string): UserProfile | undefined {
    return this.profiles.get(userId)
  }

  getOrCreate(userId: string): UserProfile {
    const existing = this.profiles.get(userId)
    if (existing) return existing
    const profile = new UserProfile({ userId })
    this.save(profile)
    return profile
  }

  renderFor(userId: string, options: UserProfileRenderOptions = {}): string {
    return this.profiles.get(userId)?.render(options) ?? ''
  }

  save(profile: UserProfile): void {
    profile.lastSeen = new Date()
    this.profiles.set(profile.userId, profile)
    if (!this.storage) return
    try {
      this.storage.save(`${PROFILE_KEY_PREFIX}${profile.userId}`, profile.toRecord())
    } catch {
      // Profile collection remains usable when optional durable storage is temporarily unavailable.
    }
  }

  private hydrate(): void {
    if (!this.storage) return
    let keys: string[]
    try {
      keys = this.storage.listKeys(PROFILE_KEY_PREFIX)
    } catch {
      return
    }
    for (const key of keys) {
      if (!key.startsWith(PROFILE_KEY_PREFIX)) continue
      try {
        const data = this.storage.load(key)
        if (!isRecord(data)) continue
        const profile = UserProfile.fromRecord(data)
        this.profiles.set(profile.userId, profile)
      } catch {
        // One invalid persisted row must not prevent unrelated profiles from loading.
      }
    }
  }
}

function clampConfidence(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function confidentValues(value: unknown): Map<string, ConfidentValue> {
  if (!isRecord(value)) return new Map()
  const entries: Array<readonly [string, ConfidentValue]> = []
  for (const [key, record] of Object.entries(value)) {
    if (isRecord(record)) entries.push([key, ConfidentValue.fromRecord(record)])
  }
  return new Map(entries)
}

function confidentValuesRecord(values: ReadonlyMap<string, ConfidentValue>): Record<string, ConfidentValueRecord> {
  const record: Record<string, ConfidentValueRecord> = {}
  for (const [key, value] of values) record[key] = value.toRecord()
  return record
}

function decayProfile(profile: UserProfile, now: Date, halfLifeDays: number, pruneThreshold: number): number {
  let pruned = 0
  if (profile.tone) {
    decayValue(profile.tone, now, halfLifeDays)
    if (profile.tone.confidence < pruneThreshold) {
      profile.tone = undefined
      pruned += 1
    }
  }
  for (const values of [profile.expertise, profile.explicitPreferences, profile.implicitPreferences]) {
    for (const [key, value] of values) {
      decayValue(value, now, halfLifeDays)
      if (value.confidence < pruneThreshold) {
        values.delete(key)
        pruned += 1
      }
    }
  }
  return pruned
}

function decayValue(value: ConfidentValue, now: Date, halfLifeDays: number): void {
  const ageDays = Math.max(0, (now.valueOf() - value.lastUpdated.valueOf()) / 86_400_000)
  const factor = Math.pow(0.5, ageDays / Math.max(halfLifeDays, 0.001))
  value.confidence = Math.max(0, value.confidence * factor)
}

function displayValue(value: unknown): string {
  return typeof value === 'string' ? value : JSON.stringify(value)
}

function integer(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.trunc(value) : 0
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function number(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function parseDate(value: unknown): Date {
  if (value instanceof Date) return value
  if (typeof value === 'string') {
    const parsed = new Date(value)
    if (!Number.isNaN(parsed.valueOf())) return parsed
  }
  return new Date()
}

function profileFeedback(value: unknown): ProfileFeedback[] {
  if (!Array.isArray(value)) return []
  const feedback: ProfileFeedback[] = []
  for (const item of value) {
    if (!isRecord(item) || typeof item.signal !== 'string') continue
    feedback.push({
      ...item,
      signal: item.signal,
      target: typeof item.target === 'string' ? item.target : '',
      delta: number(item.delta, 1),
      timestamp: typeof item.timestamp === 'string' ? item.timestamp : new Date().toISOString(),
    })
  }
  return feedback
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}
