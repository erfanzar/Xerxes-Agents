// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile, mkdir, readFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { parseJsonlEventLog, truncateTornTail } from '../core/jsonlEventLog.js'
import type { JsonObject } from '../types/toolCalls.js'

export type ReviewStatus = 'pending' | 'approved' | 'rejected'
export type SensitivityLevel = 'public' | 'internal' | 'confidential' | 'restricted'

export interface MemoryRecord {
  readonly id: string
  readonly content: string
  readonly source: string
  readonly agentId: string
  readonly turnId?: string
  readonly createdAt: number
  readonly reviewStatus: ReviewStatus
  readonly sensitivity: SensitivityLevel
  readonly expiresAt?: number
  readonly correctionOf?: string
  readonly correctionReason?: string
  /** Optional structured projection data; legacy records omit it. */
  readonly payload?: JsonObject
}

export interface GovernedMemoryState {
  readonly records: ReadonlyMap<string, MemoryRecord>
}

/** Read-only history projection; expiration never removes an identity here. */
export interface MemoryHistoryRecord extends MemoryRecord {
  readonly everApproved: boolean
  readonly reviewedAt?: number
  readonly reviewReason?: string
  readonly expiredAt?: number
  readonly expiredReason?: string
  /** Optional transport provenance for an imported retirement. */
  readonly expiredPayload?: JsonObject
}

export interface GovernedMemoryHistory {
  readonly records: ReadonlyMap<string, MemoryHistoryRecord>
}

export interface GovernedMemoryOptions {
  readonly directory: string
  readonly now?: () => number
}

export interface RecordMemoryOptions {
  id: string
  content: string
  source: string
  agentId: string
  turnId?: string
  sensitivity?: SensitivityLevel
  expiresAt?: number
  readonly correctionOf?: string
  readonly correctionReason?: string
  /** Optional structured projection data; legacy records omit it. */
  readonly payload?: JsonObject
}

export class GovernedMemoryStore {
  readonly logPath: string
  readonly now: () => number

  constructor(options: GovernedMemoryOptions) {
    this.logPath = join(options.directory, 'governed-memory.jsonl')
    this.now = options.now ?? (() => Date.now())
  }

  async load(): Promise<GovernedMemoryState> {
    const events = await this.readEvents()
    return project(events)
  }

  async loadHistory(): Promise<GovernedMemoryHistory> {
    const records = new Map<string, MemoryHistoryRecord>()
    for (const event of await this.readEvents()) {
      if (event.type === 'recorded') {
        const previous = records.get(event.record.id)
        records.set(event.record.id, { ...previous, ...event.record,
          everApproved: previous?.everApproved === true || event.record.reviewStatus === 'approved' })
        continue
      }
      const current = records.get(event.id)
      if (!current) continue
      if (event.type === 'reviewed') {
        const { reviewReason: _previousReason, ...record } = current
        records.set(event.id, { ...record, reviewStatus: event.status, reviewedAt: event.reviewedAt,
          everApproved: current.everApproved || event.status === 'approved', ...(event.reason === undefined ? {} : { reviewReason: event.reason }) })
      } else if (event.type === 'classified') {
        records.set(event.id, { ...current, sensitivity: event.sensitivity })
      } else if (event.type === 'payload_updated') {
        records.set(event.id, { ...current, payload: { ...current.payload, ...event.patch } })
      } else {
        const { expiredReason: _previousReason, expiredPayload: _previousPayload, ...record } = current
        records.set(event.id, { ...record, expiredAt: event.expiredAt,
          ...(event.reason === undefined ? {} : { expiredReason: event.reason }), ...(event.payload === undefined ? {} : { expiredPayload: event.payload }) })
      }
    }
    return { records }
  }

  async record(options: RecordMemoryOptions): Promise<MemoryRecord> {
    const record = buildRecord(options, this.now())
    const event: GovernedMemoryEvent = { type: 'recorded', record }
    await this.append(event)
    const state = await this.load()
    const persisted = state.records.get(options.id)
    if (persisted === undefined) throw new Error('memory record creation failed')
    return persisted
  }

  async review(id: string, status: ReviewStatus, reason?: string): Promise<void> {
    await this.append({ type: 'reviewed', id, status, reviewedAt: this.now(), ...(reason === undefined ? {} : { reason }) })
  }

  /** Sharing metadata is append-only too; never re-record or resurrect an expired identity. */
  async updatePayload(id: string, patch: JsonObject): Promise<void> {
    await this.append({ type: 'payload_updated', id, patch })
  }

  async classify(id: string, sensitivity: SensitivityLevel): Promise<void> {
    await this.append({ type: 'classified', id, sensitivity, classifiedAt: this.now() })
  }

  async correct(options: { readonly originalId: string; readonly newId: string; readonly content: string; readonly reason: string; readonly payload?: JsonObject }): Promise<MemoryRecord> {
    const original = (await this.load()).records.get(options.originalId)
    if (original === undefined) throw new Error('original memory record not found')
    const recordOptions: RecordMemoryOptions = {
      id: options.newId,
      content: options.content,
      source: original.source,
      agentId: original.agentId,
      sensitivity: original.sensitivity,
      correctionOf: options.originalId,
      correctionReason: options.reason,
    }
    if (original.turnId !== undefined) recordOptions.turnId = original.turnId
    if (original.expiresAt !== undefined) recordOptions.expiresAt = original.expiresAt
    const payload = options.payload ?? original.payload
    return this.record({ ...recordOptions, ...(payload === undefined ? {} : { payload }) })
  }

  async expire(id: string, reason?: string, provenance?: { readonly at: number; readonly payload: JsonObject }): Promise<void> {
    await this.append({ type: 'expired', id, expiredAt: provenance?.at ?? this.now(), ...(reason === undefined ? {} : { reason }), ...(provenance === undefined ? {} : { payload: provenance.payload }) })
  }

  private async append(event: GovernedMemoryEvent): Promise<void> {
    await mkdir(dirname(this.logPath), { recursive: true })
    // Repair a tail torn by an earlier crash before adding to it; appending
    // onto a partial record fuses the two into one malformed middle line.
    await truncateTornTail(this.logPath)
    await appendFile(this.logPath, JSON.stringify({ ...event, loggedAt: this.now() }) + '\n', 'utf8')
  }

  private async readEvents(): Promise<readonly GovernedMemoryEvent[]> {
    let text = ''
    try { text = await readFile(this.logPath, 'utf8') } catch (error) {
      if (!isMissing(error)) throw error
    }
    // Tolerate only a torn final record — see parseJsonlEventLog. A crash
    // during an append otherwise made every later read and write throw.
    const { events } = parseJsonlEventLog(text, {
      label: 'governed memory event log',
      isValid: isGovernedMemoryEvent,
    })
    return events
  }
}

type GovernedMemoryEvent =
  | { readonly type: 'recorded'; readonly record: MemoryRecord }
  | { readonly type: 'reviewed'; readonly id: string; readonly status: ReviewStatus; readonly reviewedAt: number; readonly reason?: string }
  | { readonly type: 'classified'; readonly id: string; readonly sensitivity: SensitivityLevel; readonly classifiedAt: number }
  | { readonly type: 'payload_updated'; readonly id: string; readonly patch: JsonObject }
  | { readonly type: 'expired'; readonly id: string; readonly expiredAt: number; readonly reason?: string; readonly payload?: JsonObject }

function buildRecord(options: RecordMemoryOptions, now: number): MemoryRecord {
  const base = {
    id: options.id,
    content: options.content,
    source: options.source,
    agentId: options.agentId,
    createdAt: now,
    reviewStatus: 'pending' as const,
    sensitivity: options.sensitivity ?? 'internal' as SensitivityLevel,
  }
  const extras: { turnId?: string; expiresAt?: number; correctionOf?: string; correctionReason?: string; payload?: JsonObject } = {}
  if (options.turnId !== undefined) extras.turnId = options.turnId
  if (options.expiresAt !== undefined) extras.expiresAt = options.expiresAt
  if (options.correctionOf !== undefined) extras.correctionOf = options.correctionOf
  if (options.correctionReason !== undefined) extras.correctionReason = options.correctionReason
  if (options.payload !== undefined) extras.payload = options.payload
  return { ...base, ...extras } as MemoryRecord
}

function project(events: readonly GovernedMemoryEvent[]): GovernedMemoryState {
  const records = new Map<string, MemoryRecord>()
  for (const event of events) {
    switch (event.type) {
      case 'recorded': {
        records.set(event.record.id, event.record)
        break
      }
      case 'reviewed': {
        const existing = records.get(event.id)
        if (existing !== undefined) {
          records.set(event.id, { ...existing, reviewStatus: event.status })
        }
        break
      }
      case 'classified': {
        const existing = records.get(event.id)
        if (existing !== undefined) {
          records.set(event.id, { ...existing, sensitivity: event.sensitivity })
        }
        break
      }
      case 'payload_updated': {
        const existing = records.get(event.id)
        if (existing) records.set(event.id, { ...existing, payload: { ...existing.payload, ...event.patch } })
        break
      }
      case 'expired': {
        records.delete(event.id)
        break
      }
    }
  }
  return { records }
}

function isGovernedMemoryEvent(value: unknown): value is GovernedMemoryEvent {
  if (!isRecord(value) || typeof value.type !== 'string') return false
  switch (value.type) {
    case 'recorded':
      return isRecord(value.record) && typeof value.record.id === 'string'
    case 'payload_updated':
      return typeof value.id === 'string' && isRecord(value.patch)
    case 'reviewed':
    case 'classified':
    case 'expired':
      return typeof value.id === 'string'
    default:
      return false
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isMissing(error: unknown): boolean {
  return error instanceof Error && 'code' in error && (error as { code: unknown }).code === 'ENOENT'
}
