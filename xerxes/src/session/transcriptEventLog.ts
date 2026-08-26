// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Current schema for append-only daemon transcript events. */
export const TRANSCRIPT_EVENT_SCHEMA_VERSION = 1

export type TranscriptEventMessage = Readonly<Record<string, unknown>>
export interface TranscriptEventIdentity { readonly eventId: string; readonly sequence: number }
interface TranscriptEventBase extends TranscriptEventIdentity {
  readonly eventSchemaVersion: typeof TRANSCRIPT_EVENT_SCHEMA_VERSION
  readonly sessionId: string
}

export interface TranscriptMessageAppendedEvent {
  readonly eventId?: string
  readonly eventSchemaVersion: typeof TRANSCRIPT_EVENT_SCHEMA_VERSION
  readonly index: number
  readonly message: TranscriptEventMessage
  readonly sequence?: number
  readonly sessionId: string
  readonly type: 'message_appended'
}
export interface TranscriptTurnStartedEvent extends TranscriptEventBase {
  readonly details: Readonly<{ readonly mode?: string }>
  readonly turnId: string
  readonly type: 'turn_started'
}
export interface TranscriptRequestPreparedEvent extends TranscriptEventBase {
  readonly details: Readonly<{ readonly ephemeralFields?: readonly string[]; readonly model: string; readonly provider: string }>
  readonly turnId: string
  readonly type: 'request_prepared'
}
export interface TranscriptContextComposedEvent extends TranscriptEventBase {
  readonly details: Readonly<{ readonly messageCount: number; readonly modelVisible: boolean; readonly sources: readonly string[] }>
  readonly turnId: string
  readonly type: 'context_composed'
}
export interface TranscriptPolicyDecidedEvent extends TranscriptEventBase {
  readonly details: Readonly<{ readonly action: string; readonly decision: 'allow' | 'deny' | 'ask'; readonly reason?: string }>
  readonly turnId: string
  readonly type: 'policy_decided'
}
export interface TranscriptToolStartedEvent extends TranscriptEventBase {
  readonly details: Readonly<{ readonly arguments: Readonly<Record<string, unknown>>; readonly callId: string; readonly name: string }>
  readonly turnId: string
  readonly type: 'tool_started'
}
export interface TranscriptToolCompletedEvent extends TranscriptEventBase {
  readonly details: Readonly<{ readonly callId: string; readonly name: string; readonly ok: boolean; readonly result?: unknown }>
  readonly turnId: string
  readonly type: 'tool_completed'
}
export interface TranscriptTurnCompletedEvent extends TranscriptEventBase {
  readonly details: Readonly<{
    readonly stopReason: string
    readonly usage?: Readonly<{ readonly inputTokens: number; readonly outputTokens: number }>
  }>
  readonly turnId: string
  readonly type: 'turn_completed'
}

export type TranscriptEvent = TranscriptMessageAppendedEvent
  | TranscriptTurnStartedEvent | TranscriptRequestPreparedEvent | TranscriptContextComposedEvent
  | TranscriptPolicyDecidedEvent | TranscriptToolStartedEvent | TranscriptToolCompletedEvent
  | TranscriptTurnCompletedEvent

export interface TranscriptEventReadResult {
  readonly events: readonly TranscriptEvent[]
  readonly malformedLines: number
  readonly partialTail: boolean
}
export interface TranscriptEventRecord {
  readonly endOffset: number
  readonly event: TranscriptEvent
}
export interface TranscriptEventRecordReadResult extends TranscriptEventReadResult {
  readonly records: readonly TranscriptEventRecord[]
}

export function transcriptMessageAppendedEvent(
  sessionId: string, index: number, message: TranscriptEventMessage, identity?: TranscriptEventIdentity,
): TranscriptMessageAppendedEvent {
  if (!sessionId || !Number.isSafeInteger(index) || index < 0 || !isRecord(message)
    || (identity !== undefined && !validIdentity(identity))) throw new TypeError('invalid transcript message event')
  return { ...(identity ?? {}), eventSchemaVersion: 1, index, message: { ...message }, sessionId, type: 'message_appended' }
}

export function transcriptTurnStartedEvent(
  sessionId: string, turnId: string, details: TranscriptTurnStartedEvent['details'], identity: TranscriptEventIdentity,
): TranscriptTurnStartedEvent {
  assertLifecycle(sessionId, turnId, identity, isRecord(details))
  return lifecycle('turn_started', sessionId, turnId, { ...details }, identity)
}
export function transcriptRequestPreparedEvent(
  sessionId: string, turnId: string, details: TranscriptRequestPreparedEvent['details'], identity: TranscriptEventIdentity,
): TranscriptRequestPreparedEvent {
  assertLifecycle(sessionId, turnId, identity, typeof details.model === 'string' && typeof details.provider === 'string')
  return lifecycle('request_prepared', sessionId, turnId, { ...details }, identity)
}
export function transcriptContextComposedEvent(
  sessionId: string, turnId: string, details: TranscriptContextComposedEvent['details'], identity: TranscriptEventIdentity,
): TranscriptContextComposedEvent {
  assertLifecycle(sessionId, turnId, identity, Number.isSafeInteger(details.messageCount) && details.messageCount >= 0
    && typeof details.modelVisible === 'boolean' && stringArray(details.sources))
  return lifecycle('context_composed', sessionId, turnId, { ...details }, identity)
}
export function transcriptPolicyDecidedEvent(
  sessionId: string, turnId: string, details: TranscriptPolicyDecidedEvent['details'], identity: TranscriptEventIdentity,
): TranscriptPolicyDecidedEvent {
  assertLifecycle(sessionId, turnId, identity, typeof details.action === 'string'
    && ['allow', 'deny', 'ask'].includes(details.decision))
  return lifecycle('policy_decided', sessionId, turnId, { ...details }, identity)
}
export function transcriptToolStartedEvent(
  sessionId: string, turnId: string, details: TranscriptToolStartedEvent['details'], identity: TranscriptEventIdentity,
): TranscriptToolStartedEvent {
  assertLifecycle(sessionId, turnId, identity, !!details.callId && !!details.name && isRecord(details.arguments))
  return lifecycle('tool_started', sessionId, turnId, { ...details }, identity)
}
export function transcriptToolCompletedEvent(
  sessionId: string, turnId: string, details: TranscriptToolCompletedEvent['details'], identity: TranscriptEventIdentity,
): TranscriptToolCompletedEvent {
  assertLifecycle(sessionId, turnId, identity, !!details.callId && !!details.name && typeof details.ok === 'boolean')
  return lifecycle('tool_completed', sessionId, turnId, { ...details }, identity)
}
export function transcriptTurnCompletedEvent(
  sessionId: string, turnId: string, details: TranscriptTurnCompletedEvent['details'], identity: TranscriptEventIdentity,
): TranscriptTurnCompletedEvent {
  assertLifecycle(sessionId, turnId, identity, !!details.stopReason)
  return lifecycle('turn_completed', sessionId, turnId, { ...details }, identity)
}

function lifecycle<T extends TranscriptEvent['type'], D>(
  type: T, sessionId: string, turnId: string, details: D, identity: TranscriptEventIdentity,
): { readonly type: T; readonly sessionId: string; readonly turnId: string; readonly details: D } & TranscriptEventBase {
  return { ...identity, details, eventSchemaVersion: 1, sessionId, turnId, type }
}

export function encodeTranscriptEvent(event: TranscriptEvent): string {
  const common = {
    event_schema_version: event.eventSchemaVersion,
    type: event.type,
    session_id: event.sessionId,
    ...(event.eventId === undefined ? {} : { event_id: event.eventId }),
    ...(event.sequence === undefined ? {} : { sequence: event.sequence }),
  }
  return `${JSON.stringify(event.type === 'message_appended'
    ? { ...common, index: event.index, message: event.message }
    : { ...common, turn_id: event.turnId, details: event.details })}\n`
}

export function parseTranscriptEvent(raw: unknown, expectedSessionId: string): TranscriptEvent | undefined {
  if (!isRecord(raw)) return undefined
  if (!('type' in raw) && !('event_schema_version' in raw) && !('session_id' in raw)) {
    return validMessage(raw) ? transcriptMessageAppendedEvent(expectedSessionId, Number(raw.index), raw.message) : undefined
  }
  if (raw.event_schema_version !== 1 || raw.session_id !== expectedSessionId || typeof raw.type !== 'string') return undefined
  const identity = parseIdentity(raw)
  if (raw.type === 'message_appended') {
    if (!validMessage(raw)) return undefined
    if (!identity && ('event_id' in raw || 'sequence' in raw)) return undefined
    return transcriptMessageAppendedEvent(expectedSessionId, Number(raw.index), raw.message, identity)
  }
  if (!identity || typeof raw.turn_id !== 'string' || !raw.turn_id || !isRecord(raw.details)) return undefined
  try {
    switch (raw.type) {
      case 'turn_started': return transcriptTurnStartedEvent(expectedSessionId, raw.turn_id, raw.details, identity)
      case 'request_prepared': return transcriptRequestPreparedEvent(expectedSessionId, raw.turn_id, raw.details as TranscriptRequestPreparedEvent['details'], identity)
      case 'context_composed': return transcriptContextComposedEvent(expectedSessionId, raw.turn_id, raw.details as unknown as TranscriptContextComposedEvent['details'], identity)
      case 'policy_decided': return transcriptPolicyDecidedEvent(expectedSessionId, raw.turn_id, raw.details as unknown as TranscriptPolicyDecidedEvent['details'], identity)
      case 'tool_started': return transcriptToolStartedEvent(expectedSessionId, raw.turn_id, raw.details as unknown as TranscriptToolStartedEvent['details'], identity)
      case 'tool_completed': return transcriptToolCompletedEvent(expectedSessionId, raw.turn_id, raw.details as unknown as TranscriptToolCompletedEvent['details'], identity)
      case 'turn_completed': return transcriptTurnCompletedEvent(expectedSessionId, raw.turn_id, raw.details as unknown as TranscriptTurnCompletedEvent['details'], identity)
      default: return undefined
    }
  } catch { return undefined }
}

export function readTranscriptEventRecords(bytes: Uint8Array, expectedSessionId: string, baseOffset = 0): TranscriptEventRecordReadResult {
  const records: TranscriptEventRecord[] = []; const events: TranscriptEvent[] = []
  let malformedLines = 0; let start = 0
  for (let cursor = 0; cursor < bytes.length; cursor += 1) {
    if (bytes[cursor] !== 0x0a) continue
    const line = new TextDecoder().decode(bytes.subarray(start, cursor)); const endOffset = baseOffset + cursor + 1
    start = cursor + 1; if (!line.trim()) continue
    let raw: unknown
    try { raw = JSON.parse(line) as unknown } catch { malformedLines += 1; continue }
    const event = parseTranscriptEvent(raw, expectedSessionId)
    if (!event) { malformedLines += 1; continue }
    events.push(event); records.push({ endOffset, event })
  }
  return { events, malformedLines, partialTail: start < bytes.length, records }
}

export function readTranscriptEventLines(text: string, expectedSessionId: string): TranscriptEventReadResult {
  const { events, malformedLines, partialTail } = readTranscriptEventRecords(new TextEncoder().encode(text), expectedSessionId)
  return { events, malformedLines, partialTail }
}

function assertLifecycle(sessionId: string, turnId: string, identity: TranscriptEventIdentity, detailsValid: boolean): void {
  if (!sessionId || !turnId || !validIdentity(identity) || !detailsValid) throw new TypeError('invalid transcript lifecycle event')
}
function validIdentity(value: TranscriptEventIdentity): boolean {
  return !!value.eventId && Number.isSafeInteger(value.sequence) && value.sequence > 0
}
function parseIdentity(raw: Record<string, unknown>): TranscriptEventIdentity | undefined {
  return typeof raw.event_id === 'string' && raw.event_id && Number.isSafeInteger(raw.sequence) && Number(raw.sequence) > 0
    ? { eventId: raw.event_id, sequence: Number(raw.sequence) } : undefined
}
function validMessage(raw: Record<string, unknown>): raw is Record<string, unknown> & { index: number; message: Record<string, unknown> } {
  return Number.isSafeInteger(raw.index) && Number(raw.index) >= 0 && isRecord(raw.message)
}
function stringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value) && value.every(item => typeof item === 'string')
}
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
