// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'

/** JSON-compatible payload values forwarded across the daemon/TUI boundary. */
export type WireRecord = Readonly<Record<string, unknown>>

/**
 * The native `AgentTurnRunner` stream adapter maps portable loop events into
 * these seven external event families.
 *
 * Lifecycle, slash-command, and historical bridge-only events stay owned by
 * their respective modules until they have a native streaming producer.
 */
export type WireEventName =
  | 'approval_request'
  | 'notification'
  | 'status_update'
  | 'text_part'
  | 'think_part'
  | 'tool_call'
  | 'tool_result'

export const WIRE_EVENT_NAME_BY_INTERNAL: Readonly<Record<WireEventName, string>> = Object.freeze({
  text_part: 'TextPart',
  think_part: 'ThinkPart',
  tool_call: 'ToolCall',
  approval_request: 'ApprovalRequest',
  tool_result: 'ToolResult',
  status_update: 'StatusUpdate',
  notification: 'Notification',
})

const INTERNAL_WIRE_EVENT_NAME_BY_KIMI: Readonly<Record<string, WireEventName>> = Object.freeze(
  Object.fromEntries(
    Object.entries(WIRE_EVENT_NAME_BY_INTERNAL).map(([internal, kimi]) => [kimi, internal as WireEventName]),
  ),
)

/** A normalized internal event keeps its protocol discriminator separate from its exact payload. */
export interface WireEvent<Name extends WireEventName, Payload extends WireRecord> {
  readonly event_type: Name
  readonly payload: Payload
}

export interface TextPartPayload extends WireRecord {
  readonly text: string
}

export interface ThinkPartPayload extends WireRecord {
  readonly think: string
}

export interface ToolCallPayload extends WireRecord {
  readonly arguments: string | null
  readonly id: string
  readonly name: string
  /** Current daemon producers send this redundant correlation alias. */
  readonly tool_call_id?: string
}

export interface ApprovalRequestPayload extends WireRecord {
  readonly action: string
  readonly description: string
  readonly id?: string
  readonly inputs?: WireRecord
  readonly name?: string
  readonly request_id?: string
  readonly tool_call_id?: string
  readonly tool_name?: string
}

export interface ToolResultPayload extends WireRecord {
  readonly display_blocks: readonly WireRecord[]
  readonly duration_ms: number
  readonly permitted?: boolean
  readonly result?: string
  readonly return_value?: string
  readonly tool_call_id: string
}

/** Status payloads vary between streamed turns and daemon startup; known fields are validated when present. */
export interface StatusUpdatePayload extends WireRecord {
  readonly context_tokens?: number
  readonly max_context?: number
  readonly mcp_status?: WireRecord
  readonly mode?: string
  readonly model?: string
  readonly plan_mode?: boolean
  readonly reasoning_effort?: string
  readonly usage?: WireRecord
}

/** Notification payloads intentionally carry arbitrary source-specific context. */
export interface NotificationPayload extends WireRecord {
  readonly body?: string
  readonly level?: string
  readonly message?: string
  readonly payload?: WireRecord
  readonly severity?: string
  readonly title?: string
}

export type TextPart = WireEvent<'text_part', TextPartPayload>
export type ThinkPart = WireEvent<'think_part', ThinkPartPayload>
export type ToolCall = WireEvent<'tool_call', ToolCallPayload>
export type ApprovalRequest = WireEvent<'approval_request', ApprovalRequestPayload>
export type ToolResult = WireEvent<'tool_result', ToolResultPayload>
export type StatusUpdate = WireEvent<'status_update', StatusUpdatePayload>
export type Notification = WireEvent<'notification', NotificationPayload>

export type KnownWireEvent =
  | ApprovalRequest
  | Notification
  | StatusUpdate
  | TextPart
  | ThinkPart
  | ToolCall
  | ToolResult

/** Unknown event names remain inspectable for forward compatibility instead of being silently discarded. */
export interface GenericWireEvent {
  readonly event_type: 'generic'
  readonly raw: WireRecord
}

export type WireEventType = GenericWireEvent | KnownWireEvent

/** The event body nested inside the daemon's JSON-RPC `event` notification. */
export interface WireEventEnvelope {
  readonly payload: WireRecord
  readonly type: string
}

export interface JSONRPCEventMessage {
  readonly jsonrpc: '2.0'
  readonly method: 'event'
  readonly params: WireEventEnvelope
}

/** Explicit malformed-wire failure instead of a permissive object cast at the protocol boundary. */
export class WireEventValidationError extends ValidationError {}

/** Translate a native snake_case event name to the Kimi/PascalCase spelling used by the TUI protocol. */
export function toKimiEventName(eventType: string): string {
  return WIRE_EVENT_NAME_BY_INTERNAL[eventType as WireEventName] ?? eventType
}

/** Translate a Kimi/PascalCase spelling back to the native daemon spelling. */
export function toInternalEventName(eventType: string): string {
  return INTERNAL_WIRE_EVENT_NAME_BY_KIMI[eventType] ?? eventType
}

/**
 * Decode either a direct `{ type, ...payload }` value or the canonical
 * `{ type, payload }` envelope. Both frozen event-name spellings are accepted.
 */
export function eventFromDict(value: unknown): WireEventType {
  const record = readRecord(value, 'event')
  const wireType = record.type
  if (typeof wireType !== 'string') return { event_type: 'generic', raw: record }

  const eventType = toInternalEventName(wireType)
  if (!isWireEventName(eventType)) return { event_type: 'generic', raw: record }

  const payload = isEnvelope(record) ? readRecord(record.payload, 'event.payload') : withoutType(record)
  return createWireEvent(eventType, payload)
}

/** Validate and normalize one native event without changing daemon-provided extension fields. */
export function createWireEvent<Name extends WireEventName>(
  eventType: Name,
  value: unknown,
): Extract<KnownWireEvent, { event_type: Name }> {
  const payload = readRecord(value, `${eventType}.payload`)
  return validatePayload(eventType, payload) as Extract<KnownWireEvent, { event_type: Name }>
}

/** Serialize a normalized event as the Kimi-compatible `{ type, payload }` frame body. */
export function eventToDict(event: WireEventType): WireEventEnvelope {
  if (event.event_type === 'generic') {
    return { type: 'generic', payload: { raw: event.raw } }
  }
  return { type: toKimiEventName(event.event_type), payload: copyRecord(event.payload) }
}

/** Build the exact JSON-RPC notification envelope expected by daemon transports and UI clients. */
export function jsonRpcEventMessage(event: WireEventType): JSONRPCEventMessage {
  return { jsonrpc: '2.0', method: 'event', params: eventToDict(event) }
}

/** Validate a JSON-RPC event notification before handing it to an event reducer. */
export function eventFromJsonRpcMessage(value: unknown): WireEventType {
  const message = readRecord(value, 'jsonRpcEvent')
  if (message.jsonrpc !== '2.0') {
    throw new WireEventValidationError('jsonrpc', 'must equal "2.0"', message.jsonrpc)
  }
  if (message.method !== 'event') {
    throw new WireEventValidationError('method', 'must equal "event"', message.method)
  }
  return eventFromDict(readRecord(message.params, 'params'))
}

function validatePayload(eventType: WireEventName, payload: WireRecord): KnownWireEvent {
  switch (eventType) {
    case 'text_part':
      return { event_type: eventType, payload: { ...payload, text: requiredText(payload, 'text') } }
    case 'think_part':
      return { event_type: eventType, payload: { ...payload, think: requiredText(payload, 'think') } }
    case 'tool_call':
      return {
        event_type: eventType,
        payload: {
          ...payload,
          id: requiredText(payload, 'id'),
          name: requiredText(payload, 'name'),
          arguments: nullableText(payload, 'arguments'),
          ...optionalTextField(payload, 'tool_call_id'),
        },
      }
    case 'approval_request':
      return {
        event_type: eventType,
        payload: {
          ...payload,
          action: requiredText(payload, 'action'),
          description: requiredText(payload, 'description'),
          ...requestIdFields(payload),
          ...optionalTextField(payload, 'tool_call_id'),
          ...optionalTextField(payload, 'name'),
          ...optionalTextField(payload, 'tool_name'),
          ...optionalRecordField(payload, 'inputs'),
        },
      }
    case 'tool_result':
      return {
        event_type: eventType,
        payload: {
          ...payload,
          tool_call_id: requiredText(payload, 'tool_call_id'),
          duration_ms: finiteNumber(payload, 'duration_ms'),
          display_blocks: displayBlocks(payload),
          ...optionalTextField(payload, 'return_value'),
          ...optionalTextField(payload, 'result'),
          ...optionalBooleanField(payload, 'permitted'),
        },
      }
    case 'status_update':
      validateOptionalText(payload, ['model', 'mode', 'reasoning_effort'])
      validateOptionalFiniteNumbers(payload, ['context_tokens', 'max_context'])
      validateOptionalBoolean(payload, 'plan_mode')
      validateOptionalRecord(payload, ['mcp_status', 'usage'])
      return { event_type: eventType, payload: copyRecord(payload) }
    case 'notification':
      validateOptionalText(payload, ['level', 'message', 'severity', 'title', 'body'])
      validateOptionalRecord(payload, ['payload'])
      return { event_type: eventType, payload: copyRecord(payload) }
  }
}

function requestIdFields(payload: WireRecord): Pick<ApprovalRequestPayload, 'id' | 'request_id'> {
  const id = optionalText(payload, 'id')
  const requestId = optionalText(payload, 'request_id')
  if (!id && !requestId) {
    throw new WireEventValidationError('approval_request.id', 'must provide id or request_id', payload)
  }
  return {
    ...(id === undefined ? {} : { id }),
    ...(requestId === undefined ? {} : { request_id: requestId }),
  }
}

function displayBlocks(payload: WireRecord): readonly WireRecord[] {
  const value = payload.display_blocks
  if (!Array.isArray(value)) {
    throw new WireEventValidationError('display_blocks', 'must be an array of objects', value)
  }
  return value.map((block, index) => readRecord(block, `display_blocks[${index}]`))
}

function requiredText(payload: WireRecord, field: string): string {
  const value = payload[field]
  if (typeof value !== 'string') {
    throw new WireEventValidationError(field, 'must be a string', value)
  }
  return value
}

function nullableText(payload: WireRecord, field: string): string | null {
  const value = payload[field]
  if (value === null || typeof value === 'string') return value
  throw new WireEventValidationError(field, 'must be a string or null', value)
}

function finiteNumber(payload: WireRecord, field: string): number {
  const value = payload[field]
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new WireEventValidationError(field, 'must be a finite number', value)
  }
  return value
}

function optionalText(payload: WireRecord, field: string): string | undefined {
  const value = payload[field]
  if (value === undefined) return undefined
  if (typeof value !== 'string') throw new WireEventValidationError(field, 'must be a string', value)
  return value
}

function optionalTextField(payload: WireRecord, field: string): Partial<WireRecord> {
  const value = optionalText(payload, field)
  return value === undefined ? {} : { [field]: value }
}

function optionalRecordField(payload: WireRecord, field: string): Partial<WireRecord> {
  const value = payload[field]
  if (value === undefined) return {}
  return { [field]: readRecord(value, field) }
}

function optionalBooleanField(payload: WireRecord, field: string): Partial<WireRecord> {
  const value = payload[field]
  if (value === undefined) return {}
  if (typeof value !== 'boolean') throw new WireEventValidationError(field, 'must be a boolean', value)
  return { [field]: value }
}

function validateOptionalText(payload: WireRecord, fields: readonly string[]): void {
  for (const field of fields) optionalText(payload, field)
}

function validateOptionalFiniteNumbers(payload: WireRecord, fields: readonly string[]): void {
  for (const field of fields) {
    if (payload[field] !== undefined) finiteNumber(payload, field)
  }
}

function validateOptionalBoolean(payload: WireRecord, field: string): void {
  if (payload[field] !== undefined) optionalBooleanField(payload, field)
}

function validateOptionalRecord(payload: WireRecord, fields: readonly string[]): void {
  for (const field of fields) {
    if (payload[field] !== undefined) readRecord(payload[field], field)
  }
}

function isEnvelope(record: WireRecord): boolean {
  return Object.hasOwn(record, 'payload')
    && isRecord(record.payload)
    && Object.keys(record).every(key => key === 'type' || key === 'payload')
}

function isWireEventName(value: string): value is WireEventName {
  return Object.hasOwn(WIRE_EVENT_NAME_BY_INTERNAL, value)
}

function withoutType(record: WireRecord): WireRecord {
  const result: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(record)) {
    if (key !== 'type') result[key] = value
  }
  return result
}

function readRecord(value: unknown, field: string): WireRecord {
  if (!isRecord(value)) throw new WireEventValidationError(field, 'must be an object', value)
  return copyRecord(value)
}

function copyRecord(record: WireRecord): WireRecord {
  return { ...record }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
