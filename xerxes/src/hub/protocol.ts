// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Wire contract between the Xerxes Hub and its clients (phone PWA, desktop
 * supervisor tab). This module is browser-safe: no Node or Bun imports.
 *
 * The Hub protocol is its own versioned protocol, not a passthrough of daemon
 * protocol v35. Clients never name daemon methods; they send typed, allowlisted
 * Hub methods and `hub.command` actions, and the Hub derives the actor from the
 * authenticated connection, never from request parameters.
 */

export const HUB_PROTOCOL_VERSION = 1 as const

export const HUB_LIMITS = Object.freeze({
  /** Target size for one event batch or page. */
  normalFrameBytes: 256 * 1024,
  /** Hard cap for any single frame in either direction. */
  maxFrameBytes: 1024 * 1024,
  /** Upper bound on events per batch. */
  maxBatchEvents: 500,
  /** Upper bound on ids in one request (workspaces, sessions, recipients). */
  maxIdsPerRequest: 200,
  maxIdLength: 200,
})

export const HUB_METHODS = [
  'hub.hello',
  'hub.workspaces.list',
  'hub.snapshot',
  'hub.events.subscribe',
  'hub.events.read',
  'hub.events.ack',
  'hub.events.unsubscribe',
  'hub.history',
  'hub.command',
  'hub.command.status',
  'hub.presence.update',
  'hub.pairing.create',
  'hub.pairing.consume',
  'hub.devices.list',
  'hub.devices.revoke',
  'hub.invites.create',
  'hub.invites.accept',
  'hub.grants.update',
  'hub.notifications.subscribe',
  'hub.notifications.unsubscribe',
  'hub.usage',
  'supervisor.thread.open',
  'supervisor.message.submit',
  'supervisor.confirmation.respond',
  'supervisor.action.status',
  'hub.machines.list',
  'supervisor.config.get',
  'supervisor.config.options',
  'supervisor.config.set',
] as const
export type HubMethod = (typeof HUB_METHODS)[number]

export const HUB_COMMAND_ACTIONS = [
  'session.open',
  'turn.submit',
  'turn.steer',
  'turn.cancel',
  'permission.respond',
  'question.respond',
  'goal.set',
  'model.set',
  'subagent.message',
] as const
export type HubCommandAction = (typeof HUB_COMMAND_ACTIONS)[number]

export const HUB_EVENT_TYPES = [
  'hub.event_batch',
  'hub.stream_gap',
  'hub.presence',
  'hub.workspace_status',
  'hub.command_result',
  'hub.revoked',
  'supervisor.thread_event',
  'supervisor.confirmation',
  'supervisor.config_changed',
] as const
export type HubEventType = (typeof HUB_EVENT_TYPES)[number]

export type HubErrorCode =
  | 'unauthorized'
  | 'forbidden'
  | 'not_found'
  | 'invalid_request'
  | 'frame_too_large'
  | 'unavailable'
  | 'conflict'
  | 'submission_conflict'
  | 'interaction_already_resolved'
  | 'gap'
  | 'rate_limited'
  | 'unsupported'
  | 'unconfigured'
  | 'internal'

export type PrincipalKind = 'user' | 'agent'

/** Who did something. Always written by the Hub from authenticated context. */
export interface HubActor {
  readonly principal_kind: PrincipalKind
  readonly principal_id: string
  readonly display_name: string
  readonly user_id?: string
  readonly device_id?: string
  /** For agent principals: the human the action was requested by (attribution only). */
  readonly requested_by?: string
  /** For confirmed agent actions: the human who confirmed it. */
  readonly confirmed_by?: string
}

/** One sequenced event as clients see it. `cursor` is opaque to clients. */
export interface HubEvent {
  readonly event_id: string
  readonly sequence: number
  readonly cursor: string
  readonly workspace_id: string
  readonly session_id?: string
  readonly type: string
  readonly payload: Readonly<Record<string, unknown>>
  readonly occurred_at: string
  readonly actor?: HubActor
}

export interface HubEventBatch {
  readonly subscription_id: string
  readonly events: readonly HubEvent[]
  readonly through_cursor: string
}

export type HubGapReason = 'retention' | 'stream_replaced' | 'journal_unavailable' | 'scope_changed'

export interface HubStreamGap {
  readonly subscription_id: string
  readonly reason: HubGapReason
  readonly requested_cursor?: string
  readonly oldest_cursor?: string
  readonly watermark: string
}

export type HubCommandState = 'accepted' | 'rejected' | 'unknown' | 'pending_confirmation'

export interface HubCommandResult {
  readonly ok: boolean
  readonly command_id: string
  readonly state: HubCommandState
  readonly result?: Readonly<Record<string, unknown>>
  readonly error?: { readonly code: HubErrorCode; readonly message: string }
}

export type WorkspaceHealth = 'online' | 'connecting' | 'stale' | 'offline' | 'needs_operator'

export interface HubRequestFrame {
  readonly v: typeof HUB_PROTOCOL_VERSION
  readonly kind: 'request'
  readonly id: string
  readonly method: HubMethod
  readonly params: Readonly<Record<string, unknown>>
}

export type HubResponseFrame =
  | {
      readonly v: typeof HUB_PROTOCOL_VERSION
      readonly kind: 'response'
      readonly id: string
      readonly ok: true
      readonly result: Readonly<Record<string, unknown>>
    }
  | {
      readonly v: typeof HUB_PROTOCOL_VERSION
      readonly kind: 'response'
      readonly id: string | null
      readonly ok: false
      readonly error: { readonly code: HubErrorCode; readonly message: string }
    }

export interface HubEventFrame {
  readonly v: typeof HUB_PROTOCOL_VERSION
  readonly kind: 'event'
  readonly type: HubEventType
  readonly payload: Readonly<Record<string, unknown>>
}

export type HubServerFrame = HubResponseFrame | HubEventFrame

export class HubProtocolError extends Error {
  readonly code: HubErrorCode
  readonly requestId: string | null
  constructor(code: HubErrorCode, message: string, requestId: string | null = null) {
    super(message)
    this.name = 'HubProtocolError'
    this.code = code
    this.requestId = requestId
  }
}

/** UTF-8 byte length without allocating a buffer (browser-safe). */
export function utf8ByteLength(text: string): number {
  let bytes = 0
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index)
    if (code < 0x80) bytes += 1
    else if (code < 0x800) bytes += 2
    else if (code >= 0xd800 && code <= 0xdbff && index + 1 < text.length) {
      const next = text.charCodeAt(index + 1)
      if (next >= 0xdc00 && next <= 0xdfff) {
        bytes += 4
        index += 1
      } else bytes += 3
    } else bytes += 3
  }
  return bytes
}

const METHOD_SET: ReadonlySet<string> = new Set(HUB_METHODS)
const ACTION_SET: ReadonlySet<string> = new Set(HUB_COMMAND_ACTIONS)
const EVENT_SET: ReadonlySet<string> = new Set(HUB_EVENT_TYPES)

export function isHubMethod(value: unknown): value is HubMethod {
  return typeof value === 'string' && METHOD_SET.has(value)
}

export function isHubCommandAction(value: unknown): value is HubCommandAction {
  return typeof value === 'string' && ACTION_SET.has(value)
}

/** Parse one client frame. Throws HubProtocolError with the request id when known. */
export function parseHubRequestFrame(text: string, maxBytes: number = HUB_LIMITS.maxFrameBytes): HubRequestFrame {
  if (utf8ByteLength(text) > maxBytes) {
    throw new HubProtocolError('frame_too_large', `frame exceeds ${maxBytes} bytes`)
  }
  let value: unknown
  try {
    value = JSON.parse(text) as unknown
  } catch {
    throw new HubProtocolError('invalid_request', 'frame is not valid JSON')
  }
  if (!isRecord(value)) throw new HubProtocolError('invalid_request', 'frame must be a JSON object')
  const id = typeof value.id === 'string' && value.id.length > 0 && value.id.length <= HUB_LIMITS.maxIdLength ? value.id : null
  if (value.v !== HUB_PROTOCOL_VERSION) {
    throw new HubProtocolError('unsupported', `unsupported hub protocol version ${String(value.v)}`, id)
  }
  if (value.kind !== 'request') throw new HubProtocolError('invalid_request', 'client frames must be requests', id)
  if (id === null) throw new HubProtocolError('invalid_request', 'request id must be a non-empty string of at most 200 characters')
  if (!isHubMethod(value.method)) {
    throw new HubProtocolError('invalid_request', `unknown hub method ${JSON.stringify(value.method)}`, id)
  }
  if (value.params !== undefined && !isRecord(value.params)) {
    throw new HubProtocolError('invalid_request', 'params must be an object', id)
  }
  return { v: HUB_PROTOCOL_VERSION, kind: 'request', id, method: value.method, params: value.params ?? {} }
}

/** Parse one server frame on the client side. */
export function parseHubServerFrame(text: string, maxBytes: number = HUB_LIMITS.maxFrameBytes): HubServerFrame {
  if (utf8ByteLength(text) > maxBytes) throw new HubProtocolError('frame_too_large', `frame exceeds ${maxBytes} bytes`)
  let value: unknown
  try {
    value = JSON.parse(text) as unknown
  } catch {
    throw new HubProtocolError('invalid_request', 'frame is not valid JSON')
  }
  if (!isRecord(value) || value.v !== HUB_PROTOCOL_VERSION) {
    throw new HubProtocolError('unsupported', 'not a hub protocol v1 frame')
  }
  if (value.kind === 'event') {
    if (typeof value.type !== 'string' || !EVENT_SET.has(value.type) || !isRecord(value.payload)) {
      throw new HubProtocolError('invalid_request', 'malformed event frame')
    }
    return { v: HUB_PROTOCOL_VERSION, kind: 'event', type: value.type as HubEventType, payload: value.payload }
  }
  if (value.kind === 'response') {
    const id = typeof value.id === 'string' ? value.id : null
    if (value.ok === true && isRecord(value.result) && id !== null) {
      return { v: HUB_PROTOCOL_VERSION, kind: 'response', id, ok: true, result: value.result }
    }
    if (value.ok === false && isRecord(value.error) && typeof value.error.message === 'string') {
      return {
        v: HUB_PROTOCOL_VERSION,
        kind: 'response',
        id,
        ok: false,
        error: { code: String(value.error.code ?? 'internal') as HubErrorCode, message: value.error.message },
      }
    }
  }
  throw new HubProtocolError('invalid_request', 'malformed server frame')
}

/** Serialize a frame, refusing to emit anything over the hard cap. */
export function encodeHubFrame(frame: HubServerFrame | HubRequestFrame, maxBytes: number = HUB_LIMITS.maxFrameBytes): string {
  const text = JSON.stringify(frame)
  if (utf8ByteLength(text) > maxBytes) {
    throw new HubProtocolError('frame_too_large', `frame of ${utf8ByteLength(text)} bytes exceeds ${maxBytes}`)
  }
  return text
}

export function hubResponse(id: string, result: Readonly<Record<string, unknown>>): HubResponseFrame {
  return { v: HUB_PROTOCOL_VERSION, kind: 'response', id, ok: true, result }
}

export function hubError(id: string | null, code: HubErrorCode, message: string): HubResponseFrame {
  return { v: HUB_PROTOCOL_VERSION, kind: 'response', id, ok: false, error: { code, message } }
}

export function hubEvent(type: HubEventType, payload: Readonly<Record<string, unknown>>): HubEventFrame {
  return { v: HUB_PROTOCOL_VERSION, kind: 'event', type, payload }
}

/** Validate a bounded list of ids from untrusted params. */
export function readIdList(value: unknown, field: string, options: { required?: boolean } = {}): string[] {
  if (value === undefined) {
    if (options.required) throw new HubProtocolError('invalid_request', `${field} is required`)
    return []
  }
  if (!Array.isArray(value)) throw new HubProtocolError('invalid_request', `${field} must be an array of ids`)
  if (value.length > HUB_LIMITS.maxIdsPerRequest) {
    throw new HubProtocolError('invalid_request', `${field} accepts at most ${HUB_LIMITS.maxIdsPerRequest} ids`)
  }
  const ids: string[] = []
  for (const item of value) ids.push(readId(item, field))
  return [...new Set(ids)]
}

export function readId(value: unknown, field: string): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > HUB_LIMITS.maxIdLength) {
    throw new HubProtocolError('invalid_request', `${field} must be a non-empty string of at most ${HUB_LIMITS.maxIdLength} characters`)
  }
  return value
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
