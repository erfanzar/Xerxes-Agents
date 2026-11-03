// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Hub health, shared by the main process (which reads the saved Hub address
 * and asks the Hub's `/healthz`) and the renderer (which shows a notice when
 * no Hub is set up or the saved one does not answer). Only this validated
 * state crosses the process boundary; no credential or cookie is in it.
 */

export const HUB_HEALTH_ACTIONS = ['status', 'recheck', 'dismiss', 'remind'] as const
export type HubHealthAction = typeof HUB_HEALTH_ACTIONS[number]

/** One IPC channel per action: hubhealth:status, hubhealth:recheck, ... */
export const HUB_HEALTH_CHANNEL_PREFIX = 'hubhealth:'
/** Main pushes the state here after every check and reminder change. */
export const HUB_HEALTH_STATE_EVENT = 'hubhealth:state'

export const HUB_HEALTH_KINDS = ['unknown', 'not_configured', 'online', 'degraded', 'unreachable'] as const
export type HubHealthKind = typeof HUB_HEALTH_KINDS[number]

/** The states that show a notice; each has its own "Don't remind me". */
export const HUB_NOTICE_KINDS = ['not_configured', 'unreachable'] as const
export type HubNoticeKind = typeof HUB_NOTICE_KINDS[number]

export interface HubWorkspaceCounts {
  readonly total: number
  readonly online: number
  readonly degraded: number
}

export interface HubHealthDismissed {
  readonly not_configured: boolean
  readonly unreachable: boolean
}

export interface HubHealthState {
  readonly kind: HubHealthKind
  /** The saved SSH target of the Hub host, or null when none is saved. */
  readonly target: string | null
  /** The saved Hub web origin that was probed. */
  readonly origin: string | null
  /** Why the Hub is unreachable or degraded, in a few words. */
  readonly detail: string | null
  readonly workspaces: HubWorkspaceCounts | null
  readonly checked_at: string | null
  readonly dismissed: HubHealthDismissed
}

export type HubHealthFailureCode = 'unknown_action' | 'invalid_request' | 'failed'

export type HubHealthResult =
  | { readonly ok: true; readonly state: HubHealthState }
  | { readonly ok: false; readonly code: HubHealthFailureCode; readonly message: string }

export function isHubHealthAction(value: unknown): value is HubHealthAction {
  return typeof value === 'string' && (HUB_HEALTH_ACTIONS as readonly string[]).includes(value)
}

export function isHubNoticeKind(value: unknown): value is HubNoticeKind {
  return typeof value === 'string' && (HUB_NOTICE_KINDS as readonly string[]).includes(value)
}

/** The notice this state calls for, or null when it is fine or the person opted out of that reminder. */
export function hubNoticeKind(state: HubHealthState): HubNoticeKind | null {
  if (state.kind === 'not_configured' && !state.dismissed.not_configured) return 'not_configured'
  if (state.kind === 'unreachable' && !state.dismissed.unreachable) return 'unreachable'
  return null
}

function nullableText(value: unknown, max: number): string | null | undefined {
  if (value === null) return null
  return typeof value === 'string' && value.length <= max ? value : undefined
}

function count(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 && value <= 1_000_000 ? value : undefined
}

export function asHubWorkspaceCounts(value: unknown): HubWorkspaceCounts | null | undefined {
  if (value === null) return null
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  const raw = value as Record<string, unknown>
  const total = count(raw.total), online = count(raw.online), degraded = count(raw.degraded)
  if (total === undefined || online === undefined || degraded === undefined) return undefined
  return { total, online, degraded }
}

/** Validate a state that crossed a process boundary; anything off-contract is dropped. */
export function asHubHealthState(value: unknown): HubHealthState | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  const raw = value as Record<string, unknown>
  if (typeof raw.kind !== 'string' || !(HUB_HEALTH_KINDS as readonly string[]).includes(raw.kind)) return undefined
  const target = nullableText(raw.target, 300)
  const origin = nullableText(raw.origin, 300)
  const detail = nullableText(raw.detail, 400)
  const checkedAt = nullableText(raw.checked_at, 64)
  const workspaces = asHubWorkspaceCounts(raw.workspaces)
  if (target === undefined || origin === undefined || detail === undefined || checkedAt === undefined || workspaces === undefined) return undefined
  if (origin !== null && !/^https?:\/\//.test(origin)) return undefined
  const dismissed = raw.dismissed
  if (!dismissed || typeof dismissed !== 'object') return undefined
  const flags = dismissed as Record<string, unknown>
  if (typeof flags.not_configured !== 'boolean' || typeof flags.unreachable !== 'boolean') return undefined
  return {
    kind: raw.kind as HubHealthKind, target, origin, detail, workspaces, checked_at: checkedAt,
    dismissed: { not_configured: flags.not_configured, unreachable: flags.unreachable },
  }
}
