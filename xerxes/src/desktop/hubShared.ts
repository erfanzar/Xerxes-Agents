// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Pure helpers and payload types shared by the Electron main process (which
 * runs the Hub CLI and owns the Hub session) and the renderer (which shows the
 * "Hub & phone" panel and the XERXES tab). No Electron, DOM or Node imports:
 * this file must load in both worlds.
 *
 * Nothing in these payloads is a credential. The desktop's own pairing secret,
 * its cookies and the SSH identity stay in the main process.
 */

export const HUB_WS_SUFFIX = '/api/hub/v1/ws'

// ── Hub SSH address and other inputs ────────────────────────────────────

const USER = /^[A-Za-z0-9_][A-Za-z0-9._-]{0,63}$/
const HOST = /^[A-Za-z0-9][A-Za-z0-9._-]{0,253}$/

export type InputCheck<T> = { readonly ok: true; readonly value: T } | { readonly ok: false; readonly message: string }

/**
 * `user@host` or an SSH alias. A leading dash would be read by ssh as an
 * option, and spaces or shell syntax have no place in an address, so both are
 * refused here, before anything is spawned.
 */
export function checkHubTarget(value: unknown): InputCheck<string> {
  if (typeof value !== 'string') return { ok: false, message: 'The Hub SSH address must be text.' }
  const target = value.trim()
  if (target === '') return { ok: false, message: 'Enter the Hub SSH address, for example xerxes@209.38.102.184.' }
  if (target.length > 320) return { ok: false, message: 'That address is too long.' }
  if (target.startsWith('-')) return { ok: false, message: 'An address cannot start with a dash.' }
  const at = target.indexOf('@')
  const user = at < 0 ? undefined : target.slice(0, at)
  const host = at < 0 ? target : target.slice(at + 1)
  if (user !== undefined && !USER.test(user)) return { ok: false, message: 'The user name in that address is not valid.' }
  if (!HOST.test(host) || host.includes('..')) return { ok: false, message: 'Use a host name, an IP address or an SSH alias, like xerxes@209.38.102.184. No spaces or special characters.' }
  return { ok: true, value: target }
}

const LABEL = /^[\p{L}\p{N}][\p{L}\p{N} ._()-]{0,39}$/u

export function checkDeviceLabel(value: unknown): InputCheck<string> {
  if (typeof value !== 'string') return { ok: false, message: 'The device label must be text.' }
  const label = value.trim()
  if (!LABEL.test(label)) return { ok: false, message: 'Use 1 to 40 letters, numbers or spaces for the device label.' }
  return { ok: true, value: label }
}

export function checkPairingId(value: unknown): InputCheck<string> {
  return typeof value === 'string' && /^[A-Za-z0-9_][A-Za-z0-9_-]{0,99}$/.test(value)
    ? { ok: true, value }
    : { ok: false, message: 'That pairing id is not valid.' }
}

export function checkComparisonCode(value: unknown): InputCheck<string> {
  return typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9-]{0,19}$/.test(value.trim())
    ? { ok: true, value: value.trim() }
    : { ok: false, message: 'That code is not valid.' }
}

// ── Hub address ─────────────────────────────────────────────────────────

function loopback(hostname: string): boolean {
  return hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]' || hostname === '::1'
}

/**
 * The Hub's web origin (`https://hub.example.ts.net`) from what the Hub
 * reports. https only; plain http for loopback. Anything with a path, query,
 * fragment or credentials is refused. Throws a message fit for the panel.
 */
export function parseHubOrigin(value: string): string {
  const trimmed = value.trim()
  if (trimmed === '') throw new Error('The Hub reported no web address.')
  let url: URL
  try {
    url = new URL(trimmed.includes('://') ? trimmed : `https://${trimmed}`)
  } catch {
    throw new Error('The Hub reported a web address this app cannot read.')
  }
  if (url.username !== '' || url.password !== '') throw new Error('The Hub web address must not contain a password or token.')
  if (url.search !== '' || url.hash !== '') throw new Error('The Hub web address must not have a query or fragment.')
  if (url.pathname !== '' && url.pathname !== '/') throw new Error('The Hub web address must not have a path.')
  const secure = url.protocol === 'https:'
  const plain = url.protocol === 'http:'
  if (!secure && !plain) throw new Error('The Hub web address must start with https://.')
  if (plain && !loopback(url.hostname)) throw new Error('A Hub away from this machine must use https://. Plain http is only allowed for localhost.')
  return `${url.protocol}//${url.host}`
}

/** `wss://host/api/hub/v1/ws` for an origin accepted by `parseHubOrigin`. */
export function hubWebSocketUrlFor(origin: string): string {
  return `${origin.startsWith('https:') ? 'wss' : 'ws'}://${origin.replace(/^https?:\/\//, '')}${HUB_WS_SUFFIX}`
}

/** What the panel shows for a Hub origin. */
export function displayOrigin(origin: string): string {
  return origin.replace(/^https?:\/\//, '')
}

/**
 * The secret half of a pairing link. The Hub's link is
 * `<origin>/pair#<base64url JSON {hub_id, pairing_id, code}>`; the query-style
 * `#pairing_id=…&code=…` fragment is accepted as well. Returns `undefined` for
 * anything else. The code is a one-time secret: it must only ever travel to the
 * Hub's own consume endpoint.
 */
export function parsePairingTicket(url: string): { readonly pairingId: string; readonly code: string } | undefined {
  const hash = url.indexOf('#')
  if (hash < 0) return undefined
  const body = url.slice(hash + 1)
  if (body === '') return undefined
  if (/^[A-Za-z0-9_-]+$/.test(body) && body.length <= 2048) {
    try {
      const padded = body.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(body.length / 4) * 4, '=')
      const bytes = Uint8Array.from(atob(padded), char => char.charCodeAt(0))
      const value: unknown = JSON.parse(new TextDecoder().decode(bytes))
      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        const ticket = value as Record<string, unknown>
        const id = ticket.pairing_id
        const code = ticket.code
        if (typeof id === 'string' && typeof code === 'string' && id !== '' && code !== '' && id.length <= 200 && code.length <= 512) return { pairingId: id, code }
      }
    } catch {
      // Not a base64url ticket; try the query form below.
    }
  }
  const params = new URLSearchParams(body)
  const id = params.get('pairing_id') ?? params.get('p')
  const code = params.get('code') ?? params.get('c')
  return id && code && id.length <= 200 && code.length <= 512 ? { pairingId: id, code } : undefined
}

// ── QR code ─────────────────────────────────────────────────────────────

const QR_MAX_CHARS = 262_144

function base64Utf8(text: string): string {
  const bytes = new TextEncoder().encode(text)
  let binary = ''
  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000))
  }
  return btoa(binary)
}

/**
 * The QR as an `<img src>` data URL. It is only ever shown through `<img>`, where
 * a browser runs no script, and it is screened anyway: a document that is not a
 * plain `<svg>`, or that carries script, event handlers, foreign content or
 * entities, yields `undefined` and the panel falls back to the link and code.
 */
export function qrDataUrl(svg: unknown): string | undefined {
  if (typeof svg !== 'string') return undefined
  const text = svg.trim()
  if (text === '' || text.length > QR_MAX_CHARS) return undefined
  if (!/^(?:<\?xml[^>]*\?>\s*)?(?:<!--[\s\S]*?-->\s*)?(?:<!DOCTYPE\s+svg[^>[]*>\s*)?<svg[\s>]/i.test(text)) return undefined
  if (/<script|<foreignObject|<iframe|<object|<embed|<!ENTITY|javascript:|\son[a-z]+\s*=/i.test(text)) return undefined
  return `data:image/svg+xml;base64,${base64Utf8(text)}`
}

// ── IPC payloads (main -> renderer) ─────────────────────────────────────

export type CockpitFailureCode =
  | 'invalid_input'
  | 'no_target'
  | 'no_origin'
  | 'cli_unavailable'
  | 'cli_failed'
  | 'bad_output'
  | 'timeout'
  | 'busy'
  | 'not_signed_in'
  | 'hub_unreachable'
  | 'socket_closed'
  | 'storage_failed'

export interface CockpitFailure {
  readonly ok: false
  readonly code: CockpitFailureCode
  readonly message: string
  /** Sign-in only: the Hub refused this desktop, so signing in again will not help until it is paired anew. */
  readonly denied?: true
}

export type CockpitResult<T extends object = Record<string, unknown>> = ({ readonly ok: true } & T) | CockpitFailure

export interface CockpitStatus {
  /** `user@host` or alias the Hub is reached through; null until one is saved. */
  readonly target: string | null
  /** The Hub's web origin, learned from `hub info`. */
  readonly origin: string | null
  readonly wsUrl: string | null
  readonly fingerprint: string | null
  readonly hubVersion: string | null
  /** The Hub machine id this Mac was linked as. */
  readonly linkedMachineId: string | null
  /** The desktop holds a Hub session (a refresh cookie), not that it is still valid. */
  readonly hasSession: boolean
}

export interface CockpitProject {
  readonly workspaceId: string
  readonly projectDir: string
  readonly label: string
  readonly health: string
}

export interface CockpitMachine {
  readonly id: string
  readonly label: string
  readonly endpointKind: string
  readonly discover: boolean
  readonly projects: readonly CockpitProject[]
}

export interface CockpitLink {
  readonly machineId: string
  readonly remoteSocket: string
  readonly servicePath: string | null
}

export interface CockpitPairing {
  readonly pairingId: string
  /** The one-time link the phone opens. It is meant for the phone and expires. */
  readonly url: string
  readonly comparisonCode: string
  readonly expiresAt: string
  readonly confirmRequired: boolean
  readonly qrDataUrl: string | null
}

/** The reply body of each action (besides `ok`). */
export interface CockpitReplyMap {
  readonly status: { readonly status: CockpitStatus }
  readonly save: { readonly status: CockpitStatus }
  readonly forget: { readonly status: CockpitStatus }
  readonly link: { readonly link: CockpitLink; readonly machines: readonly CockpitMachine[]; readonly machinesError: string | null; readonly status: CockpitStatus }
  readonly pair: { readonly pairing: CockpitPairing }
  readonly confirm: { readonly deviceId: string }
  readonly machines: { readonly machines: readonly CockpitMachine[] }
  readonly signin: { readonly origin: string; readonly wsUrl: string }
  readonly 'socket.open': object
  readonly 'socket.send': object
  readonly 'socket.close': object
}

/** Events pushed from main to the renderer that owns a bridged Hub socket. */
export type HubSocketEvent =
  | { readonly id: string; readonly type: 'open' }
  | { readonly id: string; readonly type: 'message'; readonly data: string }
  | { readonly id: string; readonly type: 'binary' }
  | { readonly id: string; readonly type: 'error'; readonly message: string }
  | { readonly id: string; readonly type: 'close'; readonly code: number; readonly reason: string }

export interface CockpitProgress {
  readonly op: 'link'
  readonly line: string
}

export const COCKPIT_CHANNEL_PREFIX = 'hub.cockpit.'
export const COCKPIT_SOCKET_EVENT = 'hub.cockpit.socket'
export const COCKPIT_PROGRESS_EVENT = 'hub.cockpit.progress'

export const COCKPIT_ACTIONS = [
  'status', 'save', 'forget', 'link', 'pair', 'confirm', 'machines', 'signin',
  'socket.open', 'socket.send', 'socket.close',
] as const
export type CockpitAction = (typeof COCKPIT_ACTIONS)[number]

export function isCockpitAction(value: unknown): value is CockpitAction {
  return typeof value === 'string' && (COCKPIT_ACTIONS as readonly string[]).includes(value)
}

/** Largest Hub frame the bridge carries, in either direction (the Hub's own limit). */
export const COCKPIT_MAX_FRAME_BYTES = 1024 * 1024

export function socketIdOk(value: unknown): value is string {
  return typeof value === 'string' && /^[A-Za-z0-9_-]{1,64}$/.test(value)
}

/** A socket event as it arrives in the renderer. Anything malformed is dropped, never guessed. */
export function asHubSocketEvent(value: unknown): HubSocketEvent | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined
  const item = value as Record<string, unknown>
  const id = item.id
  if (!socketIdOk(id)) return undefined
  switch (item.type) {
    case 'open': return { id, type: 'open' }
    case 'binary': return { id, type: 'binary' }
    case 'message': return typeof item.data === 'string' && item.data.length <= COCKPIT_MAX_FRAME_BYTES ? { id, type: 'message', data: item.data } : undefined
    case 'error': return { id, type: 'error', message: typeof item.message === 'string' ? item.message.slice(0, 300) : 'The Hub link failed.' }
    case 'close': return {
      id, type: 'close',
      code: typeof item.code === 'number' && Number.isInteger(item.code) ? item.code : 1006,
      reason: typeof item.reason === 'string' ? item.reason.slice(0, 300) : '',
    }
    default: return undefined
  }
}

export function asCockpitProgress(value: unknown): CockpitProgress | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined
  const item = value as Record<string, unknown>
  return item.op === 'link' && typeof item.line === 'string' ? { op: 'link', line: item.line.slice(0, 300) } : undefined
}
