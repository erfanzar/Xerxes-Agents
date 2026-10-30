// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Pairing flow. The QR code carries the one-time secret in the URL fragment,
 * which browsers never send to a server. `takePairingFragment` reads it and
 * removes it from the address bar and history at once, before anything renders
 * or any request is made. The secret then lives only in memory and in one
 * redacted POST body.
 */

export interface PairFragment {
  readonly pairingId: string
  readonly code: string
  /** Short code the owner compares with the desktop, when the QR carries one. */
  readonly comparison?: string
}

export type PairingRead =
  | { readonly kind: 'found'; readonly fragment: PairFragment }
  | { readonly kind: 'invalid' }
  | { readonly kind: 'missing' }
  /** The page was reloaded (or discarded by iOS) after the Hub accepted the code, while waiting for the desktop. */
  | { readonly kind: 'resume'; readonly comparison?: string }

/** sessionStorage subset; access can throw (Safari private mode, storage disabled). */
export interface PairStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

const AWAIT_KEY = 'xerxes.pair.awaiting'
/** How long a reload may resume waiting for the desktop; matches the pair page's own wait plus margin. */
export const PAIR_AWAIT_TTL_MS = 10 * 60_000

/**
 * Remember that this tab is waiting for desktop confirmation. Holds no secret: the
 * pairing code was already spent and the session lives in httpOnly cookies. Only the
 * comparison code (shown on screen anyway) and a deadline are stored.
 */
export function rememberPairingAwait(storage: PairStorage | undefined, now: number, comparison?: string): void {
  try {
    storage?.setItem(AWAIT_KEY, JSON.stringify({ until: now + PAIR_AWAIT_TTL_MS, ...(comparison === undefined ? {} : { comparison: comparison.slice(0, 16) }) }))
  } catch { /* Without storage a reload shows the plain pair page, as before. */ }
}

/** This tab's sessionStorage, when the browser offers one (undefined in tests and when disabled). */
export function browserPairStorage(): PairStorage | undefined {
  try {
    return typeof sessionStorage === 'undefined' ? undefined : sessionStorage
  } catch {
    return undefined
  }
}

export function forgetPairingAwait(storage: PairStorage | undefined): void {
  try { storage?.removeItem(AWAIT_KEY) } catch { /* nothing stored */ }
}

/** The pending wait, if one was remembered and has not expired. */
export function pairingAwait(storage: PairStorage | undefined, now: number): { readonly comparison?: string } | undefined {
  let text: string | null = null
  try { text = storage?.getItem(AWAIT_KEY) ?? null } catch { return undefined }
  if (text === null) return undefined
  try {
    const value = JSON.parse(text) as unknown
    if (typeof value !== 'object' || value === null) return undefined
    const record = value as Record<string, unknown>
    if (typeof record.until !== 'number' || record.until <= now) {
      forgetPairingAwait(storage)
      return undefined
    }
    return typeof record.comparison === 'string' && record.comparison.length > 0 ? { comparison: record.comparison.slice(0, 16) } : {}
  } catch {
    forgetPairingAwait(storage)
    return undefined
  }
}

/** Read the pair route: a fresh fragment wins; otherwise resume a remembered wait. */
export function readPairRoute(win: PairWindow, storage: PairStorage | undefined, now: number): PairingRead {
  const read = takePairingFragment(win)
  if (read.kind !== 'missing') {
    // A new code supersedes any earlier wait.
    forgetPairingAwait(storage)
    return read
  }
  const waiting = pairingAwait(storage, now)
  if (waiting === undefined) return read
  return waiting.comparison === undefined ? { kind: 'resume' } : { kind: 'resume', comparison: waiting.comparison }
}

export interface PairWindow {
  readonly location: { readonly hash: string; readonly pathname: string; readonly search: string }
  readonly history: { replaceState(data: unknown, unused: string, url?: string | null): void }
}

function decodeBase64UrlJson(text: string): Readonly<Record<string, unknown>> | undefined {
  if (!/^[A-Za-z0-9_-]+$/.test(text) || text.length > 2048) return undefined
  try {
    const padded = text.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(text.length / 4) * 4, '=')
    const bytes = Uint8Array.from(atob(padded), char => char.charCodeAt(0))
    const value = JSON.parse(new TextDecoder().decode(bytes)) as unknown
    return typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined
  } catch {
    return undefined
  }
}

/**
 * The Hub's ticket is `#<base64url JSON {hub_id, pairing_id, code}>`. The
 * query-style `#pairing_id=…&code=…` form is accepted as well. The comparison
 * code is not in the fragment: the consume response returns it.
 */
export function parsePairFragment(hash: string): PairFragment | undefined {
  const body = hash.startsWith('#') ? hash.slice(1) : hash
  const ticket = decodeBase64UrlJson(body)
  if (ticket !== undefined) {
    const id = ticket.pairing_id
    const secret = ticket.code
    if (typeof id !== 'string' || typeof secret !== 'string' || !id || !secret || id.length > 200 || secret.length > 512) return undefined
    return { pairingId: id, code: secret }
  }
  const params = new URLSearchParams(body)
  const pairingId = params.get('pairing_id') ?? params.get('p')
  const code = params.get('code') ?? params.get('c')
  if (!pairingId || !code || pairingId.length > 200 || code.length > 512) return undefined
  const comparison = params.get('cc') ?? params.get('comparison')
  return { pairingId, code, ...(comparison ? { comparison: comparison.slice(0, 16) } : {}) }
}

export function takePairingFragment(win: PairWindow): PairingRead {
  const hash = win.location.hash
  if (hash.length <= 1) return { kind: 'missing' }
  const fragment = parsePairFragment(hash)
  // Cleared whether or not it parsed: a malformed secret must not linger either.
  win.history.replaceState(null, '', `${win.location.pathname}${win.location.search}`)
  return fragment ? { kind: 'found', fragment } : { kind: 'invalid' }
}

export function deviceLabelFromUserAgent(userAgent: string): string {
  if (/iPhone/i.test(userAgent)) return 'iPhone'
  if (/iPad/i.test(userAgent)) return 'iPad'
  if (/Android/i.test(userAgent)) return /Mobile/i.test(userAgent) ? 'Android phone' : 'Android tablet'
  return 'Browser'
}

export type PairState =
  | { readonly phase: 'missing' }
  | { readonly phase: 'invalid' }
  | { readonly phase: 'submitting'; readonly comparison?: string }
  | { readonly phase: 'waiting'; readonly comparison?: string }
  | { readonly phase: 'rejected'; readonly message: string }
  | { readonly phase: 'unavailable'; readonly message: string; readonly comparison?: string }
  | { readonly phase: 'timeout'; readonly comparison?: string }
  | { readonly phase: 'paired' }

export type PairAction =
  | { readonly type: 'accepted'; readonly comparison?: string }
  | { readonly type: 'rejected'; readonly message: string }
  | { readonly type: 'unavailable'; readonly message: string }
  | { readonly type: 'confirmed' }
  | { readonly type: 'expired' }
  | { readonly type: 'retry' }

function comparisonOf(state: PairState): { readonly comparison?: string } {
  return 'comparison' in state && state.comparison !== undefined ? { comparison: state.comparison } : {}
}

export function initialPairState(read: PairingRead): PairState {
  if (read.kind === 'missing') return { phase: 'missing' }
  if (read.kind === 'invalid') return { phase: 'invalid' }
  if (read.kind === 'resume') return read.comparison === undefined ? { phase: 'waiting' } : { phase: 'waiting', comparison: read.comparison }
  return read.fragment.comparison === undefined ? { phase: 'submitting' } : { phase: 'submitting', comparison: read.fragment.comparison }
}

export function reducePair(state: PairState, action: PairAction): PairState {
  switch (action.type) {
    case 'accepted': {
      const comparison = action.comparison ?? ('comparison' in state ? state.comparison : undefined)
      return comparison === undefined ? { phase: 'waiting' } : { phase: 'waiting', comparison }
    }
    case 'rejected': return { phase: 'rejected', message: action.message }
    case 'unavailable': return { phase: 'unavailable', message: action.message, ...comparisonOf(state) }
    case 'confirmed': return state.phase === 'waiting' ? { phase: 'paired' } : state
    case 'expired': return state.phase === 'waiting' ? { phase: 'timeout', ...comparisonOf(state) } : state
    case 'retry': return state.phase === 'unavailable' ? { phase: 'submitting', ...comparisonOf(state) } : state
  }
}
