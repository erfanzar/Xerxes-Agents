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
