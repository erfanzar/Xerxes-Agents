// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { parsePairFragment } from './pairing.js'

/** Longer than any ticket the Hub issues (its QR encoder stops at 213 bytes). */
export const PAIR_LINK_MAX_LENGTH = 2048

export type PairLinkCheck =
  | { readonly ok: true; readonly href: string }
  | { readonly ok: false; readonly message: string }

const NOT_A_LINK: PairLinkCheck = { ok: false, message: 'That is not a Xerxes pairing link. Scan the QR code from Pair a phone on the desktop.' }

/**
 * Check a scanned or pasted pairing link before the phone opens it. Only this Hub's own
 * `/pair#<ticket>` links pass, so a QR code can never send the phone, or its pairing
 * secret, to another site. The returned href is rebuilt from `origin`, never taken
 * verbatim, and no message repeats the secret.
 */
export function checkPairLink(raw: string, origin: string): PairLinkCheck {
  const text = raw.trim()
  if (!text) return { ok: false, message: 'Scan or paste the pairing link from the desktop.' }
  if (text.length > PAIR_LINK_MAX_LENGTH) return NOT_A_LINK
  let url: URL
  try { url = new URL(text) } catch { return NOT_A_LINK }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') return NOT_A_LINK
  if (url.username || url.password) return NOT_A_LINK
  if (url.origin !== origin) {
    return { ok: false, message: `That code is for a different Hub (${url.host}). Scan the code shown by this Hub’s desktop.` }
  }
  if (url.pathname !== '/pair') return NOT_A_LINK
  if (parsePairFragment(url.hash) === undefined) {
    return { ok: false, message: 'That pairing link is incomplete. Create a new pairing code on the desktop and scan it again.' }
  }
  return { ok: true, href: `${origin}/pair${url.hash}` }
}
