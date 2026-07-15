// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHmac, timingSafeEqual } from 'node:crypto'

export const DEFAULT_IDENTITY_SALT = 'xerxes-default-identity-salt'

const DEFAULT_SALT_WARNING = [
  'XERXES_IDENTITY_SALT is not set; using the public default identity salt.',
  'Identity hashes are not irreversible in this configuration.',
].join(' ')

let warnedForDefaultSalt = false

export interface IdentityHashOptions {
  /** Overrides XERXES_IDENTITY_SALT for this individual operation. */
  readonly salt?: string
  /** Receives the one-time insecure-default diagnostic instead of console.warn. */
  readonly onDefaultSalt?: (message: string) => void
}

/** Hash a platform identity into the stable user_<sha16> audit form. */
export function hashUser(
  platform: string,
  rawUserId: string | number,
  options: IdentityHashOptions = {},
): string {
  return 'user_' + identityDigest(platform + '|' + String(rawUserId), options).slice(0, 16)
}

/** Hash a room ID while retaining the platform prefix needed for routing. */
export function hashChat(
  platform: string,
  rawChatId: string | number,
  options: IdentityHashOptions = {},
): string {
  return platform + ':' + identityDigest(platform + '|chat|' + String(rawChatId), options).slice(0, 16)
}

/** Compare an external candidate without exposing timing differences for same-length values. */
export function matchesUser(
  platform: string,
  rawUserId: string | number,
  candidate: string,
  options: IdentityHashOptions = {},
): boolean {
  const expected = hashUser(platform, rawUserId, options)
  const expectedBytes = Buffer.from(expected)
  const candidateBytes = Buffer.from(candidate)
  return expectedBytes.length === candidateBytes.length && timingSafeEqual(expectedBytes, candidateBytes)
}

function identityDigest(value: string, options: IdentityHashOptions): string {
  return createHmac('sha256', resolveSalt(options)).update(value).digest('hex')
}

function resolveSalt(options: IdentityHashOptions): string {
  const configured = options.salt ?? process.env.XERXES_IDENTITY_SALT
  if (configured !== undefined) return configured
  if (!warnedForDefaultSalt) {
    warnedForDefaultSalt = true
    const warn = options.onDefaultSalt ?? console.warn
    warn(DEFAULT_SALT_WARNING)
  }
  return DEFAULT_IDENTITY_SALT
}
