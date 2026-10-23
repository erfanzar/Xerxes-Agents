// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Cookie helpers. Both Hub cookies use the `__Host-` prefix, which browsers only
 * accept when the cookie is `Secure`, has `Path=/` and carries no `Domain`.
 * They are `HttpOnly` (script can never read a token) and `SameSite=Strict`.
 */

export const REFRESH_COOKIE = '__Host-xerxes_refresh'
export const SESSION_COOKIE = '__Host-xerxes_session'

export function serializeHostCookie(name: string, value: string, maxAgeSeconds: number): string {
  if (!name.startsWith('__Host-')) throw new Error('Hub cookies must use the __Host- prefix')
  if (!/^[A-Za-z0-9._~-]*$/.test(value)) throw new Error('cookie value contains characters that need encoding')
  const maxAge = Math.max(0, Math.floor(maxAgeSeconds))
  return `${name}=${value}; Max-Age=${maxAge}; Path=/; Secure; HttpOnly; SameSite=Strict`
}

export function clearHostCookie(name: string): string {
  return serializeHostCookie(name, '', 0)
}

/** Parse a `Cookie` request header. The first occurrence of a name wins. */
export function parseCookieHeader(header: string | undefined): Map<string, string> {
  const cookies = new Map<string, string>()
  if (!header) return cookies
  for (const part of header.split(';')) {
    const index = part.indexOf('=')
    if (index <= 0) continue
    const name = part.slice(0, index).trim()
    if (!cookies.has(name)) cookies.set(name, part.slice(index + 1).trim())
  }
  return cookies
}
