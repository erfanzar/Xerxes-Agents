// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The desktop's cookie jar for the Hub, kept in the Electron main process.
 *
 * Only the two Hub session cookies are ever stored; anything else a response
 * sets is ignored. The jar is never handed to the renderer: the renderer asks
 * main to sign in and to open the Hub socket, and main attaches the cookies.
 * The names are the Hub's (`hub/identity/cookies.ts`); they are repeated here
 * so the main bundle does not pull in the Hub server.
 */

export const HUB_REFRESH_COOKIE = '__Host-xerxes_refresh'
export const HUB_SESSION_COOKIE = '__Host-xerxes_session'

const NAMES: ReadonlySet<string> = new Set([HUB_REFRESH_COOKIE, HUB_SESSION_COOKIE])
/** RFC 6265 cookie-octet. */
const VALUE = /^[\x21\x23-\x2b\x2d-\x3a\x3c-\x5b\x5d-\x7e]{1,4096}$/
const DEFAULT_SECONDS = 3600
const MAX_SECONDS = 60 * 60 * 24 * 400

interface Entry {
  readonly value: string
  readonly expiresAt: number
}

export class CookieJar {
  private readonly entries = new Map<string, Entry>()

  /** Apply one `Set-Cookie` header value. A zero Max-Age or an empty value removes the cookie. */
  store(setCookie: string, now: number): void {
    const parts = setCookie.split(';')
    const pair = parts[0] ?? ''
    const eq = pair.indexOf('=')
    if (eq <= 0) return
    const name = pair.slice(0, eq).trim()
    const value = pair.slice(eq + 1).trim()
    if (!NAMES.has(name)) return
    let maxAge: number | undefined
    for (const attribute of parts.slice(1)) {
      const at = attribute.indexOf('=')
      if (at < 0) continue
      if (attribute.slice(0, at).trim().toLowerCase() !== 'max-age') continue
      const seconds = Number(attribute.slice(at + 1).trim())
      if (Number.isInteger(seconds)) maxAge = seconds
    }
    if (value === '' || (maxAge !== undefined && maxAge <= 0)) {
      this.entries.delete(name)
      return
    }
    if (!VALUE.test(value)) return
    this.entries.set(name, { value, expiresAt: now + Math.min(maxAge ?? DEFAULT_SECONDS, MAX_SECONDS) * 1000 })
  }

  /** True when the cookie exists and stays valid for at least `marginMs` more. */
  has(name: string, now: number, marginMs = 0): boolean {
    const entry = this.entries.get(name)
    return entry !== undefined && entry.expiresAt - marginMs > now
  }

  /** A `Cookie` header for the named cookies that are still valid ('' when none). */
  header(names: readonly string[], now: number): string {
    const pairs: string[] = []
    for (const name of names) {
      const entry = this.entries.get(name)
      if (entry !== undefined && entry.expiresAt > now) pairs.push(`${name}=${entry.value}`)
    }
    return pairs.join('; ')
  }

  clear(): void {
    this.entries.clear()
  }

  /** Text for the encrypted store. Only unexpired cookies are written. */
  serialize(now: number): string {
    const cookies = [...this.entries].filter(([, entry]) => entry.expiresAt > now).map(([name, entry]) => ({ name, value: entry.value, expires_at: entry.expiresAt }))
    return JSON.stringify({ version: 1, cookies })
  }

  /** Load what `serialize` wrote. Anything unreadable is dropped, never guessed. */
  restore(text: string, now: number): void {
    let parsed: unknown
    try {
      parsed = JSON.parse(text)
    } catch {
      return
    }
    if (typeof parsed !== 'object' || parsed === null || (parsed as { version?: unknown }).version !== 1) return
    const cookies = (parsed as { cookies?: unknown }).cookies
    if (!Array.isArray(cookies)) return
    for (const raw of cookies.slice(0, 8) as unknown[]) {
      if (typeof raw !== 'object' || raw === null) continue
      const item = raw as Record<string, unknown>
      const { name, value, expires_at: expiresAt } = item
      if (typeof name !== 'string' || !NAMES.has(name) || typeof value !== 'string' || !VALUE.test(value)) continue
      if (typeof expiresAt !== 'number' || !Number.isFinite(expiresAt) || expiresAt <= now) continue
      this.entries.set(name, { value, expiresAt })
    }
  }
}
