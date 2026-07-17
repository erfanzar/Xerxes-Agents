// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lookup } from 'node:dns/promises'
import { isIP } from 'node:net'

export interface UrlSafetyDecision {
  readonly allowed: boolean
  readonly reason: string
  readonly url: string
}

/** Resolve every A/AAAA address for a hostname. Injectable for deterministic tests. */
export type DnsLookup = (hostname: string) => Promise<readonly string[]>

export interface UrlSafetyOptions {
  /** Hosts explicitly allowed despite an otherwise internal address classification. */
  readonly allowlist?: ReadonlySet<string>
  /** Resolver used by {@link checkUrlDns}; defaults to the system DNS via node:dns. */
  readonly dnsLookup?: DnsLookup
}

export const DENIED_URL_SCHEMES: ReadonlySet<string> = new Set(['file', 'ftp', 'gopher', 'data'])

/** Evaluate a URL against Xerxes' no-private-network fetch policy. */
export function checkUrl(url: string, options: UrlSafetyOptions = {}): UrlSafetyDecision {
  if (!url) {
    return decision(url, false, 'empty URL')
  }

  const scheme = schemeOf(url)
  if (scheme !== undefined && DENIED_URL_SCHEMES.has(scheme)) {
    return decision(url, false, `denied scheme: ${scheme}`)
  }

  const parsed = parseNetworkUrl(url)
  if (parsed === undefined) {
    return decision(url, false, 'missing scheme or host')
  }

  const host = normalizedHostname(parsed.hostname)
  if (isAllowlisted(host, options.allowlist)) {
    return decision(url, true, 'allowlisted host')
  }
  if (isInternalHost(host)) {
    return decision(url, false, `private/internal host: ${host}`)
  }
  return decision(url, true, 'public host')
}

/** Return only the allow/deny result of {@link checkUrl}. */
export function isUrlSafe(url: string, options: UrlSafetyOptions = {}): boolean {
  return checkUrl(url, options).allowed
}

/**
 * Resolve the URL's host and require every A/AAAA answer to be a public address.
 *
 * Literal IPs and allowlisted hosts never touch the resolver, and resolution
 * failures fail closed. Residual TOCTOU limitation: answers are validated here,
 * but the HTTP client resolves the host again when dialing, so a DNS-rebinding
 * attacker could return different answers on the second lookup. Pinning the
 * dialed connection to a validated address would close that gap; until then this
 * blocks hosts with static private answers, not a determined rebinding adversary.
 */
export async function checkUrlDns(url: string, options: UrlSafetyOptions = {}): Promise<UrlSafetyDecision> {
  const parsed = parseNetworkUrl(url)
  if (parsed === undefined) {
    return decision(url, false, 'missing scheme or host')
  }

  const host = normalizedHostname(parsed.hostname)
  if (isAllowlisted(host, options.allowlist)) {
    return decision(url, true, 'allowlisted host')
  }
  if (isIP(host) || host === 'localhost' || host === 'localhost.localdomain') {
    return isInternalHost(host)
      ? decision(url, false, `private/internal host: ${host}`)
      : decision(url, true, 'public host')
  }

  const resolve = options.dnsLookup ?? systemDnsLookup
  let addresses: readonly string[]
  try {
    addresses = await resolve(host)
  } catch {
    return decision(url, false, `DNS resolution failed for host: ${host}`)
  }
  if (addresses.length === 0) {
    return decision(url, false, `DNS returned no addresses for host: ${host}`)
  }
  for (const address of addresses) {
    if (!isIP(address) || isInternalHost(address)) {
      return decision(url, false, `host ${host} resolves to private/internal address: ${address}`)
    }
  }
  return decision(url, true, 'public DNS answers')
}

function decision(url: string, allowed: boolean, reason: string): UrlSafetyDecision {
  return { url, allowed, reason }
}

function schemeOf(url: string): string | undefined {
  const match = /^([a-z][a-z\d+.-]*):/i.exec(url)
  return match?.[1]?.toLowerCase()
}

function parseNetworkUrl(url: string): URL | undefined {
  const scheme = schemeOf(url)
  if (scheme === undefined || !/^[a-z][a-z\d+.-]*:\/\//i.test(url)) {
    return undefined
  }
  try {
    const parsed = new URL(url)
    return parsed.hostname ? parsed : undefined
  } catch {
    return undefined
  }
}

function normalizedHostname(host: string): string {
  return host.replace(/^\[|\]$/g, '').replace(/\.$/, '').toLowerCase()
}

function isAllowlisted(host: string, allowlist: ReadonlySet<string> | undefined): boolean {
  if (allowlist === undefined) {
    return false
  }
  for (const allowedHost of allowlist) {
    if (normalizedHostname(allowedHost) === host) {
      return true
    }
  }
  return false
}

function isInternalHost(host: string): boolean {
  if (!host || host === 'localhost' || host === 'localhost.localdomain') {
    return true
  }
  const family = isIP(host)
  if (family === 4) {
    return isInternalIpv4(host)
  }
  if (family === 6) {
    return isInternalIpv6(host)
  }
  // Literal-address checks only; hostnames are never resolved on this synchronous
  // path. Network callers must additionally run checkUrlDns so a public name
  // cannot hide private DNS answers.
  return false
}

async function systemDnsLookup(hostname: string): Promise<readonly string[]> {
  const results = await lookup(hostname, { all: true, verbatim: true })
  return results.map(result => result.address)
}

function isInternalIpv4(host: string): boolean {
  const octets = host.split('.').map(Number)
  const first = octets[0]
  const second = octets[1]
  if (first === undefined || second === undefined) {
    return true
  }

  return first === 0
    || first === 10
    || first === 127
    || first >= 224
    || (first === 100 && second >= 64 && second <= 127)
    || (first === 169 && second === 254)
    || (first === 172 && second >= 16 && second <= 31)
    || (first === 192 && second === 168)
    || (first === 192 && second === 0)
    || (first === 192 && second === 88)
    || (first === 198 && (second === 18 || second === 19))
    || (first === 198 && second === 51)
    || (first === 203 && second === 0)
  }

function isInternalIpv6(host: string): boolean {
  const groups = ipv6Groups(host)
  if (groups === undefined) {
    return true
  }
  const first = groups[0]
  const second = groups[1]
  const third = groups[2]
  const fourth = groups[3]
  const fifth = groups[4]
  const sixth = groups[5]
  if (first === undefined || second === undefined || third === undefined || fourth === undefined
    || fifth === undefined || sixth === undefined) {
    return true
  }

  const allZero = groups.every(group => group === 0)
  const loopback = groups.slice(0, 7).every(group => group === 0) && groups[7] === 1
  const ipv4Mapped = first === 0 && second === 0 && third === 0 && fourth === 0 && fifth === 0 && sixth === 0xffff
  if (ipv4Mapped) {
    const seventh = groups[6]
    const eighth = groups[7]
    if (seventh === undefined || eighth === undefined) {
      return true
    }
    return isInternalIpv4([seventh >> 8, seventh & 0xff, eighth >> 8, eighth & 0xff].join('.'))
  }

  return allZero
    || loopback
    || (first & 0xff00) === 0xff00
    || (first & 0xffc0) === 0xfe80
    || (first & 0xfe00) === 0xfc00
    || (first === 0x2001 && (second & 0xff00) === 0x0d00)
}

function ipv6Groups(host: string): number[] | undefined {
  const normalized = host.toLowerCase()
  const split = normalized.split('::')
  if (split.length > 2) {
    return undefined
  }
  const left = split[0] ? split[0].split(':') : []
  const right = split[1] ? split[1].split(':') : []
  const leftGroups = parseIpv6Side(left)
  const rightGroups = parseIpv6Side(right)
  if (leftGroups === undefined || rightGroups === undefined) {
    return undefined
  }

  const specified = leftGroups.length + rightGroups.length
  if (split.length === 1) {
    return specified === 8 ? leftGroups : undefined
  }
  if (specified >= 8) {
    return undefined
  }
  return [...leftGroups, ...Array<number>(8 - specified).fill(0), ...rightGroups]
}

function parseIpv6Side(parts: string[]): number[] | undefined {
  const groups: number[] = []
  for (const part of parts) {
    if (!/^[\da-f]{1,4}$/i.test(part)) {
      return undefined
    }
    groups.push(Number.parseInt(part, 16))
  }
  return groups
}
