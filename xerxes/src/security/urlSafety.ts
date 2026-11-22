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
  if (isIP(host) || isLocalhostName(host)) {
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

function isLocalhostName(host: string): boolean {
  return host === 'localhost'
    || host.endsWith('.localhost')
    || host === 'localhost.localdomain'
    || host.endsWith('.localhost.localdomain')
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
  if (!host || isLocalhostName(host)) {
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
  if (octets.length !== 4 || octets.some(octet => !Number.isInteger(octet) || octet < 0 || octet > 255)) {
    return true
  }
  const first = octets[0]
  const second = octets[1]
  const third = octets[2]
  if (first === undefined || second === undefined || third === undefined) {
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
    // RFC 6890 IETF protocol assignments (192.0.0.0/24) and RFC 5737 TEST-NET-1 (192.0.2.0/24).
    || (first === 192 && second === 0 && (third === 0 || third === 2))
    // RFC 3068 6to4 relay anycast (192.88.99.0/24).
    || (first === 192 && second === 88 && third === 99)
    || (first === 198 && (second === 18 || second === 19))
    // RFC 5737 TEST-NET-2 (198.51.100.0/24) and TEST-NET-3 (203.0.113.0/24).
    || (first === 198 && second === 51 && third === 100)
    || (first === 203 && second === 0 && third === 113)
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
  const seventh = groups[6]
  const eighth = groups[7]
  if (first === undefined || second === undefined || third === undefined || fourth === undefined
    || fifth === undefined || sixth === undefined || seventh === undefined || eighth === undefined) {
    return true
  }

  const allZero = groups.every(group => group === 0)
  const loopback = groups.slice(0, 7).every(group => group === 0) && eighth === 1
  if (allZero || loopback) {
    return true
  }

  // IPv4-mapped (::ffff:0:0/96), SIIT IPv4-translated (::ffff:0:0:0/96), and
  // deprecated IPv4-compatible (::/96) all embed an IPv4 in the final 32 bits.
  const zeroPrefix64 = first === 0 && second === 0 && third === 0 && fourth === 0
  if (zeroPrefix64 && (
    (fifth === 0 && sixth === 0xffff)
    || (fifth === 0xffff && sixth === 0)
    || (fifth === 0 && sixth === 0)
  )) {
    return isInternalIpv4(formatIpv4Words(seventh, eighth))
  }

  // NAT64 local-use prefix (64:ff9b:1::/48, RFC 8215) is private translation space;
  // well-known NAT64 (64:ff9b::/96, RFC 6052) embeds an IPv4 in the final 32 bits.
  if (first === 0x0064 && second === 0xff9b) {
    if (third === 1) return true
    if (third === 0 && fourth === 0 && fifth === 0 && sixth === 0) {
      return isInternalIpv4(formatIpv4Words(seventh, eighth))
    }
  }

  // 6to4 (2002::/16) embeds a plain IPv4 in the next 32 bits; Teredo
  // (2001:0000::/32) embeds the client IPv4 XOR-obfuscated in the final 32
  // bits. Either can hide a private or loopback address, so classify the
  // embedded IPv4 rather than trusting the public-looking prefix.
  if (first === 0x2002) {
    return isInternalIpv4(formatIpv4Words(second, third))
  }
  if (first === 0x2001 && second === 0) {
    return isInternalIpv4(formatIpv4Words(seventh ^ 0xffff, eighth ^ 0xffff))
  }

  return (first & 0xff00) === 0xff00
    || (first & 0xffc0) === 0xfe80
    || (first & 0xfe00) === 0xfc00
    // RFC 3849 IPv6 documentation prefix (2001:0db8::/32).
    || (first === 0x2001 && second === 0x0db8)
}

function formatIpv4Words(high: number, low: number): string {
  return [(high >> 8) & 0xff, high & 0xff, (low >> 8) & 0xff, low & 0xff].join('.')
}

function ipv6Groups(host: string): number[] | undefined {
  let normalized = host.toLowerCase()
  const lastColon = normalized.lastIndexOf(':')
  if (lastColon >= 0 && normalized.slice(lastColon + 1).includes('.')) {
    const octets = normalized.slice(lastColon + 1).split('.')
    if (octets.length !== 4 || octets.some(part => !/^\d{1,3}$/.test(part))) {
      return undefined
    }
    const nums = octets.map(Number)
    if (nums.some(octet => octet < 0 || octet > 255)) {
      return undefined
    }
    const high = (((nums[0] ?? 0) << 8) | (nums[1] ?? 0)).toString(16)
    const low = (((nums[2] ?? 0) << 8) | (nums[3] ?? 0)).toString(16)
    normalized = `${normalized.slice(0, lastColon + 1)}${high}:${low}`
  }

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
