// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

/** OSV endpoint used to vet an MCP package before an explicit install. */
export const OSV_API_URL = 'https://api.osv.dev/v1/query'

/** One day, in milliseconds. */
export const DEFAULT_CACHE_TTL_MS = 24 * 60 * 60 * 1_000

const CACHE_FILE = 'osv_cache.json'

export interface Vulnerability {
  readonly aliases: readonly string[]
  readonly id: string
  readonly severity: string
  readonly summary: string
}

export type OSVFetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

export interface CheckPackageOptions {
  /** Optional local cache directory. No cache is created unless this is supplied. */
  readonly cacheDirectory?: string
  /** Freshness period in milliseconds. */
  readonly cacheTtlMs?: number
  /** Injectable fetch boundary for tests and hosts with restricted networking. */
  readonly fetchImplementation?: OSVFetch
  /** Injectable clock for deterministic cache tests. */
  readonly now?: () => number
  readonly signal?: AbortSignal
}

interface CacheEntry {
  readonly fetchedAt: number
  readonly vulnerabilities: readonly Vulnerability[]
}

type Cache = Readonly<Record<string, CacheEntry>>

/**
 * Query OSV for known advisories affecting an MCP package.
 *
 * A failed or malformed OSV response yields an empty result instead of
 * blocking a deliberate install solely because the advisory service is down.
 * Callers still decide whether a successful result is acceptable with
 * {@link isBlocked}.
 */
export async function checkPackage(
  ecosystem: string,
  name: string,
  version: string | undefined = undefined,
  options: CheckPackageOptions = {},
): Promise<readonly Vulnerability[]> {
  const packageName = requiredText(name, 'name')
  const packageEcosystem = requiredText(ecosystem, 'ecosystem')
  const packageVersion = optionalText(version, 'version')
  const now = options.now ?? Date.now
  const cacheTtlMs = nonNegativeFinite(options.cacheTtlMs ?? DEFAULT_CACHE_TTL_MS, 'cacheTtlMs')
  const key = cacheKey(packageEcosystem, packageName, packageVersion)
  const cache = options.cacheDirectory === undefined ? {} : await loadCache(options.cacheDirectory)
  const entry = cache[key]
  if (entry !== undefined && now() - entry.fetchedAt < cacheTtlMs) {
    return cloneVulnerabilities(entry.vulnerabilities)
  }

  const payload: Record<string, unknown> = {
    package: { ecosystem: packageEcosystem, name: packageName },
  }
  if (packageVersion !== undefined) payload.version = packageVersion

  let vulnerabilities: readonly Vulnerability[]
  try {
    const response = await (options.fetchImplementation ?? fetch)(OSV_API_URL, {
      method: 'POST',
      headers: { accept: 'application/json', 'content-type': 'application/json' },
      body: JSON.stringify(payload),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    })
    if (!response.ok) return []
    vulnerabilities = parseResponse(await response.json())
  } catch {
    return []
  }

  if (options.cacheDirectory !== undefined) {
    await saveCache(options.cacheDirectory, { ...cache, [key]: { fetchedAt: now(), vulnerabilities } })
  }
  return cloneVulnerabilities(vulnerabilities)
}

/** True when an advisory represents malware or HIGH/CRITICAL severity. */
export function isBlocked(vulnerabilities: readonly Vulnerability[]): boolean {
  return vulnerabilities.some(vulnerability =>
    vulnerability.id.startsWith('MAL-')
    || /^(?:CRITICAL|HIGH)/i.test(vulnerability.severity),
  )
}

/** Parse an OSV query response into the small safe subset exposed to install policy. */
export function parseResponse(payload: unknown): readonly Vulnerability[] {
  if (!isRecord(payload) || !Array.isArray(payload.vulns)) return []
  return payload.vulns.flatMap(vulnerability => {
    if (!isRecord(vulnerability)) return []
    const databaseSpecific = isRecord(vulnerability.database_specific) ? vulnerability.database_specific : {}
    return [{
      id: stringValue(vulnerability.id),
      summary: stringValue(vulnerability.summary),
      severity: stringValue(databaseSpecific.severity),
      aliases: Object.freeze(stringArray(vulnerability.aliases)),
    }]
  })
}

async function loadCache(directory: string): Promise<Cache> {
  try {
    const payload = JSON.parse(await readFile(join(directory, CACHE_FILE), 'utf8')) as unknown
    return parseCache(payload)
  } catch {
    return {}
  }
}

async function saveCache(directory: string, cache: Cache): Promise<void> {
  try {
    await mkdir(directory, { recursive: true })
    const path = join(directory, CACHE_FILE)
    const temporary = join(directory, `.${CACHE_FILE}.${randomUUID()}.tmp`)
    try {
      await writeFile(temporary, JSON.stringify(cache, null, 2), 'utf8')
      await rename(temporary, path)
    } finally {
      await rm(temporary, { force: true })
    }
  } catch {
    // Advisory caching must never turn a successful OSV check into an install failure.
  }
}

function parseCache(payload: unknown): Cache {
  if (!isRecord(payload)) return {}
  const entries: Record<string, CacheEntry> = {}
  for (const [key, value] of Object.entries(payload)) {
    if (!isRecord(value) || typeof value.fetchedAt !== 'number' || !Number.isFinite(value.fetchedAt)) continue
    entries[key] = { fetchedAt: value.fetchedAt, vulnerabilities: parseResponse({ vulns: value.vulnerabilities }) }
  }
  return entries
}

function cacheKey(ecosystem: string, name: string, version: string | undefined): string {
  return `${ecosystem}::${name}::${version ?? ''}`
}

function cloneVulnerabilities(vulnerabilities: readonly Vulnerability[]): Vulnerability[] {
  return vulnerabilities.map(vulnerability => ({ ...vulnerability, aliases: [...vulnerability.aliases] }))
}

function requiredText(value: string, name: string): string {
  const text = value.trim()
  if (!text) throw new TypeError(`${name} must not be empty`)
  return text
}

function optionalText(value: string | undefined, name: string): string | undefined {
  if (value === undefined) return undefined
  const text = value.trim()
  if (!text) throw new TypeError(`${name} must not be empty when supplied`)
  return text
}

function nonNegativeFinite(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${name} must be a non-negative finite number`)
  return value
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
