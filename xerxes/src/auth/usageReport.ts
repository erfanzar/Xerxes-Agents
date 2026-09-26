// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * One usage report for every provider profile the user has imported — the
 * shared source for the desktop Usage tab and the TUI's /usage view.
 *
 * Each profile gets exactly one of three answers, never a silent gap:
 * `ok` (its plan windows: 5-hour, weekly, per-model, with used % and reset;
 * or, for pay-as-you-go API keys, the balance or key limit the provider reports),
 * `unsupported` (the provider publishes no usage limits Xerxes can read), or
 * `error` (the reason in plain words, e.g. not signed in). Quota endpoints
 * are fetched once per credential and cached briefly, so opening the view
 * repeatedly does not hammer the providers.
 */

import { createHash } from 'node:crypto'

import { profileLabel } from '../bridge/profiles.js'
import { ConfigurationError } from '../core/errors.js'

import {
  fetchSubscriptionUsage,
  type UsageFetch,
  PROVIDER_ALIASES,
  type ProviderUsageReport,
  type SubscriptionUsageProvider,
  type UsageProfile,
  type UsageRequestOptions,
  type UsageWindow,
} from './usage.js'

export interface UsageProfileEntry {
  readonly profile: string
  /** Display name ("Claude Code" for `cc`). */
  readonly label: string
  readonly provider: string
  readonly model: string
  readonly active: boolean
  /** Which quota endpoint answers for this profile, when one exists. */
  readonly source?: UsageSource
  readonly status: 'ok' | 'unsupported' | 'error'
  readonly plan?: string
  readonly windows: readonly UsageWindow[]
  /** Pay-as-you-go keys: money left (or spent), e.g. "$12.40 left". */
  readonly balance?: string
  readonly message?: string
  readonly fetchedAt?: number
}

/** Pay-as-you-go APIs that publish a key balance or spend limit. */
export type ApiBalanceSource = 'openrouter' | 'deepseek' | 'moonshot'
export type UsageSource = SubscriptionUsageProvider | ApiBalanceSource

const SOURCE_NAMES: Readonly<Record<UsageSource, string>> = {
  claude: 'Claude',
  codex: 'ChatGPT',
  kimi: 'Kimi Code',
  zai: 'Z.ai',
  openrouter: 'OpenRouter',
  deepseek: 'DeepSeek',
  moonshot: 'Moonshot',
}

export function usageSourceName(source: UsageSource): string {
  return SOURCE_NAMES[source]
}

/**
 * The endpoint that answers for a profile. The host decides before the
 * provider name does: a "kimi" profile pointed at api.moonshot.* is a
 * Moonshot API key (balance), not a Kimi Code plan (windows).
 */
export function usageSourceFor(provider: string, baseUrl = ''): UsageSource | undefined {
  const host = hostOf(baseUrl)
  if (host.endsWith('openrouter.ai')) return 'openrouter'
  if (host.endsWith('deepseek.com')) return 'deepseek'
  if (/(^|\.)moonshot\.(ai|cn)$/.test(host)) return 'moonshot'
  const name = provider.trim().toLowerCase()
  if (name === 'openrouter') return 'openrouter'
  if (name === 'deepseek') return 'deepseek'
  if (name.startsWith('moonshot')) return 'moonshot'
  return PROVIDER_ALIASES[name]
}

function hostOf(url: string): string {
  try { return new URL(url).hostname.toLowerCase() } catch { return '' }
}

const API_BALANCE_SOURCES = new Set<UsageSource>(['openrouter', 'deepseek', 'moonshot'])

const money = (amount: number, currency: string) => {
  const symbol = currency === 'USD' ? '$' : currency === 'CNY' ? '¥' : ''
  return symbol ? `${symbol}${amount.toFixed(2)}` : `${amount.toFixed(2)} ${currency}`
}

const numberOf = (value: unknown): number | undefined => {
  const number = typeof value === 'string' ? Number(value) : value
  return typeof number === 'number' && Number.isFinite(number) ? number : undefined
}

async function getJson(url: string, apiKey: string, options: UsageRequestOptions): Promise<Record<string, unknown>> {
  const deadline = AbortSignal.timeout(10_000)
  const signal = options.signal ? AbortSignal.any([options.signal, deadline]) : deadline
  const fetcher = options.fetchImplementation ?? (globalThis.fetch as UsageFetch)
  const response = await fetcher(url, { headers: { Authorization: `Bearer ${apiKey}`, Accept: 'application/json' }, method: 'GET', redirect: 'error', signal })
  if (!response.ok) {
    await response.body?.cancel()
    throw new Error(response.status === 401 || response.status === 403 ? `The provider rejected this key (${response.status}).` : `Usage request failed (${response.status}).`)
  }
  const body: unknown = await response.json()
  return body !== null && typeof body === 'object' && !Array.isArray(body) ? body as Record<string, unknown> : {}
}

/** Balance or key limit for a pay-as-you-go API key; windows only when the key has a spend cap. */
export async function fetchApiBalance(source: ApiBalanceSource, profile: UsageProfile, options: UsageRequestOptions = {}): Promise<ProviderUsageReport & { balance?: string }> {
  const apiKey = profile.api_key?.trim()
  if (!apiKey) throw new Error('This profile has no API key saved.')
  const fetchedAt = Date.now()
  if (source === 'openrouter') {
    const data = (await getJson('https://openrouter.ai/api/v1/key', apiKey, options)).data as Record<string, unknown> | undefined
    const used = numberOf(data?.usage) ?? 0
    const limit = numberOf(data?.limit)
    const free = data?.is_free_tier === true
    return {
      provider: source, fetchedAt, ...(free ? { planType: 'free tier' } : {}),
      windows: limit && limit > 0 ? [{ label: 'Key limit', usedPercent: Math.min(100, (used / limit) * 100), detail: `${money(used, 'USD')} of ${money(limit, 'USD')}` }] : [],
      balance: limit && limit > 0 ? `${money(Math.max(0, limit - used), 'USD')} left` : `${money(used, 'USD')} spent`,
    }
  }
  if (source === 'deepseek') {
    const body = await getJson('https://api.deepseek.com/user/balance', apiKey, options)
    const infos = Array.isArray(body.balance_infos) ? body.balance_infos as Record<string, unknown>[] : []
    const parts = infos.map(info => { const amount = numberOf(info.total_balance); return amount === undefined ? '' : money(amount, String(info.currency ?? 'USD')) }).filter(Boolean)
    if (!parts.length) throw new Error('DeepSeek returned no balance for this key.')
    return { provider: source, fetchedAt, windows: [], balance: `${parts.join(' + ')} left` }
  }
  const cn = hostOf(profile.base_url).endsWith('moonshot.cn')
  const body = await getJson(`https://api.moonshot.${cn ? 'cn' : 'ai'}/v1/users/me/balance`, apiKey, options)
  const available = numberOf((body.data as Record<string, unknown> | undefined)?.available_balance)
  if (available === undefined) throw new Error('Moonshot returned no balance for this key.')
  return { provider: source, fetchedAt, windows: [], balance: `${money(available, cn ? 'CNY' : 'USD')} left` }
}

type SourceReport = ProviderUsageReport & { readonly balance?: string }

interface CacheEntry {
  readonly at: number
  readonly result: Promise<SourceReport>
}

/** Remembers quota answers for `ttlMs` per (source, credential); failures are not cached. */
export class UsageReportCache {
  private readonly entries = new Map<string, CacheEntry>()

  constructor(private readonly ttlMs = 60_000, private readonly now: () => number = Date.now) {}

  get(key: string, load: () => Promise<SourceReport>, refresh = false): Promise<SourceReport> {
    const hit = this.entries.get(key)
    if (hit && !refresh && this.now() - hit.at < this.ttlMs) return hit.result
    const result = load()
    this.entries.set(key, { at: this.now(), result })
    void result.catch(() => { if (this.entries.get(key)?.result === result) this.entries.delete(key) })
    return result
  }
}

export interface BuildUsageReportOptions extends UsageRequestOptions {
  readonly cache?: UsageReportCache
  readonly refresh?: boolean
  /** Swap the quota fetchers (tests). */
  readonly fetchUsage?: typeof fetchSubscriptionUsage
  readonly fetchBalance?: typeof fetchApiBalance
  /**
   * Profiles listed whether or not the user ever signed in (the built-in
   * `cc` and `codex`). When they are not active and have no session, they
   * are left out instead of reporting "not signed in" for an account the
   * user never set up.
   */
  readonly hideWhenSignedOut?: ReadonlySet<string>
}

/** Build the per-profile report; never throws for one provider's failure. */
export async function buildUsageReport(
  profiles: ReadonlyArray<UsageProfile & { readonly active?: boolean }>,
  options: BuildUsageReportOptions = {},
): Promise<UsageProfileEntry[]> {
  const fetchUsage = options.fetchUsage ?? fetchSubscriptionUsage
  // One fetch per credential per report, even when `refresh` bypasses the cache.
  const fetchBalance = options.fetchBalance ?? fetchApiBalance
  const pending = new Map<string, Promise<SourceReport>>()
  const entries = await Promise.all(profiles.map(async (profile): Promise<UsageProfileEntry | null> => {
    const base = { profile: profile.name, label: profileLabel(profile.name), provider: profile.provider, model: profile.model, active: profile.active === true }
    const source = usageSourceFor(profile.provider, profile.base_url)
    if (!source) {
      return { ...base, status: 'unsupported', windows: [], message: 'This provider does not publish usage limits or a balance Xerxes can read.' }
    }
    // Profiles that share a login (e.g. two Codex profiles) share one fetch;
    // profiles with their own keys (two Z.ai keys) are fetched separately.
    const credential = profile.api_key?.trim() ? createHash('sha256').update(profile.api_key.trim()).digest('hex').slice(0, 16) : 'session'
    const key = `${source}:${credential}:${profile.base_url ?? ''}`
    const load = (): Promise<SourceReport> => API_BALANCE_SOURCES.has(source)
      ? fetchBalance(source as ApiBalanceSource, profile, options)
      : fetchUsage(source as SubscriptionUsageProvider, { ...options, profiles: [profile] })
    try {
      let request = pending.get(key)
      if (!request) {
        request = options.cache ? options.cache.get(key, load, options.refresh) : load()
        pending.set(key, request)
      }
      const report = await request
      return {
        ...base,
        source,
        status: 'ok',
        ...(report.planType ? { plan: report.planType } : {}),
        windows: report.windows,
        ...(report.balance ? { balance: report.balance } : {}),
        fetchedAt: report.fetchedAt,
      }
    } catch (error) {
      if (error instanceof ConfigurationError && !base.active && options.hideWhenSignedOut?.has(profile.name)) return null
      return { ...base, source, status: 'error', windows: [], message: readableFailure(error) }
    }
  }))
  // Active first, then the ones with numbers, then the rest — alphabetical within.
  const rank = (entry: UsageProfileEntry) => (entry.active ? 0 : 2) + (entry.status === 'ok' ? 0 : entry.status === 'error' ? 1 : 2) * 0.1
  return entries.filter((entry): entry is UsageProfileEntry => entry !== null).sort((a, b) => rank(a) - rank(b) || a.profile.localeCompare(b.profile))
}

/** The reason in the user's words: drop the internal "Configuration codex_auth:" / "Provider x:" prefixes. */
function readableFailure(error: unknown): string {
  const message = (error instanceof Error ? error.message : String(error)).replace(/^(?:Configuration|Provider|Client) [\w.-]+: /, '').trim()
  const rejected = /usage request failed \((401|403)\)/i.exec(message)
  if (rejected) return `The provider rejected the saved key or sign-in (${rejected[1]}).`
  return message || 'Usage lookup failed.'
}

export { resetIn } from './usageView.js'
