// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Subscription usage windows for quota-based providers (Claude Pro/Max,
 * ChatGPT Codex plans, Kimi Code, Z.ai coding plan). Each fetcher maps the
 * provider's quota endpoint onto one normalized report: a plan label plus
 * rolling windows (5-hour, weekly) with used percent and reset times.
 *
 * Endpoints are constants with environment overrides; when a provider
 * changes shape or a subscription does not expose a window, the fetcher
 * returns an actionable error rather than fabricating numbers.
 */

import { ConfigurationError } from '../core/errors.js'
import { AnthropicOAuthSession } from './anthropicOAuth.js'
import { claudeCodeLogin } from './claudeCodeLogin.js'
import { codexAuthHeaders, CodexSession } from './codexAuth.js'
import { KimiCodingOAuthSession } from './kimiCodingOAuth.js'

export const CLAUDE_USAGE_URL = 'https://api.anthropic.com/api/oauth/usage'
export const CODEX_USAGE_URL = 'https://chatgpt.com/backend-api/wham/usage'
export const KIMI_USAGE_URL = 'https://api.kimi.com/coding/v1/usages'
export const ZAI_USAGE_URL = 'https://api.z.ai/api/monitor/usage/quota/limit'

export type UsageFetch = (url: string, init?: RequestInit) => Promise<Response>

export interface UsageWindow {
  /** Human label, e.g. "5-hour", "weekly". */
  readonly label: string
  /** 0–100 percentage of the window's allowance already consumed. */
  readonly usedPercent: number
  /** Epoch milliseconds when the window resets, when the provider reports it. */
  readonly resetsAt?: number
  /** Seconds until reset, when the provider reports only a duration. */
  readonly resetAfterSeconds?: number
  /** Optional provider detail, e.g. "opus", "sonnet", "tokens". */
  readonly detail?: string
}

export interface ProviderUsageReport {
  readonly provider: string
  readonly planType?: string
  readonly windows: readonly UsageWindow[]
  readonly fetchedAt: number
}

export interface UsageRequestOptions {
  /** Swap the Claude Code sign-in reader (tests). */
  readonly claudeCodeLogin?: typeof claudeCodeLogin
  /** Model-facing reports reject ambiguous units and malformed percentages. */
  readonly strictUnits?: boolean
  readonly fetchImplementation?: UsageFetch
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly signal?: AbortSignal
}

const recordOf = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}

const finite = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined

const stringOf = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim() ? value.trim() : undefined

/** "2026-09-03T12:00:00Z" or epoch seconds/ms → epoch ms, when parseable. */
function epochMs(value: unknown): number | undefined {
  if (typeof value === 'string') {
    const parsed = Date.parse(value)
    return Number.isNaN(parsed) ? undefined : parsed
  }
  const number = finite(value)
  if (number === undefined) return undefined
  return number < 10_000_000_000 ? number * 1_000 : number
}

async function fetchJson(
  provider: string,
  url: string,
  headers: Record<string, string>,
  options: UsageRequestOptions,
): Promise<Record<string, unknown>> {
  options.signal?.throwIfAborted()
  const deadline = new AbortController()
  const timer = setTimeout(() => deadline.abort(new Error('Usage request timed out')), 10000)
  const signal = options.signal ? AbortSignal.any([options.signal, deadline.signal]) : deadline.signal
  try {
    const fetcher = options.fetchImplementation ?? (globalThis.fetch as UsageFetch)
    const response = await fetcher(url, { headers, method: 'GET', redirect: 'error', signal })
    signal.throwIfAborted()
    if (!response.ok) {
      await response.body?.cancel()
      throw new ConfigurationError(provider, provider + ' usage request failed (' + response.status + ')')
    }
    const text = await usageResponseText(response, signal)
    signal.throwIfAborted()
    return recordOf(JSON.parse(text))
  } finally { clearTimeout(timer) }
}

export const MAX_USAGE_RESPONSE_BYTES = 262144
async function usageResponseText(response: Response, signal: AbortSignal): Promise<string> {
  if (Number(response.headers.get('content-length')) > MAX_USAGE_RESPONSE_BYTES) {
    await response.body?.cancel()
    throw new Error('Usage response exceeds size limit')
  }
  if (!response.body) throw new Error('Usage response has no body')
  const reader = response.body.getReader(), decoder = new TextDecoder('utf-8', { fatal: true })
  let bytes = 0, text = ''
  const abort = () => { void reader.cancel(signal.reason).catch(() => {}) }
  signal.addEventListener('abort', abort, { once: true })
  try {
    signal.throwIfAborted()
    for (;;) {
      const chunk = await reader.read()
      signal.throwIfAborted()
      if (chunk.done) break
      bytes += chunk.value.byteLength
      if (bytes > MAX_USAGE_RESPONSE_BYTES) throw new Error('Usage response exceeds size limit')
      text += decoder.decode(chunk.value, { stream: true })
    }
    return text + decoder.decode()
  } catch (error) {
    try { await reader.cancel(error) } catch { /* Preserve the original transport/parse failure. */ }
    throw error
  } finally { signal.removeEventListener('abort', abort); reader.releaseLock() }
}

const percentFromFraction = (value: number): number =>
  value <= 1 ? value * 100 : value

function claudeWindow(key: string, body: Record<string, unknown>, label: string, detail?: string): UsageWindow | undefined {
  const window = recordOf(body[key])
  const utilization = finite(window.utilization)
  if (utilization === undefined) return undefined
  const resetsAt = epochMs(window.resets_at)
  return {
    label,
    usedPercent: Math.max(0, Math.min(100, utilization)),
    ...(resetsAt !== undefined ? { resetsAt } : {}),
    ...(detail ? { detail } : {}),
  }
}

/** A window's name from its length: 300 minutes → "5-hour", 7 days → "weekly". */
export function windowNameForSeconds(seconds: number): string {
  const hours = seconds / 3_600
  if (Math.abs(hours - 24) < 0.5) return 'daily'
  if (Math.abs(hours - 168) < 1) return 'weekly'
  if (hours >= 27 * 24 && hours <= 31 * 24) return 'monthly'
  if (hours < 48 && Number.isInteger(Math.round(hours * 100) / 100)) return `${Math.round(hours)}-hour`
  return hours < 48 ? `${Math.round(seconds / 60)}-minute` : `${Math.round(hours / 24)}-day`
}

const TIME_UNIT_SECONDS: Readonly<Record<string, number>> = { SECOND: 1, MINUTE: 60, HOUR: 3_600, DAY: 86_400, WEEK: 604_800, MONTH: 2_592_000 }
/** Z.ai's numeric window units (3 = hours, 6 = weeks), confirmed against live responses. */
const ZAI_UNIT_SECONDS: Readonly<Record<number, number>> = { 3: 3_600, 4: 86_400, 5: 2_592_000, 6: 604_800 }

const numberish = (value: unknown): number | undefined => {
  const number = typeof value === 'string' && value.trim() !== '' ? Number(value) : value
  return typeof number === 'number' && Number.isFinite(number) ? number : undefined
}
const firstNumber = (record: Record<string, unknown>, keys: readonly string[]): number | undefined => {
  for (const key of keys) { const value = numberish(record[key]); if (value !== undefined) return value }
  return undefined
}

function windowSeconds(record: Record<string, unknown>, zaiUnits: boolean): number | undefined {
  const window = recordOf(record.window)
  const duration = numberish(window.duration ?? record.duration)
  const unit = stringOf(window.timeUnit ?? record.timeUnit ?? window.time_unit ?? record.time_unit)?.toUpperCase()
  if (duration !== undefined && unit) {
    const scale = Object.entries(TIME_UNIT_SECONDS).find(([name]) => unit.includes(name))?.[1]
    if (scale) return duration * scale
  }
  const explicit = firstNumber(record, ['limit_window_seconds', 'window_seconds', 'windowSeconds', 'period_seconds'])
  if (explicit !== undefined) return explicit
  if (zaiUnits) {
    const scale = ZAI_UNIT_SECONDS[numberish(record.unit) ?? -1]
    const count = numberish(record.number)
    if (scale && count) return scale * count
  }
  return undefined
}

function labelFromKey(key: string): string | undefined {
  const name = key.toLowerCase()
  if (/(^|_)(5h|five_?hour|5_?hour)s?($|_)/.test(name) || name.endsWith('5h')) return '5-hour'
  if (/7d|seven_?day|week/.test(name)) return 'weekly'
  if (/(^|_)(1d|24h|day|daily)($|_)/.test(name)) return 'daily'
  if (/month/.test(name)) return 'monthly'
  return undefined
}

export interface DiscoverUsageOptions {
  /** Labels for keys whose meaning the payload does not state (Kimi's `usage` is the weekly total). */
  readonly keyLabels?: Readonly<Record<string, string>>
  /** Read Z.ai-style numeric `unit` × `number` window lengths. */
  readonly zaiUnits?: boolean
}

function quotaWindow(record: Record<string, unknown>, key: string, options: DiscoverUsageOptions): UsageWindow | undefined {
  // Kimi nests the counts one level down, beside the window length.
  const counts = Object.keys(recordOf(record.detail)).length ? recordOf(record.detail) : record
  const explicit = firstNumber(counts, ['used_percent', 'usedPercent', 'used_percentage', 'utilization', 'percentage', 'percent'])
  const ratio = firstNumber(counts, ['used_ratio', 'usedRatio'])
  const limit = firstNumber(counts, ['limit', 'total', 'quota', 'max'])
  const remaining = firstNumber(counts, ['remaining', 'left', 'available'])
  const used = firstNumber(counts, ['used', 'consumed', 'currentValue', 'current_value']) ?? (limit !== undefined && remaining !== undefined ? limit - remaining : undefined)
  const percent = explicit ?? (ratio !== undefined ? percentFromFraction(ratio) : undefined)
    ?? (limit !== undefined && limit > 0 && used !== undefined ? (used / limit) * 100 : undefined)
  if (percent === undefined) return undefined
  const seconds = windowSeconds(record, options.zaiUnits === true)
  const named = [record.name, record.title, record.label, record.scope].map(stringOf).find(text => text && !/^[A-Z0-9_]+$/.test(text))
  const label = options.keyLabels?.[key] ?? named ?? (seconds !== undefined ? windowNameForSeconds(seconds) : undefined) ?? labelFromKey(key)
  if (!label) return undefined
  const resetsAt = epochMs(counts.reset_at ?? counts.resetAt ?? counts.reset_time ?? counts.resetTime ?? counts.resets_at ?? counts.resetsAt
    ?? counts.nextResetTime ?? counts.next_reset_time ?? counts.next_reset_at)
  const resetAfterSeconds = firstNumber(counts, ['reset_after_seconds', 'reset_in', 'resetIn', 'resets_in_seconds'])
  const left = remaining ?? (limit !== undefined && used !== undefined ? limit - used : undefined)
  return {
    label,
    usedPercent: Math.max(0, Math.min(100, percent)),
    ...(resetsAt !== undefined ? { resetsAt } : resetAfterSeconds !== undefined ? { resetAfterSeconds } : {}),
    ...(left !== undefined && left >= 0 && limit !== undefined ? { detail: `${left} remaining` } : {}),
  }
}

/**
 * Fallback for any quota payload: walk the JSON and keep every record that
 * states how much of a window is used — as a percentage, a 0–1 ratio, or
 * used/limit/remaining counts — named by its window length, its own name, or
 * the key it sits under. Providers change these shapes without notice (Kimi
 * returns three at once); this reads what their own CLIs read instead of
 * failing on the first unfamiliar field. The first answer per label wins, so
 * a payload's summary beats its duplicate breakdowns.
 */
export function discoverUsageWindows(body: unknown, options: DiscoverUsageOptions = {}): UsageWindow[] {
  const windows: UsageWindow[] = []
  const seen = new Set<string>()
  const visit = (value: unknown, key: string, depth: number) => {
    if (depth > 5) return
    if (Array.isArray(value)) { for (const item of value) visit(item, key, depth + 1); return }
    if (!value || typeof value !== 'object') return
    const record = value as Record<string, unknown>
    const window = quotaWindow(record, key, options)
    if (window) {
      const id = window.label.toLowerCase()
      if (!seen.has(id)) { seen.add(id); windows.push(window) }
      return
    }
    for (const [child, nested] of Object.entries(record)) visit(nested, child, depth + 1)
  }
  visit(body, '', 0)
  return windows
}

/** Claude Pro/Max: GET api.anthropic.com/api/oauth/usage (OAuth bearer + oauth beta flag). */
export async function fetchClaudeUsage(
  accessToken: string,
  options: UsageRequestOptions = {},
): Promise<ProviderUsageReport> {
  const url = options.environment?.XERXES_CLAUDE_USAGE_URL?.trim() || CLAUDE_USAGE_URL
  const body = await fetchJson('claude', url, {
    Authorization: `Bearer ${accessToken}`,
    'anthropic-beta': 'oauth-2025-04-20',
  }, options)
  const windows = [
    claudeWindow('five_hour', body, '5-hour'),
    claudeWindow('seven_day', body, 'weekly'),
    claudeWindow('seven_day_opus', body, 'weekly', 'opus'),
    claudeWindow('seven_day_sonnet', body, 'weekly', 'sonnet'),
  ].filter((window): window is UsageWindow => window !== undefined)
  if (!windows.length && !options.strictUnits) windows.push(...discoverUsageWindows(body))
  if (!windows.length) {
    throw new ConfigurationError(
      'claude',
      'Claude usage response carried no recognized windows. If the endpoint moved, set XERXES_CLAUDE_USAGE_URL.',
    )
  }
  return { provider: 'claude', windows, fetchedAt: Date.now() }
}

function codexWindow(window: Record<string, unknown>, fallbackLabel: string, detail?: string, strict = false): UsageWindow | undefined {
  const usedPercent = finite(window.used_percent)
  if (strict && Object.keys(window).length && (usedPercent === undefined || usedPercent < 0 || usedPercent > 100)) throw new ConfigurationError('codex', 'Invalid explicit usage percentage')
  if (usedPercent === undefined) return undefined
  const seconds = finite(window.limit_window_seconds)
  const label = seconds === 18_000 ? '5-hour'
    : seconds === 604_800 ? 'weekly'
    : fallbackLabel
  const resetAfterSeconds = finite(window.reset_after_seconds)
  const resetsAt = epochMs(window.reset_at)
  return {
    label,
    usedPercent: Math.max(0, Math.min(100, usedPercent)),
    ...(resetsAt !== undefined ? { resetsAt } : {}),
    ...(resetAfterSeconds !== undefined ? { resetAfterSeconds } : {}),
    ...(detail ? { detail } : {}),
  }
}

/** ChatGPT Codex plans: GET chatgpt.com/backend-api/wham/usage (bearer + account header). */
export async function fetchCodexUsage(
  headers: Record<string, string>,
  options: UsageRequestOptions = {},
): Promise<ProviderUsageReport> {
  const url = options.environment?.XERXES_CODEX_USAGE_URL?.trim() || CODEX_USAGE_URL
  const body = await fetchJson('codex', url, headers, options)
  const rateLimit = recordOf(body.rate_limit)
  const windows = [
    codexWindow(recordOf(rateLimit.primary_window), options.strictUnits ? 'primary' : '5-hour', undefined, options.strictUnits),
    codexWindow(recordOf(rateLimit.secondary_window), options.strictUnits ? 'secondary' : 'weekly', undefined, options.strictUnits),
  ].filter((window): window is UsageWindow => window !== undefined)
  // Model-specific limits (e.g. Codex Spark) ride a side list with their own
  // 5-hour/weekly pair; tag them so they do not read as the plan's totals.
  const additional = Array.isArray(body.additional_rate_limits) ? body.additional_rate_limits : []
  for (const entry of additional) {
    const record = recordOf(entry)
    const name = stringOf(record.limit_name)
    const nested = recordOf(record.rate_limit)
    for (const window of [
      codexWindow(recordOf(nested.primary_window), options.strictUnits ? 'primary' : '5-hour', name, options.strictUnits),
      codexWindow(recordOf(nested.secondary_window), options.strictUnits ? 'secondary' : 'weekly', name, options.strictUnits),
    ]) {
      if (window) windows.push(window)
    }
  }
  if (!windows.length && !options.strictUnits) windows.push(...discoverUsageWindows(body.rate_limit ?? body))
  if (!windows.length) {
    throw new ConfigurationError(
      'codex',
      'Codex usage response carried no rate-limit windows. If the endpoint moved, set XERXES_CODEX_USAGE_URL.',
    )
  }
  const planType = stringOf(body.plan_type)
  return {
    provider: 'codex',
    ...(planType ? { planType } : {}),
    windows,
    fetchedAt: Date.now(),
  }
}

/** Kimi Code subscription: GET api.kimi.com/coding/v1/usages (device-flow bearer). */
export async function fetchKimiUsage(
  accessToken: string,
  options: UsageRequestOptions = {},
): Promise<ProviderUsageReport> {
  const url = options.environment?.XERXES_KIMI_USAGE_URL?.trim() || KIMI_USAGE_URL
  const body = await fetchJson('kimi', url, { Authorization: `Bearer ${accessToken}` }, options)
  // Kimi's own CLI reads `usage` (the weekly total) and `limits[]` (each with
  // a window length and used/limit counts); older shapes carried percentages.
  const known = kimiWindows(body, options.strictUnits)
  const windows = known.length || options.strictUnits ? known : discoverUsageWindows(body, { keyLabels: { usage: 'weekly' } })
  if (!windows.length) {
    throw new ConfigurationError(
      'kimi',
      'Kimi usage response carried no recognized windows. If the endpoint moved, set XERXES_KIMI_USAGE_URL.',
    )
  }
  const planType = stringOf(body.plan) ?? stringOf(body.plan_type) ?? stringOf(body.membership)
  return {
    provider: 'kimi',
    ...(planType ? { planType } : {}),
    windows,
    fetchedAt: Date.now(),
  }
}

function kimiWindows(body: Record<string, unknown>, strict = false): UsageWindow[] {
  const windows: UsageWindow[] = []
  // Tolerant across the shapes the coding subscription has returned: a
  // top-level list of scope-tagged rows, or keyed five-hour/weekly records.
  const rows = Array.isArray(body.usages) ? body.usages
    : Array.isArray(body.data) ? body.data
    : []
  for (const row of rows) {
    const record = recordOf(row)
    if (strict && (finite(record.used_percent) === undefined || (record.used_percent as number) < 0 || (record.used_percent as number) > 100)) throw new ConfigurationError('kimi', 'Usage row has no valid explicit percentage')
    const used = finite(record.used_percent) ?? finite(record.percentage)
    const scope = (stringOf(record.scope) ?? stringOf(record.type) ?? '').toLowerCase()
    if (used === undefined || !scope) continue
    const resetsAt = epochMs(record.reset_at ?? record.resets_at)
    windows.push({
      label: strict ? scope : scope.includes('week') ? 'weekly' : scope.includes('5') ? '5-hour' : scope,
      usedPercent: Math.max(0, Math.min(100, finite(record.used_percent) !== undefined ? used : percentFromFraction(used))),
      ...(resetsAt !== undefined ? { resetsAt } : {}),
    })
  }
  for (const [key, label] of [['five_hour', '5-hour'], ['weekly', 'weekly'], ['seven_day', 'weekly']] as const) {
    if (strict && body[key] !== undefined) {
      const percent = finite(recordOf(body[key]).utilization)
      if (percent === undefined || percent < 0 || percent > 100) throw new ConfigurationError('kimi', 'Usage window has no valid percentage')
    }
    const window = claudeWindow(key, body, label)
    if (window) windows.push(window)
  }
  return windows
}

/**
 * Z.ai coding plan: GET api.z.ai/api/monitor/usage/quota/limit (API-key
 * bearer). The CN host answers the same path under open.bigmodel.cn.
 */
export async function fetchZaiUsage(
  apiKey: string,
  options: UsageRequestOptions & { readonly host?: 'cn' | 'global' } = {},
): Promise<ProviderUsageReport> {
  const fallback = options.host === 'cn'
    ? ZAI_USAGE_URL.replace('api.z.ai', 'open.bigmodel.cn')
    : ZAI_USAGE_URL
  const url = options.environment?.XERXES_ZAI_USAGE_URL?.trim() || fallback
  const body = await fetchJson('zai', url, { Authorization: `Bearer ${apiKey}` }, options)
  const data = recordOf(body.data)
  const limits = Array.isArray(data.limits) ? data.limits : []
  const windows: UsageWindow[] = limits.flatMap((limit): UsageWindow[] => {
    const record = recordOf(limit)
    const used = finite(record.percentage)
    if (options.strictUnits && (used === undefined || used < 0 || used > 100)) throw new ConfigurationError('zai', 'Usage window has no valid percentage')
    if (used === undefined) return []
    const type = stringOf(record.type) ?? 'quota'
    // The window's length is `unit` × `number` (unit 3 = hours, 6 = weeks):
    // a coding plan's 5-hour and weekly limits share one type name.
    const seconds = windowSeconds(record, true)
    const label = options.strictUnits ? type.toLowerCase()
      : seconds !== undefined ? windowNameForSeconds(seconds)
      // Older responses carried no `number`; keep their established reading.
      : type === 'TOKENS_LIMIT' ? (finite(record.unit) === 5 ? '5-hour' : 'weekly')
      : type === 'TIME_LIMIT' ? '5-hour'
      : type.toLowerCase()
    const nextReset = epochMs(record.nextResetTime)
    const remaining = finite(record.remaining)
    return [{
      label,
      usedPercent: Math.max(0, Math.min(100, used)),
      ...(nextReset !== undefined ? { resetsAt: nextReset } : {}),
      detail: options.strictUnits ? type.toLowerCase() : remaining !== undefined ? `${remaining} remaining` : type.toLowerCase(),
    }]
  })
  if (!windows.length && !options.strictUnits) windows.push(...discoverUsageWindows(body, { zaiUnits: true }))
  if (!windows.length) {
    throw new ConfigurationError(
      'zai',
      'Z.ai usage response carried no quota limits. If the endpoint moved, set XERXES_ZAI_USAGE_URL.',
    )
  }
  return { provider: 'zai', windows, fetchedAt: Date.now() }
}

export const SUBSCRIPTION_USAGE_PROVIDERS = ['claude', 'codex', 'kimi', 'zai'] as const
export type SubscriptionUsageProvider = (typeof SUBSCRIPTION_USAGE_PROVIDERS)[number]

export const PROVIDER_ALIASES: Readonly<Record<string, SubscriptionUsageProvider>> = {
  anthropic: 'claude',
  claude: 'claude',
  'claude-code': 'claude',
  chatgpt: 'codex',
  codex: 'codex',
  'openai-codex': 'codex',
  kimi: 'kimi',
  'kimi-code': 'kimi',
  bigmodel: 'zai',
  zai: 'zai',
  'zai-coding': 'zai',
  'zai-coding-cn': 'zai',
  zhipu: 'zai',
}

export interface SubscriptionUsageError {
  readonly provider: SubscriptionUsageProvider
  readonly message: string
}

export interface SubscriptionUsageCollection {
  readonly errors: readonly SubscriptionUsageError[]
  readonly reports: readonly ProviderUsageReport[]
}

/** Minimal profile shape the collector needs from the daemon's profile store. */
export interface UsageProfile {
  readonly name: string
  readonly provider: string
  readonly api_key: string
  readonly base_url: string
  readonly model: string
}

/**
 * Resolve subscription quota for every configured profile (and OAuth
 * session), not just the active one. A profile whose provider has no quota
 * endpoint is skipped; a profile whose fetch fails lands in `errors`.
 */
export async function collectSubscriptionUsage(
  provider?: string,
  options: UsageRequestOptions & { readonly profiles?: readonly UsageProfile[] } = {},
): Promise<SubscriptionUsageCollection> {
  const normalized = provider?.trim().toLowerCase()
  const targets = normalized
    ? [PROVIDER_ALIASES[normalized] ?? (() => {
        throw new ConfigurationError(
          'usage',
          `unknown usage provider '${provider}'; expected one of ${SUBSCRIPTION_USAGE_PROVIDERS.join(', ')}`,
        )
      })()]
    : [...SUBSCRIPTION_USAGE_PROVIDERS]
  const reports: ProviderUsageReport[] = []
  const errors: SubscriptionUsageError[] = []
  await Promise.all(targets.map(async target => {
    try {
      reports.push(await fetchSubscriptionUsage(target, options))
    } catch (error) {
      errors.push({ provider: target, message: error instanceof Error ? error.message : String(error) })
    }
  }))
  const order = new Map(SUBSCRIPTION_USAGE_PROVIDERS.map((name, index) => [name, index]))
  reports.sort((a, b) => (order.get(a.provider as SubscriptionUsageProvider) ?? 0) - (order.get(b.provider as SubscriptionUsageProvider) ?? 0))
  errors.sort((a, b) => (order.get(a.provider) ?? 0) - (order.get(b.provider) ?? 0))
  return { errors, reports }
}

/** One provider's quota, using the given profiles' keys where the provider needs one. */
export async function fetchSubscriptionUsage(
  provider: SubscriptionUsageProvider,
  options: UsageRequestOptions & { readonly profiles?: readonly UsageProfile[] },
): Promise<ProviderUsageReport> {
  const environment = options.environment ?? process.env
  const profiles = options.profiles ?? []
  const profile = profiles.find(p => PROVIDER_ALIASES[p.provider] === provider)

  if (provider === 'claude') {
    // The `cc` profile runs the Claude Code CLI on its own sign-in, so its
    // plan windows come from that login; every other Claude profile uses
    // Xerxes' Anthropic session. Only `cc` ever reads Claude Code's login.
    if (profile?.provider === 'claude-code') {
      const login = await (options.claudeCodeLogin ?? claudeCodeLogin)({ environment })
      const report = await fetchClaudeUsage(login.accessToken, { ...options, environment })
      return login.subscriptionType && !report.planType ? { ...report, planType: login.subscriptionType } : report
    }
    const credential = await new AnthropicOAuthSession({ environment }).credential(options.signal)
    return fetchClaudeUsage(credential.access, { ...options, environment })
  }
  if (provider === 'codex') {
    const credential = await new CodexSession({ environment }).credential(options.signal)
    return fetchCodexUsage(codexAuthHeaders(credential), { ...options, environment })
  }
  if (provider === 'kimi') {
    // Profile API key first (Kimi Code profiles store the coding key), then
    // the device-flow OAuth session.
    const apiKey = profile?.api_key?.trim() || environment.KIMI_CODE_API_KEY?.trim()
    if (apiKey) return fetchKimiUsage(apiKey, { ...options, environment })
    const credential = await new KimiCodingOAuthSession({ environment }).credential(options.signal)
    return fetchKimiUsage(credential.access, { ...options, environment })
  }
  // zai
  const apiKey = profile?.api_key?.trim()
    || environment.ZHIPU_API_KEY?.trim()
    || environment.ZAI_API_KEY?.trim()
  if (!apiKey) {
    throw new ConfigurationError(
      'zai',
      'No Z.ai API key found. Add a zai-coding profile or set ZHIPU_API_KEY.',
    )
  }
  const isCn = profile?.base_url.includes('bigmodel.cn')
    || environment.XERXES_ZAI_USAGE_URL?.includes('bigmodel.cn')
  return fetchZaiUsage(apiKey, {
    ...options,
    environment,
    host: isCn ? 'cn' : 'global',
  })
}

/** Compact human rendering: `Claude (max) — 5-hour 42% (resets 15:00), weekly 12%`. */
export function formatUsageReport(report: ProviderUsageReport, now = Date.now()): string {
  const name = report.provider[0]!.toUpperCase() + report.provider.slice(1)
  const plan = report.planType ? ` (${report.planType})` : ''
  const windows = report.windows.map(window => {
    const reset = window.resetsAt !== undefined
      ? `resets ${new Date(window.resetsAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
      : window.resetAfterSeconds !== undefined
        ? `resets in ${Math.max(1, Math.round((window.resetAfterSeconds - Math.max(0, now - report.fetchedAt) / 1_000) / 60))}m`
        : undefined
    const detail = window.detail ? ` ${window.detail}` : ''
    return `${window.label}${detail} ${Math.round(window.usedPercent)}%${reset ? ` (${reset})` : ''}`
  })
  return `${name}${plan} — ${windows.join(', ')}`
}
