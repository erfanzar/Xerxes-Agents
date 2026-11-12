// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomUUID } from 'node:crypto'

import { ProviderAccountRegistry, type OAuthAccountKind, type ProviderSyncPayload } from '../../auth/providerAccountRegistry.js'
import { nativeGeminiBaseUrl } from '../../llms/gemini.js'
import type { RemoteTarget } from './remote.js'
import { remoteProfileReceipt, spawnSecretWriter, type RemoteProviderSyncInput } from './remoteCredentials.js'

type Call = (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>
type LoginCall = (method: string, params: Record<string, unknown>, onAuthorizationUrl: (url: string) => void) => Promise<Record<string, unknown>>
interface SyncReview { readonly id: string; readonly accountId: string; readonly target: string; readonly expiresAt: number }

const record = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object' && !Array.isArray(value)
const counter = (value: unknown): value is number => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
const price = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0
const accountIdentity = (payload: ProviderSyncPayload): string =>
  createHash('sha256').update(JSON.stringify([
    payload.kind, payload.profile.provider, payload.profile.base_url, payload.profile.api_key,
    payload.profile.oauth_account_id, payload.profile.isolated_auth,
    ...(payload.kind === 'api_key' ? [] : [payload.accountId, payload.credential]),
  ])).digest('hex')

/** Profile-tagged ledger only; this is not credential-level attribution. */
function recordedAccountUsage(report: Record<string, unknown>, profile: string): Record<string, unknown> {
  const unavailable = { status: 'unavailable', profile }
  if (report.ok === false || !Array.isArray(report.models)) return unavailable
  const models: Record<string, unknown>[] = []
  const totals = { calls: 0, input_tokens: 0, output_tokens: 0, cache_read_tokens: 0, cache_write_tokens: 0 }
  let cost = 0, priced = true
  for (const row of report.models) {
    if (!record(row) || row.profile !== profile) continue
    if (typeof row.model !== 'string' || !counter(row.calls) || !counter(row.input_tokens) || !counter(row.output_tokens)
      || !counter(row.cache_read_tokens) || !counter(row.cache_write_tokens)) return unavailable
    const counts = { calls: row.calls, input_tokens: row.input_tokens, output_tokens: row.output_tokens,
      cache_read_tokens: row.cache_read_tokens, cache_write_tokens: row.cache_write_tokens }
    for (const field of ['calls', 'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens'] as const) {
      totals[field] += counts[field]
      if (!counter(totals[field])) return unavailable
    }
    if (price(row.cost_usd)) cost += row.cost_usd
    else priced = false
    models.push({ model: row.model, ...counts, ...(price(row.cost_usd) ? { cost_usd: row.cost_usd } : {}) })
  }
  return { status: 'available', source: 'usage.report', profile, since: counter(report.models_since) ? report.models_since : null,
    ...totals, models, ...(models.length && priced && price(cost) ? { cost_usd: cost } : {}) }
}

/** Normalize OAuth's ProviderUsageReport to the existing profile-quota wire shape. */
export function providerAccountUsageWire(report: unknown): Record<string, unknown> {
  if (!report || typeof report !== 'object' || Array.isArray(report)) return { status: 'unknown', reason: 'Usage is unavailable.' }
  const value = report as Record<string, unknown>
  if (typeof value.status === 'string') return value
  if (!Array.isArray(value.windows) || value.windows.length === 0) return { status: 'unknown', reason: 'The provider reported no usage windows.' }
  const windows = value.windows.map(row => {
    if (!row || typeof row !== 'object' || Array.isArray(row)) throw new Error('Invalid provider usage window')
    const window = row as Record<string, unknown>
    if (typeof window.label !== 'string' || typeof window.usedPercent !== 'number' || !Number.isFinite(window.usedPercent)) throw new Error('Invalid provider usage window')
    const resetsAt = typeof window.resetsAt === 'number' && Number.isFinite(window.resetsAt) ? new Date(window.resetsAt).toISOString() : null
    return {
      label: window.label,
      used_percent: window.usedPercent,
      resets_at: resetsAt,
      reset_after_seconds: typeof window.resetAfterSeconds === 'number' && Number.isFinite(window.resetAfterSeconds) ? window.resetAfterSeconds : null,
      detail: typeof window.detail === 'string' ? window.detail : null,
    }
  })
  return { status: 'available', windows, observed_at: typeof value.fetchedAt === 'number' && Number.isFinite(value.fetchedAt) ? new Date(value.fetchedAt).toISOString() : null }
}

/** One global account center for local and SSH workspaces. */
export class GlobalProviderAccounts {
  private review: SyncReview | undefined
  private readonly syncedAccounts = new Map<string, string>()
  private recordedReport: Promise<Record<string, unknown>> | undefined
  constructor(
    private readonly registry: ProviderAccountRegistry,
    private readonly remote: RemoteTarget | undefined,
    private readonly call: Call,
    private readonly loginCall: LoginCall,
    private readonly now: () => number = Date.now,
    private readonly writer?: (payload: RemoteProviderSyncInput) => Promise<string>,
  ) {}

  async list(): Promise<Record<string, unknown>> {
    const [accounts, providers] = await Promise.all([this.registry.list(), this.call('provider_list', {})])
    const rows = providers.ok !== false && Array.isArray(providers.profiles) ? providers.profiles.filter(record) : []
    const listed = await Promise.all(accounts.map(async account => {
      let available = rows.some(row => row.name === account.profileName)
      if (account.provider === 'gemini') {
        available = rows.some(row => row.name === account.profileName && row.provider === 'gemini' && typeof row.base_url === 'string' && nativeGeminiBaseUrl(row.base_url) === 'https://generativelanguage.googleapis.com/v1beta')
      }
      if (available && this.remote) {
        // A name alone proves nothing about a remote key or login. A confirmed
        // transfer authorizes only this exact local credential, in this center.
        const confirmed = this.syncedAccounts.get(account.id)
        available = false
        if (confirmed !== undefined) {
          try { available = confirmed === accountIdentity(await this.registry.syncPayload(account.id)) }
          catch { /* Missing/unreadable credentials cannot retain availability. */ }
        }
      }
      return { ...account, available }
    }))
    return {
      accounts: listed,
      can_sync: Boolean(this.remote),
      destination: this.remote?.target ?? null,
    }
  }

  async login(kind: OAuthAccountKind, label: string, onAuthorizationUrl: (url: string) => void = () => {}): Promise<Record<string, unknown>> {
    const account = await this.registry.prepare(kind, label)
    try {
      const result = await this.loginCall('provider_account.login', { account_id: account.id }, onAuthorizationUrl)
      if (result.ok === false) throw new Error(typeof result.error === 'string' ? result.error : 'OAuth sign-in failed')
      const registered = (await this.registry.list()).find(row => row.id === account.id)
      if (!registered) throw new Error('Provider account sign-in did not complete')
      return { account: registered, usage: await this.registry.usage(account.id).catch(error => ({ status: 'unknown', reason: error instanceof Error ? error.message : String(error) })) }
    }
    catch (error) {
      try { await this.registry.discard(account.id) }
      catch (cleanup) { throw new AggregateError([error, cleanup], 'Sign-in failed and its temporary files could not be removed') }
      throw error
    }
  }

  async createGemini(label: string, apiKey: string, model: string): Promise<Record<string, unknown>> {
    return { ok: true, account: await this.registry.createGemini(label, apiKey, model) }
  }

  private async requireRemoteAccountIsolation(): Promise<void> {
    if (!this.remote) return
    const message = 'Provider account isolation is unavailable; update remote runtime to 0.8.35 or later before syncing or selecting accounts'
    let status: Record<string, unknown>
    try { status = await this.call('runtime.status', {}) }
    catch { throw new Error(message) }
    if (status.ok === false || status.provider_account_isolation_v1_supported !== true) throw new Error(message)
  }

  async activate(accountId: string, sessionKey: string): Promise<Record<string, unknown>> {
    const account = (await this.registry.list()).find(row => row.id === accountId)
    if (!account) throw new Error('Provider account not found')
    if (!sessionKey.trim()) throw new Error('An idle conversation is required')
    await this.requireRemoteAccountIsolation()
    if (this.remote || account.provider === 'gemini') {
      const label = account.provider === 'gemini' ? 'Gemini account' : 'Provider account'
      let listed: Record<string, unknown>
      try { listed = await this.list() }
      catch { throw new Error(`Could not check ${label} availability on this workspace`) }
      if (!Array.isArray(listed.accounts) || !listed.accounts.some(row => record(row) && row.id === accountId && row.available === true)) throw new Error(`${label} is not available on this workspace; review Sync to SSH first for a remote workspace`)
      try {
        const result = await this.call('provider_select', { name: account.profileName, session_key: sessionKey })
        if (result.ok === false) throw new Error('selection failed')
        return { ok: true }
      } catch { throw new Error(`Could not switch ${label}; ensure the conversation is idle and the saved profile is available`) }
    }
    return this.call('provider_select', { name: account.profileName, session_key: sessionKey })
  }

  async usage(accountId: string): Promise<unknown> {
    const account = (await this.registry.list()).find(row => row.id === accountId)
    if (!account) throw new Error('Provider account not found')
    const quota = providerAccountUsageWire(await this.registry.usage(accountId))
    if (account.provider !== 'gemini') return quota
    // Share concurrent ledger requests, not a stale cache. The daemon owns the
    // recorded window and any published prices; never estimate a balance here.
    const pending = this.recordedReport ??= Promise.resolve().then(() => this.call('usage.report', { refresh: false }))
    let recorded: Record<string, unknown>
    try { recorded = recordedAccountUsage(await pending, account.profileName) }
    catch { recorded = { status: 'unavailable', profile: account.profileName } }
    finally { if (this.recordedReport === pending) this.recordedReport = undefined }
    return { ...quota, recorded }
  }

  reviewSync(accountId: string): Record<string, unknown> {
    if (!this.remote) throw new Error('Open an SSH workspace to synchronize accounts')
    const id = randomUUID()
    this.review = { id, accountId, target: this.remote.target, expiresAt: this.now() + 300_000 }
    return { review: id, account_id: accountId, destination: this.remote.target, expires_at: this.review.expiresAt }
  }

  async confirmSync(reviewId: string, confirm: boolean): Promise<Record<string, unknown>> {
    const review = this.review
    if (!this.remote || !confirm || !review || review.id !== reviewId || review.target !== this.remote.target || review.expiresAt < this.now()) {
      throw new Error('Account sync review expired; review it again')
    }
    this.review = undefined
    this.syncedAccounts.delete(review.accountId)
    await this.requireRemoteAccountIsolation()
    try {
      const payload = await this.registry.syncPayload(review.accountId)
      const identity = accountIdentity(payload)
      const profile = remoteProfileReceipt(this.writer ? await this.writer(payload) : await spawnSecretWriter(this.remote, payload, new AbortController().signal))
      if (profile !== payload.profile.name) throw new Error('Remote credential receiver acknowledged a different profile')
      const reloaded = await this.call('runtime.reload', {})
      if (reloaded.ok === false) throw new Error('Remote profile reload failed')
      this.syncedAccounts.set(review.accountId, identity)
      return { ok: true, profile, destination: this.remote.target }
    } catch {
      throw new Error('Provider account sync was not acknowledged; review the transfer again before retrying')
    }
  }

  async remove(accountId: string): Promise<Record<string, unknown>> {
    await this.registry.remove(accountId)
    this.syncedAccounts.delete(accountId)
    return { ok: true }
  }
}
