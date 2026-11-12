// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'

import { ProviderAccountRegistry, type OAuthAccountKind } from '../../auth/providerAccountRegistry.js'
import type { RemoteTarget } from './remote.js'
import { remoteProfileReceipt, spawnSecretWriter, type RemoteProviderSyncInput } from './remoteCredentials.js'

type Call = (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>
type LoginCall = (method: string, params: Record<string, unknown>, onAuthorizationUrl: (url: string) => void) => Promise<Record<string, unknown>>
interface SyncReview { readonly id: string; readonly accountId: string; readonly target: string; readonly expiresAt: number }

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
    const rows = Array.isArray(providers.profiles) ? providers.profiles as Record<string, unknown>[] : []
    return {
      accounts: accounts.map(account => ({ ...account, available: rows.some(row => row.name === account.profileName) })),
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

  async activate(accountId: string, sessionKey: string): Promise<Record<string, unknown>> {
    const account = (await this.registry.list()).find(row => row.id === accountId)
    if (!account) throw new Error('Provider account not found')
    if (!sessionKey.trim()) throw new Error('An idle conversation is required')
    return this.call('provider_select', { name: account.profileName, session_key: sessionKey })
  }

  async usage(accountId: string): Promise<unknown> {
    return providerAccountUsageWire(await this.registry.usage(accountId))
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
    const payload = await this.registry.syncPayload(review.accountId)
    const profile = remoteProfileReceipt(this.writer ? await this.writer(payload) : await spawnSecretWriter(this.remote, payload, new AbortController().signal))
    if (profile !== payload.profile.name) throw new Error('Remote credential receiver acknowledged a different profile')
    await this.call('runtime.reload', {})
    return { ok: true, profile, destination: this.remote.target }
  }

  async remove(accountId: string): Promise<Record<string, unknown>> {
    if (accountId.startsWith('api:')) throw new Error('Delete API-key profiles from their profile editor')
    await this.registry.remove(accountId)
    return { ok: true }
  }
}
