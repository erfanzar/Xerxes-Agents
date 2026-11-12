// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'

import type { RemoteTarget } from './remote.js'
import { localKeyedProfiles } from './profileSync.js'
import type { RemoteProfileInput } from './remoteCredentials.js'

export type ProviderAccountCall = (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>
export type ProviderAccountWriter = (profile: RemoteProfileInput, signal: AbortSignal) => Promise<string>
export interface ProviderAccountSummary { readonly name: string; readonly provider?: string; readonly model: string; readonly bound: boolean }

interface BindReview {
  readonly id: string
  readonly profile: string
  readonly target: string
  readonly expiresAt: number
}

/** Main-process-owned account binding; the renderer never receives the key. */
export class ProviderAccountDirectory {
  private review: BindReview | undefined
  private boundProfile: string | undefined
  constructor(
    private readonly target: RemoteTarget,
    private readonly profilesFile: string,
    private readonly call: ProviderAccountCall,
    private readonly write: ProviderAccountWriter,
    private readonly now: () => number = Date.now,
  ) {}

  async list(): Promise<{ profiles: readonly ProviderAccountSummary[]; bound_profile?: string; destination: string }> {
    const profiles = await localKeyedProfiles(this.profilesFile)
    return {
      destination: this.target.target,
      profiles: profiles.map(profile => ({
        name: String(profile.name), model: String(profile.model), bound: profile.name === this.boundProfile,
        ...(typeof profile.provider === 'string' ? { provider: profile.provider } : {}),
      })),
      ...(this.boundProfile ? { bound_profile: this.boundProfile } : {}),
    }
  }

  async reviewBind(name: string): Promise<{ review: string; profile: string; destination: string; expires_at: number }> {
    const profiles = await localKeyedProfiles(this.profilesFile)
    if (!profiles.some(profile => profile.name === name)) throw new Error('Choose a saved provider account')
    const id = randomUUID(), expiresAt = this.now() + 300_000
    this.review = { id, profile: name, target: this.target.target, expiresAt }
    return { review: id, profile: name, destination: this.target.target, expires_at: expiresAt }
  }

  async confirmBind(reviewId: string, confirm: boolean, signal: AbortSignal): Promise<{ profile: string }> {
    const review = this.review
    if (!confirm || !review || review.id !== reviewId || review.target !== this.target.target || review.expiresAt < this.now()) {
      throw new Error('Provider binding review expired; review it again')
    }
    this.review = undefined
    const profile = (await localKeyedProfiles(this.profilesFile)).find(row => row.name === review.profile)
    if (!profile) throw new Error('The saved provider account changed; review it again')
    const input: RemoteProfileInput = {
      name: String(profile.name), base_url: String(profile.base_url), api_key: String(profile.api_key), model: String(profile.model),
      ...(typeof profile.provider === 'string' ? { provider: profile.provider } : {}),
      ...(profile.sampling && typeof profile.sampling === 'object' ? { sampling: profile.sampling as Record<string, unknown> } : {}),
    }
    const received = await this.write(input, signal)
    if (received !== review.profile) throw new Error('Remote credential receiver acknowledged a different profile')
    this.boundProfile = review.profile
    await this.call('runtime.reload', {})
    return { profile: review.profile }
  }

  async activate(sessionKey: string): Promise<Record<string, unknown>> {
    if (!this.boundProfile) throw new Error('Bind a provider account to this host first')
    if (!sessionKey.trim()) throw new Error('An idle conversation is required')
    return this.call('provider_select', { name: this.boundProfile, session_key: sessionKey })
  }

  quota(): Promise<Record<string, unknown>> {
    return this.call('usage.report', { refresh: false })
  }
}
