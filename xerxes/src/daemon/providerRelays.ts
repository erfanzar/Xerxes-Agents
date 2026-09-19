// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomBytes } from 'node:crypto'
import type { ProfileStore, ProviderProfile } from '../bridge/profiles.js'
import { createLlmClient, type LlmClient } from '../llms/client.js'
import { LocalProviderEndpoint } from '../security/localProviderEndpoint.js'
import { LocalProviderRelay, LocalProviderRelayError, type LocalProviderDefaults, type LocalProviderGrant, type LocalProviderGrantView, type RelayPeer } from '../security/localProviderRelay.js'
import { DEFAULT_TEMPERATURE, DEFAULT_TOP_K } from '../llms/samplingDefaults.js'
import { resolveProvider, type ProviderName } from '../llms/providerRegistry.js'
import { resolveTurnThinking } from '../runtime/thinkingLevels.js'
import { providerRouteIdentity } from './agentProvider.js'
import { profileAcceptsModel } from './sessionProvider.js'

interface OwnedRelay { readonly owner: object; readonly peer: RelayPeer; readonly token: string; readonly endpoint: LocalProviderEndpoint }
interface RetiredRelay { readonly owner: object; readonly view: LocalProviderGrantView }
const MAX_RETIRED_RELAYS = 128
export type RelayClientFactory = (model: string, profile: ProviderProfile) => LlmClient

/** Local daemon control surface. Public ids are connection-owned handles, not
 * bearer tokens. The authority engine's tokens and provider credentials never
 * leave this object. These methods are not model tools or slash commands. */
export class DaemonProviderRelays {
  private readonly authority = new LocalProviderRelay()
  private readonly relays = new Map<string, OwnedRelay>()
  // Recent terminal status only: no endpoint, authority token or client resolver.
  private readonly retired = new Map<string, RetiredRelay>()
  constructor(private readonly profiles: Pick<ProfileStore, 'get' | 'list'>,
    private readonly client: RelayClientFactory = (model, profile) => createLlmClient(model, { provider: profile.provider, api_key: profile.api_key, base_url: profile.base_url })) {}

  inventory() {
    return this.profiles.list().map(profile => {
      const provider = relayProvider(profile, profile.model)
      return { name: profile.name, provider: profile.provider, model: profile.model,
      supported: provider !== undefined && provider !== 'claude-code',
      output_limit_mode: provider === 'openai-codex' ? 'provider-controlled' : 'request-bound',
      credential_source: profile.api_key ? 'local saved profile' : 'local provider authentication or environment',
      readiness: 'configured; not verified with a provider call',
      ...(provider === undefined ? { setup: 'Correct the provider route in this local profile before sharing it.' } : {}),
      ...(provider === 'claude-code' ? { setup: 'Use an API or subscription provider profile; the Claude Code subprocess cannot use this relay.' } : {}),
      ...(provider === 'openai-codex' ? { setup: 'Codex subscription controls output length. Explicit consent to provider-controlled output is required; request, concurrency and expiry limits still apply.' } : {}),
    } })
  }

  authorize(owner: object, params: Record<string, unknown>) {
    this.retireSettled()
    if (params.consent !== true || typeof params.destination !== 'string' || typeof params.workspace !== 'string' ||
      typeof params.profile !== 'string' || typeof params.model !== 'string' || typeof params.expires_at !== 'number' ||
      typeof params.max_requests !== 'number' || (params.max_output_tokens !== null && typeof params.max_output_tokens !== 'number') || typeof params.max_concurrent !== 'number') {
      throw new LocalProviderRelayError('invalid_request')
    }
    if ([params.destination, params.workspace, params.profile, params.model].some(value => value.length > 4096 || /[\x00-\x1f\x7f]/.test(value))) throw new LocalProviderRelayError('invalid_request')
    const policy: LocalProviderGrant = { profile: params.profile, model: params.model, expiresAt: params.expires_at,
      maxRequests: params.max_requests, maxOutputTokens: params.max_output_tokens, maxConcurrent: params.max_concurrent }
    const profile = this.profiles.get(policy.profile)
    if (!profile || !profileAcceptsModel(profile, policy.model)) throw new LocalProviderRelayError('route_mismatch')
    const provider = relayProvider(profile, policy.model)
    if (provider === undefined || provider === 'claude-code') throw new LocalProviderRelayError('route_mismatch')
    if (provider === 'openai-codex' && policy.maxOutputTokens !== null) throw new LocalProviderRelayError('output_limit_unsupported')
    // Null is not an omitted/default cap. It is a separate, explicit consent
    // policy, allowed only for the native transport that cannot send a cap.
    if (policy.maxOutputTokens === null && (provider !== 'openai-codex' || params.consent_provider_controlled_output !== true)) throw new LocalProviderRelayError('invalid_request')
    const identity = providerRouteIdentity(policy.model, { provider: profile.provider, baseUrl: profile.base_url })
    const defaults = profileDefaults(profile)
    const peer = Object.freeze({ destination: params.destination, workspace: params.workspace })
    const granted = this.authority.authorize(peer, policy, () => {
      const current = this.profiles.get(policy.profile)
      if (!current || !profileAcceptsModel(current, policy.model)) throw new LocalProviderRelayError('route_changed')
      const routeIdentity = providerRouteIdentity(policy.model, { provider: current.provider, baseUrl: current.base_url })
      // Check BEFORE constructing a client. A newly selected route must not
      // even initialize an SDK while the old grant is still in use.
      if (routeIdentity !== identity) throw new LocalProviderRelayError('route_changed')
      return { routeIdentity, defaults, client: this.client(policy.model, current) }
    }, identity)
    const id = randomBytes(16).toString('hex')
    this.relays.set(id, { owner, peer, token: granted.token, endpoint: new LocalProviderEndpoint(this.authority, peer, granted.token) })
    return { id, ...granted.view }
  }

  async next(owner: object, id: unknown, frame: unknown) {
    this.retireSettled()
    const retired = this.retiredFor(owner, id)
    if (retired) return { error: retired.view.status === 'expired' ? 'grant_expired' : 'grant_revoked' }
    try { return await this.owned(owner, id).endpoint.handle(frame) }
    finally { this.retireSettled() }
  }
  status(owner: object, id: unknown) {
    this.retireSettled()
    const retired = this.retiredFor(owner, id)
    if (retired) return { ...retired.view }
    const relay = this.owned(owner, id)
    return this.authority.inspect(relay.peer, relay.token)
  }
  /** A remote tool may be running between provider requests. Keep its local
   * authority alive until the grant ends, not just while pulling a response. */
  hasLiveGrants(): boolean {
    this.retireSettled()
    for (const relay of this.relays.values()) {
      const view = this.authority.inspect(relay.peer, relay.token)
      if (view.status === 'active' || view.activeRequests > 0) return true
    }
    return false
  }
  revoke(owner: object, id: unknown): void {
    this.retireSettled()
    if (this.retiredFor(owner, id)) return
    const relay = this.owned(owner, id)
    this.authority.revoke(relay.peer, relay.token)
    relay.endpoint.close('grant_revoked')
    this.retireSettled()
  }
  disconnect(owner: object): void {
    for (const [id, relay] of this.relays) if (relay.owner === owner) {
      relay.endpoint.close(); this.authority.release(relay.peer, relay.token); this.relays.delete(id)
    }
    for (const [id, relay] of this.retired) if (relay.owner === owner) this.retired.delete(id)
  }
  close(): void { for (const relay of this.relays.values()) relay.endpoint.close(); this.relays.clear(); this.retired.clear(); this.authority.close() }
  private retiredFor(owner: object, id: unknown): RetiredRelay | undefined {
    const retired = typeof id === 'string' ? this.retired.get(id) : undefined
    if (retired && retired.owner !== owner) throw new LocalProviderRelayError('grant_unavailable')
    return retired
  }
  private retireSettled(): void {
    for (const [id, relay] of this.relays) {
      const view = this.authority.inspect(relay.peer, relay.token)
      // Keep counting noncooperative backend work until it actually settles.
      // Exhausted grants may still have buffered deltas to deliver.
      if ((view.status !== 'revoked' && view.status !== 'expired') || view.activeRequests > 0) continue
      relay.endpoint.close(view.status === 'expired' ? 'grant_expired' : 'grant_revoked')
      this.authority.release(relay.peer, relay.token)
      this.relays.delete(id)
      this.retired.set(id, { owner: relay.owner, view: Object.freeze({ ...view }) })
      while (this.retired.size > MAX_RETIRED_RELAYS) this.retired.delete(this.retired.keys().next().value!)
    }
  }
  private owned(owner: object, id: unknown): OwnedRelay {
    const relay = typeof id === 'string' ? this.relays.get(id) : undefined
    if (!relay || relay.owner !== owner) throw new LocalProviderRelayError('grant_unavailable')
    return relay
  }
}

function relayProvider(profile: ProviderProfile, model: string): ProviderName | undefined {
  try { return resolveProvider(model, { provider: profile.provider, base_url: profile.base_url }) }
  catch { return undefined } // Inventory must remain usable for invalid saved profiles.
}

/** Snapshot only supported sampling fields at consent; credentials are still
 * refreshed per call. An unrelated profile edit cannot change a live grant's
 * behavior, and arbitrary profile fields never enter a completion request. */
function profileDefaults(profile: ProviderProfile): LocalProviderDefaults {
  const sampling = profile.sampling
  const numeric = (key: string, min: number, max = Infinity, integer = false): number | undefined => {
    const value = sampling[key]
    if (value === undefined) return undefined
    if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isSafeInteger(value))) throw new LocalProviderRelayError('invalid_request')
    return value
  }
  const budgetTokens = numeric('thinking_budget', 1, Infinity, true)
  const maxTokens = numeric('max_tokens', 1, Infinity, true)
  const topP = numeric('top_p', 0, 1)
  const thinking = resolveTurnThinking({ prompt: '', ultraMode: false, defaults: {
    ...(typeof sampling.thinking === 'boolean' ? { enabled: sampling.thinking } : {}),
    ...(budgetTokens === undefined ? {} : { budgetTokens }),
    ...(typeof sampling.reasoning_effort === 'string' ? { effort: sampling.reasoning_effort } : {}),
  } })
  return Object.freeze({
    ...(maxTokens === undefined ? {} : { maxTokens }),
    temperature: numeric('temperature', 0, 2) ?? DEFAULT_TEMPERATURE,
    topK: numeric('top_k', 0, Infinity, true) ?? DEFAULT_TOP_K,
    ...(topP === undefined ? {} : { topP }),
    ...(typeof sampling.service_tier === 'string' && ['auto', 'default', 'flex', 'priority'].includes(sampling.service_tier) ? { serviceTier: sampling.service_tier } : {}),
    ...(thinking ? { thinking: Object.freeze({ budgetTokens: thinking.budgetTokens, effort: thinking.effort }) } : {}),
  })
}
