// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { classifyError, ErrorKind } from '../runtime/errorClassifier.js'
import { createHash, randomBytes } from 'node:crypto'
import type { CompletionRequest, LlmClient, LlmDelta } from '../llms/client.js'
import { OUTPUT_TOKEN_LIMIT, OutputTokenLimitError } from '../llms/outputTokenLimit.js'

/** Created by the local transport owner, never deserialized from remote input. */
export interface RelayPeer { readonly destination: string; readonly workspace: string }
/** Local host defaults, never supplied by the remote peer. */
export type LocalProviderDefaults = Readonly<Pick<CompletionRequest, 'maxTokens' | 'temperature' | 'topK' | 'topP' | 'serviceTier' | 'thinking'>>
interface LocalProviderRoute { readonly client: LlmClient; readonly routeIdentity: string; readonly defaults?: LocalProviderDefaults }
export interface LocalProviderGrant {
  readonly profile: string
  readonly model: string
  readonly expiresAt: number
  readonly maxRequests: number
  /** Null requires separate host consent to provider-controlled output. */
  readonly maxOutputTokens: number | null
  readonly maxConcurrent: number
}
export interface LocalProviderGrantView extends LocalProviderGrant {
  readonly destination: string
  readonly workspace: string
  readonly requestsUsed: number
  readonly activeRequests: number
  readonly status: 'active' | 'expired' | 'revoked' | 'exhausted'
  readonly execution: 'local provider; remote tools'
  readonly persistence: 'memory only'
  readonly outputLimitMode: 'request-bound' | 'provider-controlled'
}
export type RelayFailureCode = 'grant_unavailable' | 'grant_expired' | 'grant_revoked' | 'request_limit' | 'concurrency_limit' | 'route_mismatch' | 'route_changed' | 'invalid_request' | 'cancelled' | 'provider_failed' | 'context_overflow' | 'output_limit' | 'output_limit_unsupported'
const FAILURE_TEXT: Record<RelayFailureCode, string> = {
  output_limit: 'Provider output settings exceed the authorized limit. Lower reasoning or explicitly authorize a larger output limit.',
  output_limit_unsupported: 'This provider cannot enforce a per-request output limit. Explicitly consent to provider-controlled output in local setup, or select a provider with output limits.',
  context_overflow: 'The local provider context window was exceeded. Compact the conversation before retrying.',
  grant_unavailable: 'Local provider grant is unavailable for this connection. Review local setup before retrying.',
  grant_expired: 'Local provider grant expired. Renew it explicitly in local setup.',
  grant_revoked: 'Local provider access was revoked. Choose a provider explicitly before continuing.',
  request_limit: 'Local provider grant reached its request limit. Review usage before renewing.',
  concurrency_limit: 'Local provider grant is at its concurrency limit. Wait for an active request or cancel it.',
  route_changed: 'The selected local provider route changed. Review local setup before continuing.',
  route_mismatch: 'This model is not covered by the local provider grant. Select an authorized model or review local setup.',
  invalid_request: 'The local provider relay rejected an unsupported or oversized request.',
  cancelled: 'Local provider request cancelled.',
  provider_failed: 'The local provider request failed. Check the selected provider on the local machine; its diagnostic text was not forwarded.',
}
export function isRelayFailureCode(value: unknown): value is RelayFailureCode {
  return typeof value === 'string' && Object.hasOwn(FAILURE_TEXT, value)
}
export class LocalProviderRelayError extends Error {
  constructor(readonly code: RelayFailureCode) { super(FAILURE_TEXT[code]); this.name = 'LocalProviderRelayError' }
}
interface GrantState {
  readonly peer: RelayPeer
  readonly policy: LocalProviderGrant
  readonly destination: string
  readonly workspace: string
  readonly routeIdentity: string
  readonly cacheScope: string
  readonly resolveClient: () => LocalProviderRoute
  readonly active: Set<AbortController>
  readonly timer: ReturnType<typeof setTimeout>
  requestsUsed: number
  stopped?: 'expired' | 'revoked'
}

/** Local-only authority primitive. Its host must first obtain explicit consent
 * for the displayed destination, profile, model, expiry and limits. It does not
 * discover credentials, open network listeners, execute tools or persist grants.
 * The host resolves the exact local profile on every request and must reject a
 * changed route; remote input can never supply endpoints, headers or credentials.
 */
export class LocalProviderRelay {
  private readonly grants = new Map<string, GrantState>()
  constructor(private readonly now: () => number = Date.now) {}

  authorize(peer: RelayPeer, policy: LocalProviderGrant, resolveClient: () => LocalProviderRoute, routeIdentity: string): { token: string; view: LocalProviderGrantView } {
    const lifetime = policy.expiresAt - this.now()
    if (!routeIdentity || !peer.destination || !peer.workspace.startsWith('/') || !policy.profile || !policy.model ||
      !Number.isFinite(lifetime) || lifetime <= 0 || lifetime > 8 * 60 * 60 * 1000 ||
      !Number.isSafeInteger(policy.maxRequests) || policy.maxRequests < 1 || policy.maxRequests > 10_000 ||
      (policy.maxOutputTokens !== null && (!Number.isSafeInteger(policy.maxOutputTokens) || policy.maxOutputTokens < 1 || policy.maxOutputTokens > 1_000_000)) ||
      !Number.isSafeInteger(policy.maxConcurrent) || policy.maxConcurrent < 1 || policy.maxConcurrent > 16 || this.grants.size >= 128) {
      throw new LocalProviderRelayError('invalid_request')
    }
    const token = randomBytes(32).toString('hex')
    const state: GrantState = { peer, destination: peer.destination, workspace: peer.workspace, routeIdentity, cacheScope: randomBytes(16).toString('hex'), policy: Object.freeze({ ...policy }), resolveClient, active: new Set(), requestsUsed: 0,
      timer: setTimeout(() => this.stop(state, 'expired'), lifetime) }
    state.timer.unref?.()
    this.grants.set(token, state)
    return { token, view: this.view(state) }
  }

  inspect(peer: RelayPeer, token: string): LocalProviderGrantView { return this.view(this.owned(peer, token)) }
  revoke(peer: RelayPeer, token: string): void { this.stop(this.owned(peer, token), 'revoked') }
  release(peer: RelayPeer, token: string): void { this.stop(this.owned(peer, token), 'revoked'); this.grants.delete(token) }
  /** A dropped transport aborts its in-flight requests. Reattachment to the
   * same locally owned peer can use the remaining grant; no request replays. */
  disconnect(peer: RelayPeer): void {
    for (const grant of this.grants.values()) if (grant.peer === peer) {
      for (const controller of grant.active) controller.abort(new LocalProviderRelayError('cancelled'))
    }
  }
  close(): void { for (const state of this.grants.values()) this.stop(state, 'revoked'); this.grants.clear() }

  async *stream(peer: RelayPeer, token: string, request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    const grant = this.owned(peer, token)
    this.assertAvailable(grant)
    if (signal?.aborted) throw new LocalProviderRelayError('cancelled')
    if (request.model !== grant.policy.model) throw new LocalProviderRelayError('route_mismatch')
    if (grant.requestsUsed >= grant.policy.maxRequests) throw new LocalProviderRelayError('request_limit')
    if (grant.active.size >= grant.policy.maxConcurrent) throw new LocalProviderRelayError('concurrency_limit')
    // Defense in depth for typed host callers; the transport must also decode
    // and validate untrusted request structure before reaching this boundary.
    if (!Array.isArray(request.messages) || !request.messages.length || request.extraBody !== undefined ||
      (request.maxTokens !== undefined && (!Number.isSafeInteger(request.maxTokens) || request.maxTokens < 1 || (grant.policy.maxOutputTokens !== null && request.maxTokens > grant.policy.maxOutputTokens)))) {
      throw new LocalProviderRelayError('invalid_request')
    }
    // Network images can make an adapter fetch from the LOCAL network. Until
    // that operation has a separate network policy, accept inline data only.
    for (const message of request.messages) if (Array.isArray(message.content)) {
      for (const part of message.content) if (part.type === 'image_url' && !part.image_url.url.startsWith('data:image/')) throw new LocalProviderRelayError('invalid_request')
    }
    let size: number
    try { size = Buffer.byteLength(JSON.stringify(request)) } catch { throw new LocalProviderRelayError('invalid_request') }
    if (size > 16 * 1024 * 1024) throw new LocalProviderRelayError('invalid_request')
    const controller = new AbortController()
    const cancel = () => controller.abort(new LocalProviderRelayError('cancelled'))
    signal?.addEventListener('abort', cancel, { once: true })
    grant.active.add(controller)
    grant.requestsUsed++
    try {
      if (signal?.aborted) cancel()
      this.assertAvailable(grant)
      controller.signal.throwIfAborted()
      const route = grant.resolveClient()
      if (route.routeIdentity !== grant.routeIdentity) throw new LocalProviderRelayError('route_changed')
      const sessionId = `relay:${grant.cacheScope}:${createHash('sha256').update(request.sessionId ?? 'main').digest('hex')}`
      const defaults = route.defaults
      const temperature = request.temperature ?? defaults?.temperature
      const topK = request.topK ?? defaults?.topK
      const topP = request.topP ?? defaults?.topP
      const serviceTier = request.serviceTier ?? defaults?.serviceTier
      const thinking = request.thinking ?? defaults?.thinking
      const maxTokens = request.maxTokens ?? (grant.policy.maxOutputTokens === null ? defaults?.maxTokens
        : Math.min(defaults?.maxTokens ?? grant.policy.maxOutputTokens, grant.policy.maxOutputTokens))
      // Only sampling defaults are allowed here. In particular, this is not
      // an extraBody/config merge, and cannot introduce headers or endpoints.
      const stream = route.client.stream({ ...request, sessionId, model: grant.policy.model,
        ...(maxTokens === undefined ? {} : { maxTokens }),
        ...(grant.policy.maxOutputTokens === null ? {} : { [OUTPUT_TOKEN_LIMIT]: grant.policy.maxOutputTokens }),
        ...(temperature === undefined ? {} : { temperature }),
        ...(topK === undefined ? {} : { topK }),
        ...(topP === undefined ? {} : { topP }),
        ...(serviceTier === undefined ? {} : { serviceTier }),
        ...(thinking === undefined ? {} : { thinking }),
      }, controller.signal)
      for await (const delta of stream) {
        this.assertAvailable(grant)
        controller.signal.throwIfAborted()
        yield delta
      }
      this.assertAvailable(grant)
      controller.signal.throwIfAborted()
    } catch (error) {
      // A provider's exception can contain bearer headers or account data.
      // Emit only our own fixed failure vocabulary, never arbitrary messages.
      if (grant.stopped) throw new LocalProviderRelayError(grant.stopped === 'expired' ? 'grant_expired' : 'grant_revoked')
      if (controller.signal.aborted) throw new LocalProviderRelayError('cancelled')
      if (error instanceof LocalProviderRelayError) throw new LocalProviderRelayError(error.code)
      if (error instanceof OutputTokenLimitError) throw new LocalProviderRelayError('output_limit')
      if (classifyError(error).kind === ErrorKind.CONTEXT_OVERFLOW) throw new LocalProviderRelayError('context_overflow')
      throw new LocalProviderRelayError('provider_failed')
    } finally {
      controller.abort(new LocalProviderRelayError('cancelled'))
      signal?.removeEventListener('abort', cancel)
      grant.active.delete(controller)
    }
  }

  private owned(peer: RelayPeer, token: string): GrantState {
    const grant = this.grants.get(token)
    if (!grant || grant.peer !== peer) throw new LocalProviderRelayError('grant_unavailable')
    return grant
  }
  private assertAvailable(grant: GrantState): void {
    if (!grant.stopped && this.now() >= grant.policy.expiresAt) this.stop(grant, 'expired')
    if (grant.stopped) throw new LocalProviderRelayError(grant.stopped === 'expired' ? 'grant_expired' : 'grant_revoked')
  }
  private stop(grant: GrantState, reason: 'expired' | 'revoked'): void {
    if (grant.stopped) return
    grant.stopped = reason
    clearTimeout(grant.timer)
    for (const controller of grant.active) controller.abort(new LocalProviderRelayError(reason === 'expired' ? 'grant_expired' : 'grant_revoked'))
  }
  private view(grant: GrantState): LocalProviderGrantView {
    if (!grant.stopped && this.now() >= grant.policy.expiresAt) this.stop(grant, 'expired')
    return { ...grant.policy, destination: grant.destination, workspace: grant.workspace,
      requestsUsed: grant.requestsUsed, activeRequests: grant.active.size,
      status: grant.stopped ?? (grant.requestsUsed >= grant.policy.maxRequests ? 'exhausted' : 'active'),
      execution: 'local provider; remote tools', persistence: 'memory only',
      outputLimitMode: grant.policy.maxOutputTokens === null ? 'provider-controlled' : 'request-bound' }
  }
}
