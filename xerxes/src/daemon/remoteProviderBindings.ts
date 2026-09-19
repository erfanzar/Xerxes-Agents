// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomBytes } from 'node:crypto'
import { LocalRelayClient } from '../llms/localRelayClient.js'
import type { LlmClient } from '../llms/client.js'
import { LocalProviderRelayError } from '../security/localProviderRelay.js'
import { MAX_RELAY_DELTA_BYTES, type ProviderRelayRequest } from '../security/providerRelayProtocol.js'
import { parseLocalProviderCapabilities, type LocalProviderCapabilities } from '../protocol/localProviderCapabilities.js'

export const LOCAL_PROVIDER_BINDING = 'local_provider_binding'
export function localProviderLabel(metadata: Record<string, unknown>): string {
  if (!Object.hasOwn(metadata, LOCAL_PROVIDER_BINDING)) return ''
  const marker = metadata[LOCAL_PROVIDER_BINDING]
  const profile = marker && typeof marker === 'object' && !Array.isArray(marker) ? (marker as Record<string, unknown>).profile : undefined
  return typeof profile === 'string' && profile.length <= 512 && !/[\x00-\x1f\x7f]/.test(profile)
    ? 'Local provider: ' + profile + ' · requires local access' : 'Local provider required · review access'
}
interface Session { id: string; cwd: string; metadata: Record<string, unknown> }
export interface LocalProviderSelection { source: string; profile: string; model: string; capabilities?: LocalProviderCapabilities }
export function localProviderCapabilities(metadata: Record<string, unknown>, model: string): LocalProviderCapabilities | undefined {
  const marker = metadata[LOCAL_PROVIDER_BINDING]
  if (!marker || typeof marker !== 'object' || Array.isArray(marker) || !('model' in marker) || marker.model !== model) return undefined
  try { return parseLocalProviderCapabilities('capabilities' in marker ? marker.capabilities : undefined, model) }
  catch { return undefined } // Old or damaged metadata cannot authorize guessed controls.
}
interface Marker extends LocalProviderSelection { version: 1; binding: string; workspace: string }
export interface RemoteProviderRequest { binding: string; request_id: string; frame: ProviderRelayRequest }
interface Pending { settle(value: unknown): void; reject(error: Error): void }
interface Binding {
  owner: object
  marker: Marker
  pending: Map<string, Pending>
  streams: Set<string>
  send: (request: RemoteProviderRequest) => boolean
}

/** Remote session routing stores only a requirement and display metadata.
 * Authority and local credentials remain on the local host. Private requests
 * must bypass transcript/event journals and fail when their transport is gone.
 */
export class RemoteProviderBindings {
  private readonly bindings = new Map<string, Binding>()
  constructor(private readonly timeoutMs = 60_000) {}

  bind(owner: object, session: Session, selection: LocalProviderSelection,
    send: Binding['send']): Marker {
    if ([selection.source, selection.profile, selection.model].some(value => typeof value !== 'string' || !value.trim() || value.length > 512 || /[\x00-\x1f\x7f]/.test(value))) throw new LocalProviderRelayError('invalid_request')
    let capabilities: LocalProviderCapabilities | undefined
    try { capabilities = parseLocalProviderCapabilities(selection.capabilities, selection.model) }
    catch { throw new LocalProviderRelayError('invalid_request') }
    const old = this.bindings.get(session.id)
    if (old && old.owner !== owner) throw new LocalProviderRelayError('grant_unavailable')
    if (old?.pending.size) throw new LocalProviderRelayError('concurrency_limit')
    if (!old && this.bindings.size >= 128) throw new LocalProviderRelayError('concurrency_limit')
    const marker: Marker = { version: 1, binding: randomBytes(16).toString('hex'), workspace: session.cwd, source: selection.source, profile: selection.profile, model: selection.model, ...(capabilities ? {capabilities} : {}) }
    if (old) this.cancelStreams(old)
    this.bindings.set(session.id, { owner, marker, pending: new Map(), streams: new Set(), send })
    session.metadata[LOCAL_PROVIDER_BINDING] = { ...marker }
    return { ...marker }
  }

  /** Undefined means this session never chose a local provider. A persisted
   * but disconnected marker is an error, never permission to use a fallback. */
  client(session: Session, model: string): LlmClient | undefined {
    if (!Object.hasOwn(session.metadata, LOCAL_PROVIDER_BINDING)) return undefined
    const binding = this.bindings.get(session.id)
    const marker = session.metadata[LOCAL_PROVIDER_BINDING]
    if (!binding || !marker || typeof marker !== 'object' || Array.isArray(marker) ||
      (marker as Record<string, unknown>).binding !== binding.marker.binding || session.cwd !== binding.marker.workspace) throw new LocalProviderRelayError('grant_unavailable')
    if (model !== binding.marker.model) throw new LocalProviderRelayError('route_mismatch')
    return new LocalRelayClient((frame, signal) => {
      if (this.bindings.get(session.id) !== binding) return Promise.reject(new LocalProviderRelayError('grant_unavailable'))
      return this.call(binding, frame, signal)
    })
  }

  /** Child snapshots keep only a fingerprint. Every execution still needs live authority. */
  sourceClient(session: Session, model: string, explicitProfile?: string): { llm: LlmClient; route: string } | undefined {
    const llm = this.client(session, model)
    if (!llm) return undefined
    // An agent definition cannot silently replace a user's local binding.
    if (explicitProfile) throw new LocalProviderRelayError('route_mismatch')
    const binding = this.bindings.get(session.id)!
    const route = createHash('sha256').update(JSON.stringify(binding.marker)).digest('hex')
    return { llm, route }
  }

  reply(owner: object, bindingId: unknown, requestId: unknown, value: unknown): void {
    const binding = [...this.bindings.values()].find(item => item.marker.binding === bindingId && item.owner === owner)
    const pending = typeof requestId === 'string' ? binding?.pending.get(requestId) : undefined
    if (!pending) throw new LocalProviderRelayError('grant_unavailable')
    try {
      if (Buffer.byteLength(JSON.stringify(value)) > MAX_RELAY_DELTA_BYTES + 4096) throw new Error('oversized')
    } catch { pending.reject(new LocalProviderRelayError('provider_failed')); return }
    // LocalRelayClient validates delta/error shapes before exposing content.
    pending.settle(value)
  }

  /** Explicit user selection of a remote profile removes the local requirement.
   * Refuse to invalidate an active provider request. */
  useRemote(session: Session): void {
    const binding = this.bindings.get(session.id)
    if (binding && (binding.pending.size || binding.streams.size)) throw new LocalProviderRelayError('concurrency_limit')
    this.bindings.delete(session.id)
    delete session.metadata[LOCAL_PROVIDER_BINDING]
  }

  disconnect(owner: object): void {
    for (const [id, binding] of this.bindings) if (binding.owner === owner) {
      this.bindings.delete(id)
      this.cancelStreams(binding)
      for (const pending of [...binding.pending.values()]) pending.reject(new LocalProviderRelayError('grant_unavailable'))
    }
  }
  close(): void { for (const binding of [...this.bindings.values()]) this.disconnect(binding.owner) }

  private cancelStreams(binding: Binding): void {
    for (const id of binding.streams) {
      try { binding.send({ binding: binding.marker.binding, request_id: randomBytes(16).toString('hex'), frame: { op: 'cancel', id } }) }
      catch { /* Closed transports are also bounded by local endpoint expiry. */ }
    }
    binding.streams.clear()
  }

  private call(binding: Binding, frame: ProviderRelayRequest, signal?: AbortSignal): Promise<unknown> {
    if (signal?.aborted) return Promise.reject(new LocalProviderRelayError('cancelled'))
    if (binding.pending.size >= 16) return Promise.reject(new LocalProviderRelayError('concurrency_limit'))
    if (frame.op === 'next' && !binding.streams.has(frame.id) && binding.streams.size >= 16) return Promise.reject(new LocalProviderRelayError('concurrency_limit'))
    if (frame.op === 'cancel') binding.streams.delete(frame.id)
    else binding.streams.add(frame.id)
    const id = randomBytes(16).toString('hex')
    return new Promise((resolve, reject) => {
      const finish = (value: unknown, error?: Error) => {
        if (!binding.pending.delete(id)) return
        clearTimeout(timer); signal?.removeEventListener('abort', cancel)
        if (value && typeof value === 'object' && ('error' in value || ('done' in value && value.done === true))) binding.streams.delete(frame.id)
        if (error) reject(error); else resolve(value)
      }
      const cancel = () => finish(undefined, new LocalProviderRelayError('cancelled'))
      const timer = setTimeout(() => finish(undefined, new LocalProviderRelayError('grant_unavailable')), this.timeoutMs)
      timer.unref?.()
      binding.pending.set(id, { settle: value => finish(value), reject: error => finish(undefined, error) })
      signal?.addEventListener('abort', cancel, { once: true })
      if (signal?.aborted) { cancel(); return }
      try {
        if (!binding.send({ binding: binding.marker.binding, request_id: id, frame })) finish(undefined, new LocalProviderRelayError('grant_unavailable'))
      } catch { finish(undefined, new LocalProviderRelayError('grant_unavailable')) }
    })
  }
}
