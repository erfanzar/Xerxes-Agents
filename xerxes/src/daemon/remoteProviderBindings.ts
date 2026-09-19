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
  const profile = typeof metadata.local_provider_profile === 'string' ? metadata.local_provider_profile : marker && typeof marker === 'object' && !Array.isArray(marker) ? (marker as Record<string, unknown>).profile : undefined
  return typeof profile === 'string' && profile.length <= 512 && !/[\x00-\x1f\x7f]/.test(profile)
    ? 'Local provider: ' + profile + ' · requires local access' : 'Local provider required · review access'
}
interface Session { id: string; cwd: string; metadata: Record<string, unknown> }
export interface LocalProviderSelection { source: string; profile: string; model: string; capabilities?: LocalProviderCapabilities; alternatives?: readonly LocalProviderSelection[] }
export function localProviderCapabilities(metadata: Record<string, unknown>, model: string): LocalProviderCapabilities | undefined {
  const marker = localProviderSelections(metadata).find(route => route.model === model && (!metadata.local_provider_profile || route.profile === metadata.local_provider_profile))
  if (!marker) return undefined
  try { return parseLocalProviderCapabilities('capabilities' in marker ? marker.capabilities : undefined, model) }
  catch { return undefined } // Old or damaged metadata cannot authorize guessed controls.
}
interface Marker extends LocalProviderSelection { version: 1; binding: string; workspace: string; alternatives?: readonly Marker[] }
/** Display-only persisted choices; live authority is always checked separately. */
export function localProviderSelections(metadata: Record<string, unknown>): readonly LocalProviderSelection[] {
  const raw = metadata[LOCAL_PROVIDER_BINDING]
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return []
  const marker = raw as Record<string, unknown>
  const values = [marker, ...(Array.isArray(marker.alternatives) ? marker.alternatives.slice(0,31) : [])]
  return values.flatMap(value => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return []
    const r = value as Record<string, unknown>
    if (typeof r.source !== 'string' || typeof r.profile !== 'string' || typeof r.model !== 'string') return []
    try { const capabilities = parseLocalProviderCapabilities(r.capabilities, r.model)
      return [{source:r.source, profile:r.profile, model:r.model, ...(capabilities ? {capabilities} : {})}]
    } catch { return [] }
  })
}
export interface RemoteProviderRequest { binding: string; request_id: string; frame: ProviderRelayRequest }
interface Pending { settle(value: unknown): void; reject(error: Error): void }
interface Binding {
  alternatives?: Binding[]
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
    if (selection.alternatives !== undefined && (!Array.isArray(selection.alternatives) || selection.alternatives.length > 31)) throw new LocalProviderRelayError('invalid_request')
    const choices = [selection, ...(selection.alternatives ?? [])]
    const markers = choices.map((choice,index): Marker => {
      if (!choice || (index > 0 && choice.alternatives !== undefined) || [choice.source, choice.profile, choice.model].some(value => typeof value !== 'string' || !value.trim() || value.length > 512 || /[\x00-\x1f\x7f]/.test(value))) throw new LocalProviderRelayError('invalid_request')
      let capabilities: LocalProviderCapabilities | undefined
      try { capabilities = parseLocalProviderCapabilities(choice.capabilities, choice.model) }
      catch { throw new LocalProviderRelayError('invalid_request') }
      return {version:1, binding:randomBytes(16).toString('hex'),workspace:session.cwd,source:choice.source,profile:choice.profile,model:choice.model,...(capabilities ? {capabilities} : {})}
    })
    if (new Set(markers.map(m => JSON.stringify([m.profile,m.model]))).size !== markers.length) throw new LocalProviderRelayError('invalid_request')
    const old = this.bindings.get(session.id)
    if (old && old.owner !== owner) throw new LocalProviderRelayError('grant_unavailable')
    if (old && this.group(old).some(route => route.pending.size)) throw new LocalProviderRelayError('concurrency_limit')
    if (!old && this.bindings.size >= 128) throw new LocalProviderRelayError('concurrency_limit')
    const routes = markers.map(marker => ({owner,marker,pending:new Map<string,Pending>(),streams:new Set<string>(),send}))
    const primary: Binding = routes[0]!
    if (routes.length > 1) { primary.alternatives = routes.slice(1); primary.marker.alternatives = markers.slice(1) }
    if (old) for (const route of this.group(old)) this.cancelStreams(route)
    this.bindings.set(session.id, primary)
    session.metadata[LOCAL_PROVIDER_BINDING] = structuredClone(primary.marker)
    session.metadata.local_provider_profile = primary.marker.profile
    return structuredClone(primary.marker)
  }

  /** Undefined means this session never chose a local provider. A persisted
   * but disconnected marker is an error, never permission to use a fallback. */
  client(session: Session, model: string, explicitProfile?: string): LlmClient | undefined {
    if (!Object.hasOwn(session.metadata, LOCAL_PROVIDER_BINDING)) return undefined
    const binding = this.bindings.get(session.id)
    const marker = session.metadata[LOCAL_PROVIDER_BINDING]
    if (!binding || !marker || typeof marker !== 'object' || Array.isArray(marker) ||
      (marker as Record<string, unknown>).binding !== binding.marker.binding || session.cwd !== binding.marker.workspace) throw new LocalProviderRelayError('grant_unavailable')
    const candidates = this.group(binding).filter(route => route.marker.model === model)
    const preferred = explicitProfile ?? (typeof session.metadata.local_provider_profile === 'string' ? session.metadata.local_provider_profile : binding.marker.profile)
    const selected = candidates.find(route => route.marker.profile === preferred) ?? (!explicitProfile && candidates.length === 1 ? candidates[0] : undefined)
    if (!selected) throw new LocalProviderRelayError('route_mismatch')
    return new LocalRelayClient((frame, signal) => {
      if (this.bindings.get(session.id) !== binding) return Promise.reject(new LocalProviderRelayError('grant_unavailable'))
      return this.call(selected, frame, signal)
    })
  }

  /** Child snapshots keep only a fingerprint. Every execution still needs live authority. */
  sourceClient(session: Session, model: string, explicitProfile?: string): { llm: LlmClient; route: string; profile: string } | undefined {
    const llm = this.client(session, model, explicitProfile)
    if (!llm) return undefined
    const root = this.bindings.get(session.id)!
    const candidates = this.group(root).filter(route => route.marker.model === model)
    const profile = explicitProfile ?? (typeof session.metadata.local_provider_profile === 'string' ? session.metadata.local_provider_profile : root.marker.profile)
    const selected = candidates.find(route => route.marker.profile === profile) ?? candidates[0]!
    const route = createHash('sha256').update(JSON.stringify(selected.marker)).digest('hex')
    return { llm, route, profile:selected.marker.profile }
  }

  reply(owner: object, bindingId: unknown, requestId: unknown, value: unknown): void {
    const binding = [...this.bindings.values()].flatMap(item => this.group(item)).find(item => item.marker.binding === bindingId && item.owner === owner)
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
    if (binding && this.group(binding).some(route => route.pending.size || route.streams.size)) throw new LocalProviderRelayError('concurrency_limit')
    this.bindings.delete(session.id)
    delete session.metadata[LOCAL_PROVIDER_BINDING]
    delete session.metadata.local_provider_profile
  }

  disconnect(owner: object): void {
    for (const [id, binding] of this.bindings) if (binding.owner === owner) {
      this.bindings.delete(id)
      for (const route of this.group(binding)) {
        this.cancelStreams(route)
        for (const pending of [...route.pending.values()]) pending.reject(new LocalProviderRelayError('grant_unavailable'))
      }
    }
  }
  close(): void { for (const binding of [...this.bindings.values()]) this.disconnect(binding.owner) }

  private group(binding: Binding): Binding[] { return [binding, ...(binding.alternatives ?? [])] }

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
