// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { GatewayClient } from '../gatewayClient.js'
import { createLocalProviderBroker, requestLocalProviderBroker, type LocalProviderBroker } from './localProviderBroker.js'
import type { PreparedRemoteWorkspace, RemoteWorkspacePreparation } from './machineHandoff.js'

type Rpc = (method: string, params: Record<string, unknown>) => Promise<unknown>
export interface ShareableLocalProfile {
  readonly name: string
  readonly model: string
  readonly credentialSource: string
  readonly supported: boolean
  readonly providerControlledOutput: boolean
  readonly setup: string
}
export interface RemoteTaskReview {
  readonly destination: string
  readonly workspace: string
  readonly sessionId: string
  readonly remoteModel: string
  readonly remoteProfile: string
  readonly localRequirement: string
  readonly running: boolean
  readonly localBindingSupported: boolean
  readonly localBundleSupported?: boolean
  readonly profiles: readonly ShareableLocalProfile[]
  readonly inventoryError: string
}
export type RemoteTaskDecision = { readonly kind: 'remote' } | {
  readonly kind: 'local'
  readonly profile: string
  readonly profiles?: readonly string[]
  readonly durationMinutes: number
  readonly maxRequests: number
  readonly maxOutputTokens: number | null
  readonly maxConcurrent: number
  readonly consentProviderControlledOutput: boolean
}
const record = (v: unknown): Record<string, unknown> => v && typeof v === 'object' && !Array.isArray(v) ? v as Record<string, unknown> : {}
const label = (v: unknown, maximum = 512): string => typeof v === 'string' && v.length <= maximum && !/[\x00-\x1f\x7f]/.test(v) ? v : ''
const failure = () => new Error('Remote task setup failed. Review the connection and selected provider before retrying.')

export function shareableLocalProfiles(value: unknown): readonly ShareableLocalProfile[] {
  const result = record(value)
  if (result.ok !== true || !Array.isArray(result.profiles) || result.profiles.length > 1000) throw failure()
  return Object.freeze(result.profiles.map(raw => {
    const p = record(raw)
    const name = label(p.name), model = label(p.model)
    if (!name || !model) throw failure()
    return Object.freeze({name, model, credentialSource: label(p.credential_source) || 'local provider configuration',
      supported: p.supported === true, providerControlledOutput: p.output_limit_mode === 'provider-controlled', setup: label(p.setup)})
  }))
}

/** Prepare a task before asking for authority. The parent owns both connections;
 * the child renderer gets only the remote session ID/key, never the local grant or
 * broker path. Closing this handoff or losing either transport revokes access. */
export async function prepareRemoteTask(
  remote: RemoteWorkspacePreparation,
  localRpc: Rpc,
  review: (value: RemoteTaskReview, signal: AbortSignal) => Promise<RemoteTaskDecision>,
): Promise<PreparedRemoteWorkspace> {
  remote.signal.throwIfAborted()
  const lifetime = new AbortController()
  const signal = AbortSignal.any([remote.signal, lifetime.signal])
  const brokers: LocalProviderBroker[] = []
  const routes = new Map<string,LocalProviderBroker>()
  let closing: Promise<void> | undefined
  const gateway = new GatewayClient({externalSocketPath: remote.socketPath, projectDir: remote.projectDir,
    providerRelay: (id, frame, abort) => routes.has(id)
      ? requestLocalProviderBroker(routes.get(id)!.path, frame, abort)
      : Promise.resolve({error: 'grant_unavailable'}),
  })
  const close = (): Promise<void> => {
    if (closing) return closing
    closing = Promise.resolve().then(async () => {
      lifetime.abort()
      gateway.close()
      await Promise.all(brokers.map(broker => broker.close()))
      routes.clear()
    })
    return closing
  }
  const abort = () => { void close().catch(() => {}) }
  signal.addEventListener('abort', abort, {once: true})
  gateway.on('event', event => { if (event.type === 'gateway.closed') abort() })
  try {
    await gateway.start()
    signal.throwIfAborted()
    const session = record(await gateway.request(remote.resumeSessionId ? 'session.resume' : 'session.create', remote.resumeSessionId ? {session_id: remote.resumeSessionId, history_limit: 0} : {}))
    const sessionId = label(session.session_id)
    if (!/^[a-zA-Z0-9_-]{1,128}$/.test(sessionId) || (remote.resumeSessionId && sessionId !== remote.resumeSessionId)) throw failure()
    // Use the authoritative current session after initialization. Never bind a
    // running task, and never infer capability support from a version string.
    const status = record(await gateway.request('session.status', {history_limit: 0, structured: true}))
    const current = record(status.session)
    const info = record(session.info)
    const sessionKey = label(current.key)
    if (current.cwd !== remote.projectDir || !sessionKey) throw failure()
    let profiles: readonly ShareableLocalProfile[] = [], inventoryError = ''
    try { profiles = shareableLocalProfiles(await localRpc('provider.relay.inventory', {})) }
    catch { inventoryError = 'Local provider inventory unavailable. Configure a local profile with /provider, or use remote setup.' }
    signal.throwIfAborted()
    const input: RemoteTaskReview = Object.freeze({destination: remote.machine.target, workspace: remote.projectDir, sessionId,
      remoteModel: label(current.model), remoteProfile: label(current.profile_name), localRequirement: label(current.local_provider_label, 600), running: current.status !== 'idle',
      localBindingSupported: info.remote_provider_binding_supported === true, localBundleSupported: info.remote_provider_bundle_supported === true, profiles, inventoryError})
    const decision = await review(input, signal)
    signal.throwIfAborted()
    if (decision.kind === 'local') {
      const names = decision.profiles ?? [decision.profile]
      if (!Array.isArray(names) || !names.length || names.length > 32 || new Set(names).size !== names.length || !names.includes(decision.profile) || (names.length > 1 && !input.localBundleSupported)) throw failure()
      const selected = [decision.profile, ...names.filter(name => name !== decision.profile)].map(name => profiles.find(p => p.name === name))
      if (!input.localBindingSupported || input.running || selected.some(p => !p?.supported) ||
        !Number.isSafeInteger(decision.durationMinutes) || decision.durationMinutes < 1 || decision.durationMinutes > 480 ||
        !Number.isSafeInteger(decision.maxRequests) || decision.maxRequests < 1 || decision.maxRequests > 10000 ||
        !Number.isSafeInteger(decision.maxConcurrent) || decision.maxConcurrent < 1 || decision.maxConcurrent > 16 ||
        selected.some(profile => profile!.providerControlledOutput ? !decision.consentProviderControlledOutput
          : !Number.isSafeInteger(decision.maxOutputTokens) || decision.maxOutputTokens === null || decision.maxOutputTokens < 1 || decision.maxOutputTokens > 1_000_000)) throw failure()
      const expiresAt = Date.now() + decision.durationMinutes * 60_000
      const selections = []
      for (const profile of selected) {
        signal.throwIfAborted()
        const broker = await createLocalProviderBroker(localRpc, {consent:true,destination:input.destination,workspace:input.workspace,
          profile:profile!.name,model:profile!.model,expires_at:expiresAt,max_requests:decision.maxRequests,
          max_output_tokens:profile!.providerControlledOutput ? null : decision.maxOutputTokens,max_concurrent:decision.maxConcurrent,
          ...(profile!.providerControlledOutput ? {consent_provider_controlled_output:true} : {}),
        },signal)
        brokers.push(broker)
        selections.push({source:'local workstation (this SSH window)',profile:profile!.name,model:profile!.model,...(broker.capabilities ? {capabilities:broker.capabilities} : {})})
      }
      signal.throwIfAborted()
      const bound = record(await gateway.request('provider.remote.bind', {consent:true,...selections[0],...(selections.length > 1 ? {alternatives:selections.slice(1)} : {})}))
      const marker = record(bound.binding)
      const markers = [marker, ...(Array.isArray(marker.alternatives) ? marker.alternatives.map(record) : [])]
      if (bound.ok !== true || markers.length !== selected.length) throw failure()
      for (const [index, current] of markers.entries()) {
        if (current.workspace !== input.workspace || current.model !== selected[index]!.model || current.profile !== selected[index]!.name ||
          typeof current.binding !== 'string' || !/^[a-f0-9]{32}$/.test(current.binding) || routes.has(current.binding)) throw failure()
        routes.set(current.binding, brokers[index]!)
      }
    }
    signal.throwIfAborted()
    return {sessionId, sessionKey, close}
  } catch {
    await close()
    throw failure()
  }
}
