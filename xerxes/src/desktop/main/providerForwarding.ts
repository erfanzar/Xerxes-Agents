// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { randomUUID } from 'node:crypto'
import { createLocalProviderBroker, requestLocalProviderBroker, type LocalProviderBroker } from '../../ui/lib/localProviderBroker.js'
import { shareableLocalProfiles, type ShareableLocalProfile } from '../../protocol/shareableLocalProfiles.js'

type Rpc = (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>
const record = (value: unknown): Record<string, unknown> => value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
interface Review { id: string; sessionKey: string; profiles: readonly ShareableLocalProfile[] }
interface Access { brokers: LocalProviderBroker[]; bindings: string[]; expiresAt: number; profiles: string[] }
const working = (session: Record<string, unknown>) => session.status !== 'idle' || Boolean(session.active_turn_id) ||
  Array.isArray(session.subagent_snapshots) && session.subagent_snapshots.some(raw => ['working','running','starting','waiting'].includes(String(record(raw).status)))

/** A desktop surface owns authority, not the renderer or the remote filesystem.
 * Selection is reviewed for one session. No consent survives transport loss. */
export class DesktopProviderForwarding {
  private readonly routes = new Map<string, LocalProviderBroker>()
  private readonly sessions = new Map<string, Access>()
  private review: Review | undefined
  private lifetime = new AbortController()
  private busy = false
  constructor(private readonly local: Rpc, private readonly remote: Rpc,
    private readonly destination: string, private readonly workspace: string) {}

  async inspect() {
    const response = await this.remote('session.status', { history_limit: 0 })
    const session = record(response.session)
    if (response.provider_binding_session_guard_supported !== true) throw new Error('Update the SSH runtime when idle to enable local provider access, then reconnect.')
    if (response.ok !== true || typeof session.key !== 'string' || session.cwd !== this.workspace) throw new Error('Open the SSH conversation before choosing its provider source.')
    const profiles = shareableLocalProfiles(await this.local('provider.relay.inventory', {}))
    const review = {id: randomUUID(), sessionKey: session.key, profiles}
    this.review = review
    const access = this.sessions.get(session.key)
    const running = working(session) || response.provider_binding_busy === true
    return {ok:true, review:review.id, destination:this.destination, workspace:this.workspace, sessionKey:session.key,
      profiles, running, remoteProfile:session.profile_name, model:session.model,
      source:session.local_provider_label ? access && Date.now() < access.expiresAt ? 'local' : 'local-unavailable' : 'remote',
      expiresAt:access?.expiresAt, sharedProfiles:access?.profiles ?? []}
  }

  async authorize(params: Record<string, unknown>) {
    if (this.busy) throw new Error('Provider setup is already in progress.')
    const review = this.review
    if (!review || params.review !== review.id || params.consent !== true) throw new Error('Review this conversation’s provider source again before sharing.')
    const names = params.profiles
    if (!Array.isArray(names) || !names.length || names.length > 32 || new Set(names).size !== names.length || !names.includes(params.profile)) throw new Error('Choose a primary profile and up to 32 local profiles.')
    const profiles = [params.profile, ...names.filter(name => name !== params.profile)].map(name => review.profiles.find(profile => profile.name === name))
    if (profiles.some(profile => !profile?.supported)) throw new Error('A selected provider cannot be forwarded. Review its setup guidance.')
    if (profiles.some(profile => profile!.providerControlledOutput) && params.providerControlledOutput !== true) throw new Error('Acknowledge provider-controlled output for the selected subscription providers.')
    const minutes = params.minutes
    if (!Number.isSafeInteger(minutes) || (minutes as number) < 1 || (minutes as number) > 480) throw new Error('Choose an access duration from 1 to 480 minutes.')
    this.busy = true
    this.review = undefined
    const signal = this.lifetime.signal
    const brokers: LocalProviderBroker[] = []
    try {
      const status = await this.remote('session.status', {history_limit:0})
      const current = record(status.session)
      if (current.key !== review.sessionKey || current.cwd !== this.workspace || working(current) || status.provider_binding_busy === true) throw new Error('The conversation changed or is working. Review setup when it is idle.')
      if (!this.sessions.has(review.sessionKey) && this.sessions.size >= 128) throw new Error('This window already shares providers with 128 tasks. Revoke unused access first.')
      const fresh = shareableLocalProfiles(await this.local('provider.relay.inventory', {}))
      if (profiles.some(profile => !fresh.some(value => JSON.stringify(value) === JSON.stringify(profile)))) throw new Error('Local provider configuration changed. Review it again.')
      const expiresAt = Date.now() + (minutes as number) * 60_000
      for (const profile of profiles) {
        signal.throwIfAborted()
        brokers.push(await createLocalProviderBroker(this.local, {consent:true, destination:this.destination, workspace:this.workspace,
          profile:profile!.name, model:profile!.model, expires_at:expiresAt, max_requests:10000, max_concurrent:16,
          max_output_tokens:profile!.providerControlledOutput ? null : 32768,
          ...(profile!.providerControlledOutput ? {consent_provider_controlled_output:true} : {})}, signal))
      }
      signal.throwIfAborted()
      const choices = profiles.map((profile,index) => ({source:'local workstation (this desktop window)', profile:profile!.name, model:profile!.model,
        ...(brokers[index]!.capabilities ? {capabilities:brokers[index]!.capabilities} : {})}))
      const bound = await this.remote('provider.remote.bind', {consent:true, session_key:review.sessionKey, ...choices[0],
        ...(choices.length > 1 ? {alternatives:choices.slice(1)} : {})})
      const marker = record(bound.binding)
      const markers = [marker, ...(Array.isArray(marker.alternatives) ? marker.alternatives.map(record) : [])]
      if (bound.ok !== true || markers.length !== choices.length || markers.some((value,index) =>
        value.workspace !== this.workspace || value.profile !== choices[index]!.profile || value.model !== choices[index]!.model ||
        typeof value.binding !== 'string' || !/^[a-f0-9]{32}$/.test(value.binding))) throw new Error('The remote runtime could not bind local providers. Update it when idle, then review setup again.')
      signal.throwIfAborted()
      await this.revokeSession(review.sessionKey)
      signal.throwIfAborted()
      const bindings = markers.map(value => String(value.binding))
      bindings.forEach((binding,index) => this.routes.set(binding,brokers[index]!))
      this.sessions.set(review.sessionKey,{brokers,bindings,expiresAt,profiles:profiles.map(profile=>profile!.name)})
      return {ok:true}
    } catch (error) {
      await Promise.all(brokers.map(broker => broker.close()))
      throw error
    } finally { this.busy = false }
  }

  relay(binding: string, frame: Readonly<Record<string, unknown>>, signal: AbortSignal): Promise<unknown> {
    const broker = this.routes.get(binding)
    return broker ? requestLocalProviderBroker(broker.path,frame,signal) : Promise.resolve({error:'grant_unavailable'})
  }

  async revoke(sessionKey: unknown) {
    if (typeof sessionKey !== 'string') throw new Error('Choose a conversation before revoking access.')
    await this.revokeSession(sessionKey)
    return {ok:true}
  }
  private async revokeSession(key: string) {
    const access = this.sessions.get(key)
    if (!access) return
    this.sessions.delete(key)
    access.bindings.forEach(binding => this.routes.delete(binding))
    await Promise.all(access.brokers.map(broker => broker.close()))
  }
  /** Called on either transport loss and surface closure. No automatic reauthorization. */
  async disconnect() {
    this.review = undefined
    this.lifetime.abort()
    this.lifetime = new AbortController()
    await Promise.all([...this.sessions.keys()].map(key => this.revokeSession(key)))
  }
}
