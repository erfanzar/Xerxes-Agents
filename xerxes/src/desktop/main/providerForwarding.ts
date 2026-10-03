// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { randomUUID } from 'node:crypto'
import { createLocalProviderBroker, requestLocalProviderBroker, type LocalProviderBroker } from '../../ui/lib/localProviderBroker.js'
import { shareableLocalProfiles, type ShareableLocalProfile } from '../../protocol/shareableLocalProfiles.js'

type Rpc = (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>
const record = (value: unknown): Record<string, unknown> => value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
interface Review { id: string; sessionKey: string; profiles: readonly ShareableLocalProfile[] }
interface Access { brokers: LocalProviderBroker[]; bindings: string[]; expiresAt: number; profiles: string[] }
/** What the local daemon reports for a profile with no saved key: a sign-in or environment credential. */
const SIGN_IN_CREDENTIAL = 'local provider authentication or environment'
/** Following access is renewed this long before it would expire. */
const RENEW_BEFORE_MS = 10 * 60_000
const FOLLOW_MINUTES = 480
/**
 * What has gone through this computer for one SSH conversation, so the person
 * can see that its prompts run here rather than on the host. Counts outlive
 * the access (a revoked or expired grant still shows what it carried).
 */
export interface RelayActivity {
  readonly requests: number
  readonly inFlight: number
  readonly profile?: string
  readonly model?: string
  readonly lastAt?: number
  readonly lastError?: string
}
interface ActivityState { requests: number; inFlight: Set<string>; profile?: string; model?: string; lastAt?: number; lastError?: string }
const MAX_ACTIVITY = 128
export type FollowResult =
  | { readonly status: 'bound' }
  | { readonly status: 'skipped' | 'failed'; readonly reason: string }
/** The profile and model a session's saved local requirement names (restored exactly after a dropped link). */
const savedRoute = (session: Record<string, unknown>): { profile: string; model: string } | undefined => {
  const route = record(session.local_provider_route)
  return typeof route.profile === 'string' && typeof route.model === 'string' ? { profile: route.profile, model: route.model } : undefined
}
/** Retry delays while restoring a lost binding: quick at first, then every 15 s, for as long as the window is connected. */
const RESTORE_BACKOFF_MS = [1_000, 2_000, 4_000, 8_000, 15_000]
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
  private following: Promise<FollowResult> | undefined
  private restoreTimer: ReturnType<typeof setTimeout> | undefined
  private restoreAttempt = 0
  private restoreFailure: string | undefined
  private followAgain = false
  private renewal: ReturnType<typeof setTimeout> | undefined
  private readonly bindingOwners = new Map<string, { sessionKey: string; profile: string }>()
  /** Bindings the person revoked; their requests fail as revoked, not as a lapse to retry. */
  private readonly revoked = new Set<string>()
  private readonly activity = new Map<string, ActivityState>()
  /** The profile each conversation was put on; survives transport loss so renewal keeps it. */
  private readonly chosen = new Map<string, string>()
  private changed: (() => void) | undefined
  /** `preferred` names this computer's active profile; without it nothing is followed automatically. */
  constructor(private readonly local: Rpc, private readonly remote: Rpc,
    private readonly destination: string, private readonly workspace: string,
    private readonly preferred?: () => Promise<string | undefined>) {}

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
      turns: typeof session.turn_count === 'number' ? session.turn_count : undefined,
      // A grant held here does not help if the host's binding dropped with its link.
      source:session.local_provider_label ? access && Date.now() < access.expiresAt && response.provider_binding_lost !== true ? 'local' : 'local-unavailable' : 'remote',
      expiresAt:access?.expiresAt, sharedProfiles:access?.profiles ?? [],
      ...(savedRoute(session) ? { savedRoute: savedRoute(session)! } : {})}
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
      // Restoring the exact route a dropped link served is allowed mid-turn
      // (the runtime checks it is the same profile and model); anything else waits.
      const restoring = params.restore === true && status.provider_binding_lost === true
      if (current.key !== review.sessionKey || current.cwd !== this.workspace || (!restoring && (working(current) || status.provider_binding_busy === true))) throw new Error('The conversation changed or is working. Review setup when it is idle.')
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
      const bound = await this.remote('provider.remote.bind', {consent:true, restorable:true, session_key:review.sessionKey, ...choices[0],
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
      bindings.forEach((binding,index) => {
        this.routes.set(binding,brokers[index]!)
        this.bindingOwners.set(binding,{sessionKey:review.sessionKey, profile:profiles[index]!.name})
      })
      this.sessions.set(review.sessionKey,{brokers,bindings,expiresAt,profiles:profiles.map(profile=>profile!.name)})
      this.changed?.()
      return {ok:true}
    } catch (error) {
      await Promise.all(brokers.map(broker => broker.close()))
      throw error
    } finally { this.busy = false }
  }

  /**
   * Keep the open SSH conversation on this computer's sign-in provider with no
   * manual review. ChatGPT and Claude Code logins cannot be copied to a host:
   * they live in this computer's keychain or are replaced on every refresh, so
   * a copy would sign one machine out. Their requests run here instead, with
   * every other shareable profile offered as an alternative. Applies when this
   * computer's active profile is such a provider: to a new conversation, and to
   * one already following it after a reconnect or near expiry. A conversation
   * already run on the host's own providers is left as it is. Keyed profiles
   * are copied to the host on connect and need none of this.
   * Calls arriving while one runs are coalesced into one more pass.
   */
  follow(): Promise<FollowResult> {
    if (this.following) { this.followAgain = true; return this.following }
    this.following = this.followOnce().then(result => {
      // A conversation that should run here but could not be bound again keeps
      // being retried until it is; nothing waits for the person to notice.
      clearTimeout(this.restoreTimer)
      if (result.status === 'failed') {
        this.restoreFailure = result.reason
        const delay = RESTORE_BACKOFF_MS[Math.min(this.restoreAttempt, RESTORE_BACKOFF_MS.length - 1)]!
        this.restoreAttempt += 1
        this.restoreTimer = setTimeout(() => { void this.follow() }, delay)
        this.restoreTimer.unref?.()
      } else {
        this.restoreAttempt = 0
        this.restoreFailure = undefined
      }
      this.changed?.()
      return result
    }).finally(() => {
      this.following = undefined
      if (this.followAgain) { this.followAgain = false; void this.follow() }
    })
    return this.following
  }

  private async followOnce(): Promise<FollowResult> {
    if (!this.preferred) return { status: 'skipped', reason: 'automatic provider access is off' }
    try {
      const review = await this.inspect()
      if (review.source === 'local' && (review.expiresAt ?? 0) - Date.now() > RENEW_BEFORE_MS) {
        this.scheduleRenewal(review.expiresAt!)
        return { status: 'skipped', reason: 'already following' }
      }
      const followsHere = review.source !== 'remote'
      let primary: ShareableLocalProfile | undefined
      if (followsHere) {
        // Renewing or restoring: the route the session saved (exactly what its
        // dropped link served), else the profile it was put on here (a switch
        // to Claude Code stays Claude Code), else this computer's active one.
        const chosen = review.savedRoute?.profile ?? this.chosen.get(review.sessionKey) ?? await this.preferred()
        primary = review.profiles.find(profile => profile.name === chosen && profile.supported)
        if (!primary) return { status: 'skipped', reason: 'the profile this conversation used is no longer shareable from this computer' }
      } else {
        if (review.turns !== 0) return { status: 'skipped', reason: 'this conversation uses the host\'s providers' }
        const name = await this.preferred()
        primary = review.profiles.find(profile => profile.name === name)
        if (!primary || !primary.supported || primary.credentialSource !== SIGN_IN_CREDENTIAL) return { status: 'skipped', reason: 'this computer\'s active profile is copied to the host' }
      }
      // A working conversation whose link dropped is restored at once — its
      // turn is retrying meanwhile. Any other change waits for it to finish.
      const restore = followsHere && review.running
      if (review.running && !restore) return { status: 'skipped', reason: 'the conversation is working' }
      await this.bindPrimary(review, primary, restore)
      return { status: 'bound' }
    } catch (error) {
      return { status: 'failed', reason: error instanceof Error ? error.message : String(error) }
    }
  }

  /**
   * Put the open SSH conversation on one of this computer's profiles because
   * the person chose it (Settings → switch on ChatGPT/Codex or Claude Code).
   * Unlike follow(), this applies to a conversation that already ran on the
   * host's providers, and the choice is kept through renewals.
   */
  async useLocal(name: unknown): Promise<{ ok: true }> {
    if (typeof name !== 'string' || !name) throw new Error('Choose a provider profile to run on this computer.')
    const review = await this.inspect()
    const primary = review.profiles.find(profile => profile.name === name)
    if (!primary) throw new Error(`This computer has no ${name} profile to share. Set it up in a local workspace's Models & Providers settings.`)
    if (!primary.supported) throw new Error(primary.setup || `${name} cannot run through this computer.`)
    if (review.running) throw new Error('This task or its agents are working. Wait until they finish before changing its provider.')
    await this.bindPrimary(review, primary)
    return { ok: true }
  }

  private async bindPrimary(review: Awaited<ReturnType<DesktopProviderForwarding['inspect']>>, primary: ShareableLocalProfile, restore = false): Promise<void> {
    const profiles = [primary, ...review.profiles.filter(profile => profile.supported && profile.name !== primary.name)].slice(0, 32)
    await this.authorize({ consent: true, restore, review: review.review, profiles: profiles.map(profile => profile.name), profile: primary.name,
      minutes: FOLLOW_MINUTES, providerControlledOutput: profiles.some(profile => profile.providerControlledOutput) })
    this.chosen.set(review.sessionKey, primary.name)
    while (this.chosen.size > MAX_ACTIVITY) this.chosen.delete(this.chosen.keys().next().value!)
    this.scheduleRenewal(Date.now() + FOLLOW_MINUTES * 60_000)
  }

  private scheduleRenewal(expiresAt: number) {
    clearTimeout(this.renewal)
    this.renewal = setTimeout(() => { void this.follow() }, Math.max(1000, expiresAt - Date.now() - RENEW_BEFORE_MS + 1000))
    this.renewal.unref?.()
  }

  relay(binding: string, frame: Readonly<Record<string, unknown>>, signal: AbortSignal): Promise<unknown> {
    const broker = this.routes.get(binding)
    // Revoked on purpose: final. Unknown (its access lapsed with a dropped
    // link): unavailable, which the host retries while this window restores it.
    if (!broker) return Promise.resolve({error:this.revoked.has(binding) ? 'grant_revoked' : 'grant_unavailable'})
    const owner = this.bindingOwners.get(binding)
    const id = typeof frame.id === 'string' ? frame.id : ''
    const state = owner ? this.activityOf(owner.sessionKey) : undefined
    // A request starts with the frame that carries it; later pulls stream it.
    if (state && frame.op === 'next' && frame.request && typeof frame.request === 'object') {
      const request = frame.request as Record<string, unknown>
      state.requests += 1
      state.inFlight.add(id)
      state.lastAt = Date.now()
      state.profile = owner!.profile
      if (typeof request.model === 'string') state.model = request.model
      this.changed?.()
    }
    const settle = (error?: string) => {
      if (!state || !state.inFlight.delete(id)) return
      if (error) state.lastError = error
      this.changed?.()
    }
    if (frame.op === 'cancel') settle()
    return requestLocalProviderBroker(broker.path,frame,signal).then(reply => {
      const value = record(reply)
      if (typeof value.error === 'string') settle(value.error)
      else if (value.done === true) settle()
      return reply
    }, error => { settle(error instanceof Error ? error.message : String(error)); throw error })
  }

  /** Called whenever access or relayed traffic changes; one listener. */
  onChange(listener: () => void): void { this.changed = listener }

  /** Relayed traffic for one conversation, and whether its access is live now. */
  activityFor(sessionKey: string): RelayActivity & { readonly live: boolean; readonly expiresAt?: number; readonly profiles: readonly string[] } {
    const state = this.activity.get(sessionKey)
    const access = this.sessions.get(sessionKey)
    const live = Boolean(access && Date.now() < access.expiresAt)
    const profile = state?.profile ?? access?.profiles[0]
    return { live, profiles: access?.profiles ?? [], ...(access ? { expiresAt: access.expiresAt } : {}),
      requests: state?.requests ?? 0, inFlight: state?.inFlight.size ?? 0,
      ...(profile ? { profile } : {}),
      ...(state?.model ? { model: state.model } : {}), ...(state?.lastAt ? { lastAt: state.lastAt } : {}), ...(state?.lastError ? { lastError: state.lastError } : {}) }
  }

  /** The open conversation's routing: bound to this computer, live, and what it carried. */
  async current() {
    const response = await this.remote('session.status', { history_limit: 0 })
    const session = record(response.session)
    if (response.ok !== true || typeof session.key !== 'string') return { ok: true, bound: false }
    const bound = typeof session.local_provider_label === 'string' && session.local_provider_label !== ''
    const activity = this.activityFor(session.key)
    const route = savedRoute(session)
    return { ok: true, bound, sessionKey: session.key, destination: this.destination, ...activity,
      ...(!activity.profile && route ? { profile: route.profile, model: route.model } : {}),
      // Not live but bound: this window is restoring it and retries until it is back.
      ...(bound && !activity.live ? { reconnecting: true, ...(this.restoreFailure ? { lastFailure: this.restoreFailure } : {}) } : {}) }
  }

  private activityOf(sessionKey: string): ActivityState {
    let state = this.activity.get(sessionKey)
    if (!state) {
      state = { requests: 0, inFlight: new Set() }
      this.activity.set(sessionKey, state)
      while (this.activity.size > MAX_ACTIVITY) this.activity.delete(this.activity.keys().next().value!)
    }
    return state
  }

  async revoke(sessionKey: unknown) {
    if (typeof sessionKey !== 'string') throw new Error('Choose a conversation before revoking access.')
    for (const binding of this.sessions.get(sessionKey)?.bindings ?? []) this.revoked.add(binding)
    while (this.revoked.size > 1024) this.revoked.delete(this.revoked.values().next().value!)
    await this.revokeSession(sessionKey)
    return {ok:true}
  }
  private async revokeSession(key: string) {
    const access = this.sessions.get(key)
    if (!access) return
    this.sessions.delete(key)
    access.bindings.forEach(binding => { this.routes.delete(binding); this.bindingOwners.delete(binding) })
    this.activity.get(key)?.inFlight.clear()
    this.changed?.()
    await Promise.all(access.brokers.map(broker => broker.close()))
  }
  /** Called on either transport loss and surface closure. Only follow() authorizes again. */
  async disconnect() {
    clearTimeout(this.restoreTimer)
    clearTimeout(this.renewal)
    this.renewal = undefined
    this.review = undefined
    this.lifetime.abort()
    this.lifetime = new AbortController()
    await Promise.all([...this.sessions.keys()].map(key => this.revokeSession(key)))
  }
}
