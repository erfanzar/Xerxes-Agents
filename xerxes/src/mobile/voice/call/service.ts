// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { VoiceModelTier } from '../../../client/voiceModelManifest.js'
import type { CallEngine, CallSnapshot } from './engine.js'
import { chooseCallTier, type CallModelsStatus } from './logic.js'

export type CallEngineControl = Pick<CallEngine, 'start' | 'hangup' | 'mute' | 'interrupt' | 'visibility' | 'setDebugCaptureEnabled' | 'getSnapshot' | 'subscribe'>
  & Partial<Pick<CallEngine, 'setConnected'>>
export interface VoiceCallSnapshot {
  readonly view: 'closed' | 'voice' | 'chat'
  readonly call: CallSnapshot
  readonly tier: VoiceModelTier
  readonly connected: boolean
  readonly checking: boolean
  readonly models?: CallModelsStatus
  readonly error?: string
  readonly downloadNoticeSeen: boolean
  readonly debugCaptureEnabled: boolean
  /** Bytes of the selected tier already in this browser's caches (display-only). */
  readonly phoneCache?: { readonly cachedBytes: number; readonly totalBytes: number }
  readonly storagePersisted?: 'granted' | 'denied' | 'unsupported'
}
export interface VoiceCallServiceDeps {
  readonly engine: () => CallEngineControl
  readonly modelsStatus: (signal: AbortSignal) => Promise<CallModelsStatus>
  readonly canStart: () => boolean
  readonly beforeStart?: () => void
  readonly deviceMemory?: number
  readonly noticeSeen?: () => boolean
  readonly rememberNotice?: () => void
  readonly debugEnabled?: () => boolean
  readonly rememberDebugEnabled?: (enabled: boolean) => void
  /** Reports how much of a tier is already cached on the phone. */
  readonly phoneCache?: (tier: VoiceModelTier) => Promise<{ readonly cachedBytes: number; readonly totalBytes: number }>
  /** Asks the browser not to evict cached models (navigator.storage.persist). */
  readonly persistStorage?: () => Promise<boolean>
}
export const isCallActive = (call: CallSnapshot): boolean => call.phase !== 'idle' && call.phase !== 'error'

/** App-owned call session. View/navigation state never owns the microphone or model lifetime. */
export class VoiceCallService {
  private engine: CallEngineControl | undefined
  private stopObserving: (() => void) | undefined
  private check: AbortController | undefined
  private snapshot: VoiceCallSnapshot
  private readonly observers = new Set<() => void>()
  private persistRequested = false
  private cacheProbe = 0
  constructor(private readonly deps: VoiceCallServiceDeps) {
    const tier = chooseCallTier(deps.deviceMemory)
    this.snapshot = { view: 'closed', tier, connected: true, checking: false, downloadNoticeSeen: deps.noticeSeen?.() ?? false,
      debugCaptureEnabled: deps.debugEnabled?.() ?? false, call: { phase: 'idle', tier, muted: false, heard: '', reply: '' } }
  }
  readonly getSnapshot = (): VoiceCallSnapshot => this.snapshot
  readonly subscribe = (observer: () => void): (() => void) => { this.observers.add(observer); return () => { this.observers.delete(observer) } }
  private update(patch: Partial<VoiceCallSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch }
    for (const observer of this.observers) observer()
  }
  open(): void {
    this.update({ view: 'voice' })
    if (!this.snapshot.models && !this.snapshot.checking) void this.loadStatus()
  }
  /** Chat/Escape only hides the voice screen. The engine, microphone, models and observers are untouched. */
  minimize(): void { this.update({ view: isCallActive(this.snapshot.call) ? 'chat' : 'closed' }) }
  /** A dropped Hub link pauses the call (the engine holds what was said); it never ends it. */
  setConnected(connected: boolean): void {
    this.engine?.setConnected?.(connected)
    if (connected !== this.snapshot.connected) this.update({ connected })
  }
  setTier(tier: VoiceModelTier): void {
    if (isCallActive(this.snapshot.call)) return
    const { phoneCache: _stale, ...previous } = this.snapshot
    this.snapshot = previous
    this.update({ tier })
    void this.refreshPhoneCache()
  }
  /** Display-only: a failed probe leaves the previous (or no) cache figure in place. */
  private async refreshPhoneCache(): Promise<void> {
    const probe = this.deps.phoneCache
    if (!probe) return
    const tier = this.snapshot.tier
    const id = ++this.cacheProbe
    try {
      const phoneCache = await probe(tier)
      if (id === this.cacheProbe && tier === this.snapshot.tier) this.update({ phoneCache })
    } catch { /* cache status only changes the progress wording */ }
  }
  /** Asked once per service lifetime; never delays or fails a call. */
  private requestPersistentStorage(): void {
    if (this.persistRequested) return
    this.persistRequested = true
    const persist = this.deps.persistStorage
    if (!persist) { this.update({ storagePersisted: 'unsupported' }); return }
    void (async () => {
      try { this.update({ storagePersisted: (await persist()) ? 'granted' : 'denied' }) } catch { this.update({ storagePersisted: 'unsupported' }) }
    })()
  }
  setDebugCaptureEnabled(enabled: boolean): void {
    this.deps.rememberDebugEnabled?.(enabled)
    this.engine?.setDebugCaptureEnabled(enabled)
    this.update({ debugCaptureEnabled: enabled })
  }
  async loadStatus(): Promise<void> {
    this.check?.abort()
    const check = new AbortController()
    this.check = check
    const { error: _error, ...previous } = this.snapshot
    this.snapshot = previous
    this.update({ checking: true })
    try {
      const [models] = await Promise.all([this.deps.modelsStatus(check.signal), this.refreshPhoneCache()])
      if (!check.signal.aborted) this.update({ models, checking: false })
    } catch (error) {
      if (!check.signal.aborted) this.update({ checking: false, error: error instanceof Error ? error.message : 'Voice models are unavailable.' })
    }
  }
  async start(): Promise<void> {
    if (!this.snapshot.models || !this.deps.canStart() || isCallActive(this.snapshot.call)) return
    const { error: _error, ...previous } = this.snapshot
    this.snapshot = previous
    this.requestPersistentStorage()
    try {
      this.deps.beforeStart?.()
      if (!this.engine) {
        const engine = this.deps.engine()
        this.engine = engine
        this.stopObserving = engine.subscribe(() => {
          const call = engine.getSnapshot()
          const ended = isCallActive(this.snapshot.call) && call.phase === 'idle'
          const stopped = isCallActive(this.snapshot.call) && !isCallActive(call)
          // An unrequested end (hidden too long, microphone taken) keeps the voice screen open with its reason.
          const explained = call.phase === 'error' || (ended && call.endReason !== undefined)
          this.update({ call, ...(explained ? { view: 'voice' as const } : ended ? { view: 'closed' as const } : {}) })
          if (stopped) void this.refreshPhoneCache()
        })
        engine.setConnected?.(this.snapshot.connected)
      }
      this.engine.setDebugCaptureEnabled(this.snapshot.debugCaptureEnabled)
      this.deps.rememberNotice?.()
      this.update({ view: 'voice', downloadNoticeSeen: true })
      await this.engine.start(this.snapshot.tier)
    } catch (error) {
      this.engine?.hangup()
      this.update({ view: 'voice', error: error instanceof Error ? error.message : 'Voice mode could not start.' })
    }
  }
  mute(): void { this.engine?.mute(!this.snapshot.call.muted) }
  interrupt(): void { this.engine?.interrupt() }
  visibility(hidden: boolean): void { this.engine?.visibility(hidden) }
  end(): void { this.engine?.hangup(); this.update({ view: 'closed' }) }
  /** Only app teardown (not a tab/view unmount) disposes this service. */
  dispose(): void {
    this.check?.abort()
    this.stopObserving?.()
    this.stopObserving = undefined
    this.engine?.hangup()
    this.engine = undefined
  }
}
