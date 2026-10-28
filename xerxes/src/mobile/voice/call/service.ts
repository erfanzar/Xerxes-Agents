// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { VoiceModelTier } from '../../../client/voiceModelManifest.js'
import type { CallEngine, CallSnapshot } from './engine.js'
import { chooseCallTier, type CallModelsStatus } from './logic.js'

export type CallEngineControl = Pick<CallEngine, 'start' | 'hangup' | 'mute' | 'interrupt' | 'visibility' | 'getSnapshot' | 'subscribe'>
export interface VoiceCallSnapshot {
  readonly view: 'closed' | 'voice' | 'chat'
  readonly call: CallSnapshot
  readonly tier: VoiceModelTier
  readonly connected: boolean
  readonly checking: boolean
  readonly models?: CallModelsStatus
  readonly error?: string
  readonly downloadNoticeSeen: boolean
}
export interface VoiceCallServiceDeps {
  readonly engine: () => CallEngineControl
  readonly modelsStatus: (signal: AbortSignal) => Promise<CallModelsStatus>
  readonly canStart: () => boolean
  readonly beforeStart?: () => void
  readonly deviceMemory?: number
  readonly noticeSeen?: () => boolean
  readonly rememberNotice?: () => void
}
export const isCallActive = (call: CallSnapshot): boolean => call.phase !== 'idle' && call.phase !== 'error'

/** App-owned call session. View/navigation state never owns the microphone or model lifetime. */
export class VoiceCallService {
  private engine: CallEngineControl | undefined
  private stopObserving: (() => void) | undefined
  private check: AbortController | undefined
  private snapshot: VoiceCallSnapshot
  private readonly observers = new Set<() => void>()
  constructor(private readonly deps: VoiceCallServiceDeps) {
    const tier = chooseCallTier(deps.deviceMemory)
    this.snapshot = { view: 'closed', tier, connected: true, checking: false, downloadNoticeSeen: deps.noticeSeen?.() ?? false,
      call: { phase: 'idle', tier, muted: false, heard: '', reply: '' } }
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
  setConnected(connected: boolean): void { if (connected !== this.snapshot.connected) this.update({ connected }) }
  setTier(tier: VoiceModelTier): void { if (!isCallActive(this.snapshot.call)) this.update({ tier }) }
  async loadStatus(): Promise<void> {
    this.check?.abort()
    const check = new AbortController()
    this.check = check
    const { error: _error, ...previous } = this.snapshot
    this.snapshot = previous
    this.update({ checking: true })
    try {
      const models = await this.deps.modelsStatus(check.signal)
      if (!check.signal.aborted) this.update({ models, checking: false })
    } catch (error) {
      if (!check.signal.aborted) this.update({ checking: false, error: error instanceof Error ? error.message : 'Voice models are unavailable.' })
    }
  }
  async start(): Promise<void> {
    if (!this.snapshot.models || !this.deps.canStart() || isCallActive(this.snapshot.call)) return
    const { error: _error, ...previous } = this.snapshot
    this.snapshot = previous
    try {
      this.deps.beforeStart?.()
      if (!this.engine) {
        const engine = this.deps.engine()
        this.engine = engine
        this.stopObserving = engine.subscribe(() => {
          const call = engine.getSnapshot()
          const ended = isCallActive(this.snapshot.call) && call.phase === 'idle'
          this.update({ call, ...(call.phase === 'error' ? { view: 'voice' as const } : ended ? { view: 'closed' as const } : {}) })
        })
      }
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
