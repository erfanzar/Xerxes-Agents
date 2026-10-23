// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { LiveVoiceControl } from '../../client/voiceControl.js'
import { PcmAggregator, floatToPcm16, pcm16ToBytes } from '../../hub/voice/pcm.js'
import type { VoiceCapabilities } from '../../hub/voice/protocol.js'
import { MicrophoneError } from './capture.js'
import { VoiceRemoteError, type VoiceClientCore } from './core.js'
import type { LiveCapture, LiveCaptureDeps } from './liveCapture.js'
import { supportsLive } from './probe.js'

/**
 * Optional hands-free session. Capture is mono PCM16 in 20 ms frames
 * aggregated to ~60 ms messages at the rate the browser reports (sent to the
 * Hub in `start`). The Hub/provider VAD decides when a turn is committed; the
 * client only drives the UI and a local barge-in flush so playback stops at
 * once while the Hub catches up.
 */

export interface LiveVoiceOptions {
  readonly core: VoiceClientCore
  readonly capture: LiveCaptureDeps
  readonly capabilities: VoiceCapabilities
  readonly conversationId: () => string | undefined
  readonly now?: () => number
  readonly newId?: () => string
  readonly readyTimeoutMs?: number
  readonly levelIntervalMs?: number
  /** Message size sent to the Hub, 40-100 ms. */
  readonly messageMs?: number
  /** Raw level (0-1) above which, for `bargeInFrames` 20 ms frames, playback is flushed locally. */
  readonly bargeInLevel?: number
  readonly bargeInFrames?: number
  readonly language?: string
}

interface LiveRun {
  readonly abort: AbortController
  readonly id: string
  readonly startedAt: number
  capture: LiveCapture | undefined
  generation: number
  sequence: number
  ready: boolean
  preRoll: Int16Array[]
  level: number
  loud: number
  levelTimer: ReturnType<typeof setInterval> | undefined
  offClosed: (() => void) | undefined
}

const PRE_ROLL_FRAMES = 6

function randomId(): string {
  const crypto = globalThis.crypto
  if (crypto && typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  const bytes = new Uint8Array(12)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')
}

function peak(samples: Float32Array): number {
  let max = 0
  for (const sample of samples) max = Math.max(max, Math.abs(sample))
  return Math.min(1, max)
}

export function createLiveVoice(options: LiveVoiceOptions): LiveVoiceControl {
  const { core, capabilities } = options
  const state = core.stateStore
  const now = options.now ?? Date.now
  const newId = options.newId ?? randomId
  const levelIntervalMs = options.levelIntervalMs ?? 100
  const bargeInLevel = options.bargeInLevel ?? 0.2
  const bargeInFrames = options.bargeInFrames ?? 3
  let run: LiveRun | undefined

  function release(target: LiveRun, notifyHub: boolean): void {
    if (target.levelTimer !== undefined) clearInterval(target.levelTimer)
    target.levelTimer = undefined
    target.offClosed?.()
    target.offClosed = undefined
    target.abort.abort()
    try { target.capture?.close() } catch { /* already closed */ }
    target.capture = undefined
    if (notifyHub) core.activeLink?.sendControl({ type: 'stop' })
    core.dropLink()
    core.setLive(false)
    if (run === target) run = undefined
  }

  function failRun(target: LiveRun, code: string, message: string): void {
    if (run !== target) return
    release(target, true)
    state.dispatch({ type: 'failed', code, message })
  }

  function send(target: LiveRun, samples: Int16Array): boolean {
    const link = core.activeLink
    if (link === undefined) return false
    const ok = link.sendFrame(
      { stream: 'mic', codec: 'pcm16', generation: target.generation, sequence: target.sequence, timestampMs: now() - target.startedAt, final: false, clipEnd: true },
      pcm16ToBytes(samples),
    )
    if (ok) target.sequence += 1
    return ok
  }

  async function start(): Promise<void> {
    if (!supportsLive(capabilities)) throw new Error('Live voice is not available on this Hub.')
    if (run !== undefined) throw new Error('A live session is already open.')
    const conversation = options.conversationId()
    if (conversation === undefined) throw new Error('Open the XERXES tab first, then talk.')
    const target: LiveRun = {
      abort: new AbortController(), id: newId(), startedAt: now(), capture: undefined, generation: 0, sequence: 0,
      ready: false, preRoll: [], level: 0, loud: 0, levelTimer: undefined, offClosed: undefined,
    }
    run = target
    core.setLive(true)
    state.dispatch({ type: 'live', active: true })
    core.unlockAudio()
    try {
      // Capabilities were checked above; only now is the microphone opened.
      const capture = await options.capture.open(target.abort.signal)
      target.capture = capture
      if (target.abort.signal.aborted) throw new MicrophoneError('cancelled', 'The live session was cancelled.')
      const rate = capture.sampleRate
      const aggregator = new PcmAggregator(rate, Math.min(100, Math.max(40, options.messageMs ?? 60)))
      capture.onFrame(samples => {
        if (run !== target || target.abort.signal.aborted) return
        const level = peak(samples)
        target.level = level
        if (core.playback.playing && level >= bargeInLevel) {
          target.loud += 1
          if (target.loud >= bargeInFrames) { target.loud = 0; core.interruptPlayback() }
        } else target.loud = 0
        const batch = aggregator.push(floatToPcm16(samples))
        if (batch === undefined) return
        if (!target.ready) {
          target.preRoll.push(batch)
          if (target.preRoll.length > PRE_ROLL_FRAMES) target.preRoll.shift()
          return
        }
        if (!send(target, batch)) failRun(target, 'VOICE_OVERLOADED', 'The connection is too slow to stream audio.')
      })
      const link = await core.ensureLink()
      if (target.abort.signal.aborted) throw new MicrophoneError('cancelled', 'The live session was cancelled.')
      target.offClosed = core.onClosed((code, reason) => {
        if (run !== target) return
        release(target, false)
        state.dispatch({ type: 'failed', code: 'VOICE_DISCONNECTED', message: reason !== '' ? `The live session ended: ${reason}` : `The live session ended (${code}).` })
      })
      const ready = core.waitReady(target.id, options.readyTimeoutMs ?? 10_000)
      ready.catch(() => undefined)
      link.sendControl({
        type: 'start', mode: 'live', conversation_id: conversation, utterance_id: target.id, codec: 'pcm16', sample_rate: rate,
        ...(options.language === undefined ? {} : { language: options.language }),
      })
      target.generation = await ready
      if (target.abort.signal.aborted) throw new MicrophoneError('cancelled', 'The live session was cancelled.')
      target.ready = true
      for (const batch of target.preRoll.splice(0)) {
        if (!send(target, batch)) { failRun(target, 'VOICE_OVERLOADED', 'The connection is too slow to stream audio.'); return }
      }
    } catch (error) {
      const cancelled = target.abort.signal.aborted || (error instanceof MicrophoneError && error.kind === 'cancelled')
      if (run === target) release(target, !cancelled)
      if (!cancelled) {
        const code = error instanceof MicrophoneError ? (error.kind === 'denied' ? 'VOICE_DENIED' : 'VOICE_PERMISSION')
          : error instanceof VoiceRemoteError ? error.code : 'VOICE_DISCONNECTED'
        state.dispatch({ type: 'failed', code, message: error instanceof Error ? error.message : 'Live voice could not start.' })
      } else state.dispatch({ type: 'live', active: false })
      throw error instanceof Error ? error : new Error('Live voice could not start.')
    }
    state.dispatch({ type: 'listening' })
    if (levelIntervalMs > 0) target.levelTimer = setInterval(() => state.dispatch({ type: 'level', level: target.level }), levelIntervalMs)
  }

  function stop(): void {
    const target = run
    if (target === undefined) return
    release(target, true)
    state.dispatch({ type: 'live', active: false })
  }

  return {
    get active() { return run !== undefined },
    start,
    stop,
  }
}
