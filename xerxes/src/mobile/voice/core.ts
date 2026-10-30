// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { VoiceLink, VoiceLinkEvent } from '../../client/voiceLink.js'
import type { VoiceStateController } from '../../client/voiceState.js'
import type { VoiceServerMessage } from '../../hub/voice/protocol.js'
import { PlaybackError, ResponsePlayback, type AudioOutput } from './playback.js'

/**
 * Shared phone-side voice plumbing: one Hub voice link, the playback of Hub
 * speech, and the translation of Hub messages into voice-state events. The
 * push-to-talk and live controllers sit on top of it.
 *
 * A dropped link is never resumed and nothing is resent: the next press opens
 * a fresh session, so audio is never replayed.
 */

export class VoiceRemoteError extends Error {
  readonly code: string
  constructor(code: string, message: string) {
    super(message)
    this.name = 'VoiceRemoteError'
    this.code = code
  }
}

export interface VoiceClientDeps {
  readonly createLink: () => VoiceLink
  readonly output: AudioOutput | undefined
  readonly state: VoiceStateController
}

interface Waiter<T> {
  readonly resolve: (value: T) => void
  readonly reject: (error: Error) => void
  readonly timer: ReturnType<typeof setTimeout>
}

const NULL_OUTPUT: AudioOutput = {
  schedule: () => Promise.reject(new PlaybackError('unsupported', 'This browser cannot play audio.')),
  flush: () => undefined,
}

export class VoiceClientCore {
  readonly playback: ResponsePlayback
  private link: VoiceLink | undefined
  private offLink: (() => void) | undefined
  private readonly ready = new Map<string, Waiter<number>>()
  private readonly transcripts = new Map<string, Waiter<string>>()
  private readonly closeListeners = new Set<(code: number, reason: string) => void>()
  private liveMode = false
  private audioUnlocked = false

  constructor(private readonly deps: VoiceClientDeps) {
    const { state } = deps
    this.playback = new ResponsePlayback(deps.output ?? NULL_OUTPUT, {
      started: (responseId, generation) => state.dispatch({ type: 'playback_started', responseId, generation }),
      ended: (responseId, generation) => state.dispatch({ type: 'playback_ended', responseId, generation }),
      empty: (responseId, generation) => state.dispatch({ type: 'playback_empty', responseId, generation }),
      failed: (responseId, _generation, error) => {
        this.link?.sendControl({ type: 'interrupt', response_id: responseId })
        state.dispatch({ type: 'failed', code: error.kind === 'blocked' ? 'VOICE_BLOCKED' : 'VOICE_FORMAT', message: error.message })
      },
    })
  }

  get stateStore(): VoiceStateController {
    return this.deps.state
  }

  get activeLink(): VoiceLink | undefined {
    return this.link?.open === true ? this.link : undefined
  }

  setLive(active: boolean): void {
    this.liveMode = active
  }

  /** Reuse the open link or open a new one. Rejects when the Hub cannot be reached. */
  async ensureLink(): Promise<VoiceLink> {
    const existing = this.link
    if (existing?.open === true) return existing
    this.dropLink()
    const link = this.deps.createLink()
    this.link = link
    this.offLink = link.onEvent(event => this.onEvent(link, event))
    try {
      await link.connect()
    } catch (error) {
      if (this.link === link) this.dropLink()
      throw error
    }
    return link
  }

  dropLink(): void {
    const link = this.link
    this.link = undefined
    this.offLink?.()
    this.offLink = undefined
    link?.close(1000, 'done')
  }

  onClosed(listener: (code: number, reason: string) => void): () => void {
    this.closeListeners.add(listener)
    return () => { this.closeListeners.delete(listener) }
  }

  waitReady(utteranceId: string, timeoutMs: number): Promise<number> {
    return this.wait(this.ready, utteranceId, timeoutMs, 'The Hub did not start the voice session in time.')
  }

  waitTranscript(utteranceId: string, timeoutMs: number): Promise<string> {
    return this.wait(this.transcripts, utteranceId, timeoutMs, 'The transcription took too long.')
  }

  /** Flush playback and tell the Hub. Local only: it never cancels an agent. */
  interruptPlayback(): void {
    const active = this.playback.activeResponse
    this.playback.interrupt()
    this.link?.sendControl({ type: 'interrupt', ...(active === undefined ? {} : { response_id: active.responseId }) })
    this.deps.state.dispatch({ type: 'interrupt' })
  }

  /**
   * Call inside a user gesture so a later spoken reply is allowed to play. WebKit may not count a
   * pointerdown as an audio gesture, so callers also call this on release; it is only done once the
   * output reports it runs.
   */
  unlockAudio(): void {
    if (this.audioUnlocked) return
    const output = this.deps.output
    if (output?.unlock === undefined) return
    // Called as a method: class-based outputs read `this`.
    void output.unlock().then(() => { this.audioUnlocked = true }, () => undefined)
  }

  /** Whether a gesture has already unlocked output audio. */
  get audioReady(): boolean {
    return this.audioUnlocked
  }

  private wait<T>(map: Map<string, Waiter<T>>, id: string, timeoutMs: number, timeoutMessage: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        map.delete(id)
        reject(new VoiceRemoteError('VOICE_PROVIDER', timeoutMessage))
      }, timeoutMs)
      map.set(id, { resolve, reject, timer })
    })
  }

  private settle<T>(map: Map<string, Waiter<T>>, id: string, value: T): void {
    const waiter = map.get(id)
    if (waiter === undefined) return
    map.delete(id)
    clearTimeout(waiter.timer)
    waiter.resolve(value)
  }

  private rejectAll(error: Error, utteranceId?: string): void {
    for (const map of [this.ready, this.transcripts] as const) {
      for (const [id, waiter] of [...map]) {
        if (utteranceId !== undefined && id !== utteranceId) continue
        map.delete(id)
        clearTimeout(waiter.timer)
        waiter.reject(error)
      }
    }
  }

  private onEvent(link: VoiceLink, event: VoiceLinkEvent): void {
    if (this.link !== link) return
    switch (event.kind) {
      case 'frame': this.playback.frame(event.frame); return
      case 'closed':
        this.link = undefined
        this.offLink?.()
        this.offLink = undefined
        this.playback.interrupt()
        this.rejectAll(new VoiceRemoteError('VOICE_DISCONNECTED', 'The voice connection dropped.'))
        for (const listener of [...this.closeListeners]) listener(event.code, event.reason)
        return
      case 'control': this.onControl(event.message); return
    }
  }

  private onControl(message: VoiceServerMessage): void {
    const { state } = this.deps
    switch (message.type) {
      case 'state':
        state.dispatch({ type: 'server_state', state: message.state })
        if (message.state === 'listening' && message.utterance_id !== undefined) this.settle(this.ready, message.utterance_id, message.generation)
        return
      case 'transcript':
        state.dispatch({ type: 'transcript', text: message.text, final: message.final, live: this.liveMode })
        if (message.final) this.settle(this.transcripts, message.utterance_id, message.text)
        return
      case 'submitted': state.dispatch({ type: 'submitted' }); return
      case 'reply': state.dispatch({ type: 'reply', spoken: message.spoken }); return
      case 'audio_start':
        state.dispatch({ type: 'audio_start', responseId: message.response_id, generation: message.generation })
        if (this.deps.output !== undefined) {
          this.playback.start({
            responseId: message.response_id, generation: message.generation, codec: message.codec,
            ...(message.sample_rate === undefined ? {} : { sampleRate: message.sample_rate }),
          })
        }
        return
      case 'audio_end':
        // The state learns the stream ended first; playback then reports ended/empty when the speaker is done.
        state.dispatch({ type: 'audio_end', responseId: message.response_id, generation: message.generation, interrupted: message.interrupted })
        this.playback.end({ responseId: message.response_id, generation: message.generation, interrupted: message.interrupted })
        // Without an output device nothing plays, so nothing would ever end the response.
        if (this.deps.output === undefined && !message.interrupted) {
          state.dispatch({ type: 'playback_empty', responseId: message.response_id, generation: message.generation })
        }
        return
      case 'error': {
        this.rejectAll(new VoiceRemoteError(message.code, message.message), message.utterance_id)
        // A late or stale frame is housekeeping, not a failure the person should see.
        if (message.code !== 'VOICE_STALE') state.dispatch({ type: 'failed', code: message.code, message: message.message })
        return
      }
      case 'capabilities':
      case 'usage':
      case 'lease':
        return
    }
  }
}
