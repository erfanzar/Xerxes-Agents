// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { VoiceControl } from '../../client/voiceControl.js'
import type { VoiceCapabilities } from '../../hub/voice/protocol.js'
import { MicrophoneError, startCapture, type CaptureSession, type MicrophoneDeps } from './capture.js'
import { VoiceRemoteError, type VoiceClientCore } from './core.js'
import { chooseRecorderFormat } from './mime.js'
import { supportsPushToTalk } from './probe.js'

/**
 * `VoiceControl` over the Hub voice endpoint: hold to record with
 * MediaRecorder, send 250-1000 ms slices in order, let the Hub run speech to
 * text once, and return the text for an editable draft. Nothing here submits a
 * message: the phone sends the edited text through the normal supervisor path.
 */

export interface PushToTalkOptions {
  readonly core: VoiceClientCore
  readonly mic: MicrophoneDeps
  readonly capabilities: VoiceCapabilities
  /** The supervisor thread to speak into; undefined until the XERXES tab has opened it. */
  readonly conversationId: () => string | undefined
  readonly now?: () => number
  readonly newId?: () => string
  readonly readyTimeoutMs?: number
  readonly transcriptTimeoutMs?: number
  /** Microphone level refresh; 0 disables it. */
  readonly levelIntervalMs?: number
  readonly language?: string
}

interface Run {
  readonly id: string
  readonly abort: AbortController
  readonly startedAt: number
  readonly codec: 'webm-opus' | 'ogg-opus' | 'mp4-aac'
  generation: number
  sequence: number
  capture: CaptureSession | undefined
  levelTimer: ReturnType<typeof setInterval> | undefined
  failed: Error | undefined
}

function randomId(): string {
  const crypto = globalThis.crypto
  if (crypto && typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  const bytes = new Uint8Array(12)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')
}

export function createPushToTalk(options: PushToTalkOptions): VoiceControl {
  const { core, mic, capabilities } = options
  const state = core.stateStore
  const now = options.now ?? Date.now
  const newId = options.newId ?? randomId
  const readyTimeoutMs = options.readyTimeoutMs ?? 8000
  const transcriptTimeoutMs = options.transcriptTimeoutMs ?? 45_000
  const levelIntervalMs = options.levelIntervalMs ?? 100
  let current: Run | undefined

  function cleanup(run: Run): void {
    if (run.levelTimer !== undefined) clearInterval(run.levelTimer)
    run.levelTimer = undefined
    run.capture?.cancel()
    run.capture = undefined
    if (current === run) current = undefined
  }

  function fail(run: Run, error: Error, code: string): void {
    run.failed ??= error
    run.abort.abort()
    core.activeLink?.sendControl({ type: 'interrupt' })
    cleanup(run)
    state.dispatch({ type: 'failed', code, message: error.message })
  }

  async function start(): Promise<void> {
    if (!supportsPushToTalk(capabilities)) throw new Error('Voice is not available on this Hub.')
    if (current !== undefined) throw new Error('A recording is already in progress.')
    const conversation = options.conversationId()
    if (conversation === undefined) throw new Error('Open the XERXES tab first, then talk.')
    // Capabilities and format support are checked before the microphone is touched.
    const format = chooseRecorderFormat(mimeType => mic.isTypeSupported(mimeType))
    if (format === undefined || (format.codec !== 'webm-opus' && format.codec !== 'ogg-opus' && format.codec !== 'mp4-aac')) {
      const error = new MicrophoneError('format', 'This browser cannot record in a format the Hub accepts. Type instead.')
      state.dispatch({ type: 'failed', code: 'VOICE_FORMAT', message: error.message })
      throw error
    }
    const run: Run = {
      id: newId(), abort: new AbortController(), startedAt: now(), codec: format.codec,
      generation: 0, sequence: 0, capture: undefined, levelTimer: undefined, failed: undefined,
    }
    current = run
    state.dispatch({ type: 'connecting' })
    core.unlockAudio()
    try {
      const link = await core.ensureLink()
      if (run.abort.signal.aborted) throw new MicrophoneError('cancelled', 'The recording was cancelled.')
      const ready = core.waitReady(run.id, readyTimeoutMs)
      ready.catch(() => undefined)
      link.sendControl({
        type: 'start', mode: 'push_to_talk', conversation_id: conversation, utterance_id: run.id, codec: format.codec,
        ...(options.language === undefined ? {} : { language: options.language }),
      })
      run.generation = await ready
      if (run.abort.signal.aborted) throw new MicrophoneError('cancelled', 'The recording was cancelled.')
      run.capture = await startCapture(mic, {
        maxSeconds: capabilities.limits.max_utterance_seconds,
        maxBytes: capabilities.limits.max_utterance_bytes,
        signal: run.abort.signal,
        onChunk: bytes => {
          const sent = core.activeLink?.sendFrame(
            { stream: 'mic', codec: run.codec, generation: run.generation, sequence: run.sequence, timestampMs: now() - run.startedAt, final: false, clipEnd: false },
            bytes,
          )
          if (sent !== true) {
            fail(run, new Error('The connection is too slow to send audio.'), 'VOICE_OVERLOADED')
            return
          }
          run.sequence += 1
        },
        onLimit: () => undefined,
        onError: error => fail(run, error, 'VOICE_FORMAT'),
      })
      if (run.abort.signal.aborted) {
        run.capture.cancel()
        throw new MicrophoneError('cancelled', 'The recording was cancelled.')
      }
    } catch (error) {
      const wasCancelled = run.abort.signal.aborted || (error instanceof MicrophoneError && error.kind === 'cancelled')
      cleanup(run)
      if (!wasCancelled) {
        const code = error instanceof MicrophoneError ? (error.kind === 'denied' ? 'VOICE_DENIED' : error.kind === 'format' ? 'VOICE_FORMAT' : 'VOICE_PERMISSION')
          : error instanceof VoiceRemoteError ? error.code : 'VOICE_DISCONNECTED'
        state.dispatch({ type: 'failed', code, message: error instanceof Error ? error.message : 'Voice could not start.' })
        core.activeLink?.sendControl({ type: 'interrupt' })
      }
      throw error instanceof Error ? error : new Error('Voice could not start.')
    }
    state.dispatch({ type: 'listening' })
    if (levelIntervalMs > 0) {
      run.levelTimer = setInterval(() => state.dispatch({ type: 'level', level: run.capture?.level() ?? 0 }), levelIntervalMs)
    }
  }

  async function stop(): Promise<string | undefined> {
    const run = current
    if (run === undefined || run.capture === undefined) return undefined
    if (run.failed !== undefined) throw run.failed
    state.dispatch({ type: 'transcribing' })
    try {
      const transcript = core.waitTranscript(run.id, transcriptTimeoutMs)
      transcript.catch(() => undefined)
      await run.capture.stop()
      if (run.failed !== undefined) throw run.failed
      const link = core.activeLink
      if (link === undefined) throw new VoiceRemoteError('VOICE_DISCONNECTED', 'The voice connection dropped.')
      link.sendFrame(
        { stream: 'mic', codec: run.codec, generation: run.generation, sequence: run.sequence, timestampMs: now() - run.startedAt, final: true, clipEnd: false },
        new Uint8Array(0),
      )
      link.sendControl({ type: 'commit', utterance_id: run.id })
      const text = (await transcript).trim()
      cleanup(run)
      return text === '' ? undefined : text
    } catch (error) {
      const failure = error instanceof Error ? error : new Error('Could not finish the recording.')
      if (current === run) {
        cleanup(run)
        state.dispatch({ type: 'failed', code: error instanceof VoiceRemoteError ? error.code : 'VOICE_PROVIDER', message: failure.message })
      }
      throw failure
    }
  }

  function cancel(): void {
    const run = current
    if (run === undefined) return
    run.abort.abort()
    core.activeLink?.sendControl({ type: 'interrupt' })
    cleanup(run)
    state.dispatch({ type: 'reset' })
  }

  async function armReply(enabled: boolean): Promise<void> {
    try {
      if (!enabled && core.activeLink === undefined) return
      const conversation = options.conversationId()
      if (conversation === undefined) return
      const link = await core.ensureLink()
      link.sendControl({ type: 'expect_reply', enabled, conversation_id: conversation })
    } catch {
      // The reply is still shown on screen; it just will not be spoken.
    }
  }

  return {
    start,
    stop,
    cancel,
    state,
    interrupt: () => core.interruptPlayback(),
    armReply,
  }
}
