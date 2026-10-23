// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError } from '../../../core/errors.js'
import type { RealtimeSessionOptions, RealtimeVoicePort, RealtimeVoiceSession } from '../ports.js'
import { VoiceError } from '../protocol.js'
import {
  bunRealtimeSocketFactory,
  decodeBase64Audio,
  encodeBase64Audio,
  openRealtimeSocket,
  parseJsonObject,
  type RealtimeEmitter,
  type RealtimeSocketFactory,
} from './realtimeSocket.js'

/**
 * Gemini Live adapter behind the same `RealtimeVoicePort` contract as the
 * OpenAI one. The key travels in the `x-goog-api-key` header of the Hub-side
 * socket (never in the URL, so it cannot land in a log line) and nowhere else.
 *
 * Differences the relay has to live with, stated honestly:
 * - Gemini has no response id or item id, so ids are synthesized per turn.
 * - There is no truncate call (`supportsTruncate` is false).
 * - There is no explicit response cancel: `cancelResponse` stops relaying the
 *   current turn's audio, the provider generation may continue until its own
 *   `turnComplete`, and sending new user audio interrupts it provider-side.
 */
export interface GeminiLiveOptions {
  readonly apiKey: string
  readonly model?: string
  readonly baseUrl?: string
  readonly voice?: string
  readonly vadSilenceMs?: number
  readonly socketFactory?: RealtimeSocketFactory
  readonly connectTimeoutMs?: number
  readonly maxBufferedBytes?: number
}

const DEFAULT_URL = 'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent'
const DEFAULT_MODEL = 'gemini-2.0-flash-live-001'
const INPUT_RATE = 16_000
const OUTPUT_RATE = 24_000

interface TurnState {
  responseSeq: number
  inputSeq: number
  inputText: string
  inputFinalized: boolean
  suppress: boolean
  callNames: Map<string, string>
}

export class GeminiLivePort implements RealtimeVoicePort {
  readonly providerName = 'gemini-live'
  readonly sampleRate = INPUT_RATE
  readonly outputSampleRate = OUTPUT_RATE
  private readonly apiKey: string
  private readonly model: string
  private readonly baseUrl: string
  private readonly voice: string | undefined
  private readonly vadSilenceMs: number
  private readonly factory: RealtimeSocketFactory
  private readonly connectTimeoutMs: number
  private readonly maxBufferedBytes: number

  constructor(options: GeminiLiveOptions) {
    this.apiKey = options.apiKey.trim()
    if (this.apiKey === '') throw new ConfigurationError('hub.voice.realtime', 'the Gemini Live provider requires an API key')
    this.model = (options.model?.trim() || DEFAULT_MODEL).replace(/^models\//, '')
    const base = options.baseUrl ?? DEFAULT_URL
    let parsed: URL
    try { parsed = new URL(base) } catch (error) {
      throw new ConfigurationError('hub.voice.realtime.base_url', 'must be an absolute ws(s) URL', {}, { cause: error })
    }
    if (parsed.protocol !== 'wss:' && parsed.protocol !== 'ws:') throw new ConfigurationError('hub.voice.realtime.base_url', 'must use ws or wss')
    if (parsed.search !== '') throw new ConfigurationError('hub.voice.realtime.base_url', 'must not carry a query string; credentials go in headers')
    this.baseUrl = parsed.toString()
    this.voice = options.voice?.trim() || undefined
    const silence = options.vadSilenceMs ?? 500
    if (!Number.isInteger(silence) || silence < 100 || silence > 3000) {
      throw new ConfigurationError('hub.voice.realtime.vad_silence_ms', 'must be an integer from 100 to 3000')
    }
    this.vadSilenceMs = silence
    this.factory = options.socketFactory ?? bunRealtimeSocketFactory
    this.connectTimeoutMs = options.connectTimeoutMs ?? 10_000
    this.maxBufferedBytes = options.maxBufferedBytes ?? 512 * 1024
  }

  async connect(options: RealtimeSessionOptions, signal: AbortSignal): Promise<RealtimeVoiceSession> {
    const state: TurnState = { responseSeq: 1, inputSeq: 1, inputText: '', inputFinalized: false, suppress: false, callNames: new Map() }
    let markReady: () => void = () => undefined
    const ready = new Promise<void>(resolve => { markReady = resolve })
    const connected = openRealtimeSocket(
      this.factory,
      { url: this.baseUrl, headers: { 'x-goog-api-key': this.apiKey } },
      signal,
      this.connectTimeoutMs,
      (data, emitter) => mapEvent(data, emitter, state, markReady),
    )
    await connected.opened
    const { socket, emitter } = connected
    const send = (message: Readonly<Record<string, unknown>>): boolean => socket.send(JSON.stringify(message))
    send({
      setup: {
        model: `models/${this.model}`,
        generationConfig: {
          responseModalities: ['AUDIO'],
          ...(this.voice === undefined ? {} : { speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: this.voice } } } }),
        },
        systemInstruction: { parts: [{ text: options.instructions }] },
        tools: options.tools.length === 0
          ? []
          : [{ functionDeclarations: options.tools.map(tool => ({ name: tool.name, description: tool.description, parameters: tool.parameters })) }],
        inputAudioTranscription: {},
        realtimeInputConfig: { automaticActivityDetection: { silenceDurationMs: this.vadSilenceMs, prefixPaddingMs: 300 } },
      },
    })
    await waitForReady(ready, signal, this.connectTimeoutMs, () => { try { socket.close(1000, 'setup failed') } catch { /* closed */ } })
    const maxBuffered = this.maxBufferedBytes
    return {
      supportsTruncate: false,
      onEvent: listener => emitter.on(listener),
      sendAudio(pcm) {
        if (socket.bufferedAmount() > maxBuffered) return false
        return send({ realtimeInput: { audio: { data: encodeBase64Audio(pcm), mimeType: `audio/pcm;rate=${INPUT_RATE}` } } })
      },
      cancelResponse() {
        state.suppress = true
      },
      truncate() {
        // Gemini cannot trim a stored response; the relay reports supportsTruncate=false.
      },
      sendToolResult(callId, output) {
        send({ toolResponse: { functionResponses: [{ id: callId, name: state.callNames.get(callId) ?? 'unknown', response: { output } }] } })
        state.callNames.delete(callId)
      },
      say(text) {
        state.suppress = false
        send({
          clientContent: {
            turns: [{ role: 'user', parts: [{ text: `Read the following message to the user aloud, word for word, and add nothing:\n${text}` }] }],
            turnComplete: true,
          },
        })
      },
      close() {
        try { socket.close(1000, 'session ended') } catch { /* already closed */ }
      },
    }
  }
}

function waitForReady(ready: Promise<void>, signal: AbortSignal, timeoutMs: number, abort: () => void): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => { abort(); reject(new VoiceError('VOICE_PROVIDER', 'the realtime provider did not finish setup in time')) }, timeoutMs)
    const onAbort = (): void => { clearTimeout(timer); abort(); reject(new VoiceError('VOICE_CANCELLED', 'the realtime connection was cancelled')) }
    if (signal.aborted) { onAbort(); return }
    signal.addEventListener('abort', onAbort, { once: true })
    void ready.then(() => { clearTimeout(timer); signal.removeEventListener('abort', onAbort); resolve() })
  })
}

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined
}

function finalizeInput(emitter: RealtimeEmitter, state: TurnState): void {
  if (state.inputFinalized || state.inputText.trim() === '') return
  state.inputFinalized = true
  emitter.emit({ type: 'transcript', itemId: `in-${state.inputSeq}`, text: state.inputText.trim(), final: true })
}

function mapEvent(data: string, emitter: RealtimeEmitter, state: TurnState, markReady: () => void): void {
  const message = parseJsonObject(data)
  if (message === undefined) return
  if (message.setupComplete !== undefined) {
    markReady()
    emitter.emit({ type: 'ready' })
    return
  }
  const content = record(message.serverContent)
  if (content !== undefined) {
    const input = record(content.inputTranscription)
    if (input !== undefined && typeof input.text === 'string' && input.text !== '') {
      if (state.inputFinalized) {
        state.inputSeq += 1
        state.inputText = ''
        state.inputFinalized = false
      }
      if (state.inputText === '') emitter.emit({ type: 'speech_started' })
      state.inputText += input.text
      emitter.emit({ type: 'transcript', itemId: `in-${state.inputSeq}`, text: state.inputText.trim(), final: false })
    }
    if (content.interrupted === true) {
      state.suppress = false
      state.responseSeq += 1
      emitter.emit({ type: 'speech_started' })
    }
    const turn = record(content.modelTurn)
    const parts = turn !== undefined && Array.isArray(turn.parts) ? turn.parts : []
    for (const part of parts) {
      const inline = record(record(part)?.inlineData)
      if (inline === undefined) continue
      finalizeInput(emitter, state)
      const audio = decodeBase64Audio(inline.data)
      if (audio === undefined || state.suppress) continue
      emitter.emit({ type: 'audio', responseId: `resp-${state.responseSeq}`, audio })
    }
    if (content.turnComplete === true || content.generationComplete === true) {
      finalizeInput(emitter, state)
      if (!state.suppress) emitter.emit({ type: 'audio_done', responseId: `resp-${state.responseSeq}` })
      state.suppress = false
      state.responseSeq += 1
    }
  }
  const toolCall = record(message.toolCall)
  if (toolCall !== undefined && Array.isArray(toolCall.functionCalls)) {
    finalizeInput(emitter, state)
    for (const call of toolCall.functionCalls) {
      const entry = record(call)
      if (entry === undefined || typeof entry.id !== 'string' || typeof entry.name !== 'string') continue
      state.callNames.set(entry.id, entry.name)
      emitter.emit({ type: 'tool_call', callId: entry.id, name: entry.name, arguments: JSON.stringify(record(entry.args) ?? {}) })
    }
  }
  const usage = record(message.usageMetadata)
  if (usage !== undefined) {
    const out: { type: 'usage'; inputTokens?: number; outputTokens?: number } = { type: 'usage' }
    if (typeof usage.promptTokenCount === 'number') out.inputTokens = usage.promptTokenCount
    if (typeof usage.responseTokenCount === 'number') out.outputTokens = usage.responseTokenCount
    emitter.emit(out)
  }
  if (message.error !== undefined) {
    // Provider error text can echo request content; only a fixed message leaves this module.
    emitter.emit({ type: 'error', error: new VoiceError('VOICE_PROVIDER', 'the realtime provider reported an error') })
  }
}
