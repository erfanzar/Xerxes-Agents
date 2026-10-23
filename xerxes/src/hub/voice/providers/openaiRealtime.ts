// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError } from '../../../core/errors.js'
import type { RealtimeEvent, RealtimeSessionOptions, RealtimeVoicePort, RealtimeVoiceSession } from '../ports.js'
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
 * OpenAI Realtime adapter. The API key is placed in a request header on the
 * Hub-side socket and nowhere else: it is not part of any event, error or
 * capability the phone can see.
 */
export interface OpenAiRealtimeOptions {
  readonly apiKey: string
  readonly model?: string
  /** `wss://` (or `ws://` for a local gateway) endpoint, without the model query. */
  readonly baseUrl?: string
  readonly voice?: string
  readonly transcriptionModel?: string
  /** Silence that ends a user turn (provider VAD). Tuned between 300 and 600 ms. */
  readonly vadSilenceMs?: number
  readonly socketFactory?: RealtimeSocketFactory
  readonly connectTimeoutMs?: number
  readonly maxBufferedBytes?: number
}

const DEFAULT_URL = 'wss://api.openai.com/v1/realtime'
const DEFAULT_MODEL = 'gpt-4o-realtime-preview'
const RATE = 24_000

export class OpenAiRealtimePort implements RealtimeVoicePort {
  readonly providerName = 'openai-realtime'
  readonly sampleRate = RATE
  readonly outputSampleRate = RATE
  private readonly apiKey: string
  private readonly model: string
  private readonly baseUrl: URL
  private readonly voice: string
  private readonly transcriptionModel: string
  private readonly vadSilenceMs: number
  private readonly factory: RealtimeSocketFactory
  private readonly connectTimeoutMs: number
  private readonly maxBufferedBytes: number

  constructor(options: OpenAiRealtimeOptions) {
    this.apiKey = options.apiKey.trim()
    if (this.apiKey === '') throw new ConfigurationError('hub.voice.realtime', 'the OpenAI realtime provider requires an API key')
    this.model = options.model?.trim() || DEFAULT_MODEL
    try {
      this.baseUrl = new URL(options.baseUrl ?? DEFAULT_URL)
    } catch (error) {
      throw new ConfigurationError('hub.voice.realtime.base_url', 'must be an absolute ws(s) URL', {}, { cause: error })
    }
    if (this.baseUrl.protocol !== 'wss:' && this.baseUrl.protocol !== 'ws:') {
      throw new ConfigurationError('hub.voice.realtime.base_url', 'must use ws or wss')
    }
    this.voice = options.voice?.trim() || 'alloy'
    this.transcriptionModel = options.transcriptionModel?.trim() || 'whisper-1'
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
    const url = new URL(this.baseUrl)
    url.searchParams.set('model', this.model)
    const connected = openRealtimeSocket(
      this.factory,
      { url: url.toString(), headers: { Authorization: `Bearer ${this.apiKey}`, 'OpenAI-Beta': 'realtime=v1' } },
      signal,
      this.connectTimeoutMs,
      (data, emitter) => mapEvent(data, emitter),
    )
    await connected.opened
    const { socket, emitter } = connected
    const maxBuffered = this.maxBufferedBytes
    const send = (event: Readonly<Record<string, unknown>>): boolean => socket.send(JSON.stringify(event))
    send({
      type: 'session.update',
      session: {
        modalities: ['audio', 'text'],
        instructions: options.instructions,
        voice: this.voice,
        input_audio_format: 'pcm16',
        output_audio_format: 'pcm16',
        input_audio_transcription: { model: this.transcriptionModel, ...(options.language === undefined ? {} : { language: options.language }) },
        turn_detection: { type: 'server_vad', threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: this.vadSilenceMs, create_response: true },
        tools: options.tools.map(tool => ({ type: 'function', name: tool.name, description: tool.description, parameters: tool.parameters })),
        tool_choice: 'auto',
      },
    })
    return {
      supportsTruncate: true,
      onEvent: listener => emitter.on(listener),
      sendAudio(pcm) {
        if (socket.bufferedAmount() > maxBuffered) return false
        return send({ type: 'input_audio_buffer.append', audio: encodeBase64Audio(pcm) })
      },
      cancelResponse() {
        send({ type: 'response.cancel' })
      },
      truncate(itemId, playedMs) {
        send({ type: 'conversation.item.truncate', item_id: itemId, content_index: 0, audio_end_ms: Math.max(0, Math.floor(playedMs)) })
      },
      sendToolResult(callId, output) {
        send({ type: 'conversation.item.create', item: { type: 'function_call_output', call_id: callId, output } })
        send({ type: 'response.create' })
      },
      say(text) {
        send({
          type: 'response.create',
          response: { modalities: ['audio', 'text'], instructions: `Read the following message to the user aloud, word for word, and add nothing:\n${text}` },
        })
      },
      close() {
        try { socket.close(1000, 'session ended') } catch { /* already closed */ }
      },
    }
  }
}

function str(value: unknown): string | undefined {
  return typeof value === 'string' && value !== '' ? value : undefined
}

function mapEvent(data: string, emitter: RealtimeEmitter): void {
  const event = parseJsonObject(data)
  if (event === undefined) return
  switch (event.type) {
    case 'session.updated':
      emitter.emit({ type: 'ready' })
      return
    case 'input_audio_buffer.speech_started':
      emitter.emit({ type: 'speech_started' })
      return
    case 'input_audio_buffer.speech_stopped': {
      const itemId = str(event.item_id)
      emitter.emit(itemId === undefined ? { type: 'speech_stopped' } : { type: 'speech_stopped', itemId })
      return
    }
    case 'conversation.item.input_audio_transcription.completed': {
      const itemId = str(event.item_id)
      if (itemId !== undefined && typeof event.transcript === 'string') emitter.emit({ type: 'transcript', itemId, text: event.transcript.trim(), final: true })
      return
    }
    case 'response.audio.delta':
    case 'response.output_audio.delta': {
      const responseId = str(event.response_id)
      const audio = decodeBase64Audio(event.delta)
      if (responseId === undefined || audio === undefined) return
      const itemId = str(event.item_id)
      emitter.emit(itemId === undefined ? { type: 'audio', responseId, audio } : { type: 'audio', responseId, itemId, audio })
      return
    }
    case 'response.audio.done':
    case 'response.output_audio.done': {
      const responseId = str(event.response_id)
      if (responseId !== undefined) emitter.emit({ type: 'audio_done', responseId })
      return
    }
    case 'response.function_call_arguments.done': {
      const callId = str(event.call_id)
      const name = str(event.name)
      if (callId !== undefined && name !== undefined) emitter.emit({ type: 'tool_call', callId, name, arguments: typeof event.arguments === 'string' ? event.arguments : '{}' })
      return
    }
    case 'response.done': {
      const response = event.response
      const usage = typeof response === 'object' && response !== null ? (response as Record<string, unknown>).usage : undefined
      if (typeof usage === 'object' && usage !== null) {
        const record = usage as Record<string, unknown>
        const out: { type: 'usage'; inputTokens?: number; outputTokens?: number } = { type: 'usage' }
        if (typeof record.input_tokens === 'number') out.inputTokens = record.input_tokens
        if (typeof record.output_tokens === 'number') out.outputTokens = record.output_tokens
        emitter.emit(out)
      }
      return
    }
    case 'error':
      // The provider's message can echo request content; only a fixed text leaves this module.
      emitter.emit({ type: 'error', error: new VoiceError('VOICE_PROVIDER', 'the realtime provider reported an error') } satisfies RealtimeEvent)
      return
    default:
      return
  }
}
