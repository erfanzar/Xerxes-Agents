// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { normalizeAudioInput, type TranscriptionPort } from '../../tools/transcription.js'
import type { AudioFormat, SynthesizedAudio, TextToSpeechPort as BaseTextToSpeechPort, TextToSpeechRequest } from '../../tools/tts.js'
import { VoiceError, type VoiceCodec } from './protocol.js'

/**
 * Provider ports of the Hub voice subsystem. Every method takes an
 * AbortSignal and honours it: a cancelled call rejects with
 * `VOICE_CANCELLED` and its result, if any, is discarded. Failures are typed
 * `VoiceError`s that never carry a provider response body; nothing here
 * invents a transcript or claims audio was played.
 */

// ---------------------------------------------------------------- speech to text

export interface SpeechToTextRequest {
  readonly audio: Uint8Array
  /** Bare or parameterized recorder MIME type, e.g. `audio/webm;codecs=opus`. */
  readonly mediaType: string
  readonly language?: string
}

export interface SpeechToTextResult {
  /** Trimmed recognized text. Empty when nothing was heard; it is never a placeholder. */
  readonly text: string
  readonly provider: string
  readonly model: string
}

export interface SpeechToTextPort {
  readonly providerName: string
  transcribe(request: SpeechToTextRequest, signal: AbortSignal): Promise<SpeechToTextResult>
}

const EXTENSIONS: Readonly<Record<string, string>> = {
  'audio/webm': 'webm',
  'audio/ogg': 'ogg',
  'audio/mp4': 'mp4',
  'audio/mpeg': 'mp3',
  'audio/wav': 'wav',
  'audio/flac': 'flac',
  'audio/aac': 'aac',
}

export function throwIfAborted(signal: AbortSignal, what: string): void {
  if (signal.aborted) throw new VoiceError('VOICE_CANCELLED', `${what} was cancelled`)
}

/** Adapt the existing `TranscriptionPort` (batch, multipart) to the voice STT port. */
export function adaptTranscriptionPort(port: TranscriptionPort, options: { readonly defaultLanguage?: string } = {}): SpeechToTextPort {
  return {
    providerName: port.providerName,
    async transcribe(request, signal) {
      throwIfAborted(signal, 'transcription')
      const bare = request.mediaType.split(';')[0]?.trim().toLowerCase() ?? ''
      let audio
      try {
        audio = normalizeAudioInput({ bytes: request.audio, filename: `utterance.${EXTENSIONS[bare] ?? 'bin'}`, mediaType: request.mediaType })
      } catch (error) {
        throw new VoiceError('VOICE_FORMAT', 'the recording is empty or has an unsupported audio type', error)
      }
      const language = request.language ?? options.defaultLanguage
      let result
      try {
        result = await port.transcribe({ audio, ...(language === undefined ? {} : { language }) }, signal)
      } catch (error) {
        if (signal.aborted) throw new VoiceError('VOICE_CANCELLED', 'transcription was cancelled', error)
        throw new VoiceError('VOICE_PROVIDER', 'the transcription provider failed', error)
      }
      throwIfAborted(signal, 'transcription')
      return { text: result.text.trim(), provider: result.backend, model: result.model }
    },
  }
}

// ---------------------------------------------------------------- text to speech

/**
 * TTS port. `synthesize` is the existing buffered call. `delivery` states
 * honestly whether `stream` produces audio while synthesis is still running
 * ('streaming') or whether the whole clip is produced first ('buffered').
 */
export interface TextToSpeechPort extends BaseTextToSpeechPort {
  readonly delivery: 'buffered' | 'streaming'
  stream?(request: TextToSpeechRequest, signal: AbortSignal): AsyncIterable<Uint8Array>
}

/** Wrap a buffered `TextToSpeechPort` (such as the OpenAI-compatible adapter). */
export function adaptBufferedTextToSpeech(port: BaseTextToSpeechPort): TextToSpeechPort {
  return {
    providerName: port.providerName,
    delivery: 'buffered',
    async synthesize(request, signal) {
      if (signal !== undefined) throwIfAborted(signal, 'speech synthesis')
      try {
        const audio = await port.synthesize(request, signal)
        if (signal !== undefined) throwIfAborted(signal, 'speech synthesis')
        return audio
      } catch (error) {
        if (error instanceof VoiceError) throw error
        if (signal?.aborted === true) throw new VoiceError('VOICE_CANCELLED', 'speech synthesis was cancelled', error)
        throw new VoiceError('VOICE_PROVIDER', 'the speech provider failed', error)
      }
    },
  }
}

const FORMAT_CODEC: Readonly<Record<AudioFormat, VoiceCodec>> = {
  aac: 'aac',
  flac: 'flac',
  mp3: 'mp3',
  opus: 'opus',
  pcm: 'pcm16',
  wav: 'wav',
}

export function codecForAudioFormat(format: AudioFormat): VoiceCodec {
  return FORMAT_CODEC[format]
}

export interface SpeechChunk {
  readonly bytes: Uint8Array
  readonly codec: VoiceCodec
}

/**
 * Produce audio chunks for one piece of text. A streaming port yields as
 * synthesis proceeds; a buffered port yields the finished clip in slices.
 * Cancellation stops the iteration; no chunk is yielded after the signal fires.
 */
export async function* speechChunks(
  port: TextToSpeechPort,
  request: TextToSpeechRequest,
  signal: AbortSignal,
  chunkBytes: number,
): AsyncGenerator<SpeechChunk> {
  throwIfAborted(signal, 'speech synthesis')
  if (port.delivery === 'streaming' && port.stream !== undefined) {
    const codec = codecForAudioFormat(request.format ?? 'mp3')
    try {
      for await (const bytes of port.stream(request, signal)) {
        throwIfAborted(signal, 'speech synthesis')
        for (let offset = 0; offset < bytes.byteLength; offset += chunkBytes) yield { bytes: bytes.subarray(offset, offset + chunkBytes), codec }
      }
    } catch (error) {
      if (error instanceof VoiceError) throw error
      if (signal.aborted) throw new VoiceError('VOICE_CANCELLED', 'speech synthesis was cancelled', error)
      throw new VoiceError('VOICE_PROVIDER', 'the speech provider failed', error)
    }
    return
  }
  const clip: SynthesizedAudio = await port.synthesize(request, signal)
  throwIfAborted(signal, 'speech synthesis')
  if (clip.audio.byteLength === 0) throw new VoiceError('VOICE_PROVIDER', 'the speech provider returned no audio')
  const codec = codecForAudioFormat(clip.format)
  for (let offset = 0; offset < clip.audio.byteLength; offset += chunkBytes) {
    throwIfAborted(signal, 'speech synthesis')
    yield { bytes: clip.audio.subarray(offset, offset + chunkBytes), codec }
  }
}

// ---------------------------------------------------------------- realtime

export interface RealtimeToolSpec {
  readonly name: string
  readonly description: string
  readonly parameters: Readonly<Record<string, unknown>>
}

export interface RealtimeSessionOptions {
  readonly instructions: string
  readonly tools: readonly RealtimeToolSpec[]
  /** Rate of PCM16 audio exchanged with the relay (both directions). */
  readonly sampleRate: number
  readonly language?: string
}

/** Provider events, normalized. Audio is mono PCM16 at the port's `sampleRate`. */
export type RealtimeEvent =
  | { readonly type: 'ready' }
  | { readonly type: 'speech_started' }
  | { readonly type: 'speech_stopped'; readonly itemId?: string }
  | { readonly type: 'transcript'; readonly itemId: string; readonly text: string; readonly final: boolean }
  | { readonly type: 'audio'; readonly responseId: string; readonly itemId?: string; readonly audio: Uint8Array }
  | { readonly type: 'audio_done'; readonly responseId: string }
  | { readonly type: 'tool_call'; readonly callId: string; readonly name: string; readonly arguments: string }
  | { readonly type: 'usage'; readonly inputTokens?: number; readonly outputTokens?: number }
  | { readonly type: 'error'; readonly error: VoiceError }
  | { readonly type: 'closed'; readonly reason: string }

export interface RealtimeVoiceSession {
  /** Whether `truncate` actually trims the provider's stored response. */
  readonly supportsTruncate: boolean
  onEvent(listener: (event: RealtimeEvent) => void): () => void
  /** Returns false when the provider socket is backed up beyond its bound (the caller ends the stream). */
  sendAudio(pcm: Uint8Array): boolean
  cancelResponse(): void
  /** Trim the assistant item to what the user actually heard. */
  truncate(itemId: string, playedMs: number): void
  sendToolResult(callId: string, output: string): void
  /** Have the model say exactly this supervisor-authored text. */
  say(text: string): void
  close(): void
}

export interface RealtimeVoicePort {
  readonly providerName: string
  /** Rate of PCM16 the relay sends to the provider. */
  readonly sampleRate: number
  /** Rate of PCM16 the provider returns; defaults to `sampleRate`. */
  readonly outputSampleRate?: number
  connect(options: RealtimeSessionOptions, signal: AbortSignal): Promise<RealtimeVoiceSession>
}
