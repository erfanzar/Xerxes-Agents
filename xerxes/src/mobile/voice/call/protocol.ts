// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { VoiceModelTier } from '../../../client/voiceModelManifest.js'

export interface CallProgress { readonly loaded: number; readonly total: number }
export interface CallAudio { readonly samples: Float32Array; readonly sampleRate: number }
export type CallWorkerRequest =
  | { readonly id: number; readonly type: 'load'; readonly tier: VoiceModelTier; readonly forceWasm?: boolean }
  | { readonly id: number; readonly type: 'vad'; readonly samples: Float32Array }
  | { readonly id: number; readonly type: 'transcribe'; readonly samples: Float32Array }
  | { readonly id: number; readonly type: 'synthesize'; readonly text: string }
  | { readonly id: number; readonly type: 'dispose' }
  | { readonly id: number; readonly type: 'reset-vad' }
  /** id identifies the existing job to cancel; cancellation has no response. */
  | { readonly id: number; readonly type: 'cancel' }
export type CallWorkerResponse =
  | { readonly id: number; readonly type: 'progress'; readonly progress: CallProgress }
  | { readonly id: number; readonly type: 'loaded'; readonly tier: VoiceModelTier; readonly notice?: string }
  | { readonly id: number; readonly type: 'probability'; readonly probability: number }
  | { readonly id: number; readonly type: 'transcript'; readonly text: string }
  | { readonly id: number; readonly type: 'retry-wasm' }
  /** WASM inference poisoned this runtime. Drop the utterance and replace the entire speech worker. */
  | { readonly id: number; readonly type: 'restart-speech' }
  | { readonly id: number; readonly type: 'audio'; readonly audio: CallAudio }
  | { readonly id: number; readonly type: 'done' }
  | { readonly id: number; readonly type: 'error'; readonly message: string }

/**
 * One step of a call failed (a stalled playback, a speech job that timed out, a phone voice error) but the
 * call itself can continue: the engine drops that step, shows the message and keeps listening.
 * Any other error ends the call.
 */
export class CallStepError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CallStepError'
  }
}

/**
 * Advice for a failed worker job. Light still runs the same VAD and Moonshine workers, so it is offered only
 * when Kokoro (synthesis) failed; anything else points at reloading the app or updating the Hub.
 */
export function workerFailureMessage(job: string, detail: string): string {
  if (job === 'synthesize') return `On-phone speech output failed (${detail}). Choose Light in voice settings to use the phone’s own voice.`
  return `On-phone ${job} failed (${detail}). Reload the app; if it keeps happening, update the Hub so its call models are available.`
}

export interface CallTranscriptionRecovery {
  restarting(): void
  progress(value: CallProgress): void
}

export interface CallModels {
  load(tier: VoiceModelTier, progress: (value: CallProgress) => void, signal: AbortSignal): Promise<{ tier: VoiceModelTier; notice?: string }>
  vad(frame: Float32Array, signal: AbortSignal): Promise<number>
  resetVad(): void
  /** Null means a failed utterance was dropped after successfully replacing the speech runtime. */
  transcribe(samples: Float32Array, signal: AbortSignal, recovery?: CallTranscriptionRecovery): Promise<string | null>
  synthesize(text: string, signal: AbortSignal): Promise<CallAudio>
  /** Cancels pending RPCs, requests model disposal, then terminates the worker (also frees a still-loading model). */
  dispose(): void
}
