// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError, ValidationError } from '../core/errors.js'
import {
  normalizeAudioInput,
  type AudioInput,
  type TranscriptionPort,
  type TranscriptionRequest,
  type TranscriptionResult,
} from './transcription.js'

/**
 * Host-owned microphone capture port. Browser, native desktop, or hardware
 * adapters must implement this; the Bun runtime does not claim microphone
 * access merely because voice mode is enabled.
 */
export interface VoiceRecorder {
  readonly recording: boolean
  start(signal?: AbortSignal): Promise<void> | void
  stop(signal?: AbortSignal): Promise<AudioInput> | AudioInput
}

/** Deliberate default for servers and terminals without an injected microphone adapter. */
export class UnavailableVoiceRecorder implements VoiceRecorder {
  readonly recording = false

  start(): never {
    throw new ConfigurationError(
      'voice.capture',
      'no microphone capture adapter is configured; inject a browser, desktop, or hardware VoiceRecorder before enabling voice mode.',
    )
  }

  stop(): never {
    throw new ConfigurationError('voice.capture', 'cannot stop recording because no microphone capture adapter is configured')
  }
}

export interface VoiceModeOptions {
  readonly recorder: VoiceRecorder
  readonly transcription: TranscriptionPort
}

/**
 * Push-to-talk coordinator. Capture remains injected while transcription is a
 * configured port, so this class never silently records audio or invokes a
 * model with missing credentials.
 */
export class VoiceModeController {
  private readonly recorder: VoiceRecorder
  private readonly transcription: TranscriptionPort

  constructor(options: VoiceModeOptions) {
    this.recorder = options.recorder
    this.transcription = options.transcription
  }

  get recording(): boolean {
    return this.recorder.recording
  }

  async start(signal?: AbortSignal): Promise<void> {
    if (this.recorder.recording) return
    await this.recorder.start(signal)
  }

  async stopAndTranscribe(
    options: Omit<TranscriptionRequest, 'audio'> = {},
    signal?: AbortSignal,
  ): Promise<TranscriptionResult> {
    if (!this.recorder.recording) {
      throw new ValidationError('voice_mode', 'cannot transcribe because push-to-talk is not recording')
    }
    const audio = normalizeAudioInput(await this.recorder.stop(signal))
    return this.transcription.transcribe({ ...options, audio }, signal)
  }
}
