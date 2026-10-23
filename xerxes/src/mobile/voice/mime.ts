// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { RECORDER_CODECS, codecForMediaType, type VoiceCodec } from '../../hub/voice/protocol.js'

export interface RecorderFormat {
  /** Exactly what is passed to MediaRecorder. */
  readonly mimeType: string
  /** What the Hub is told. The Hub derives the media type from this, never from the phone's string. */
  readonly codec: VoiceCodec
}

/**
 * Preference order: WebM/Opus where supported, MP4/AAC for Safari, then Ogg/Opus.
 * Bare types come last for browsers that only accept the container.
 */
const CANDIDATES: readonly string[] = [
  'audio/webm;codecs=opus',
  'audio/mp4;codecs=mp4a.40.2',
  'audio/mp4',
  'audio/ogg;codecs=opus',
  'audio/webm',
  'audio/ogg',
]

export function chooseRecorderFormat(
  isTypeSupported: (mimeType: string) => boolean,
  allowed: readonly VoiceCodec[] = RECORDER_CODECS,
): RecorderFormat | undefined {
  for (const mimeType of CANDIDATES) {
    const codec = codecForMediaType(mimeType)
    if (codec === undefined || !allowed.includes(codec)) continue
    let supported = false
    try {
      supported = isTypeSupported(mimeType)
    } catch {
      supported = false
    }
    if (supported) return { mimeType, codec }
  }
  return undefined
}
