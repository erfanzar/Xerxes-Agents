// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  OpenAiCompatibleTranscriptionPort,
  type TranscriptionPort,
} from '../../tools/transcription.js'

export const MAX_DICTATION_BYTES = 8 * 1024 * 1024
export function dictationPort(
  env: Readonly<Record<string, string | undefined>>,
): TranscriptionPort {
  const apiKey = env.XERXES_DICTATION_API_KEY,
    baseUrl = env.XERXES_DICTATION_BASE_URL
  if (!apiKey || !baseUrl)
    throw new Error(
      'Configure XERXES_DICTATION_BASE_URL and XERXES_DICTATION_API_KEY for a transcription provider, then relaunch. Dictation does not use your chat model credentials.',
    )
  return new OpenAiCompatibleTranscriptionPort({
    apiKey,
    baseUrl,
    defaultModel: env.XERXES_DICTATION_MODEL || 'whisper-1',
  })
}
export async function transcribeDictation(
  value: unknown,
  port: TranscriptionPort,
  signal: AbortSignal,
): Promise<string> {
  if (!value || typeof value !== 'object') throw new Error('Invalid recording')
  const row = value as Record<string, unknown>
  if (
    !(row.bytes instanceof Uint8Array) ||
    !row.bytes.length ||
    row.bytes.length > MAX_DICTATION_BYTES
  )
    throw new Error('Recording must contain 1 byte to 8 MiB')
  if (
    typeof row.mediaType !== 'string' ||
    !/^audio\/(webm|ogg|mp4)(;codecs=[a-z0-9.-]+)?$/.test(row.mediaType)
  )
    throw new Error('Unsupported recording format')
  const extension = row.mediaType.split('/')[1]!.split(';')[0]
  const result = await port.transcribe(
    { audio: { bytes: row.bytes, mediaType: row.mediaType, filename: 'dictation.' + extension } },
    signal,
  )
  return result.text
}
