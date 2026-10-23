// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { requiredMediaType } from '../src/tools/mediaHttp.js'
import { OpenAiCompatibleTranscriptionPort, normalizeAudioInput } from '../src/tools/transcription.js'

describe('requiredMediaType: parameterized browser MIME types (regression)', () => {
  test('desktop dictation MIME strings are accepted and reduced to the bare type', () => {
    expect(requiredMediaType('audio/webm;codecs=opus', 'audio/', 'media_type')).toBe('audio/webm')
    expect(requiredMediaType('audio/ogg;codecs=opus', 'audio/', 'media_type')).toBe('audio/ogg')
    expect(requiredMediaType('Audio/MP4; codecs="mp4a.40.2"', 'audio/', 'media_type')).toBe('audio/mp4')
    expect(requiredMediaType('audio/webm', 'audio/', 'media_type')).toBe('audio/webm')
  })

  test('images keep working and bare types are unchanged', () => {
    expect(requiredMediaType('image/png', 'image/', 'media_type')).toBe('image/png')
    expect(requiredMediaType(' IMAGE/JPEG ', 'image/', 'media_type')).toBe('image/jpeg')
  })

  test('the wrong prefix, a second type and header smuggling are still refused', () => {
    expect(() => requiredMediaType('text/html;charset=utf-8', 'audio/', 'media_type')).toThrow('valid audio/')
    expect(() => requiredMediaType('audio/webm;codecs=opus\r\nX-Evil: 1', 'audio/', 'media_type')).toThrow('parameter')
    expect(() => requiredMediaType('audio/webm;codecs=opus,audio/ogg', 'audio/', 'media_type')).toThrow('parameter')
    expect(requiredMediaType('audio/webm;codecs=opus,vp8', 'audio/', 'media_type')).toBe('audio/webm')
    expect(() => requiredMediaType('audio/webm;', 'audio/', 'media_type')).toThrow('parameter')
    expect(() => requiredMediaType('audio/webm;codecs', 'audio/', 'media_type')).toThrow('parameter')
    expect(() => requiredMediaType('audio/webm;a=1;b=2;c=3;d=4;e=5', 'audio/', 'media_type')).toThrow('too many')
    expect(() => requiredMediaType('audio/webm;codecs=<script>', 'audio/', 'media_type')).toThrow('parameter')
  })

  test('normalizeAudioInput carries the bare type to the provider', () => {
    const input = normalizeAudioInput({ bytes: new Uint8Array([1, 2, 3]), filename: 'dictation.webm', mediaType: 'audio/webm;codecs=opus' })
    expect(input.mediaType).toBe('audio/webm')
  })

  test('the OpenAI-compatible transcription adapter sends a parameterized recording', async () => {
    let seenType: string | undefined
    const port = new OpenAiCompatibleTranscriptionPort({
      apiKey: 'test-key',
      baseUrl: 'https://stt.example.test/v1',
      fetchImplementation: async (_url, init) => {
        const file = (init?.body as FormData).get('file') as File
        seenType = file.type
        return new Response(JSON.stringify({ text: 'hello there' }), { status: 200, headers: { 'content-type': 'application/json' } })
      },
    })
    const result = await port.transcribe({ audio: { bytes: new Uint8Array([1, 2, 3]), filename: 'dictation.webm', mediaType: 'audio/webm;codecs=opus' } })
    expect(result.text).toBe('hello there')
    expect(seenType).toBe('audio/webm')
  })
})
