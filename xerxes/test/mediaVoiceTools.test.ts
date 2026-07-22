// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ConfigurationError } from '../src/core/errors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { HttpMediaClient } from '../src/tools/mediaHttp.js'
import {
  OpenAiCompatibleImageGenerationPort,
  writeGeneratedImage,
} from '../src/tools/imageGeneration.js'
import {
  createOpenAiCompatibleMediaTools,
  registerMediaTools,
} from '../src/tools/mediaTools.js'
import { OpenAiCompatibleTranscriptionPort } from '../src/tools/transcription.js'
import { OpenAiCompatibleTextToSpeechPort, serializableSynthesizedAudio } from '../src/tools/tts.js'
import { UnavailableVoiceRecorder, VoiceModeController, type VoiceRecorder } from '../src/tools/voiceMode.js'
import {
  AnthropicVisionPort,
  OpenAiCompatibleVisionPort,
  visionImageFromBase64,
} from '../src/tools/vision.js'
import type { ToolCall } from '../src/types/toolCalls.js'

test('media HTTP ports require an explicit credential unless a local gateway is deliberately allowed', () => {
  expect(() => new OpenAiCompatibleImageGenerationPort({ baseUrl: 'https://media.test/v1' }))
    .toThrow(ConfigurationError)
  expect(() => new OpenAiCompatibleTextToSpeechPort({ baseUrl: 'https://media.test/v1' }))
    .toThrow('explicit API key')
})

test('OpenAI-compatible image generation posts a serializable request and can persist an explicit output', async () => {
  let endpoint = ''
  let authorization = ''
  let payload: Record<string, unknown> = {}
  const port = new OpenAiCompatibleImageGenerationPort({
    apiKey: 'image-key',
    baseUrl: 'https://media.test/v1',
    fetchImplementation: async (input, init) => {
      endpoint = input.toString()
      authorization = new Headers(init?.headers).get('Authorization') ?? ''
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return jsonResponse({ data: [{ b64_json: 'UE5H', revised_prompt: 'A sharper fox' }] })
    },
  })

  const result = await port.generate({ count: 1, prompt: 'a fox', size: '512x512' })
  expect(endpoint).toBe('https://media.test/v1/images/generations')
  expect(authorization).toBe('Bearer image-key')
  expect(payload).toMatchObject({ prompt: 'a fox', n: 1, size: '512x512', response_format: 'b64_json' })
  expect(result).toEqual({
    model: 'gpt-image-1',
    size: '512x512',
    count: 1,
    images: [{ b64: 'UE5H', format: 'png', revisedPrompt: 'A sharper fox' }],
  })

  const directory = await mkdtemp(join(tmpdir(), 'xerxes-media-image-'))
  try {
    const path = join(directory, 'output.png')
    const saved = await writeGeneratedImage(result.images[0]!, path)
    expect(saved).toEqual({ bytes: 3, path })
    expect(await Bun.file(path).text()).toBe('PNG')
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

test('OpenAI-compatible vision supports base64 images and direct Anthropic vision uses x-api-key', async () => {
  let openAiPayload: Record<string, unknown> = {}
  const openAi = new OpenAiCompatibleVisionPort({
    apiKey: 'vision-key',
    baseUrl: 'https://media.test/v1',
    defaultModel: 'openai-vision-model',
    fetchImplementation: async (_input, init) => {
      openAiPayload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return jsonResponse({ choices: [{ message: { content: 'A blue square.' } }] })
    },
  })
  const openAiResult = await openAi.analyze({ image: visionImageFromBase64('UE5H'), prompt: 'What is shown?' })
  expect(openAiResult.response).toBe('A blue square.')
  const messages = openAiPayload.messages as Array<Record<string, unknown>>
  const content = messages[0]?.content as Array<Record<string, unknown>>
  expect((content[1]?.image_url as { url: string }).url).toBe('data:image/png;base64,UE5H')

  let anthropicHeaders = new Headers()
  const anthropic = new AnthropicVisionPort({
    apiKey: 'anthropic-key',
    baseUrl: 'https://api.anthropic.test',
    defaultModel: 'anthropic-vision-model',
    fetchImplementation: async (_input, init) => {
      anthropicHeaders = new Headers(init?.headers)
      return jsonResponse({ content: [{ type: 'text', text: 'A tiny PNG.' }] })
    },
  })
  const anthropicResult = await anthropic.analyze({ image: visionImageFromBase64('UE5H') })
  expect(anthropicResult.response).toBe('A tiny PNG.')
  expect(anthropicHeaders.get('x-api-key')).toBe('anthropic-key')
  expect(anthropicHeaders.get('Authorization')).toBeNull()
})

test('TTS and transcription use bytes/form data at the provider boundary and return serializable results', async () => {
  let ttsPayload: Record<string, unknown> = {}
  const tts = new OpenAiCompatibleTextToSpeechPort({
    apiKey: 'tts-key',
    baseUrl: 'https://media.test/v1',
    fetchImplementation: async (_input, init) => {
      ttsPayload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return new Response(new Uint8Array([1, 2, 3]), { status: 200, headers: { 'Content-Type': 'audio/mpeg' } })
    },
  })
  const speech = await tts.synthesize({ text: 'hello', voice: 'nova' })
  expect(ttsPayload).toEqual({ model: 'tts-1', input: 'hello', voice: 'nova', format: 'mp3' })
  expect(serializableSynthesizedAudio(speech)).toEqual({
    audio_b64: 'AQID', bytes: 3, format: 'mp3', model: 'tts-1', voice: 'nova',
  })

  let receivedForm: FormData | undefined
  const transcription = new OpenAiCompatibleTranscriptionPort({
    apiKey: 'stt-key',
    baseUrl: 'https://media.test/v1',
    fetchImplementation: async (_input, init) => {
      expect(init?.body).toBeInstanceOf(FormData)
      receivedForm = init?.body as FormData
      return jsonResponse({ text: 'hello from audio' })
    },
  })
  const transcript = await transcription.transcribe({
    audio: { bytes: new Uint8Array([82, 73, 70, 70]), filename: 'clip.wav', mediaType: 'audio/wav' },
    language: 'en',
  })
  expect(receivedForm?.get('model')).toBe('whisper-1')
  expect(receivedForm?.get('language')).toBe('en')
  expect(receivedForm?.get('file')).toBeInstanceOf(File)
  expect(transcript).toEqual({ backend: 'openai-compatible-transcription', model: 'whisper-1', text: 'hello from audio' })
})

test('registered media tool adapters only require JSON-safe payloads', async () => {
  const ports = createOpenAiCompatibleMediaTools({
    apiKey: 'test-key',
    baseUrl: 'https://media.test/v1',
    visionModel: 'configured-vision-model',
    fetchImplementation: async (input) => {
      const path = new URL(input.toString()).pathname
      if (path.endsWith('/images/generations')) return jsonResponse({ data: [{ b64_json: 'UE5H' }] })
      if (path.endsWith('/chat/completions')) return jsonResponse({ choices: [{ message: { content: 'a test image' } }] })
      if (path.endsWith('/audio/speech')) return new Response(new Uint8Array([9, 8]), { status: 200 })
      if (path.endsWith('/audio/transcriptions')) return jsonResponse({ text: 'test transcript' })
      return new Response('', { status: 404 })
    },
  })
  const registry = new ToolRegistry()
  registerMediaTools(registry, ports)
  expect(registry.definitions().map(definition => definition.function.name).sort()).toEqual([
    'image_generate', 'text_to_speech', 'transcribe_audio', 'vision_analyze',
  ])

  const image = JSON.parse(await registry.execute(toolCall('image_generate', { prompt: 'test image' }), { metadata: {} }))
  const vision = JSON.parse(await registry.execute(toolCall('vision_analyze', { image_b64: 'UE5H' }), { metadata: {} }))
  const speech = JSON.parse(await registry.execute(toolCall('text_to_speech', { text: 'hello' }), { metadata: {} }))
  const transcript = JSON.parse(await registry.execute(toolCall('transcribe_audio', { audio_b64: 'UklGRg==' }), { metadata: {} }))
  expect(image.images[0].b64).toBe('UE5H')
  expect(vision.answer).toBe('a test image')
  expect(speech.audio_b64).toBe('CQg=')
  expect(transcript.text).toBe('test transcript')
})

test('vision ports never infer a provider model when neither the port nor request configures one', async () => {
  let requested = false
  const port = new OpenAiCompatibleVisionPort({
    apiKey: 'vision-key',
    baseUrl: 'https://media.test/v1',
    fetchImplementation: async () => {
      requested = true
      return jsonResponse({ choices: [] })
    },
  })

  await expect(port.analyze({ image: visionImageFromBase64('UE5H') }))
    .rejects.toBeInstanceOf(ConfigurationError)
  expect(requested).toBeFalse()
})

test('voice mode requires an injected recorder and coordinates it with a real transcription port', async () => {
  const unavailable = new UnavailableVoiceRecorder()
  expect(() => unavailable.start()).toThrow('no microphone capture adapter')

  const recorder = new FakeRecorder()
  const calls: string[] = []
  const controller = new VoiceModeController({
    recorder,
    transcription: {
      providerName: 'fake-stt',
      async transcribe(request) {
        calls.push(`${request.audio.filename}:${request.audio.bytes.byteLength}`)
        return { backend: 'fake-stt', model: request.model ?? 'fake', text: 'captured text' }
      },
    },
  })
  await controller.start()
  expect(controller.recording).toBe(true)
  await expect(controller.stopAndTranscribe({ model: 'fake-model' })).resolves.toEqual({
    backend: 'fake-stt', model: 'fake-model', text: 'captured text',
  })
  expect(calls).toEqual(['push-to-talk.wav:4'])
  expect(controller.recording).toBe(false)
})

test('media HTTP client rejects response bodies over the configured byte caps', async () => {
  const oversized = (bytes: number) => new Response(new Uint8Array(bytes), { status: 200 })
  const binary = new HttpMediaClient({
    apiKey: 'key',
    baseUrl: 'https://media.test/v1',
    fetchImplementation: async () => oversized(17),
    maxBinaryResponseBytes: 16,
    maxJsonResponseBytes: 16,
  })
  await expect(binary.postBinary('audio/speech', { input: 'hello' }))
    .rejects.toThrow('exceeded the 16-byte limit')

  const json = new HttpMediaClient({
    apiKey: 'key',
    baseUrl: 'https://media.test/v1',
    fetchImplementation: async () => jsonResponse({ text: 'x'.repeat(32) }),
    maxJsonResponseBytes: 16,
  })
  await expect(json.postJson('audio/transcriptions', {}))
    .rejects.toThrow('exceeded the 16-byte limit')

  // A declared content-length over the cap fails before the body is read.
  // The 4-byte body can never trip the 16-byte streaming cap, so a rejection
  // here proves the content-length pre-check fired without consuming it.
  const declared = new HttpMediaClient({
    apiKey: 'key',
    baseUrl: 'https://media.test/v1',
    fetchImplementation: async () =>
      new Response(new ReadableStream({
        pull(controller) {
          controller.enqueue(new Uint8Array(4))
          controller.close()
        },
      }), { status: 200, headers: { 'Content-Length': '64' } }),
    maxBinaryResponseBytes: 16,
  })
  await expect(declared.postBinary('audio/speech', {})).rejects.toThrow('exceeded the 16-byte limit')

  // Bodies within the cap still decode normally.
  const within = new HttpMediaClient({
    apiKey: 'key',
    baseUrl: 'https://media.test/v1',
    fetchImplementation: async () => oversized(3),
    maxBinaryResponseBytes: 16,
  })
  expect(await within.postBinary('audio/speech', {})).toEqual(new Uint8Array(3))
  expect(() => new HttpMediaClient({
    apiKey: 'key',
    baseUrl: 'https://media.test/v1',
    maxBinaryResponseBytes: 0,
  })).toThrow(ConfigurationError)
})

test('media HTTP client rejects parent-directory segments that escape the API path prefix', async () => {
  let endpoint = ''
  const client = new HttpMediaClient({
    apiKey: 'key',
    baseUrl: 'https://media.test/v1',
    fetchImplementation: async input => {
      endpoint = input.toString()
      return jsonResponse({ ok: true })
    },
  })

  await expect(client.postJson('../admin', {})).rejects.toThrow('parent-directory')
  await expect(client.postJson('images/../../admin', {})).rejects.toThrow('parent-directory')
  expect(endpoint).toBe('')

  await client.postJson('images/generations', {})
  expect(endpoint).toBe('https://media.test/v1/images/generations')
})

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), { status: 200, headers: { 'Content-Type': 'application/json' } })
}

function toolCall(name: string, arguments_: Record<string, string>): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name, arguments: arguments_ } }
}

class FakeRecorder implements VoiceRecorder {
  private active = false

  get recording(): boolean {
    return this.active
  }

  start(): void {
    this.active = true
  }

  stop(): { readonly bytes: Uint8Array; readonly filename: string; readonly mediaType: string } {
    this.active = false
    return { bytes: new Uint8Array([82, 73, 70, 70]), filename: 'push-to-talk.wav', mediaType: 'audio/wav' }
  }
}
