// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { ConfigurationError } from '../src/core/errors.js'
import { buildCapabilities } from '../src/hub/voice/config.js'
import type { RealtimeEvent } from '../src/hub/voice/ports.js'
import { VoiceError } from '../src/hub/voice/protocol.js'
import { GeminiLivePort } from '../src/hub/voice/providers/geminiLive.js'
import { OpenAiRealtimePort } from '../src/hub/voice/providers/openaiRealtime.js'
import type { RealtimeSocket, RealtimeSocketFactory, RealtimeSocketHandlers, RealtimeSocketRequest } from '../src/hub/voice/providers/realtimeSocket.js'
import { createVoiceRuntimeConfig } from '../src/hub/voice/providerFactory.js'

type Json = Record<string, unknown>

class FakeProviderSocket implements RealtimeSocket {
  readonly sent: Json[] = []
  closed = false
  buffered = 0
  handlers!: RealtimeSocketHandlers
  request!: RealtimeSocketRequest
  onSend: ((message: Json) => void) | undefined
  send(data: string): boolean {
    if (this.closed) return false
    const message = JSON.parse(data) as Json
    this.sent.push(message)
    this.onSend?.(message)
    return true
  }
  bufferedAmount(): number { return this.buffered }
  close(): void { this.closed = true }
  server(message: Json): void { this.handlers.message(JSON.stringify(message)) }
  ofType(type: string): Json[] { return this.sent.filter(message => message.type === type) }
}

function factoryFor(fake: FakeProviderSocket, mode: 'open' | 'refuse' = 'open'): RealtimeSocketFactory {
  return (request, handlers) => {
    fake.request = request
    fake.handlers = handlers
    queueMicrotask(() => { if (mode === 'open') handlers.open(); else handlers.close(1006, 'denied sk-secret') })
    return fake
  }
}

const SIGNAL = (): AbortSignal => new AbortController().signal
const TOOLS = [{ name: 'supervisor_status', description: 'status', parameters: { type: 'object', properties: {} } }]
const b64 = (bytes: number[]): string => Buffer.from(bytes).toString('base64')

function collect(session: { onEvent(listener: (event: RealtimeEvent) => void): () => void }): RealtimeEvent[] {
  const events: RealtimeEvent[] = []
  session.onEvent(event => events.push(event))
  return events
}

describe('OpenAI Realtime adapter (fake socket)', () => {
  const connect = async (extra: Partial<ConstructorParameters<typeof OpenAiRealtimePort>[0]> = {}) => {
    const fake = new FakeProviderSocket()
    const port = new OpenAiRealtimePort({ apiKey: 'sk-secret', model: 'rt-model', socketFactory: factoryFor(fake), ...extra })
    const session = await port.connect({ instructions: 'be brief', tools: TOOLS, sampleRate: 24_000, language: 'en' }, SIGNAL())
    return { fake, port, session, events: collect(session) }
  }

  test('keeps the key in the socket header only and configures server VAD, pcm16 and the three tools', async () => {
    const { fake, port } = await connect()
    expect(fake.request.headers.Authorization).toBe('Bearer sk-secret')
    expect(fake.request.url).toContain('model=rt-model')
    expect(fake.request.url).not.toContain('sk-secret')
    expect(port.sampleRate).toBe(24_000)
    const update = fake.ofType('session.update')[0] as { session: Json }
    expect(JSON.stringify(update)).not.toContain('sk-secret')
    expect(update.session).toMatchObject({ input_audio_format: 'pcm16', output_audio_format: 'pcm16', instructions: 'be brief' })
    expect((update.session.turn_detection as Json).type).toBe('server_vad')
    expect((update.session.tools as Json[]).map(tool => tool.name)).toEqual(['supervisor_status'])
  })

  test('maps provider events: VAD, transcript, audio, tool call, usage', async () => {
    const { fake, events } = await connect()
    fake.server({ type: 'session.updated' })
    fake.server({ type: 'input_audio_buffer.speech_started' })
    fake.server({ type: 'input_audio_buffer.speech_stopped', item_id: 'i1' })
    fake.server({ type: 'conversation.item.input_audio_transcription.completed', item_id: 'i1', transcript: '  run tests ' })
    fake.server({ type: 'response.audio.delta', response_id: 'r1', item_id: 'a1', delta: b64([1, 2, 3, 4]) })
    fake.server({ type: 'response.audio.done', response_id: 'r1' })
    fake.server({ type: 'response.function_call_arguments.done', call_id: 'c1', name: 'supervisor_status', arguments: '{}' })
    fake.server({ type: 'response.done', response: { usage: { input_tokens: 10, output_tokens: 5 } } })
    expect(events.map(event => event.type)).toEqual(['ready', 'speech_started', 'speech_stopped', 'transcript', 'audio', 'audio_done', 'tool_call', 'usage'])
    expect(events[3]).toMatchObject({ itemId: 'i1', text: 'run tests', final: true })
    const audio = events[4] as Extract<RealtimeEvent, { type: 'audio' }>
    expect([...audio.audio]).toEqual([1, 2, 3, 4])
    expect(audio).toMatchObject({ responseId: 'r1', itemId: 'a1' })
    expect(events[7]).toMatchObject({ inputTokens: 10, outputTokens: 5 })
  })

  test('malformed audio and unknown events are ignored; provider error text is replaced by a fixed message', async () => {
    const { fake, events } = await connect()
    fake.handlers.message('not json')
    fake.server({ type: 'response.audio.delta', response_id: 'r1', delta: '' })
    fake.server({ type: 'rate_limits.updated' })
    fake.server({ type: 'error', error: { message: 'bad request echoing sk-secret' } })
    expect(events).toHaveLength(1)
    const error = events[0] as Extract<RealtimeEvent, { type: 'error' }>
    expect(error.error).toBeInstanceOf(VoiceError)
    expect(error.error.code).toBe('VOICE_PROVIDER')
    expect(error.error.message).not.toContain('sk-secret')
  })

  test('sends audio, cancels, truncates to the played offset, returns tool results and speaks supervisor text', async () => {
    const { fake, session } = await connect()
    expect(session.supportsTruncate).toBe(true)
    expect(session.sendAudio(new Uint8Array([1, 2, 3]))).toBe(true)
    expect(fake.ofType('input_audio_buffer.append')[0]).toMatchObject({ audio: 'AQID' })
    session.cancelResponse()
    session.truncate('a1', 120.9)
    session.sendToolResult('c1', '{"ok":true}')
    session.say('All quiet.')
    expect(fake.ofType('response.cancel')).toHaveLength(1)
    expect(fake.ofType('conversation.item.truncate')[0]).toMatchObject({ item_id: 'a1', audio_end_ms: 120 })
    expect(fake.ofType('conversation.item.create')[0]).toMatchObject({ item: { type: 'function_call_output', call_id: 'c1', output: '{"ok":true}' } })
    const creates = fake.ofType('response.create')
    expect(creates).toHaveLength(2)
    expect(JSON.stringify(creates[1])).toContain('All quiet.')
    session.close()
    expect(fake.closed).toBe(true)
  })

  test('a backed-up socket makes sendAudio report failure', async () => {
    const { fake, session } = await connect({ maxBufferedBytes: 10 })
    fake.buffered = 100
    expect(session.sendAudio(new Uint8Array([1]))).toBe(false)
    expect(fake.ofType('input_audio_buffer.append')).toHaveLength(0)
  })

  test('a refused connection is a typed provider error without the provider reason; an aborted one is cancelled', async () => {
    const refused = new OpenAiRealtimePort({ apiKey: 'sk-secret', socketFactory: factoryFor(new FakeProviderSocket(), 'refuse') })
    const error = await refused.connect({ instructions: '', tools: [], sampleRate: 24_000 }, SIGNAL()).catch((reason: unknown) => reason)
    expect(error).toBeInstanceOf(VoiceError)
    expect((error as VoiceError).code).toBe('VOICE_PROVIDER')
    expect((error as VoiceError).message).not.toContain('sk-secret')

    const controller = new AbortController()
    controller.abort()
    const aborted = new OpenAiRealtimePort({ apiKey: 'k', socketFactory: factoryFor(new FakeProviderSocket()) })
    const cancelled = await aborted.connect({ instructions: '', tools: [], sampleRate: 24_000 }, controller.signal).catch((reason: unknown) => reason)
    expect((cancelled as VoiceError).code).toBe('VOICE_CANCELLED')
  })

  test('a connection that never opens times out as a provider error', async () => {
    const port = new OpenAiRealtimePort({ apiKey: 'k', connectTimeoutMs: 20, socketFactory: (_request, _handlers) => new FakeProviderSocket() })
    const error = await port.connect({ instructions: '', tools: [], sampleRate: 24_000 }, SIGNAL()).catch((reason: unknown) => reason)
    expect((error as VoiceError).code).toBe('VOICE_PROVIDER')
  })

  test('provider close after open is reported once as closed', async () => {
    const { fake, events } = await connect()
    fake.handlers.close(1000, 'bye')
    expect(events.at(-1)?.type).toBe('closed')
  })

  test('configuration is validated', () => {
    expect(() => new OpenAiRealtimePort({ apiKey: '  ' })).toThrow(ConfigurationError)
    expect(() => new OpenAiRealtimePort({ apiKey: 'k', baseUrl: 'https://x.test' })).toThrow('ws or wss')
    expect(() => new OpenAiRealtimePort({ apiKey: 'k', baseUrl: 'nope' })).toThrow(ConfigurationError)
    expect(() => new OpenAiRealtimePort({ apiKey: 'k', vadSilenceMs: 10 })).toThrow('100 to 3000')
  })
})

describe('Gemini Live adapter (fake socket)', () => {
  const connect = async () => {
    const fake = new FakeProviderSocket()
    fake.onSend = message => { if ('setup' in message) queueMicrotask(() => fake.server({ setupComplete: {} })) }
    const port = new GeminiLivePort({ apiKey: 'g-secret', model: 'models/gem-live', socketFactory: factoryFor(fake) })
    const session = await port.connect({ instructions: 'be brief', tools: TOOLS, sampleRate: 16_000 }, SIGNAL())
    const events = collect(session)
    // `ready` fired on setupComplete before any listener existed; the emitter replays it first.
    expect(events.map(event => event.type)).toEqual(['ready'])
    events.length = 0
    return { fake, port, session, events }
  }

  test('the key travels in a header, never in the URL or the setup message', async () => {
    const { fake, port } = await connect()
    expect(fake.request.headers['x-goog-api-key']).toBe('g-secret')
    expect(fake.request.url).not.toContain('g-secret')
    expect(JSON.stringify(fake.sent)).not.toContain('g-secret')
    const setup = (fake.sent[0] as { setup: Json }).setup
    expect(setup.model).toBe('models/gem-live')
    expect(JSON.stringify(setup.tools)).toContain('supervisor_status')
    expect(port.sampleRate).toBe(16_000)
    expect(port.outputSampleRate).toBe(24_000)
  })

  test('synthesizes ids; finalizes the transcript when the model starts answering; reports turn end', async () => {
    const { fake, events } = await connect()
    fake.server({ serverContent: { inputTranscription: { text: 'run the ' } } })
    fake.server({ serverContent: { inputTranscription: { text: 'tests' } } })
    fake.server({ serverContent: { modelTurn: { parts: [{ inlineData: { data: b64([9, 9]), mimeType: 'audio/pcm;rate=24000' } }] } } })
    fake.server({ serverContent: { turnComplete: true } })
    fake.server({ serverContent: { modelTurn: { parts: [{ inlineData: { data: b64([8]), mimeType: 'audio/pcm;rate=24000' } }] } } })
    const summary = events.map(event => (event.type === 'transcript' ? `transcript:${event.text}:${event.final}` : event.type === 'audio' ? `audio:${event.responseId}` : event.type))
    expect(summary).toEqual(['speech_started', 'transcript:run the:false', 'transcript:run the tests:false', 'transcript:run the tests:true', 'audio:resp-1', 'audio_done', 'audio:resp-2'])
  })

  test('cancelResponse stops relaying the current turn; it cannot truncate and says so', async () => {
    const { fake, session, events } = await connect()
    expect(session.supportsTruncate).toBe(false)
    session.cancelResponse()
    session.truncate('item', 100)
    fake.server({ serverContent: { modelTurn: { parts: [{ inlineData: { data: b64([1]) } }] } } })
    fake.server({ serverContent: { turnComplete: true } })
    expect(events.filter(event => event.type === 'audio' || event.type === 'audio_done')).toHaveLength(0)
    fake.server({ serverContent: { modelTurn: { parts: [{ inlineData: { data: b64([1]) } }] } } })
    expect(events.filter(event => event.type === 'audio')).toHaveLength(1)
  })

  test('tool calls and results use the provider call id and name', async () => {
    const { fake, session, events } = await connect()
    fake.server({ toolCall: { functionCalls: [{ id: 'f1', name: 'supervisor_status', args: {} }] } })
    expect(events[0]).toMatchObject({ type: 'tool_call', callId: 'f1', name: 'supervisor_status', arguments: '{}' })
    session.sendToolResult('f1', '{"ok":true}')
    expect(fake.sent.at(-1)).toMatchObject({ toolResponse: { functionResponses: [{ id: 'f1', name: 'supervisor_status', response: { output: '{"ok":true}' } }] } })
  })

  test('sends 16 kHz audio with its rate, relays say(), and reports a backed-up socket', async () => {
    const { fake, session } = await connect()
    expect(session.sendAudio(new Uint8Array([1, 2]))).toBe(true)
    expect(fake.sent.at(-1)).toMatchObject({ realtimeInput: { audio: { mimeType: 'audio/pcm;rate=16000' } } })
    session.say('Done.')
    expect(JSON.stringify(fake.sent.at(-1))).toContain('Done.')
    fake.buffered = 10_000_000
    expect(session.sendAudio(new Uint8Array([1]))).toBe(false)
  })

  test('provider errors become a fixed message; setup that never completes times out', async () => {
    const { fake, events } = await connect()
    fake.server({ error: { message: 'echo g-secret' } })
    const error = events[0] as Extract<RealtimeEvent, { type: 'error' }>
    expect(error.error.message).not.toContain('g-secret')

    const silent = new FakeProviderSocket()
    const port = new GeminiLivePort({ apiKey: 'k', connectTimeoutMs: 20, socketFactory: factoryFor(silent) })
    const failure = await port.connect({ instructions: '', tools: [], sampleRate: 16_000 }, SIGNAL()).catch((reason: unknown) => reason)
    expect((failure as VoiceError).code).toBe('VOICE_PROVIDER')
    expect(silent.closed).toBe(true)
  })

  test('configuration is validated', () => {
    expect(() => new GeminiLivePort({ apiKey: '' })).toThrow(ConfigurationError)
    expect(() => new GeminiLivePort({ apiKey: 'k', baseUrl: 'wss://x.test/ws?key=abc' })).toThrow('credentials go in headers')
    expect(() => new GeminiLivePort({ apiKey: 'k', baseUrl: 'https://x.test' })).toThrow('ws or wss')
  })
})

describe('provider factory: explicit configuration only', () => {
  test('nothing configured and voice disabled yields no ports and a disabled capability reply', () => {
    const config = createVoiceRuntimeConfig({ env: { XERXES_VOICE_STT_API_KEY: 'k', XERXES_HUB_VOICE_STT_BASE_URL: 'https://stt.example.test/v1' } })
    expect(config.settings.enabled).toBe(false)
    expect(config.stt).toBeUndefined()
    const caps = buildCapabilities(config)
    expect(caps).toMatchObject({ enabled: false, modes: [], audio_retention: 'none' })
    expect(caps.stt).toEqual({ available: false, reason: 'voice is disabled on this hub' })
  })

  test('enabled without providers is unavailable with reasons, and nothing is substituted', () => {
    const config = createVoiceRuntimeConfig({ settings: { enabled: true } })
    const caps = buildCapabilities(config)
    expect(caps.enabled).toBe(false)
    expect(caps.stt).toMatchObject({ available: false, reason: 'not configured' })
    expect(caps.tts).toMatchObject({ available: false })
    expect(caps.realtime).toMatchObject({ available: false })
  })

  test('STT from settings plus an environment credential; capability reply carries no secret or URL', async () => {
    const seen: { url: string; auth: string | undefined }[] = []
    const config = createVoiceRuntimeConfig({
      settings: { enabled: true, stt: { base_url: 'https://stt.example.test/v1', model: 'whisper-x' } },
      env: { XERXES_VOICE_STT_API_KEY: 'stt-secret-key' },
      fetchImplementation: async (input, init) => {
        seen.push({ url: String(input), auth: (init?.headers as Record<string, string>).Authorization })
        return new Response(JSON.stringify({ text: ' hi there ' }), { status: 200, headers: { 'content-type': 'application/json' } })
      },
    })
    expect(config.stt).toBeDefined()
    const result = await config.stt!.transcribe({ audio: new Uint8Array([1, 2]), mediaType: 'audio/webm;codecs=opus' }, SIGNAL())
    expect(result.text).toBe('hi there')
    expect(seen[0]).toEqual({ url: 'https://stt.example.test/v1/audio/transcriptions', auth: 'Bearer stt-secret-key' })
    const caps = buildCapabilities(config)
    expect(caps).toMatchObject({ enabled: true, modes: ['push_to_talk'], stt: { available: true } })
    const text = JSON.stringify(caps)
    expect(text).not.toContain('stt-secret-key')
    expect(text).not.toContain('stt.example.test')
  })

  test('settings win over the environment; environment keys alone can configure a section', () => {
    const fromEnv = createVoiceRuntimeConfig({
      env: {
        XERXES_HUB_VOICE_ENABLED: 'true',
        XERXES_HUB_VOICE_STT_BASE_URL: 'https://stt.example.test/v1',
        XERXES_HUB_VOICE_TTS_BASE_URL: 'https://tts.example.test/v1',
        XERXES_HUB_VOICE_MAX_SESSION_USD: '0.5',
        XERXES_VOICE_STT_API_KEY: 'a',
        XERXES_VOICE_TTS_API_KEY: 'b',
      },
    })
    expect(fromEnv.settings).toMatchObject({ enabled: true, maxSessionUsd: 0.5 })
    expect(fromEnv.stt).toBeDefined()
    expect(fromEnv.tts?.delivery).toBe('buffered')
    const override = createVoiceRuntimeConfig({ settings: { enabled: false }, env: { XERXES_HUB_VOICE_ENABLED: 'true' } })
    expect(override.settings.enabled).toBe(false)
  })

  test('missing credentials, unknown providers and missing base_url are reported, not papered over', () => {
    const config = createVoiceRuntimeConfig({
      settings: {
        enabled: true,
        stt: { base_url: 'https://stt.example.test/v1' },
        tts: { provider: 'elevenlabs', base_url: 'https://tts.example.test' },
        realtime: { provider: 'azure', model: 'x' },
      },
      env: { XERXES_VOICE_REALTIME_API_KEY: 'k' },
    })
    expect(config.stt).toBeUndefined()
    expect(config.tts).toBeUndefined()
    expect(config.realtime).toBeUndefined()
    expect(config.reasons).toEqual({ stt: 'missing credentials', tts: 'unsupported speech provider', realtime: 'unsupported realtime provider' })
    expect(createVoiceRuntimeConfig({ settings: { enabled: true, stt: { model: 'm' } } }).reasons?.stt).toBe('missing base_url')
    expect(createVoiceRuntimeConfig({ settings: { enabled: true, realtime: { model: 'm' } }, env: { XERXES_VOICE_REALTIME_API_KEY: 'k' } }).reasons?.realtime).toBe('missing provider')
  })

  test('a named credential variable is honoured; an unauthenticated gateway needs an explicit opt-in', () => {
    const named = createVoiceRuntimeConfig({ settings: { enabled: true, stt: { base_url: 'https://stt.example.test/v1', api_key_env: 'MY_STT_KEY' } }, env: { MY_STT_KEY: 'k', XERXES_VOICE_STT_API_KEY: '' } })
    expect(named.stt).toBeDefined()
    const open = createVoiceRuntimeConfig({ settings: { enabled: true, stt: { base_url: 'http://127.0.0.1:9000/v1', allow_unauthenticated: true } } })
    expect(open.stt).toBeDefined()
  })

  test('realtime picks the named provider; neither is a fallback for the other', () => {
    const env = { XERXES_VOICE_REALTIME_API_KEY: 'rt-key' }
    const openai = createVoiceRuntimeConfig({ settings: { enabled: true, mode: 'both', realtime: { provider: 'openai' } }, env })
    const gemini = createVoiceRuntimeConfig({ settings: { enabled: true, mode: 'live', realtime: { provider: 'gemini' } }, env })
    expect(openai.realtime?.providerName).toBe('openai-realtime')
    expect(gemini.realtime?.providerName).toBe('gemini-live')
    expect(buildCapabilities(gemini)).toMatchObject({ modes: ['live'], realtime: { available: true, provider: 'gemini-live' } })
    expect(JSON.stringify(buildCapabilities(openai))).not.toContain('rt-key')
  })

  test('literal credentials and bad values are refused at the boundary', () => {
    expect(() => createVoiceRuntimeConfig({ settings: { enabled: true, stt: { base_url: 'https://x.test', api_key: 'sk-live' } } })).toThrow('credentials must not be stored in settings')
    expect(() => createVoiceRuntimeConfig({ settings: { audio_retention: 'full' } })).toThrow('audio is never stored')
    expect(() => createVoiceRuntimeConfig({ settings: { mode: 'always' } })).toThrow(ConfigurationError)
    expect(() => createVoiceRuntimeConfig({ settings: { max_session_usd: -1 } })).toThrow(ConfigurationError)
    expect(() => createVoiceRuntimeConfig({ settings: { language: 'not a tag' } })).toThrow(ConfigurationError)
    expect(() => createVoiceRuntimeConfig({ settings: 'nope' })).toThrow('must be an object')
    expect(createVoiceRuntimeConfig({ settings: { audio_retention: 'none' } }).settings.audioRetention).toBe('none')
  })

  test('pricing and limits flow into the settings', () => {
    const config = createVoiceRuntimeConfig({
      settings: { enabled: true, max_session_usd: 1, max_daily_user_usd: 5, max_utterance_seconds: 30, pricing: { stt_usd_per_minute: 0.006 }, tts: { format: 'opus', base_url: 'https://t.test', allow_unauthenticated: true } },
    })
    expect(config.settings).toMatchObject({ maxSessionUsd: 1, maxDailyUserUsd: 5, maxUtteranceSeconds: 30, ttsFormat: 'opus', pricing: { sttUsdPerMinute: 0.006 } })
    expect(buildCapabilities(config).tts).toMatchObject({ available: true, delivery: 'buffered', format: 'opus' })
  })
})
