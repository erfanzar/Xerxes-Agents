// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { VoiceModelTier } from '../../client/voiceModelManifest.js'
import { TalkButton } from '../components/TalkButton.js'
import { MobileApp } from '../MobileApp.js'
import { AUDIO_PAUSE_LIMIT_MS, CALL_AUDIO_LOST, CALL_AUDIO_PAUSED, CALL_AUDIO_STALLED, CALL_ENDED_HIDDEN, CALL_RECONNECTING, CALL_RESENDING, CALL_SYNTH_STALLED,
  CALL_TRANSCRIBE_STALLED, CALL_WAITING_FOR_HUB, CallEngine, PLAYBACK_GRACE_MS, RESEND_DELAY_MS, TRANSCRIBE_DEADLINE_MS, type CallOutput } from '../voice/call/engine.js'
import { browserCallModels, browserCallOutput, CALL_AUDIO_BLOCKED, LIGHT_NO_VOICE, SYNTHESIZE_JOB_TIMEOUT_MS, TRANSCRIBE_JOB_TIMEOUT_MS } from '../voice/call/browser.js'
import { CALL_FINISHED_WITHOUT_REPLY, CALL_SEND_FAILED, CALL_SEND_REFUSED, CallSendError, REPLY_IDLE_GRACE_MS, waitForCallReply, type CallReply } from '../voice/call/logic.js'
import { voiceStateLine } from '../voice/call/presentation.js'
import { CallStepError, workerFailureMessage, type CallModels, type CallWorkerRequest, type CallWorkerResponse } from '../voice/call/protocol.js'
import { browserLiveCaptureDeps, type CaptureInterruption, type LiveCapture } from '../voice/liveCapture.js'
import { createPushToTalk } from '../voice/pushToTalk.js'
import { liveState, withSupervisorConfig } from './helpers.js'
import { makeServices } from './renderHelpers.js'
import { voiceServiceFixture } from './voiceCallServiceFakes.js'
import { FakeOutput, flush, voiceRig } from './voiceFakes.js'

const frame = (value = 0): Float32Array => new Float32Array(512).fill(value)
const settle = async (): Promise<void> => { for (let i = 0; i < 60; i++) await Promise.resolve() }
const plain = (html: string): string => html.replaceAll('<!-- -->', '').replaceAll('&#x27;', "'")

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers() })

function rig() {
  let latest: { frame: (samples: Float32Array) => void; interrupt: (event: CaptureInterruption) => void } = { frame: () => undefined, interrupt: () => undefined }
  const closes: ReturnType<typeof vi.fn>[] = []
  let resumeResult = true
  const open = vi.fn(async (): Promise<LiveCapture> => {
    const entry = { frame: (_samples: Float32Array): void => undefined, interrupt: (_event: CaptureInterruption): void => undefined }
    latest = entry
    const close = vi.fn()
    closes.push(close)
    return {
      sampleRate: 16_000,
      onFrame: next => { entry.frame = next },
      onInterruption: next => { entry.interrupt = next },
      resume: async () => resumeResult,
      close,
    }
  })
  const model: CallModels = {
    load: vi.fn(async (tier: VoiceModelTier) => ({ tier })),
    vad: vi.fn(async samples => Number(samples[0])), resetVad: vi.fn(),
    transcribe: vi.fn(async () => 'What needs me?'),
    synthesize: vi.fn(async () => ({ samples: frame(), sampleRate: 24_000 })),
    dispose: vi.fn(),
  }
  const output: CallOutput = {
    unlock: vi.fn(async () => undefined),
    play: vi.fn(async (_audio, signal) => { await new Promise<void>(resolve => { signal.addEventListener('abort', () => resolve(), { once: true }) }) }),
    speak: vi.fn(async () => undefined), flush: vi.fn(), close: vi.fn(),
  }
  const reply = vi.fn(async (_text: string, _signal: AbortSignal, _onTimeout: () => void, _messageId: string): Promise<CallReply> => ({ id: 'reply', role: 'xerxes', text: 'First sentence. Second sentence.' }))
  let ids = 0
  const engine = new CallEngine({ capture: { open }, models: () => model, output: () => output, reply, messageId: () => `m${++ids}` })
  const feed = async (count: number, probability: number): Promise<void> => { for (let i = 0; i < count; i++) { latest.frame(frame(probability)); await settle() } }
  const utterance = async (): Promise<void> => { await feed(10, 0.8); await feed(22, 0); await settle() }
  return { engine, model, output, reply, open, closes, feed, utterance, interrupt: (event: CaptureInterruption) => latest.interrupt(event), setResume: (value: boolean) => { resumeResult = value } }
}

describe('V2: a Hub drop or failed send never ends the call', () => {
  it('resends an unconfirmed message once with the same id and keeps the models loaded', async () => {
    vi.useFakeTimers()
    const f = rig()
    f.reply.mockRejectedValueOnce(new CallSendError(CALL_SEND_FAILED, 'failed'))
    await f.engine.start('standard')
    await f.utterance()
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'listening', notice: CALL_RESENDING })
    expect(f.model.dispose).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(RESEND_DELAY_MS)
    await settle()
    expect(f.reply).toHaveBeenCalledTimes(2)
    expect(f.reply.mock.calls[1]![3]).toBe(f.reply.mock.calls[0]![3])
    expect(f.engine.getSnapshot().phase).toBe('speaking')
    f.engine.hangup()
    expect(vi.getTimerCount()).toBe(0)
  })
  it('holds what was said while offline and sends it with the same id when the Hub is back', async () => {
    const f = rig()
    await f.engine.start('standard')
    f.engine.setConnected(false)
    expect(f.engine.getSnapshot().notice).toBe(CALL_RECONNECTING)
    await f.utterance()
    expect(f.reply).not.toHaveBeenCalled()
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'listening', heard: 'What needs me?', notice: CALL_WAITING_FOR_HUB })
    f.engine.setConnected(true)
    await settle()
    expect(f.reply).toHaveBeenCalledOnce()
    expect(f.engine.getSnapshot().phase).toBe('speaking')
    expect(f.closes[0]).not.toHaveBeenCalled()
    f.engine.hangup()
  })
  it('a send that fails while the link is down waits for the reconnect, then reuses its id', async () => {
    const f = rig()
    let fail = (_error: Error): void => undefined
    f.reply.mockImplementationOnce(() => new Promise<CallReply>((_resolve, reject) => { fail = reject }))
    await f.engine.start('standard')
    await f.utterance()
    expect(f.engine.getSnapshot().phase).toBe('thinking')
    f.engine.setConnected(false)
    fail(new CallSendError(CALL_SEND_FAILED, 'failed'))
    await settle()
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'listening', notice: CALL_WAITING_FOR_HUB })
    f.engine.setConnected(true)
    await settle()
    expect(f.reply).toHaveBeenCalledTimes(2)
    expect(f.reply.mock.calls[1]![3]).toBe(f.reply.mock.calls[0]![3])
    f.engine.hangup()
  })
  it('a refused message is shown, not resent, and the call keeps listening', async () => {
    vi.useFakeTimers()
    const f = rig()
    f.reply.mockRejectedValueOnce(new CallSendError(CALL_SEND_REFUSED, 'refused'))
    await f.engine.start('standard')
    await f.utterance()
    await vi.advanceTimersByTimeAsync(RESEND_DELAY_MS * 2)
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'listening', notice: CALL_SEND_REFUSED })
    expect(f.reply).toHaveBeenCalledOnce()
    expect(f.model.dispose).not.toHaveBeenCalled()
    f.engine.hangup()
  })
})

describe('V3: a failed XERXES turn ends the wait at once', () => {
  it('matches the Hub’s "XERXES stopped" notice for this message as the outcome', async () => {
    const messages: CallReply[] = []
    let listener = (): void => undefined
    const result = waitForCallReply({ messageId: 'submit', messages: () => messages, subscribe: next => { listener = next; return () => undefined }, send: async () => true, signal: new AbortController().signal })
    await settle()
    messages.push({ id: 'other', role: 'system', text: 'XERXES stopped: someone else', causalMessageId: 'elsewhere' })
    messages.push({ id: 'stop', role: 'system', text: 'XERXES stopped: the provider failed.', causalMessageId: 'submit' })
    listener()
    await expect(result).resolves.toMatchObject({ id: 'stop', role: 'system' })
  })
  it('ends with a visible step error when the turn goes idle without any reply', async () => {
    vi.useFakeTimers()
    let working: boolean | undefined = true
    let listener = (): void => undefined
    const result = waitForCallReply({ messageId: 'submit', messages: () => [], subscribe: next => { listener = next; return () => undefined }, send: async () => true, signal: new AbortController().signal, working: () => working })
    const assertion = expect(result).rejects.toBeInstanceOf(CallStepError)
    await vi.advanceTimersByTimeAsync(REPLY_IDLE_GRACE_MS * 2)
    working = false
    listener()
    await vi.advanceTimersByTimeAsync(REPLY_IDLE_GRACE_MS - 1)
    working = true
    listener() // working again within the grace: the idle verdict is withdrawn
    await vi.advanceTimersByTimeAsync(REPLY_IDLE_GRACE_MS)
    working = false
    listener()
    await vi.advanceTimersByTimeAsync(REPLY_IDLE_GRACE_MS)
    await assertion
    await expect(result).rejects.toThrow(CALL_FINISHED_WITHOUT_REPLY)
    expect(vi.getTimerCount()).toBe(0)
  })
  it('the call speaks the failure and keeps it on screen', async () => {
    const f = rig()
    f.reply.mockResolvedValueOnce({ id: 'stop', role: 'system', text: 'XERXES stopped: the provider failed.' })
    await f.engine.start('standard')
    await f.utterance()
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'speaking', notice: 'XERXES stopped: the provider failed.' })
    expect(f.model.synthesize).toHaveBeenCalledWith(expect.stringContaining('XERXES stopped'), expect.any(AbortSignal))
    f.engine.hangup()
  })
  it('a finished-without-reply turn returns to listening without unloading the call', async () => {
    const f = rig()
    f.reply.mockRejectedValueOnce(new CallStepError(CALL_FINISHED_WITHOUT_REPLY))
    await f.engine.start('standard')
    await f.utterance()
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'listening', notice: CALL_FINISHED_WITHOUT_REPLY })
    expect(f.model.dispose).not.toHaveBeenCalled()
    f.engine.hangup()
  })
})

describe('V4: microphone and audio start inside the tap', () => {
  it('opens the microphone synchronously in start(), before the model download; a denial ends the call at once', async () => {
    const f = rig()
    let ready = (): void => undefined
    vi.mocked(f.model.load).mockImplementation(async tier => { await new Promise<void>(resolve => { ready = resolve }); return { tier } })
    f.open.mockRejectedValueOnce(new Error('Microphone permission was denied.'))
    const start = f.engine.start('standard')
    expect(f.open).toHaveBeenCalledOnce()
    expect(f.model.load).not.toHaveBeenCalled()
    await settle()
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'error', error: 'Microphone permission was denied.' })
    ready()
    await start
    expect(f.engine.getSnapshot().phase).toBe('error')
  })
})

describe('V5/V1: stuck speech, synthesis or playback never hangs the call', () => {
  it('a playback that never ends (suspended audio) is cut by its watchdog and the call listens again', async () => {
    vi.useFakeTimers()
    const f = rig()
    vi.mocked(f.output.play).mockImplementation(() => new Promise<void>(() => undefined))
    await f.engine.start('standard')
    await f.utterance()
    expect(f.engine.getSnapshot().phase).toBe('speaking')
    await vi.advanceTimersByTimeAsync(PLAYBACK_GRACE_MS + 100)
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'listening', notice: CALL_AUDIO_STALLED })
    expect(f.output.flush).toHaveBeenCalled()
    expect(f.model.dispose).not.toHaveBeenCalled()
    f.engine.hangup()
  })
  it('a transcription that never answers gives up, asks again and the next utterance works', async () => {
    vi.useFakeTimers()
    const f = rig()
    vi.mocked(f.model.transcribe).mockImplementationOnce(() => new Promise<string | null>(() => undefined))
    await f.engine.start('standard')
    await f.utterance()
    expect(f.engine.getSnapshot().phase).toBe('transcribing')
    await vi.advanceTimersByTimeAsync(TRANSCRIBE_DEADLINE_MS)
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'listening', notice: CALL_TRANSCRIBE_STALLED })
    await f.utterance()
    expect(f.reply).toHaveBeenCalledOnce()
    f.engine.hangup()
  })
})

describe('V1/V8: phone interruptions recover or end with a visible reason', () => {
  it('reopens a microphone whose track ended, and after repeated losses ends the call with the reason', async () => {
    const f = rig()
    await f.engine.start('standard')
    f.interrupt('lost')
    await settle()
    expect(f.open).toHaveBeenCalledTimes(2)
    expect(f.closes[0]).toHaveBeenCalledOnce()
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'listening', notice: '' })
    f.interrupt('lost')
    await settle()
    expect(f.open).toHaveBeenCalledTimes(3)
    f.interrupt('lost')
    await settle()
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'idle', endReason: CALL_AUDIO_LOST })
    expect(f.model.dispose).toHaveBeenCalledOnce()
  })
  it('shows a paused notice, clears it on resume, and reopens the mic when the pause lasts too long', async () => {
    vi.useFakeTimers()
    const f = rig()
    await f.engine.start('standard')
    f.interrupt('paused')
    expect(f.engine.getSnapshot().notice).toBe(CALL_AUDIO_PAUSED)
    f.interrupt('resumed')
    expect(f.engine.getSnapshot().notice).toBe('')
    await vi.advanceTimersByTimeAsync(AUDIO_PAUSE_LIMIT_MS)
    expect(f.open).toHaveBeenCalledOnce()
    f.interrupt('paused')
    await vi.advanceTimersByTimeAsync(AUDIO_PAUSE_LIMIT_MS)
    await settle()
    expect(f.open).toHaveBeenCalledTimes(2)
    expect(f.engine.getSnapshot().phase).toBe('listening')
    f.engine.hangup()
  })
  it('rechecks audio when the app is visible again and reopens a capture that cannot resume', async () => {
    const f = rig()
    await f.engine.start('standard')
    f.setResume(false)
    f.engine.visibility(true)
    f.engine.visibility(false)
    await settle()
    expect(f.open).toHaveBeenCalledTimes(2)
    f.engine.hangup()
  })
  it('records why the call ended after 60 s hidden; the service keeps the voice screen open with it', async () => {
    vi.useFakeTimers()
    const f = rig()
    await f.engine.start('standard')
    f.engine.visibility(true)
    await vi.advanceTimersByTimeAsync(60_000)
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'idle', endReason: CALL_ENDED_HIDDEN })
    const s = voiceServiceFixture()
    await s.service.loadStatus()
    await s.service.start()
    s.service.minimize()
    s.emit({ phase: 'idle', endReason: CALL_ENDED_HIDDEN })
    expect(s.service.getSnapshot().view).toBe('voice')
    expect(voiceStateLine(s.service.getSnapshot())).toBe('Voice ended')
    const html = plain(renderToStaticMarkup(<MobileApp services={makeServices(withSupervisorConfig(liveState(), { status: 'configured' }), '/xerxes', { call: s.service })} />))
    expect(html).toContain(CALL_ENDED_HIDDEN)
    expect(html).toContain('>Restart voice</button>')
    s.service.dispose()
  })
  it('the service forwards the Hub link state to the engine', async () => {
    const s = voiceServiceFixture()
    const setConnected = vi.fn()
    Object.assign(s.engine, { setConnected })
    await s.service.loadStatus()
    await s.service.start()
    s.service.setConnected(false)
    expect(setConnected).toHaveBeenLastCalledWith(false)
    expect(s.engine.hangup).not.toHaveBeenCalled()
    s.service.dispose()
  })
})

describe('V10/V11: the microphone stays live while XERXES thinks', () => {
  it('a cough while thinking keeps the pending reply; real new words replace it', async () => {
    const f = rig()
    const signals: AbortSignal[] = []
    f.reply.mockImplementation((_text, signal) => { signals.push(signal); return new Promise<CallReply>(() => undefined) })
    await f.engine.start('standard')
    await f.utterance()
    expect(f.engine.getSnapshot().phase).toBe('thinking')
    await f.feed(10, 0.8)
    expect(f.engine.getSnapshot().phase).toBe('hearing')
    vi.mocked(f.model.transcribe).mockResolvedValueOnce('')
    await f.feed(22, 0)
    await settle()
    expect(f.engine.getSnapshot().phase).toBe('thinking')
    expect(signals[0]?.aborted).toBe(false)
    await f.utterance()
    expect(signals[0]?.aborted).toBe(true)
    expect(f.reply).toHaveBeenCalledTimes(2)
    f.engine.hangup()
  })
  it('a reply that lands while the person talks is held, then spoken when the noise turns out empty', async () => {
    const f = rig()
    let answer = (_reply: CallReply): void => undefined
    f.reply.mockImplementationOnce(() => new Promise<CallReply>(resolve => { answer = resolve }))
    await f.engine.start('standard')
    await f.utterance()
    await f.feed(10, 0.8)
    answer({ id: 'r', role: 'xerxes', text: 'Here is the answer.' })
    await settle()
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'hearing', reply: 'Here is the answer.' })
    expect(f.model.synthesize).not.toHaveBeenCalled()
    vi.mocked(f.model.transcribe).mockResolvedValueOnce('')
    await f.feed(22, 0)
    await settle()
    expect(f.engine.getSnapshot().phase).toBe('speaking')
    expect(f.model.synthesize).toHaveBeenCalledWith('Here is the answer.', expect.any(AbortSignal))
    f.engine.hangup()
  })
})

describe('V12: Light without a local voice is caught before the call', () => {
  it('refuses a chosen Light call up front with the reason', async () => {
    const f = rig()
    f.output.prepare = vi.fn(async () => LIGHT_NO_VOICE)
    await f.engine.start('light')
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'error', error: LIGHT_NO_VOICE })
    expect(f.closes[0]).toHaveBeenCalledOnce()
  })
  it('a fallback to Light without a voice keeps the call as text-only instead of ending it', async () => {
    const f = rig()
    f.output.prepare = vi.fn(async (tier: VoiceModelTier) => tier === 'light' ? LIGHT_NO_VOICE : undefined)
    vi.mocked(f.model.load).mockResolvedValue({ tier: 'light', notice: 'Kokoro could not load.' })
    await f.engine.start('standard')
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'listening', textOnly: true, notice: expect.stringContaining(LIGHT_NO_VOICE) })
    await f.utterance()
    expect(f.output.speak).not.toHaveBeenCalled()
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'listening', spokenSentence: 'First sentence. Second sentence.' })
    f.engine.hangup()
  })
})

class FakeWorker {
  static instances: FakeWorker[] = []
  readonly messages: CallWorkerRequest[] = []
  onmessage: ((event: MessageEvent<CallWorkerResponse>) => void) | undefined
  onerror: ((event: ErrorEvent) => void) | undefined
  terminate = vi.fn()
  constructor(readonly url: string) { FakeWorker.instances.push(this) }
  postMessage(request: CallWorkerRequest): void { this.messages.push(request) }
  reply(message: CallWorkerResponse): void { this.onmessage?.(new MessageEvent('message', { data: message })) }
}

async function loadedModels() {
  FakeWorker.instances = []
  vi.stubGlobal('Worker', FakeWorker)
  const model = browserCallModels()
  const loading = model.load('standard', () => undefined, new AbortController().signal)
  const [vad, speech] = FakeWorker.instances
  vad!.reply({ id: vad!.messages.at(-1)!.id, type: 'loaded', tier: 'standard' })
  await settle()
  speech!.reply({ id: speech!.messages.at(-1)!.id, type: 'loaded', tier: 'standard' })
  await loading
  return model
}

describe('V5/V13: worker deadlines and accurate advice', () => {
  it('only offers Light for a speech-output failure', () => {
    expect(workerFailureMessage('synthesize', 'x')).toContain('Light')
    for (const job of ['load', 'transcribe', 'vad']) {
      expect(workerFailureMessage(job, 'x')).not.toContain('Light')
      expect(workerFailureMessage(job, 'x')).toContain('Reload the app')
    }
  })
  it('a worker that fails to load names the file and does not suggest Light', async () => {
    FakeWorker.instances = []
    vi.useFakeTimers()
    vi.stubGlobal('Worker', FakeWorker)
    const model = browserCallModels()
    const loading = model.load('light', () => undefined, new AbortController().signal)
    FakeWorker.instances[0]!.onerror?.({ preventDefault() {}, message: 'SyntaxError: bad token', filename: 'voice-vad-worker.js' } as ErrorEvent)
    const error = await loading.catch((reason: unknown) => reason)
    expect(String(error)).toContain('voice-vad-worker.js')
    expect(String(error)).toContain('Reload the app')
    expect(String(error)).not.toContain('Light')
    model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
  })
  it('a transcription past its deadline replaces the speech runtime and asks again', async () => {
    vi.useFakeTimers()
    const model = await loadedModels()
    const restarting = vi.fn()
    const pending = model.transcribe(new Float32Array(1600), new AbortController().signal, { restarting, progress: () => undefined })
    const assertion = expect(pending).rejects.toThrow(CALL_TRANSCRIBE_STALLED)
    await vi.advanceTimersByTimeAsync(TRANSCRIBE_JOB_TIMEOUT_MS)
    await settle()
    expect(FakeWorker.instances[1]!.terminate).toHaveBeenCalled()
    const replacement = FakeWorker.instances[2]!
    expect(restarting).toHaveBeenCalledOnce()
    replacement.reply({ id: replacement.messages.at(-1)!.id, type: 'loaded', tier: 'standard' })
    await assertion
    model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
  })
  it('a synthesis past its deadline replaces the speech runtime, even when that speech was interrupted', async () => {
    vi.useFakeTimers()
    const model = await loadedModels()
    const speech = new AbortController()
    const pending = model.synthesize('Hello.', speech.signal)
    const assertion = expect(pending).rejects.toThrow(CALL_SYNTH_STALLED)
    await vi.advanceTimersByTimeAsync(SYNTHESIZE_JOB_TIMEOUT_MS)
    await settle()
    speech.abort()
    const replacement = FakeWorker.instances[2]!
    expect(replacement.messages.at(-1)).toMatchObject({ type: 'load', forceWasm: true })
    expect(replacement.terminate).not.toHaveBeenCalled()
    replacement.reply({ id: replacement.messages.at(-1)!.id, type: 'loaded', tier: 'standard' })
    await assertion
    expect(replacement.terminate).not.toHaveBeenCalled()
    model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
  })
})

describe('browser call output', () => {
  it('reports whether Light can speak before the call starts', async () => {
    vi.stubGlobal('AudioContext', class { state = 'running'; resume = async () => undefined; close = async () => undefined })
    vi.stubGlobal('speechSynthesis', { getVoices: () => [{ localService: false, lang: 'en-US' }], speak: vi.fn(), cancel: vi.fn() })
    expect(await browserCallOutput().prepare?.('light')).toBe(LIGHT_NO_VOICE)
    expect(await browserCallOutput().prepare?.('standard')).toBeUndefined()
    vi.stubGlobal('speechSynthesis', { getVoices: () => [{ localService: true, lang: 'en-GB' }], speak: vi.fn(), cancel: vi.fn() })
    expect(await browserCallOutput().prepare?.('light')).toBeUndefined()
  })
  it('a context the phone keeps paused drops that clip as a step error instead of waiting forever', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('AudioContext', class { state = 'interrupted'; resume = () => new Promise<void>(() => undefined); close = async () => undefined })
    const output = browserCallOutput()
    const playing = output.play({ samples: new Float32Array(512), sampleRate: 16_000 }, new AbortController().signal)
    const assertion = expect(playing).rejects.toThrow(CALL_AUDIO_BLOCKED)
    await vi.advanceTimersByTimeAsync(2000)
    await assertion
    const resumed = output.resume?.()
    await vi.advanceTimersByTimeAsync(2000)
    expect(await resumed).toBe(false)
  })
})

describe('V1: live capture reports phone interruptions', () => {
  it('maps track end/mute and context state changes, resumes, and goes quiet after close', async () => {
    const contextListeners = new Map<string, () => void>()
    const trackListeners = new Map<string, () => void>()
    let contextState = 'running'
    const resume = vi.fn(async () => undefined)
    vi.stubGlobal('AudioContext', class {
      get state(): string { return contextState }
      sampleRate = 48_000
      destination = {}
      audioWorklet = { addModule: async () => undefined }
      resume = resume
      close = async () => undefined
      addEventListener(type: string, listener: () => void): void { contextListeners.set(type, listener) }
      removeEventListener(type: string): void { contextListeners.delete(type) }
      createMediaStreamSource() { return { connect() {}, disconnect() {} } }
      createGain() { return { gain: { value: 1 }, connect() {} } }
    })
    vi.stubGlobal('AudioWorkletNode', class { port = { onmessage: null }; connect() {} })
    const track = { stop: vi.fn(), muted: false, readyState: 'live',
      addEventListener: (type: string, listener: () => void) => { trackListeners.set(type, listener) }, removeEventListener: (type: string) => { trackListeners.delete(type) } }
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [track], getAudioTracks: () => [track] }) } })
    const capture = await browserLiveCaptureDeps()!.open(new AbortController().signal)
    const events: CaptureInterruption[] = []
    capture.onInterruption?.(event => events.push(event))
    trackListeners.get('mute')?.()
    trackListeners.get('unmute')?.()
    contextState = 'interrupted'
    contextListeners.get('statechange')?.()
    await settle()
    contextState = 'running'
    contextListeners.get('statechange')?.()
    trackListeners.get('ended')?.()
    expect(events).toEqual(['paused', 'resumed', 'paused', 'resumed', 'lost'])
    expect(resume).toHaveBeenCalled()
    track.readyState = 'ended'
    expect(await capture.resume?.()).toBe(false)
    capture.close()
    expect(contextListeners.size).toBe(0)
    expect(trackListeners.size).toBe(0)
    expect(track.stop).toHaveBeenCalled()
  })
})

describe('V7/V15: hold-to-talk gestures', () => {
  it('the talk button disables browser panning so a small finger movement does not cancel it', () => {
    const voice = { start: async () => undefined, stop: async () => undefined, cancel: () => undefined }
    const html = renderToStaticMarkup(<TalkButton voice={voice} onTranscript={() => undefined} />)
    expect(html).toContain('touch-action:none')
    expect(html).toContain('Hold to talk')
  })
  it('repeats the audio unlock on the release gesture when the press did not count, and stops once it runs', async () => {
    class GestureOutput extends FakeOutput {
      attempts = 0
      override async unlock(): Promise<void> {
        this.attempts += 1
        if (this.attempts === 1) throw new Error('not a qualifying gesture')
      }
    }
    const listeners = new Map<string, () => void>()
    vi.stubGlobal('addEventListener', (type: string, listener: () => void) => { listeners.set(type, listener) })
    vi.stubGlobal('removeEventListener', (type: string) => { listeners.delete(type) })
    const output = new GestureOutput()
    const r = voiceRig({ output })
    const voice = createPushToTalk({ core: r.core, mic: r.mic, capabilities: r.hub.capabilities, conversationId: () => 'thread-1', levelIntervalMs: 0, newId: () => 'utt-1' })
    await voice.start()
    await flush()
    expect(output.attempts).toBe(1)
    expect(r.core.audioReady).toBe(false)
    expect([...listeners.keys()].sort()).toEqual(['click', 'keyup', 'pointerup', 'touchend'])
    listeners.get('pointerup')?.()
    await flush()
    expect(output.attempts).toBe(2)
    expect(r.core.audioReady).toBe(true)
    expect(listeners.size).toBe(0)
    r.core.unlockAudio()
    expect(output.attempts).toBe(2)
    voice.cancel()
  })
})
