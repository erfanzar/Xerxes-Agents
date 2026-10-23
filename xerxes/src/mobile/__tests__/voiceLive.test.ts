// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { VoiceLink } from '../../client/voiceLink.js'
import { createLiveVoice } from '../voice/live.js'
import { createVoiceControl } from '../voice/index.js'
import { probeVoiceCapabilities, supportsLive, supportsPushToTalk } from '../voice/probe.js'
import { FakeOutput, FakeVoiceHub, caps, deferred, flush, voiceRig, ttsFrameBytes } from './voiceFakes.js'

function live(rig: ReturnType<typeof voiceRig>, ...rest: [conversation?: string | undefined]) {
  // A default parameter would turn an explicit `undefined` into 'thread-1'; check the argument count instead.
  const conversation = rest.length === 0 ? 'thread-1' : rest[0]
  return createLiveVoice({
    core: rig.core, capture: rig.live, capabilities: rig.hub.capabilities, conversationId: () => conversation,
    levelIntervalMs: 0, newId: () => 'live-1',
  })
}

describe('live voice controller', () => {
  it('opens capture, negotiates the capture rate with the Hub, and streams PCM16 frames in order', async () => {
    const rig = voiceRig()
    const control = live(rig)
    await control.start()
    expect(control.active).toBe(true)
    expect(rig.hub.socket.ofType('start')[0]).toMatchObject({ mode: 'live', codec: 'pcm16', sample_rate: 48_000, conversation_id: 'thread-1' })
    rig.live.capture.frames(6, 0.01)
    const frames = rig.hub.socket.frames
    expect(frames.length).toBeGreaterThan(0)
    expect(frames.map(f => f.header.sequence)).toEqual(frames.map((_, index) => index))
    expect(frames.every(f => f.header.codec === 'pcm16' && f.header.generation === 7)).toBe(true)
    // 60 ms messages at 48 kHz are 2880 samples = 5760 bytes.
    expect(frames[0]?.payload.byteLength).toBe(5760)
    expect(rig.state.getSnapshot()).toMatchObject({ live: true, phase: 'listening' })
  })

  it('checks capabilities and the conversation before the microphone', async () => {
    const off = voiceRig({ caps: caps({ modes: ['push_to_talk'], realtime: { available: false } }) })
    await expect(live(off).start()).rejects.toThrow('not available')
    expect(off.live.opens).toBe(0)
    const none = voiceRig()
    await expect(live(none, undefined).start()).rejects.toThrow('XERXES tab')
    expect(none.live.opens).toBe(0)
  })

  it('permission denial fails the orb and leaves no socket open', async () => {
    const rig = voiceRig()
    rig.live.denied = true
    await expect(live(rig).start()).rejects.toThrow('denied')
    expect(rig.state.getSnapshot()).toMatchObject({ phase: 'failed', failure: { code: 'VOICE_DENIED' } })
    expect(rig.hub.sockets).toHaveLength(0)
  })

  it('a grant that arrives after stop is released and nothing is sent', async () => {
    const rig = voiceRig()
    const gate = deferred()
    rig.live.gate = gate.promise
    const control = live(rig)
    const started = control.start().then(() => 'started', (error: unknown) => error)
    await flush()
    control.stop()
    gate.resolve()
    await started
    expect(rig.live.capture.closed).toBeGreaterThan(0)
    expect(rig.hub.sockets).toHaveLength(0)
    expect(control.active).toBe(false)
  })

  it('stop releases the microphone, tells the Hub and closes the socket', async () => {
    const rig = voiceRig()
    const control = live(rig)
    await control.start()
    const socket = rig.hub.socket
    control.stop()
    expect(rig.live.capture.closed).toBe(1)
    expect(socket.ofType('stop')).toHaveLength(1)
    expect(socket.closedWith).toBe(1000)
    expect(rig.state.getSnapshot()).toMatchObject({ live: false, phase: 'idle' })
  })

  it('a dropped socket ends the session with a disconnected failure and releases the microphone', async () => {
    const rig = voiceRig()
    const control = live(rig)
    await control.start()
    rig.hub.socket.drop(1011)
    expect(rig.live.capture.closed).toBe(1)
    expect(control.active).toBe(false)
    expect(rig.state.getSnapshot()).toMatchObject({ phase: 'failed', failure: { code: 'VOICE_DISCONNECTED' } })
  })

  it('a backed-up link ends the session instead of queueing', async () => {
    const rig = voiceRig()
    const control = live(rig)
    await control.start()
    rig.hub.buffered = 10 * 1024 * 1024
    rig.live.capture.frames(6, 0.01)
    expect(rig.state.getSnapshot()).toMatchObject({ phase: 'failed', failure: { code: 'VOICE_OVERLOADED' } })
    expect(rig.live.capture.closed).toBe(1)
  })

  it('speaking over the reply flushes playback locally before the Hub catches up', async () => {
    const output = new FakeOutput()
    const rig = voiceRig({ output })
    const control = live(rig)
    await control.start()
    rig.hub.socket.server({ type: 'audio_start', response_id: 'r1', generation: 9, delivery: 'streaming', codec: 'pcm16', sample_rate: 24_000 })
    rig.hub.socket.serverFrame(ttsFrameBytes(9, 0, new Uint8Array(480), { codec: 'pcm16', clipEnd: true }))
    output.scheduled[0]?.start()
    expect(rig.core.playback.playing).toBe(true)
    rig.live.capture.frames(4, 0.8)
    expect(output.flushes).toBeGreaterThan(0)
    expect(rig.hub.socket.ofType('interrupt')).toHaveLength(1)
    // Quiet audio does not barge in.
    expect(rig.hub.socket.ofType('interrupt')[0]).not.toHaveProperty('agent')
  })
})

describe('capability probe and voice control factory', () => {
  it('returns the Hub capabilities and closes the probe socket', async () => {
    const rig = voiceRig()
    const result = await probeVoiceCapabilities(rig.createLink, 500)
    expect(result?.stt.available).toBe(true)
    expect(rig.hub.socket.closedWith).toBe(1000)
  })

  it('resolves undefined when the Hub refuses, stays silent or reports disabled', async () => {
    const refused = voiceRig()
    refused.hub.refuse = true
    expect(await probeVoiceCapabilities(refused.createLink, 500)).toBeUndefined()
    const silent = voiceRig()
    silent.hub.silent = true
    expect(await probeVoiceCapabilities(silent.createLink, 30)).toBeUndefined()
    expect(supportsPushToTalk(undefined)).toBe(false)
    expect(supportsPushToTalk(caps({ enabled: false }))).toBe(false)
    expect(supportsPushToTalk(caps({ stt: { available: false } }))).toBe(false)
    expect(supportsLive(caps({ modes: ['push_to_talk'] }))).toBe(false)
    expect(supportsLive(caps())).toBe(true)
  })

  it('createVoiceControl yields nothing without a microphone or without Hub capability, and never opens the mic', async () => {
    const hub = new FakeVoiceHub()
    const rig = voiceRig()
    const location = { protocol: 'https:', host: 'hub.test' }
    expect(await createVoiceControl({ location, conversationId: () => 't', socketFactory: hub.factory, mic: undefined })).toBeUndefined()
    hub.capabilities = caps({ enabled: false })
    expect(await createVoiceControl({ location, conversationId: () => 't', socketFactory: hub.factory, mic: rig.mic, probeTimeoutMs: 200 })).toBeUndefined()
    expect(rig.mic.opened).toBe(0)
    hub.capabilities = caps()
    const control = await createVoiceControl({ location, conversationId: () => 't', socketFactory: hub.factory, mic: rig.mic, output: undefined, liveCapture: rig.live, probeTimeoutMs: 200 })
    expect(control).toBeDefined()
    expect(control?.live).toBeDefined()
    expect(rig.mic.opened).toBe(0)
    const pttOnly = await createVoiceControl({ location, conversationId: () => 't', socketFactory: hub.factory, mic: rig.mic, output: undefined, liveCapture: undefined, probeTimeoutMs: 200 })
    expect(pttOnly?.live).toBeUndefined()
  })
})
