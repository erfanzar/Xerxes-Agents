// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { PlaybackError, ResponsePlayback, createWebAudioOutput, type AudioContextLike } from '../voice/playback.js'
import { FakeOutput, flush, ttsFrame, ttsFrameBytes, voiceRig } from './voiceFakes.js'

function playbackRig() {
  const output = new FakeOutput()
  const events: string[] = []
  const playback = new ResponsePlayback(output, {
    started: (responseId, generation) => { events.push(`started:${responseId}:${generation}`) },
    ended: (responseId, generation) => { events.push(`ended:${responseId}:${generation}`) },
    failed: (responseId, _generation, error) => { events.push(`failed:${responseId}:${error.kind}`) },
  })
  return { output, events, playback }
}

describe('ResponsePlayback', () => {
  it('accepts frames only for the current response and generation, in order', () => {
    const { playback, output } = playbackRig()
    expect(playback.frame(ttsFrame(1, 0, [1], { clipEnd: true }))).toBe(false)
    playback.start({ responseId: 'r1', generation: 1, codec: 'mp3' })
    expect(playback.frame(ttsFrame(2, 0, [1], { clipEnd: true }))).toBe(false)
    expect(playback.frame(ttsFrame(1, 0, [1], { clipEnd: true }))).toBe(true)
    // Late and duplicate frames are refused.
    expect(playback.frame(ttsFrame(1, 0, [1], { clipEnd: true }))).toBe(false)
    expect(output.scheduled).toHaveLength(1)
  })

  it('assembles a container clip from several frames before scheduling it', () => {
    const { playback, output } = playbackRig()
    playback.start({ responseId: 'r1', generation: 3, codec: 'mp3' })
    playback.frame(ttsFrame(3, 0, [1, 2]))
    playback.frame(ttsFrame(3, 1, [3]))
    expect(output.scheduled).toHaveLength(0)
    playback.frame(ttsFrame(3, 2, [], { clipEnd: true }))
    expect(output.scheduled).toHaveLength(1)
    expect([...output.scheduled[0]!.clip.bytes]).toEqual([1, 2, 3])
    expect(output.scheduled[0]?.clip.codec).toBe('mp3')
  })

  it('schedules every PCM16 frame with the negotiated sample rate', () => {
    const { playback, output } = playbackRig()
    playback.start({ responseId: 'r1', generation: 1, codec: 'pcm16', sampleRate: 24_000 })
    playback.frame(ttsFrame(1, 0, [0, 0, 1, 1], { codec: 'pcm16', clipEnd: true }))
    playback.frame(ttsFrame(1, 1, [], { codec: 'pcm16', clipEnd: true }))
    expect(output.scheduled).toHaveLength(1)
    expect(output.scheduled[0]?.clip).toMatchObject({ codec: 'pcm16', sampleRate: 24_000 })
  })

  it('reports started on the first sound and ended only after the stream ended and the last clip finished', async () => {
    const { playback, output, events } = playbackRig()
    playback.start({ responseId: 'r1', generation: 1, codec: 'mp3' })
    playback.frame(ttsFrame(1, 0, [1], { clipEnd: true }))
    playback.frame(ttsFrame(1, 1, [2], { clipEnd: true }))
    output.scheduled[0]?.start()
    output.scheduled[0]?.start()
    expect(events).toEqual(['started:r1:1'])
    expect(playback.playing).toBe(true)
    output.scheduled[0]?.finish()
    output.scheduled[1]?.finish()
    await flush()
    // The Hub has not said it is done: more audio could still arrive.
    expect(events).toEqual(['started:r1:1'])
    playback.end({ responseId: 'r1', generation: 1, interrupted: false })
    expect(events).toEqual(['started:r1:1', 'ended:r1:1'])
    expect(playback.activeResponse).toBeUndefined()
  })

  it('does not report ended for a response that never sounded', async () => {
    const { playback, output, events } = playbackRig()
    playback.start({ responseId: 'r1', generation: 1, codec: 'mp3' })
    playback.end({ responseId: 'r1', generation: 1, interrupted: false })
    expect(events).toEqual([])
    expect(output.scheduled).toHaveLength(0)
    expect(playback.activeResponse).toBeUndefined()
  })

  it('interrupt flushes the device, aborts what was scheduled and rejects later frames and events', async () => {
    const { playback, output, events } = playbackRig()
    playback.start({ responseId: 'r1', generation: 1, codec: 'mp3' })
    playback.frame(ttsFrame(1, 0, [1], { clipEnd: true }))
    const scheduled = output.scheduled[0]!
    playback.interrupt()
    expect(output.flushes).toBe(1)
    expect(scheduled.signal.aborted).toBe(true)
    expect(playback.frame(ttsFrame(1, 1, [2], { clipEnd: true }))).toBe(false)
    scheduled.start()
    scheduled.finish()
    playback.end({ responseId: 'r1', generation: 1, interrupted: false })
    await flush()
    expect(events).toEqual([])
    // Interrupting with nothing playing is harmless.
    playback.interrupt()
    expect(output.flushes).toBe(1)
  })

  it('an interrupted end from the Hub flushes, and an end for another response is ignored', () => {
    const { playback, output } = playbackRig()
    playback.start({ responseId: 'r2', generation: 2, codec: 'mp3' })
    playback.end({ responseId: 'r1', generation: 2, interrupted: true })
    playback.end({ responseId: 'r2', generation: 1, interrupted: true })
    expect(output.flushes).toBe(0)
    playback.end({ responseId: 'r2', generation: 2, interrupted: true })
    expect(output.flushes).toBe(1)
  })

  it('a new response replaces the old one and flushes it', () => {
    const { playback, output } = playbackRig()
    playback.start({ responseId: 'r1', generation: 1, codec: 'mp3' })
    playback.start({ responseId: 'r2', generation: 2, codec: 'mp3' })
    expect(output.flushes).toBe(1)
    expect(playback.activeResponse).toEqual({ responseId: 'r2', generation: 2 })
    expect(playback.frame(ttsFrame(1, 0, [1], { clipEnd: true }))).toBe(false)
  })

  it('an out-of-order frame fails the response and flushes', () => {
    const { playback, output, events } = playbackRig()
    playback.start({ responseId: 'r1', generation: 1, codec: 'mp3' })
    expect(playback.frame(ttsFrame(1, 3, [1], { clipEnd: true }))).toBe(false)
    expect(events).toEqual(['failed:r1:decode'])
    expect(output.flushes).toBe(1)
    expect(playback.activeResponse).toBeUndefined()
  })

  it('a browser that blocks sound (autoplay) fails the response with a typed error', async () => {
    const { playback, output, events } = playbackRig()
    playback.start({ responseId: 'r1', generation: 1, codec: 'mp3' })
    playback.frame(ttsFrame(1, 0, [1], { clipEnd: true }))
    output.scheduled[0]?.fail(new PlaybackError('blocked', 'blocked'))
    await flush()
    expect(events).toEqual(['failed:r1:blocked'])
    expect(output.flushes).toBe(1)
    playback.end({ responseId: 'r1', generation: 1, interrupted: false })
    expect(events).toHaveLength(1)
  })

  it('any other device failure becomes a decode error', async () => {
    const { playback, output, events } = playbackRig()
    playback.start({ responseId: 'r1', generation: 1, codec: 'mp3' })
    playback.frame(ttsFrame(1, 0, [1], { clipEnd: true }))
    output.scheduled[0]?.fail(new Error('device exploded'))
    await flush()
    expect(events).toEqual(['failed:r1:decode'])
  })
})

describe('Hub speech through the voice client core', () => {
  async function connected() {
    const output = new FakeOutput()
    const rig = voiceRig({ output })
    await rig.core.ensureLink()
    return { ...rig, output }
  }

  it('plays a buffered clip and ends the orb when playback ends, not when the stream does', async () => {
    const { hub, output, state } = await connected()
    hub.socket.server({ type: 'audio_start', response_id: 'r1', generation: 3, delivery: 'buffered', codec: 'mp3' })
    expect(state.getSnapshot().phase).toBe('buffering')
    hub.socket.serverFrame(ttsFrameBytes(3, 0, [1, 2]))
    hub.socket.serverFrame(ttsFrameBytes(3, 1, [3], { clipEnd: true, final: true }))
    expect(output.scheduled).toHaveLength(1)
    output.scheduled[0]?.start()
    expect(state.getSnapshot().phase).toBe('speaking')
    hub.socket.server({ type: 'audio_end', response_id: 'r1', generation: 3, interrupted: false })
    expect(state.getSnapshot().phase).toBe('speaking')
    output.scheduled[0]?.finish()
    await flush()
    expect(state.getSnapshot()).toMatchObject({ phase: 'idle', responseId: undefined })
  })

  it('interrupt flushes playback, tells the Hub, and late audio for that response is stale', async () => {
    const { hub, output, state, core } = await connected()
    hub.socket.server({ type: 'audio_start', response_id: 'r1', generation: 3, delivery: 'buffered', codec: 'mp3' })
    hub.socket.serverFrame(ttsFrameBytes(3, 0, [1], { clipEnd: true }))
    const first = output.scheduled[0]!
    core.interruptPlayback()
    expect(output.flushes).toBe(1)
    expect(first.signal.aborted).toBe(true)
    expect(hub.socket.ofType('interrupt')).toEqual([expect.objectContaining({ response_id: 'r1' })])
    expect(state.getSnapshot().phase).toBe('interrupted')
    hub.socket.serverFrame(ttsFrameBytes(3, 1, [2], { clipEnd: true }))
    first.start()
    first.finish()
    await flush()
    expect(output.scheduled).toHaveLength(1)
    expect(state.getSnapshot().phase).toBe('interrupted')
    // Nothing in this path cancels agents: only an interrupt control message reaches the Hub.
    expect(hub.socket.controls.map(message => message.type)).toEqual(['interrupt'])
  })

  it('blocked autoplay fails the orb with the tap-to-allow recovery and tells the Hub to stop sending', async () => {
    const { hub, output, state } = await connected()
    hub.socket.server({ type: 'audio_start', response_id: 'r1', generation: 3, delivery: 'buffered', codec: 'mp3' })
    hub.socket.serverFrame(ttsFrameBytes(3, 0, [1], { clipEnd: true }))
    output.scheduled[0]?.fail(new PlaybackError('blocked', 'The browser blocked sound.'))
    await flush()
    expect(state.getSnapshot()).toMatchObject({ phase: 'failed', failure: { code: 'VOICE_BLOCKED' } })
    expect(state.getSnapshot().failure?.recovery).toContain('Tap the button')
    expect(hub.socket.ofType('interrupt')).toHaveLength(1)
  })

  it('a frame for an unknown generation is dropped', async () => {
    const { hub, output } = await connected()
    hub.socket.server({ type: 'audio_start', response_id: 'r1', generation: 3, delivery: 'buffered', codec: 'mp3' })
    hub.socket.serverFrame(ttsFrameBytes(9, 0, [1], { clipEnd: true }))
    expect(output.scheduled).toHaveLength(0)
  })

  it('a dropped link ends playback and the orb', async () => {
    const { hub, output, state } = await connected()
    hub.socket.server({ type: 'audio_start', response_id: 'r1', generation: 3, delivery: 'buffered', codec: 'mp3' })
    hub.socket.drop()
    expect(output.flushes).toBe(1)
    expect(state.getSnapshot().phase).toBe('buffering')
  })
})

describe('Web Audio output', () => {
  function context(over: Partial<AudioContextLike> = {}): AudioContextLike {
    const sources: unknown[] = []
    return {
      state: 'running', currentTime: 0,
      destination: {} as AudioNode,
      resume: async () => undefined,
      decodeAudioData: async () => { throw new Error('bad data') },
      createBuffer: (_channels: number, length: number, rate: number) => ({ duration: length / rate, getChannelData: () => new Float32Array(length) }) as unknown as AudioBuffer,
      createBufferSource: () => {
        const source = { buffer: null as unknown, onended: null as (() => void) | null, started: false, stopped: false, connect() {}, start() { source.started = true }, stop() { source.stopped = true } }
        sources.push(source)
        return source as unknown as AudioBufferSourceNode
      },
      ...over,
    }
  }

  it('reports blocked sound when the context cannot resume', async () => {
    const output = createWebAudioOutput(context({ state: 'suspended', resume: async () => { throw new Error('gesture required') } }))
    const result = await output.schedule({ codec: 'pcm16', bytes: new Uint8Array(4), sampleRate: 24_000 }, new AbortController().signal, () => undefined).catch((error: unknown) => error)
    expect(result).toBeInstanceOf(PlaybackError)
    expect(result).toMatchObject({ kind: 'blocked' })

    const stays = createWebAudioOutput(context({ state: 'suspended' }))
    await expect(stays.schedule({ codec: 'pcm16', bytes: new Uint8Array(4) }, new AbortController().signal, () => undefined)).rejects.toMatchObject({ kind: 'blocked' })
  })

  it('reports a decode error for audio the browser cannot decode', async () => {
    const output = createWebAudioOutput(context())
    await expect(output.schedule({ codec: 'mp3', bytes: new Uint8Array(4) }, new AbortController().signal, () => undefined)).rejects.toMatchObject({ kind: 'decode' })
  })

  it('flush stops what is sounding and resolves it without reporting a start for an aborted clip', async () => {
    const created: { stopped: boolean; started: boolean }[] = []
    const ctx = context({
      createBufferSource: () => {
        const source = { buffer: null as unknown, onended: null as (() => void) | null, started: false, stopped: false, connect() {}, start() { source.started = true }, stop() { source.stopped = true } }
        created.push(source)
        return source as unknown as AudioBufferSourceNode
      },
    })
    const output = createWebAudioOutput(ctx)
    const abort = new AbortController()
    let started = 0
    const done = output.schedule({ codec: 'pcm16', bytes: new Uint8Array(480), sampleRate: 24_000 }, abort.signal, () => { started += 1 })
    await flush(2)
    expect(created[0]?.started).toBe(true)
    abort.abort()
    output.flush()
    await done
    expect(created[0]?.stopped).toBe(true)
    await flush(8)
    expect(started).toBe(0)
  })
})
