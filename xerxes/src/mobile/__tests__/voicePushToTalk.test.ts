// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { createPushToTalk } from '../voice/pushToTalk.js'
import { FakeOutput, caps, deferred, flush, voiceRig, type VoiceRig } from './voiceFakes.js'

function ptt(rig: VoiceRig, over: { conversation?: string | undefined; capabilities?: ReturnType<typeof caps> } = {}) {
  const conversation = 'conversation' in over ? over.conversation : 'thread-1'
  return createPushToTalk({
    core: rig.core,
    mic: rig.mic,
    capabilities: over.capabilities ?? rig.hub.capabilities,
    conversationId: () => conversation,
    levelIntervalMs: 0,
    newId: () => 'utt-1',
  })
}

describe('push-to-talk controller', () => {
  it('records, sends ordered frames with the Hub generation, commits once and returns the transcript', async () => {
    const rig = voiceRig()
    const voice = ptt(rig)
    await voice.start()
    expect(rig.hub.socket.ofType('start')).toEqual([expect.objectContaining({ mode: 'push_to_talk', conversation_id: 'thread-1', utterance_id: 'utt-1', codec: 'webm-opus' })])
    expect(rig.state.getSnapshot().phase).toBe('listening')
    rig.mic.recorders[0]?.emit([1, 2])
    rig.mic.recorders[0]?.emit([3])
    await flush()
    const text = await voice.stop()
    expect(text).toBe('hello there')
    const frames = rig.hub.socket.frames
    expect(frames.map(frame => frame.header.sequence)).toEqual([0, 1, 2])
    expect(frames.every(frame => frame.header.generation === 7 && frame.header.stream === 'mic' && frame.header.codec === 'webm-opus')).toBe(true)
    expect(frames.map(frame => frame.header.final)).toEqual([false, false, true])
    expect(rig.hub.socket.ofType('commit')).toHaveLength(1)
    expect(rig.mic.streams[0]?.track.stopped).toBe(1)
    expect(rig.state.getSnapshot().phase).toBe('idle')
  })

  it('is never submitted by the controller: only start, frames and commit reach the Hub', async () => {
    const rig = voiceRig()
    const voice = ptt(rig)
    await voice.start()
    rig.mic.recorders[0]?.emit([1])
    await flush()
    await voice.stop()
    expect(rig.hub.socket.controls.map(message => message.type)).toEqual(['start', 'commit'])
  })

  it('empty recognition resolves undefined', async () => {
    const rig = voiceRig()
    rig.hub.transcript = ''
    const voice = ptt(rig)
    await voice.start()
    expect(await voice.stop()).toBeUndefined()
    expect(rig.state.getSnapshot().notice).toBe('Nothing was heard.')
  })

  it('checks capabilities, the conversation and the format before opening the microphone', async () => {
    const off = voiceRig({ caps: caps({ enabled: false }) })
    await expect(ptt(off).start()).rejects.toThrow('not available')
    expect(off.mic.opened).toBe(0)

    const sttOff = voiceRig({ caps: caps({ stt: { available: false, reason: 'not configured' } }) })
    await expect(ptt(sttOff).start()).rejects.toThrow('not available')
    expect(sttOff.mic.opened).toBe(0)

    const noThread = voiceRig()
    await expect(ptt(noThread, { conversation: undefined }).start()).rejects.toThrow('XERXES tab')
    expect(noThread.mic.opened).toBe(0)

    const noFormat = voiceRig()
    noFormat.mic.supported.clear()
    await expect(ptt(noFormat).start()).rejects.toMatchObject({ kind: 'format' })
    expect(noFormat.mic.opened).toBe(0)
    expect(noFormat.state.getSnapshot()).toMatchObject({ phase: 'failed', failure: { code: 'VOICE_FORMAT' } })
    expect(noFormat.hub.sockets).toHaveLength(0)
  })

  it('negotiates MP4/AAC for Safari', async () => {
    const rig = voiceRig()
    rig.mic.supported = new Set(['audio/mp4'])
    await ptt(rig).start()
    expect(rig.mic.recorders[0]?.mimeType).toBe('audio/mp4')
    expect(rig.hub.socket.ofType('start')[0]).toMatchObject({ codec: 'mp4-aac' })
  })

  it('permission denial surfaces a readable error, a recovery, and leaves nothing open', async () => {
    const rig = voiceRig()
    rig.mic.denied = { name: 'NotAllowedError' }
    const voice = ptt(rig)
    await expect(voice.start()).rejects.toThrow('denied')
    expect(rig.mic.recorders).toHaveLength(0)
    expect(rig.state.getSnapshot()).toMatchObject({ phase: 'failed', failure: { code: 'VOICE_DENIED' } })
    expect(rig.state.getSnapshot().failure?.recovery).toContain('microphone')
    // The Hub is told to drop the pending utterance, and a new attempt is allowed.
    expect(rig.hub.socket.ofType('interrupt')).toHaveLength(1)
    rig.mic.denied = undefined
    await expect(voice.start()).resolves.toBeUndefined()
  })

  it('a permission grant that arrives after cancel is released and never records', async () => {
    const rig = voiceRig()
    const gate = deferred()
    rig.mic.gate = gate.promise
    const voice = ptt(rig)
    const started = voice.start().then(() => 'started', (error: unknown) => error)
    await flush()
    expect(rig.mic.opened).toBe(1)
    voice.cancel()
    gate.resolve()
    expect(await started).toMatchObject({ kind: 'cancelled' })
    expect(rig.mic.streams[0]?.track.stopped).toBe(1)
    expect(rig.mic.recorders).toHaveLength(0)
    expect(rig.state.getSnapshot().phase).toBe('idle')
    expect(rig.state.getSnapshot().failure).toBeUndefined()
  })

  it('cancel while recording releases the microphone, drops chunks and tells the Hub', async () => {
    const rig = voiceRig()
    const voice = ptt(rig)
    await voice.start()
    rig.mic.recorders[0]?.emit([1, 2, 3])
    voice.cancel()
    await flush()
    expect(rig.mic.streams[0]?.track.stopped).toBe(1)
    expect(rig.hub.socket.frames).toHaveLength(0)
    expect(rig.hub.socket.ofType('commit')).toHaveLength(0)
    expect(rig.hub.socket.ofType('interrupt')).toHaveLength(1)
    expect(rig.state.getSnapshot().phase).toBe('idle')
    expect(await voice.stop()).toBeUndefined()
  })

  it('a second start while recording is refused', async () => {
    const rig = voiceRig()
    const voice = ptt(rig)
    await voice.start()
    await expect(voice.start()).rejects.toThrow('already in progress')
    voice.cancel()
  })

  it('a Hub refusal at start (budget) fails the orb and never opens the microphone', async () => {
    const rig = voiceRig()
    rig.hub.startError = { code: 'VOICE_BUDGET', message: 'limit reached' }
    await expect(ptt(rig).start()).rejects.toThrow('limit reached')
    expect(rig.mic.opened).toBe(0)
    expect(rig.state.getSnapshot()).toMatchObject({ phase: 'failed', failure: { code: 'VOICE_BUDGET' } })
  })

  it('a Hub that never acknowledges the start times out without recording', async () => {
    const rig = voiceRig()
    rig.hub.holdStart = true
    const voice = createPushToTalk({
      core: rig.core, mic: rig.mic, capabilities: rig.hub.capabilities, conversationId: () => 'thread-1', levelIntervalMs: 0, readyTimeoutMs: 15,
    })
    await expect(voice.start()).rejects.toThrow('did not start')
    expect(rig.mic.opened).toBe(0)
  })

  it('a transcription failure rejects stop and fails the orb', async () => {
    const rig = voiceRig()
    rig.hub.commitError = { code: 'VOICE_PROVIDER', message: 'the transcription provider failed' }
    const voice = ptt(rig)
    await voice.start()
    rig.mic.recorders[0]?.emit([1])
    await flush()
    await expect(voice.stop()).rejects.toThrow('transcription provider failed')
    expect(rig.state.getSnapshot()).toMatchObject({ phase: 'failed', failure: { code: 'VOICE_PROVIDER' } })
    expect(rig.mic.streams[0]?.track.stopped).toBe(1)
  })

  it('a link that is too slow ends the recording instead of queueing audio', async () => {
    const rig = voiceRig()
    const voice = ptt(rig)
    await voice.start()
    rig.hub.buffered = 10 * 1024 * 1024
    rig.mic.recorders[0]?.emit([1, 2, 3])
    await flush()
    expect(rig.state.getSnapshot()).toMatchObject({ phase: 'failed', failure: { code: 'VOICE_OVERLOADED' } })
    expect(rig.mic.streams[0]?.track.stopped).toBe(1)
    expect(rig.hub.socket.frames).toHaveLength(0)
  })

  it('a dropped socket mid-recording rejects stop; the next press opens a fresh socket and replays nothing', async () => {
    const rig = voiceRig()
    const voice = ptt(rig)
    await voice.start()
    rig.mic.recorders[0]?.emit([1, 2, 3])
    await flush()
    expect(rig.hub.socket.frames).toHaveLength(1)
    rig.hub.socket.drop()
    await expect(voice.stop()).rejects.toThrow('dropped')
    expect(rig.mic.streams[0]?.track.stopped).toBe(1)
    expect(rig.state.getSnapshot()).toMatchObject({ phase: 'failed', failure: { code: 'VOICE_DISCONNECTED' } })

    await voice.start()
    expect(rig.hub.sockets).toHaveLength(2)
    expect(rig.hub.sockets[1]?.frames).toHaveLength(0)
    expect(rig.hub.sockets[1]?.ofType('start')).toHaveLength(1)
    voice.cancel()
  })

  it('armReply asks the Hub to speak the next reply on this device, and never throws', async () => {
    const rig = voiceRig()
    const voice = ptt(rig)
    await voice.armReply?.(true)
    expect(rig.hub.socket.ofType('expect_reply')).toEqual([expect.objectContaining({ enabled: true, conversation_id: 'thread-1' })])
    await voice.armReply?.(false)
    expect(rig.hub.socket.ofType('expect_reply')).toHaveLength(2)

    const refused = voiceRig()
    refused.hub.refuse = true
    await expect(ptt(refused).armReply?.(true)).resolves.toBeUndefined()

    // Disarming with no connection does not open one.
    const quiet = voiceRig()
    await ptt(quiet).armReply?.(false)
    expect(quiet.hub.sockets).toHaveLength(0)
  })

  it('interrupt flushes playback and only sends an interrupt control: no agent is cancelled', async () => {
    const rig = voiceRig({ output: new FakeOutput() })
    const voice = ptt(rig)
    await rig.core.ensureLink()
    rig.hub.socket.server({ type: 'audio_start', response_id: 'r1', generation: 4, delivery: 'buffered', codec: 'mp3' })
    voice.interrupt?.()
    expect(rig.hub.socket.controls.map(message => message.type)).toEqual(['interrupt'])
    expect(rig.hub.socket.controls[0]).toMatchObject({ response_id: 'r1' })
    expect(rig.state.getSnapshot().phase).toBe('interrupted')
  })
})
