// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { MicrophoneError, classifyMediaError, startCapture, type CaptureOptions, type MicrophoneDeps } from '../voice/capture.js'
import { chooseRecorderFormat } from '../voice/mime.js'
import { FakeMic, deferred, flush } from './voiceFakes.js'
import { FakeTimers } from './fakes.js'

function setup(over: Partial<CaptureOptions> = {}) {
  const chunks: Uint8Array[] = []
  const limits: string[] = []
  const errors: Error[] = []
  const abort = new AbortController()
  const timers = new FakeTimers()
  const base: CaptureOptions = {
    maxSeconds: 60, maxBytes: 1000, signal: abort.signal, timers,
    onChunk: bytes => { chunks.push(bytes) }, onLimit: reason => { limits.push(reason) }, onError: error => { errors.push(error) },
  }
  return { options: { ...base, ...over }, chunks, limits, errors, abort, timers }
}

describe('recorder format negotiation', () => {
  const supports = (...types: string[]) => (type: string) => types.includes(type)

  it('prefers WebM/Opus, then MP4/AAC for Safari, then Ogg/Opus', () => {
    expect(chooseRecorderFormat(supports('audio/webm;codecs=opus', 'audio/mp4'))).toEqual({ mimeType: 'audio/webm;codecs=opus', codec: 'webm-opus' })
    expect(chooseRecorderFormat(supports('audio/mp4'))).toEqual({ mimeType: 'audio/mp4', codec: 'mp4-aac' })
    expect(chooseRecorderFormat(supports('audio/mp4;codecs=mp4a.40.2', 'audio/mp4'))?.mimeType).toBe('audio/mp4;codecs=mp4a.40.2')
    expect(chooseRecorderFormat(supports('audio/ogg;codecs=opus'))).toEqual({ mimeType: 'audio/ogg;codecs=opus', codec: 'ogg-opus' })
    expect(chooseRecorderFormat(supports('audio/webm'))).toEqual({ mimeType: 'audio/webm', codec: 'webm-opus' })
  })

  it('returns nothing when no accepted format is supported, and treats a throwing probe as unsupported', () => {
    expect(chooseRecorderFormat(supports())).toBeUndefined()
    expect(chooseRecorderFormat(() => { throw new Error('boom') })).toBeUndefined()
    expect(chooseRecorderFormat(supports('audio/webm;codecs=opus'), ['mp4-aac'])).toBeUndefined()
  })
})

describe('microphone errors', () => {
  it('classifies permission denial separately from a missing or busy device', () => {
    expect(classifyMediaError({ name: 'NotAllowedError' })).toMatchObject({ kind: 'denied' })
    expect(classifyMediaError({ name: 'SecurityError' })).toMatchObject({ kind: 'denied' })
    expect(classifyMediaError({ name: 'NotFoundError' })).toMatchObject({ kind: 'unavailable' })
    expect(classifyMediaError({ name: 'NotReadableError' }).message).toContain('busy')
    expect(classifyMediaError('weird')).toMatchObject({ kind: 'unavailable' })
    expect(classifyMediaError({ name: 'NotAllowedError' }).message).toContain('denied')
  })
})

describe('startCapture', () => {
  it('checks the format before touching the microphone', async () => {
    const mic = new FakeMic()
    mic.supported.clear()
    const { options } = setup()
    await expect(startCapture(mic, options)).rejects.toMatchObject({ kind: 'format' })
    expect(mic.opened).toBe(0)
  })

  it('surfaces permission denial and opens nothing else', async () => {
    const mic = new FakeMic()
    mic.denied = { name: 'NotAllowedError' }
    const { options } = setup()
    await expect(startCapture(mic, options)).rejects.toMatchObject({ kind: 'denied' })
    expect(mic.recorders).toHaveLength(0)
  })

  it('releases a permission grant that arrives after cancel and never records', async () => {
    const mic = new FakeMic()
    const gate = deferred()
    mic.gate = gate.promise
    const { options, abort } = setup()
    const outcome = startCapture(mic, options).then(() => 'started', (error: unknown) => error)
    await flush()
    abort.abort()
    gate.resolve()
    expect(await outcome).toMatchObject({ kind: 'cancelled' })
    expect(mic.streams[0]?.track.stopped).toBe(1)
    expect(mic.recorders).toHaveLength(0)
  })

  it('delivers chunks in order and clamps the slice length to 250-1000 ms', async () => {
    const mic = new FakeMic()
    const { options, chunks } = setup({ sliceMs: 50 })
    await startCapture(mic, options)
    expect(mic.recorders[0]?.startedWith).toBe(250)
    mic.recorders[0]?.emit([1])
    mic.recorders[0]?.emit([2, 2])
    mic.recorders[0]?.emit([])
    mic.recorders[0]?.emit([3, 3, 3])
    await flush()
    expect(chunks.map(chunk => chunk.byteLength)).toEqual([1, 2, 3])

    const other = new FakeMic()
    await startCapture(other, setup({ sliceMs: 5000 }).options)
    expect(other.recorders[0]?.startedWith).toBe(1000)
    const defaults = new FakeMic()
    await startCapture(defaults, setup().options)
    expect(defaults.recorders[0]?.startedWith).toBe(500)
  })

  it('stop delivers pending chunks, stops the recorder and releases every track', async () => {
    const mic = new FakeMic()
    const { options, chunks } = setup()
    const session = await startCapture(mic, options)
    expect(session.format.codec).toBe('webm-opus')
    mic.recorders[0]?.emit([1, 2, 3])
    await session.stop()
    expect(chunks).toHaveLength(1)
    expect(mic.recorders[0]?.state).toBe('inactive')
    expect(mic.streams[0]?.track.stopped).toBe(1)
  })

  it('cancel drops chunks that were still being read and releases the microphone', async () => {
    const mic = new FakeMic()
    const { options, chunks } = setup()
    const session = await startCapture(mic, options)
    mic.recorders[0]?.emit([1, 2, 3])
    session.cancel()
    await flush()
    expect(chunks).toHaveLength(0)
    expect(mic.streams[0]?.track.stopped).toBe(1)
    expect(mic.recorders[0]?.state).toBe('inactive')
  })

  it('stops at the size limit without delivering the chunk that crossed it', async () => {
    const mic = new FakeMic()
    const { options, chunks, limits } = setup({ maxBytes: 5 })
    await startCapture(mic, options)
    mic.recorders[0]?.emit([1, 2, 3, 4])
    mic.recorders[0]?.emit([5, 6, 7, 8])
    await flush()
    expect(chunks).toHaveLength(1)
    expect(limits).toEqual(['size'])
    expect(mic.recorders[0]?.state).toBe('inactive')
  })

  it('stops at the time limit', async () => {
    const mic = new FakeMic()
    const { options, limits, timers } = setup({ maxSeconds: 5 })
    await startCapture(mic, options)
    timers.advance(4999)
    expect(limits).toEqual([])
    timers.advance(2)
    expect(limits).toEqual(['time'])
    expect(mic.recorders[0]?.state).toBe('inactive')
  })

  it('releases the microphone when the recorder cannot be created or started', async () => {
    const noRecorder = new FakeMic()
    noRecorder.createRecorder = () => { throw new Error('no recorder') }
    await expect(startCapture(noRecorder, setup().options)).rejects.toMatchObject({ kind: 'unsupported' })
    expect(noRecorder.streams[0]?.track.stopped).toBe(1)

    const noStart = new FakeMic()
    noStart.onCreate = recorder => { recorder.start = () => { throw new Error('cannot start') } }
    await expect(startCapture(noStart, setup().options)).rejects.toBeInstanceOf(MicrophoneError)
    expect(noStart.streams[0]?.track.stopped).toBe(1)
  })

  it('reports recorder errors', async () => {
    const mic = new FakeMic()
    const { options, errors } = setup()
    await startCapture(mic, options)
    mic.recorders[0]?.onerror?.()
    expect(errors).toHaveLength(1)
    expect(errors[0]?.message).toContain('failed')
  })

  it('reads the input level from the meter and closes it on release; a failing meter reads zero', async () => {
    const mic = new FakeMic()
    let closed = 0
    let throwing = false
    const deps: MicrophoneDeps = {
      getUserMedia: () => mic.getUserMedia(),
      createRecorder: (stream, mimeType) => mic.createRecorder(stream, mimeType),
      isTypeSupported: mimeType => mic.isTypeSupported(mimeType),
      level: () => ({ read: () => { if (throwing) throw new Error('x'); return 0.5 }, close: () => { closed += 1 } }),
    }
    const session = await startCapture(deps, setup().options)
    expect(session.level()).toBe(0.5)
    throwing = true
    expect(session.level()).toBe(0)
    session.cancel()
    expect(closed).toBe(1)
  })
})
