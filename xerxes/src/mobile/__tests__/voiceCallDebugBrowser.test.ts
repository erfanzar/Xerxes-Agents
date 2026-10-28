// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, it, vi } from 'vitest'

import { playDebugCapture, saveDebugCapture } from '../voice/call/debugAudio.js'
import { createDebugCapture } from '../voice/call/debugCapture.js'

const capture = () => createDebugCapture(Float32Array.of(0.1, -0.2, 0.3, -1), { startSample: 1, endSample: 3, preRollSamples: 1 })
const settle = async () => { for (let i = 0; i < 5; i++) await Promise.resolve() }
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers() })

function audioFixture() {
  const received = new Float32Array(4)
  const source = { buffer: undefined as unknown, connect: vi.fn(), disconnect: vi.fn(), start: vi.fn(), stop: vi.fn(), onended: null as (() => void) | null }
  const resume = vi.fn(async (): Promise<void> => undefined)
  const close = vi.fn(async () => undefined)
  const createBuffer = vi.fn(() => ({ getChannelData: () => received }))
  vi.stubGlobal('AudioContext', class {
    state = 'running'
    destination = {}
    resume = resume
    close = close
    createBuffer = createBuffer
    createBufferSource = () => source
  })
  return { source, received, resume, close, createBuffer }
}

describe('local captured-audio playback', () => {
  it('plays the exact Float32 model input at 16 kHz without normalising or re-encoding it, then frees Web Audio', async () => {
    const f = audioFixture()
    const clip = capture()
    const audio = playDebugCapture(clip)
    await settle()
    expect(f.createBuffer).toHaveBeenCalledWith(1, clip.samples.length, 16000)
    expect(f.received).toEqual(clip.samples)
    expect(f.source.start).toHaveBeenCalledOnce()
    f.source.onended?.()
    await audio.finished
    expect(f.close).toHaveBeenCalledOnce()
    expect(f.source.disconnect).toHaveBeenCalledOnce()
    expect(f.source.onended).toBeNull()
  })
  it('stops playback and closes its context on cancellation', async () => {
    const f = audioFixture()
    const audio = playDebugCapture(capture())
    await settle()
    audio.stop()
    await audio.finished
    expect(f.source.stop).toHaveBeenCalledOnce()
    expect(f.close).toHaveBeenCalledOnce()
    expect(f.source.disconnect).toHaveBeenCalledOnce()
  })
  it('can cancel before audio resumes, never starts later and closes immediately', async () => {
    const f = audioFixture()
    let resumed = (): void => undefined
    f.resume.mockImplementation(() => new Promise<void>(resolve => { resumed = resolve }))
    const audio = playDebugCapture(capture())
    audio.stop()
    await audio.finished
    expect(f.close).toHaveBeenCalledOnce()
    resumed()
    await settle()
    expect(f.source.start).not.toHaveBeenCalled()
  })
  it('makes a playback failure observable and frees the context', async () => {
    const f = audioFixture()
    f.resume.mockRejectedValue(new Error('Audio blocked'))
    const audio = playDebugCapture(capture())
    await expect(audio.finished).rejects.toThrow('Audio blocked')
    expect(f.close).toHaveBeenCalledOnce()
    expect(f.source.start).not.toHaveBeenCalled()
  })
})

describe('explicit WAV save/share only', () => {
  it('shares a real mono PCM WAV File only when file sharing is supported, without a download URL', async () => {
    const share = vi.fn(async (_data: ShareData) => undefined)
    const canShare = vi.fn(() => true)
    vi.stubGlobal('navigator', { share, canShare })
    const createUrl = vi.spyOn(URL, 'createObjectURL')
    await saveDebugCapture(capture())
    const file = share.mock.calls[0]![0].files![0]!
    expect(file).toBeInstanceOf(File)
    expect(file.name).toBe('xerxes-utterance.wav')
    expect(file.type).toBe('audio/wav')
    const wav = new DataView(await file.arrayBuffer())
    expect(wav.byteLength).toBe(44 + 4 * 2)
    expect(wav.getUint32(24, true)).toBe(16000)
    expect(wav.getInt16(50, true)).toBe(-32768)
    expect(canShare).toHaveBeenCalledWith({ files: [file] })
    expect(createUrl).not.toHaveBeenCalled()
  })
  it.each([false, true])('downloads locally when file sharing is absent/unsupported and revokes its object URL (API present: %s)', async apiPresent => {
    vi.useFakeTimers()
    const share = vi.fn()
    vi.stubGlobal('navigator', apiPresent ? { share, canShare: () => false } : {})
    const link = { href: '', download: '', click: vi.fn(), remove: vi.fn() }
    const append = vi.fn()
    vi.stubGlobal('document', { createElement: vi.fn(() => link), body: { append } })
    const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:local-debug')
    const revoke = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    await saveDebugCapture(capture())
    expect(createUrl).toHaveBeenCalledWith(expect.any(Blob))
    expect(link.href).toBe('blob:local-debug')
    expect(link.download).toBe('xerxes-utterance.wav')
    expect(append).toHaveBeenCalledWith(link)
    expect(link.click).toHaveBeenCalledOnce()
    expect(link.remove).toHaveBeenCalledOnce()
    expect(revoke).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(1000)
    expect(revoke).toHaveBeenCalledExactlyOnceWith('blob:local-debug')
    expect(share).not.toHaveBeenCalled()
  })
  it('does not turn a dismissed share sheet into an unwanted download', async () => {
    vi.stubGlobal('navigator', { canShare: () => true, share: vi.fn(async () => { throw new DOMException('Cancelled', 'AbortError') }) })
    const createUrl = vi.spyOn(URL, 'createObjectURL')
    await expect(saveDebugCapture(capture())).resolves.toBeUndefined()
    expect(createUrl).not.toHaveBeenCalled()
  })
  it('surfaces non-cancellation share failures instead of silently sending or saving elsewhere', async () => {
    vi.stubGlobal('navigator', { canShare: () => true, share: vi.fn(async () => { throw new Error('Share denied') }) })
    await expect(saveDebugCapture(capture())).rejects.toThrow('Share denied')
  })
  it('revokes the object URL and removes the link even if the browser refuses the download', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('navigator', {})
    const remove = vi.fn()
    vi.stubGlobal('document', { createElement: () => ({ click() { throw new Error('Download blocked') }, remove }), body: { append: vi.fn() } })
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:failed-debug')
    const revoke = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    await expect(saveDebugCapture(capture())).rejects.toThrow('Download blocked')
    await vi.advanceTimersByTimeAsync(1000)
    expect(remove).toHaveBeenCalledOnce()
    expect(revoke).toHaveBeenCalledWith('blob:failed-debug')
  })
})
