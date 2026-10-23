// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { VoiceBudget, estimateUsd } from '../src/hub/voice/budgets.js'
import { FrameGate } from '../src/hub/voice/frameGate.js'
import { PcmAggregator, bytesToPcm16, floatToPcm16, pcm16ToBytes, resamplePcm16 } from '../src/hub/voice/pcm.js'
import {
  VOICE_LIMITS, VoiceError, decodeVoiceFrame, encodeVoiceFrame, parseVoiceClientMessage, parseVoiceServerMessage, codecForMediaType,
  type VoiceFrame,
} from '../src/hub/voice/protocol.js'
import { SpeakerLeases } from '../src/hub/voice/speakerLease.js'
import { prepareSpeech } from '../src/hub/voice/speech.js'
import { ExactlyOnceSubmitter } from '../src/hub/voice/supervisorBridge.js'
import { FakeBridge, START, userContext } from './hubVoiceHarness.js'

function frame(over: Partial<VoiceFrame['header']> = {}, payload = new Uint8Array(10)): VoiceFrame {
  return { header: { stream: 'mic', codec: 'webm-opus', generation: 1, sequence: 0, timestampMs: 0, final: false, clipEnd: false, ...over }, payload }
}

describe('voice binary frames', () => {
  test('round-trips the header and payload', () => {
    const payload = new Uint8Array([1, 2, 3, 4])
    const out = decodeVoiceFrame(encodeVoiceFrame({ stream: 'tts', codec: 'mp3', generation: 7, sequence: 9, timestampMs: 1234, final: true, clipEnd: true }, payload))
    expect(out.header).toEqual({ stream: 'tts', codec: 'mp3', generation: 7, sequence: 9, timestampMs: 1234, final: true, clipEnd: true })
    expect([...out.payload]).toEqual([1, 2, 3, 4])
  })

  test('rejects short, oversized, wrong-version and unknown frames with typed errors', () => {
    expect(() => decodeVoiceFrame(new Uint8Array(4))).toThrow(VoiceError)
    const big = new Uint8Array(VOICE_LIMITS.headerBytes + VOICE_LIMITS.maxFramePayloadBytes + 1)
    big[0] = 1; big[1] = 1; big[2] = 2
    try { decodeVoiceFrame(big); throw new Error('expected failure') } catch (error) { expect((error as VoiceError).code).toBe('VOICE_LIMIT') }
    const bad = encodeVoiceFrame(frame().header, new Uint8Array(1))
    bad[0] = 9
    expect(() => decodeVoiceFrame(bad)).toThrow('version')
    const stream = encodeVoiceFrame(frame().header, new Uint8Array(1)); stream[1] = 8
    expect(() => decodeVoiceFrame(stream)).toThrow('stream')
    const codec = encodeVoiceFrame(frame().header, new Uint8Array(1)); codec[2] = 99
    expect(() => decodeVoiceFrame(codec)).toThrow('codec')
    expect(() => encodeVoiceFrame(frame({ generation: -1 }).header, new Uint8Array(1))).toThrow('generation')
  })
})

describe('voice control messages', () => {
  test('parse start, commit, interrupt, stop, capabilities', () => {
    expect(parseVoiceClientMessage('{"type":"capabilities"}')).toEqual({ type: 'capabilities' })
    expect(parseVoiceClientMessage('{"v":1,"type":"stop"}')).toEqual({ type: 'stop' })
    expect(parseVoiceClientMessage('{"type":"start","mode":"push_to_talk","conversation_id":"t1","utterance_id":"u1","codec":"webm-opus","language":"en-US"}'))
      .toMatchObject({ type: 'start', mode: 'push_to_talk', language: 'en-US' })
    expect(parseVoiceClientMessage('{"type":"commit","utterance_id":"u1"}')).toEqual({ type: 'commit', utterance_id: 'u1' })
    expect(parseVoiceClientMessage('{"type":"interrupt","response_id":"r1","played_ms":120.7}')).toEqual({ type: 'interrupt', response_id: 'r1', played_ms: 120 })
  })

  test('refuse malformed input with a typed error', () => {
    const bad = (text: string): string => { try { parseVoiceClientMessage(text); return 'ok' } catch (error) { return (error as VoiceError).code } }
    expect(bad('not json')).toBe('VOICE_PROTOCOL')
    expect(bad('[]')).toBe('VOICE_PROTOCOL')
    expect(bad('{"v":2,"type":"stop"}')).toBe('VOICE_PROTOCOL')
    expect(bad('{"type":"nope"}')).toBe('VOICE_PROTOCOL')
    expect(bad('{"type":"start","mode":"push_to_talk","conversation_id":"a b","utterance_id":"u","codec":"webm-opus"}')).toBe('VOICE_PROTOCOL')
    expect(bad('{"type":"start","mode":"push_to_talk","conversation_id":"t","utterance_id":"u","codec":"flac3"}')).toBe('VOICE_FORMAT')
    expect(bad('{"type":"interrupt","played_ms":-1}')).toBe('VOICE_PROTOCOL')
    expect(bad(`{"type":"submit","utterance_id":"u","text":"${'x'.repeat(VOICE_LIMITS.maxTextChars + 1)}"}`)).toBe('VOICE_PROTOCOL')
    expect(bad(JSON.stringify({ type: 'capabilities', pad: 'x'.repeat(VOICE_LIMITS.maxControlBytes) }))).toBe('VOICE_LIMIT')
  })

  test('server parser drops unknown or unversioned messages', () => {
    expect(parseVoiceServerMessage('{"v":1,"type":"state","state":"idle","generation":1}')).toMatchObject({ type: 'state' })
    expect(parseVoiceServerMessage('{"type":"state"}')).toBeUndefined()
    expect(parseVoiceServerMessage('{"v":1,"type":"zzz"}')).toBeUndefined()
    expect(parseVoiceServerMessage('garbage')).toBeUndefined()
  })

  test('codecForMediaType understands parameterized recorder types', () => {
    expect(codecForMediaType('audio/webm;codecs=opus')).toBe('webm-opus')
    expect(codecForMediaType('Audio/MP4; codecs="mp4a.40.2"')).toBe('mp4-aac')
    expect(codecForMediaType('audio/ogg;codecs=opus')).toBe('ogg-opus')
    expect(codecForMediaType('video/mp4')).toBeUndefined()
  })
})

describe('FrameGate', () => {
  const make = (over: Partial<ConstructorParameters<typeof FrameGate>[0]> = {}, clock = { t: START }): { gate: FrameGate; clock: { t: number } } =>
    ({ gate: new FrameGate({ generation: 1, codec: 'webm-opus', maxBytes: 100, maxSeconds: 10, now: () => clock.t, ...over }), clock })

  test('accepts the contiguous stream and counts bytes', () => {
    const { gate } = make()
    expect(gate.check(frame({ sequence: 0 })).ok).toBe(true)
    expect(gate.check(frame({ sequence: 1, timestampMs: 500 })).ok).toBe(true)
    expect(gate.byteCount).toBe(20)
    expect(gate.durationMs).toBe(500)
  })

  test('rejects wrong generation, late, out-of-order, wrong codec, wrong direction', () => {
    const { gate } = make()
    gate.check(frame({ sequence: 0 }))
    const stale = gate.check(frame({ generation: 0, sequence: 1 }))
    expect(stale.ok === false && stale.error.code).toBe('VOICE_STALE')
    const late = gate.check(frame({ sequence: 0 }))
    expect(late.ok === false && late.error.code).toBe('VOICE_STALE')
    const gap = gate.check(frame({ sequence: 5 }))
    expect(gap.ok === false && gap.fatal).toBe(true)
    const codec = gate.check(frame({ sequence: 1, codec: 'mp4-aac' }))
    expect(codec.ok === false && codec.error.code).toBe('VOICE_FORMAT')
    const wrongWay = gate.check(frame({ sequence: 1, stream: 'tts' }))
    expect(wrongWay.ok === false && wrongWay.error.code).toBe('VOICE_PROTOCOL')
  })

  test('enforces the byte, sender-time and Hub-clock limits', () => {
    const bytes = make()
    expect(bytes.gate.check(frame({ sequence: 0 }, new Uint8Array(100))).ok).toBe(true)
    const over = bytes.gate.check(frame({ sequence: 1 }, new Uint8Array(1)))
    expect(over.ok === false && over.error.code === 'VOICE_LIMIT' && over.fatal).toBe(true)
    const time = make()
    const slow = time.gate.check(frame({ sequence: 0, timestampMs: 10_001 }))
    expect(slow.ok === false && slow.error.code).toBe('VOICE_LIMIT')
    // A sender that lies about timestamps is still bounded by the Hub's own clock.
    const lying = make()
    lying.clock.t += 16_000
    const hubClock = lying.gate.check(frame({ sequence: 0, timestampMs: 0 }))
    expect(hubClock.ok === false && hubClock.error.code).toBe('VOICE_LIMIT')
  })

  test('nothing is accepted after the final frame', () => {
    const { gate } = make()
    expect(gate.check(frame({ sequence: 0, final: true })).ok).toBe(true)
    expect(gate.check(frame({ sequence: 1 })).ok).toBe(false)
  })
})

describe('SpeakerLeases', () => {
  const holder = (userId: string, deviceId: string, connectionId: string) => ({ userId, deviceId, connectionId })

  test('one speaker per conversation; the same user and device may take it back', () => {
    const now = { t: START }
    const leases = new SpeakerLeases({ ttlMs: 1000, clock: () => now.t })
    expect(leases.acquire('c', holder('alice', 'd1', 'k1')).ok).toBe(true)
    const rival = leases.acquire('c', holder('bob', 'd2', 'k2'))
    expect(rival.ok === false && rival.heldBy.userId).toBe('alice')
    // Same account on another device is a different speaker.
    expect(leases.acquire('c', holder('alice', 'd9', 'k3')).ok).toBe(false)
    // Reconnect of the same device takes it over; the old socket stops being the holder.
    expect(leases.acquire('c', holder('alice', 'd1', 'k4')).ok).toBe(true)
    expect(leases.isHolder('c', holder('alice', 'd1', 'k1'))).toBe(false)
    expect(leases.isHolder('c', holder('alice', 'd1', 'k4'))).toBe(true)
  })

  test('expires, renews, and is not revived after expiry', () => {
    const now = { t: START }
    const leases = new SpeakerLeases({ ttlMs: 1000, clock: () => now.t })
    leases.acquire('c', holder('alice', 'd1', 'k1'))
    now.t += 900
    expect(leases.renew('c', holder('alice', 'd1', 'k1')).ok).toBe(true)
    now.t += 900
    expect(leases.isHolder('c', holder('alice', 'd1', 'k1'))).toBe(true)
    now.t += 1100
    expect(leases.renew('c', holder('alice', 'd1', 'k1')).ok).toBe(false)
    expect(leases.acquire('c', holder('bob', 'd2', 'k2')).ok).toBe(true)
  })

  test('release and releaseConnection free the lease; renew by a non-holder fails', () => {
    const leases = new SpeakerLeases({ ttlMs: 1000, clock: () => START })
    leases.acquire('c', holder('alice', 'd1', 'k1'))
    expect(leases.renew('c', holder('bob', 'd2', 'k2')).ok).toBe(false)
    expect(leases.release('c', holder('bob', 'd2', 'k2'))).toBe(false)
    leases.releaseConnection('k1')
    expect(leases.size).toBe(0)
  })
})

describe('VoiceBudget', () => {
  test('reserve refuses over the session and daily limits, and settles actual spend', () => {
    const budget = new VoiceBudget({ clock: () => START, limits: { maxSessionUsd: 1, maxDailyUserUsd: 1.5 } })
    const first = budget.reserve('alice', 's1', 'stt', 0.6)
    expect(() => budget.reserve('alice', 's1', 'stt', 0.6)).toThrow('session reached')
    budget.settle(first, 0.5)
    expect(budget.usage('alice', 's1')).toMatchObject({ sessionUsd: 0.5, dailyUsd: 0.5, unknownCost: false, maxSessionUsd: 1, maxDailyUsd: 1.5 })
    budget.reserve('alice', 's2', 'stt', 0.9)
    expect(() => budget.reserve('alice', 's3', 'stt', 0.2)).toThrow('daily')
    // Another tenant has its own daily ledger.
    expect(() => budget.reserve('bob', 's4', 'stt', 0.9)).not.toThrow()
  })

  test('an unknown price is never zero: refused under a limit, flagged without one', () => {
    const limited = new VoiceBudget({ clock: () => START, limits: { maxSessionUsd: 5 } })
    try { limited.reserve('alice', 's', 'tts', undefined); throw new Error('expected refusal') } catch (error) {
      expect((error as VoiceError).code).toBe('VOICE_BUDGET')
    }
    const open = new VoiceBudget({ clock: () => START })
    const reservation = open.reserve('alice', 's', 'tts', undefined)
    open.settle(reservation)
    expect(open.usage('alice', 's').unknownCost).toBe(true)
    expect(estimateUsd({}, 'stt', 1)).toBeUndefined()
    expect(estimateUsd({ ttsUsdPerMillionChars: 30 }, 'tts', 1000)).toBeCloseTo(0.03)
  })

  test('release returns the reservation; daily ledger rolls over at midnight', () => {
    const now = { t: Date.parse('2026-10-22T23:59:00Z') }
    const budget = new VoiceBudget({ clock: () => now.t, limits: { maxDailyUserUsd: 1 } })
    budget.release(budget.reserve('alice', 's', 'stt', 1))
    budget.settle(budget.reserve('alice', 's', 'stt', 1), 1)
    expect(() => budget.reserve('alice', 's', 'stt', 0.1)).toThrow('daily')
    now.t += 2 * 60_000
    expect(() => budget.reserve('alice', 's2', 'stt', 0.1)).not.toThrow()
  })
})

describe('ExactlyOnceSubmitter', () => {
  const actor = userContext('alice').actor

  test('a repeated key reaches the bridge once, concurrently and afterwards', async () => {
    const bridge = new FakeBridge()
    const submitter = new ExactlyOnceSubmitter(bridge)
    const [a, b] = await Promise.all([submitter.submit('t', 'hi', actor, 'k1'), submitter.submit('t', 'hi', actor, 'k1')])
    const c = await submitter.submit('t', 'hi', actor, 'k1')
    expect(bridge.submits).toHaveLength(1)
    expect([a.duplicate, b.duplicate, c.duplicate].filter(Boolean)).toHaveLength(2)
    expect((await submitter.submit('t', 'hi', actor, 'k2')).duplicate).toBe(false)
    expect(bridge.submits).toHaveLength(2)
  })

  test('a failed submission is forgotten so an explicit retry can reach the bridge', async () => {
    const bridge = new FakeBridge()
    const submitter = new ExactlyOnceSubmitter(bridge)
    bridge.failNextSubmit = true
    await expect(submitter.submit('t', 'hi', actor, 'k')).rejects.toThrow('bridge down')
    expect((await submitter.submit('t', 'hi', actor, 'k')).duplicate).toBe(false)
  })
})

describe('pcm helpers and speech preparation', () => {
  test('float to pcm16 clips and silences NaN; bytes round-trip', () => {
    const pcm = floatToPcm16(new Float32Array([0, 1, -1, 2, Number.NaN]))
    expect([...pcm]).toEqual([0, 32767, -32768, 32767, 0])
    expect([...bytesToPcm16(pcm16ToBytes(pcm))]).toEqual([...pcm])
    expect(bytesToPcm16(new Uint8Array(5))).toHaveLength(2)
  })

  test('resample changes length proportionally and refuses bad rates', () => {
    expect(resamplePcm16(new Int16Array(480), 48_000, 24_000)).toHaveLength(240)
    expect(resamplePcm16(new Int16Array(10), 16_000, 16_000)).toHaveLength(10)
    expect(() => resamplePcm16(new Int16Array(1), 0, 24_000)).toThrow(RangeError)
  })

  test('aggregator releases a message once enough audio has gathered', () => {
    const aggregator = new PcmAggregator(16_000, 60)
    expect(aggregator.push(new Int16Array(320))).toBeUndefined()
    expect(aggregator.push(new Int16Array(320))).toBeUndefined()
    expect(aggregator.push(new Int16Array(320))?.length).toBe(960)
    expect(aggregator.push(new Int16Array(100))).toBeUndefined()
    expect(aggregator.flush()?.length).toBe(100)
  })

  test('speech drops code, links and markup and stops at the character budget', () => {
    expect(prepareSpeech('Done. See https://x.test/a now.\n```ts\nconst a = 1\n```\n**Next** step.', 500)).toEqual(['Done.', 'See now.', 'Next step.'])
    expect(prepareSpeech('```code only```', 100)).toEqual([])
    expect(prepareSpeech('One. Two. Three.', 9)).toEqual(['One.', 'Two.'])
    expect(prepareSpeech('x'.repeat(50), 10)).toEqual(['x'.repeat(10)])
  })
})
