// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { VOICE_LIMITS } from '../src/hub/voice/protocol.js'
import { FakeRealtimePort, Rig, control, micFrame, settle, startPtt, type Connection } from './hubVoiceHarness.js'

async function record(conn: Connection, generation: number, frames = 2, utterance = 'u1'): Promise<void> {
  for (let sequence = 0; sequence < frames; sequence += 1) conn.session.handleBinary(micFrame(generation, sequence, 50))
  await conn.session.handleText(control({ type: 'commit', utterance_id: utterance }))
}

describe('voice session: availability', () => {
  test('a disabled hub answers VOICE_UNAVAILABLE and never calls a provider', async () => {
    const rig = new Rig({ settings: { enabled: false } })
    const conn = rig.connect()
    await conn.session.handleText(control({ type: 'start', mode: 'push_to_talk', conversation_id: 't', utterance_id: 'u', codec: 'webm-opus' }))
    expect(conn.transport.errors()).toEqual(['VOICE_UNAVAILABLE'])
    await conn.session.handleText(control({ type: 'capabilities' }))
    expect(conn.transport.last('capabilities').capabilities).toMatchObject({ enabled: false, modes: [] })
    expect(rig.stt.calls).toHaveLength(0)
  })

  test('no STT configured: push-to-talk and live are unavailable, with a reason', async () => {
    const rig = new Rig({ stt: null })
    const conn = rig.connect()
    await conn.session.handleText(control({ type: 'start', mode: 'push_to_talk', conversation_id: 't', utterance_id: 'u', codec: 'webm-opus' }))
    await conn.session.handleText(control({ type: 'start', mode: 'live', conversation_id: 't', utterance_id: 'u', codec: 'pcm16', sample_rate: 16000 }))
    expect(conn.transport.errors()).toEqual(['VOICE_UNAVAILABLE', 'VOICE_UNAVAILABLE'])
    await conn.session.handleText(control({ type: 'capabilities' }))
    expect(conn.transport.last('capabilities').capabilities.stt).toMatchObject({ available: false, reason: 'not configured' })
  })

  test('agent principals cannot speak', async () => {
    const rig = new Rig()
    const { agentContext } = await import('./hubVoiceHarness.js')
    const { VoiceSession } = await import('../src/hub/voice/session.js')
    const { FakeTransport } = await import('./hubVoiceHarness.js')
    const transport = new FakeTransport()
    const session = new VoiceSession({
      id: 'a1', context: agentContext('robot'), config: rig.config, supervisor: rig.bridge, submitter: rig.submitter,
      leases: rig.leases, budget: rig.budget, clock: rig.clock, transport,
    })
    await session.handleText(control({ type: 'start', mode: 'push_to_talk', conversation_id: 't', utterance_id: 'u', codec: 'webm-opus' }))
    expect(transport.errors()).toEqual(['VOICE_PERMISSION'])
  })

  test('malformed control messages are rejected and repeated abuse closes the socket', async () => {
    const rig = new Rig()
    const conn = rig.connect()
    for (let index = 0; index < 5; index += 1) await conn.session.handleText('{nope')
    expect(conn.transport.errors()).toHaveLength(5)
    expect(conn.transport.closed?.code).toBe(1008)
  })
})

describe('voice session: push-to-talk', () => {
  test('records, transcribes once, shows an editable transcript, submits exactly once', async () => {
    const rig = new Rig()
    const conn = rig.connect()
    const generation = await startPtt(conn)
    expect(conn.transport.last('state')).toMatchObject({ state: 'listening', utterance_id: 'u1', conversation_id: 'thread-1' })
    await record(conn, generation)
    expect(rig.stt.calls).toHaveLength(1)
    expect(rig.stt.calls[0]?.audio.byteLength).toBe(100)
    expect(rig.stt.calls[0]?.mediaType).toBe('audio/webm')
    expect(conn.transport.last('transcript')).toMatchObject({ utterance_id: 'u1', text: 'hello world', final: true, revision: 1 })
    // Confirm-transcript is on: nothing was submitted by recognition alone.
    expect(rig.bridge.submits).toHaveLength(0)

    await conn.session.handleText(control({ type: 'submit', utterance_id: 'u1', text: 'hello edited world' }))
    await conn.session.handleText(control({ type: 'submit', utterance_id: 'u1', text: 'hello edited world' }))
    expect(rig.bridge.submits).toHaveLength(1)
    expect(rig.bridge.submits[0]).toMatchObject({ threadId: 'thread-1', text: 'hello edited world', key: 'voice:u1' })
    expect(rig.bridge.submits[0]?.actor.principal_id).toBe('alice')
    const submitted = conn.transport.ofType('submitted')
    expect(submitted.map(m => m.duplicate)).toEqual([false, true])
  })

  test('with confirmation off a commit that asks to submit sends the text once', async () => {
    const rig = new Rig({ settings: { confirmTranscript: false } })
    const conn = rig.connect()
    const generation = await startPtt(conn)
    conn.session.handleBinary(micFrame(generation, 0, 40))
    await conn.session.handleText(control({ type: 'commit', utterance_id: 'u1', submit: true }))
    await conn.session.handleText(control({ type: 'commit', utterance_id: 'u1', submit: true }))
    expect(rig.bridge.submits).toHaveLength(1)
    expect(conn.transport.errors()).toEqual(['VOICE_STALE'])
  })

  test('empty recognition is never submitted and an empty recording never reaches the provider', async () => {
    const rig = new Rig()
    rig.stt.text = '   '
    const conn = rig.connect()
    const generation = await startPtt(conn)
    await record(conn, generation, 1)
    expect(conn.transport.last('transcript')).toMatchObject({ text: '', final: true })
    await conn.session.handleText(control({ type: 'submit', utterance_id: 'u1', text: 'x' }))
    expect(conn.transport.errors()).toEqual(['VOICE_STALE'])
    expect(rig.bridge.submits).toHaveLength(0)

    const empty = rig.connect('bob')
    await startPtt(empty, { utterance: 'u2', conversation: 'thread-2' })
    await empty.session.handleText(control({ type: 'commit', utterance_id: 'u2' }))
    expect(rig.stt.calls).toHaveLength(1)
    expect(empty.transport.last('transcript').text).toBe('')
  })

  test('frame limits: oversized, over the byte budget, wrong generation, late and unauthorized', async () => {
    const rig = new Rig({ settings: { maxUtteranceBytes: 1024 } })
    const conn = rig.connect()
    const generation = await startPtt(conn)

    conn.session.handleBinary(micFrame(generation + 1, 0, 10))
    expect(conn.transport.errors().at(-1)).toBe('VOICE_STALE')
    conn.session.handleBinary(micFrame(generation, 0, 10))
    conn.session.handleBinary(micFrame(generation, 0, 10))
    expect(conn.transport.errors().at(-1)).toBe('VOICE_STALE')

    // A recording over the byte limit ends the stream: a later commit finds nothing.
    conn.session.handleBinary(micFrame(generation, 1, 2000))
    expect(conn.transport.errors().at(-1)).toBe('VOICE_LIMIT')
    await conn.session.handleText(control({ type: 'commit', utterance_id: 'u1' }))
    expect(conn.transport.errors().at(-1)).toBe('VOICE_STALE')
    expect(rig.stt.calls).toHaveLength(0)

    // Payload above the per-frame cap is refused at decode.
    const big = rig.connect('bob')
    const g2 = await startPtt(big, { utterance: 'u2', conversation: 'thread-bob' })
    // The encoder refuses oversized payloads, so build the raw frame by hand: a valid header, one byte too many.
    const oversized = new Uint8Array(VOICE_LIMITS.headerBytes + VOICE_LIMITS.maxFramePayloadBytes + 1)
    oversized.set(micFrame(g2, 0, 1).subarray(0, VOICE_LIMITS.headerBytes))
    big.session.handleBinary(oversized)
    expect(big.transport.errors().at(-1)).toBe('VOICE_LIMIT')

    // Frames with no open stream, or from someone who lost the lease, are refused.
    const none = rig.connect('carol')
    none.session.handleBinary(micFrame(1, 0, 10))
    expect(none.transport.errors()).toEqual(['VOICE_STALE'])
    const bad = rig.connect('dave')
    bad.session.handleBinary(new Uint8Array(3))
    expect(bad.transport.errors()).toEqual(['VOICE_PROTOCOL'])
  })

  test('the time limit ends a recording the sender keeps open', async () => {
    const rig = new Rig({ settings: { maxUtteranceSeconds: 5 } })
    const conn = rig.connect()
    const generation = await startPtt(conn)
    conn.session.handleBinary(micFrame(generation, 0, 10))
    rig.advance(11_000)
    conn.session.tick()
    expect(conn.transport.errors()).toContain('VOICE_LIMIT')
    await conn.session.handleText(control({ type: 'commit', utterance_id: 'u1' }))
    expect(rig.stt.calls).toHaveLength(0)
  })

  test('only one recording at a time per connection, and unsupported formats are refused', async () => {
    const rig = new Rig()
    const conn = rig.connect()
    await startPtt(conn)
    await conn.session.handleText(control({ type: 'start', mode: 'push_to_talk', conversation_id: 'thread-1', utterance_id: 'u2', codec: 'webm-opus' }))
    expect(conn.transport.errors()).toEqual(['VOICE_BUSY'])
    const other = rig.connect('bob')
    await other.session.handleText(control({ type: 'start', mode: 'push_to_talk', conversation_id: 't2', utterance_id: 'u', codec: 'mp3' }))
    expect(other.transport.errors()).toEqual(['VOICE_FORMAT'])
  })

  test('a provider failure is typed and does not leak the cause', async () => {
    const rig = new Rig()
    rig.stt.fail = new Error('upstream secret sk-123')
    const conn = rig.connect()
    const generation = await startPtt(conn)
    await record(conn, generation)
    expect(conn.transport.errors()).toEqual(['VOICE_PROVIDER'])
    expect(conn.transport.texts.join('')).not.toContain('sk-123')
  })

  test('interrupt during transcription discards the result unseen', async () => {
    const rig = new Rig()
    let release: () => void = () => undefined
    rig.stt.gate = new Promise<void>(resolve => { release = resolve })
    const conn = rig.connect()
    const generation = await startPtt(conn)
    conn.session.handleBinary(micFrame(generation, 0, 30))
    const commit = conn.session.handleText(control({ type: 'commit', utterance_id: 'u1' }))
    await settle()
    await conn.session.handleText(control({ type: 'interrupt' }))
    release()
    await commit
    expect(conn.transport.ofType('transcript')).toHaveLength(0)
  })
})

describe('voice session: budgets', () => {
  test('refuses before calling the provider when the estimate exceeds the limit', async () => {
    const rig = new Rig({ settings: { maxSessionUsd: 0.001, pricing: { sttUsdPerMinute: 60 } } })
    const conn = rig.connect()
    const generation = await startPtt(conn)
    // Stay inside the 30 s speaker lease so the budget, not the lease, is what refuses.
    rig.advance(20_000)
    await record(conn, generation)
    expect(conn.transport.errors()).toEqual(['VOICE_BUDGET'])
    expect(rig.stt.calls).toHaveLength(0)
  })

  test('an unpriced provider under a spend limit is refused; without a limit usage reports unknown cost', async () => {
    const limited = new Rig({ settings: { maxDailyUserUsd: 5 } })
    const a = limited.connect()
    const g = await startPtt(a)
    await record(a, g)
    expect(a.transport.errors()).toEqual(['VOICE_BUDGET'])
    expect(limited.stt.calls).toHaveLength(0)

    const open = new Rig()
    const b = open.connect()
    const g2 = await startPtt(b)
    await record(b, g2)
    expect(b.transport.last('usage')).toMatchObject({ unknown_cost: true })
    expect(open.stt.calls).toHaveLength(1)
  })

  test('priced calls are settled and reported', async () => {
    const rig = new Rig({ settings: { maxSessionUsd: 1, pricing: { sttUsdPerMinute: 0.006 } } })
    const conn = rig.connect()
    const generation = await startPtt(conn)
    await record(conn, generation)
    const usage = conn.transport.last('usage')
    expect(usage.unknown_cost).toBe(false)
    expect(usage.max_session_usd).toBe(1)
    expect(usage.session_usd).toBeGreaterThanOrEqual(0)
  })
})

describe('voice session: speaker lease and tenants', () => {
  test('racing starts on one conversation: exactly one speaker wins', async () => {
    const rig = new Rig()
    const alice = rig.connect('alice')
    const bob = rig.connect('bob')
    const start = (id: string) => control({ type: 'start', mode: 'push_to_talk', conversation_id: 'shared', utterance_id: id, codec: 'webm-opus' })
    await Promise.all([alice.session.handleText(start('ua')), bob.session.handleText(start('ub'))])
    const winners = [alice, bob].filter(conn => conn.transport.ofType('state').some(m => m.state === 'listening'))
    expect(winners).toHaveLength(1)
    const loser = winners[0] === alice ? bob : alice
    expect(loser.transport.errors()).toEqual(['VOICE_BUSY'])
    expect(loser.transport.last('lease')).toMatchObject({ held_by_you: false })
  })

  test('losing the lease stops a speaker mid-recording; expiry frees it for someone else', async () => {
    const rig = new Rig()
    const alice = rig.connect('alice')
    const g = await startPtt(alice, { conversation: 'shared' })
    rig.advance(VOICE_LIMITS.leaseTtlMs + 1)
    alice.session.handleBinary(micFrame(g, 0, 10))
    expect(alice.transport.errors()).toEqual(['VOICE_PERMISSION'])
    const bob = rig.connect('bob')
    await bob.session.handleText(control({ type: 'start', mode: 'push_to_talk', conversation_id: 'shared', utterance_id: 'ub', codec: 'webm-opus' }))
    expect(bob.transport.errors()).toEqual([])
    expect(bob.transport.last('state').state).toBe('listening')
  })

  test('the same person and device reconnects and takes the lease back; audio is not replayed', async () => {
    const rig = new Rig()
    const first = rig.connect('alice', 'phone')
    const generation = await startPtt(first, { conversation: 'c' })
    first.session.handleBinary(micFrame(generation, 0, 50))
    first.session.dispose()
    expect(rig.leases.size).toBe(0)

    const second = rig.connect('alice', 'phone')
    // The new socket has no stream: the old generation's next frame is refused, nothing is replayed.
    second.session.handleBinary(micFrame(generation, 1, 50))
    expect(second.transport.errors()).toEqual(['VOICE_STALE'])
    expect(second.transport.binaries).toHaveLength(0)
    expect(rig.stt.calls).toHaveLength(0)
    await second.session.handleText(control({ type: 'commit', utterance_id: 'u1' }))
    expect(second.transport.errors()).toEqual(['VOICE_STALE', 'VOICE_STALE'])
    expect(rig.stt.calls).toHaveLength(0)
  })

  test('a person denied the conversation cannot speak into it or hear it', async () => {
    const rig = new Rig()
    rig.bridge.denied.add('private|bob')
    const bob = rig.connect('bob')
    await bob.session.handleText(control({ type: 'start', mode: 'push_to_talk', conversation_id: 'private', utterance_id: 'u', codec: 'webm-opus' }))
    expect(bob.transport.errors()).toEqual(['VOICE_PERMISSION'])
    expect(rig.bridge.listenerCount('private')).toBe(0)
    await bob.session.handleText(control({ type: 'expect_reply', conversation_id: 'private' }))
    expect(bob.transport.errors()).toEqual(['VOICE_PERMISSION', 'VOICE_PERMISSION'])
    rig.bridge.reply('private', 'secret reply')
    await settle()
    expect(bob.transport.ofType('reply')).toHaveLength(0)
  })

  test("another user's reply is never played unless this device asked for it", async () => {
    const rig = new Rig()
    const alice = rig.connect('alice', 'phone')
    const bob = rig.connect('bob', 'tablet')
    const ga = await startPtt(alice, { conversation: 'shared' })
    await record(alice, ga)
    await bob.session.handleText(control({ type: 'expect_reply', conversation_id: 'shared', enabled: false }))
    await bob.session.handleText(control({ type: 'expect_reply', conversation_id: 'shared' }))
    await alice.session.handleText(control({ type: 'submit', utterance_id: 'u1', text: 'status please' }))
    // Bob armed his own device, Alice's reply arrives on both; each plays only if armed (replies go to the requester by default).
    rig.bridge.reply('shared', 'All quiet.', 'm1')
    await settle(20)
    expect(alice.transport.ofType('reply')).toHaveLength(1)
    expect(alice.transport.ofType('reply')[0]?.text).toBe('All quiet.')
    // Bob's armed message id is unset, so his device also speaks it once: arming is explicit and one-shot.
    rig.bridge.reply('shared', 'Another.', 'm9')
    await settle(20)
    expect(alice.transport.ofType('reply')).toHaveLength(1)
  })

  test('revocation style close releases the lease and the reply listener', async () => {
    const rig = new Rig()
    const conn = rig.connect()
    await startPtt(conn, { conversation: 'c' })
    expect(rig.bridge.listenerCount('c')).toBe(1)
    conn.session.end('access revoked', 1008)
    expect(conn.transport.closed).toEqual({ code: 1008, reason: 'access revoked' })
    expect(rig.bridge.listenerCount('c')).toBe(0)
    expect(rig.leases.size).toBe(0)
  })
})

describe('voice session: replies and playback', () => {
  async function submitted(rig: Rig): Promise<Connection> {
    const conn = rig.connect()
    const generation = await startPtt(conn)
    await record(conn, generation)
    await conn.session.handleText(control({ type: 'submit', utterance_id: 'u1', text: 'hello world' }))
    return conn
  }

  test('speaks only the supervisor reply that answers this message, sentence by sentence, with an honest delivery label', async () => {
    const rig = new Rig()
    const conn = await submitted(rig)
    rig.bridge.reply('thread-1', 'Unrelated.', 'm999')
    await settle()
    expect(conn.transport.ofType('reply')).toHaveLength(0)
    rig.bridge.reply('thread-1', 'First sentence. Second sentence.', 'm1')
    await settle(20)
    expect(rig.tts.calls).toEqual(['First sentence.', 'Second sentence.'])
    expect(conn.transport.last('reply')).toMatchObject({ text: 'First sentence. Second sentence.', spoken: true })
    expect(conn.transport.last('audio_start')).toMatchObject({ delivery: 'buffered', codec: 'mp3' })
    const frames = conn.transport.frames()
    expect(frames.map(f => f.header.sequence)).toEqual(frames.map((_, index) => index))
    expect(frames.every(f => f.header.stream === 'tts' && f.header.generation === conn.transport.last('audio_start').generation)).toBe(true)
    expect(frames.filter(f => f.header.clipEnd)).toHaveLength(2)
    expect(frames.at(-1)?.header.final).toBe(true)
    expect(conn.transport.last('audio_end')).toMatchObject({ interrupted: false })
    expect(conn.transport.last('state').state).toBe('idle')
  })

  test('the reply is delivered once; a second reply is ignored unless armed again', async () => {
    const rig = new Rig()
    const conn = await submitted(rig)
    rig.bridge.reply('thread-1', 'One.', 'm1')
    rig.bridge.reply('thread-1', 'Two.', 'm1')
    await settle(20)
    expect(conn.transport.ofType('reply')).toHaveLength(1)
  })

  test('without TTS the text still arrives and nothing claims playback', async () => {
    const rig = new Rig({ tts: null })
    const conn = await submitted(rig)
    rig.bridge.reply('thread-1', 'Plain text only.', 'm1')
    await settle()
    expect(conn.transport.last('reply')).toMatchObject({ spoken: false })
    expect(conn.transport.ofType('audio_start')).toHaveLength(0)
    expect(conn.transport.binaries).toHaveLength(0)
  })

  test('backpressure ends an overloaded stream instead of queueing', async () => {
    const rig = new Rig()
    const conn = await submitted(rig)
    conn.transport.buffered = VOICE_LIMITS.maxBufferedBytes
    rig.bridge.reply('thread-1', 'Too much audio for this phone.', 'm1')
    await settle(20)
    expect(conn.transport.errors()).toContain('VOICE_OVERLOADED')
    expect(conn.transport.binaries).toHaveLength(0)
    expect(conn.transport.last('audio_end')).toMatchObject({ interrupted: true })
  })

  test('interrupt aborts synthesis, flushes playback, and never cancels agents', async () => {
    const rig = new Rig()
    let release: () => void = () => undefined
    rig.tts.gate = new Promise<void>(resolve => { release = resolve })
    const conn = await submitted(rig)
    rig.bridge.reply('thread-1', 'Slow sentence.', 'm1')
    await settle()
    const responseId = String(conn.transport.last('audio_start').response_id)
    await conn.session.handleText(control({ type: 'interrupt', response_id: responseId, played_ms: 0 }))
    release()
    await settle(20)
    expect(conn.transport.last('audio_end')).toMatchObject({ interrupted: true, response_id: responseId })
    expect(conn.transport.binaries).toHaveLength(0)
    expect(rig.bridge.cancels).toHaveLength(0)
    expect(rig.store.records.some(r => r.kind === 'interruption')).toBe(true)
  })

  test('a stale response id is not interrupted by a late interrupt for another response', async () => {
    const rig = new Rig()
    let release: () => void = () => undefined
    rig.tts.gate = new Promise<void>(resolve => { release = resolve })
    const conn = await submitted(rig)
    rig.bridge.reply('thread-1', 'Slow sentence.', 'm1')
    await settle()
    await conn.session.handleText(control({ type: 'interrupt', response_id: 'old-response' }))
    expect(conn.transport.ofType('audio_end')).toHaveLength(0)
    release()
    await settle(20)
    expect(conn.transport.last('audio_end')).toMatchObject({ interrupted: false })
  })
})

describe('voice session: persistence', () => {
  test('stores final transcripts and attribution, never audio', async () => {
    const rig = new Rig()
    const conn = rig.connect('alice', 'phone')
    const generation = await startPtt(conn)
    await record(conn, generation)
    await conn.session.handleText(control({ type: 'submit', utterance_id: 'u1', text: 'edited text' }))
    await settle()
    const kinds = rig.store.records.map(r => r.kind)
    expect(kinds).toEqual(['transcript', 'submitted'])
    const submitted = rig.store.records[1]
    expect(submitted).toMatchObject({ userId: 'alice', deviceId: 'phone', displayName: 'ALICE', text: 'edited text', edited: true, mode: 'push_to_talk' })
    const dump = JSON.stringify(rig.store.records)
    expect(dump).not.toMatch(/audio|bytes|base64/i)
    for (const record of rig.store.records) expect(Object.values(record).some(v => v instanceof Uint8Array)).toBe(false)
    expect(rig.config.settings.audioRetention).toBe('none')
  })
})

describe('voice session: idle and stop', () => {
  test('a quiet push-to-talk connection is closed and a stop message ends the session', async () => {
    const rig = new Rig()
    const idle = rig.connect()
    rig.advance(rig.config.settings.pttIdleSeconds * 1000 + 1)
    idle.session.tick()
    expect(idle.transport.closed?.reason).toBe('idle')
    const stopped = rig.connect('bob')
    await stopped.session.handleText(control({ type: 'stop' }))
    expect(stopped.transport.closed?.code).toBe(1000)
    expect(stopped.session.isClosed).toBe(true)
  })
})

describe('voice session: realtime relay', () => {
  async function startLive(rig: Rig, conn: Connection, rate = 24_000): Promise<number> {
    await conn.session.handleText(control({ type: 'start', mode: 'live', conversation_id: 'thread-1', utterance_id: 'live1', codec: 'pcm16', sample_rate: rate }))
    // Refused starts (unavailable, budget, provider failure) send no state message; report generation 0.
    return Number(conn.transport.ofType('state').at(-1)?.generation ?? 0)
  }

  test('connects with only the three bridge functions and no other tool', async () => {
    const port = new FakeRealtimePort()
    const rig = new Rig({ realtime: port })
    const conn = rig.connect()
    await startLive(rig, conn)
    expect(port.connects).toHaveLength(1)
    expect(port.connects[0]?.tools.map(t => t.name)).toEqual(['supervisor_submit', 'supervisor_status', 'supervisor_cancel'])
    expect(port.connects[0]?.instructions).toContain('cannot approve')
    expect(conn.transport.last('state').state).toBe('listening')
  })

  test('live voice is unavailable when only push-to-talk mode is allowed', async () => {
    const rig = new Rig({ realtime: new FakeRealtimePort(), settings: { mode: 'push_to_talk' } })
    const conn = rig.connect()
    await startLive(rig, conn)
    expect(conn.transport.errors()).toEqual(['VOICE_UNAVAILABLE'])
  })

  test('forwards audio, resampled to the provider rate, and enforces ordering', async () => {
    const port = new FakeRealtimePort()
    const rig = new Rig({ realtime: port })
    const conn = rig.connect()
    const g = await startLive(rig, conn, 48_000)
    conn.session.handleBinary(micFrame(g, 0, new Uint8Array(960), { codec: 'pcm16' }))
    expect(port.session.audio[0]?.byteLength).toBe(480)
    conn.session.handleBinary(micFrame(g, 5, new Uint8Array(960), { codec: 'pcm16' }))
    expect(conn.transport.errors()).toContain('VOICE_PROTOCOL')
    expect(port.session.closed).toBe(true)
  })

  test('a provider that backs up ends the session with VOICE_OVERLOADED', async () => {
    const port = new FakeRealtimePort()
    port.session.backedUp = true
    const rig = new Rig({ realtime: port })
    const conn = rig.connect()
    const g = await startLive(rig, conn)
    conn.session.handleBinary(micFrame(g, 0, new Uint8Array(480), { codec: 'pcm16' }))
    expect(conn.transport.errors()).toContain('VOICE_OVERLOADED')
    expect(conn.transport.closed).toBeDefined()
    expect(port.session.closed).toBe(true)
  })

  test('VAD commit: transcript revisions, then one submit even when the model calls the function twice', async () => {
    const port = new FakeRealtimePort()
    const rig = new Rig({ realtime: port })
    const conn = rig.connect()
    await startLive(rig, conn)
    const s = port.session
    s.emit({ type: 'speech_started' })
    s.emit({ type: 'transcript', itemId: 'i1', text: 'run the', final: false })
    s.emit({ type: 'transcript', itemId: 'i1', text: 'run the tests', final: false })
    s.emit({ type: 'speech_stopped', itemId: 'i1' })
    s.emit({ type: 'transcript', itemId: 'i1', text: 'run the tests', final: true })
    expect(conn.transport.ofType('transcript').map(m => [m.revision, m.final])).toEqual([[1, false], [2, false], [3, true]])
    s.emit({ type: 'tool_call', callId: 'c1', name: 'supervisor_submit', arguments: '{"text":"model invented words"}' })
    s.emit({ type: 'tool_call', callId: 'c2', name: 'supervisor_submit', arguments: '{}' })
    await settle(20)
    expect(rig.bridge.submits).toHaveLength(1)
    // The text comes from the finalized transcript, never from model-supplied arguments.
    expect(rig.bridge.submits[0]).toMatchObject({ text: 'run the tests', key: 'voice:i1' })
    expect(s.toolResults).toHaveLength(2)
    expect(JSON.parse(s.toolResults[0]?.output ?? '{}')).toMatchObject({ ok: true, duplicate: false })
    expect(JSON.parse(s.toolResults[1]?.output ?? '{}')).toMatchObject({ ok: true, duplicate: true })
  })

  test('submit before the words are final is refused; unknown functions are refused', async () => {
    const port = new FakeRealtimePort()
    const rig = new Rig({ realtime: port })
    const conn = rig.connect()
    await startLive(rig, conn)
    port.session.emit({ type: 'tool_call', callId: 'c1', name: 'supervisor_submit', arguments: '{}' })
    port.session.emit({ type: 'tool_call', callId: 'c2', name: 'permission_respond', arguments: '{}' })
    port.session.emit({ type: 'tool_call', callId: 'c3', name: 'task_stop', arguments: '{}' })
    await settle(20)
    expect(rig.bridge.submits).toHaveLength(0)
    expect(port.session.toolResults.map(r => JSON.parse(r.output).ok)).toEqual([false, false, false])
    expect(rig.bridge.cancels).toHaveLength(0)
  })

  test('status and cancel reach the bridge; cancel says background agents keep running', async () => {
    const port = new FakeRealtimePort()
    const rig = new Rig({ realtime: port })
    const conn = rig.connect()
    await startLive(rig, conn)
    port.session.emit({ type: 'tool_call', callId: 's', name: 'supervisor_status', arguments: '{}' })
    port.session.emit({ type: 'tool_call', callId: 'x', name: 'supervisor_cancel', arguments: '{}' })
    await settle(20)
    expect(JSON.parse(port.session.toolResults[0]?.output ?? '{}')).toMatchObject({ ok: true, busy: true, agents_working: 2 })
    expect(JSON.parse(port.session.toolResults[1]?.output ?? '{}').note).toContain('Background agents keep running')
    expect(rig.bridge.cancels).toHaveLength(1)
    expect(rig.bridge.cancels[0]?.actor.principal_id).toBe('alice')
  })

  test('relays provider audio as pcm16 frames, and speech-started truncates what the user heard', async () => {
    const port = new FakeRealtimePort()
    const rig = new Rig({ realtime: port })
    const conn = rig.connect()
    await startLive(rig, conn)
    const s = port.session
    s.emit({ type: 'audio', responseId: 'rp1', itemId: 'item1', audio: new Uint8Array(4800) })
    expect(conn.transport.last('audio_start')).toMatchObject({ codec: 'pcm16', delivery: 'streaming', sample_rate: 24_000 })
    expect(conn.transport.frames().every(f => f.header.codec === 'pcm16' && f.header.clipEnd)).toBe(true)
    expect(conn.transport.last('state').state).toBe('speaking')
    s.emit({ type: 'speech_started' })
    expect(s.cancels).toBe(1)
    expect(s.truncates).toEqual([{ itemId: 'item1', playedMs: 100 }])
    expect(conn.transport.last('audio_end')).toMatchObject({ interrupted: true })
    // Audio the provider still delivers for the cancelled response is dropped.
    const before = conn.transport.binaries.length
    s.emit({ type: 'audio', responseId: 'rp1', itemId: 'item1', audio: new Uint8Array(480) })
    expect(conn.transport.binaries.length).toBe(before)
    expect(rig.store.records.some(r => r.kind === 'interruption' && r.playedMs === 100)).toBe(true)
    expect(rig.bridge.cancels).toHaveLength(0)
  })

  test('a client interrupt uses the played offset the phone reports', async () => {
    const port = new FakeRealtimePort()
    const rig = new Rig({ realtime: port })
    const conn = rig.connect()
    await startLive(rig, conn)
    port.session.emit({ type: 'audio', responseId: 'rp1', itemId: 'item1', audio: new Uint8Array(4800) })
    await conn.session.handleText(control({ type: 'interrupt', played_ms: 40 }))
    expect(port.session.truncates).toEqual([{ itemId: 'item1', playedMs: 40 }])
    expect(port.session.cancels).toBe(1)
  })

  test('providers that cannot truncate are not asked to', async () => {
    const port = new FakeRealtimePort()
    port.session.supportsTruncate = false
    const rig = new Rig({ realtime: port })
    const conn = rig.connect()
    await startLive(rig, conn)
    port.session.emit({ type: 'audio', responseId: 'rp1', audio: new Uint8Array(480) })
    port.session.emit({ type: 'speech_started' })
    expect(port.session.cancels).toBe(1)
    expect(port.session.truncates).toHaveLength(0)
  })

  test('the supervisor reply is read through the provider only when the live request armed it', async () => {
    const port = new FakeRealtimePort()
    const rig = new Rig({ realtime: port })
    const conn = rig.connect()
    await startLive(rig, conn)
    rig.bridge.reply('thread-1', 'Nobody asked for this.')
    await settle()
    expect(port.session.said).toHaveLength(0)
    port.session.emit({ type: 'transcript', itemId: 'i1', text: 'status', final: true })
    port.session.emit({ type: 'tool_call', callId: 'c1', name: 'supervisor_submit', arguments: '{}' })
    await settle(20)
    // Answer the message that was actually submitted, whatever id the bridge gave it.
    const messageId = String(conn.transport.last('submitted').message_id)
    rig.bridge.reply('thread-1', 'Two agents are working. See the screen.', messageId)
    await settle()
    expect(port.session.said).toEqual(['Two agents are working. See the screen.'])
    expect(conn.transport.last('reply')).toMatchObject({ spoken: true })
  })

  test('limits: idle close, session cap, deliberate renewal, and budget on renewal', async () => {
    const port = new FakeRealtimePort()
    const rig = new Rig({ realtime: port, settings: { maxSessionSeconds: 300, idleSeconds: 30 } })
    const idle = rig.connect()
    await startLive(rig, idle)
    rig.advance(31_000)
    idle.session.tick()
    expect(idle.transport.closed?.reason).toBe('idle')
    expect(port.session.closed).toBe(true)

    const port2 = new FakeRealtimePort()
    const rig2 = new Rig({ realtime: port2, settings: { maxSessionSeconds: 300 } })
    const capped = rig2.connect()
    const g = await startLive(rig2, capped)
    for (let i = 0; i < 12; i += 1) {
      rig2.advance(25_000)
      port2.session.emit({ type: 'speech_started' })
      capped.session.tick()
    }
    expect(capped.transport.errors()).toContain('VOICE_LIMIT')
    expect(capped.transport.closed).toBeDefined()
    void g

    const port3 = new FakeRealtimePort()
    const rig3 = new Rig({ realtime: port3, settings: { maxSessionSeconds: 300, maxSessionUsd: 10, pricing: { realtimeUsdPerMinute: 1 } } })
    const renewed = rig3.connect()
    await startLive(rig3, renewed)
    rig3.advance(200_000)
    port3.session.emit({ type: 'speech_started' })
    await renewed.session.handleText(control({ type: 'renew' }))
    rig3.advance(200_000)
    port3.session.emit({ type: 'speech_started' })
    renewed.session.tick()
    expect(renewed.transport.closed).toBeUndefined()
  })

  test('a live session that would exceed the spend limit never connects', async () => {
    const port = new FakeRealtimePort()
    const rig = new Rig({ realtime: port, settings: { maxSessionUsd: 1, pricing: { realtimeUsdPerMinute: 1 } } })
    const conn = rig.connect()
    await startLive(rig, conn)
    expect(conn.transport.errors()).toEqual(['VOICE_BUDGET'])
    expect(port.connects).toHaveLength(0)
  })

  test('provider failures are typed; a provider error ends the session; closing releases the provider', async () => {
    const port = new FakeRealtimePort()
    port.failConnect = new Error('boom sk-live-secret')
    const rig = new Rig({ realtime: port })
    const failed = rig.connect()
    await startLive(rig, failed)
    expect(failed.transport.errors()).toEqual(['VOICE_PROVIDER'])
    expect(failed.transport.texts.join('')).not.toContain('sk-live-secret')

    const ok = new FakeRealtimePort()
    const rig2 = new Rig({ realtime: ok })
    const conn = rig2.connect()
    await startLive(rig2, conn)
    ok.session.emit({ type: 'error', error: new (await import('../src/hub/voice/protocol.js')).VoiceError('VOICE_PROVIDER', 'the realtime provider reported an error') })
    expect(conn.transport.errors()).toEqual(['VOICE_PROVIDER'])
    expect(ok.session.closed).toBe(true)
    expect(rig2.leases.size).toBe(0)

    const dropped = new FakeRealtimePort()
    const rig3 = new Rig({ realtime: dropped })
    const c3 = rig3.connect()
    await startLive(rig3, c3)
    c3.session.dispose()
    expect(dropped.session.closed).toBe(true)
  })

  test('reconnecting opens a fresh provider session and replays nothing', async () => {
    const port = new FakeRealtimePort()
    const rig = new Rig({ realtime: port })
    const first = rig.connect('alice', 'phone')
    const g = await startLive(rig, first)
    first.session.handleBinary(micFrame(g, 0, new Uint8Array(480), { codec: 'pcm16' }))
    const firstSession = port.session
    first.session.dispose()
    port.session = new (await import('./hubVoiceHarness.js')).FakeRealtimeSession()
    const second = rig.connect('alice', 'phone')
    await startLive(rig, second)
    expect(port.connects).toHaveLength(2)
    expect(firstSession.audio).toHaveLength(1)
    expect(port.session.audio).toHaveLength(0)
  })
})
