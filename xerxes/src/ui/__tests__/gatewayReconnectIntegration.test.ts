// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import type { Socket } from 'node:net'
import { expect, it, vi } from 'vitest'
import { InMemoryDaemonRuntime, type TurnRunner } from '../../daemon/runtime.js'
import { DaemonServer } from '../../daemon/server.js'
import { DaemonInteractionBoard } from '../../daemon/interactions.js'
import type { ConnectionLeases } from '../../daemon/connectionLease.js'
import { GatewayClient } from '../gatewayClient.js'
import type { AnyEvent, SessionCreateResponse, SessionResumeResponse } from '../gatewayTypes.js'

function gate() {
  let release!: () => void
  const promise = new Promise<void>(resolve => { release = resolve })
  return { promise, release }
}

async function host(runner: TurnRunner, interactions?: DaemonInteractionBoard) {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xr-replay-')))
  const socketPath = join(directory, 'rpc.sock')
  const runtime = new InMemoryDaemonRuntime(runner, { model: 'fixture', currentProjectDirectory: directory,
    sessionDirectory: join(directory, 'sessions'), interactions })
  const server = new DaemonServer({ runtime, socketPath, projectDirectory: directory, interactions })
  const client = new GatewayClient({ externalSocketPath: socketPath, projectDir: directory })
  await server.start()
  await client.start()
  const session = await client.request<SessionCreateResponse>('session.create')
  return { client, runtime, session, server, async close() { client.close(); await server.stop(); await rm(directory, { force: true, recursive: true }) } }
}

async function drop(client: GatewayClient) {
  ;(client as unknown as { socket: Socket }).socket.destroy()
  await vi.waitFor(() => expect(client.connected).toBe(false))
  // Wait for the real server to observe EOF before reclaiming its owner.
  await Bun.sleep(20)
}

it('keeps the real daemon turn alive and delivers missed deltas once, before new live output', async () => {
  const continueOffline = gate(), offlineWritten = gate(), finish = gate(), finalWritten = gate()
  const fixture = await host({ async *run(_session, _text, signal) {
    yield { type: 'text_part', payload: { text: 'before ' } }
    await continueOffline.promise
    expect(signal.aborted).toBe(false)
    yield { type: 'text_part', payload: { text: 'offline ' } }
    offlineWritten.release()
    await finish.promise
    yield { type: 'text_part', payload: { text: 'after' } }
    finalWritten.release()
  } })
  const { client, session } = fixture
  const deltas: string[] = [], transcripts: AnyEvent[] = []
  client.on('message.delta', event => deltas.push(event.payload.text))
  client.on('transcript.append', event => transcripts.push(event))
  try {
    expect(client.hasConnectionLease).toBe(true)
    await client.request('turn.submit', { text: 'stream' })
    await vi.waitFor(() => expect(deltas).toEqual(['before ']))
    await drop(client)
    continueOffline.release()
    await offlineWritten.promise
    await client.start()
    const restored = await client.request<SessionResumeResponse>('session.resume', { session_id: session.session_id, preserve_view: true })
    expect(restored.reconnected).toBe(true)
    expect(restored.running).toBe(true)
    expect(deltas).toEqual(['before '])
    finish.release()
    await finalWritten.promise
    // A reply on the same ordered socket fences live events that arrived
    // after initialize but before React adopted the restored session.
    await client.request('runtime.status')
    expect(deltas).toEqual(['before '])
    client.finishSessionRecovery('wrong-session')
    expect(deltas).toEqual(['before '])
    client.finishSessionRecovery(session.session_id)
    client.finishSessionRecovery(session.session_id)
    expect(deltas).toEqual(['before ', 'offline ', 'after'])
    expect(transcripts).toEqual([])
  } finally { continueOffline.release(); finish.release(); await fixture.close() }
})

it('restores only the current question after reclaim and keeps response ownership', async () => {
  const board = new DaemonInteractionBoard()
  const fixture = await host({ async *run(session, _text, signal) {
    const answer = await board.ask(session.id, { question: 'Continue?', options: ['yes', 'no'] }, signal)
    yield { type: 'text_part', payload: { text: answer } }
  } }, board)
  const { client, session } = fixture
  const questions: AnyEvent[] = [], answers: string[] = []
  client.on('event', event => { if (event.type === 'clarify.request') questions.push(event) })
  client.on('message.delta', event => answers.push(event.payload.text))
  try {
    await client.request('turn.submit', { text: 'question' })
    await vi.waitFor(() => expect(questions).toHaveLength(1))
    await drop(client)
    await client.start()
    const restored = await client.request<SessionResumeResponse>('session.resume', { session_id: session.session_id, preserve_view: true })
    expect(restored.reconnected).toBe(true)
    client.finishSessionRecovery(session.session_id)
    client.finishSessionRecovery(session.session_id)
    expect(questions).toHaveLength(2)
    expect(questions[1]).toEqual(questions[0])
    const pending = board.pendingQuestionIds()[0]!
    expect(await client.request('question_response', { request_id: pending, answers: { answer: 'yes' } })).toMatchObject({ ok: true })
    await vi.waitFor(() => expect(answers).toEqual(['yes']))
  } finally { await fixture.close() }
})

it('restores an authoritative snapshot when another drop discards an undelivered journal', async () => {
  const offline = gate(), written = gate(), finish = gate()
  const fixture = await host({ async *run() {
    yield { type: 'text_part', payload: { text: 'before ' } }
    await offline.promise
    yield { type: 'text_part', payload: { text: 'offline' } }
    written.release()
    await finish.promise
  } })
  const { client, session } = fixture
  const deltas: string[] = []
  client.on('message.delta', event => deltas.push(event.payload.text))
  try {
    await client.request('turn.submit', { text: 'stream' })
    await vi.waitFor(() => expect(deltas).toEqual(['before ']))
    await drop(client)
    offline.release()
    await written.promise
    await client.start()
    const first = await client.request<SessionResumeResponse>('session.resume', { session_id: session.session_id, preserve_view: true })
    expect(first.reconnected).toBe(true)
    // Lose transport before the owning UI consumes the first journal.
    await drop(client)
    await client.start()
    const second = await client.request<SessionResumeResponse>('session.resume', { session_id: session.session_id, preserve_view: true })
    expect(second.reconnected).toBe(false)
    expect(second.running).toBe(true)
    expect(JSON.stringify(second.inflight)).toContain('before offline')
    expect(deltas).toEqual(['before '])
  } finally { offline.release(); finish.release(); await fixture.close() }
})

it('reattaches to running session-owned work after lease expiry and still supports explicit stop', async () => {
  let calls = 0, cancelled = false
  const fixture = await host({ async *run(_session, _text, signal) {
    calls++
    yield { type: 'text_part', payload: { text: 'retained before expiry' } }
    await new Promise<void>(resolve => signal.addEventListener('abort', () => { cancelled = true; resolve() }, { once: true }))
  } })
  const { client, session, server } = fixture
  try {
    await client.request('turn.submit', { text: 'wait' })
    await vi.waitFor(() => expect(calls).toBe(1))
    await drop(client)
    // Drive the real expiry cleanup directly; do not spend 30s on a timer test.
    ;(server as unknown as { connectionLeases: ConnectionLeases }).connectionLeases.close()
    expect(cancelled).toBe(false)
    await client.start()
    const restored = await client.request<SessionResumeResponse>('session.resume', { session_id: session.session_id, preserve_view: true })
    expect(restored.reconnected).toBe(false)
    expect(restored.running).toBe(true)
    expect(JSON.stringify(restored.inflight)).toContain('retained before expiry')
    expect(calls).toBe(1)
    expect(client.hasConnectionLease).toBe(true)
    await client.request('turn.cancel')
    await vi.waitFor(() => expect(cancelled).toBe(true))
  } finally { await fixture.close() }
})

it('failed explicit resume preserves the real daemon and client selection for the next turn', async () => {
  const owners: string[] = []
  const fixture = await host({ async *run(session) { owners.push(session.id); yield { type: 'text_part', payload: { text: 'same conversation' } } } })
  try {
    const before = fixture.runtime.listSessions().map(session => session.id)
    await expect(fixture.client.request('session.resume', { session_id: 'missing123' })).rejects.toThrow('saved conversation is missing')
    expect(fixture.runtime.listSessions().map(session => session.id)).toEqual(before)
    await fixture.client.request('turn.submit', { text: 'keep working here' })
    await vi.waitFor(() => expect(owners).toEqual([fixture.session.session_id]))
    expect(fixture.runtime.sessionStatus('missing123')).toBeUndefined()
  } finally { await fixture.close() }
})
