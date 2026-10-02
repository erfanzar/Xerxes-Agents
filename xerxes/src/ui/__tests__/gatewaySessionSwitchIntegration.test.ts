// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { expect, it, vi } from 'vitest'
import { InMemoryDaemonRuntime, type TurnRunner } from '../../daemon/runtime.js'
import { DaemonServer } from '../../daemon/server.js'
import { DaemonInteractionBoard } from '../../daemon/interactions.js'
import { queueSessionNotification } from '../../daemon/sessionNotifications.js'
import { GatewayClient } from '../gatewayClient.js'
import type { AnyEvent, SessionActivateResponse, SessionCreateResponse } from '../gatewayTypes.js'

function gate() {
  let release!: () => void
  const promise = new Promise<void>(resolve => { release = resolve })
  return { promise, release }
}

async function host(runner: TurnRunner, interactions?: DaemonInteractionBoard) {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xr-switch-')))
  const socketPath = join(directory, 'rpc.sock')
  const runtime = new InMemoryDaemonRuntime(runner, { model: 'fixture', currentProjectDirectory: directory,
    sessionDirectory: join(directory, 'sessions'), interactions })
  const server = new DaemonServer({ runtime, socketPath, projectDirectory: directory, interactions })
  const client = new GatewayClient({ externalSocketPath: socketPath, projectDir: directory })
  await server.start()
  await client.start()
  const first = await client.request<SessionCreateResponse>('session.create')
  return { client, first, runtime, server, async close() { client.close(); await server.stop(); await rm(directory, { force: true, recursive: true }) } }
}

it('re-offers a question the background tab raised while the user watched another tab', async () => {
  const board = new DaemonInteractionBoard()
  const ask = gate()
  const fixture = await host({ async *run(session, _text, signal) {
    await ask.promise
    const answer = await board.ask(session.id, { question: 'Continue?', options: ['yes', 'no'] }, signal)
    yield { type: 'text_part', payload: { text: answer } }
  } }, board)
  const { client, first } = fixture
  const questions: AnyEvent[] = [], answers: string[] = []
  client.on('event', event => { if (event.type === 'clarify.request') questions.push(event) })
  client.on('message.delta', event => answers.push(event.payload.text))
  try {
    await client.request('turn.submit', { text: 'question' })
    await client.request<SessionCreateResponse>('session.create')
    ask.release()
    await vi.waitFor(() => expect(board.pendingQuestionIds()).toHaveLength(1))
    // The question went only to observers of the first session.
    await client.request('runtime.status')
    expect(questions).toEqual([])

    const back = await client.request<SessionActivateResponse>('session.activate', { session_id: first.session_id })
    expect(back.recovery_pending).toBe(true)
    client.finishSessionRecovery(back.session_id)
    expect(questions).toHaveLength(1)

    const pending = board.pendingQuestionIds()[0]!
    expect(await client.request('question_response', { request_id: pending, answers: { answer: 'yes' } })).toMatchObject({ ok: true })
    await vi.waitFor(() => expect(answers).toEqual(['yes']))
  } finally { ask.release(); await fixture.close() }
})

it('holds the activated session\'s stream until the view has been reset onto it', async () => {
  const more = gate(), written = gate()
  const fixture = await host({ async *run() {
    yield { type: 'text_part', payload: { text: 'one ' } }
    await more.promise
    yield { type: 'text_part', payload: { text: 'two' } }
    written.release()
  } })
  const { client, first } = fixture
  const deltas: string[] = []
  client.on('message.delta', event => deltas.push(event.payload.text))
  try {
    await client.request('turn.submit', { text: 'stream' })
    await vi.waitFor(() => expect(deltas).toEqual(['one ']))
    await client.request<SessionCreateResponse>('session.create')

    const back = await client.request<SessionActivateResponse>('session.activate', { session_id: first.session_id })
    expect(back.inflight?.assistant).toBe('one ')
    // Output produced after session.open moved the connection but before the
    // UI committed the snapshot must not reach the view it is about to wipe.
    more.release()
    await written.promise
    await client.request('runtime.status')
    expect(deltas).toEqual(['one '])

    client.finishSessionRecovery(back.session_id)
    expect(deltas).toEqual(['one ', 'two'])
  } finally { more.release(); await fixture.close() }
})

it('does not replay output that session.open already folded into its snapshot', async () => {
  const more = gate(), written = gate(), last = gate()
  const fixture = await host({ async *run() {
    yield { type: 'text_part', payload: { text: 'one ' } }
    await more.promise
    yield { type: 'text_part', payload: { text: 'two' } }
    written.release()
    await last.promise
    yield { type: 'text_part', payload: { text: ' three' } }
  } })
  const { client, first, server } = fixture
  const deltas: string[] = []
  client.on('message.delta', event => deltas.push(event.payload.text))
  try {
    await client.request('turn.submit', { text: 'stream' })
    await vi.waitFor(() => expect(deltas).toEqual(['one ']))
    await client.request<SessionCreateResponse>('session.create')

    // session.open has already moved the connection onto the first session
    // when it awaits the goal re-arm; output streamed there reaches the client
    // and is also part of the snapshot built after the await.
    const internals = server as unknown as { rearmGoalAfterRestart: (...args: unknown[]) => Promise<void> }
    const rearm = internals.rearmGoalAfterRestart.bind(server)
    internals.rearmGoalAfterRestart = async (...args: unknown[]) => {
      more.release()
      await written.promise
      return rearm(...args)
    }
    const back = await client.request<SessionActivateResponse>('session.activate', { session_id: first.session_id })
    expect(back.inflight?.assistant).toBe('one two')

    client.finishSessionRecovery(back.session_id)
    expect(deltas).toEqual(['one '])
    // Output after the answer still flows onto the committed view.
    last.release()
    await vi.waitFor(() => expect(deltas).toEqual(['one ', ' three']))
  } finally { more.release(); last.release(); await fixture.close() }
})

it('delivers notices drained by session.open after the switch commits, not before', async () => {
  const fixture = await host({ async *run() { yield { type: 'text_part', payload: { text: 'ok' } } } })
  const { client, first, runtime } = fixture
  const notices: string[] = []
  client.on('event', event => {
    if (event.type === 'notification.show') notices.push(String((event.payload as { text?: unknown }).text ?? ''))
  })
  try {
    await client.request<SessionCreateResponse>('session.create')
    const parent = runtime.listSessions().find(session => session.id === first.session_id)!
    queueSessionNotification(parent.metadata, { at: Date.now(), level: 'error', message: 'Background task bg-1 failed: boom' })

    const back = await client.request<SessionActivateResponse>('session.activate', { session_id: first.session_id })
    // The drain is at-most-once; shown now it would be cleared by the reset
    // that follows and never offered again.
    expect(notices).toEqual([])
    client.finishSessionRecovery(back.session_id)
    expect(notices.join('\n')).toContain('Background task bg-1 failed: boom')
  } finally { await fixture.close() }
})
