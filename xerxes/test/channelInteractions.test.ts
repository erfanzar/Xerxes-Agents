// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ChannelManager,
  ChannelTurnRouter,
  MessageDirection,
  createChannelMessage,
  type Channel,
  type ChannelInteractionPort,
  type ChannelMessage,
  type InboundHandler,
} from '../src/channels/index.js'
import type {
  DaemonEvent,
  DaemonRuntime,
  DaemonSession,
  OpenSessionOptions,
} from '../src/daemon/runtime.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'

class RecordingChannel implements Channel {
  readonly name = 'telegram'
  readonly sent: ChannelMessage[] = []
  private inbound: InboundHandler | undefined

  async send(message: ChannelMessage): Promise<void> {
    this.sent.push(message)
  }

  async start(onInbound: InboundHandler): Promise<void> {
    this.inbound = onInbound
  }

  async stop(): Promise<void> {
    this.inbound = undefined
  }

  async receive(message: ChannelMessage): Promise<void> {
    if (!this.inbound) throw new Error('channel has not been enabled')
    await this.inbound(message)
  }
}

class FakeInteractionPort implements ChannelInteractionPort {
  readonly permissions: Array<{ readonly requestId: string; readonly response: string }> = []
  readonly questions: Array<{ readonly requestId: string; readonly answers: Readonly<Record<string, string>> }> = []
  permissionOutcome = true
  private waiters: Array<() => void> = []

  respondPermission(requestId: string, response: string): boolean {
    if (!this.permissionOutcome) return false
    this.permissions.push({ requestId, response })
    this.flush()
    return true
  }

  respondQuestion(requestId: string, answers: Readonly<Record<string, string>>): boolean {
    this.questions.push({ requestId, answers })
    this.flush()
    return true
  }

  /** Resolve once the channel has answered the parked request. */
  nextAnswer(): Promise<void> {
    return new Promise(resolve => this.waiters.push(resolve))
  }

  private flush(): void {
    const waiters = this.waiters.splice(0)
    for (const resolve of waiters) resolve()
  }
}

/**
 * Runtime whose turn raises one approval or question request and then parks
 * until the interaction port records the channel's answer — the same await
 * the real loop performs on its permission broker.
 */
class InteractionRuntime implements DaemonRuntime {
  readonly submitted: Array<{ readonly key: string; readonly prompt: string }> = []
  private readonly sessions = new Map<string, DaemonSession>()

  constructor(private readonly port: FakeInteractionPort) {}

  cancelAllTurns(): number {
    return 0
  }

  cancelTurn(): boolean {
    return false
  }

  evictSession(): void {}

  async flushSessions(): Promise<void> {}

  async listSavedSessions(): Promise<readonly []> {
    return []
  }

  listSessions(): readonly DaemonSession[] {
    return [...this.sessions.values()]
  }

  async openSession(sessionKey: string, agentId = 'default', _options: OpenSessionOptions = {}): Promise<DaemonSession> {
    const existing = this.sessions.get(sessionKey)
    if (existing) return existing
    const session = {
      agentId,
      id: sessionKey,
      sessionKey,
      cwd: '/workspace',
      messages: [],
      metadata: {},
      status: 'idle',
    } as unknown as DaemonSession
    this.sessions.set(sessionKey, session)
    return session
  }

  reload(): JsonRpcPayload {
    return {}
  }

  async setSessionMode(): Promise<undefined> {
    return undefined
  }

  sessionStatus(sessionKey: string): DaemonSession | undefined {
    return this.sessions.get(sessionKey)
  }

  steerTurn(): boolean {
    return false
  }

  status(): JsonRpcPayload {
    return { runtime: 'bun-typescript' }
  }

  async submitTurn(sessionKey: string, text: string, emit: (event: DaemonEvent) => void): Promise<void> {
    this.submitted.push({ key: sessionKey, prompt: text })
    if (text.includes('ask something')) {
      emit({
        type: 'question_request',
        payload: {
          id: 'question-1',
          tool_call_id: '',
          questions: [{ id: 'answer', question: 'Pick a lane', options: ['alpha', 'beta'] }],
        },
      })
      await this.port.nextAnswer()
      emit({ type: 'text_part', payload: { text: 'lane recorded' } })
      return
    }
    emit({
      type: 'approval_request',
      payload: { id: 'perm-1', name: 'Bash', description: 'Run: rm -rf build' },
    })
    await this.port.nextAnswer()
    emit({ type: 'text_part', payload: { text: 'command finished' } })
  }
}

function inbound(text: string): ChannelMessage {
  return createChannelMessage({
    channel: 'telegram',
    channelUserId: 'user-7',
    direction: MessageDirection.INBOUND,
    roomId: 'chat-9',
    text,
  })
}

async function waitForText(channel: RecordingChannel, marker: string): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (channel.sent.some(message => message.text.includes(marker))) return
    await Bun.sleep(5)
  }
  throw new Error(`channel never received a message containing '${marker}'`)
}

function harness(): {
  channel: RecordingChannel
  port: FakeInteractionPort
  runtime: InteractionRuntime
} {
  const channel = new RecordingChannel()
  const port = new FakeInteractionPort()
  const runtime = new InteractionRuntime(port)
  return { channel, port, runtime }
}

test('approval requests reach the channel and a yes answer releases the parked turn', async () => {
  const { channel, port, runtime } = await enable(harness())

  const turn = channel.receive(inbound('run it'))
  await waitForText(channel, 'Approval needed: Bash')
  // The turn is parked on the answer; nothing else was submitted meanwhile.
  expect(runtime.submitted).toHaveLength(1)

  await channel.receive(inbound('yes'))
  await turn

  expect(port.permissions).toEqual([{ requestId: 'perm-1', response: 'approve' }])
  expect(channel.sent.map(message => message.text)).toEqual([
    'Approval needed: Bash\nRun: rm -rf build\nReply yes to approve once, session to approve for this session, or no to deny.',
    'Approved.',
    'command finished',
  ])
})

test('conversational answers map onto deny and session-scope decisions', async () => {
  const denied = await enable(harness())
  const deniedTurn = denied.channel.receive(inbound('run it'))
  await waitForText(denied.channel, 'Approval needed')
  await denied.channel.receive(inbound('no'))
  await deniedTurn
  expect(denied.port.permissions).toEqual([{ requestId: 'perm-1', response: 'reject' }])
  expect(denied.channel.sent.some(message => message.text === 'Denied.')).toBe(true)

  const session = await enable(harness())
  const sessionTurn = session.channel.receive(inbound('run it'))
  await waitForText(session.channel, 'Approval needed')
  await session.channel.receive(inbound('session'))
  await sessionTurn
  expect(session.port.permissions).toEqual([{ requestId: 'perm-1', response: 'approve_for_session' }])
})

test('an unrecognized reply re-prompts the usage instead of answering or starting a turn', async () => {
  const { channel, port, runtime } = await enable(harness())

  const turn = channel.receive(inbound('run it'))
  await waitForText(channel, 'Approval needed')
  await channel.receive(inbound('maybe later'))

  expect(port.permissions).toEqual([])
  expect(runtime.submitted).toHaveLength(1)
  const usage = channel.sent.filter(message => message.text.startsWith('Reply yes to approve'))
  expect(usage).toHaveLength(1)

  await channel.receive(inbound('yes'))
  await turn
  expect(port.permissions).toEqual([{ requestId: 'perm-1', response: 'approve' }])
})

test('a stale approval reports instead of resolving or queueing a turn', async () => {
  const { channel, port, runtime } = await enable(harness())
  port.permissionOutcome = false

  const turn = channel.receive(inbound('run it'))
  await waitForText(channel, 'Approval needed')
  await channel.receive(inbound('yes'))

  expect(port.permissions).toEqual([])
  expect(runtime.submitted).toHaveLength(1)
  expect(channel.sent.some(message => message.text.includes('no longer pending'))).toBe(true)

  // Release the parked runtime so the test does not leak an open turn.
  port.permissionOutcome = true
  port.respondPermission('perm-1', 'approve')
  await turn
})

test('question requests reach the channel and numbered or freeform replies answer them', async () => {
  const numbered = await enable(harness())
  const numberedTurn = numbered.channel.receive(inbound('ask something'))
  await waitForText(numbered.channel, 'Question: Pick a lane')
  await numbered.channel.receive(inbound('2'))
  await numberedTurn
  expect(numbered.port.questions).toEqual([{ requestId: 'question-1', answers: { answer: 'beta' } }])
  expect(numbered.channel.sent.map(message => message.text)).toContain('Answer sent.')

  const freeform = await enable(harness())
  const freeformTurn = freeform.channel.receive(inbound('ask something'))
  await waitForText(freeform.channel, 'Question: Pick a lane')
  await freeform.channel.receive(inbound('the fast lane'))
  await freeformTurn
  expect(freeform.port.questions).toEqual([{ requestId: 'question-1', answers: { answer: 'the fast lane' } }])
})

test('without an interaction port the router ignores interaction events entirely', async () => {
  const channel = new RecordingChannel()
  const port = new FakeInteractionPort()
  const runtime = new InteractionRuntime(port)
  const manager = new ChannelManager({ channels: [['telegram', channel]] })
  const router = new ChannelTurnRouter({ channels: manager, runtime })
  manager.setInboundHandler(message => router.handle(message))
  await manager.enable('telegram')

  // The runtime parks on an answer that can never arrive through the router;
  // cancel by releasing it directly after observing no prompt was forwarded.
  const turn = channel.receive(inbound('run it'))
  await Bun.sleep(20)
  expect(channel.sent.some(message => message.text.includes('Approval needed'))).toBe(false)
  port.respondPermission('perm-1', 'approve')
  await turn
})

async function enable(setup: {
  channel: RecordingChannel
  port: FakeInteractionPort
  runtime: InteractionRuntime
}): Promise<typeof setup & { manager: ChannelManager }> {
  const manager = new ChannelManager({ channels: [['telegram', setup.channel]] })
  const router = new ChannelTurnRouter({
    channels: manager,
    interactions: setup.port,
    runtime: setup.runtime,
  })
  manager.setInboundHandler(message => router.handle(message))
  await manager.enable('telegram')
  return { ...setup, manager }
}
