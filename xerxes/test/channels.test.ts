// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type {
  Channel,
  ChannelMessage,
  InboundHandler,
  WebhookFailure,
  WebhookHeaders,
} from '../src/channels/index.js'
import {
  ChannelRegistry,
  createChannelMessage,
  gatherInbound,
  MessageDirection,
  UnknownChannelError,
  WebhookChannel,
  WebhookDispatcher,
  parseJsonBody,
} from '../src/channels/index.js'

class RecordingChannel implements Channel {
  readonly name: string
  readonly sent: ChannelMessage[] = []
  starts = 0
  stops = 0
  handler: InboundHandler | undefined
  private readonly startError: Error | undefined
  private readonly stopError: Error | undefined

  constructor(
    name: string,
    options: { readonly startError?: Error; readonly stopError?: Error } = {},
  ) {
    this.name = name
    this.startError = options.startError
    this.stopError = options.stopError
  }

  async start(onInbound: InboundHandler): Promise<void> {
    this.starts += 1
    if (this.startError) {
      throw this.startError
    }
    this.handler = onInbound
  }

  async stop(): Promise<void> {
    this.stops += 1
    if (this.stopError) {
      throw this.stopError
    }
    this.handler = undefined
  }

  async send(message: ChannelMessage): Promise<void> {
    this.sent.push(message)
  }
}

class TestWebhookChannel extends WebhookChannel {
  readonly name = 'test-webhook'
  failParse = false
  readonly outbound: ChannelMessage[] = []
  parsedMessages: readonly ChannelMessage[] = []

  protected parseInbound(
    _headers: WebhookHeaders,
    _body: Uint8Array,
  ): readonly ChannelMessage[] {
    if (this.failParse) {
      throw new Error('bad payload')
    }
    return this.parsedMessages
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    this.outbound.push(message)
  }
}

test('channel messages receive isolated Xerxes defaults and preserve explicit fields', () => {
  const timestamp = new Date('2026-07-13T12:00:00.000Z')
  const first = createChannelMessage({
    channel: 'telegram',
    text: 'hello',
    attachments: [{ file: 'one.png' }],
    metadata: { chatType: 'private' },
    timestamp,
  })
  const second = createChannelMessage({ channel: 'telegram', text: 'again' })

  expect(first.direction).toBe(MessageDirection.INBOUND)
  expect(first.messageId).toEqual(expect.any(String))
  expect(first.timestamp).toEqual(timestamp)
  expect(first.timestamp).not.toBe(timestamp)
  expect(first.attachments).toEqual([{ file: 'one.png' }])
  expect(first.metadata).toEqual({ chatType: 'private' })
  expect(first.attachments).not.toBe(second.attachments)
  expect(first.metadata).not.toBe(second.metadata)
  expect(second.direction).toBe(MessageDirection.INBOUND)
})

test('registry routes outbound messages and owns idempotent lifecycle state', async () => {
  const registry = new ChannelRegistry()
  const channel = new RecordingChannel('telegram')
  registry.register('telegram', channel)
  registry.setHandler(async () => {})

  await registry.startAll()
  await registry.startAll()
  expect(channel.starts).toBe(1)
  expect(registry.startedNames()).toEqual(['telegram'])

  const message = createChannelMessage({
    channel: 'telegram',
    direction: MessageDirection.OUTBOUND,
    text: 'reply',
  })
  await registry.send(message)
  expect(channel.sent).toEqual([message])

  const snapshot = registry.all()
  snapshot.delete('telegram')
  expect(registry.get('telegram')).toBe(channel)

  await registry.stopAll()
  expect(channel.stops).toBe(1)
  expect(registry.startedNames()).toEqual([])

  await expect(
    registry.send(createChannelMessage({ channel: 'missing', text: 'nope' })),
  ).rejects.toBeInstanceOf(UnknownChannelError)
})

test('registry requires a handler and isolates adapter lifecycle failures', async () => {
  const reported: string[] = []
  const failedStart = new Error('start failed')
  const failedStop = new Error('stop failed')
  const registry = new ChannelRegistry({
    onFailure: (failure) =>
      reported.push(`${failure.operation}:${failure.channel}`),
  })
  const broken = new RecordingChannel('broken', { startError: failedStart })
  const healthy = new RecordingChannel('healthy', { stopError: failedStop })
  registry.register('broken', broken)
  registry.register('healthy', healthy)

  await expect(registry.startAll()).rejects.toThrow(
    'ChannelRegistry.setHandler must be called before startAll()',
  )
  registry.setHandler(async () => {})
  await registry.startAll()
  expect(broken.starts).toBe(1)
  expect(healthy.starts).toBe(1)
  expect(registry.startedNames()).toEqual(['healthy'])

  await registry.stopAll()
  expect(healthy.stops).toBe(1)
  expect(registry.startedNames()).toEqual([])
  expect(registry.lifecycleFailures()).toEqual([
    { channel: 'broken', error: failedStart, operation: 'start' },
    { channel: 'healthy', error: failedStop, operation: 'stop' },
  ])
  expect(reported).toEqual(['start:broken', 'stop:healthy'])
})

test('gatherInbound starts independent registries concurrently', async () => {
  const first = new ChannelRegistry()
  const second = new ChannelRegistry()
  const firstChannel = new RecordingChannel('first')
  const secondChannel = new RecordingChannel('second')
  first.register('first', firstChannel)
  second.register('second', secondChannel)
  first.setHandler(async () => {})
  second.setHandler(async () => {})

  await gatherInbound(first, second)
  expect(firstChannel.starts).toBe(1)
  expect(secondChannel.starts).toBe(1)
})

test('webhook dispatcher preserves raw requests and contains handler errors', async () => {
  const failures: WebhookFailure[] = []
  const dispatcher = new WebhookDispatcher({
    onFailure: (failure) => failures.push(failure),
  })
  const headers = { 'x-signature': 'exact-value' }
  const body = new TextEncoder().encode('{"event":"message"}')
  dispatcher.register('service', async (receivedHeaders, receivedBody) => {
    expect(receivedHeaders).toBe(headers)
    expect(receivedBody).toBe(body)
    return { status: 202, body: 'accepted', headers: { 'x-result': 'ok' } }
  })

  expect(await dispatcher.dispatch('missing', headers, body)).toEqual({
    status: 404,
    body: "unknown channel 'missing'",
  })
  expect(await dispatcher.dispatch('service', headers, body)).toEqual({
    status: 202,
    body: 'accepted',
    headers: { 'x-result': 'ok' },
  })

  dispatcher.register('broken', async () => {
    throw new Error('handler failed')
  })
  expect(await dispatcher.dispatch('broken', headers, body)).toEqual({
    status: 500,
    body: '',
  })
  expect(failures).toHaveLength(1)
  expect(failures[0]).toMatchObject({
    channel: 'broken',
    source: 'dispatcher',
  })
})

test('webhook channel normalizes transport lifecycle and failure responses', async () => {
  const failures: WebhookFailure[] = []
  const channel = new TestWebhookChannel({
    onFailure: (failure) => failures.push(failure),
  })
  const headers = { 'x-provider': 'test' }
  const body = new Uint8Array()
  const first = createChannelMessage({ channel: channel.name, text: 'first' })
  const broken = createChannelMessage({
    channel: channel.name,
    text: 'broken',
  })

  expect(await channel.handleWebhook(headers, body)).toEqual({
    status: 503,
    body: 'channel not started',
  })
  channel.parsedMessages = [first, broken]
  const handled: string[] = []
  await channel.start(async (message) => {
    handled.push(message.text)
    if (message.text === 'broken') {
      throw new Error('runtime failed')
    }
  })
  expect(await channel.handleWebhook(headers, body)).toEqual({
    status: 500,
    body: 'ok',
  })
  expect(handled).toEqual(['first', 'broken'])

  channel.failParse = true
  expect(await channel.handleWebhook(headers, body)).toEqual({
    status: 400,
    body: 'invalid payload',
  })
  expect(failures.map((failure) => failure.source)).toEqual([
    'inbound_handler',
    'parse',
  ])

  const outbound = createChannelMessage({
    channel: channel.name,
    direction: MessageDirection.OUTBOUND,
    text: 'reply',
  })
  await channel.send(outbound)
  expect(channel.outbound).toEqual([outbound])

  await channel.stop()
  expect(await channel.handleWebhook(headers, body)).toEqual({
    status: 503,
    body: 'channel not started',
  })
})

test('parseJsonBody accepts only object payloads', () => {
  expect(parseJsonBody(new TextEncoder().encode('{"ok":true}'))).toEqual({
    ok: true,
  })
  expect(
    parseJsonBody(new TextEncoder().encode('["not", "an", "object"]')),
  ).toEqual({})
  expect(parseJsonBody(new TextEncoder().encode('not json'))).toEqual({})
  expect(parseJsonBody(new Uint8Array())).toEqual({})
})
