// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { Channel, InboundHandler } from '../src/channels/base.js'
import { ChannelManager } from '../src/channels/manager.js'
import { TelegramChannel } from '../src/channels/telegram.js'
import { ChannelWebhookServer } from '../src/channels/webhookServer.js'
import type { ChannelMessage } from '../src/channels/types.js'
import type { WebhookHeaders, WebhookResponse } from '../src/channels/webhooks.js'
import { DaemonServer } from '../src/daemon/server.js'

class TestWebhookChannel implements Channel {
  readonly name = 'test'
  received: { body: Uint8Array; headers: WebhookHeaders } | undefined
  private inbound: InboundHandler | undefined

  async handleWebhook(headers: WebhookHeaders, body: Uint8Array): Promise<WebhookResponse> {
    this.received = { headers, body }
    await this.inbound?.({
      attachments: [],
      channel: this.name,
      direction: 'inbound',
      messageId: 'incoming',
      metadata: {},
      text: new TextDecoder().decode(body),
      timestamp: new Date(),
    })
    return { status: 202, body: 'accepted', headers: { 'X-Webhook': 'yes' } }
  }

  async send(_message: ChannelMessage): Promise<void> {}

  async start(onInbound: InboundHandler): Promise<void> {
    this.inbound = onInbound
  }

  async stop(): Promise<void> {
    this.inbound = undefined
  }
}

test('webhook server exposes configured channel status and forwards raw webhook requests', async () => {
  const channel = new TestWebhookChannel()
  const inbound: string[] = []
  const manager = new ChannelManager({
    channels: [['incoming', channel]],
    onInbound: async message => {
      inbound.push(message.text)
    },
  })
  await manager.enable('incoming')
  const server = new ChannelWebhookServer({ manager, port: 0 })
  server.start()
  const base = server.url
  if (!base) throw new Error('webhook server did not start')
  try {
    const listing = await fetch(new URL('/channels', base))
    expect(listing.status).toBe(200)
    expect(await listing.json()).toMatchObject({
      ok: true,
      channels: [expect.objectContaining({ name: 'incoming', enabled: true })],
    })

    const delivered = await fetch(new URL('/channels/incoming/webhook', base), {
      method: 'POST',
      headers: { 'X-Provider-Signature': 'raw' },
      body: 'hello channel',
    })
    expect(delivered.status).toBe(202)
    expect(delivered.headers.get('x-webhook')).toBe('yes')
    expect(await delivered.text()).toBe('accepted')
    expect(new TextDecoder().decode(channel.received?.body)).toBe('hello channel')
    expect(channel.received?.headers['x-provider-signature']).toBe('raw')
    expect(inbound).toEqual(['hello channel'])

    expect((await fetch(new URL('/channels/missing/webhook', base), { method: 'POST' })).status).toBe(404)
    expect((await fetch(new URL('/channels/incoming/webhook', base))).status).toBe(405)
  } finally {
    await server.stop()
    await manager.stopAll()
  }
})

test('webhook server enforces its body limit before a provider adapter runs', async () => {
  const channel = new TestWebhookChannel()
  const manager = new ChannelManager({
    channels: [['incoming', channel]],
    onInbound: async () => undefined,
  })
  await manager.enable('incoming')
  const server = new ChannelWebhookServer({ manager, maxBodyBytes: 4, port: 0 })
  server.start()
  const base = server.url
  if (!base) throw new Error('webhook server did not start')
  try {
    const response = await fetch(new URL('/channels/incoming/webhook', base), {
      method: 'POST',
      body: 'oversized',
    })
    expect(response.status).toBe(413)
    expect(channel.received).toBeUndefined()
  } finally {
    await server.stop()
    await manager.stopAll()
  }
})

test('webhook server honors a Telegram adapter payload cap below its global limit', async () => {
  const received: ChannelMessage[] = []
  const channel = new TelegramChannel({ token: 'test-token', maxPayloadBytes: 4 })
  const manager = new ChannelManager({
    channels: [['telegram', channel]],
    onInbound: async message => { received.push(message) },
  })
  await manager.enable('telegram')
  const server = new ChannelWebhookServer({ manager, maxBodyBytes: 1_024, port: 0 })
  server.start()
  const base = server.url
  if (!base) throw new Error('webhook server did not start')
  try {
    const response = await fetch(new URL('/channels/telegram/webhook', base), {
      method: 'POST',
      body: 'oversized',
    })
    expect(response.status).toBe(413)
    expect(received).toEqual([])
  } finally {
    await server.stop()
    await manager.stopAll()
  }
})

test('daemon lifecycle owns an optional configured channel webhook listener', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-channel-daemon-'))
  const channel = new TestWebhookChannel()
  const manager = new ChannelManager({
    channels: [['incoming', channel]],
    onInbound: async () => undefined,
  })
  await manager.enable('incoming')
  const daemon = new DaemonServer({
    socketPath: join(directory, 'daemon.sock'),
    channelManager: manager,
    channelWebhook: { port: 0 },
  })
  try {
    await daemon.start()
    const base = daemon.channelWebhookUrl
    if (!base) throw new Error('daemon webhook listener did not start')
    const response = await fetch(new URL('/channels/incoming/webhook', base), {
      method: 'POST',
      body: 'through daemon',
    })
    expect(response.status).toBe(202)
    expect(new TextDecoder().decode(channel.received?.body)).toBe('through daemon')
  } finally {
    await daemon.stop()
    await rm(directory, { force: true, recursive: true })
  }
})
