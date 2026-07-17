// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createServer } from 'node:net'

import {
  ChannelConfigurationError,
  ConfiguredChannelManager,
  MessageDirection,
  createConfiguredChannel,
  createChannelMessage,
  type Channel,
  type ChannelMessage,
  type InboundHandler,
  type TelegramUpdatesOptions,
  type WebhookCapableChannel,
  type WebhookHeaders,
  type WebhookResponse,
} from '../src/channels/index.js'

class FixedNameChannel implements Channel {
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

  async receive(text: string): Promise<void> {
    if (!this.inbound) throw new Error('channel is not running')
    await this.inbound(createChannelMessage({
      channel: this.name,
      direction: MessageDirection.INBOUND,
      text,
    }))
  }
}

class PollingFixedNameChannel extends FixedNameChannel {
  calls = 0
  deletes = 0

  async deleteWebhook(): Promise<void> {
    this.deletes += 1
  }

  async getUpdates(options: TelegramUpdatesOptions = {}): Promise<Readonly<Record<string, unknown>>> {
    this.calls += 1
    if (this.calls === 1) return { result: [{ update_id: 7 }] }
    return new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true })
    })
  }

  async handleWebhook(_headers: WebhookHeaders, _body: Uint8Array): Promise<WebhookResponse> {
    await this.receive('from long poll')
    return { status: 200, body: 'ok' }
  }
}

test('configured channel manager retains disabled declarations and constructs resolved adapter settings', async () => {
  const manager = new ConfiguredChannelManager({
    channels: {
      telegram: {
        type: 'telegram',
        enabled: false,
        settings: {
          token: 'resolved-token',
          accept_edited_messages: true,
          transport: 'webhook',
        },
      },
    },
    onInbound: async () => {},
  })

  expect(manager.hasConfiguredChannels).toBe(true)
  expect(manager.list()).toEqual([{ name: 'telegram', adapterName: 'telegram', enabled: false }])
  expect(manager.registry.get('telegram')?.name).toBe('telegram')
  await manager.startConfigured()
  expect(manager.status('telegram')?.enabled).toBe(false)
  await manager.enable('telegram')
  expect(manager.status('telegram')?.enabled).toBe(true)
  await manager.stopAll()
})

test('configured manager starts enabled webhook adapters and routes their inbound delivery', async () => {
  const received: string[] = []
  const manager = new ConfiguredChannelManager({
    channels: {
      intake: {
        type: 'generic_webhook',
        enabled: true,
        settings: { outbound_url: 'https://example.invalid/outbound' },
      },
    },
    onInbound: async message => { received.push(message.text) },
  })

  await manager.startConfigured()
  const channel = manager.registry.get('intake')
  expect(manager.hasWebhookChannels()).toBe(true)
  expect(manager.status('intake')).toEqual({
    name: 'intake',
    adapterName: 'generic_webhook',
    enabled: true,
  })
  if (!isWebhookChannel(channel)) throw new Error('generic webhook adapter was not constructed')
  const result = await channel.handleWebhook({}, new TextEncoder().encode(JSON.stringify({ text: 'real message' })))
  expect(result).toEqual({ status: 200, body: 'ok' })
  expect(received).toEqual(['real message'])
  await manager.stopAll()
})

test('configured manager maps an adapter fixed name onto its configured daemon name in both directions', async () => {
  const adapter = new FixedNameChannel()
  const received: ChannelMessage[] = []
  const manager = new ConfiguredChannelManager({
    channels: {
      product_telegram: {
        type: 'telegram',
        enabled: true,
        settings: { transport: 'webhook' },
      },
    },
    factory: () => adapter,
    onInbound: async message => { received.push(message) },
  })

  await manager.startConfigured()
  await adapter.receive('inbound')
  await manager.registry.send(createChannelMessage({
    channel: 'product_telegram',
    direction: MessageDirection.OUTBOUND,
    text: 'outbound',
  }))

  expect(received[0]?.channel).toBe('product_telegram')
  expect(adapter.sent[0]?.channel).toBe('telegram')
  expect(manager.status('product_telegram')).toMatchObject({
    adapterName: 'telegram',
    enabled: true,
    name: 'product_telegram',
  })
  await manager.stopAll()
})

test('configured manager starts and stops Telegram long polling for configured polling transport', async () => {
  const adapter = new PollingFixedNameChannel()
  const received: ChannelMessage[] = []
  const manager = new ConfiguredChannelManager({
    channels: {
      bot: {
        type: 'telegram',
        enabled: true,
        settings: { transport: 'polling', polling_timeout: 0, polling_retry_delay: 0 },
      },
    },
    factory: () => adapter,
    onInbound: async message => { received.push(message) },
  })

  await manager.startConfigured()
  await eventually(() => received.length === 1)
  expect(received[0]).toMatchObject({ channel: 'bot', text: 'from long poll' })
  expect(adapter.deletes).toBe(1)
  await manager.stopAll()
})

test('configured manager accepts a decimal string Telegram polling timeout', async () => {
  const adapter = new TimeoutRecordingPollingChannel()
  const manager = new ConfiguredChannelManager({
    channels: {
      bot: {
        type: 'telegram',
        enabled: true,
        settings: { transport: 'polling', polling_timeout: '45', polling_retry_delay: 0 },
      },
    },
    factory: () => adapter,
    onInbound: async () => {},
  })

  await manager.startConfigured()
  await eventually(() => adapter.updatesOptions.length >= 1)
  expect(adapter.updatesOptions[0]?.timeout).toBe(45)
  await manager.stopAll()
})

test('configured factory accepts a decimal string SMTP port for the email adapter', async () => {
  const rcpt: string[] = []
  await withSmtpFixture(async port => {
    const channel = createConfiguredChannel('email', 'mail', {
      from_address: 'xerxes@example.test',
      smtp_host: '127.0.0.1',
      smtp_port: String(port),
    })
    await channel.send(createChannelMessage({
      channel: 'mail',
      direction: MessageDirection.OUTBOUND,
      channelUserId: 'alice@example.test',
      text: 'string port delivery',
    }))
  }, rcpt)

  expect(rcpt).toContain('RCPT TO:<alice@example.test>')
})

class TimeoutRecordingPollingChannel extends FixedNameChannel {
  readonly updatesOptions: TelegramUpdatesOptions[] = []

  async deleteWebhook(): Promise<void> {}

  async getUpdates(options: TelegramUpdatesOptions = {}): Promise<Readonly<Record<string, unknown>>> {
    this.updatesOptions.push(options)
    return new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true })
    })
  }

  async handleWebhook(_headers: WebhookHeaders, _body: Uint8Array): Promise<WebhookResponse> {
    return { status: 200, body: 'ok' }
  }
}

async function withSmtpFixture(run: (port: number) => Promise<void>, commands: string[]): Promise<void> {
  const server = createServer(socket => {
    let buffer = ''
    let dataMode = false
    socket.write('220 fixture SMTP\r\n')
    socket.on('data', chunk => {
      buffer += chunk.toString()
      while (true) {
        const index = buffer.indexOf('\n')
        if (index < 0) return
        const line = buffer.slice(0, index).replace(/\r$/, '')
        buffer = buffer.slice(index + 1)
        if (dataMode) {
          if (line === '.') {
            dataMode = false
            socket.write('250 accepted\r\n')
          }
          continue
        }
        commands.push(line)
        if (line === 'DATA') {
          dataMode = true
          socket.write('354 continue\r\n')
        } else if (line === 'QUIT') {
          socket.write('221 closing\r\n')
          socket.end()
        } else {
          socket.write('250 accepted\r\n')
        }
      }
    })
  })
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => resolve())
  })
  const address = server.address()
  if (!address || typeof address === 'string') {
    server.close()
    throw new Error('SMTP fixture did not bind a TCP port')
  }
  try {
    await run(address.port)
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()))
  }
}

test('configured manager contains invalid declarations without silently inventing a transport', async () => {
  const manager = new ConfiguredChannelManager({
    channels: {
      discord: {
        type: 'discord',
        enabled: true,
        settings: {},
      },
      legacy_mail: {
        type: 'email',
        enabled: false,
        settings: {},
      },
    },
    onInbound: async () => {},
  })

  await expect(manager.startConfigured()).resolves.toHaveLength(2)
  expect(manager.list()).toEqual([
    {
      name: 'discord',
      adapterName: 'discord',
      enabled: false,
      lastError: 'missing required setting bot_token for discord',
    },
    {
      name: 'legacy_mail',
      adapterName: 'email',
      enabled: false,
    },
  ])
  await expect(manager.enable('discord')).rejects.toBeInstanceOf(ChannelConfigurationError)
})

test('configured factory accepts Python daemon snake-case fields for concrete adapters', async () => {
  const matrix = createConfiguredChannel('matrix', 'matrix', {
    access_token: 'matrix-secret',
    homeserver_url: 'https://matrix.example',
  })
  const twilio = createConfiguredChannel('sms', 'sms', {
    account_sid: 'AC123',
    auth_token: 'auth-secret',
    from_number: '+15555550100',
  })

  expect(matrix.name).toBe('matrix')
  expect(twilio.name).toBe('sms')

  const telegram = createConfiguredChannel('telegram', 'telegram', {
    token: '123:testing-token',
    allowed_user_ids: '77',
    bot_username: 'xerxes_bot',
    require_allowed_sender: true,
  })
  if (!isWebhookChannel(telegram)) throw new Error('Telegram adapter was not constructed')
  const received: ChannelMessage[] = []
  await telegram.start(async message => { received.push(message) })
  await telegram.handleWebhook({}, new TextEncoder().encode(JSON.stringify({
    message: {
      message_id: 3,
      text: '@xerxes_bot hi',
      from: { id: 77 },
      chat: { id: -100, type: 'supergroup' },
    },
  })))
  expect(received).toHaveLength(1)
})

function isWebhookChannel(channel: unknown): channel is WebhookCapableChannel {
  return typeof channel === 'object'
    && channel !== null
    && 'handleWebhook' in channel
    && typeof channel.handleWebhook === 'function'
}

async function eventually(predicate: () => boolean): Promise<void> {
  for (let index = 0; index < 100; index += 1) {
    if (predicate()) return
    await Bun.sleep(1)
  }
  throw new Error('condition was not reached')
}
