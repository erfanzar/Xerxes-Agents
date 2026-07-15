// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ChannelInboundHandlerUnavailableError,
  ChannelManager,
  ChannelNotConfiguredError,
  MessageDirection,
  createChannelMessage,
  type Channel,
  type ChannelMessage,
  type InboundHandler,
} from '../src/channels/index.js'

class RecordingChannel implements Channel {
  readonly name: string
  starts = 0
  stops = 0
  private handler: InboundHandler | undefined
  private readonly startError: Error | undefined

  constructor(name: string, options: { readonly startError?: Error } = {}) {
    this.name = name
    this.startError = options.startError
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
    this.handler = undefined
  }

  async send(_message: ChannelMessage): Promise<void> {}

  async receive(text: string): Promise<void> {
    if (!this.handler) {
      throw new Error('channel is not enabled')
    }
    await this.handler(createChannelMessage({
      channel: this.name,
      direction: MessageDirection.INBOUND,
      text,
    }))
  }
}

test('channel manager only enables configured adapters with a host inbound route', async () => {
  const channel = new RecordingChannel('telegram')
  const received: string[] = []
  const manager = new ChannelManager({
    channels: [['telegram', channel]],
    onInbound: async message => { received.push(message.text) },
  })

  expect(manager.list()).toEqual([{ name: 'telegram', adapterName: 'telegram', enabled: false }])
  expect(await manager.enable('telegram')).toEqual({ name: 'telegram', adapterName: 'telegram', enabled: true })
  await channel.receive('real inbound')
  expect(received).toEqual(['real inbound'])
  expect(channel.starts).toBe(1)

  expect(await manager.disable('telegram')).toEqual({ name: 'telegram', adapterName: 'telegram', enabled: false })
  expect(channel.stops).toBe(1)
  await expect(manager.enable('missing')).rejects.toBeInstanceOf(ChannelNotConfiguredError)
})

test('channel manager reports missing host routing and concrete lifecycle failures', async () => {
  const noHandler = new ChannelManager({ channels: [['telegram', new RecordingChannel('telegram')]] })
  await expect(noHandler.enable('telegram')).rejects.toBeInstanceOf(ChannelInboundHandlerUnavailableError)

  const broken = new RecordingChannel('broken', { startError: new Error('adapter refused startup') })
  const manager = new ChannelManager({
    channels: [['broken', broken]],
    onInbound: async () => {},
  })
  await expect(manager.enable('broken')).rejects.toThrow('adapter refused startup')
  expect(manager.status('broken')).toEqual({
    name: 'broken',
    adapterName: 'broken',
    enabled: false,
    lastOperation: 'start',
    lastError: 'adapter refused startup',
  })
})
