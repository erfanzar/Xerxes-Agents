// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  MessageDirection,
  MarkdownAgentWorkspace,
  createChannelMessage,
  type Channel,
  type ChannelMessage,
  type InboundHandler,
} from '../src/channels/index.js'
import {
  createDaemonChannelManager,
  daemonChannelWebhookOptions,
} from '../src/daemon/channels.js'
import type { DaemonConfig } from '../src/daemon/config.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'

class RecordingAdapter implements Channel {
  readonly name = 'fixed-adapter'
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
    if (!this.inbound) throw new Error('adapter is not running')
    await this.inbound(createChannelMessage({
      channel: this.name,
      channelUserId: 'user',
      direction: MessageDirection.INBOUND,
      text,
    }))
  }
}

class PreviewAdapter extends RecordingAdapter {
  readonly previews: Array<{ readonly kind: 'edit' | 'send'; readonly text: string }> = []

  async sendText(_chatId: string, text: string): Promise<Readonly<Record<string, unknown>>> {
    this.previews.push({ kind: 'send', text })
    return { result: { message_id: 'preview-1' } }
  }

  async editText(_chatId: string, _messageId: string, text: string): Promise<Readonly<Record<string, unknown>>> {
    this.previews.push({ kind: 'edit', text })
    return {}
  }
}

test('daemon channel host binds configured adapters to native runtime turns and preserves adapter-facing names', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-daemon-channels-'))
  const adapter = new RecordingAdapter()
  const config = testConfig(directory, {
    support: { type: 'test-adapter', enabled: true, settings: {} },
  })
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, 'sessions'),
  })
  const workspace = new MarkdownAgentWorkspace(join(directory, 'channel-workspace'))
  const manager = createDaemonChannelManager(config, runtime, {
    environment: {},
    factory: () => adapter,
    workspace,
  })

  try {
    await manager.startConfigured()
    await adapter.receive('hello from a channel')

    expect(manager.status('support')).toMatchObject({
      adapterName: 'test-adapter',
      enabled: true,
      name: 'support',
    })
    expect(adapter.sent).toHaveLength(1)
    expect(adapter.sent[0]).toMatchObject({
      channel: 'fixed-adapter',
      direction: MessageDirection.OUTBOUND,
      text: expect.stringContaining('hello from a channel'),
    })
    expect((await workspace.loadContext()).prompt).toContain('hello from a channel')
  } finally {
    await manager.stopAll()
    await rm(directory, { recursive: true, force: true })
  }
})

test('daemon channel webhook listener settings inherit control host and validate port fallback', () => {
  const config = testConfig('/workspace', {})
  const options = daemonChannelWebhookOptions({
    ...config,
    control: { websocket_host: '0.0.0.0', webhook_port: 'not-a-port' },
  })

  expect(options).toEqual({ host: '0.0.0.0', port: 11997 })
})

test('configured daemon channel settings can disable streamed editable previews', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-channel-preview-config-'))
  const adapter = new PreviewAdapter()
  const config = testConfig(directory, {
    support: { type: 'test-adapter', enabled: true, settings: { stream_previews: false } },
  })
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, 'sessions'),
  })
  const manager = createDaemonChannelManager(config, runtime, {
    environment: {},
    factory: () => adapter,
    workspace: new MarkdownAgentWorkspace(join(directory, 'channel-workspace')),
  })
  try {
    await manager.startConfigured()
    await adapter.receive('send only a final reply')

    expect(adapter.previews).toEqual([])
    expect(adapter.sent).toHaveLength(1)
  } finally {
    await manager.stopAll()
    await rm(directory, { recursive: true, force: true })
  }
})

function testConfig(directory: string, channels: DaemonConfig['channels']): DaemonConfig {
  return {
    channels,
    control: { websocket_host: '127.0.0.1', websocket_port: 0 },
    maxConcurrentTurns: 8,
    projectDirectory: directory,
    runtime: {},
    workspace: {},
  }
}
