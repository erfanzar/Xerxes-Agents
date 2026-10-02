// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  ChannelSessionIndexError,
  JsonChannelSessionIndex,
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
import { DaemonInteractionBoard } from '../src/daemon/interactions.js'
import { InMemoryDaemonRuntime, type TurnRunner } from '../src/daemon/runtime.js'
import type { PermissionRequest } from '../src/streaming/events.js'

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

test('daemon channel webhook listener uses an independent loopback host and validates port fallback', () => {
  const config = testConfig('/workspace', {})
  const options = daemonChannelWebhookOptions({
    ...config,
    control: { websocket_host: '0.0.0.0', webhook_port: 'not-a-port' },
  })

  expect(options).toEqual({ host: '127.0.0.1', port: 11997 })
})

test('daemon channel webhook options reject an unauthenticated public bind even with an irrelevant channel secret', () => {
  const config = testConfig('/workspace', {
    relay: { type: 'generic_webhook', enabled: true, settings: { signing_secret: 'not-used-by-generic-webhook' } },
  })

  expect(() => daemonChannelWebhookOptions({
    ...config,
    control: { webhook_host: '0.0.0.0' },
  })).toThrow('non-loopback webhook listeners require control.auth_token')
})

test('daemon channel webhook options allow a public bind with shared bearer authentication', () => {
  const config = testConfig('/workspace', {})
  const options = daemonChannelWebhookOptions({
    ...config,
    control: { webhook_host: '0.0.0.0', auth_token: ' edge-secret ' },
  })

  expect(options).toEqual({ authToken: 'edge-secret', host: '0.0.0.0', port: 11997 })
})

test('daemon channel webhook listener receives the configured control auth token', () => {
  const config = testConfig('/workspace', {})
  const options = daemonChannelWebhookOptions({
    ...config,
    control: { websocket_host: '127.0.0.1', websocket_port: 0, auth_token: '  edge-secret  ' },
  })

  expect(options).toEqual({ authToken: 'edge-secret', host: '127.0.0.1', port: 11997 })
})

test('daemon channel webhook options keep loopback unauthenticated for compatibility', () => {
  const options = daemonChannelWebhookOptions(testConfig('/workspace', {
    relay: { type: 'generic_webhook', enabled: true, settings: {} },
  }))

  expect(options).toEqual({ host: '127.0.0.1', port: 11997 })
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

test('daemon channel host answers a turn\'s approval from the chat through the daemon interaction board', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-channel-approval-'))
  const adapter = new RecordingAdapter()
  const config = testConfig(directory, {
    support: { type: 'test-adapter', enabled: true, settings: {} },
  })
  const interactions = new DaemonInteractionBoard()
  const runner: TurnRunner = {
    async *run(session, _text, signal) {
      const request: PermissionRequest = {
        requestId: 'channel-approval',
        description: 'Send the build report.',
        inputs: {},
        toolCall: { id: 'send-1', type: 'function', function: { name: 'send_message', arguments: {} } },
      }
      yield { type: 'approval_request', payload: { id: request.requestId, request_id: request.requestId, description: request.description } }
      yield { type: 'text_part', payload: { text: 'approval:' + await interactions.permissionBroker(session.id).request(request, signal) } }
    },
  }
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    interactions,
    sessionDirectory: join(directory, 'sessions'),
  })
  const manager = createDaemonChannelManager(config, runtime, {
    environment: {},
    factory: () => adapter,
    interactions,
    workspace: new MarkdownAgentWorkspace(join(directory, 'channel-workspace')),
  })
  try {
    await manager.startConfigured()
    const turn = adapter.receive('send the build report')
    for (let tries = 0; tries < 200 && !adapter.sent.some(sent => sent.text.startsWith('Approval needed')); tries++) await Bun.sleep(5)
    expect(adapter.sent.at(-1)?.text).toContain('Send the build report.')
    await adapter.receive('/approve')
    await turn
    expect(adapter.sent.at(-1)?.text).toBe('approval:approve')
  } finally {
    await manager.stopAll()
    await rm(directory, { recursive: true, force: true })
  }
})

test('a channel conversation continues its saved session after a daemon restart', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-channel-restart-'))
  const config = testConfig(directory, {
    support: { type: 'test-adapter', enabled: true, settings: { stream_previews: false } },
  })
  const boot = () => {
    const adapter = new RecordingAdapter()
    const runtime = new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, 'sessions'),
    })
    const manager = createDaemonChannelManager(config, runtime, {
      environment: {},
      factory: () => adapter,
      workspace: new MarkdownAgentWorkspace(join(directory, 'channel-workspace')),
    })
    return { adapter, manager, runtime }
  }
  const sessionKey = 'support:private:user'
  try {
    const before = boot()
    await before.manager.startConfigured()
    await before.adapter.receive('remember the code word: heron')
    const original = before.runtime.sessionStatus(sessionKey)
    expect(original).toBeDefined()
    await before.runtime.flushSessions()
    await before.manager.stopAll()

    const after = boot()
    await after.manager.startConfigured()
    await after.adapter.receive('what was the code word?')
    const resumed = after.runtime.sessionStatus(sessionKey)

    expect(resumed?.id).toBe(original!.id)
    expect(JSON.stringify(resumed?.messages)).toContain('remember the code word: heron')
    // /new is remembered too: the next restart must not revive the old chat.
    await after.adapter.receive('/new')
    await after.runtime.flushSessions()
    await after.manager.stopAll()

    const reset = boot()
    await reset.manager.startConfigured()
    await reset.adapter.receive('starting over')
    const fresh = reset.runtime.sessionStatus(sessionKey)
    expect(fresh?.id).not.toBe(original!.id)
    expect(JSON.stringify(fresh?.messages)).not.toContain('heron')
    await reset.manager.stopAll()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('the channel session index rejects a corrupt file with an actionable error', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-channel-index-'))
  const path = join(directory, 'channel-sessions.json')
  try {
    await Bun.write(path, '{ not json')
    const index = new JsonChannelSessionIndex(path)
    await expect(index.get('telegram:private:7')).rejects.toBeInstanceOf(ChannelSessionIndexError)

    await Bun.write(path, JSON.stringify({ sessions: { 'telegram:private:7': 'abc123def456' } }))
    expect(await index.get('telegram:private:7')).toBe('abc123def456')
    await index.set('telegram:private:8', 'fedcba987654')
    expect(await new JsonChannelSessionIndex(path).get('telegram:private:8')).toBe('fedcba987654')
  } finally {
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
