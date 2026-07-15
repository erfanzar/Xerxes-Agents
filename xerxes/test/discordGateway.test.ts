// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  BunDiscordGatewayWebSocketPort,
  ConfiguredChannelManager,
  DiscordChannel,
  FetchDiscordApplicationRestPort,
  FetchDiscordGatewayRestPort,
  type DiscordApplicationCommandRequest,
  type DiscordApplicationRestPort,
  type DiscordGuildApplicationCommandRequest,
  type DiscordInteractionAcknowledgement,
  type DiscordGatewayClock,
  type DiscordGatewayCloseEvent,
  type DiscordGatewayConnectionHandlers,
  type DiscordGatewayRestPort,
  type DiscordGatewaySocket,
  type DiscordGatewayWebSocketPort,
} from '../src/channels/index.js'
import type { ChannelMessage } from '../src/channels/types.js'

test('Discord Gateway identifies, heartbeats, routes messages, and resumes without external credentials', async () => {
  const clock = new FakeClock()
  const rest = new FakeRestPort('wss://gateway.fixture.test')
  const webSocket = new FakeWebSocketPort()
  const received: ChannelMessage[] = []
  const channel = new DiscordChannel({
    botToken: 'fixture-bot-token',
    gatewayPorts: { clock, random: () => 0, rest, webSocket },
    registerCommands: false,
    requireMention: true,
    transport: 'gateway',
  })

  await channel.start(async message => { received.push(message) })
  expect(rest.tokens).toEqual(['fixture-bot-token'])
  const first = onlySocket(webSocket)
  expect(new URL(first.url)).toMatchObject({ hostname: 'gateway.fixture.test', protocol: 'wss:' })

  first.emit({ op: 10, d: { heartbeat_interval: 25 } })
  await eventually(() => sentFrame(first, 1) !== undefined && sentFrame(first, 2) !== undefined)
  expect(sentFrame(first, 2)?.d).toMatchObject({
    intents: 37_377,
    properties: { $browser: 'xerxes', $device: 'xerxes', $os: 'bun' },
    token: 'fixture-bot-token',
  })
  expect(sentFrame(first, 1)?.d).toBeNull()

  first.emit({
    d: {
      resume_gateway_url: 'wss://resume.fixture.test/gateway',
      session_id: 'session-1',
      user: { id: 'bot-1' },
    },
    op: 0,
    s: 7,
    t: 'READY',
  })
  first.emit({
    d: {
      author: { id: 'user-1', username: 'fixture-user' },
      channel_id: 'channel-1',
      content: '<@bot-1> status',
      guild_id: 'guild-1',
      id: 'message-1',
      mentions: [{ id: 'bot-1' }],
    },
    op: 0,
    s: 8,
    t: 'MESSAGE_CREATE',
  })
  await eventually(() => received.length === 1)
  expect(received[0]).toMatchObject({
    channel: 'discord',
    channelUserId: 'user-1',
    platformMessageId: 'message-1',
    roomId: 'channel-1',
    text: 'status',
  })

  first.emit({ op: 11, d: null })
  await clock.runNext(25)
  await eventually(() => sentFrames(first, 1).length === 2)
  expect(sentFrames(first, 1)[1]?.d).toBe(8)

  first.emit({ op: 7, d: null })
  await clock.runNext(1_000)
  await eventually(() => webSocket.connections.length === 2)
  const second = webSocket.connections[1]
  if (!second) throw new Error('Discord Gateway reconnect did not create a second socket')
  expect(new URL(second.url)).toMatchObject({ hostname: 'resume.fixture.test', pathname: '/gateway' })
  second.emit({ op: 10, d: { heartbeat_interval: 25 } })
  await eventually(() => sentFrame(second, 6) !== undefined)
  expect(sentFrame(second, 6)?.d).toEqual({ seq: 8, session_id: 'session-1', token: 'fixture-bot-token' })

  await channel.stop()
  expect(second.closeEvents).toContainEqual({ code: 1000, reason: 'Xerxes Discord gateway stopped' })
})

test('Discord Gateway caches guild and thread metadata for name-routed raw messages', async () => {
  const webSocket = new FakeWebSocketPort()
  const received: ChannelMessage[] = []
  const channel = new DiscordChannel({
    allowedChannelNames: ['ops'],
    botToken: 'fixture-bot-token',
    gatewayPorts: { rest: new FakeRestPort('wss://gateway.fixture.test'), webSocket },
    registerCommands: false,
    transport: 'gateway',
  })

  await channel.start(async message => { received.push(message) })
  const socket = onlySocket(webSocket)
  socket.emit({
    d: {
      channels: [{ id: 'channel-1', name: 'ops' }],
      id: 'guild-1',
      name: 'Operations',
      threads: [{ id: 'thread-1', name: 'deployment-incident', parent_id: 'channel-1' }],
    },
    op: 0,
    s: 1,
    t: 'GUILD_CREATE',
  })
  socket.emit({
    d: {
      author: { global_name: 'Fixture User', id: 'user-1', username: 'fixture-user' },
      channel_id: 'thread-1',
      content: 'inspect the deploy',
      guild_id: 'guild-1',
      id: 'message-1',
      member: { nick: 'On-call' },
    },
    op: 0,
    s: 2,
    t: 'MESSAGE_CREATE',
  })

  await eventually(() => received.length === 1)
  expect(received[0]).toMatchObject({
    channelUserId: 'user-1',
    roomId: 'thread-1',
    text: 'inspect the deploy',
    metadata: {
      author_display_name: 'On-call',
      channel_name: 'deployment-incident',
      guild_id: 'guild-1',
      guild_name: 'Operations',
      parent_channel_id: 'channel-1',
      thread_id: 'thread-1',
    },
  })
  await channel.stop()
})

test('Discord Gateway resets an invalid non-resumable session before identifying again', async () => {
  const clock = new FakeClock()
  const webSocket = new FakeWebSocketPort()
  const channel = new DiscordChannel({
    botToken: 'fixture-bot-token',
    gatewayPorts: { clock, random: () => 0, rest: new FakeRestPort('wss://gateway.fixture.test'), webSocket },
    registerCommands: false,
    transport: 'gateway',
  })

  await channel.start(async () => {})
  const first = onlySocket(webSocket)
  first.emit({ op: 10, d: { heartbeat_interval: 20 } })
  await eventually(() => sentFrame(first, 2) !== undefined)
  first.emit({
    d: { resume_gateway_url: 'wss://resume.fixture.test', session_id: 'session-1' },
    op: 0,
    s: 4,
    t: 'READY',
  })

  first.emit({ op: 9, d: false })
  await clock.runNext(1_000)
  await eventually(() => webSocket.connections.length === 2)
  const second = webSocket.connections[1]
  if (!second) throw new Error('Discord Gateway invalid-session reconnect did not open a socket')
  second.emit({ op: 10, d: { heartbeat_interval: 20 } })
  await eventually(() => sentFrame(second, 2) !== undefined)
  expect(sentFrame(second, 6)).toBeUndefined()

  await channel.stop()
})

test('Discord Gateway requires explicit ports and surfaces discovery failures instead of reporting a fake start', async () => {
  expect(() => new DiscordChannel({ botToken: 'fixture-bot-token', transport: 'gateway' }))
    .toThrow('requires explicit gatewayPorts')

  const channel = new DiscordChannel({
    botToken: 'fixture-bot-token',
    gatewayPorts: {
      rest: { gatewayBot: async () => { throw new Error('fixture discovery failed') } },
      webSocket: new FakeWebSocketPort(),
    },
    registerCommands: false,
    transport: 'gateway',
  })
  await expect(channel.start(async () => {})).rejects.toThrow('fixture discovery failed')
  expect(channel.gatewayError).toBe('fixture discovery failed')
})

test('Discord Gateway registers app commands, acknowledges interactions, and routes normalized command messages', async () => {
  const clock = new FakeClock()
  const events: string[] = []
  const applicationRest = new FakeApplicationRestPort(events)
  const webSocket = new FakeWebSocketPort()
  const received: ChannelMessage[] = []
  const channel = new DiscordChannel({
    allowedGuildIds: ['guild-allowed'],
    applicationId: 'application-configured',
    applicationRest,
    botToken: 'fixture-bot-token',
    gatewayPorts: {
      clock,
      rest: new FakeRestPort('wss://gateway.fixture.test'),
      webSocket,
    },
    transport: 'gateway',
  })

  await channel.start(async message => {
    events.push(`inbound:${message.text}`)
    received.push(message)
  })
  const socket = onlySocket(webSocket)
  socket.emit({
    d: { application: { id: 'application-ready' }, guilds: [{ id: 'guild-from-ready' }], user: { id: 'bot-1' } },
    op: 0,
    s: 1,
    t: 'READY',
  })
  await eventually(() => applicationRest.guildCommands.length === 1)
  expect(applicationRest.globalCommands).toEqual([])
  expect(applicationRest.guildCommands[0]).toMatchObject({
    applicationId: 'application-configured',
    botToken: 'fixture-bot-token',
    guildId: 'guild-allowed',
  })
  expect(applicationRest.guildCommands[0]?.commands.map(command => command.name)).toEqual([
    'ask', 'skills', 'skill', 'status',
  ])

  const emitInteraction = (
    id: string,
    name: string,
    options: readonly Readonly<Record<string, unknown>>[] = [],
    guildId = 'guild-allowed',
  ): void => socket.emit({
    d: {
      channel: { id: 'channel-1', name: 'orders', parent_id: 'parent-1' },
      channel_id: 'channel-1',
      data: { name, options },
      guild: { name: 'Fixture Guild' },
      guild_id: guildId,
      id,
      member: { user: { id: 'user-1' } },
      token: `${id}-token`,
      type: 2,
    },
    op: 0,
    s: received.length + 2,
    t: 'INTERACTION_CREATE',
  })

  emitInteraction('ask-1', 'ask', [{ name: 'prompt', value: 'inspect the deployment' }])
  await eventually(() => received.length === 1)
  emitInteraction('skills-1', 'skills')
  await eventually(() => received.length === 2)
  emitInteraction('skill-1', 'skill', [
    { name: 'name', value: 'review:security' },
    { name: 'prompt', value: 'audit this PR' },
  ])
  await eventually(() => received.length === 3)
  emitInteraction('status-1', 'status')
  await eventually(() => received.length === 4)

  expect(received.map(message => message.text)).toEqual([
    'inspect the deployment',
    '/skills',
    '/skill review:security audit this PR',
    '/status',
  ])
  expect(received[0]).toMatchObject({
    channel: 'discord',
    channelUserId: 'user-1',
    platformMessageId: '',
    roomId: 'channel-1',
    metadata: {
      channel_name: 'orders',
      chat_type: 'group',
      discord_interaction: true,
      guild_id: 'guild-allowed',
      guild_name: 'Fixture Guild',
      parent_channel_id: 'parent-1',
      thread_id: 'channel-1',
    },
  })
  expect(applicationRest.acknowledgements).toEqual([
    { content: 'Queued.', ephemeral: true, interactionId: 'ask-1', interactionToken: 'ask-1-token' },
    { content: 'Queued.', ephemeral: true, interactionId: 'skills-1', interactionToken: 'skills-1-token' },
    { content: 'Queued.', ephemeral: true, interactionId: 'skill-1', interactionToken: 'skill-1-token' },
    { content: 'Queued.', ephemeral: true, interactionId: 'status-1', interactionToken: 'status-1-token' },
  ])
  expect(events).toEqual([
    'ack:ask-1',
    'inbound:inspect the deployment',
    'ack:skills-1',
    'inbound:/skills',
    'ack:skill-1',
    'inbound:/skill review:security audit this PR',
    'ack:status-1',
    'inbound:/status',
  ])

  emitInteraction('blocked-1', 'status', [], 'guild-blocked')
  await eventually(() => applicationRest.acknowledgements.length === 5)
  expect(applicationRest.acknowledgements.at(-1)).toEqual({
    content: 'This Xerxes instance is not configured for this channel.',
    ephemeral: true,
    interactionId: 'blocked-1',
    interactionToken: 'blocked-1-token',
  })
  expect(received).toHaveLength(4)
  await channel.stop()
})

test('Discord Gateway uses the READY application id for a global command sync when no guild is configured', async () => {
  const applicationRest = new FakeApplicationRestPort()
  const webSocket = new FakeWebSocketPort()
  const channel = new DiscordChannel({
    applicationRest,
    botToken: 'fixture-bot-token',
    gatewayPorts: { rest: new FakeRestPort('wss://gateway.fixture.test'), webSocket },
    transport: 'gateway',
  })

  await channel.start(async () => {})
  onlySocket(webSocket).emit({ d: { application: { id: 'application-ready' } }, op: 0, s: 1, t: 'READY' })
  await eventually(() => applicationRest.globalCommands.length === 1)
  expect(applicationRest.globalCommands[0]).toMatchObject({ applicationId: 'application-ready' })
  await channel.stop()
})

test('configured Discord Gateway declarations receive only the host-injected network ports', async () => {
  const clock = new FakeClock()
  const rest = new FakeRestPort('wss://gateway.fixture.test')
  const webSocket = new FakeWebSocketPort()
  const applicationRest = new FakeApplicationRestPort()
  const manager = new ConfiguredChannelManager({
    channels: {
      discord: {
        enabled: true,
        settings: { application_id: 'configured-app', bot_token: 'fixture-bot-token', transport: 'gateway' },
        type: 'discord',
      },
    },
    discordApplicationRest: applicationRest,
    discordGatewayPorts: { clock, rest, webSocket },
    onInbound: async () => {},
  })

  await manager.startConfigured()
  expect(rest.tokens).toEqual(['fixture-bot-token'])
  onlySocket(webSocket).emit({ d: { application: { id: 'ready-app' } }, op: 0, s: 1, t: 'READY' })
  await eventually(() => applicationRest.globalCommands.length === 1)
  expect(applicationRest.globalCommands[0]).toMatchObject({ applicationId: 'configured-app' })
  expect(manager.status('discord')).toMatchObject({ enabled: true })
  await manager.stopAll()
})

test('Discord application REST port bulk-replaces commands and acknowledges interactions without external I/O', async () => {
  const calls: Array<{ readonly body: unknown; readonly headers: Headers; readonly method: string; readonly url: string }> = []
  const port = new FetchDiscordApplicationRestPort({
    apiBaseUrl: 'https://discord.fixture.test/api/v10/',
    fetchImplementation: async (input, init) => {
      calls.push({
        body: init?.body === undefined ? undefined : JSON.parse(String(init.body)),
        headers: new Headers(init?.headers),
        method: init?.method ?? '',
        url: String(input),
      })
      return Response.json([])
    },
  })
  const commands = [{ description: 'Fixture command', name: 'fixture' }]

  await port.replaceGlobalCommands({ applicationId: 'app-1', botToken: 'bot-token', commands })
  await port.replaceGuildCommands({ applicationId: 'app-1', botToken: 'bot-token', commands, guildId: 'guild-1' })
  await port.acknowledgeInteraction({
    content: 'Queued.',
    ephemeral: true,
    interactionId: 'interaction-1',
    interactionToken: 'interaction-token',
  })

  expect(calls).toEqual([
    {
      body: commands,
      headers: expect.any(Headers),
      method: 'PUT',
      url: 'https://discord.fixture.test/api/v10/applications/app-1/commands',
    },
    {
      body: commands,
      headers: expect.any(Headers),
      method: 'PUT',
      url: 'https://discord.fixture.test/api/v10/applications/app-1/guilds/guild-1/commands',
    },
    {
      body: { data: { content: 'Queued.', flags: 64 }, type: 4 },
      headers: expect.any(Headers),
      method: 'POST',
      url: 'https://discord.fixture.test/api/v10/interactions/interaction-1/interaction-token/callback',
    },
  ])
  expect(calls[0]?.headers.get('authorization')).toBe('Bot bot-token')
  expect(calls[1]?.headers.get('authorization')).toBe('Bot bot-token')
  expect(calls[2]?.headers.get('authorization')).toBeNull()
})

test('injected Discord REST and Bun WebSocket ports perform the wire path against a local Gateway only', async () => {
  const received: ChannelMessage[] = []
  const clientFrames: GatewayFrame[] = []
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    fetch(request, gateway) {
      if (gateway.upgrade(request)) return
      return new Response('Not Found', { status: 404 })
    },
    websocket: {
      message(socket, value) {
        const frame = JSON.parse(String(value)) as GatewayFrame
        clientFrames.push(frame)
        if (frame.op === 1) {
          socket.send(JSON.stringify({ op: 11, d: null }))
        }
        if (frame.op !== 2) return
        socket.send(JSON.stringify({
          d: { resume_gateway_url: `ws://127.0.0.1:${server.port}`, session_id: 'local-session', user: { id: 'local-bot' } },
          op: 0,
          s: 1,
          t: 'READY',
        }))
        socket.send(JSON.stringify({
          d: { author: { id: 'local-user' }, channel_id: 'dm-1', content: 'hello local gateway', id: 'local-message' },
          op: 0,
          s: 2,
          t: 'MESSAGE_CREATE',
        }))
      },
      open(socket) {
        socket.send(JSON.stringify({ op: 10, d: { heartbeat_interval: 60_000 } }))
      },
    },
  })
  const port = new FetchDiscordGatewayRestPort({
    apiBaseUrl: 'https://discord.fixture.test/api/v10/',
    fetchImplementation: async (input, init) => {
      expect(String(input)).toBe('https://discord.fixture.test/api/v10/gateway/bot')
      expect(init?.headers).toMatchObject({ Authorization: 'Bot fixture-bot-token' })
      expect(init?.method).toBe('GET')
      return Response.json({ url: `ws://127.0.0.1:${server.port}` })
    },
  })
  const channel = new DiscordChannel({
    botToken: 'fixture-bot-token',
    gatewayPorts: { rest: port, webSocket: new BunDiscordGatewayWebSocketPort() },
    registerCommands: false,
    transport: 'gateway',
  })

  try {
    await channel.start(async message => { received.push(message) })
    await eventually(() => received.length === 1 && clientFrames.some(frame => frame.op === 2))
    expect(received[0]).toMatchObject({ roomId: 'dm-1', text: 'hello local gateway' })
    expect(clientFrames.find(frame => frame.op === 2)?.d).toMatchObject({ token: 'fixture-bot-token' })
  } finally {
    await channel.stop()
    await server.stop(true)
  }
})

interface GatewayFrame {
  readonly d?: unknown
  readonly op: number
  readonly s?: number
  readonly t?: string
}

class FakeRestPort implements DiscordGatewayRestPort {
  readonly tokens: string[] = []

  constructor(private readonly url: string) {}

  async gatewayBot(botToken: string): Promise<{ readonly url: string }> {
    this.tokens.push(botToken)
    return { url: this.url }
  }
}

class FakeApplicationRestPort implements DiscordApplicationRestPort {
  readonly acknowledgements: DiscordInteractionAcknowledgement[] = []
  readonly globalCommands: DiscordApplicationCommandRequest[] = []
  readonly guildCommands: DiscordGuildApplicationCommandRequest[] = []

  constructor(private readonly events: string[] = []) {}

  async acknowledgeInteraction(request: DiscordInteractionAcknowledgement): Promise<void> {
    this.acknowledgements.push(request)
    this.events.push(`ack:${request.interactionId}`)
  }

  async replaceGlobalCommands(request: DiscordApplicationCommandRequest): Promise<void> {
    this.globalCommands.push(request)
  }

  async replaceGuildCommands(request: DiscordGuildApplicationCommandRequest): Promise<void> {
    this.guildCommands.push(request)
  }
}

class FakeWebSocketPort implements DiscordGatewayWebSocketPort {
  readonly connections: FakeGatewaySocket[] = []

  async connect(url: string, handlers: DiscordGatewayConnectionHandlers): Promise<DiscordGatewaySocket> {
    const socket = new FakeGatewaySocket(url, handlers)
    this.connections.push(socket)
    handlers.onOpen(socket)
    return socket
  }
}

class FakeGatewaySocket implements DiscordGatewaySocket {
  readonly closeEvents: DiscordGatewayCloseEvent[] = []
  readonly frames: GatewayFrame[] = []

  constructor(
    readonly url: string,
    private readonly handlers: DiscordGatewayConnectionHandlers,
  ) {}

  close(code = 1000, reason = ''): void {
    this.closeEvents.push({ code, reason })
  }

  emit(frame: GatewayFrame): void {
    this.handlers.onMessage(this, JSON.stringify(frame))
  }

  send(payload: string): void {
    this.frames.push(JSON.parse(payload) as GatewayFrame)
  }
}

class FakeClock implements DiscordGatewayClock {
  private nextId = 1
  private readonly timers = new Map<number, { readonly callback: () => void; readonly milliseconds: number }>()

  clearTimeout(timer: unknown): void {
    if (typeof timer === 'number') this.timers.delete(timer)
  }

  setTimeout(callback: () => void, milliseconds: number): number {
    const id = this.nextId
    this.nextId += 1
    this.timers.set(id, { callback, milliseconds })
    return id
  }

  async runNext(milliseconds: number): Promise<void> {
    const candidate = [...this.timers.entries()].find(([, timer]) => timer.milliseconds === milliseconds)
    if (!candidate) throw new Error(`No scheduled timer for ${milliseconds}ms`)
    const [id, timer] = candidate
    this.timers.delete(id)
    timer.callback()
    await settle()
  }
}

function onlySocket(port: FakeWebSocketPort): FakeGatewaySocket {
  const socket = port.connections[0]
  if (!socket) throw new Error('Discord Gateway did not open its first socket')
  return socket
}

function sentFrame(socket: FakeGatewaySocket, opcode: number): GatewayFrame | undefined {
  return socket.frames.find(frame => frame.op === opcode)
}

function sentFrames(socket: FakeGatewaySocket, opcode: number): readonly GatewayFrame[] {
  return socket.frames.filter(frame => frame.op === opcode)
}

async function eventually(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) return
    await Bun.sleep(2)
  }
  throw new Error('Timed out waiting for Discord Gateway state')
}

async function settle(): Promise<void> {
  await Promise.resolve()
  await Promise.resolve()
  await Bun.sleep(0)
}
