// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createServer } from 'node:net'

import {
  decodeMimeHeader,
  EmailChannel,
  EmailChannelConfigurationError,
  EmailTransportUnavailableError,
  htmlToEmailText,
  type EmailInboundMessage,
  type EmailSmtpSendRequest,
} from '../src/channels/emailImap.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from '../src/channels/types.js'

const encoder = new TextEncoder()

test('email channel normalizes webhook and IMAP inbound mail, then supplies SMTP MIME requests', async () => {
  const smtpRequests: EmailSmtpSendRequest[] = []
  let imapInbound: ((message: EmailInboundMessage) => Promise<void>) | undefined
  let imapStops = 0
  const channel = new EmailChannel({
    smtpHost: 'smtp.example.test',
    smtpPort: 587,
    smtpUser: 'smtp-user',
    smtpPassword: 'smtp-password',
    fromAddress: 'xerxes@example.test',
    smtpTransport: { send: async request => { smtpRequests.push(request) } },
    imapTransport: {
      start: async onInbound => { imapInbound = onInbound },
      stop: async () => { imapStops += 1 },
    },
  })
  const received: ChannelMessage[] = []
  await channel.start(async message => { received.push(message) })

  expect(await channel.handleWebhook({}, encoder.encode(JSON.stringify({
    from: 'Alice <alice@example.test>',
    to: 'xerxes@example.test',
    subject: '=?UTF-8?Q?Re=3A_status?=',
    text: 'Hello=20world=0Aagain',
    text_encoding: 'quoted-printable',
    message_id: 'message-1',
    in_reply_to: 'parent-1',
    attachments: [{ filename: 'report.txt', content_type: 'text/plain' }],
  })))).toEqual({ status: 200, body: 'ok' })
  expect(received[0]).toMatchObject({
    channel: 'email',
    text: 'Hello world\nagain',
    channelUserId: 'alice@example.test',
    platformMessageId: 'message-1',
    replyTo: 'parent-1',
    metadata: { subject: 'Re: status' },
    attachments: [{ filename: 'report.txt', content_type: 'text/plain' }],
  })
  // Our own recipient address must not become the routing room.
  expect(received[0]?.roomId).toBeUndefined()

  if (!imapInbound) {
    throw new Error('IMAP transport did not receive an inbound callback')
  }
  await imapInbound({
    from: 'Bob <bob@example.test>',
    to: ['xerxes@example.test'],
    subject: 'HTML only',
    html: '<p>Only&nbsp;<strong>HTML</strong></p>',
    messageId: 'message-2',
  })
  expect(received[1]).toMatchObject({
    text: 'Only HTML',
    channelUserId: 'bob@example.test',
    platformMessageId: 'message-2',
    metadata: { subject: 'HTML only' },
  })
  expect(received[1]?.roomId).toBeUndefined()

  await channel.send(createChannelMessage({
    channel: 'email',
    direction: MessageDirection.OUTBOUND,
    roomId: 'alice@example.test',
    replyTo: 'message-1',
    text: 'agent reply',
    metadata: { subject: 'Re: Status' },
  }))
  expect(smtpRequests).toHaveLength(1)
  expect(smtpRequests[0]).toMatchObject({
    host: 'smtp.example.test',
    port: 587,
    from: 'xerxes@example.test',
    to: 'alice@example.test',
    subject: 'Re: Status',
    text: 'agent reply',
    startTls: true,
    authentication: { username: 'smtp-user', password: 'smtp-password' },
  })
  expect(smtpRequests[0]?.mime).toContain('Content-Type: text/plain; charset=utf-8')
  expect(smtpRequests[0]?.mime).toContain('In-Reply-To: message-1')
  expect(smtpRequests[0]?.mime).toContain('YWdlbnQgcmVwbHk=')

  await channel.stop()
  expect(imapStops).toBe(1)
})

test('email reply follows the original sender instead of our own mailbox', async () => {
  const smtpRequests: EmailSmtpSendRequest[] = []
  const channel = new EmailChannel({
    smtpHost: 'smtp.example.test',
    fromAddress: 'xerxes@example.test',
    smtpTransport: { send: async request => { smtpRequests.push(request) } },
  })
  const received: ChannelMessage[] = []
  await channel.start(async message => { received.push(message) })

  expect(await channel.handleWebhook({}, encoder.encode(JSON.stringify({
    from: 'Alice <alice@example.test>',
    to: 'xerxes@example.test',
    subject: 'question',
    text: 'how does this work?',
    message_id: 'inbound-1',
  })))).toEqual({ status: 200, body: 'ok' })
  const inbound = received[0]
  if (!inbound) throw new Error('inbound email was not delivered')

  // Mirror ChannelTurnRouter.reply: copy the correspondent identity onto the outbound message.
  await channel.send(createChannelMessage({
    channel: 'email',
    direction: MessageDirection.OUTBOUND,
    metadata: inbound.metadata,
    text: 'here is the answer',
    ...(inbound.channelUserId === undefined ? {} : { channelUserId: inbound.channelUserId }),
    ...(inbound.platformMessageId === undefined ? {} : { replyTo: inbound.platformMessageId }),
    ...(inbound.roomId === undefined ? {} : { roomId: inbound.roomId }),
  }))

  expect(smtpRequests).toHaveLength(1)
  expect(smtpRequests[0]?.to).toBe('alice@example.test')
  expect(smtpRequests[0]?.mime).toContain('To: alice@example.test')
  expect(smtpRequests[0]?.mime).not.toContain('To: xerxes@example.test')
  await channel.stop()
})

test('email channel uses Bun native SMTP when no host sender is injected', async () => {
  const commands: string[] = []
  const bodies: string[] = []
  await withSmtpFixture(async port => {
    const channel = new EmailChannel({
      fromAddress: 'xerxes@example.test',
      smtpHost: '127.0.0.1',
      smtpPort: port,
    })
    await channel.send(createChannelMessage({
      channel: 'email',
      direction: MessageDirection.OUTBOUND,
      roomId: 'alice@example.test',
      text: '.leading line\nsecond line',
    }))
  }, commands, bodies)

  expect(commands).toEqual([
    'EHLO xerxes',
    'MAIL FROM:<xerxes@example.test>',
    'RCPT TO:<alice@example.test>',
    'DATA',
    'QUIT',
  ])
  expect(bodies[0]).toContain('Content-Transfer-Encoding: base64')
  expect(bodies[0]).toContain('LmxlYWRpbmcgbGluZQpzZWNvbmQgbGluZQ==')
})

test('email channel validates configuration and fails clearly when direct SMTP is explicitly disabled', async () => {
  expect(() => new EmailChannel({ smtpPort: 0 })).toThrow(EmailChannelConfigurationError)
  expect(() => new EmailChannel({ smtpUser: 'user' })).toThrow(EmailChannelConfigurationError)
  expect(() => new EmailChannel({ fromAddress: 'bad\naddress@example.test' }))
    .toThrow(EmailChannelConfigurationError)

  const outbound = new EmailChannel({ directSmtp: false, fromAddress: 'xerxes@example.test' })
  await expect(outbound.send(createChannelMessage({
    channel: 'email',
    direction: MessageDirection.OUTBOUND,
    roomId: 'recipient@example.test',
    text: 'reply',
  }))).rejects.toBeInstanceOf(EmailTransportUnavailableError)

  const inbound = new EmailChannel({ requireImapTransport: true })
  await expect(inbound.start(async () => {})).rejects.toBeInstanceOf(EmailTransportUnavailableError)
  expect(decodeMimeHeader('=?UTF-8?B?4pyT?=')).toBe('✓')
  expect(htmlToEmailText('<p>Hello&nbsp;<b>world</b></p>')).toBe('Hello world')
})

async function withSmtpFixture(
  run: (port: number) => Promise<void>,
  commands: string[],
  bodies: string[],
): Promise<void> {
  const server = createServer(socket => {
    let buffer = ''
    let dataLines: string[] | undefined
    socket.write('220 fixture SMTP\r\n')
    socket.on('data', chunk => {
      buffer += chunk.toString()
      while (true) {
        const index = buffer.indexOf('\n')
        if (index < 0) return
        const line = buffer.slice(0, index).replace(/\r$/, '')
        buffer = buffer.slice(index + 1)
        if (dataLines !== undefined) {
          if (line === '.') {
            bodies.push(dataLines.join('\r\n'))
            dataLines = undefined
            socket.write('250 accepted\r\n')
          } else {
            dataLines.push(line)
          }
          continue
        }
        commands.push(line)
        if (line.startsWith('EHLO ')) {
          socket.write('250-fixture\r\n250 PIPELINING\r\n')
        } else if (line === 'DATA') {
          dataLines = []
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
