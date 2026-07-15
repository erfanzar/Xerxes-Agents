// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  BLUEBUBBLES_TRANSPORT,
  BlueBubblesChannel,
  createChannelMessage,
  DINGTALK_TRANSPORT,
  DingTalkChannel,
  FEISHU_TRANSPORT,
  FeishuChannel,
  HOME_ASSISTANT_TRANSPORT,
  HomeAssistantChannel,
  MATRIX_TRANSPORT,
  MatrixChannel,
  MATTERMOST_TRANSPORT,
  MattermostChannel,
  MessageDirection,
  SIGNAL_TRANSPORT,
  SignalChannel,
  TWILIO_SMS_TRANSPORT,
  TwilioSmsChannel,
  UNSUPPORTED_CHANNEL_TRANSPORTS,
  WECOM_TRANSPORT,
  WeComChannel,
  WHATSAPP_TRANSPORT,
  WhatsAppChannel,
  whatsAppWebhookChallenge,
  type ChannelFetch,
  type ChannelMessage,
} from '../src/channels/index.js'

const encoder = new TextEncoder()

interface FetchCall {
  readonly body: string
  readonly headers: Headers
  readonly method: string
  readonly url: string
}

function recordingFetch(calls: FetchCall[]): ChannelFetch {
  return async (input, init) => {
    calls.push({
      url: String(input),
      method: init?.method ?? '',
      headers: new Headers(init?.headers),
      body: requestBody(init?.body),
    })
    return Response.json({ ok: true })
  }
}

function requestBody(body: BodyInit | null | undefined): string {
  if (body instanceof URLSearchParams) {
    return body.toString()
  }
  if (typeof body === 'string') {
    return body
  }
  return body === undefined || body === null ? '' : String(body)
}

function outbound(channel: string, fields: Partial<ChannelMessage> = {}): ChannelMessage {
  return createChannelMessage({
    channel,
    direction: MessageDirection.OUTBOUND,
    text: fields.text ?? 'agent reply',
    ...(fields.channelUserId ? { channelUserId: fields.channelUserId } : {}),
    ...(fields.replyTo ? { replyTo: fields.replyTo } : {}),
    ...(fields.roomId ? { roomId: fields.roomId } : {}),
  })
}

function requiredCall(calls: readonly FetchCall[]): FetchCall {
  const call = calls[0]
  if (!call) {
    throw new Error('expected a provider HTTP request')
  }
  return call
}

test('relay-only adapters expose their unsupported persistent transports', () => {
  expect(MATTERMOST_TRANSPORT).toMatchObject({ inbound: 'webhook-relay', outbound: 'http-api' })
  expect(MATRIX_TRANSPORT.unsupported).toContain('Matrix /sync polling')
  expect(FEISHU_TRANSPORT.unsupported).toContain('persistent WebSocket event delivery')
  expect(WECOM_TRANSPORT.unsupported).toContain('encrypted XML callback decryption')
  expect(DINGTALK_TRANSPORT.unsupported).toContain('DingTalk stream-mode connections')
  expect(HOME_ASSISTANT_TRANSPORT.unsupported).toContain('Home Assistant WebSocket event subscriptions')
  expect(BLUEBUBBLES_TRANSPORT.unsupported).toContain('BlueBubbles persistent event socket')
  expect(SIGNAL_TRANSPORT.unsupported).toContain('signal-cli receive loop')
  expect(WHATSAPP_TRANSPORT.unsupported).toContain('persistent WhatsApp socket transport')
  expect(TWILIO_SMS_TRANSPORT.unsupported).toContain('Twilio X-Twilio-Signature verification')
  expect(UNSUPPORTED_CHANNEL_TRANSPORTS.email_imap.reason).toContain('direct SMTP delivery')
})

test('Mattermost relay parses outgoing-webhook messages and posts threaded REST replies', async () => {
  const calls: FetchCall[] = []
  const channel = new MattermostChannel({
    baseUrl: 'https://mattermost.test',
    botToken: 'mattermost-token',
    fetchImplementation: recordingFetch(calls),
  })
  const received: ChannelMessage[] = []
  await channel.start(async message => { received.push(message) })

  await channel.handleWebhook({}, encoder.encode(JSON.stringify({
    text: 'hello', user_id: 'U1', channel_id: 'C1', post_id: 'P1', team_id: 'T1',
  })))
  expect(received[0]).toMatchObject({
    channel: 'mattermost', text: 'hello', channelUserId: 'U1', roomId: 'C1',
    platformMessageId: 'P1', metadata: { team_id: 'T1' },
  })

  await channel.send(outbound('mattermost', { roomId: 'C1', replyTo: 'P1' }))
  expect(calls[0]).toMatchObject({
    method: 'POST',
    url: 'https://mattermost.test/api/v4/posts',
    body: JSON.stringify({ channel_id: 'C1', message: 'agent reply', root_id: 'P1' }),
  })
  expect(calls[0]?.headers.get('authorization')).toBe('Bearer mattermost-token')
})

test('Matrix relay accepts room events and sends an idempotent client-server PUT', async () => {
  const calls: FetchCall[] = []
  const channel = new MatrixChannel({
    homeserverUrl: 'https://matrix.test',
    accessToken: 'matrix-token',
    transactionId: () => 'txn-1',
    fetchImplementation: recordingFetch(calls),
  })
  const received: ChannelMessage[] = []
  await channel.start(async message => { received.push(message) })

  await channel.handleWebhook({}, encoder.encode(JSON.stringify({ events: [{
    type: 'm.room.message', sender: '@erfan:example.test', room_id: '!room:example.test',
    event_id: '$event', content: { msgtype: 'm.text', body: 'matrix hello' },
  }, { type: 'm.room.member' }] })))
  expect(received).toHaveLength(1)
  expect(received[0]).toMatchObject({ text: 'matrix hello', roomId: '!room:example.test' })

  await channel.send(outbound('matrix', { roomId: '!room:example.test' }))
  const request = requiredCall(calls)
  expect(request?.method).toBe('PUT')
  expect(new URL(request.url).pathname).toBe(
    '/_matrix/client/v3/rooms/!room%3Aexample.test/send/m.room.message/txn-1',
  )
  expect(request?.body).toBe(JSON.stringify({ msgtype: 'm.text', body: 'agent reply' }))
  expect(request?.headers.get('authorization')).toBe('Bearer matrix-token')
})

test('Feishu URL verification, inbound content decoding, and refreshed-token output work through HTTP', async () => {
  const calls: FetchCall[] = []
  const channel = new FeishuChannel({
    apiBaseUrl: 'https://feishu.test',
    tokenProvider: () => 'fresh-token',
    fetchImplementation: recordingFetch(calls),
  })
  expect(await channel.handleWebhook({}, encoder.encode(JSON.stringify({
    type: 'url_verification', challenge: 'challenge-value',
  })))).toEqual({
    status: 200,
    body: 'challenge-value',
    headers: { 'content-type': 'text/plain; charset=utf-8' },
  })

  const received: ChannelMessage[] = []
  await channel.start(async message => { received.push(message) })
  await channel.handleWebhook({}, encoder.encode(JSON.stringify({ event: {
    sender: { sender_id: { open_id: 'ou_1' } },
    message: {
      chat_id: 'oc_1', message_id: 'om_1', message_type: 'text',
      content: JSON.stringify({ text: 'Lark hello' }),
    },
  } })))
  expect(received[0]).toMatchObject({
    text: 'Lark hello', channelUserId: 'ou_1', roomId: 'oc_1', platformMessageId: 'om_1',
  })

  await channel.send(outbound('feishu', { roomId: 'oc_1' }))
  const request = requiredCall(calls)
  expect(new URL(request.url).searchParams.get('receive_id_type')).toBe('chat_id')
  expect(request?.body).toBe(JSON.stringify({
    receive_id: 'oc_1', msg_type: 'text', content: JSON.stringify({ text: 'agent reply' }),
  }))
  expect(request?.headers.get('authorization')).toBe('Bearer fresh-token')
})

test('WeCom, DingTalk, and Home Assistant map practical webhook/API shapes', async () => {
  const wecomCalls: FetchCall[] = []
  const wecom = new WeComChannel({
    accessToken: 'wecom-token', agentId: 42, apiBaseUrl: 'https://wecom.test',
    fetchImplementation: recordingFetch(wecomCalls),
  })
  const wecomInbound: ChannelMessage[] = []
  await wecom.start(async message => { wecomInbound.push(message) })
  await wecom.handleWebhook({}, encoder.encode(JSON.stringify({
    Content: 'enterprise hello', FromUserName: 'alice', MsgId: 'm1', Event: 'text',
  })))
  expect(wecomInbound[0]).toMatchObject({ text: 'enterprise hello', roomId: 'alice' })
  await wecom.send(outbound('wecom', { channelUserId: 'alice' }))
  expect(new URL(requiredCall(wecomCalls).url).searchParams.get('access_token')).toBe('wecom-token')
  expect(wecomCalls[0]?.body).toBe(JSON.stringify({
    touser: 'alice', msgtype: 'text', agentid: 42, text: { content: 'agent reply' },
  }))

  const dingCalls: FetchCall[] = []
  const ding = new DingTalkChannel({
    webhookUrl: 'https://dingtalk.test/robot/send?access_token=token',
    fetchImplementation: recordingFetch(dingCalls),
  })
  const dingInbound: ChannelMessage[] = []
  await ding.start(async message => { dingInbound.push(message) })
  await ding.handleWebhook({}, encoder.encode(JSON.stringify({
    text: { content: 'ding hello' }, senderId: 'staff-1', conversationId: 'conv-1',
    msgId: 'msg-1', senderNick: 'Erfan',
  })))
  expect(dingInbound[0]).toMatchObject({
    text: 'ding hello', roomId: 'conv-1', metadata: { sender_nick: 'Erfan' },
  })
  await ding.send(outbound('dingtalk'))
  expect(dingCalls[0]?.body).toBe(JSON.stringify({ msgtype: 'text', text: { content: 'agent reply' } }))

  const homeCalls: FetchCall[] = []
  const home = new HomeAssistantChannel({
    baseUrl: 'https://home.test', accessToken: 'ha-token', notificationTitle: 'Assistant',
    fetchImplementation: recordingFetch(homeCalls),
  })
  const homeInbound: ChannelMessage[] = []
  await home.start(async message => { homeInbound.push(message) })
  await home.handleWebhook({}, encoder.encode(JSON.stringify({
    input: { text: 'turn lights on' }, user_id: 'home-user', conversation_id: 'conv',
    event_id: 'event', language: 'tr',
  })))
  expect(homeInbound[0]).toMatchObject({ text: 'turn lights on', metadata: { language: 'tr' } })
  await home.send(outbound('home_assistant'))
  expect(homeCalls[0]?.url).toBe('https://home.test/api/services/persistent_notification/create')
  expect(JSON.parse(homeCalls[0]?.body ?? '{}')).toMatchObject({
    title: 'Assistant', message: 'agent reply', notification_id: expect.any(String),
  })
  expect(homeCalls[0]?.headers.get('authorization')).toBe('Bearer ha-token')
})

test('zero-is-success provider envelopes do not hide API-level failures', async () => {
  const channel = new DingTalkChannel({
    webhookUrl: 'https://dingtalk.test/robot/send?access_token=token',
    fetchImplementation: async () => Response.json({ errcode: 310000, errmsg: 'rejected' }),
  })
  await expect(channel.send(outbound('dingtalk'))).rejects.toThrow('DingTalk API request failed')
})

test('BlueBubbles and Signal support relay-in/API-out without owning their gateway loops', async () => {
  const blueCalls: FetchCall[] = []
  const blue = new BlueBubblesChannel({
    serverUrl: 'https://blue.test', password: 'secret pass', fetchImplementation: recordingFetch(blueCalls),
  })
  const blueInbound: ChannelMessage[] = []
  await blue.start(async message => { blueInbound.push(message) })
  await blue.handleWebhook({}, encoder.encode(JSON.stringify({ data: {
    body: 'iMessage hello', guid: 'message-guid', chats: [{ guid: 'chat-guid' }],
    handle: { address: '+15550001' },
  } })))
  expect(blueInbound[0]).toMatchObject({ text: 'iMessage hello', roomId: 'chat-guid' })
  await blue.send(outbound('bluebubbles', { roomId: 'chat-guid' }))
  expect(new URL(requiredCall(blueCalls).url).searchParams.get('password')).toBe('secret pass')
  expect(blueCalls[0]?.body).toBe(JSON.stringify({
    chatGuid: 'chat-guid', message: 'agent reply', method: 'private-api',
  }))

  const signalCalls: FetchCall[] = []
  const signal = new SignalChannel({
    restBaseUrl: 'https://signal.test', senderNumber: '+15550100', fetchImplementation: recordingFetch(signalCalls),
  })
  const signalInbound: ChannelMessage[] = []
  await signal.start(async message => { signalInbound.push(message) })
  await signal.handleWebhook({}, encoder.encode(JSON.stringify({ envelope: {
    sourceNumber: '+15550001', timestamp: 123, dataMessage: { message: 'signal hello' },
  } })))
  expect(signalInbound[0]).toMatchObject({ text: 'signal hello', roomId: '+15550001' })
  await signal.send(outbound('signal', { roomId: '+15550001' }))
  expect(signalCalls[0]).toMatchObject({
    url: 'https://signal.test/v2/send',
    body: JSON.stringify({
      number: '+15550100', recipients: ['+15550001'], message: 'agent reply',
    }),
  })
})

test('WhatsApp unpacks batched Cloud API webhooks, sends Graph text, and exposes verification helper', async () => {
  const calls: FetchCall[] = []
  const channel = new WhatsAppChannel({
    apiBaseUrl: 'https://graph.test', apiVersion: 'v99.0', accessToken: 'wa-token', phoneNumberId: 'phone-id',
    fetchImplementation: recordingFetch(calls),
  })
  const received: ChannelMessage[] = []
  await channel.start(async message => { received.push(message) })
  await channel.handleWebhook({}, encoder.encode(JSON.stringify({ entry: [{ changes: [{ value: { messages: [
    { from: '15550001', id: 'wamid-1', type: 'text', text: { body: 'WhatsApp hello' } },
    { from: '15550002', id: 'wamid-2', type: 'interactive', interactive: { button_reply: { title: 'Choose me' } } },
  ] } }] }] })))
  expect(received.map(message => message.text)).toEqual(['WhatsApp hello', 'Choose me'])
  await channel.send(outbound('whatsapp', { roomId: '15550001' }))
  expect(calls[0]).toMatchObject({
    url: 'https://graph.test/v99.0/phone-id/messages',
    body: JSON.stringify({
      messaging_product: 'whatsapp', to: '15550001', type: 'text', text: { body: 'agent reply' },
    }),
  })
  expect(calls[0]?.headers.get('authorization')).toBe('Bearer wa-token')
  expect(whatsAppWebhookChallenge({
    'hub.mode': 'subscribe', 'hub.verify_token': 'match', 'hub.challenge': 'challenge',
  }, 'match')).toBe('challenge')
  expect(whatsAppWebhookChallenge({ 'hub.mode': 'subscribe' }, 'match')).toBeUndefined()
})

test('Twilio parses form callbacks and sends a Basic-auth URL-encoded SMS request', async () => {
  const calls: FetchCall[] = []
  const channel = new TwilioSmsChannel({
    accountSid: 'AC123', authToken: 'auth-token', fromNumber: '+15550100', apiBaseUrl: 'https://twilio.test',
    fetchImplementation: recordingFetch(calls),
  })
  const received: ChannelMessage[] = []
  await channel.start(async message => { received.push(message) })
  await channel.handleWebhook({}, encoder.encode('Body=SMS+hello&From=%2B15550001&To=%2B15550100&MessageSid=SM1'))
  expect(received[0]).toMatchObject({
    text: 'SMS hello', channelUserId: '+15550001', roomId: '+15550001', metadata: { to: '+15550100' },
  })
  await channel.send(outbound('sms', { roomId: '+15550001' }))
  expect(calls[0]?.url).toBe('https://twilio.test/2010-04-01/Accounts/AC123/Messages.json')
  expect([...new URLSearchParams(calls[0]?.body).entries()]).toEqual(expect.arrayContaining([
    ['From', '+15550100'], ['To', '+15550001'], ['Body', 'agent reply'],
  ]))
  expect(calls[0]?.headers.get('authorization')).toBe(
    `Basic ${Buffer.from('AC123:auth-token').toString('base64')}`,
  )
  expect(calls[0]?.headers.get('content-type')).toContain('application/x-www-form-urlencoded')
})
