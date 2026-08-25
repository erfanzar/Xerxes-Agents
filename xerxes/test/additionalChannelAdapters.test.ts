// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHmac } from 'node:crypto'

import { expect, spyOn, test } from 'bun:test'

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

/** Compute the expected Twilio signature: base64 HMAC-SHA1 over URL + sorted concatenated values. */
function twilioSignature(url: string, formBody: string, authToken: string): string {
  const values = [...new URLSearchParams(formBody).entries()]
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([, value]) => value)
    .join('')
  return createHmac('sha1', authToken).update(url + values).digest('base64')
}

/** Compute Meta's expected X-Hub-Signature-256 value for a raw webhook body. */
function hubSignature(body: Uint8Array, appSecret: string): string {
  return 'sha256=' + createHmac('sha256', appSecret).update(body).digest('hex')
}

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
  expect(WHATSAPP_TRANSPORT.unsupported).not.toContain('webhook signature verification')
  expect(TWILIO_SMS_TRANSPORT.unsupported).not.toContain('Twilio X-Twilio-Signature verification')
  expect(TWILIO_SMS_TRANSPORT.unsupported).toContain('MMS media download and delivery callbacks')
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
  await blue.handleWebhook({}, encoder.encode(JSON.stringify({ data: {
    body: 'agent echo', guid: 'self-message-guid', isFromMe: true, chats: [{ guid: 'chat-guid' }],
    handle: { address: '+15550001' },
  } })))
  expect(blueInbound).toHaveLength(1)
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
  // No appSecret is configured here, so the first webhook warns once about
  // unverified signatures; capture it instead of leaking noise into test output.
  const warnings: unknown[][] = []
  const warnSpy = spyOn(console, 'warn').mockImplementation((...args: unknown[]) => {
    warnings.push(args)
  })
  let response: Awaited<ReturnType<typeof channel.handleWebhook>>
  try {
    response = await channel.handleWebhook({}, encoder.encode(JSON.stringify({ entry: [{ changes: [{ value: { messages: [
      { from: '15550001', id: 'wamid-1', type: 'text', text: { body: 'WhatsApp hello' } },
      { from: '15550002', id: 'wamid-2', type: 'interactive', interactive: { button_reply: { title: 'Choose me' } } },
    ] } }] }] })))
  } finally {
    warnSpy.mockRestore()
  }
  expect(response).toEqual({ status: 200, body: 'ok' })
  expect(received.map(message => message.text)).toEqual(['WhatsApp hello', 'Choose me'])
  expect(warnings).toHaveLength(1)
  expect(String(warnings[0]?.[0])).toContain('app_secret')
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

test('Twilio verifies X-Twilio-Signature and parses form callbacks before sending a Basic-auth SMS', async () => {
  const calls: FetchCall[] = []
  const webhookUrl = 'https://edge.test/channels/sms/webhook'
  const channel = new TwilioSmsChannel({
    accountSid: 'AC123', authToken: 'auth-token', fromNumber: '+15550100', apiBaseUrl: 'https://twilio.test',
    fetchImplementation: recordingFetch(calls),
    webhookUrl,
  })
  const received: ChannelMessage[] = []
  await channel.start(async message => { received.push(message) })
  const form = 'Body=SMS+hello&From=%2B15550001&To=%2B15550100&MessageSid=SM1'
  const response = await channel.handleWebhook(
    { 'X-Twilio-Signature': twilioSignature(webhookUrl, form, 'auth-token') },
    encoder.encode(form),
  )
  expect(response).toEqual({ status: 200, body: 'ok' })
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

test('Twilio rejects inbound webhooks with a missing or unverifiable X-Twilio-Signature', async () => {
  const webhookUrl = 'https://edge.test/channels/sms/webhook'
  const channel = new TwilioSmsChannel({
    accountSid: 'AC123', authToken: 'auth-token', fromNumber: '+15550100', webhookUrl,
  })
  let dispatched = 0
  await channel.start(async () => { dispatched += 1 })
  const form = 'Body=spoofed&From=%2B15559999'
  const body = encoder.encode(form)

  // No signature header at all.
  expect(await channel.handleWebhook({}, body)).toEqual({ status: 401, body: 'unauthorized' })
  // Malformed signature.
  expect(await channel.handleWebhook({ 'X-Twilio-Signature': 'garbage' }, body))
    .toEqual({ status: 401, body: 'unauthorized' })
  // Valid signature computed over a different URL (spoofed edge).
  expect(await channel.handleWebhook(
    { 'X-Twilio-Signature': twilioSignature('https://evil.test/channels/sms/webhook', form, 'auth-token') },
    body,
  )).toEqual({ status: 401, body: 'unauthorized' })
  // Signature for the wrong auth token.
  expect(await channel.handleWebhook(
    { 'x-twilio-signature': twilioSignature(webhookUrl, form, 'other-token') },
    body,
  )).toEqual({ status: 401, body: 'unauthorized' })
  expect(dispatched).toBe(0)
})

test('Twilio reconstructs the signed URL from forwarded headers when no webhookUrl is configured', async () => {
  const channel = new TwilioSmsChannel({
    accountSid: 'AC123', authToken: 'auth-token', fromNumber: '+15550100',
  })
  const received: ChannelMessage[] = []
  await channel.start(async message => { received.push(message) })
  const form = 'Body=proxied&From=%2B15550001'
  const reconstructedUrl = 'https://proxy.test/channels/sms/webhook'
  const response = await channel.handleWebhook(
    {
      Host: 'internal.test',
      'X-Forwarded-Proto': 'https',
      'X-Forwarded-Host': 'proxy.test',
      'X-Twilio-Signature': twilioSignature(reconstructedUrl, form, 'auth-token'),
    },
    encoder.encode(form),
  )
  expect(response).toEqual({ status: 200, body: 'ok' })
  expect(received[0]?.text).toBe('proxied')

  // The same form signed for the wrong scheme fails: with no forwarded proto
  // the adapter reconstructs http://internal.test, not https.
  const mismatch = await channel.handleWebhook(
    {
      Host: 'internal.test',
      'X-Twilio-Signature': twilioSignature('https://internal.test/channels/sms/webhook', form, 'auth-token'),
    },
    encoder.encode(form),
  )
  expect(mismatch).toEqual({ status: 401, body: 'unauthorized' })
})

test('WhatsApp enforces X-Hub-Signature-256 when an appSecret is configured', async () => {
  const payload = JSON.stringify({ entry: [{ changes: [{ value: { messages: [
    { from: '15550001', id: 'wamid-signed', type: 'text', text: { body: 'signed hello' } },
  ] } }] }] })
  const body = encoder.encode(payload)
  const channel = new WhatsAppChannel({ accessToken: 'wa-token', phoneNumberId: 'phone-id', appSecret: 'meta-secret' })
  const received: ChannelMessage[] = []
  await channel.start(async message => { received.push(message) })

  // Missing header and wrong digest are rejected before parsing.
  expect(await channel.handleWebhook({}, body)).toEqual({ status: 401, body: 'unauthorized' })
  expect(await channel.handleWebhook({ 'X-Hub-Signature-256': 'sha256=' + '0'.repeat(64) }, body))
    .toEqual({ status: 401, body: 'unauthorized' })
  expect(received).toEqual([])

  // A correctly signed delivery passes and parses.
  expect(await channel.handleWebhook({ 'X-Hub-Signature-256': hubSignature(body, 'meta-secret') }, body))
    .toEqual({ status: 200, body: 'ok' })
  expect(received.map(message => message.text)).toEqual(['signed hello'])
})
