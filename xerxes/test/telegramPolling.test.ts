// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  TelegramChannel,
  TelegramPollingLoop,
  type TelegramUpdatesOptions,
  type WebhookHeaders,
  type WebhookResponse,
} from '../src/channels/index.js'

class PollingChannel {
  readonly calls: TelegramUpdatesOptions[] = []
  readonly delivered: unknown[] = []

  async getUpdates(options: TelegramUpdatesOptions = {}): Promise<Readonly<Record<string, unknown>>> {
    this.calls.push(options)
    if (this.calls.length === 1) {
      return { result: [{ update_id: 41, message: { text: 'from poll' } }] }
    }
    return new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new Error('request aborted')), { once: true })
    })
  }

  async handleWebhook(_headers: WebhookHeaders, body: Uint8Array): Promise<WebhookResponse> {
    this.delivered.push(JSON.parse(new TextDecoder().decode(body)))
    return { status: 200, body: 'ok' }
  }
}

test('telegram polling replays Bot API updates through the adapter webhook and advances offset', async () => {
  const channel = new PollingChannel()
  const loop = new TelegramPollingLoop({ channel, timeout: 0, retryDelay: 0 })

  await eventually(() => channel.calls.length >= 2)
  expect(channel.delivered).toEqual([{ update_id: 41, message: { text: 'from poll' } }])
  expect(channel.calls[0]).toMatchObject({ timeout: 0 })
  expect(channel.calls[1]).toMatchObject({ offset: 42, timeout: 0 })

  await loop.stop()
  expect(loop.stopped).toBe(true)
})

test('telegram polling delivers updates even when a webhook secret token is configured', async () => {
  const received: string[] = []
  let polls = 0
  const telegram = new TelegramChannel({
    token: 'token',
    webhookSecretToken: 'configured-secret',
    fetchImplementation: async (_input, init) => {
      polls += 1
      if (polls > 1) {
        await new Promise<void>(resolve => {
          init?.signal?.addEventListener('abort', () => resolve(), { once: true })
        })
        return new Response(JSON.stringify({ ok: true, result: [] }))
      }
      return new Response(JSON.stringify({
        ok: true,
        result: [{
          update_id: 41,
          message: { message_id: 9, text: 'polled hello', from: { id: 7 }, chat: { id: 7, type: 'private' } },
        }],
      }))
    },
  })
  await telegram.start(async message => { received.push(message.text) })

  // The public webhook path still rejects a missing secret token...
  expect(await telegram.handleWebhook({}, new TextEncoder().encode('{}'))).toEqual({ status: 401, body: 'unauthorized' })

  // ...while the bot-token-authenticated polling path ingests the same adapter parser.
  const loop = new TelegramPollingLoop({ channel: telegram, timeout: 0, retryDelay: 0 })
  await eventually(() => received.length === 1 && polls >= 2)
  expect(received).toEqual(['polled hello'])
  await loop.stop()
})

test('telegram polling acknowledges the offset only after successful delivery', async () => {
  const channel = new FlakyPollingChannel()
  const errors: unknown[] = []
  const loop = new TelegramPollingLoop({
    channel,
    timeout: 0,
    retryDelay: 1,
    onError: error => { errors.push(error) },
  })

  await eventually(() => channel.calls.length >= 3)
  // The first delivery failed, so the second poll must not have acknowledged update 41.
  expect(channel.calls[0]?.offset).toBeUndefined()
  expect(channel.calls[1]?.offset).toBeUndefined()
  expect(channel.calls[2]?.offset).toBe(42)
  expect(channel.attempts).toBe(2)
  expect(channel.delivered).toEqual([41])
  expect(errors).toHaveLength(1)

  await loop.stop()
})

class FlakyPollingChannel {
  readonly calls: TelegramUpdatesOptions[] = []
  readonly delivered: number[] = []
  attempts = 0

  async getUpdates(options: TelegramUpdatesOptions = {}): Promise<Readonly<Record<string, unknown>>> {
    this.calls.push(options)
    if (this.calls.length <= 2) {
      return { result: [{ update_id: 41, message: { text: 'retry me' } }] }
    }
    return new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new Error('request aborted')), { once: true })
    })
  }

  async handleWebhook(_headers: WebhookHeaders, body: Uint8Array): Promise<WebhookResponse> {
    this.attempts += 1
    if (this.attempts === 1) {
      return { status: 500, body: 'handler failed' }
    }
    this.delivered.push((JSON.parse(new TextDecoder().decode(body)) as { update_id: number }).update_id)
    return { status: 200, body: 'ok' }
  }
}

test('telegram long-poll requests forward AbortSignal through the HTTP adapter', async () => {
  let seenSignal: AbortSignal | null = null
  const telegram = new TelegramChannel({
    token: 'token',
    fetchImplementation: async (_input, init) => {
      seenSignal = init?.signal ?? null
      return new Response(JSON.stringify({ ok: true, result: [] }))
    },
  })
  const controller = new AbortController()

  await telegram.getUpdates({ signal: controller.signal })

  expect(seenSignal === controller.signal).toBe(true)
})

async function eventually(predicate: () => boolean): Promise<void> {
  for (let index = 0; index < 100; index += 1) {
    if (predicate()) return
    await Bun.sleep(1)
  }
  throw new Error('condition was not reached')
}
