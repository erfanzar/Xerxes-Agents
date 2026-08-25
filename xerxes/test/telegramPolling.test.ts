// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { getEventListeners } from 'node:events'

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

test('telegram polling dead-letters a deterministically failing update after its retry budget', async () => {
  const channel = new PoisonedUpdateChannel()
  const errors: unknown[] = []
  const loop = new TelegramPollingLoop({
    channel,
    timeout: 0,
    retryDelay: 0,
    onError: error => { errors.push(error) },
  })

  // Polling must move past the poisoned update and deliver the ones behind it.
  await eventually(() => channel.delivered.includes(43))
  await loop.stop()

  // Default budget: five failed deliveries for update 41, then it is skipped.
  expect(channel.poisonAttempts).toBe(5)
  expect(channel.delivered).toContain(42)
  expect(channel.delivered).toContain(43)
  expect(channel.calls.some(call => call.offset === 43)).toBeTrue()
  const deadLetters = errors.map(String).filter(text => text.includes('dead-lettered'))
  expect(deadLetters).toHaveLength(1)
  expect(deadLetters[0]).toContain('update_id 41')
})

class PoisonedUpdateChannel {
  readonly calls: TelegramUpdatesOptions[] = []
  readonly delivered: number[] = []
  poisonAttempts = 0

  async getUpdates(options: TelegramUpdatesOptions = {}): Promise<Readonly<Record<string, unknown>>> {
    this.calls.push(options)
    if ((options.offset ?? 0) >= 44) {
      // One delivery past the poisoned window, then behave like a real
      // long-poll: block until aborted instead of re-serving update 43.
      return new Promise((_resolve, reject) => {
        options.signal?.addEventListener('abort', () => reject(new Error('request aborted')), { once: true })
      })
    }
    if ((options.offset ?? 0) >= 43) {
      return { result: [{ update_id: 43, message: { text: 'after the poison' } }] }
    }
    return {
      result: [
        { update_id: 41, message: { text: 'deterministically failing' } },
        { update_id: 42, message: { text: 'queued behind the poison' } },
      ],
    }
  }

  async handleWebhook(_headers: WebhookHeaders, body: Uint8Array): Promise<WebhookResponse> {
    const updateId = (JSON.parse(new TextDecoder().decode(body)) as { update_id: number }).update_id
    if (updateId === 41) {
      this.poisonAttempts += 1
      return { status: 500, body: 'handler always fails' }
    }
    this.delivered.push(updateId)
    return { status: 200, body: 'ok' }
  }
}

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

test('telegram polling retry sleep removes its abort listener when the timer wins', async () => {
  const retryEntered = Promise.withResolvers<void>()
  const releaseRetry = Promise.withResolvers<void>()
  let signal: AbortSignal | undefined
  let calls = 0
  const channel = {
    async getUpdates(options: TelegramUpdatesOptions = {}): Promise<Readonly<Record<string, unknown>>> {
      calls += 1
      signal = options.signal
      if (calls === 1) throw new Error('retry')
      retryEntered.resolve()
      await releaseRetry.promise
      throw new Error('released')
    },
    async handleWebhook(): Promise<WebhookResponse> {
      return { status: 200, body: 'ok' }
    },
  }
  const loop = new TelegramPollingLoop({ channel, retryDelay: 1, timeout: 0 })

  await retryEntered.promise
  expect(signal).toBeDefined()
  expect(getEventListeners(signal!, 'abort')).toHaveLength(0)

  const stopping = loop.stop()
  releaseRetry.resolve()
  await stopping
})

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
