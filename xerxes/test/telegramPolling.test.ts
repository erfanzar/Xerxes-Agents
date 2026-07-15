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
