// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { SlackTransportError } from './types.js'
import { record } from './wire.js'

export type SlackFetch = (url: string, init: RequestInit) => Promise<Response>
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024
const methods = {
  'oauth.v2.access': 'POST', 'auth.test': 'POST', 'users.info': 'GET', 'users.lookupByEmail': 'GET',
  'conversations.open': 'POST', 'conversations.info': 'GET', 'conversations.history': 'GET', 'chat.postMessage': 'POST',
} as const

type SlackMethod = keyof typeof methods

export function checkAbort(signal?: AbortSignal): void {
  if (signal?.aborted) throw new SlackTransportError('aborted')
}

async function parseBody(response: Response, signal: AbortSignal | null | undefined): Promise<Record<string, unknown>> {
  if (!response.body) throw new SlackTransportError('invalid_response')
  const reader = response.body.getReader()
  const cancel = (): void => { void reader.cancel().catch(() => {}) }
  signal?.addEventListener('abort', cancel, { once: true })
  const decoder = new TextDecoder('utf-8', { fatal: true })
  let bytes = 0
  let text = ''
  try {
    while (true) {
      const next = await reader.read()
      if (next.done) break
      bytes += next.value.byteLength
      if (bytes > MAX_RESPONSE_BYTES) throw new SlackTransportError('invalid_response')
      text += decoder.decode(next.value, { stream: true })
    }
    const parsed = record(JSON.parse(text + decoder.decode()))
    if (!parsed || typeof parsed.ok !== 'boolean') throw new SlackTransportError('invalid_response')
    return parsed
  } catch {
    cancel()
    throw new SlackTransportError('invalid_response')
  } finally {
    signal?.removeEventListener('abort', cancel)
    reader.releaseLock()
  }
}

export class SlackHttp {
  constructor(private readonly fetchPort: SlackFetch, private readonly timeoutMs: number) {
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 60_000) throw new SlackTransportError('invalid_request')
  }

  async request(method: SlackMethod, authorization: string, input: URLSearchParams | Record<string, unknown>, signal?: AbortSignal): Promise<Record<string, unknown>> {
    checkAbort(signal)
    const url = new URL(`https://slack.com/api/${method}`)
    const headers = new Headers({ authorization, accept: 'application/json' })
    const verb = methods[method]
    const aborter = new AbortController()
    const init: RequestInit = { method: verb, headers, redirect: 'error', signal: aborter.signal }
    if (verb === 'GET') url.search = input.toString()
    else if (input instanceof URLSearchParams) {
      headers.set('content-type', 'application/x-www-form-urlencoded')
      init.body = input.toString()
    } else {
      headers.set('content-type', 'application/json')
      init.body = JSON.stringify(input)
    }
    const cancel = (): void => aborter.abort(new SlackTransportError('aborted'))
    signal?.addEventListener('abort', cancel, { once: true })
    const timer = setTimeout(() => aborter.abort(new SlackTransportError('timeout')), this.timeoutMs)
    const deadline = new Promise<never>((_, reject) => {
      aborter.signal.addEventListener('abort', () => reject(aborter.signal.reason), { once: true })
    })
    try {
      const data = await Promise.race([deadline, this.#exchange(method, url.toString(), init)])
      checkAbort(signal)
      return data
    } finally {
      clearTimeout(timer)
      signal?.removeEventListener('abort', cancel)
    }
  }

  async #exchange(method: SlackMethod, url: string, init: RequestInit): Promise<Record<string, unknown>> {
    let response: Response
    try { response = await this.fetchPort(url, init) } catch { throw new SlackTransportError('network_error') }
    if (init.signal?.aborted || response.redirected || !response.ok) {
      void response.body?.cancel().catch(() => {})
      if (init.signal?.aborted) throw init.signal.reason
      if (response.status === 429) {
        const header = response.headers.get('retry-after') ?? ''
        const delay = /^\d{1,8}$/u.test(header) ? Number(header) : undefined
        throw new SlackTransportError('rate_limited', delay)
      }
      throw new SlackTransportError('slack_rejected')
    }
    const data = await parseBody(response, init.signal)
    if (data.ok === true) return data
    if (data.error === 'ratelimited') throw new SlackTransportError('rate_limited')
    if ((method === 'users.info' && data.error === 'user_not_found') || (method === 'users.lookupByEmail' && data.error === 'users_not_found')) {
      throw new SlackTransportError('not_found')
    }
    throw new SlackTransportError('slack_rejected')
  }
}
