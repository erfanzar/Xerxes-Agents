// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { SlackTransportError } from './types.js'
import { record } from './wire.js'

export type SlackFetch = (url: string, init: RequestInit) => Promise<Response>
/** The invented authorization code used only to check a client ID and secret. */
export const SLACK_CLIENT_CHECK_CODE = 'xerxes-credential-check'
/** apps.manifest.create errors that mean the configuration token itself is unusable. */
const TOKEN_ERRORS = new Set(['invalid_auth', 'not_authed', 'token_expired', 'token_revoked', 'not_allowed_token_type', 'missing_scope', 'account_inactive'])
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024
const methods = {
  'oauth.v2.access': 'POST', 'auth.test': 'POST', 'users.info': 'GET', 'users.lookupByEmail': 'GET',
  'conversations.open': 'POST', 'conversations.info': 'GET', 'conversations.history': 'GET', 'chat.postMessage': 'POST',
} as const

type SlackMethod = keyof typeof methods

function abortError(signal: AbortSignal): SlackTransportError {
  const reason: unknown = signal.reason
  if (reason instanceof SlackTransportError) return reason
  if (typeof reason === 'object' && reason !== null && 'name' in reason && reason.name === 'TimeoutError') {
    return new SlackTransportError('timeout')
  }
  return new SlackTransportError('aborted')
}

export function checkAbort(signal?: AbortSignal): void {
  if (signal?.aborted) throw new SlackTransportError('aborted')
}

export async function parseSlackJson(response: Response, signal: AbortSignal | null | undefined, maxBytes = MAX_RESPONSE_BYTES): Promise<Record<string, unknown>> {
  if (signal?.aborted) {
    void response.body?.cancel().catch(() => {})
    throw abortError(signal)
  }
  if (!response.body) throw new SlackTransportError('invalid_response')
  const reader = response.body.getReader()
  const cancel = (): void => { void reader.cancel().catch(() => {}) }
  let onAbort: (() => void) | undefined
  const aborted = signal
    ? new Promise<never>((_, reject) => {
        onAbort = (): void => {
          cancel()
          reject(abortError(signal))
        }
        signal.addEventListener('abort', onAbort, { once: true })
      })
    : undefined
  aborted?.catch(() => {})
  const decoder = new TextDecoder('utf-8', { fatal: true })
  let bytes = 0
  let text = ''
  try {
    while (true) {
      const next = aborted ? await Promise.race([reader.read(), aborted]) : await reader.read()
      if (signal?.aborted) throw abortError(signal)
      if (next.done) break
      bytes += next.value.byteLength
      if (bytes > maxBytes) throw new SlackTransportError('invalid_response')
      text += decoder.decode(next.value, { stream: true })
    }
    if (signal?.aborted) throw abortError(signal)
    const parsed = record(JSON.parse(text + decoder.decode()))
    if (!parsed) throw new SlackTransportError('invalid_response')
    return parsed
  } catch (error) {
    cancel()
    if (signal?.aborted) throw abortError(signal)
    if (error instanceof SlackTransportError) throw error
    throw new SlackTransportError('invalid_response')
  } finally {
    if (onAbort) signal?.removeEventListener('abort', onAbort)
    try { reader.releaseLock() } catch { /* reader still cancelling */ }
  }
}

export class SlackHttp {
  constructor(private readonly fetchPort: SlackFetch, private readonly timeoutMs: number) {
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 60_000) throw new SlackTransportError('invalid_request')
  }

  /**
   * Check a Slack app's client ID and secret without a real authorization code. Slack answers an
   * invented code with `invalid_code` when the credentials are right and names a wrong ID or secret.
   */
  async checkClient(clientId: string, clientSecret: string, redirectUri: string, signal?: AbortSignal): Promise<'ok' | 'bad_client'> {
    checkAbort(signal)
    const deadline = AbortSignal.timeout(this.timeoutMs)
    const combined = AbortSignal.any([deadline, ...(signal === undefined ? [] : [signal])])
    let response: Response
    try {
      response = await this.fetchPort('https://slack.com/api/oauth.v2.access', {
        method: 'POST', redirect: 'error', signal: combined,
        headers: new Headers({ authorization: `Basic ${Buffer.from(`${clientId}:${clientSecret}`).toString('base64')}`,
          accept: 'application/json', 'content-type': 'application/x-www-form-urlencoded' }),
        body: new URLSearchParams({ code: SLACK_CLIENT_CHECK_CODE, redirect_uri: redirectUri }).toString(),
      })
    } catch {
      checkAbort(signal)
      throw new SlackTransportError(deadline.aborted ? 'timeout' : 'network_error')
    }
    if (response.redirected || !response.ok) {
      void response.body?.cancel().catch(() => {})
      throw new SlackTransportError(response.status === 429 ? 'rate_limited' : 'slack_rejected')
    }
    let data: Record<string, unknown>
    try {
      data = await parseSlackJson(response, combined)
    } catch (error) {
      checkAbort(signal)
      if (deadline.aborted) throw new SlackTransportError('timeout')
      throw error
    }
    if (data.ok === true) return 'ok'
    if (data.error === 'invalid_client_id' || data.error === 'bad_client_secret' || data.error === 'invalid_client') return 'bad_client'
    if (data.error === 'ratelimited') throw new SlackTransportError('rate_limited')
    return 'ok'
  }

  /**
   * Create the owner's Slack app from a manifest (apps.manifest.create) with a one-time app
   * configuration token. Returns the new app's credentials as Slack reported them (unvalidated).
   */
  async createAppFromManifest(configToken: string, manifest: string, signal?: AbortSignal): Promise<
    { readonly kind: 'created'; readonly clientId: unknown; readonly clientSecret: unknown } | { readonly kind: 'bad_token' | 'refused' }> {
    checkAbort(signal)
    const deadline = AbortSignal.timeout(this.timeoutMs)
    const combined = AbortSignal.any([deadline, ...(signal === undefined ? [] : [signal])])
    let response: Response
    try {
      response = await this.fetchPort('https://slack.com/api/apps.manifest.create', {
        method: 'POST', redirect: 'error', signal: combined,
        headers: new Headers({ accept: 'application/json', 'content-type': 'application/x-www-form-urlencoded' }),
        body: new URLSearchParams({ token: configToken, manifest }).toString(),
      })
    } catch {
      checkAbort(signal)
      throw new SlackTransportError(deadline.aborted ? 'timeout' : 'network_error')
    }
    if (response.redirected || !response.ok) {
      void response.body?.cancel().catch(() => {})
      throw new SlackTransportError(response.status === 429 ? 'rate_limited' : 'slack_rejected')
    }
    let data: Record<string, unknown>
    try {
      data = await parseSlackJson(response, combined)
    } catch (error) {
      checkAbort(signal)
      if (deadline.aborted) throw new SlackTransportError('timeout')
      throw error
    }
    if (data.ok === true) {
      const credentials = record(data.credentials)
      return { kind: 'created', clientId: credentials?.client_id, clientSecret: credentials?.client_secret }
    }
    if (data.error === 'ratelimited') throw new SlackTransportError('rate_limited')
    if (typeof data.error === 'string' && TOKEN_ERRORS.has(data.error)) return { kind: 'bad_token' }
    return { kind: 'refused' }
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
      const result = await Promise.race([deadline, this.#exchange(method, url.toString(), init)])
      checkAbort(signal)
      return result
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
    const data = await parseSlackJson(response, init.signal)
    if (typeof data.ok !== 'boolean') throw new SlackTransportError('invalid_response')
    if (data.ok === true) return data
    if (data.error === 'ratelimited') throw new SlackTransportError('rate_limited')
    if ((method === 'users.info' && data.error === 'user_not_found') || (method === 'users.lookupByEmail' && data.error === 'users_not_found')) {
      throw new SlackTransportError('not_found')
    }
    throw new SlackTransportError('slack_rejected')
  }
}
