// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Injectable native-fetch boundary for messaging providers. */
export type ChannelFetch = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>

export interface JsonRequestOptions {
  readonly body?: unknown
  readonly fetchImplementation?: ChannelFetch
  readonly headers?: Readonly<Record<string, string>>
  readonly signal?: AbortSignal
}

export interface FormRequestOptions {
  readonly body: Readonly<Record<string, string | number | boolean | undefined>>
  readonly fetchImplementation?: ChannelFetch
  readonly headers?: Readonly<Record<string, string>>
  readonly signal?: AbortSignal
}

/** A failed provider request with a body-safe diagnostic message. */
export class ChannelHttpError extends Error {
  readonly status: number

  constructor(status: number, message = `channel HTTP request failed (${status})`) {
    super(message)
    this.name = new.target.name
    this.status = status
  }
}

/**
 * POST a JSON document through native fetch and decode a JSON success body.
 *
 * Response bodies are deliberately excluded from failure messages because
 * provider error payloads can echo user content or credentials.
 */
export async function postJson<T = unknown>(
  url: string,
  options: JsonRequestOptions = {},
): Promise<T | undefined> {
  return requestJson('POST', url, options)
}

/** PUT a JSON document through native fetch and decode a JSON success body. */
export async function putJson<T = unknown>(
  url: string,
  options: JsonRequestOptions = {},
): Promise<T | undefined> {
  return requestJson('PUT', url, options)
}

/** Post an URL-encoded form body for APIs such as Twilio's Messages endpoint. */
export async function postForm<T = unknown>(
  url: string,
  options: FormRequestOptions,
): Promise<T | undefined> {
  validateHttpUrl(url)
  const body = new URLSearchParams()
  for (const [name, value] of Object.entries(options.body)) {
    if (value !== undefined) {
      body.set(name, String(value))
    }
  }
  const response = await (options.fetchImplementation ?? fetch)(url, {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
      ...options.headers,
    },
    body,
    ...(options.signal ? { signal: options.signal } : {}),
  })
  return decodeResponse<T>(response)
}

async function requestJson<T>(
  method: 'POST' | 'PUT',
  url: string,
  options: JsonRequestOptions,
): Promise<T | undefined> {
  validateHttpUrl(url)
  const response = await (options.fetchImplementation ?? fetch)(url, {
    method,
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...(options.body === undefined ? {} : { body: JSON.stringify(options.body) }),
    ...(options.signal ? { signal: options.signal } : {}),
  })
  return decodeResponse<T>(response)
}

async function decodeResponse<T>(response: Response): Promise<T | undefined> {
  if (!response.ok) {
    throw new ChannelHttpError(response.status)
  }
  const text = await response.text()
  if (!text) {
    return undefined
  }
  try {
    return JSON.parse(text) as T
  } catch {
    return undefined
  }
}

/** Join a provider-relative path to an HTTP(S) base URL. */
export function providerUrl(baseUrl: string, path: string): string {
  validateHttpUrl(baseUrl)
  const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`
  // Prefix a relative marker: a provider path such as `bot123:token/sendMessage`
  // would otherwise be interpreted as a custom URL scheme because of its colon.
  return new URL(`./${path.replace(/^\/+/, '')}`, normalizedBase).toString()
}

function validateHttpUrl(url: string): void {
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch (error) {
    throw new TypeError('channel URL must be absolute', { cause: error })
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new TypeError('channel URL must use HTTP or HTTPS')
  }
}
