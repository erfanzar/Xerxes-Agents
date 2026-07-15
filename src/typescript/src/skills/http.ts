// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Fetch signature used by bundled skill modules. Hosts can inject it in tests or constrained runtimes. */
export type SkillFetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

/** JSON-like object returned from a third-party skill service. */
export type SkillJsonObject = Readonly<Record<string, unknown>>

/** Failure raised when a bundled skill cannot complete an HTTP or response-decoding operation. */
export class SkillHttpError extends Error {
  readonly url: string

  constructor(message: string, url: string, options: { readonly cause?: unknown } = {}) {
    super(message, options)
    this.name = 'SkillHttpError'
    this.url = url
  }
}

/** Return a caller-provided non-blank string without silently coercing it. */
export function requireSkillText(value: string, name: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError(`${name} must be a non-empty string`)
  }
  return value.trim()
}

/** Narrow unknown JSON data to a record. */
export function skillJsonObject(value: unknown, context: string): SkillJsonObject {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${context} must be a JSON object`)
  }
  return value as SkillJsonObject
}

/** Narrow unknown JSON data to an array. */
export function skillJsonArray(value: unknown, context: string): readonly unknown[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${context} must be a JSON array`)
  }
  return value
}

/** Fetch a response body as text and include HTTP context in failures. */
export async function skillFetchText(
  fetchImplementation: SkillFetch,
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<string> {
  const url = typeof input === 'string' ? input : input.toString()
  let response: Response
  try {
    response = await fetchImplementation(input, init)
  } catch (error) {
    throw new SkillHttpError(`request failed: ${url}`, url, { cause: error })
  }
  if (!response.ok) {
    let body = ''
    try {
      body = await response.text()
    } catch {
      // A status code is still actionable when a proxy closed the response body.
    }
    const statusText = response.statusText ? ` ${response.statusText}` : ''
    throw new SkillHttpError(
      `request failed (${response.status}${statusText}): ${body.slice(0, 4_096)}`,
      url,
    )
  }
  return response.text()
}

/** Fetch a response body as JSON while keeping transport and payload errors distinct. */
export async function skillFetchJson(
  fetchImplementation: SkillFetch,
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<unknown> {
  const url = typeof input === 'string' ? input : input.toString()
  const text = await skillFetchText(fetchImplementation, input, init)
  try {
    return JSON.parse(text) as unknown
  } catch (error) {
    throw new SkillHttpError(`response was not valid JSON: ${url}`, url, { cause: error })
  }
}
