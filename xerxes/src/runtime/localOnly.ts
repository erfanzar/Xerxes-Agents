// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Fully local mode: Xerxes reaches only this machine.
 *
 * One setting, `runtime.local_only` (or `XERXES_LOCAL_ONLY`), turns it on for
 * the daemon, one-shot runs and ACP. With it on:
 * - `fetch` and `WebSocket` reach only loopback (localhost, 127.0.0.0/8, ::1)
 *   and unix sockets. Anything else is refused with a ConfigurationError that
 *   names the address, so a refused call is visible, not silent.
 * - Model clients are built only for a model served on this machine, so a
 *   cloud profile, a sub-agent on a cloud model, the advisor and Claude Code
 *   (which runs its own CLI) are refused before anything starts.
 * - The features that would otherwise try the network are off: the models.dev
 *   catalogue, Claude Code updates and installs, the web tools and image
 *   generation, and git fetch, pull, push and plugin clones.
 *
 * The guard is installed before any client is built: clients keep the `fetch`
 * they were created with.
 */
import { ConfigurationError, ValidationError } from '../core/errors.js'
import { isLoopbackBaseUrl } from '../llms/providerRegistry.js'

export const LOCAL_ONLY_ENV = 'XERXES_LOCAL_ONLY'

const TRUE_WORDS = new Set(['1', 'true', 'yes', 'on'])
const FALSE_WORDS = new Set(['0', 'false', 'no', 'off'])

/** `XERXES_LOCAL_ONLY` wins over `runtime.local_only`; the default is off. */
export function parseLocalOnlyConfig(
  runtime: Readonly<Record<string, unknown>>,
  env: Readonly<Record<string, string | undefined>>,
): boolean {
  const fromEnv = env[LOCAL_ONLY_ENV]?.trim().toLowerCase()
  if (fromEnv) {
    if (TRUE_WORDS.has(fromEnv)) return true
    if (FALSE_WORDS.has(fromEnv)) return false
    throw new ValidationError(LOCAL_ONLY_ENV, 'must be one of 1, true, yes, on, 0, false, no, off', fromEnv)
  }
  const fromFile = runtime.local_only
  if (fromFile === undefined || fromFile === null) return false
  if (typeof fromFile !== 'boolean') throw new ValidationError('local_only', 'must be true or false', fromFile)
  return fromFile
}

let active = false

/** Whether fully local mode is on in this process. */
export function localOnlyActive(): boolean {
  return active
}

/** The error every refusal raises: fatal, so a turn reports it instead of retrying. */
export function localOnlyRefusal(what: string): ConfigurationError {
  return new ConfigurationError(
    'local_only',
    `fully local mode refuses ${what}; only this machine is reachable. Turn it off with runtime.local_only: false or ${LOCAL_ONLY_ENV}=0.`,
  )
}

/** Throw when fully local mode is on; `what` names the refused action. */
export function assertNotLocalOnly(what: string): void {
  if (active) throw localOnlyRefusal(what)
}

/** Whether `url` is on this machine: loopback http(s) or ws(s), or a unix socket. */
export function isLocalUrl(url: string | URL): boolean {
  return isLoopbackBaseUrl(String(url))
}

function requestUrl(input: unknown): string {
  if (typeof input === 'string') return input
  if (input instanceof URL) return input.href
  if (input instanceof Request) return input.url
  return String(input)
}

/** The existing off switches local mode sets, so child processes inherit them too. */
const LOCAL_ONLY_SWITCHES: Readonly<Record<string, string>> = {
  [LOCAL_ONLY_ENV]: '1',
  XERXES_MODELS_DEV: '0',
  XERXES_CLAUDE_CODE_AUTOUPDATE: '0',
  XERXES_AUTO_INSTALL_CLAUDE_CODE: '0',
}

/**
 * Turn fully local mode on for this process: guard `fetch` and `WebSocket`,
 * and set the off switches for models.dev and Claude Code updates. Returns a
 * function that puts everything back (for tests and embedders). Calling it
 * again while on changes nothing and returns a no-op.
 */
export function enableLocalOnly(env: Record<string, string | undefined> = process.env): () => void {
  if (active) return () => undefined
  active = true
  const previousEnv = Object.fromEntries(Object.keys(LOCAL_ONLY_SWITCHES).map(key => [key, env[key]]))
  Object.assign(env, LOCAL_ONLY_SWITCHES)

  const originalFetch = globalThis.fetch
  const guardedFetch = (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]): Promise<Response> => {
    const url = requestUrl(input)
    if (!isLocalUrl(url)) return Promise.reject(localOnlyRefusal(`a request to ${safeOrigin(url)}`))
    return originalFetch(input, init)
  }
  globalThis.fetch = Object.assign(guardedFetch, originalFetch) as typeof fetch

  const OriginalWebSocket = globalThis.WebSocket
  if (OriginalWebSocket !== undefined) {
    class LocalOnlyWebSocket extends OriginalWebSocket {
      constructor(url: string | URL, protocols?: string | string[]) {
        if (!isLocalUrl(url)) throw localOnlyRefusal(`a WebSocket to ${safeOrigin(String(url))}`)
        super(url, protocols)
      }
    }
    globalThis.WebSocket = LocalOnlyWebSocket as typeof WebSocket
  }

  return () => {
    globalThis.fetch = originalFetch
    if (OriginalWebSocket !== undefined) globalThis.WebSocket = OriginalWebSocket
    for (const [key, value] of Object.entries(previousEnv)) {
      if (value === undefined) delete env[key]
      else env[key] = value
    }
    active = false
  }
}

/** The origin only: a refused URL's path and query may carry tokens. */
function safeOrigin(url: string): string {
  try {
    return new URL(url).origin
  } catch {
    return 'an unparseable address'
  }
}
