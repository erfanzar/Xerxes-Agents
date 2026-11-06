// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * User-facing wording for provider transport failures. Bun's fetch errors are
 * written for the developer holding the `fetch()` call ("pass `verbose: true`
 * in the second argument to fetch()"); the person in the terminal needs what
 * happened and what to check. Hooks, audit and logs keep the raw message.
 */
import { errorMessage } from '../executors/toolRegistry.js'

const BUN_FETCH_HINT = /\s*For more information, pass `?verbose: true`? in the second argument to fetch\(\)\.?/gi

const KNOWN: ReadonlyArray<readonly [RegExp, string]> = [
  [/socket connection was closed unexpectedly|other side closed|ECONNRESET|connection reset by peer/i,
    'The model server closed the connection mid-reply. It may have crashed or run out of memory'],
  [/Unable to connect\. Is the computer able to access the url\?|ECONNREFUSED|connection refused/i,
    'Could not connect to the model server. It may not be running'],
]

export function providerErrorText(error: unknown): string {
  const raw = errorMessage(error)
  for (const [pattern, text] of KNOWN) {
    if (pattern.test(raw)) return text
  }
  return raw.replace(BUN_FETCH_HINT, '').trim() || raw
}
