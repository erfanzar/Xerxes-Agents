// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Local authority annotation. Symbols cannot arrive in JSON or leak into a
 * provider body, transcript, or transport frame. Preserve it when mapping a
 * native request, and check the final provider payload before sending it. */
export const OUTPUT_TOKEN_LIMIT = Symbol('xerxes.outputTokenLimit')
export interface OutputTokenBound { readonly [OUTPUT_TOKEN_LIMIT]?: number }
export class OutputTokenLimitError extends Error {
  constructor() { super('Provider output settings exceed the authorized limit. Lower reasoning or explicitly authorize a larger limit.'); this.name = 'OutputTokenLimitError' }
}

export function assertOutputTokenLimit(request: OutputTokenBound, wireLimit: unknown): void {
  const limit = request[OUTPUT_TOKEN_LIMIT]
  if (limit === undefined) return
  if (!Number.isSafeInteger(limit) || limit < 1 || typeof wireLimit !== 'number' ||
    !Number.isSafeInteger(wireLimit) || wireLimit < 1 || wireLimit > limit) throw new OutputTokenLimitError()
}
