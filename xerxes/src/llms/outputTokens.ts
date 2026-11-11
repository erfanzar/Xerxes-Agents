// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'

export const DEFAULT_MAX_OUTPUT_TOKENS = 32_768

/** Native OpenAI and local compatible routes; not the Codex subscription backend. */
export function isOpenAiOutputTokenProvider(provider: string): boolean {
  return provider === 'openai' || provider === 'custom' || provider === 'lmstudio' || provider === 'ollama'
}

/** -1 removes the caller cap; known model and context limits still bound a request. */
export function resolveMaxOutputTokens(requested: number | undefined, modelLimit?: number): number {
  if (requested !== undefined && requested !== -1 && (!Number.isSafeInteger(requested) || requested <= 0)) {
    throw new ValidationError('max_tokens', 'must be -1 or a positive safe integer', requested)
  }
  const resolved = requested ?? DEFAULT_MAX_OUTPUT_TOKENS
  if (modelLimit === undefined || !Number.isSafeInteger(modelLimit) || modelLimit <= 0) return resolved
  return resolved === -1 ? modelLimit : Math.min(resolved, modelLimit)
}
