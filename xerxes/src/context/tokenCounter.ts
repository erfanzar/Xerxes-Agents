// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type TokenCountInput = string | readonly Record<string, unknown>[]

/** Provider-aware token estimator with a deterministic dependency-free fallback. */
export class ProviderTokenCounter {
  static countTokensForProvider(input: TokenCountInput, provider?: string, model?: string): number {
    const text = typeof input === 'string' ? input : this.messagesToText(input)
    const resolved = provider ?? (model ? this.detectProvider(model) : undefined)
    const fallback = estimateTokens(text)
    return resolved === 'google' ? Math.ceil(fallback * 1.1) : fallback
  }

  static detectProvider(model: string): string | undefined {
    const normalized = model.toLowerCase()
    if (normalized.includes('gpt') || normalized.includes('o1')) return 'openai'
    if (normalized.includes('claude')) return 'anthropic'
    if (normalized.includes('gemini') || normalized.includes('palm')) return 'google'
    if (normalized.includes('llama')) return 'meta'
    if (normalized.includes('mistral') || normalized.includes('mixtral')) return 'mistral'
    return undefined
  }

  static messagesToText(messages: readonly Record<string, unknown>[]): string {
    return messages.map(message => {
      const role = typeof message.role === 'string' ? message.role : ''
      const parts = [`${role}: ${contentToText(message.content)}`]
      // tool_calls arguments are often the largest payload in a window; count every
      // additional token-bearing field instead of silently treating it as free.
      for (const [key, value] of Object.entries(message)) {
        if (key === 'role' || key === 'content' || value === undefined || value === null) {
          continue
        }
        const serialized = contentToText(value)
        if (serialized) {
          parts.push(`${key}=${serialized}`)
        }
      }
      return parts.join(' ')
    }).join('\n')
  }
}

export interface SmartTokenCounterOptions {
  readonly model?: string
  readonly provider?: string
}

export class SmartTokenCounter {
  readonly model: string | undefined
  readonly provider: string | undefined

  constructor(options: SmartTokenCounterOptions = {}) {
    this.model = options.model
    this.provider = options.provider ?? (options.model ? ProviderTokenCounter.detectProvider(options.model) : undefined)
  }

  countRemainingCapacity(input: TokenCountInput, maxTokens: number): number {
    return Math.max(0, maxTokens - this.countTokens(input))
  }

  countTokens(input: TokenCountInput): number {
    return ProviderTokenCounter.countTokensForProvider(input, this.provider, this.model)
  }

  estimateCompressionRatio(original: string, compressed: string): number {
    const originalTokens = this.countTokens(original)
    return originalTokens === 0 ? 0 : 1 - this.countTokens(compressed) / originalTokens
  }
}

function contentToText(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(contentToText).join(' ')
  if (value === undefined || value === null) return ''
  return JSON.stringify(value)
}

function estimateTokens(text: string): number {
  if (!text) return 0
  // Words and numbers count individually while a run of punctuation/symbols counts once
  // per run. One token per non-space character over-counted punctuation-dense content
  // (code, JSON, diffs) by roughly 2x versus BPE merge behavior.
  const lexical = text.match(/\p{L}+[\p{L}\p{N}_-]*|\p{N}+|[^\s\p{L}\p{N}]+/gu)?.length ?? 0
  return Math.max(1, lexical, Math.ceil(text.length / 4))
}
