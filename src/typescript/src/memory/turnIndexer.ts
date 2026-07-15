// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Memory } from './base.js'

export interface TurnIndexInput {
  readonly agentId?: string
  readonly response?: unknown
}

export interface TurnIndexerOptions {
  readonly importance?: number
  readonly memoryType?: string
  readonly minChars?: number
}

/** Build a resilient post-turn hook that indexes useful assistant output into a memory tier. */
export function makeTurnIndexerHook(memory: Memory, options: TurnIndexerOptions = {}): (input: TurnIndexInput) => void {
  const minimum = options.minChars ?? 32
  const importance = options.importance ?? 0.5
  const memoryType = options.memoryType ?? 'turn'
  return input => {
    const content = coerceText(input.response).trim()
    if (content.length < minimum) return
    try {
      memory.save(content, { source: 'turn_indexer' }, {
        ...(input.agentId ? { agentId: input.agentId } : {}),
        importance,
        memoryType,
      })
    } catch {
      // A best-effort post-turn index must not change the completed turn's outcome.
    }
  }
}

/** Build a tolerant `(agentId, count) -> content[]` context provider for the agent loop. */
export function makeMemoryProvider(memory: Memory, useSemantic = true): (agentId: string | undefined, count: number) => string[] {
  return (agentId, count) => {
    try {
      return memory.search(agentId ?? 'context', count, undefined, { useSemantic }).slice(0, count).map(item => item.content).filter(Boolean)
    } catch {
      return []
    }
  }
}

export function coerceText(response: unknown): string {
  if (response === undefined || response === null) return ''
  if (typeof response === 'string') return response
  if (Array.isArray(response)) return response.map(coerceText).filter(Boolean).join('\n')
  if (typeof response === 'object') {
    const value = response as Record<string, unknown>
    if (typeof value.content === 'string') return value.content
    if (typeof value.text === 'string') return value.text
    if (Array.isArray(value.content)) return value.content.map(coerceText).filter(Boolean).join('\n')
  }
  return String(response)
}
