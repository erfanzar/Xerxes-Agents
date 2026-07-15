// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ToolRegistry } from '../executors/toolRegistry.js'
import type { SearchHit, SearchOptions, SessionStore } from '../session/index.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { optionalInteger, optionalString, requiredString } from './inputs.js'

export interface HistorySearcher {
  search(query: string, options?: SearchOptions): readonly SearchHit[]
}

export interface SearchHistoryToolOptions {
  readonly defaultLimit?: number
  readonly index?: HistorySearcher
  readonly store?: SessionStore
}

/** Search structured history from either a durable session store or a supplied index. */
export class SearchHistoryTool {
  readonly defaultLimit: number
  private readonly searcher: HistorySearcher

  constructor(options: SearchHistoryToolOptions) {
    if (!options.index && !options.store) {
      throw new Error('SearchHistoryTool requires a store or an index')
    }
    this.searcher = options.index ?? options.store as SessionStore
    this.defaultLimit = positiveLimit(options.defaultLimit ?? 5)
  }

  search(query: string, options: { readonly agentId?: string; readonly limit?: number; readonly sessionId?: string } = {}): JsonObject {
    const cleaned = query.trim()
    if (!cleaned) return { query, count: 0, hits: [] }
    const k = positiveLimit(options.limit ?? this.defaultLimit)
    const hits = this.searcher.search(cleaned, {
      k,
      ...(options.agentId === undefined ? {} : { agentId: options.agentId }),
      ...(options.sessionId === undefined ? {} : { sessionId: options.sessionId }),
    })
    return {
      query,
      count: hits.length,
      hits: hits.map(hit => historyHitRecord(hit)),
    }
  }
}

export const SEARCH_HISTORY_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'search_history',
    description: 'Search indexed historical agent prompts and responses through a configured session store.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        query: { type: 'string' },
        limit: { type: 'integer', minimum: 1, default: 5 },
        agent_id: { type: 'string' },
        session_id: { type: 'string' },
      },
      required: ['query'],
    },
  },
}

export function registerSearchHistoryTool(registry: ToolRegistry, tool: SearchHistoryTool): void {
  registry.register(SEARCH_HISTORY_DEFINITION, inputs => {
    const agentId = optionalString(inputs, 'agent_id')
    const sessionId = optionalString(inputs, 'session_id')
    return tool.search(requiredString(inputs, 'query'), {
      limit: optionalInteger(inputs, 'limit', tool.defaultLimit),
      ...(agentId === undefined ? {} : { agentId }),
      ...(sessionId === undefined ? {} : { sessionId }),
    })
  })
}

function historyHitRecord(hit: SearchHit): JsonObject {
  return {
    session_id: hit.sessionId,
    turn_id: hit.turnId,
    agent_id: hit.agentId,
    prompt: hit.prompt,
    response: hit.response,
    score: Math.round(hit.score * 10_000) / 10_000,
    timestamp: hit.timestamp,
  }
}

function positiveLimit(value: number): number {
  if (!Number.isInteger(value) || value < 1) throw new Error('limit must be a positive integer')
  return value
}
