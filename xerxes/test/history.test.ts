// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { SearchHistoryTool, registerSearchHistoryTool } from '../src/tools/history.js'
import type { SearchHit } from '../src/session/search.js'

const HIT: SearchHit = {
  agentId: 'reviewer',
  bm25Score: 0.3,
  metadata: {},
  prompt: 'How was auth implemented?',
  response: 'With an explicit token store.',
  score: 0.876543,
  semanticScore: 0.8,
  sessionId: 'session-1',
  timestamp: '2026-07-13T00:00:00.000Z',
  turnId: 'turn-1',
}

test('history tool maps session hits to the Python-compatible public shape', async () => {
  const searches: unknown[] = []
  const tool = new SearchHistoryTool({
    defaultLimit: 4,
    index: {
      search(query, options) {
        searches.push({ query, options })
        return [HIT]
      },
    },
  })
  expect(tool.search(' auth ', { agentId: 'reviewer' })).toEqual({
    query: ' auth ',
    count: 1,
    hits: [{
      session_id: 'session-1',
      turn_id: 'turn-1',
      agent_id: 'reviewer',
      prompt: 'How was auth implemented?',
      response: 'With an explicit token store.',
      score: 0.8765,
      timestamp: '2026-07-13T00:00:00.000Z',
    }],
  })
  expect(searches).toEqual([{ query: 'auth', options: { k: 4, agentId: 'reviewer' } }])

  const registry = new ToolRegistry()
  registerSearchHistoryTool(registry, tool)
  const output = JSON.parse(await registry.execute({
    id: 'history',
    type: 'function',
    function: { name: 'search_history', arguments: { query: 'auth', limit: 2, session_id: 'session-1' } },
  }, { metadata: {} }))
  expect(output.count).toBe(1)
  expect(searches.at(-1)).toEqual({ query: 'auth', options: { k: 2, sessionId: 'session-1' } })
})

test('history tool requires a search source and handles empty queries without calling it', () => {
  expect(() => new SearchHistoryTool({})).toThrow('requires a store or an index')
  let called = false
  const tool = new SearchHistoryTool({
    index: {
      search() {
        called = true
        return []
      },
    },
  })
  expect(tool.search('   ')).toEqual({ query: '   ', count: 0, hits: [] })
  expect(called).toBeFalse()
})
