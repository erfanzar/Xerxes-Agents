// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, it } from 'vitest'

import {
  clearSpawnHistory,
  getSpawnHistory,
  pushSnapshot,
  reconcileSpawnHistorySubagent,
  spawnHistoryForSession
} from '../app/spawnHistoryStore.js'
import type { Msg, SubagentProgress } from '../types.js'

import { reconcileArchivedSubagent } from './subagentProgress.js'

const archivedAgent: SubagentProgress = {
  agentType: 'researcher',
  depth: 1,
  goal: 'inspect runtime',
  id: 'child-1',
  index: 0,
  name: 'runtime-audit',
  notes: [],
  parentId: null,
  status: 'running',
  taskCount: 1,
  thinking: [],
  toolCount: 0,
  tools: []
}

describe('reconcileArchivedSubagent', () => {
  afterEach(clearSpawnHistory)

  it('updates the original archived row without adding or reordering transcript messages', () => {
    const messages: Msg[] = [
      { role: 'user', text: 'inspect this' },
      { kind: 'trail', role: 'system', subagents: [archivedAgent], text: '' },
      { role: 'assistant', text: 'The background audit is still running.' },
      { role: 'user', text: 'continue while it finishes' }
    ]
    const next = reconcileArchivedSubagent(
      messages,
      {
        agent_name: 'runtime-audit',
        agent_type: 'researcher',
        goal: 'inspect runtime',
        status: 'completed',
        subagent_id: 'child-1',
        summary: 'runtime verified',
        task_index: 0
      },
      current => ({
        notes: [...current.notes, '✓ ReadFile — complete'],
        status: 'completed',
        summary: 'runtime verified'
      })
    )

    expect(next).toHaveLength(messages.length)
    expect(next.map(message => message.text)).toEqual(messages.map(message => message.text))
    expect(next[1]?.subagents?.[0]).toMatchObject({
      id: 'child-1',
      name: 'runtime-audit',
      notes: ['✓ ReadFile — complete'],
      status: 'completed',
      summary: 'runtime verified'
    })
    expect(next[3]).toBe(messages[3])
  })

  it('returns the same transcript when the subagent was never archived', () => {
    const messages: Msg[] = [{ role: 'assistant', text: 'No delegated work.' }]
    const next = reconcileArchivedSubagent(
      messages,
      { goal: 'missing', subagent_id: 'unknown', task_index: 0 },
      () => ({ status: 'completed' })
    )

    expect(next).toBe(messages)
  })

  it('reconciles the matching spawn-history row used by the agents overlay', () => {
    pushSnapshot([{ ...archivedAgent, status: 'interrupted' }], { sessionId: 'session-a', startedAt: 1 })
    reconcileSpawnHistorySubagent(
      { goal: 'inspect runtime', status: 'completed', subagent_id: 'child-1', task_index: 0 },
      () => ({ status: 'completed', summary: 'runtime verified' })
    )

    expect(getSpawnHistory()[0]?.subagents[0]).toMatchObject({
      id: 'child-1',
      status: 'completed',
      summary: 'runtime verified'
    })
  })

  it('returns only the active session history from the warm global cache', () => {
    pushSnapshot([archivedAgent], { sessionId: 'session-a', startedAt: 1 })
    pushSnapshot([{ ...archivedAgent, id: 'child-2' }], { sessionId: 'session-b', startedAt: 2 })

    expect(spawnHistoryForSession(getSpawnHistory(), 'session-a').map(snapshot => snapshot.sessionId)).toEqual([
      'session-a'
    ])
    expect(spawnHistoryForSession(getSpawnHistory(), 'session-b')[0]?.subagents[0]?.id).toBe('child-2')
    expect(spawnHistoryForSession(getSpawnHistory(), null)).toEqual([])
    expect(spawnHistoryForSession(getSpawnHistory(), '  ')).toEqual([])
  })
})
