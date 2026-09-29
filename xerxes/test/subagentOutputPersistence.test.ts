// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import {
  mergePersistedSubagentSnapshots,
  replacePersistedSubagentSnapshots,
  SUBAGENT_SNAPSHOT_METADATA_KEY,
} from '../src/agents/subagentPersistence.js'
import type { SpawnedAgentSnapshot } from '../src/operators/subagents.js'

function agent(index: number, output: string, grouped = true): SpawnedAgentSnapshot {
  const second = String(index % 60).padStart(2, '0')
  const minute = String(Math.floor(index / 60)).padStart(2, '0')
  return {
    agentId: 'coder',
    closed: true,
    createdAt: '2026-01-01T00:00:00.000Z',
    id: `agent-${index}`,
    name: `agent-${index}`,
    promptProfile: 'coder',
    queueSize: 0,
    status: 'completed',
    title: `agent-${index}`,
    updatedAt: `2026-01-01T00:${minute}:${second}.000Z`,
    lastOutput: output,
    ...(grouped ? { group: { id: 'run-1', label: 'audit', phase: 'Investigate' } } : {}),
  }
}

function saved(metadata: Record<string, unknown>): readonly Record<string, unknown>[] {
  return metadata[SUBAGENT_SNAPSHOT_METADATA_KEY] as readonly Record<string, unknown>[]
}

describe('saved subagent output', () => {
  test('a workflow agent keeps its whole result, not a 2,000-character excerpt', () => {
    const metadata: Record<string, unknown> = {}
    const result = JSON.stringify({ slice: 'gpt2_f16', root_cause: 'x'.repeat(6_000) })
    replacePersistedSubagentSnapshots(metadata, [agent(1, result)])
    expect(saved(metadata)[0]).toMatchObject({ last_output: result })
    expect(saved(metadata)[0]).not.toHaveProperty('last_output_truncated')
  })

  test('output past the per-agent cap is cut and marked as trimmed', () => {
    const metadata: Record<string, unknown> = {}
    replacePersistedSubagentSnapshots(metadata, [agent(1, 'y'.repeat(20_000), false)])
    const row = saved(metadata)[0]!
    expect(String(row.last_output).length).toBe(16_000)
    expect(row.last_output_truncated).toBe(true)
  })

  test('past the session budget the oldest workflow agents are trimmed, the newest kept whole', () => {
    const metadata: Record<string, unknown> = {}
    // 100 agents × 15,000 characters is past the 1,000,000 budget.
    const agents = Array.from({ length: 100 }, (_, index) => agent(index, `${index}:`.padEnd(15_000, 'z')))
    replacePersistedSubagentSnapshots(metadata, agents)
    const rows = saved(metadata)
    const newest = rows.find(row => row.id === 'agent-99')!
    const oldest = rows.find(row => row.id === 'agent-0')!
    expect(String(newest.last_output).length).toBe(15_000)
    expect(newest).not.toHaveProperty('last_output_truncated')
    expect(String(oldest.last_output).length).toBe(2_000)
    expect(oldest.last_output_truncated).toBe(true)
    const total = rows.reduce((sum, row) => sum + String(row.last_output ?? '').length, 0)
    expect(total).toBeLessThanOrEqual(1_000_000 + 100 * 2_000)
    // Merging later progress keeps the manifest inside the same budget.
    mergePersistedSubagentSnapshots(metadata, [agent(100, '100:'.padEnd(15_000, 'z'))])
    const merged = saved(metadata)
    expect(String(merged.find(row => row.id === 'agent-100')!.last_output).length).toBe(15_000)
    expect(merged).toHaveLength(101)
  })
})
