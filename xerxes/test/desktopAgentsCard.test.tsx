// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { AgentsCard } from '../src/desktop/renderer/AgentsCard.js'
import { foldAgentEvent } from '../src/desktop/renderer/agentEvents.js'
import { syncAgentMembers } from '../src/desktop/renderer/agentMembers.js'
import { spawnMembersOf } from '../src/desktop/renderer/store.js'
import type { AgentMember, SessionRow } from '../src/desktop/renderer/types.js'

const group = (phase: string) => ({ id: 'wf_1', label: 'Review vnext', phase })

test('a workflow card is titled by the run and grouped by phase, with live activity and models', () => {
  const now = Date.now()
  const members: AgentMember[] = [
    { key: 'a', runtimeId: 'a', title: 'Slice files', status: 'completed', group: group('Slice'), model: 'claude-code/opus', startedAt: now - 60_000, finishedAt: now - 50_000, summary: '272 files into 15 slices' },
    { key: 'b', runtimeId: 'b', title: 'S01 attention kernels', status: 'working', group: group('Review'), model: 'claude-code/sonnet', startedAt: now - 30_000, activity: 'Reading kda.py', tokens: 12_400 },
    { key: 'c', runtimeId: 'c', title: 'S09 ejkernel', status: 'failed', group: group('Review'), model: 'claude-code/sonnet', error: 'Prompt too long\nstack' },
  ]
  const html = renderToStaticMarkup(createElement(AgentsCard, { members }))
  expect(html).toContain('aria-label="Workflow: Review vnext"')
  expect(html).toContain('>Review vnext<')
  expect(html).toContain('3 agents · 2 phases · opus, sonnet')
  expect(html.indexOf('>Slice<')).toBeLessThan(html.indexOf('>Review<'))
  expect(html).toContain('Reading kda.py')
  expect(html).toContain('12K tok')
  expect(html).toContain('272 files into 15 slices')
  expect(html).toContain('>Prompt too long</span>')
  expect(html).toContain('1 running')
})

test('a phase of thousands draws as a dot grid with rows only for live and failed agents', () => {
  const members: AgentMember[] = Array.from({ length: 2_000 }, (_, index) => ({
    key: `m${index}`,
    runtimeId: `m${index}`,
    title: `shard ${index}`,
    status: index < 20 ? 'working' : index < 25 ? 'failed' : 'completed',
    group: group('Classify'),
  }))
  const html = renderToStaticMarkup(createElement(AgentsCard, { members }))
  expect(html.match(/class="acard__dot"/g)?.length).toBe(2_000)
  // 8 running + 5 failed rows, never all two thousand.
  expect(html.match(/class="acard__row"/g)?.length).toBe(13)
  expect(html).toContain('Show all 2000')
  expect(html).toContain('20 running')
})

test('workflow agents join the card from their own events, matched by id and never quadratically', () => {
  let fleet: readonly SessionRow[] = []
  for (let index = 0; index < 500; index++) {
    fleet = foldAgentEvent(fleet, {
      agent_id: `sub_${index}`,
      title: `shard ${index}`,
      model: 'fast-model',
      group: { id: 'wf_9', label: 'Sweep', phase: 'Scan' },
      event: { type: 'turn_begin', payload: { status: 'running' } },
    })
  }
  fleet = foldAgentEvent(fleet, {
    agent_id: 'sub_3',
    event: { type: 'tool_call', payload: { id: 't1', name: 'ReadFile', arguments: JSON.stringify({ file_path: 'src/model/kda.py' }) } },
  })
  const members = new Map<string, AgentMember>()
  const started = performance.now()
  expect(syncAgentMembers(members, fleet)).toBe(true)
  expect(performance.now() - started).toBeLessThan(250)
  expect(members.size).toBe(500)
  const three = members.get('sub_3')!
  expect(three.group).toEqual({ id: 'wf_9', label: 'Sweep', phase: 'Scan' })
  expect(three.activity).toBe('Reading kda.py')
  expect(three.model).toBe('fast-model')
  // A second pass with nothing new reports no change.
  expect(syncAgentMembers(members, fleet)).toBe(false)
  fleet = foldAgentEvent(fleet, { agent_id: 'sub_3', summary: 'Found 3 issues\nmore', event: { type: 'turn_end', payload: { status: 'completed' } } })
  expect(syncAgentMembers(members, fleet)).toBe(true)
  expect(members.get('sub_3')).toMatchObject({ status: 'completed', summary: 'Found 3 issues' })
  expect(members.get('sub_3')?.activity).toBeUndefined()
})

test('a Workflow call opens no placeholder member, and spawn batches are no longer cut at 24', () => {
  expect(spawnMembersOf('Workflow', JSON.stringify({ name: 'Review', script: 'return 1' }), 'call-1')).toEqual([])
  const batch = spawnMembersOf('SpawnAgents', JSON.stringify({ agents: Array.from({ length: 32 }, (_, i) => ({ title: `t${i}`, prompt: 'p' })) }), 'call-2')
  expect(batch).toHaveLength(32)
})
