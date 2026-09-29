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
    { key: 'b', runtimeId: 'b', title: 'S01 attention kernels', status: 'working', group: group('Review'), model: 'claude-code/sonnet', startedAt: now - 30_000, activity: 'Reading kda.py', tokens: 12_400, toolUses: 9, recentTools: ['Searching for shard_map', 'Reading kda.py'] },
    { key: 'c', runtimeId: 'c', title: 'S09 ejkernel', status: 'failed', group: group('Review'), model: 'claude-code/sonnet', error: 'Prompt too long\nstack' },
  ]
  const html = renderToStaticMarkup(createElement(AgentsCard, { members }))
  expect(html).toContain('aria-label="Workflow: Review vnext"')
  expect(html).toContain('>Review vnext<')
  expect(html).toContain('3 agents · 2 phases · opus, sonnet · 9 tool uses · 12.4K tokens')
  expect(html.indexOf('>Slice<')).toBeLessThan(html.indexOf('>Review<'))
  expect(html).toContain('Reading kda.py')
  // Claude Code's shape: the branch's stats, then ⎿ what it is doing now.
  expect(html).toContain('9 tool uses · 12.4K tokens')
  expect(html).toContain('⎿')
  expect(html).toContain('+8 more tool uses')
  expect(html).toContain('>Done (10s)<')
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

test('a child tool preview reads as a command line, not key=value pairs', async () => {
  const { previewArguments } = await import('../src/desktop/renderer/agentMembers.js')
  expect(previewArguments('cmd=env, args=JAX_PLATFORMS=cpu,mps python bench.py, workdir=.')).toEqual({ cmd: 'env', args: 'JAX_PLATFORMS=cpu,mps python bench.py', workdir: '.' })
  expect(previewArguments('{"file_path":"src/a.ts"}')).toEqual({ file_path: 'src/a.ts' })
  let fleet: readonly SessionRow[] = foldAgentEvent([], { agent_id: 'x', title: 'bench', event: { type: 'turn_begin', payload: {} } })
  fleet = foldAgentEvent(fleet, { agent_id: 'x', event: { type: 'tool_call', payload: { id: 't', name: 'exec_command', arguments: 'cmd=env, args=JAX_PLATFORMS=cpu,mps python bench.py' } } })
  const members = new Map<string, AgentMember>()
  syncAgentMembers(members, fleet)
  expect(members.get('x')?.activity).toBe('Running env JAX_PLATFORMS=cpu,mps python bench.py')
  expect(members.get('x')?.toolUses).toBe(1)
})

test('workflow runs get their own panel card, grouped by phase, apart from other agents', async () => {
  const { RailWorkflows, workflowRuns } = await import('../src/desktop/renderer/RailLists.js')
  const row = (id: string, status: string, phase?: string, label = 'f16 perf gaps'): SessionRow => ({
    id, key: id, title: id, status, age: '', current: false, kind: 'subagent', turns: 0, messages: 0, cwd: '', untitled: false,
    agentDetails: { summary: '', error: '', model: 'claude-code/sonnet', filesRead: [], filesWritten: [], ...(phase ? { group: { id: `wf-${label}`, label, phase } } : {}) },
  })
  const rows = [row('softmax_f16', 'running', 'Investigate'), row('gpt2_f16', 'completed', 'Investigate'), row('verify', 'running', 'Verify'), row('Map native core', 'completed')]
  const runs = workflowRuns(rows)
  expect(runs).toHaveLength(1)
  expect(runs[0]!.phases.map(phase => [phase.name, phase.rows.length])).toEqual([['Investigate', 2], ['Verify', 1]])
  const html = renderToStaticMarkup(createElement(RailWorkflows, { rows, onInspect: () => {} }))
  expect(html).toContain('aria-label="Workflows"')
  expect(html).toContain('f16 perf gaps')
  expect(html).toContain('2 working · 1 done')
  // A running workflow opens to its agents by phase; plain agents stay out.
  expect(html).toContain('Investigate')
  expect(html).toContain('softmax_f16')
  expect(html).not.toContain('Map native core')
})

test('a finished run shows its cost in the card header', () => {
  const g = { id: 'wf_c', label: 'Review', phase: 'Find', costUsd: 0.4217 }
  const html = renderToStaticMarkup(createElement(AgentsCard, { members: [
    { key: 'a', runtimeId: 'a', title: 'slice 1', status: 'completed', group: g },
    { key: 'b', runtimeId: 'b', title: 'slice 2', status: 'completed', group: { ...g, costPartial: true } },
  ] }))
  expect(html).toContain('$0.42')
})

test('the Agents chip menu lists off, auto and eager with auto as the default', async () => {
  const { DELEGATION_CHOICES } = await import('../src/desktop/renderer/Overlays.js')
  expect(DELEGATION_CHOICES.map(choice => choice.mode)).toEqual(['off', 'auto', 'eager'])
})
