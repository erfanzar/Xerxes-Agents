// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { groupActivity, isDisclosedActivity } from '../src/desktop/renderer/activityGroups.js'
import { activitySummary, liveActivityPhrase, toolPhrase } from '../src/desktop/renderer/activityPhrase.js'
import type { Block, ToolItem } from '../src/desktop/renderer/types.js'

const tool = (name: string, arg: string, state: ToolItem['state'] = 'done', path?: string): ToolItem => ({
  id: `${name}:${arg}:${Math.random()}`, name, verb: name.toLowerCase(), arg, dur: '', state, input: '', output: '',
  ...(path ? { path } : {}),
})
const tools = (...items: ToolItem[]): Block => ({ kind: 'tools', id: 1, running: items.some(i => i.state === 'working'), items })

test('a running call reads as what the agent is doing, across wire spellings', () => {
  expect(toolPhrase(tool('exec_command', 'bash -c JAX_PLATFORMS=cpu pytest tests/'))).toBe('Running pytest tests/')
  expect(toolPhrase(tool('FileEditTool', 'src/desktop/renderer/store.ts', 'working', 'src/desktop/renderer/store.ts'))).toBe('Editing store.ts')
  expect(toolPhrase(tool('ReadFile', '/repo/README.md'))).toBe('Reading README.md')
  expect(toolPhrase(tool('GrepTool', 'turn_end'))).toBe('Searching for turn_end')
  expect(toolPhrase(tool('SpawnAgents', ''))).toBe('Spawning agents')
  expect(toolPhrase(tool('SendMessageTool', 'reviewer'))).toBe('Messaging reviewer')
  expect(toolPhrase(tool('mcp.github:AwaitAgents', ''))).toBe('Waiting on agents')
  expect(toolPhrase(tool('ha_call_service', 'light.on'))).toBe('Using ha call service')
})

test('the live phrase follows the newest running call, then explains any wait', () => {
  expect(liveActivityPhrase([tools(tool('ReadFile', 'a.ts'), tool('exec_command', 'bun test', 'working'))])).toBe('Running bun test')
  expect(liveActivityPhrase([tools(tool('ReadFile', 'a.ts')), { kind: 'thinking', id: 2, text: 'hm', streaming: true }])).toBe('Thinking')
  expect(liveActivityPhrase([
    tools(tool('SpawnAgents', '')),
    { kind: 'agents', id: 2, members: [{ key: 'a', title: 'A', status: 'working' }, { key: 'b', title: 'B', status: 'working' }, { key: 'c', title: 'C', status: 'completed' }] },
  ])).toBe('Waiting on 2 agents')
})

test('a finished group summarizes by outcome, counting distinct files', () => {
  const summary = activitySummary([tools(
    tool('exec_command', 'bun test'), tool('exec_command', 'bun run check'), tool('exec_command', 'git diff'),
    tool('FileEditTool', 'a.ts', 'done', 'a.ts'), tool('FileEditTool', 'a.ts', 'done', 'a.ts'), tool('WriteFile', 'b.ts', 'done', 'b.ts'),
    tool('ReadFile', 'c.ts'),
  )])
  expect(summary).toBe('Ran 3 commands, edited 2 files and read 1 file')
  expect(activitySummary([tools(tool('GrepTool', 'x'))])).toBe('Ran 1 search')
  expect(activitySummary([{ kind: 'agents', id: 1, members: [{ key: 'a', title: 'A', status: 'completed' }] }])).toBe('Started an agent')
})

test('groups with no work in them are not folded behind a disclosure', () => {
  expect(isDisclosedActivity([{ kind: 'checkpoint', id: 1, turn: 2, adds: 3, dels: 1 }])).toBe(false)
  expect(isDisclosedActivity([{ kind: 'notice', id: 1, error: false, text: 'compacted' }])).toBe(false)
  // An error stays folded (the failure card states it) but under a real label.
  expect(isDisclosedActivity([{ kind: 'notice', id: 1, error: true, text: 'Provider failed' }])).toBe(true)
  expect(activitySummary([{ kind: 'notice', id: 1, error: true, text: 'Provider failed' }])).toBe('Runtime error')
  expect(isDisclosedActivity([tools(tool('ReadFile', 'a.ts')), { kind: 'checkpoint', id: 2, turn: 2, adds: 3, dels: 1 }])).toBe(true)
})

test('a turn-end checkpoint folds into its turn, so the transcript ends on the answer', () => {
  const blocks: Block[] = [
    { kind: 'user', id: 1, text: 'fix it' },
    tools(tool('FileEditTool', 'a.ts', 'done', 'a.ts')),
    { kind: 'agent', id: 3, text: 'Fixed.', streaming: false },
    { kind: 'checkpoint', id: 4, turn: 1, adds: 1, dels: 1 },
    { kind: 'user', id: 5, text: 'hello' },
    { kind: 'agent', id: 6, text: 'hi', streaming: false },
    { kind: 'checkpoint', id: 7, turn: 2, adds: 1, dels: 1 },
  ]
  const groups = groupActivity(blocks)
  expect(groups.map(group => group.map(block => block.kind))).toEqual([
    ['user'], ['tools', 'checkpoint'], ['agent'], ['user'], ['agent'],
    // No activity in this turn to join: it stays a plain row, never a box.
    ['checkpoint'],
  ])
})
