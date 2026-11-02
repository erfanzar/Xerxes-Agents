// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import {
  agentHistoryBlocks, agentLiveEventOf, agentTranscriptBlocks, liveAgentBlocks, prependAgentHistory, readAgentHistory,
  refreshAgentHistory, retainLiveEvents, settleSavedTools, toolIdsOf, type HistoryAction,
} from '../src/desktop/renderer/agentTranscript.js'
import { AgentTranscriptFeed } from '../src/desktop/renderer/AgentInspector.js'
import { keyedActivityGroups } from '../src/desktop/renderer/activityGroups.js'
import type { Block, ToolItem } from '../src/desktop/renderer/types.js'

const say = (id: string, content: string, thinking?: string): HistoryAction => ({ id, messages: [{ role: 'assistant', content }], executions: [], thinking: thinking ? [thinking] : [] })
const call = (id: string, callId: string, name: string, args: unknown, result?: string): HistoryAction => ({
  id,
  messages: [{ role: 'assistant', content: '', tool_calls: [{ id: callId, type: 'function', function: { name, arguments: JSON.stringify(args) } }] }, ...(result === undefined ? [] : [{ role: 'tool', tool_call_id: callId, content: result }])],
  executions: [{ toolCallId: callId, name, durationMs: 1500 }],
  thinking: [],
})
const tools = (blocks: readonly Block[]): ToolItem[] => blocks.flatMap(block => block.kind === 'tools' ? [...block.items] : [])

const actions: HistoryAction[] = [
  { id: '0:message', messages: [{ role: 'user', content: 'Fix reconnect' }], executions: [], thinking: [] },
  say('1:message', 'Reading the controller first.', 'Find where the session is restored.'),
  call('2:tool:0', 'c-run', 'exec_command', { cmd: 'bun', args: ['test', 'recovery'] }, JSON.stringify({ stdout: '1 fail\n', stderr: 'boom\n', exitCode: 1 })),
  call('3:tool:0', 'c-edit', 'FileEditTool', { file_path: 'src/a.ts', old_string: 'a\n', new_string: 'b\nc\n' }, 'Applied 1 replacement(s).'),
  call('4:tool:0', 'c-denied', 'exec_command', { cmd: 'rm' }, 'Tool execution failed: denied'),
  call('5:tool:0', 'c-pending', 'exec_command', { cmd: 'bun', args: ['test'] }),
]

test('a saved page folds into the main transcript blocks: prose, reasoning, commands, edits and failures', () => {
  const blocks = agentHistoryBlocks(actions, new Map(), true)
  // One tools block per saved action, as the main history fold makes; the
  // feed groups consecutive ones into a single "Ran N commands" disclosure.
  expect(blocks.map(block => block.kind)).toEqual(['user', 'thinking', 'agent', 'tools', 'tools', 'tools', 'tools'])
  const groups = keyedActivityGroups(blocks)
  expect(groups.map(group => group.blocks.map(block => block.kind))).toEqual([['user'], ['thinking'], ['agent'], ['tools', 'tools', 'tools', 'tools']])
  const items = tools(blocks)
  expect(items.map(item => item.id)).toEqual(['c-run', 'c-edit', 'c-denied', 'c-pending'])
  expect(items.find(item => item.id === 'c-run')).toMatchObject({ state: 'done', dur: '1.5s' })
  expect(items.find(item => item.id === 'c-edit')).toMatchObject({ path: 'src/a.ts', diff: { adds: 2, dels: 1 } })
  expect(items.find(item => item.id === 'c-denied')).toMatchObject({ state: 'failed' })
  // A call with no saved result is still running while the agent works…
  expect(items.find(item => item.id === 'c-pending')).toMatchObject({ state: 'working' })
  expect(items.find(item => item.id === 'c-pending')?.error).toBeUndefined()
  // …and says it never returned once the agent stopped.
  const stopped = tools(agentHistoryBlocks(actions, new Map(), false)).find(item => item.id === 'c-pending')
  expect(stopped).toMatchObject({ state: 'failed', error: 'The agent stopped before this call returned a result.' })
})

test('block ids stay stable across reads of the same actions', () => {
  const ids = new Map<string, number>()
  const first = agentHistoryBlocks(actions.slice(0, 3), ids, true).map(block => block.id)
  const second = agentHistoryBlocks(actions, ids, true).map(block => block.id)
  expect(second.slice(0, first.length - 1)).toEqual(first.slice(0, -1))
})

test('the inspect reply says whether history is unsupported, unavailable or a page', () => {
  expect(readAgentHistory({ ok: true, agent: {} })).toEqual({ kind: 'unsupported' })
  expect(readAgentHistory({ ok: true, history: null, history_note: 'Not saved.' })).toEqual({ kind: 'unavailable', note: 'Not saved.' })
  const read = readAgentHistory({ ok: true, history: { actions, has_more: false, before: null, clipped: true } })
  expect(read.kind).toBe('page')
  if (read.kind === 'page') expect(read.clipped).toBe(true)
  expect(() => readAgentHistory({ history: { actions: 'no', has_more: false, before: null } })).toThrow()
})

test('a fresh newest page keeps older loaded actions and paging prepends without duplicates', () => {
  const page = (list: HistoryAction[], before: string | null) => ({ actions: list, has_more: before !== null, before })
  const opened = refreshAgentHistory(null, page(actions.slice(3), 'cursor-3'), false)
  const paged = prependAgentHistory(opened, page(actions.slice(0, 4), null), false)
  expect(paged.actions.map(action => action.id)).toEqual(actions.map(action => action.id))
  expect(paged.more).toBe(false)
  // The child kept working: the newest window moved past 3:tool:0.
  const later = refreshAgentHistory(paged, page([...actions.slice(4), say('6:message', 'Done.')], 'cursor-4'), false)
  expect(later.actions.map(action => action.id)).toEqual([...actions.map(action => action.id), '6:message'])
  expect(later.more).toBe(false)
})

test('live subagent_event envelopes fold for exactly this agent, using the full fields', () => {
  const envelope = (agent: string, type: string, payload: Record<string, unknown>) => ({ agent_id: agent, event: { type, payload } })
  // Grandchildren arrive nested; only the innermost envelope names the owner.
  const nested = { agent_id: 'parent-agent', event: { type: 'subagent_event', payload: envelope('me', 'text_part', { text: '…tail', text_full: 'Whole **reply**' }) } }
  expect(agentLiveEventOf(nested, 'me')?.type).toBe('text_part')
  expect(agentLiveEventOf(nested, 'parent-agent')).toBeNull()
  expect(agentLiveEventOf(envelope('other', 'tool_call', { id: 'x' }), 'me')).toBeNull()
  expect(agentLiveEventOf(envelope('me', 'usage', {}), 'me')).toBeNull()

  const events = [
    agentLiveEventOf(envelope('me', 'think_part', { think: 'Plan it.' }), 'me')!,
    agentLiveEventOf(nested, 'me')!,
    agentLiveEventOf(envelope('me', 'tool_call', { id: 'c-run', name: 'exec_command', arguments: 'preview' }), 'me')!,
    agentLiveEventOf(envelope('me', 'tool_call', { id: 'l-1', name: 'exec_command', arguments: 'bun …', arguments_full: JSON.stringify({ cmd: 'bun', args: ['test'] }) }), 'me')!,
    agentLiveEventOf(envelope('me', 'tool_result', { tool_call_id: 'l-1', name: 'exec_command', return_value: 'short', result_full: JSON.stringify({ stdout: 'all 8 pass\n', exitCode: 0 }), duration_ms: 900 }), 'me')!,
    agentLiveEventOf(envelope('me', 'tool_call', { id: 'l-2', name: 'exec_command', arguments_full: JSON.stringify({ cmd: 'rm' }) }), 'me')!,
    agentLiveEventOf(envelope('me', 'tool_result', { tool_call_id: 'l-2', name: 'exec_command', return_value: 'Permission denied', permitted: false }), 'me')!,
  ]
  const saved = agentHistoryBlocks(actions, new Map(), true)
  const live = liveAgentBlocks(events, toolIdsOf(saved), true)
  expect(live.map(block => block.kind)).toEqual(['thinking', 'agent', 'tools'])
  expect(live.find(block => block.kind === 'agent')).toMatchObject({ text: 'Whole **reply**' })
  const items = tools(live)
  // c-run is already on the saved page: never drawn twice.
  expect(items.map(item => item.id)).toEqual(['l-1', 'l-2'])
  expect(JSON.parse(items[0]!.input)).toEqual({ cmd: 'bun', args: ['test'] })
  expect(items[0]).toMatchObject({ state: 'done', dur: '0.9s' })
  expect(items[0]!.output).toContain('all 8 pass')
  expect(items[1]).toMatchObject({ state: 'failed' })
  // The live tail streams; saved replies do not.
  const whole = agentTranscriptBlocks(saved, live, true)
  expect(whole.length).toBe(saved.length + live.length)
})

test('a live result settles a saved running call, and a fresh page retires covered events', () => {
  const saved = agentHistoryBlocks(actions, new Map(), true)
  const result = { type: 'tool_result', payload: { tool_call_id: 'c-pending', name: 'exec_command', result_full: JSON.stringify({ stdout: 'ok\n', exitCode: 0 }), duration_ms: 2000 }, seq: 1 }
  const settled = tools(settleSavedTools(saved, [result])).find(item => item.id === 'c-pending')
  expect(settled).toMatchObject({ state: 'done', dur: '2.0s' })
  expect(settled?.output).toContain('ok')

  const text = { type: 'text_part', payload: { text: 'old' }, seq: 2 }
  const newer = { type: 'text_part', payload: { text: 'new' }, seq: 5 }
  const page = { actions, has_more: false, before: null }
  // The page has not answered c-pending yet: its result stays; covered text goes.
  expect(retainLiveEvents([result, text, newer], 3, page).map(event => event.seq)).toEqual([1, 5])
  const answered = { actions: [...actions.slice(0, 5), call('5:tool:0', 'c-pending', 'exec_command', {}, 'ok')], has_more: false, before: null }
  expect(retainLiveEvents([result, text, newer], 3, answered).map(event => event.seq)).toEqual([5])
})

test('a saved reply still being written carries the streaming caret only while the agent works', () => {
  const saved = agentHistoryBlocks([say('0:message', 'Partial reply')], new Map(), true)
  expect(agentTranscriptBlocks(saved, [], true).at(-1)).toMatchObject({ kind: 'agent', streaming: true })
  expect(agentTranscriptBlocks(saved, [], false).at(-1)).toMatchObject({ kind: 'agent', streaming: false })
})

test('the inspector feed renders with the main conversation components', () => {
  const blocks = agentHistoryBlocks(actions.slice(0, 4), new Map(), false)
  const html = renderToStaticMarkup(createElement(AgentTranscriptFeed, { blocks, active: false }))
  expect(html).toContain('msg msg--agent')
  expect(html).toContain('Reading the controller first.')
  expect(html).toContain('class="activity-group')
  expect(html).toContain('msg--user')
  expect(renderToStaticMarkup(createElement(AgentTranscriptFeed, { blocks: [], active: true }))).toContain('Waiting for the agent')
})
