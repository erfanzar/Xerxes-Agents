// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { AgentMenu, agentMenuRows } from '../src/desktop/renderer/AgentSwitcher.js'
import { AgentDetailsCard, AgentInspector, retainedAgentBlocks } from '../src/desktop/renderer/AgentInspector.js'
import type { SessionRow } from '../src/desktop/renderer/types.js'

const agent = (id: string, title: string, status: string, extra: Partial<SessionRow> = {}): SessionRow => ({
  id, key: id, title, status, age: '', current: false, kind: 'subagent', turns: 0, messages: 0, cwd: '', untitled: false, ...extra,
})

test('the agent menu lists working agents first, keeps runtime order within a state, and filters by title or id', () => {
  const rows = [agent('a', 'Finished review', 'completed'), agent('b', 'Live probe', 'running'), agent('c', 'Broken build', 'failed'), agent('d', 'Second probe', 'working')]
  expect(agentMenuRows(rows).map(row => row.id)).toEqual(['b', 'd', 'c', 'a'])
  expect(agentMenuRows(rows, 'PROBE').map(row => row.id)).toEqual(['b', 'd'])
  expect(agentMenuRows(rows, 'c').map(row => row.id)).toContain('c')
  expect(agentMenuRows(rows, 'nothing-matches')).toEqual([])
})

test('the closed agent menu is one labelled trigger and renders no rows', () => {
  const rows = [agent('a', 'Unique agent title', 'running')]
  const html = renderToStaticMarkup(createElement(AgentMenu, { rows, label: '1 agent', title: 'Open a sub-agent', onPick: () => {} }))
  expect(html).toContain('aria-haspopup="menu"')
  expect(html).toContain('aria-expanded="false"')
  expect(html).toContain('1 agent')
  expect(html).not.toContain('Unique agent title')
})

test('an open working agent takes the pane as a plain chat, with its details in the rail card', () => {
  const row = agent('child-1', 'Inspect reconnect', 'running', { agentDetails: { goal: 'Inspect the reconnect path', model: 'glm-5', baseAgent: 'researcher', summary: '', error: '', filesRead: [], filesWritten: ['src/a.ts'] } })
  const html = renderToStaticMarkup(createElement(AgentInspector, { row, rows: [row], sessionKey: 'parent', online: true }))
  expect(html).toContain('class="agent-inspector agent-pane"')
  expect(html).toContain('aria-label="Agent conversation"')
  // The task opens the chat as the person's message, the way the main conversation does.
  expect(html).toContain('class="msg-turn msg-turn--user"')
  expect(html).toContain('Inspect the reconnect path')
  // Identity and metadata are not in the chat.
  for (const gone of ['Assigned task', 'Base agent', 'glm-5', 'Runtime details', 'src/a.ts']) expect(html).not.toContain(gone)
  // Before the first read answers, only the task and a loading note show; no summary poses as a reply.
  const loadingRow = agent('child-1b', 'Inspect reconnect', 'running', { agentDetails: { goal: 'Inspect the reconnect path', summary: 'Partial summary text', error: '', model: '', filesRead: [], filesWritten: [] } })
  const loading = renderToStaticMarkup(createElement(AgentInspector, { row: loadingRow, rows: [loadingRow], sessionKey: 'parent', online: true }))
  expect(loading).toContain('Loading the agent’s conversation…')
  expect(loading).not.toContain('Partial summary text')
  const card = renderToStaticMarkup(createElement(AgentDetailsCard, { row, rows: [row], sessionKey: 'parent' }))
  for (const shown of ['aria-label="Agent details"', 'Inspect reconnect', 'Working', 'Base agent', 'researcher', 'glm-5', 'Assigned task', 'Files changed', 'src/a.ts', 'Runtime details']) expect(card).toContain(shown)
  expect(html).toContain('aria-label="Message this agent"')
  expect(html).toContain('aria-label="Stop agent"')
  expect(html).toContain('aria-label="Send to agent"')
  expect(html).toContain('The task and other agents keep running')
  expect(html).not.toContain('aria-label="Retry agent"')
  // The old rail inspector's back link is gone: the header breadcrumb returns.
  expect(html).not.toContain('agent-inspector__back')
})

test('an older runtime’s retained summary still reads as a chat: task bubble, replies, one closed work group', () => {
  const calls = [{ id: 'c1', name: 'exec_command', verb: 'exec_command', arg: 'ls', dur: '0.1s', state: 'done' as const }]
  const blocks = retainedAgentBlocks({ summary: '', error: '', model: '', filesRead: [], filesWritten: [], notes: ['I’ll list the folder.', '  '], toolCalls: calls }, 'List the renderer folder', 'Listed 40 files.')
  expect(blocks.map(block => block.kind)).toEqual(['user', 'agent', 'tools', 'agent'])
  expect(blocks[0]).toMatchObject({ kind: 'user', text: 'List the renderer folder' })
  expect(blocks.at(-1)).toMatchObject({ kind: 'agent', text: 'Listed 40 files.' })
  expect(retainedAgentBlocks(undefined, '', '')).toEqual([])
})

test('a finished agent offers retry instead of stop, and a disconnected one disables its dock', () => {
  const row = agent('child-2', 'Done review', 'completed')
  const html = renderToStaticMarkup(createElement(AgentInspector, { row, rows: [row], sessionKey: 'parent', online: true }))
  expect(html).toContain('aria-label="Retry agent"')
  expect(html).toContain('aria-label="Follow-up for retry"')
  expect(html).not.toContain('aria-label="Stop agent"')
  const offline = renderToStaticMarkup(createElement(AgentInspector, { row: agent('child-3', 'Live', 'running'), rows: [], sessionKey: 'parent', online: false }))
  expect(offline).toMatch(/<fieldset class="agent-dock composer-wrap" disabled/)
  expect(offline).toContain('Disconnected')
})

test('an unconfirmed spawn request shows a read-only bar and no controls', () => {
  const row = agent('call:0', 'Pending spawn', 'starting', { agentDetails: { provisional: true, goal: 'Review code', summary: '', error: '', model: '', filesRead: [], filesWritten: [] } })
  const html = renderToStaticMarkup(createElement(AgentInspector, { row, rows: [row], sessionKey: 'parent', online: true }))
  expect(html).toContain('agent-dock--readonly')
  expect(html).toContain('Not controllable yet.')
  expect(html).not.toContain('aria-label="Stop agent"')
  expect(html).not.toContain('aria-label="Message this agent"')
})
