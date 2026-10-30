// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { createDraftStore, draftAfterAccepted, type DraftStorage } from '../../client/drafts.js'
import { AGENTS_REFRESH_MS, AGENTS_UNSUPPORTED, TYPING_IDLE_MS, createHubController } from '../../client/hubController.js'
import { foldSummaryEvent, parseHistoryPage, type SupervisorConfirmation } from '../../client/hubParsers.js'
import { createHubStore, reduceHub, type HubState } from '../../client/hubStore.js'
import { EMPTY_TAIL, TOOL_OUTPUT_LIMIT, foldAll, foldSessionEvent, toolDetails } from '../../client/sessionFold.js'
import { pendingDetailsUnavailable } from '../viewModel.js'
import { batch, ev, goLive, harness, settle, type Harness } from './fakes.js'
import { hubEvent, liveState, summary } from './helpers.js'

const page = (actions: readonly unknown[]) => ({ history: { actions, before: null, has_more: false } })
const message = (id: string, role: string, content: string) => ({ id, messages: [{ role, content }], executions: [], thinking: [] })

async function taskSetup(extra: (h: Harness) => void = () => undefined) {
  const h = harness()
  h.hub.handlers.set('hub.events.read', () => ({ watermark: 'live-edge' }))
  h.hub.handlers.set('hub.presence.update', () => ({}))
  h.hub.handlers.set('hub.history', () => page([message('0:message', 'user', 'question')]))
  h.hub.handlers.set('agents.list', () => ({ agents: [] }))
  extra(h)
  const socket = await goLive(h)
  const store = createHubStore(liveState([summary({ id: 's1', status: 'working' })]))
  const controller = createHubController({ connection: h.conn, store, timers: h.timers, now: () => h.timers.now })
  return { h, socket, store, controller }
}

function selectedState(): HubState {
  return reduceHub(liveState([summary({ id: 's1', status: 'working' })]), { type: 'select', workspaceId: 'w1', sessionId: 's1' })
}

describe('tool calls carry their exact arguments, diffs, output and duration', () => {
  it('shows a shell command verbatim, an edit as a diff, and other arguments as pretty JSON', () => {
    expect(toolDetails('exec_command', { cmd: 'bun', args: ['test', 'a b'] })).toEqual({ args: "bun test 'a b'" })
    expect(toolDetails('Bash', JSON.stringify({ command: 'ls -la | head' }))).toEqual({ args: 'ls -la | head' })
    const edit = toolDetails('FileEditTool', { file_path: 'src/a.ts', old_string: 'one', new_string: 'two', replace_all: false })
    expect(edit.diff).toEqual({ path: 'src/a.ts', before: 'one', after: 'two' })
    expect(edit.args).toBe(JSON.stringify({ file_path: 'src/a.ts', replace_all: false }, null, 2))
    expect(toolDetails('WriteFile', { file_path: 'b.md', content: 'hello' }).diff).toEqual({ path: 'b.md', before: '', after: 'hello' })
    expect(toolDetails('ReadFile', { file_path: 'c.ts', limit: 10 }).args).toBe('{\n  "file_path": "c.ts",\n  "limit": 10\n}')
  })

  it('keeps full args, a long output and the duration for a live call', () => {
    let tail = foldSessionEvent(EMPTY_TAIL, { id: 'e1', type: 'tool_call', payload: { id: 'call-1', name: 'exec_command', arguments: JSON.stringify({ command: 'bun test --timeout 15000 xerxes/test/a.test.ts' }) } })
    tail = foldSessionEvent(tail, { id: 'e2', type: 'tool_result', payload: { tool_call_id: 'call-1', result: 'x'.repeat(20_000), permitted: true, duration_ms: 1234 } })
    const block = tail.blocks[0]
    expect(block).toMatchObject({ kind: 'tool', args: 'bun test --timeout 15000 xerxes/test/a.test.ts', state: 'done', durationMs: 1234 })
    expect(block?.kind === 'tool' ? block.output?.length : 0).toBe(TOOL_OUTPUT_LIMIT)
  })

  it('folds history pages the same way, including a diff and the duration', () => {
    const parsed = parseHistoryPage(page([{
      id: '1:tool:0',
      messages: [
        { role: 'assistant', content: '', tool_calls: [{ id: 'call-9', function: { name: 'FileEditTool', arguments: { file_path: 'x.ts', old_string: 'a', new_string: 'b' } } }] },
        { role: 'tool', tool_call_id: 'call-9', content: 'Edited x.ts' },
      ],
      executions: [{ toolCallId: 'call-9', duration_ms: 42 }],
      thinking: [],
    }]))
    expect(parsed.blocks[0]).toMatchObject({ kind: 'tool', callId: 'call-9', diff: { path: 'x.ts', before: 'a', after: 'b' }, output: 'Edited x.ts', durationMs: 42 })
  })
})

describe('todo list', () => {
  const todos = [{ content: 'Read the code', status: 'completed' }, { content: 'Fix it', status: 'in_progress' }, { content: 'Test', status: 'bogus' }]

  it('comes from the latest live TodoWriteTool call', () => {
    const tail = foldAll(EMPTY_TAIL, [
      { id: 'a', type: 'tool_call', payload: { id: 't1', name: 'TodoWriteTool', arguments: JSON.stringify({ todos: [{ content: 'old', status: 'pending' }] }) } },
      { id: 'b', type: 'tool_call', payload: { id: 't2', name: 'todo_write', arguments: { todos } } },
    ])
    expect(tail.todos).toEqual([{ content: 'Read the code', status: 'completed' }, { content: 'Fix it', status: 'in_progress' }, { content: 'Test', status: 'pending' }])
  })

  it('comes from history when the open task has no live list', () => {
    const history = page([{ id: '3:tool:0', messages: [{ role: 'assistant', content: '', tool_calls: [{ id: 't9', function: { name: 'TodoWriteTool', arguments: JSON.stringify({ todos }) } }] }], executions: [], thinking: [] }])
    const parsed = parseHistoryPage(history)
    expect(parsed.todos?.map(item => item.content)).toEqual(['Read the code', 'Fix it', 'Test'])
    const state = reduceHub(selectedState(), { type: 'historyLoaded', workspaceId: 'w1', sessionId: 's1', page: parsed, throughIds: [] })
    expect(state.selected?.tail.todos?.[1]).toEqual({ content: 'Fix it', status: 'in_progress' })
  })
})

describe('streaming text and history reconciliation', () => {
  it('streams text_part into one assistant block and replaces it with history without a duplicate', () => {
    let state = selectedState()
    state = reduceHub(state, { type: 'events', events: [hubEvent('text_part', { text: 'Here is ' }), hubEvent('text_part', { text: 'the answer.' })] })
    const live = state.selected!.tail.blocks
    expect(live).toHaveLength(1)
    expect(live[0]).toMatchObject({ kind: 'assistant', text: 'Here is the answer.', streaming: true })
    const parsed = parseHistoryPage(page([message('0:message', 'user', 'q'), message('1:message', 'assistant', 'Here is the answer.')]))
    state = reduceHub(state, { type: 'historyLoaded', workspaceId: 'w1', sessionId: 's1', page: parsed, throughIds: live.map(block => block.id) })
    expect(state.selected!.tail.blocks).toHaveLength(0)
    expect(state.selected!.older.filter(block => block.kind === 'assistant')).toHaveLength(1)
  })

  it('drops passing notices once the turn ended and history is read (S13), keeping errors', () => {
    let state = selectedState()
    state = reduceHub(state, { type: 'events', events: [hubEvent('notification', { message: 'Compacted context' }), hubEvent('turn_end', {})] })
    expect(state.selected!.tail.blocks.some(block => block.kind === 'notice')).toBe(true)
    const parsed = parseHistoryPage(page([message('0:message', 'assistant', 'Final answer')]))
    state = reduceHub(state, { type: 'historyLoaded', workspaceId: 'w1', sessionId: 's1', page: parsed, throughIds: [] })
    expect(state.selected!.tail.blocks.some(block => block.kind === 'notice')).toBe(false)
  })
})

describe('resolutions and pending interactions', () => {
  it('Inbox: an approval_response for a count-only task clears Needs you (S4)', () => {
    const task = summary({ id: 's1', status: 'needs_you', pendingCount: 1, interactionIds: [] })
    const next = foldSummaryEvent('w1', task, hubEvent('approval_response', { request_id: 'r1', response: 'approve' }))
    expect(next).toMatchObject({ pendingCount: 0, status: 'working' })
  })

  it('Task: a question_response leaves Needs you for working when nothing else waits (S4)', () => {
    const start = { ...EMPTY_TAIL, status: 'needs_you' as const }
    const tail = foldAll(start, [
      { id: 'q', type: 'question_request', payload: { request_id: 'r1', questions: [{ id: 'q1', question: 'Which?', options: ['a', 'b'] }] } },
      { id: 'a', type: 'question_response', payload: { request_id: 'r1', answers: { q1: 'a' } } },
    ])
    expect(tail.pending).toEqual([])
    expect(tail.status).toBe('working')
  })

  it('says the waiting details are unavailable when nothing could be recovered (S8)', () => {
    let state = reduceHub(liveState([summary({ id: 's1', status: 'needs_you', pendingCount: 1 })]), { type: 'select', workspaceId: 'w1', sessionId: 's1' })
    expect(pendingDetailsUnavailable(state, state.selected!)).toBe(false)
    state = reduceHub(state, { type: 'pendingRestored', workspaceId: 'w1', sessionId: 's1', pending: [] })
    expect(pendingDetailsUnavailable(state, state.selected!)).toBe(true)
  })

  it('restores pending interactions served with hub.history, with no journal replay', async () => {
    const { h, store, controller } = await taskSetup(h => {
      h.hub.handlers.set('hub.history', () => ({
        ...page([message('0:message', 'user', 'q')]),
        pending_interactions: [{ type: 'approval_request', payload: { request_id: 'p1', tool_name: 'exec_command', arguments: { command: 'rm -rf build' } } }],
      }))
    })
    store.dispatch({ type: 'summaries', workspaceId: 'w1', tasks: [summary({ id: 's1', status: 'needs_you', pendingCount: 1 })] })
    controller.openSession('w1', 's1')
    await settle()
    expect(store.getSnapshot().selected?.tail.pending.map(item => [item.id, item.command])).toEqual([['p1', 'rm -rf build']])
    expect(h.hub.all('hub.events.subscribe')).toHaveLength(1)
    controller.dispose()
  })
})

describe('command answers stay with their task (S12)', () => {
  it('ignores a send result for a task that is no longer open', () => {
    let state = reduceHub(selectedState(), { type: 'select', workspaceId: 'w1', sessionId: 's2' })
    state = reduceHub(state, { type: 'send', state: { phase: 'failed', message: 'nope' }, workspaceId: 'w1', sessionId: 's1' })
    expect(state.selected?.send.phase).toBe('idle')
    state = reduceHub(state, { type: 'send', state: { phase: 'failed', message: 'mine' }, workspaceId: 'w1', sessionId: 's2' })
    expect(state.selected?.send).toEqual({ phase: 'failed', message: 'mine' })
  })
})

describe('agents of the open task', () => {
  it('loads the list on open, inspects one agent, closes it', async () => {
    const { h, store, controller } = await taskSetup(h => {
      h.hub.handlers.set('agents.list', () => ({ agents: [{ id: 'a1', name: 'reviewer', title: 'Review cancellation', status: 'running', tool_calls: 3, started_at: '2026-10-29T10:00:00Z' }, { name: 'no id' }] }))
      h.hub.handlers.set('agents.inspect', params => ({
        agent: { id: params.agent_id, name: 'reviewer', title: 'Review cancellation', status: 'running' },
        steps: [{ kind: 'tool', name: 'exec_command', args: 'bun test', output: '3 pass', state: 'done' }, { kind: 'text', text: 'All good.' }, { kind: 'bogus' }],
        truncated: true,
      }))
    })
    controller.openSession('w1', 's1')
    await settle()
    expect(h.hub.all('agents.list')[0]?.params).toEqual({ workspace_id: 'w1', session_id: 's1' })
    expect(store.getSnapshot().selected?.agents).toEqual({ phase: 'ready', rows: [{ id: 'a1', name: 'reviewer', title: 'Review cancellation', status: 'running', toolCalls: 3, startedAt: '2026-10-29T10:00:00Z' }] })
    await controller.inspectAgent('a1')
    expect(h.hub.all('agents.inspect')[0]?.params).toMatchObject({ workspace_id: 'w1', session_id: 's1', agent_id: 'a1' })
    const detail = store.getSnapshot().selected?.agentDetail
    expect(detail).toMatchObject({ phase: 'ready', agentId: 'a1', truncated: true })
    expect(detail?.steps).toEqual([{ kind: 'tool', name: 'exec_command', args: 'bun test', output: '3 pass', state: 'done' }, { kind: 'text', text: 'All good.' }])
    controller.closeAgent()
    expect(store.getSnapshot().selected?.agentDetail.phase).toBe('idle')
    controller.dispose()
  })

  it('refreshes at most once per two seconds while subagent events arrive', async () => {
    const { h, socket, controller } = await taskSetup()
    controller.openSession('w1', 's1')
    await settle()
    const count = h.hub.all('agents.list').length
    batch(socket, 'sub1', [ev(1, { type: 'subagent_event', payload: { agent_id: 'a1', event: { type: 'tool_call', payload: {} } } }), ev(2, { type: 'subagent_event', payload: { event: { type: 'text_part', payload: {} } } }), ev(3, { type: 'subagent_event', payload: {} })], 'c3')
    await settle()
    expect(h.hub.all('agents.list')).toHaveLength(count)
    h.timers.advance(AGENTS_REFRESH_MS)
    await settle()
    expect(h.hub.all('agents.list')).toHaveLength(count + 1)
    controller.dispose()
  })

  it('reports a Hub without agents.list as a readable error, not an empty list', async () => {
    const { store, controller } = await taskSetup(h => {
      h.hub.handlers.set('agents.list', () => ({ __error: { code: 'invalid_request', message: 'unknown hub method "agents.list"' } }))
      h.hub.handlers.set('agents.inspect', () => ({ __error: { code: 'not_found', message: 'no agent' } }))
    })
    controller.openSession('w1', 's1')
    await settle()
    expect(store.getSnapshot().selected?.agents).toEqual({ phase: 'error', rows: [], error: AGENTS_UNSUPPORTED })
    await controller.inspectAgent('gone')
    expect(store.getSnapshot().selected?.agentDetail).toMatchObject({ phase: 'error', agentId: 'gone', error: 'This agent is no longer known to the task.' })
    controller.dispose()
  })
})

describe('presence', () => {
  it('leaving a task says idle and stops its subscription (S5/UI-13)', async () => {
    const { h, controller } = await taskSetup()
    controller.openSession('w1', 's1')
    await settle()
    controller.closeSession()
    expect(h.hub.all('hub.presence.update').at(-1)?.params).toEqual({ workspace_id: 'w1', session_id: 's1', state: 'idle' })
    expect(h.hub.all('hub.events.unsubscribe').some(frame => frame.params.subscription_id === 'sub1')).toBe(true)
    const reads = h.hub.all('hub.history').length
    h.timers.advance(60_000)
    await settle()
    expect(h.hub.all('hub.history')).toHaveLength(reads)
    controller.dispose()
  })

  it('typing falls back to viewing after a quiet spell (S14)', async () => {
    const { h, controller } = await taskSetup()
    controller.openSession('w1', 's1')
    await settle()
    controller.setPresence('typing')
    controller.setPresence('typing')
    const states = () => h.hub.all('hub.presence.update').map(frame => frame.params.state)
    expect(states().filter(state => state === 'typing')).toHaveLength(1)
    h.timers.advance(TYPING_IDLE_MS)
    expect(states().at(-1)).toBe('viewing')
    controller.dispose()
  })
})

describe('reconnects and retries', () => {
  it('re-reads the XERXES thread after a reconnect: missed replies land and working clears (S2/XT-1)', async () => {
    const h = harness()
    let opens = 0
    h.hub.handlers.set('supervisor.thread.open', () => {
      opens += 1
      return opens === 1
        ? { thread_id: 'shared', busy: true, messages: [{ id: 'm1', role: 'user', text: 'status?', at: '2026-10-29T10:00:00Z' }], confirmations: [] }
        : { thread_id: 'shared', busy: false, messages: [{ id: 'm1', role: 'user', text: 'status?', at: '2026-10-29T10:00:00Z' }, { id: 'm2', role: 'xerxes', text: 'All green.', at: '2026-10-29T10:01:00Z' }], confirmations: [] }
    })
    const socket = await goLive(h)
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store, timers: h.timers })
    controller.start()
    await controller.supervisorOpen()
    expect(store.getSnapshot().supervisor.working).toBe(true)
    socket.drop(1006)
    h.timers.advance(10_000)
    await settle()
    h.hub.socket(h.hub.sockets.length - 1).open()
    await settle()
    await settle()
    expect(opens).toBe(2)
    expect(store.getSnapshot().supervisor.messages.map(m => m.id)).toEqual(['m1', 'm2'])
    expect(store.getSnapshot().supervisor.working).toBe(false)
    controller.dispose()
  })

  it('retries a failed Inbox subscribe instead of leaving live updates dead (S3)', async () => {
    const h = harness()
    let subscribes = 0
    h.hub.handlers.set('hub.workspaces.list', () => ({ workspaces: [{ id: 'w1', label: 'Mac', health: 'online' }] }))
    h.hub.handlers.set('hub.events.read', () => ({ watermark: 'W' }))
    h.hub.handlers.set('hub.snapshot', () => ({ workspaces: [{ id: 'w1', sessions: [] }] }))
    h.hub.handlers.set('hub.events.subscribe', () => (++subscribes === 1
      ? { __error: { code: 'unavailable', message: 'busy' } }
      : { subscription_id: `sub${subscribes}`, watermark: 'W', oldest_cursor: 'O', reset_required: false }))
    await goLive(h)
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store, timers: h.timers })
    await controller.refreshInbox()
    await settle()
    expect(store.getSnapshot().inboxError).toContain('Live updates paused')
    h.timers.advance(1_000)
    await settle()
    await settle()
    expect(subscribes).toBe(2)
    expect(store.getSnapshot().inboxError).toBeUndefined()
    controller.dispose()
  })
})

describe('XERXES chat', () => {
  it('a retry after an uncertain failure reuses the message id and starts no second turn (S11/XT-6)', async () => {
    const h = harness()
    let calls = 0
    h.hub.handlers.set('supervisor.message.submit', params => (++calls === 1 ? undefined : { thread_id: 'shared', duplicate: true, messages: [{ id: params.message_id, role: 'user', text: 'hi', at: '2026-10-29T10:00:00Z' }] }))
    await goLive(h)
    const store = createHubStore()
    const controller = createHubController({ connection: h.conn, store, timers: h.timers })
    const first = controller.supervisorSend('hi')
    await settle()
    h.timers.advance(15_000)
    expect(await first).toBe(false)
    expect(store.getSnapshot().supervisor.error).toBeDefined()
    expect(await controller.supervisorSend('hi')).toBe(true)
    const ids = h.hub.all('supervisor.message.submit').map(frame => frame.params.message_id)
    expect(ids[0]).toBe(ids[1])
    expect(store.getSnapshot().supervisor.working).not.toBe(true)
    expect(store.getSnapshot().supervisor.error).toBeUndefined()
    controller.dispose()
  })

  it('clears a stale error banner when a new send starts (XT-5)', () => {
    let state = reduceHub(liveState(), { type: 'supervisor', patch: { error: 'old problem' } })
    state = reduceHub(state, { type: 'supervisor', patch: { busy: true } })
    expect(state.supervisor.error).toBeUndefined()
  })

  it('gives the buttons back after a transient refusal and never marks an unknown outcome failed (XT-7)', async () => {
    const h = harness()
    let respond = 0
    h.hub.handlers.set('supervisor.confirmation.respond', () => (++respond === 1 ? { __error: { code: 'unavailable', message: 'Machine offline.' } } : undefined))
    h.hub.handlers.set('supervisor.action.status', () => ({ __error: { code: 'unavailable', message: 'still offline' } }))
    await goLive(h)
    const store = createHubStore()
    const card: SupervisorConfirmation = { id: 'c1', tool: 'task.steer', summary: 'Steer', targets: [], state: 'pending' }
    store.dispatch({ type: 'supervisorConfirmations', confirmations: [card] })
    const controller = createHubController({ connection: h.conn, store, timers: h.timers })
    await controller.supervisorRespond('c1', 'confirm')
    expect(store.getSnapshot().supervisor.confirmations[0]).toMatchObject({ state: 'pending' })
    const second = controller.supervisorRespond('c1', 'confirm')
    await settle()
    h.timers.advance(15_000)
    await second
    expect(store.getSnapshot().supervisor.confirmations[0]?.state).toBe('confirming')
    controller.dispose()
  })
})

describe('drafts', () => {
  function fakeStorage(): DraftStorage & { data: Map<string, string>; full: boolean } {
    const data = new Map<string, string>()
    const storage = {
      data, full: false,
      getItem: (key: string) => data.get(key) ?? null,
      setItem: (key: string, value: string) => { if (storage.full) throw new Error('QuotaExceededError'); data.set(key, value) },
      removeItem: (key: string) => { data.delete(key) },
    }
    return storage
  }

  it('survive the app being evicted, stay bounded, and tolerate a full storage (S9/UI-05)', () => {
    const storage = fakeStorage()
    const drafts = createDraftStore(2, { storage, maxLength: 10 })
    drafts.set('a', 'one')
    drafts.set('b', 'a very long draft text')
    drafts.set('c', 'three')
    const restored = createDraftStore(2, { storage })
    expect([restored.get('a'), restored.get('b'), restored.get('c')]).toEqual(['', 'a very lon', 'three'])
    storage.full = true
    restored.set('c', 'changed')
    expect(restored.get('c')).toBe('changed')
    storage.data.set('xerxes.drafts.v1', '{broken')
    expect(createDraftStore(2, { storage }).get('c')).toBe('')
  })

  it('keeps text typed while a send was in flight (UI-02)', () => {
    expect(draftAfterAccepted('hello', 'hello')).toBe('')
    expect(draftAfterAccepted('hello and more', 'hello')).toBe('and more')
    expect(draftAfterAccepted('different', 'hello')).toBe('different')
  })
})
