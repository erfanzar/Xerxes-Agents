// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { createHubController, HISTORY_REFRESH_MS } from '../../client/hubController.js'
import { parseHistoryPage } from '../../client/hubParsers.js'
import { createHubStore, reduceHub } from '../../client/hubStore.js'
import { batch, ev, goLive, harness, settle } from './fakes.js'
import { liveState, summary } from './helpers.js'

const message = (id: string, role: string, content: string) => ({ id, messages: [{ role, content }], executions: [], thinking: [] })
const tool = (id: string, callId: string, result?: string) => ({
  id, messages: [
    { role: 'assistant', content: '', tool_calls: [{ id: callId, function: { name: 'exec_command', arguments: { command: 'bun test' } } }] },
    ...(result === undefined ? [] : [{ role: 'tool', tool_call_id: callId, content: result }]),
  ], executions: [], thinking: [],
})
const page = (actions: readonly unknown[], before: string | null = null) => ({ history: { actions, before, has_more: before !== null } })

async function setup() {
  const h = harness()
  h.hub.handlers.set('hub.events.read', () => ({ watermark: 'live-edge' }))
  h.hub.handlers.set('hub.presence.update', () => ({}))
  const socket = await goLive(h)
  const store = createHubStore(liveState([summary({ id: 's1', status: 'working' })]))
  const controller = createHubController({ connection: h.conn, store, timers: h.timers })
  return { h, socket, store, controller }
}

const visible = (store: ReturnType<typeof createHubStore>) => {
  const selected = store.getSnapshot().selected
  return [...selected?.older ?? [], ...selected?.tail.blocks ?? []]
}

describe('task history-first controller', () => {
  it('shows loading, asks for 30 actions first, renders newest last, then subscribes at the live edge', async () => {
    const { h, socket, store, controller } = await setup()
    h.hub.handlers.set('hub.history', () => page([message('0:message', 'user', 'question'), message('1:message', 'assistant', 'answer')], 'older'))
    controller.openSession('w1', 's1')
    expect(store.getSnapshot().selected?.phase).toBe('loading')
    expect(socket.sent.at(-2)?.method).toBe('hub.history')
    expect(h.hub.all('hub.history')[0]?.params).toEqual({ workspace_id: 'w1', session_id: 's1', limit: 30 })
    expect(h.hub.all('hub.events.subscribe')).toHaveLength(0)
    await settle()
    expect(store.getSnapshot().selected).toMatchObject({ phase: 'ready', hasOlder: true, olderCursor: 'older' })
    expect(visible(store).map(block => 'text' in block ? block.text : block.kind)).toEqual(['question', 'answer'])
    expect(h.hub.all('hub.events.subscribe')).toHaveLength(1)
    expect(h.hub.all('hub.events.subscribe')[0]?.params.after_cursor).toBe('live-edge')
    controller.dispose()
  })

  it('prepends one older page, guards overlapping loads, and stops at has_more:false', async () => {
    const { h, store, controller } = await setup()
    h.hub.handlers.set('hub.history', params => params.before === undefined
      ? page([message('10:message', 'assistant', 'newest')], 'before-10')
      : page([message('0:message', 'user', 'oldest'), message('1:message', 'assistant', 'older')]))
    controller.openSession('w1', 's1')
    await settle()
    const count = h.hub.all('hub.history').length
    await Promise.all([controller.loadOlder(), controller.loadOlder()])
    expect(h.hub.all('hub.history')).toHaveLength(count + 1)
    expect(h.hub.all('hub.history').at(-1)?.params.before).toBe('before-10')
    expect(visible(store).map(block => 'text' in block ? block.text : '')).toEqual(['oldest', 'older', 'newest'])
    expect(store.getSnapshot().selected).toMatchObject({ hasOlder: false, loadingOlder: false })
    await controller.loadOlder()
    expect(h.hub.all('hub.history')).toHaveLength(count + 1)
    controller.dispose()
  })

  it('reads assistant text at turn_end and replaces covered live steps without duplicating tools', async () => {
    const { h, socket, store, controller } = await setup()
    let actions: unknown[] = [message('0:message', 'user', 'question')]
    h.hub.handlers.set('hub.history', () => page(actions))
    controller.openSession('w1', 's1')
    await settle()
    batch(socket, 'sub1', [ev(1, { type: 'tool_call', payload: { id: 'call-1', name: 'exec_command' } })], 'c1')
    expect(visible(store).filter(block => block.kind === 'tool')).toHaveLength(1)
    actions = [...actions, tool('1:tool:0', 'call-1', '2 pass'), message('2:message', 'assistant', 'Here is the actual reply.')]
    batch(socket, 'sub1', [ev(2, { type: 'tool_result', payload: { tool_call_id: 'call-1', result: '2 pass', permitted: true } }), ev(3, { type: 'turn_end', payload: {} })], 'c3')
    await settle()
    expect(visible(store).filter(block => block.kind === 'tool')).toHaveLength(1)
    expect(visible(store).at(-1)).toMatchObject({ kind: 'assistant', text: 'Here is the actual reply.' })
    expect(store.getSnapshot().selected?.tail.blocks).toHaveLength(0)
    controller.dispose()
  })

  it('coalesces busy tool activity into a four-second refresh, keeping newer in-flight tools', async () => {
    const { h, socket, store, controller } = await setup()
    let actions: unknown[] = [message('0:message', 'user', 'question')]
    h.hub.handlers.set('hub.history', () => page(actions))
    controller.openSession('w1', 's1')
    await settle()
    const count = h.hub.all('hub.history').length
    batch(socket, 'sub1', [ev(1, { type: 'tool_call', payload: { id: 'call-1', name: 'exec_command' } })], 'c1')
    h.timers.advance(HISTORY_REFRESH_MS - 1)
    expect(h.hub.all('hub.history')).toHaveLength(count)
    actions = [...actions, tool('1:tool:0', 'call-1'), message('2:message', 'assistant', 'Progress report')]
    h.timers.advance(1)
    batch(socket, 'sub1', [ev(2, { type: 'tool_call', payload: { id: 'call-2', name: 'ReadFile' } })], 'c2')
    await settle()
    expect(h.hub.all('hub.history')).toHaveLength(count + 1)
    expect(visible(store).filter(block => block.kind === 'tool').map(block => block.callId)).toEqual(['call-1', 'call-2'])
    expect(visible(store).some(block => block.kind === 'assistant' && block.text === 'Progress report')).toBe(true)
    batch(socket, 'sub1', [ev(3, { type: 'tool_result', payload: { tool_call_id: 'call-1', permitted: false, result: 'Denied' } })], 'c3')
    expect(visible(store).find(block => block.kind === 'tool' && block.callId === 'call-1')).toMatchObject({ state: 'error', output: 'Denied' })
    controller.dispose()
    h.timers.advance(HISTORY_REFRESH_MS)
    expect(h.hub.all('hub.history')).toHaveLength(count + 1)
  })

  it('queues a final read when turn_end arrives during an in-flight refresh', async () => {
    const { h, socket, store, controller } = await setup()
    h.hub.handlers.set('hub.history', () => page([message('0:message', 'user', 'question')]))
    controller.openSession('w1', 's1')
    await settle()
    h.hub.handlers.set('hub.history', () => undefined)
    batch(socket, 'sub1', [ev(1, { type: 'tool_call', payload: { id: 'call-1' } })], 'c1')
    h.timers.advance(HISTORY_REFRESH_MS)
    const delayed = h.hub.all('hub.history').at(-1)!
    batch(socket, 'sub1', [ev(2, { type: 'turn_end', payload: {} })], 'c2')
    h.hub.handlers.set('hub.history', () => page([message('0:message', 'user', 'question'), tool('1:tool:0', 'call-1', 'ok'), message('2:message', 'assistant', 'Final reply')]))
    socket.respond(delayed.id, page([message('0:message', 'user', 'question')]))
    await settle()
    expect(h.hub.all('hub.history').at(-1)?.id).not.toBe(delayed.id)
    expect(visible(store).at(-1)).toMatchObject({ kind: 'assistant', text: 'Final reply' })
    controller.dispose()
  })

  it('makes history failures visible and ignores responses after switching tasks', async () => {
    const { h, socket, store, controller } = await setup()
    h.hub.handlers.set('hub.history', () => ({ __error: { code: 'unavailable', message: 'History unavailable' } }))
    controller.openSession('w1', 's1')
    await settle()
    expect(store.getSnapshot().selected).toMatchObject({ phase: 'error', error: 'History unavailable' })
    h.hub.handlers.set('hub.history', () => undefined)
    controller.openSession('w1', 's1')
    const old = h.hub.all('hub.history').at(-1)!
    h.hub.handlers.set('hub.history', () => page([message('0:message', 'assistant', 'other task')]))
    controller.openSession('w1', 's2')
    socket.respond(old.id, page([message('0:message', 'assistant', 'wrong task')]))
    await settle()
    expect(visible(store).at(-1)).toMatchObject({ text: 'other task' })
    controller.dispose()
  })

  it('refreshes on gaps and resumes a watermark, never a full-journal replay', async () => {
    const { h, socket, store, controller } = await setup()
    h.hub.handlers.set('hub.history', () => page([message('0:message', 'assistant', 'initial')]))
    controller.openSession('w1', 's1')
    await settle()
    h.hub.handlers.set('hub.history', () => page([message('10:message', 'assistant', 'after gap')]))
    socket.event('hub.stream_gap', { subscription_id: 'sub1', reason: 'retention', watermark: 'gap-edge' })
    await settle()
    expect(visible(store).map(block => 'text' in block ? block.text : '')).toEqual(['after gap'])
    expect(h.hub.all('hub.events.subscribe').map(frame => frame.params.after_cursor)).toEqual(['live-edge', 'live-edge'])
    controller.dispose()
  })

  it('shows reset failures and refuses to resubscribe without a safe cursor', async () => {
    const { h, socket, store, controller } = await setup()
    h.hub.handlers.set('hub.history', () => page([message('0:message', 'assistant', 'initial')]))
    controller.openSession('w1', 's1')
    await settle()
    h.hub.handlers.set('hub.events.read', () => ({ __error: { code: 'unavailable', message: 'No live boundary' } }))
    socket.event('hub.stream_gap', { subscription_id: 'sub1', reason: 'retention' })
    await settle()
    expect(store.getSnapshot().selected?.error).toBe('No live boundary')
    expect(h.hub.all('hub.events.subscribe')).toHaveLength(1)
    expect(visible(store).at(-1)).toMatchObject({ text: 'initial' })
    controller.dispose()
  })

  it('recovers pre-existing waiting interactions separately, without replaying tools/text into the view or resurrecting live resolutions', async () => {
    const { h, socket, store, controller } = await setup()
    store.dispatch({ type: 'summaries', workspaceId: 'w1', tasks: [summary({ id: 's1', status: 'needs_you', pendingCount: 2 })] })
    h.hub.handlers.set('hub.history', () => page([message('0:message', 'assistant', 'Latest history only')]))
    controller.openSession('w1', 's1')
    await settle()
    expect(h.hub.all('hub.events.subscribe')).toHaveLength(2)
    batch(socket, 'sub1', [ev(20, { type: 'permission_resolved', payload: { request_id: 'already-done' } })], 'c20')
    batch(socket, 'sub2', [
      ev(1, { type: 'text_part', payload: { text: 'Never replay this old text' } }),
      ev(2, { type: 'tool_call', payload: { id: 'old-tool', name: 'Bash' } }),
      ev(3, { type: 'approval_request', payload: { id: 'already-done', tool_name: 'Bash' } }),
      ev(4, { type: 'question_request', payload: { id: 'waiting', questions: [{ id: 'q1', question: 'Choose?', options: ['yes', 'no'] }] } }),
    ], 'W')
    await settle()
    expect(visible(store)).toHaveLength(1)
    expect(visible(store)[0]).toMatchObject({ text: 'Latest history only' })
    expect(store.getSnapshot().selected?.tail.pending.map(item => item.id)).toEqual(['waiting'])
    expect(h.hub.all('hub.events.unsubscribe').some(frame => frame.params.subscription_id === 'sub2')).toBe(true)
    controller.dispose()
  })

  it('keeps older-history errors retryable without discarding the conversation', async () => {
    const { h, store, controller } = await setup()
    h.hub.handlers.set('hub.history', params => params.before === undefined
      ? page([message('10:message', 'assistant', 'newest')], 'before-10')
      : { __error: { code: 'unavailable', message: 'Try again later' } })
    controller.openSession('w1', 's1')
    await settle()
    await controller.loadOlder()
    expect(store.getSnapshot().selected).toMatchObject({ hasOlder: true, loadingOlder: false, olderError: 'Try again later' })
    expect(visible(store).at(-1)).toMatchObject({ text: 'newest' })
    h.hub.handlers.set('hub.history', () => page([message('0:message', 'user', 'oldest')]))
    await controller.loadOlder()
    expect(store.getSnapshot().selected?.olderError).toBeUndefined()
    expect(visible(store)).toHaveLength(2)
    controller.dispose()
  })
})

describe('authoritative history windows', () => {
  it('keeps loaded earlier pages, replaces changed action contents, and bounds memory', () => {
    let state = reduceHub(liveState(), { type: 'select', workspaceId: 'w1', sessionId: 's1' })
    const load = (actions: unknown[], before: string | null = null) => {
      state = reduceHub(state, { type: 'historyLoaded', workspaceId: 'w1', sessionId: 's1', page: parseHistoryPage(page(actions, before)), throughIds: [] })
    }
    load([message('10:message', 'assistant', 'partial')], 'before-10')
    state = reduceHub(state, { type: 'olderLoaded', blocks: parseHistoryPage(page([message('0:message', 'user', 'old')])).blocks, hasMore: false })
    load([message('10:message', 'assistant', 'complete'), message('11:message', 'assistant', 'new')], 'before-10')
    expect(state.selected?.older.map(block => 'text' in block ? block.text : '')).toEqual(['old', 'complete', 'new'])
    expect(state.selected?.hasOlder).toBe(false)
    load(Array.from({ length: 1000 }, (_, i) => message(`${i}:message`, 'assistant', String(i))))
    expect(state.selected?.older.length).toBeLessThanOrEqual(360)
    expect(state.selected?.historyActionIds.length).toBeLessThanOrEqual(360)
  })

  it('rejects unreadable pages rather than treating missing history as an empty conversation', () => {
    expect(() => parseHistoryPage({ history: null })).toThrow('did not return a history page')
    expect(parseHistoryPage(page([]))).toMatchObject({ blocks: [], actionIds: [], hasMore: false })
  })

  it('parses actual daemon tool-call messages with no executions and preserves full assistant text', () => {
    const text = 'answer '.repeat(2000)
    const parsed = parseHistoryPage(page([tool('0:tool:0', 'call-1', 'Tool execution failed: bad'), message('1:message', 'assistant', text)]))
    expect(parsed.blocks[0]).toMatchObject({ kind: 'tool', name: 'exec_command', callId: 'call-1', summary: 'bun test', state: 'error' })
    expect(parsed.blocks.at(-1)).toMatchObject({ kind: 'assistant', text })
  })
})
