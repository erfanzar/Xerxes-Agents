// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { classifyInput, shouldQueue } from '../app/slash.js'
import { decideSubmit, dequeue, enqueue, replaceLast } from '../app/queue.js'
import { executePlan, planSubmit, respondApproval, respondQuestion, type ClientLike } from '../app/controller.js'
import { findSlashCommand } from '../app/slash/registry.js'
import { appendHistory, HistoryCursor, loadHistory } from '../lib/history.js'
import { initialState, reduce, type WireEvt } from '../app/gatewayState.js'

describe('classifyInput', () => {
  it('routes plain text to message', () => {
    expect(classifyInput('hello there')).toEqual({ kind: 'message', text: 'hello there' })
  })
  it('routes !cmd to shell', () => {
    expect(classifyInput('!ls -la')).toEqual({ kind: 'shell', command: 'ls -la' })
  })
  it('handles client-local commands + aliases', () => {
    expect(classifyInput('/quit').kind).toBe('exit')
    expect(classifyInput('/q').kind).toBe('exit')
    expect(classifyInput('/clear').kind).toBe('clear')
    expect(classifyInput('/details expanded')).toEqual({ kind: 'details', arg: 'expanded' })
  })
  it('falls unknown slashes through to remote', () => {
    expect(classifyInput('/skills')).toEqual({ kind: 'remote', command: '/skills' })
    expect(classifyInput('/model gpt')).toEqual({ kind: 'remote', command: '/model gpt' })
    expect(classifyInput('/compact')).toEqual({ kind: 'remote', command: '/compact' })
  })
  it('/compact is not a client-local display toggle', () => {
    expect(findSlashCommand('compact')).toBeUndefined()
    expect(findSlashCommand('ui-compact')?.help).toBe('toggle compact transcript display')
  })
  it('empty → noop, only messages queue', () => {
    expect(classifyInput('   ').kind).toBe('noop')
    expect(shouldQueue(classifyInput('hi'))).toBe(true)
    expect(shouldQueue(classifyInput('/skills'))).toBe(false)
  })
})

describe('queue ops + decideSubmit', () => {
  it('enqueue trims and skips empties', () => {
    expect(enqueue(['a'], '  b ')).toEqual(['a', 'b'])
    expect(enqueue(['a'], '   ')).toEqual(['a'])
  })
  it('dequeue pulls the oldest', () => {
    expect(dequeue(['a', 'b'])).toEqual({ next: 'a', rest: ['b'] })
    expect(dequeue([])).toEqual({ next: undefined, rest: [] })
  })
  it('replaceLast swaps the newest', () => {
    expect(replaceLast(['a', 'b'], 'c')).toEqual(['a', 'c'])
  })
  it('decides send/queue/interrupt/drain', () => {
    expect(decideSubmit('hi', false, 0)).toEqual({ kind: 'send', text: 'hi' })
    expect(decideSubmit('hi', true, 0)).toEqual({ kind: 'queue', text: 'hi' })
    expect(decideSubmit('', true, 2).kind).toBe('interrupt')
    expect(decideSubmit('', false, 2).kind).toBe('drain')
    expect(decideSubmit('', false, 0).kind).toBe('noop')
  })
})

describe('planSubmit (pure)', () => {
  it('sends when idle, queues when busy', () => {
    expect(planSubmit('hi', false, 0)).toEqual({ do: 'send', text: 'hi' })
    expect(planSubmit('hi', true, 0)).toEqual({ do: 'queue', text: 'hi' })
  })
  it('maps empty-Enter to interrupt/drain by busy state', () => {
    expect(planSubmit('', true, 1)).toEqual({ do: 'interrupt' })
    expect(planSubmit('', false, 1)).toEqual({ do: 'drain' })
    expect(planSubmit('', false, 0)).toEqual({ do: 'noop' })
  })
  it('routes slash + shell', () => {
    expect(planSubmit('/skills', false, 0)).toEqual({ do: 'remote', command: '/skills' })
    expect(planSubmit('!pwd', false, 0)).toEqual({ do: 'shell', command: 'pwd' })
    expect(planSubmit('/quit', false, 0)).toEqual({ do: 'exit' })
  })
})

describe('executePlan (fake client)', () => {
  let calls: Array<{ method: string; params?: Record<string, unknown> }>
  let dispatched: WireEvt[]
  let client: ClientLike

  beforeEach(() => {
    calls = []
    dispatched = []
    client = {
      sessionKey: 'tui:test',
      request: (method, params) => {
        calls.push({ method, params })
        return Promise.resolve({ ok: true })
      },
      stderrSnapshot: () => 'last log line'
    }
  })
  const dispatch = (e: unknown) => dispatched.push(e as WireEvt)

  it('send issues prompt + optimistic user echo', () => {
    executePlan(client, dispatch, { do: 'send', text: 'hi' }, { queue: [] })
    expect(calls[0]!.method).toBe('prompt')
    expect(calls[0]!.params).toMatchObject({ user_input: 'hi', session_key: 'tui:test' })
    expect(dispatched.some(e => e.type === '__user')).toBe(true)
  })
  it('queue dispatches enqueue, no RPC', () => {
    executePlan(client, dispatch, { do: 'queue', text: 'later' }, { queue: ['a'] })
    expect(calls).toHaveLength(0)
    expect(dispatched.some(e => e.type === '__enqueue')).toBe(true)
  })
  it('interrupt calls cancel_all', () => {
    executePlan(client, dispatch, { do: 'interrupt' }, { queue: [] })
    expect(calls[0]!.method).toBe('cancel_all')
  })
  it('drain sends the head of the queue', () => {
    executePlan(client, dispatch, { do: 'drain' }, { queue: ['first', 'second'] })
    expect(dispatched.some(e => e.type === '__dequeue')).toBe(true)
    expect(calls[0]!.params).toMatchObject({ user_input: 'first' })
  })
  it('remote + shell + help go through slash', () => {
    executePlan(client, dispatch, { do: 'remote', command: '/skills' }, { queue: [] })
    executePlan(client, dispatch, { do: 'shell', command: 'ls' }, { queue: [] })
    executePlan(client, dispatch, { do: 'help' }, { queue: [] })
    expect(calls.map(c => c.params!.command)).toEqual(['/skills', '!ls', '/help'])
  })
  it('clear both wipes locally and resets the daemon session', () => {
    executePlan(client, dispatch, { do: 'clear' }, { queue: [] })
    expect(dispatched.some(e => e.type === '__clear')).toBe(true)
    expect(calls[0]!.params).toMatchObject({ command: '/clear' })
  })
  it('exit returns exit:true', () => {
    expect(executePlan(client, dispatch, { do: 'exit' }, { queue: [] }).exit).toBe(true)
  })

  it('respondApproval / respondQuestion call the right RPCs and clear pending', () => {
    respondApproval(client, dispatch, 'req1', 'approve_for_session')
    expect(calls[0]).toMatchObject({
      method: 'permission_response',
      params: { request_id: 'req1', response: 'approve_for_session' }
    })
    expect(dispatched.some(e => e.type === '__approval_done')).toBe(true)

    respondQuestion(client, dispatch, 'req2', { q1: 'yes' })
    expect(calls[1]).toMatchObject({
      method: 'question_response',
      params: { request_id: 'req2', answers: { q1: 'yes' } }
    })
    expect(dispatched.some(e => e.type === '__question_done')).toBe(true)
  })
})

describe('prompt-flow reducer state', () => {
  it('approval_request populates pendingApproval; __approval_done clears it', () => {
    const open = reduce(initialState, {
      type: 'approval_request',
      payload: { id: 'a1', tool_call_id: 'tc1', action: 'exec_command', description: 'rm -rf' }
    })
    expect(open.pendingApproval).toMatchObject({ id: 'a1', action: 'exec_command' })
    const done = reduce(open, { type: '__approval_done', payload: {} })
    expect(done.pendingApproval).toBeNull()
  })
  it('question_request populates pendingQuestion with items', () => {
    const open = reduce(initialState, {
      type: 'question_request',
      payload: {
        id: 'q1',
        tool_call_id: 'tc1',
        questions: [{ id: 'x', question: 'pick', options: ['a', 'b'], allow_free_form: false }]
      }
    })
    expect(open.pendingQuestion?.questions).toHaveLength(1)
    expect(reduce(open, { type: '__question_done', payload: {} }).pendingQuestion).toBeNull()
  })
  it('queue synthetics push/pop', () => {
    const a = reduce(initialState, { type: '__enqueue', payload: { text: 'one' } })
    const b = reduce(a, { type: '__enqueue', payload: { text: 'two' } })
    expect(b.queue).toEqual(['one', 'two'])
    expect(reduce(b, { type: '__dequeue', payload: {} }).queue).toEqual(['two'])
  })
})

describe('input history', () => {
  let dir: string
  let path: string
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'xerxes-hist-'))
    path = join(dir, '.tui_history')
  })
  afterEach(() => rmSync(dir, { recursive: true, force: true }))

  it('appends and loads, flattening newlines', () => {
    appendHistory('first', path)
    appendHistory('second\nline', path)
    expect(loadHistory(path)).toEqual(['first', 'second line'])
    expect(readFileSync(path, 'utf8')).toContain('second line')
  })
  it('cursor walks prev/next and returns to the live draft', () => {
    const cur = new HistoryCursor(['a', 'b', 'c'])
    expect(cur.atLive()).toBe(true)
    expect(cur.prev()).toBe('c')
    expect(cur.prev()).toBe('b')
    expect(cur.next()).toBe('c')
    expect(cur.next()).toBe('') // back on the live draft
    expect(cur.atLive()).toBe(true)
  })
  it('push de-dupes consecutive repeats and resets the cursor', () => {
    const cur = new HistoryCursor(['a'])
    cur.push('a') // dup, ignored
    cur.push('b')
    expect(cur.prev()).toBe('b')
    expect(cur.prev()).toBe('a')
  })
})

// keep vi import used even if a future test stubs timers
vi.useRealTimers?.()
