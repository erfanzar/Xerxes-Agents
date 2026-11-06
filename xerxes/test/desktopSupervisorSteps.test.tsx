// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { parseSupervisorStep, type SupervisorMessage, type SupervisorStep } from '../src/client/hubParsers.js'
import { initialHubState, reduceHub, withStep } from '../src/client/hubStore.js'
import { liveSteps, stepPhrase, stepsFinished } from '../src/client/supervisorSteps.js'
import { LiveSteps, ThreadMessages } from '../src/desktop/renderer/SupervisorThread.js'

const at = (second: number): string => `2026-11-06T10:00:${String(second).padStart(2, '0')}.000Z`
const step = (id: string, patch: Partial<SupervisorStep> = {}): SupervisorStep => ({
  id, turnId: 't1', kind: 'tool', text: 'Listing workspaces', state: 'running', at: at(1), ...patch,
})
const plain = (html: string): string => html.replaceAll('<!-- -->', '')

test('a wire step parses; malformed ones are dropped and an unknown state reads as done', () => {
  expect(parseSupervisorStep({ id: 's1', turn_id: 't1', kind: 'tool', tool: 'tasks_list', text: 'Listing tasks', state: 'running', at: at(1) }))
    .toEqual({ id: 's1', turnId: 't1', kind: 'tool', tool: 'tasks_list', text: 'Listing tasks', state: 'running', at: at(1) })
  expect(parseSupervisorStep({ id: 's1', turn_id: 't1', kind: 'say', text: 'Checking.', state: 'weird' })?.state).toBe('done')
  for (const bad of [null, {}, { id: 's', turn_id: 't', kind: 'tool', text: '' }, { id: 's', kind: 'tool', text: 'x' }, { id: 's', turn_id: 't', kind: 'shell', text: 'x' }]) {
    expect(parseSupervisorStep(bad)).toBeUndefined()
  }
})

test('a tool step replaces its own running state; a new turn id starts a fresh list', () => {
  let steps = withStep(undefined, step('a'))
  steps = withStep(steps, step('a', { state: 'done' }))
  steps = withStep(steps, step('b', { text: 'Searching every task' }))
  expect(steps.map(s => [s.id, s.state])).toEqual([['a', 'done'], ['b', 'running']])
  const next = withStep(steps, step('c', { turnId: 't2' }))
  expect(next.map(s => s.id)).toEqual(['c'])
})

test('the store routes steps by thread and keeps them through working on/off', () => {
  let state = reduceHub(initialHubState(), { type: 'supervisorStep', thread: 'shared', step: step('a') })
  state = reduceHub(state, { type: 'supervisor', patch: { working: true } })
  state = reduceHub(state, { type: 'supervisorStep', thread: 'autopilot', step: step('z', { turnId: 'w1' }) })
  expect(state.supervisor.steps?.map(s => s.id)).toEqual(['a'])
  expect(state.autopilotThread.steps?.map(s => s.id)).toEqual(['z'])
  state = reduceHub(state, { type: 'supervisor', patch: { working: false } })
  expect(state.supervisor.steps?.map(s => s.id)).toEqual(['a'])
})

test('steps are live until a XERXES reply or a stop notice lands after them; mid-turn notices do not end them', () => {
  const steps = [step('a', { state: 'done' }), step('b', { at: at(3) })]
  const notice: SupervisorMessage = { id: 'n', role: 'system', text: 'XERXES did: task.steer', at: at(4), source: 'text' }
  const older: SupervisorMessage = { id: 'o', role: 'xerxes', text: 'old answer', at: at(0), source: 'text' }
  expect(stepsFinished(steps, [older, notice])).toBe(false)
  expect(liveSteps(steps, [older, notice])).toHaveLength(2)
  expect(stepsFinished(steps, [{ id: 'r', role: 'xerxes', text: 'Done.', at: at(5), source: 'text' }])).toBe(true)
  expect(stepsFinished(steps, [{ id: 's', role: 'system', text: 'XERXES stopped: timeout', at: at(5), source: 'text' }])).toBe(true)
  expect(liveSteps(undefined, [])).toEqual([])
})

test('the status phrase names the running tool, else the last narration line', () => {
  expect(stepPhrase([])).toBeUndefined()
  expect(stepPhrase([step('s', { kind: 'say', text: 'Checking every workspace.', state: 'done' })])).toBe('Checking every workspace.')
  expect(stepPhrase([step('s', { kind: 'say', text: 'Checking.', state: 'done' }), step('a')])).toBe('Listing workspaces…')
  expect(stepPhrase([step('a', { state: 'done' })])).toBe('Working…')
})

test('live: the status line shows what runs now, then every step with its state', () => {
  const html = plain(renderToStaticMarkup(createElement(LiveSteps, {
    steps: [step('s', { kind: 'say', text: 'Checking every workspace first.', state: 'done' }), step('a', { state: 'done' }), step('b', { text: 'Searching every task for “shader”' })],
    seconds: 4,
  })))
  expect(html).toContain('role="status" aria-live="polite"')
  expect(html).toContain('>Searching every task for “shader”…<')
  expect(html).toContain('class="xstep xstep--say">Checking every workspace first.<')
  expect(html).toContain('class="xstep" data-state="done"')
  expect(html).toContain('class="xstep" data-state="running"')
  expect(html).toContain('4s')
  // Before the first step it still says something.
  expect(plain(renderToStaticMarkup(createElement(LiveSteps, { steps: [] })))).toContain('>Thinking…<')
})

test('finished: the steps fold above the reply that ended them, as "Ran N steps"', () => {
  const messages: SupervisorMessage[] = [
    { id: 'u', role: 'user', text: 'status?', at: at(0), source: 'text' },
    { id: 'r', role: 'xerxes', text: 'All good.', at: at(9), source: 'text' },
  ]
  const steps = [step('a', { state: 'done' }), step('b', { state: 'error', at: at(2) })]
  const html = plain(renderToStaticMarkup(createElement(ThreadMessages, { messages, steps, now: Date.parse(at(30)), label: 'Conversation' })))
  expect(html).toContain('<summary>Ran 2 steps · 1 failed</summary>')
  expect(html.indexOf('Ran 2 steps')).toBeGreaterThan(html.indexOf('status?'))
  expect(html.indexOf('Ran 2 steps')).toBeLessThan(html.indexOf('All good.'))
  // While the turn still runs, nothing folds into the thread.
  const running = plain(renderToStaticMarkup(createElement(ThreadMessages, { messages: messages.slice(0, 1), steps, now: Date.parse(at(30)), label: 'Conversation' })))
  expect(running).not.toContain('Ran 2 steps')
})

test('step text from the Hub is rendered as text, never HTML', () => {
  const html = renderToStaticMarkup(createElement(LiveSteps, { steps: [step('x', { text: '<img src=x onerror=alert(1)>' })] }))
  expect(html).not.toContain('<img')
  expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;')
})
