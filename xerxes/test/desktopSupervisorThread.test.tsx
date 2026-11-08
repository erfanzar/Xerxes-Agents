// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import type { SupervisorConfirmation, SupervisorMessage } from '../src/client/hubParsers.js'
import { isOpenCard, withRecords } from '../src/client/supervisorRecords.js'
import { ThreadMessages, VISIBLE_EVENTS, noticeTone, threadItems } from '../src/desktop/renderer/SupervisorThread.js'

const at = (minute: number): string => `2026-11-05T10:${String(minute).padStart(2, '0')}:00.000Z`
const system = (id: string, text: string, minute = 0): SupervisorMessage => ({ id, role: 'system', text, at: at(minute), source: 'text' })
const user = (id: string, text: string, minute = 0): SupervisorMessage => ({ id, role: 'user', text, at: at(minute), source: 'text', author: 'Erfan' })
const xerxes = (id: string, text: string, minute = 0): SupervisorMessage => ({ id, role: 'xerxes', text, at: at(minute), source: 'text' })
const render = (messages: readonly SupervisorMessage[]): string =>
  renderToStaticMarkup(createElement(ThreadMessages, { messages, now: Date.parse(at(59)), label: 'Conversation' })).replaceAll('<!-- -->', '')

test('notice tones: a stop or failure is an error, attention is amber, Autopilot is its own tone', () => {
  expect(noticeTone('XERXES stopped: it could not reach its model')).toBe('error')
  expect(noticeTone('Needs attention (goal blocked): EasyDeL')).toBe('attention')
  expect(noticeTone('Autopilot: woke for a check-in in “EasyDeL”.')).toBe('autopilot')
  expect(noticeTone('Approved a permission request')).toBe('info')
})

test('a run of notices is one events item; identical neighbours merge with a count and the latest time', () => {
  const items = threadItems([
    system('a', 'Autopilot: woke for a blocked goal.', 1),
    system('b', 'Autopilot: woke for a blocked goal.', 2),
    system('c', 'Needs attention (goal blocked): X', 3),
    user('u', 'status?', 4),
    system('d', 'XERXES stopped: it could not reach its model', 5),
  ])
  expect(items.map(item => item.kind)).toEqual(['events', 'message', 'events'])
  const first = items[0]
  if (first?.kind !== 'events') throw new Error('expected events')
  expect(first.lines).toHaveLength(2)
  expect(first.lines[0]).toMatchObject({ count: 2, at: at(2), tone: 'autopilot' })
  expect(first.lines[1]).toMatchObject({ count: 1, tone: 'attention' })
  // Non-adjacent duplicates are not merged: order of events stays true.
  const split = threadItems([system('a', 'same'), system('b', 'other'), system('c', 'same')])
  expect(split[0]?.kind === 'events' ? split[0].lines.map(line => line.count) : []).toEqual([1, 1, 1])
})

test('rendered: events are one-line rows, repeats show ×N, an error row is marked, messages keep their roles', () => {
  const html = render([
    system('a', 'Autopilot: woke for a check-in.', 1),
    system('b', 'Autopilot: woke for a check-in.', 2),
    user('u', 'man what is the status', 3),
    xerxes('x', '**All good**', 4),
    system('d', 'XERXES stopped: it could not reach its model', 5),
  ])
  expect(html).toContain('class="xevent" data-tone="autopilot"')
  expect(html).toContain('aria-label="2 times">×2<')
  expect(html).toContain('class="xevent" data-tone="error"')
  expect(html).toContain('class="supervisor__message supervisor__message--user"')
  expect(html).toContain('Erfan · ')
  expect(html).toContain('<strong>All good</strong>')
  // No more "Notice" headings or yellow notice boxes.
  expect(html).not.toContain('supervisor__notice')
  expect(html).not.toContain('>Notice')
})

test('a long run of notices folds its older lines behind "N earlier events"', () => {
  const messages = Array.from({ length: VISIBLE_EVENTS + 4 }, (_, index) => system(`n${index}`, `event ${index}`, index))
  const html = render(messages)
  expect(html).toContain('<details class="xevents__more"><summary>4 earlier events</summary>')
  expect(html.match(/class="xevent"/g)).toHaveLength(VISIBLE_EVENTS + 4)
  // One line too many is not worth a fold.
  expect(render(messages.slice(0, VISIBLE_EVENTS + 1))).not.toContain('xevents__more')
})

test('notice text from the Hub is rendered as text, never HTML', () => {
  const html = render([system('a', '<img src=x onerror=alert(1)>')])
  expect(html).not.toContain('<img')
  expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;')
})

test('nothing renders for an empty thread', () => {
  expect(render([])).toBe('')
})

const plain = (html: string): string => html.replaceAll('<!-- -->', '')

const card = (id: string, patch: Partial<SupervisorConfirmation> = {}): SupervisorConfirmation => ({
  id, tool: 'task.steer', summary: 'Steer it', targets: [], state: 'confirmed', auto: true, createdAt: at(2), ...patch,
})

test('only pending or confirming cards stay cards; every settled one is a record', () => {
  expect(isOpenCard(card('a', { state: 'pending' }))).toBe(true)
  expect(isOpenCard(card('a', { state: 'confirming' }))).toBe(true)
  for (const state of ['confirmed', 'cancelled', 'expired', 'failed'] as const) expect(isOpenCard(card('a', { state }))).toBe(false)
})

test('a settled card is placed in the timeline at its own time, never piled under newer replies', () => {
  const messages = [user('u1', 'first', 1), xerxes('x1', 'answer one', 3), user('u2', 'second', 5), xerxes('x2', 'answer two', 7)]
  const merged = withRecords(messages, [card('c1', { createdAt: at(2) }), card('c2', { createdAt: at(6), state: 'failed', message: 'not found' })])
  expect(merged.map(m => m.id)).toEqual(['u1', 'card:c1', 'x1', 'u2', 'card:c2', 'x2'])
  expect(merged[1]!.text).toBe('XERXES did: task.steer: Steer it')
  expect(merged[4]!.text).toBe('XERXES failed: task.steer: Steer it (not found)')
  // An older Hub sends no time: the record goes after the newest message.
  const old = withRecords(messages, [card('c3', { createdAt: undefined } as Partial<SupervisorConfirmation>)])
  expect(old.at(-1)!.id).toBe('card:c3')
})

test('rendered: records are quiet event lines, and a failed one is marked as an error', () => {
  const html = plain(renderToStaticMarkup(createElement(ThreadMessages, {
    messages: [user('u1', 'kill it', 1), xerxes('x1', 'Done.', 4)], now: Date.parse(at(59)), label: 'Conversation',
    records: [card('c1', { createdAt: at(2) }), card('c2', { createdAt: at(3), state: 'failed' })],
  })))
  expect(html).toContain('class="xevent" data-tone="autopilot"')
  expect(html).toContain('class="xevent" data-tone="error"')
  expect(html).not.toContain('action-card')
  expect(html.indexOf('XERXES did:')).toBeLessThan(html.indexOf('Done.'))
})
