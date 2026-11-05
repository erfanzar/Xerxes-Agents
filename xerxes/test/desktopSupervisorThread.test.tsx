// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import type { SupervisorMessage } from '../src/client/hubParsers.js'
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
