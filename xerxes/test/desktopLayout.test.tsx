// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { parsePanelLayout, PanelDivider } from '../src/desktop/renderer/layout.js'
import { ActivityDetails } from '../src/desktop/renderer/App.js'
import { BackgroundActivity } from '../src/desktop/renderer/DesktopPanels.js'
import { Store } from '../src/desktop/renderer/store.js'

test('panel preferences validate persisted data and bound widths', () => {
  expect(parsePanelLayout(null)).toEqual({ sidebarWidth: 240, inspectorWidth: 340, sidebarHidden: false })
  expect(parsePanelLayout({ sidebarWidth: -1, inspectorWidth: 5000, sidebarHidden: true })).toEqual({ sidebarWidth: 180, inspectorWidth: 1400, sidebarHidden: true })
  // A wide rail is kept (it used to be capped at 520px).
  expect(parsePanelLayout({ inspectorWidth: 900 }).inspectorWidth).toBe(900)
  expect(parsePanelLayout({ sidebarWidth: NaN, inspectorWidth: '500', sidebarHidden: 'false' })).toEqual(parsePanelLayout(null))
})
test('resizable panels expose keyboard-operable separator values', () => {
  const html = renderToStaticMarkup(createElement(PanelDivider, { label: 'Resize sessions', min: 180, max: 360, value: 240, onChange() {} }))
  expect(html).toContain('role="separator"')
  expect(html).toContain('aria-valuenow="240"')
  expect(html).toContain('tabindex="0"')
})
test('long goals and eight agents retain all content without zero-turn noise', () => {
  const store = new Store()
  const objective = 'Preserve every integration while making conversation, code review and task navigation readable. '.repeat(8)
  const snap = { ...store.getSnapshot(), goal: `Objective: ${objective}\nStatus: active`, fleet: Array.from({ length: 8 }, (_, i) => ({ id: `${i}`, key: `${i}`, title: `Agent ${i}: investigate cancellation and transport recovery`, status: 'working', age: '', current: false, kind: 'subagent', turns: 0, messages: 0, cwd: '/repo', untitled: false })) }
  const html = renderToStaticMarkup(createElement(ActivityDetails, { snap }))
  expect(html).toContain(objective.trim())
  expect(html).toContain('Show full goal')
  // The rail lists agents as flat rows; eight working agents are all
  // visible, because only a finished tail is ever capped.
  expect(html.match(/class="railrow"/g)).toHaveLength(8)
  expect(html).not.toContain('more<')
  // Statistics sit in the status card as plain rows, like context — a question you
  // have after the fact, not while eight agents are running.
  expect(html).toContain('<dl class="session-diagnostics session-diagnostics__values" aria-label="Session statistics">')
  expect(html).not.toContain('0 turns')
})

test('appearance preferences validate stored choices independently of session state', async () => {
  const { parseAppearance } = await import('../src/desktop/renderer/appearance.js')
  expect(parseAppearance({ theme: 'light', font: '13' })).toEqual({ theme: 'light', font: '13' })
  // Unreadable choices fall back to a fresh install's defaults (XS text).
  expect(parseAppearance({ theme: 'invalid', font: 90 })).toEqual({ theme: 'system', font: '11' })
})

test('background history is collapsed while active and unknown states remain visible', () => {
  const rows = [
    { id: 'live', state: 'running', title: 'Live watch' },
    { id: 'future', state: 'waiting_for_input', title: 'Needs input' },
    { id: 'old', state: 'archived', title: 'Archived watch' },
    { id: 'failed', state: 'failed', title: 'Failed command' },
  ]
  const html = renderToStaticMarkup(createElement(BackgroundActivity, { rows, renderRow: row => createElement('p', { key: String(row.id) }, String(row.title)) }))
  const disclosure = html.indexOf('<details')
  expect(html.indexOf('Live watch')).toBeLessThan(disclosure)
  expect(html.indexOf('Needs input')).toBeLessThan(disclosure)
  expect(html.indexOf('Archived watch')).toBeGreaterThan(disclosure)
  expect(html).toContain('2 · 1 failed')
  expect(html).not.toContain('<details open')
  // "Past" now means the tail beyond the few kept on screen.
  expect(html).toContain('aria-label="Earlier background activity" tabindex="0"')
  expect(renderToStaticMarkup(createElement(BackgroundActivity, { rows: [], renderRow: () => createElement('p') }))).toBe('')
})
