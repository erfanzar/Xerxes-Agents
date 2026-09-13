// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { parsePanelLayout, PanelDivider } from '../src/desktop/renderer/layout.js'
import { ActivityDetails } from '../src/desktop/renderer/App.js'
import { Store } from '../src/desktop/renderer/store.js'

test('panel preferences validate persisted data and bound widths', () => {
  expect(parsePanelLayout(null)).toEqual({ sidebarWidth: 240, inspectorWidth: 340, sidebarHidden: false })
  expect(parsePanelLayout({ sidebarWidth: -1, inspectorWidth: 900, sidebarHidden: true })).toEqual({ sidebarWidth: 180, inspectorWidth: 520, sidebarHidden: true })
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
  expect(html.match(/class="agent-row"/g)).toHaveLength(8)
  expect(html).not.toContain('0 turns')
})

test('appearance preferences validate stored choices independently of session state', async () => {
  const { parseAppearance } = await import('../src/desktop/renderer/appearance.js')
  expect(parseAppearance({ theme: 'light', font: '13' })).toEqual({ theme: 'light', font: '13' })
  expect(parseAppearance({ theme: 'invalid', font: 90 })).toEqual({ theme: 'system', font: '12' })
})
