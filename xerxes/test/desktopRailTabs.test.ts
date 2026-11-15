// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { MAX_RAIL_TABS, railFitOf, closeTab, openTab, panelOfKey, retitleTab, tabKey, type RailTab } from '../src/desktop/renderer/railTabs.js'

test('a section and one artifact are separate tabs; reopening one keeps a single copy', () => {
  let tabs: readonly RailTab[] = []
  tabs = openTab(tabs, { panel: 'files' })
  tabs = openTab(tabs, { panel: 'docs', item: 'abc' })
  tabs = openTab(tabs, { panel: 'files' })
  expect(tabs.map(tabKey)).toEqual(['files', 'docs:abc'])
  expect(panelOfKey('docs:abc')).toBe('docs')
})

test('closing the active tab focuses its right neighbour, then its left; the last closes the panel', () => {
  const tabs: readonly RailTab[] = [{ panel: 'files' }, { panel: 'review' }, { panel: 'terminal' }]
  expect(closeTab(tabs, 'review', 'review')).toEqual({ tabs: [{ panel: 'files' }, { panel: 'terminal' }], active: 'terminal' })
  expect(closeTab(tabs, 'terminal', 'terminal').active).toBe('review')
  expect(closeTab(tabs, 'files', 'terminal').active).toBe('terminal')
  expect(closeTab([{ panel: 'files' }], 'files', 'files')).toEqual({ tabs: [], active: null })
})

test('the strip is bounded and never drops the active tab', () => {
  let tabs: readonly RailTab[] = [{ panel: 'files' }]
  for (let i = 0; i < MAX_RAIL_TABS + 3; i++) tabs = openTab(tabs, { panel: 'docs', item: `a${i}` }, 'files')
  expect(tabs.length).toBe(MAX_RAIL_TABS)
  expect(tabs.map(tabKey)).toContain('files')
})

test('the panel width picks how much the header shows', () => {
  expect(railFitOf(600)).toBe('wide')
  expect(railFitOf(460)).toBe('wide')
  expect(railFitOf(420)).toBe('compact')
  expect(railFitOf(359)).toBe('narrow')
})

test('an artifact tab takes its title; an empty title keeps the old one', () => {
  const tabs = retitleTab([{ panel: 'docs', item: 'abc' }], 'docs:abc', ' Plan ')
  expect(tabs[0]?.title).toBe('Plan')
  expect(retitleTab(tabs, 'docs:abc', '')).toBe(tabs)
})
