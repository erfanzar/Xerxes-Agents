// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { MENU_SECTIONS, SectionsMenuPanel, filterChanges } from '../src/desktop/renderer/SectionsMenu.js'
import type { DiffFile } from '../src/desktop/renderer/types.js'

const change = (path: string, isNew = false): DiffFile => ({ path, adds: 3, dels: 1, isNew, hunks: [], turn: 1 })
const changes = [change('xerxes/src/a.ts'), change('docs/new.md', true), change('xerxes/src/b.tsx')]
const base = { hoverTop: 0, needle: '', setNeedle: () => {}, agents: [], artifacts: null, changes, onHover: () => {}, onPick: () => {} }

test('the Sections menu lists the five sections in order, with a preview chevron on all but Terminal', () => {
  expect(MENU_SECTIONS.map(section => section.label)).toEqual(['Subagents', 'Artifacts', 'Changes', 'Files', 'Terminal'])
  const html = renderToStaticMarkup(<SectionsMenuPanel {...base} hover={null} />)
  for (const label of ['Subagents', 'Artifacts', 'Changes', 'Files', 'Terminal']) expect(html).toContain(label)
  expect(html.match(/aria-haspopup="menu"/g)?.length).toBe(4)
  expect(html).not.toContain('sections-menu__preview')
})

test('hovering Files previews the changed files with M/A badges and a filter', () => {
  const html = renderToStaticMarkup(<SectionsMenuPanel {...base} hover="files" />)
  expect(html).toContain('Search in files')
  expect(html).toContain('Changed files')
  expect(html).toContain('a.ts')
  expect(html).toContain('data-change="added"')
  expect(html).toContain('data-change="modified"')
})

test('the Files filter matches paths case-insensitively and an empty filter keeps everything', () => {
  expect(filterChanges(changes, 'SRC').map(item => item.path)).toEqual(['xerxes/src/a.ts', 'xerxes/src/b.tsx'])
  expect(filterChanges(changes, '  ')).toHaveLength(3)
  expect(filterChanges(changes, 'nothing')).toHaveLength(0)
})

test('an older runtime without artifacts says so instead of showing a count', () => {
  const html = renderToStaticMarkup(<SectionsMenuPanel {...base} hover="docs" />)
  expect(html).toContain('Artifacts load when the workspace runtime supports them.')
  expect(html).not.toContain('sections-menu__count')
})
