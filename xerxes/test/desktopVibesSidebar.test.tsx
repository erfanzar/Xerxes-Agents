// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { Sidebar, VibesSideNav } from '../src/desktop/renderer/App.js'
import { Store } from '../src/desktop/renderer/store.js'

const holder = globalThis as { document?: unknown }
const original = holder.document
afterEach(() => { holder.document = original })

const stamp = (style: string | null): void => {
  holder.document = { documentElement: { dataset: {}, getAttribute: (name: string) => name === 'data-style' ? style : null } }
}

test('Vibes sidebar shows Claude Code rows (XERXES lives in the XERXES | Code switch, not a row), and the other Xerxes destinations under More', () => {
  const snap = new Store().getSnapshot()
  const html = renderToStaticMarkup(createElement(VibesSideNav, { snap, page: null, filter: '', setFilter: () => {}, defaultMoreOpen: true }))
  const rows = html.slice(0, html.indexOf('role="menu"'))
  expect(rows).not.toContain('<span>XERXES</span>')
  const order = ['New', 'Search', 'Artifacts', 'Customize', 'More'].map(label => rows.indexOf(`<span>${label}</span>`))
  expect(order.every(at => at >= 0)).toBe(true)
  expect([...order].sort((a, b) => a - b)).toEqual(order)
  const menu = html.slice(html.indexOf('role="menu"'))
  for (const label of ['Tasks', 'Agents', 'Scheduled jobs', 'Filter tasks…']) expect(menu).toContain(`<span>${label}</span>`)
  expect(menu).not.toContain('<span>XERXES</span>')
})

test('Vibes and Neo share the Claude sidebar; only an unstyled page (VS Code high contrast) keeps the old rows', () => {
  const snap = new Store().getSnapshot()
  for (const style of ['vibes', 'neo']) {
    stamp(style)
    const claude = renderToStaticMarkup(createElement(Sidebar, { snap, page: null }))
    expect(claude).toContain('vibes-nav')
    expect(claude).toContain('vibes-foot__settings')
    expect(claude).toContain('vibes-mode')
    expect(claude).not.toContain('sidebar-wordmark')
    expect(claude).not.toContain('Search message history')
  }

  for (const style of [null]) {
    stamp(style)
    const classic = renderToStaticMarkup(createElement(Sidebar, { snap, page: null }))
    expect(classic).not.toContain('vibes-nav')
    expect(classic).not.toContain('vibes-foot')
    expect(classic).toContain('sidebar-wordmark')
    expect(classic).toContain('Skills &amp; tools')
    expect(classic).toContain('Scheduled jobs')
    expect(classic).toContain('Search message history')
    expect(classic).toContain('<span>Settings</span>')
  }
})
