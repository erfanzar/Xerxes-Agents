// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { Screens } from '../src/desktop/renderer/Screens.js'
import { INITIAL_STATE, buildView } from '../src/desktop/renderer/viewModel.js'

/** Every screen the app's own navigation offers. */
const SCREEN_IDS = buildView(INITIAL_STATE, () => {})
  .nav.filter((item: { id?: string }) => item.id)
  .map((item: { id: string }) => item.id)

const render = (screen: string, turn = INITIAL_STATE.turn) =>
  renderToStaticMarkup(
    createElement(Screens, { state: { ...INITIAL_STATE, screen, turn }, set: () => undefined }),
  )

/**
 * A screen that throws or renders a stub is invisible to a type check and to a
 * build — both pass on markup nobody can read. This is the cheapest thing that
 * actually looks at the output.
 */
test('every navigable screen renders real markup', () => {
  expect(SCREEN_IDS.length).toBeGreaterThan(15)

  for (const screen of SCREEN_IDS) {
    const html = render(screen)
    expect({ screen, rendered: html.length > 8_000 }).toEqual({ screen, rendered: true })
  }
})

test('the session screen renders each of its turn states', () => {
  // Idle, streaming, fan-out and approval are four different screens wearing
  // one name; a regression in any of them hides behind the other three.
  for (const turn of ['idle', 'transcript', 'fanout', 'approval']) {
    const html = render('session', turn)
    expect({ turn, rendered: html.length > 8_000 }).toEqual({ turn, rendered: true })
  }
})

test('screens are distinct from one another', () => {
  // Guards the failure where a switch falls through and several ids quietly
  // render the same pane.
  const lengths = new Map<string, number>()
  for (const screen of SCREEN_IDS) lengths.set(screen, render(screen).length)
  expect(new Set(lengths.values()).size).toBeGreaterThan(SCREEN_IDS.length * 0.7)
})

test('no colour is hard-coded past the generated palette', () => {
  // The design's tokens came from ui/theme.ts; the screens must keep reading
  // them through custom properties so the desktop cannot drift from the TUI.
  const html = render('style')
  const literals = html.match(/#[0-9a-fA-F]{6}\b/g) ?? []
  expect(literals).toEqual([])
})
