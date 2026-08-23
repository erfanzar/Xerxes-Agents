// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Paging, for keyboards that have no PageUp/PageDown.
//
// Every scrolling panel advertised `PgUp/PgDn` in its footer, which on a
// compact laptop keyboard is a key you cannot press. The design is explicit
// that "every key listed maps to a real capability" — a hint you physically
// cannot honour is the same failure as one that opens a "not implemented"
// dialog, so the panels take vim's `ctrl-b`/`ctrl-f` as well and say so.
//
// `ctrl-d` is deliberately NOT bound: it is the global quit chord
// (`useInputHandlers`), and a half-page scroll that sometimes exits the app
// is worse than no half-page scroll.

interface PageKeyEvent {
  ctrl?: boolean
  meta?: boolean
  name?: string
  shift?: boolean
  super?: boolean
}

const isChord = (event: PageKeyEvent, letter: string): boolean =>
  event.name === letter && event.ctrl === true && !event.meta && !event.super && !event.shift

/** PageUp, or `ctrl-b`. */
export const isPageUpKey = (event: PageKeyEvent): boolean => event.name === 'pageup' || isChord(event, 'b')

/** PageDown, or `ctrl-f`. */
export const isPageDownKey = (event: PageKeyEvent): boolean => event.name === 'pagedown' || isChord(event, 'f')

/** What the footers print, so the hint and the binding cannot drift. */
export const PAGE_KEY_HINT = '⌃b/⌃f page'
