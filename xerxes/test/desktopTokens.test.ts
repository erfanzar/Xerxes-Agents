// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { NOCTURNE_DARK, NOCTURNE_LIGHT } from '../src/ui/theme.js'
import { TOKEN_NAMES, themeStylesheet } from '../src/desktop/tokens.js'

test('every palette token reaches the desktop stylesheet', () => {
  const sheet = themeStylesheet()

  // A palette field with no custom property would render as an empty value in
  // the browser — a silently invisible element rather than a build error. The
  // mapped type makes a missing entry a compile error; this covers the other
  // direction, that what is declared is actually emitted.
  for (const [key, name] of Object.entries(TOKEN_NAMES)) {
    expect(sheet).toContain(`${name}: ${NOCTURNE_LIGHT[key as keyof typeof NOCTURNE_LIGHT]};`)
    expect(sheet).toContain(`${name}: ${NOCTURNE_DARK[key as keyof typeof NOCTURNE_DARK]};`)
  }
})

test('the desktop palette is the TUI palette, not a copy of it', () => {
  const sheet = themeStylesheet()

  // The design was authored against ui/theme.ts and its values match byte for
  // byte. This asserts the generated sheet still tracks that source: a colour
  // changed in the TUI must move the desktop app with it, or the two surfaces
  // drift the way every other pair in this repo has.
  expect(sheet).toContain(`--x-working: ${NOCTURNE_DARK.working};`)
  expect(sheet).toContain(`--x-done: ${NOCTURNE_DARK.done};`)
  expect(sheet).toContain(`--x-failed: ${NOCTURNE_DARK.failed};`)
  expect(NOCTURNE_DARK.working).toBe('#6ea8fe')
})

test('all three theme states are covered', () => {
  const sheet = themeStylesheet()

  // The viewer has three states, not two: an explicit choice stamps data-theme,
  // and the default "system" setting stamps nothing — so the media query must
  // carry dark on its own while still losing to an explicit light choice.
  expect(sheet).toContain(':root {')
  expect(sheet).toContain('@media (prefers-color-scheme: dark)')
  expect(sheet).toContain(':root:not([data-theme="light"])')
  expect(sheet).toContain(':root[data-theme="dark"]')
  expect(sheet).toContain(':root[data-theme="light"]')
})
