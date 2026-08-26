// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * CSS custom properties for the desktop app, derived from the TUI's palette.
 *
 * The desktop design was authored against `ui/theme.ts` and its tokens match it
 * byte for byte. Copying them into a stylesheet would create a second source of
 * truth for the product's colours — the same split that let the TUI and the
 * daemon disagree about interaction mode, and the deferral flag disagree with
 * the code that read it. So the sheet is GENERATED from the palette, and a test
 * asserts every declared token is emitted.
 *
 * Only the palette lives here. Layout, type and spacing belong to the
 * stylesheet, because nothing else consumes them.
 */

import { NOCTURNE_DARK, NOCTURNE_LIGHT, type NocturnePalette } from '../ui/theme.js'

/**
 * Palette key → CSS custom property name.
 *
 * Written out rather than derived by camel-to-kebab so that renaming a palette
 * field is a compile error here instead of a silently missing variable: the
 * mapped type below requires an entry for every key of NocturnePalette.
 */
export const TOKEN_NAMES: { readonly [K in keyof NocturnePalette]: string } = {
  working: '--x-working',
  done: '--x-done',
  failed: '--x-failed',
  needsInput: '--x-needs',
  activity: '--x-activity',
  structure: '--x-structure',
  needsInputText: '--x-needs-text',
  failedText: '--x-failed-text',
  backdrop: '--x-backdrop',
  sunken: '--x-sunken',
  screen: '--x-screen',
  chrome: '--x-chrome',
  card: '--x-card',
  selected: '--x-selected',
  hairline: '--x-hairline',
  divider: '--x-divider',
  focusEdge: '--x-focus',
  strong: '--x-strong',
  title: '--x-title',
  prose: '--x-prose',
  secondary: '--x-secondary',
  meta: '--x-meta',
  numeric: '--x-numeric',
  caption: '--x-caption',
  separator: '--x-separator',
  rule: '--x-rule',
  leader: '--x-leader',
  leaderQuiet: '--x-leader-quiet',
  workingCardBg: '--x-working-bg',
  needsInputCardBg: '--x-needs-bg',
  needsInputCardBorder: '--x-needs-border',
  doneCardBg: '--x-done-bg',
  doneCardBorder: '--x-done-border',
  failedCardBg: '--x-failed-bg',
  failedCardBorder: '--x-failed-border',
  diffContext: '--x-diff-context',
  diffAddRow: '--x-diff-add',
  diffDelRow: '--x-diff-del',
  diffAddWordBg: '--x-diff-add-word-bg',
  diffDelWordBg: '--x-diff-del-word-bg',
  diffAddWordFg: '--x-diff-add-word-fg',
  diffDelWordFg: '--x-diff-del-word-fg',
  diffHunkBg: '--x-diff-hunk',
  diffFoldBg: '--x-diff-fold',
}

/**
 * Chrome the terminal has no equivalent for.
 *
 * The menu bar exists only in a window, so these five have no palette role to
 * derive from and are stated per theme. Kept in their own layer rather than
 * added to NocturnePalette: the TUI would carry fields it can never render, and
 * the palette is the contract both surfaces share.
 */
export const CHROME_TOKENS: Readonly<Record<'dark' | 'light', Readonly<Record<string, string>>>> = {
  dark: {
    '--x-menubar': 'rgba(20,22,30,0.92)',
    '--x-menubar-edge': '#14161e',
    '--x-menubar-fg': '#b2b6ca',
    '--x-menubar-strong': '#e4e7f5',
    '--x-menubar-meta': '#75798c',
  },
  light: {
    '--x-menubar': 'rgba(241,243,245,0.94)',
    '--x-menubar-edge': '#dbe2ea',
    '--x-menubar-fg': '#3f5162',
    '--x-menubar-strong': '#172533',
    '--x-menubar-meta': '#647486',
  },
}

const chromeDeclarations = (mode: 'dark' | 'light'): string =>
  Object.entries(CHROME_TOKENS[mode]).map(([name, value]) => `  ${name}: ${value};`).join('\n')

/** Emit `--name: value;` declarations for one palette, sorted for a stable diff. */
export function paletteDeclarations(palette: NocturnePalette): string {
  return (Object.keys(TOKEN_NAMES) as (keyof NocturnePalette)[])
    .map(key => `  ${TOKEN_NAMES[key]}: ${palette[key]};`)
    .sort()
    .join('\n')
}

/**
 * The full custom-property sheet.
 *
 * Three selectors, not two: an explicit choice stamps `data-theme`, and the
 * default "follow the system" state stamps nothing, so the media query has to
 * carry the dark palette on its own while still losing to an explicit light
 * choice.
 */
export function themeStylesheet(): string {
  return [
    ':root {',
    paletteDeclarations(NOCTURNE_LIGHT),
    chromeDeclarations('light'),
    '}',
    '',
    '@media (prefers-color-scheme: dark) {',
    '  :root:not([data-theme="light"]) {',
    paletteDeclarations(NOCTURNE_DARK).replace(/^/gm, '  '),
    chromeDeclarations('dark').replace(/^/gm, '  '),
    '  }',
    '}',
    '',
    ':root[data-theme="dark"] {',
    paletteDeclarations(NOCTURNE_DARK),
    chromeDeclarations('dark'),
    '}',
    '',
    ':root[data-theme="light"] {',
    paletteDeclarations(NOCTURNE_LIGHT),
    chromeDeclarations('light'),
    '}',
    '',
  ].join('\n')
}
