// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

// With a custom background the palette makes --x-card translucent (palettes.ts),
// while --x-popover stays solid. A floating menu painted with --x-card lets the
// conversation show through it: the agent switcher did exactly that.

// Comments are dropped first: a comment just before a rule would otherwise become part of its selector.
const css = (await Bun.file(new URL('../src/desktop/renderer/atelier.css', import.meta.url)).text()).replace(/\/\*[\s\S]*?\*\//g, '')
const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map(match => ({ selector: match[1]!.trim(), body: match[2]! }))

test('the agent switcher menu and the dictation error use the solid popover surface', () => {
  for (const selector of ['.atelier .agent-menu__pop', '.dictation-error']) {
    const rule = rules.find(entry => entry.selector === selector)
    expect(rule).toBeDefined()
    expect(rule!.body).toContain('background:var(--x-popover,var(--x-dialog))')
  }
})

test('no floating (absolute or fixed) surface is painted with the translucent card colour', () => {
  const offenders = rules
    .filter(entry => /position:(absolute|fixed)/.test(entry.body) && /background:var\(--x-(card|chip|chrome)\)/.test(entry.body))
    .map(entry => entry.selector)
  expect(offenders).toEqual([])
})
