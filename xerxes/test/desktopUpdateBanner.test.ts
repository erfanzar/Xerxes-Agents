// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

const ATELIER = new URL('../src/desktop/renderer/atelier.css', import.meta.url)

function rule(css: string, selector: string): string {
  const head = `${selector}{`
  const start = css.indexOf(head)
  if (start < 0) throw new Error(`missing rule ${selector}`)
  return css.slice(start + head.length, css.indexOf('}', start))
}

test('the update banner and its buttons are never part of the window drag region', async () => {
  // The banner floats over the toolbar, which is -webkit-app-region:drag. In
  // Electron a drag region wins over anything painted above it unless that
  // element opts out, so clicks on the overlapped part of a button dragged the
  // window instead ("the click box is different from what is drawn").
  const css = await Bun.file(ATELIER).text()
  expect(rule(css, '.atelier .appbanner')).toContain('-webkit-app-region:no-drag')
  expect(rule(css, '.atelier .appbanner *')).toContain('-webkit-app-region:no-drag')
})
