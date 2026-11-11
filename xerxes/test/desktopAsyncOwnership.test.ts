// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'

test('commit detail requests have selection ownership before updating the detail or preselection', async () => {
  const source = await Bun.file(new URL('../src/desktop/renderer/GitPanel.tsx', import.meta.url)).text()
  expect(source.includes('epoch !== requestEpoch.current')).toBe(true)
})

test('terminal close has acknowledged pending and recoverable error states', async () => {
  const source = await Bun.file(new URL('../src/desktop/renderer/TerminalPanel.tsx', import.meta.url)).text()
  expect(source.includes('Closing…')).toBe(true)
  expect(source.includes('Retry close')).toBe(true)
  expect(source.includes('/* already gone */')).toBe(false)
})

test('monitor and delivery inspectors use the viewport dialog instead of following long lists', async () => {
  for (const name of ['Monitors', 'Deliveries']) {
    const source = await Bun.file(new URL(`../src/desktop/renderer/${name}.tsx`, import.meta.url)).text()
    expect(source.includes('<ActionDialog')).toBe(true)
  }
})
