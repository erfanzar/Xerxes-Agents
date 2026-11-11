// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'

test('ARIA modal dialogs own global keyboard shortcuts just like native dialogs', async () => {
  const source = await Bun.file(new URL('../src/desktop/renderer/App.tsx', import.meta.url)).text()
  expect(source.includes('[role="dialog"][aria-modal="true"]')).toBe(true)
  const update = await Bun.file(new URL('../src/desktop/renderer/AppUpdatePrompt.tsx', import.meta.url)).text()
  expect(update).toContain('useDialogFocus(')
  expect(update).toContain('onKeyDown={')
})

test('file-save completions are scoped to the document and workspace that initiated them', async () => {
  const source = await Bun.file(new URL('../src/desktop/renderer/DesktopPanels.tsx', import.meta.url)).text()
  expect(source.includes('selection.current !== savingSelection')).toBe(true)
})
