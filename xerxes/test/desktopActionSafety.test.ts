// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'

test('modal dialogs own shortcuts while the update banner stays non-modal with Escape handling', async () => {
  const source = await Bun.file(new URL('../src/desktop/renderer/App.tsx', import.meta.url)).text()
  expect(source.includes('[role="dialog"][aria-modal="true"]')).toBe(true)
  const update = await Bun.file(new URL('../src/desktop/renderer/AppUpdatePrompt.tsx', import.meta.url)).text()
  expect(update).not.toContain('useDialogFocus(')
  expect(update).not.toContain('aria-modal')
  expect(update).toContain('role="status"')
  expect(update).toContain('onKeyDown={onEscape}')
  expect(update).toContain("event.key !== 'Escape'")
  expect(update).toContain('if (notesOpen) setNotesOpen(false)')
  expect(update).toContain("else void store.appUpdateAction('dismiss')")
})

test('file-save completions are scoped to the document and workspace that initiated them', async () => {
  const source = await Bun.file(new URL('../src/desktop/renderer/DesktopPanels.tsx', import.meta.url)).text()
  expect(source.includes('selection.current !== savingSelection')).toBe(true)
})
