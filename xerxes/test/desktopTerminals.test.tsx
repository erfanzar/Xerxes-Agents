// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { TerminalCard } from '../src/desktop/renderer/TerminalsPanel.js'
import type { TerminalRow } from '../src/desktop/renderer/types.js'

test('terminal rows show the complete command once and expose only supported controls', () => {
  const row: TerminalRow = { id: 'terminal-1', kind: 'foreground', label: 'bun test src/…', command: 'bun test src/deep/workspace/session.test.ts', cwd: '/repo', running: false, startedAt: 1000, endedAt: 2000, exitCode: 1, outputChars: 100, canWrite: false, canInterrupt: false, canKill: false }
  const html = renderToStaticMarkup(createElement(TerminalCard, { row, online: true }))
  expect(html.split(row.command)).toHaveLength(2)
  expect(html).toContain('exit 1')
  expect(html).toContain('ended')
  expect(html).not.toContain('Interrupt')
  expect(html).not.toContain('Kill')
  const running = renderToStaticMarkup(createElement(TerminalCard, { row: { ...row, running: true, exitCode: null, canInterrupt: true }, online: true }))
  expect(running).toContain('Interrupt')
  expect(running).not.toContain('Kill')
})
