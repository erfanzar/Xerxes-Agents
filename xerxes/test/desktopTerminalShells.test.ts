// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { userShells } from '../src/desktop/renderer/terminalShells.js'

test('the terminal tab lists only live shells the user opened, not agent PTYs or background jobs', () => {
  expect(userShells([
    { id: 'pty_user', kind: 'pty', command: 'zsh', label: 'User shell', running: true },
    { id: 'pty_agent', kind: 'pty', command: 'python3 -i', label: 'python3 -i', running: true },
    { id: 'pty_old', kind: 'pty', command: 'zsh', label: 'User shell', running: false },
    { id: 'bg_1', kind: 'background', command: 'make', label: 'User shell', running: true },
  ])).toEqual([{ id: 'pty_user', running: true }])
})
