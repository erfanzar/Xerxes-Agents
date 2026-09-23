// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { plainTerminalText, readableOutput } from '../src/desktop/renderer/OutputViewer.js'

test('terminal output reads as plain text: colors, shell-integration noise and prompts sequences are gone', () => {
  // The exact kind of stream the Background panel showed raw.
  const raw = '\x1b]1337;RemoteHost=erfan@mac\x07\x1b]1337;CurrentDir=/repo\x07\x1b]133;A\x07\x1b[49m\x1b[38;2;87;199;255m~/repo\x1b[0m \x1b[38;5;242mmain\x1b[0m ❯ \x1b]133;B\x07\x1b[?2004hls\r\n\x1b]133;C;\x07\x1b[1;34mdocs\x1b[0m  README.md\r\n'
  expect(plainTerminalText(raw)).toBe('~/repo main ❯ ls\ndocs  README.md\n')
})

test('carriage returns resolve like a terminal, so a progress bar shows its last frame', () => {
  expect(plainTerminalText('Downloading  10%\rDownloading  55%\rDownloading 100%\ndone\n')).toBe('Downloading 100%\ndone\n')
  expect(plainTerminalText('abc\x08\x08XY\n')).toBe('aXY\n')
})

test('plain logs and JSON results are untouched apart from escapes', () => {
  expect(plainTerminalText('just text\nline two')).toBe('just text\nline two')
  expect(readableOutput(JSON.stringify({ stdout: '\x1b[32mok\x1b[0m\n' }))).toBe('ok\n')
  expect(readableOutput('{"partial": ')).toBe('{"partial": ')
})
