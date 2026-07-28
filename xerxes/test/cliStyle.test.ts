// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  CliWriter,
  createCliStyle,
  detectColorDepth,
  supportsUnicode,
  terminalWidth,
} from '../src/runtime/cliStyle.js'

const ESC = '\u001B'

test('NO_COLOR disables colour regardless of terminal capability', () => {
  // Presence is what counts by convention, including an empty value — a user who
  // sets NO_COLOR= means it.
  expect(detectColorDepth({ env: { NO_COLOR: '1', COLORTERM: 'truecolor', TERM: 'xterm-256color' }, isTTY: true }))
    .toBe('none')
  expect(detectColorDepth({ env: { NO_COLOR: '', COLORTERM: 'truecolor' }, isTTY: true })).toBe('none')
})

test('a non-TTY stdout is plain even on a capable terminal', () => {
  // This is the case that matters most: piping to a file or another program must
  // not embed escape sequences in the data.
  expect(detectColorDepth({ env: { COLORTERM: 'truecolor', TERM: 'xterm-256color' }, isTTY: false })).toBe('none')
  expect(detectColorDepth({ env: { TERM: 'dumb' }, isTTY: true })).toBe('none')
  expect(detectColorDepth({ env: {}, isTTY: true })).toBe('none')
})

test('FORCE_COLOR overrides capability detection in both directions', () => {
  expect(detectColorDepth({ env: { FORCE_COLOR: '0', COLORTERM: 'truecolor' }, isTTY: true })).toBe('none')
  expect(detectColorDepth({ env: { FORCE_COLOR: '1' }, isTTY: false })).toBe('ansi16')
  expect(detectColorDepth({ env: { FORCE_COLOR: '2' }, isTTY: false })).toBe('ansi256')
  expect(detectColorDepth({ env: { FORCE_COLOR: '3' }, isTTY: false })).toBe('truecolor')
  // NO_COLOR still wins: an explicit opt-out beats an explicit opt-in.
  expect(detectColorDepth({ env: { FORCE_COLOR: '3', NO_COLOR: '1' }, isTTY: true })).toBe('none')
})

test('terminal capability maps to the expected depth', () => {
  expect(detectColorDepth({ env: { COLORTERM: 'truecolor', TERM: 'xterm' }, isTTY: true })).toBe('truecolor')
  expect(detectColorDepth({ env: { COLORTERM: '24bit', TERM: 'xterm' }, isTTY: true })).toBe('truecolor')
  expect(detectColorDepth({ env: { TERM: 'xterm-256color' }, isTTY: true })).toBe('ansi256')
  expect(detectColorDepth({ env: { TERM: 'xterm' }, isTTY: true })).toBe('ansi16')
})

test('a disabled styler is exactly the identity function', () => {
  // The plain path must not merely avoid colour, it must return the input
  // untouched — anything else would change output for scripts and tests.
  const style = createCliStyle('none')
  expect(style.enabled).toBe(false)
  for (const input of ['plain', '', 'with spaces', 'Git: current (HEAD abc)']) {
    expect(style.bold(input)).toBe(input)
    expect(style.dim(input)).toBe(input)
    expect(style.color('error', input)).toBe(input)
  }
})

test('each depth emits the escape form that depth actually supports', () => {
  expect(createCliStyle('truecolor').color('error', 'x')).toBe(`${ESC}[38;2;224;85;107mx${ESC}[39m`)
  // 6x6x6 cube: #e0556b = (224,85,107) quantizes to (4,2,2)
  // -> 16 + 36*4 + 6*2 + 2 = 174.
  expect(createCliStyle('ansi256').color('error', 'x')).toBe(`${ESC}[38;5;174mx${ESC}[39m`)
  expect(createCliStyle('ansi16').color('error', 'x')).toBe(`${ESC}[91mx${ESC}[39m`)
})

test('a nested colour does not strip the enclosing bold', () => {
  // A blanket `0m` reset would clear bold when the inner colour closed, so the
  // tail of the string would silently lose its emphasis.
  const style = createCliStyle('ansi16')
  const rendered = style.bold(`a${style.color('ok', 'b')}c`)
  expect(rendered).toBe(`${ESC}[1ma${ESC}[92mb${ESC}[39mc${ESC}[22m`)
  // Bold is opened once and closed once, and the close is the bold-specific code.
  expect(rendered).not.toContain(`${ESC}[0m`)
})

test('styling an empty string adds nothing', () => {
  // Wrapping an empty string would emit a bare escape pair, which some terminals
  // render as a stray artifact and which pads captured output for no reason.
  for (const depth of ['truecolor', 'ansi256', 'ansi16'] as const) {
    expect(createCliStyle(depth).color('ok', '')).toBe('')
    expect(createCliStyle(depth).bold('')).toBe('')
  }
})

test('status rows keep their glyph and message when colour is off', () => {
  const lines: string[] = []
  const writer = new CliWriter({ style: createCliStyle('none'), unicode: true, write: line => lines.push(line) })
  writer.status('ok', 'bun', 'Bun 1.3.12')
  writer.status('warn', 'provider-keys', 'No provider API key is set', 'Set OPENAI_API_KEY')
  expect(lines[0]).toBe('✓ bun: Bun 1.3.12')
  expect(lines[1]).toBe('! provider-keys: No provider API key is set')
  expect(lines[2]).toBe('    → Set OPENAI_API_KEY')
})

test('status rows fall back to ASCII markers when unicode is not trusted', () => {
  // A glyph the terminal cannot render shifts every column after it, so the
  // ASCII form is a correctness measure rather than a preference.
  const lines: string[] = []
  const writer = new CliWriter({ style: createCliStyle('none'), unicode: false, write: line => lines.push(line) })
  writer.status('fail', 'daemon', 'not reachable', 'start it with xerxes daemon')
  expect(lines[0]).toBe('x daemon: not reachable')
  expect(lines[1]).toBe('    -> start it with xerxes daemon')
})

test('a multi-line hint indents every line, not just the first', () => {
  const lines: string[] = []
  const writer = new CliWriter({ style: createCliStyle('none'), unicode: true, write: line => lines.push(line) })
  writer.status('warn', 'config', 'two sources disagree', 'first line\nsecond line')
  expect(lines[1]).toBe('    → first line')
  expect(lines[2]).toBe('    → second line')
})

test('colour never carries meaning on its own', () => {
  // Same content at every depth: strip the escapes from a coloured render and it
  // must equal the plain render exactly.
  for (const depth of ['truecolor', 'ansi256', 'ansi16'] as const) {
    const styled: string[] = []
    const plain: string[] = []
    for (const [sink, style] of [[styled, createCliStyle(depth)], [plain, createCliStyle('none')]] as const) {
      const writer = new CliWriter({ style, unicode: true, write: line => (sink as string[]).push(line) })
      writer.status('warn', 'label', 'message', 'hint')
      writer.field('key', 'value', 8)
      writer.step('a step', 2)
    }
    const stripped = styled.map(line => line.replaceAll(/\u001B\[[0-9;]*m/g, ''))
    expect(stripped).toEqual(plain)
  }
})

test('headings fill the available width without exceeding it', () => {
  const lines: string[] = []
  const writer = new CliWriter({ style: createCliStyle('none'), unicode: true, write: line => lines.push(line) })
  writer.heading('Update', 40)
  expect(lines[0]).toHaveLength(40)
  expect(lines[0]?.startsWith('Update ─')).toBe(true)

  // A heading wider than the target must not produce a negative-length rule.
  writer.heading('a'.repeat(50), 40)
  expect(lines[1]).toBe('a'.repeat(50))
})

test('terminal width is clamped so unknown or extreme sizes stay readable', () => {
  expect(terminalWidth(undefined)).toBe(80)
  expect(terminalWidth(0)).toBe(80)
  expect(terminalWidth(20)).toBe(40)
  expect(terminalWidth(300)).toBe(100)
  expect(terminalWidth(72)).toBe(72)
})

test('unicode support requires evidence rather than being assumed', () => {
  expect(supportsUnicode({ env: { LANG: 'en_US.UTF-8' } })).toBe(true)
  expect(supportsUnicode({ env: { LC_ALL: 'C.utf8' } })).toBe(true)
  expect(supportsUnicode({ env: { LANG: 'C' } })).toBe(false)
  expect(supportsUnicode({ env: {} })).toBe(false)
  // Windows hosts set no POSIX locale variable but do render UTF-8.
  expect(supportsUnicode({ env: { WT_SESSION: '1' } })).toBe(true)
  // An explicit opt-out wins over every signal.
  expect(supportsUnicode({ env: { LANG: 'en_US.UTF-8', XERXES_CLI_ASCII: '1' } })).toBe(false)
})
