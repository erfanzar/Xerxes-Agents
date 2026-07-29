// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { sanitizeTelegramOutbound } from '../src/channels/telegram.js'
import { sanitizeJournalOutput } from '../src/channels/turnRouter.js'

test('windows user paths are redacted from telegram outbound text', () => {
  expect(sanitizeTelegramOutbound('log written to C:\\Users\\amin\\.xerxes\\daemon.log done')).toBe(
    'log written to [path redacted] done',
  )
  expect(sanitizeTelegramOutbound('open D:\\work\\project\\.xerxes\\state for details')).toBe(
    'open [path redacted] for details',
  )
  expect(sanitizeTelegramOutbound('state at ~\\.xerxes\\daemon\\x.pid here')).toBe(
    'state at [path redacted] here',
  )
})

test('windows user paths are redacted from channel journal output', () => {
  expect(sanitizeJournalOutput('see C:\\Users\\amin\\AppData\\file.txt')).toBe('see [path redacted]')
})

test('existing POSIX redaction behavior is unchanged', () => {
  expect(sanitizeTelegramOutbound('read /home/amin/.xerxes/daemon.log please')).toBe(
    'read [path redacted] please',
  )
  expect(sanitizeTelegramOutbound('read ~/.xerxes/daemon.log please')).toBe(
    'read [path redacted] please',
  )
  expect(sanitizeJournalOutput('temp at /tmp/xerxes-build/out')).toBe('temp at [path redacted]')
})

test('benign text without user paths passes through untouched', () => {
  expect(sanitizeTelegramOutbound('build finished in 42s')).toBe('build finished in 42s')
  expect(sanitizeJournalOutput('relative path src/index.ts unchanged')).toBe(
    'relative path src/index.ts unchanged',
  )
})
