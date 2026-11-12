// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { remoteProfileReceipt } from '../src/desktop/main/remoteCredentials.js'

test('a receipt is read past harmless remote noise', () => {
  // A login shell or the runtime can print before the receiver's own line.
  expect(remoteProfileReceipt('Welcome to the host\n{"ok":true,"profile":"Personal"}\n')).toBe('Personal')
  expect(remoteProfileReceipt('{"ok":true,"profile":"Personal"}')).toBe('Personal')
})

test('only a real acknowledgement counts as a receipt', () => {
  for (const output of [
    '',
    'not json',
    '{"ok":false,"profile":"Personal"}',
    '{"ok":true}',
    '{"ok":true,"profile":""}',
    '{"ok":true,"profile":7}',
  ]) {
    expect(() => remoteProfileReceipt(output)).toThrow('not acknowledged')
  }
})