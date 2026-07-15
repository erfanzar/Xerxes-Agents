// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ApprovalScope,
  ApprovalStore,
  REDACTED,
  redactPayload,
  redactString,
} from '../src/security/index.js'

test('redaction parity removes token, PII, and cloud-credential-shaped values from strings', () => {
  const fixtures = [
    'sk-abcdefghijklmnop',
    'sk-ant-abcdefghijklmnop',
    'foo@example.com',
    '555-123-4567',
    'AKIAIOSFODNN7EXAMPLE',
    'abc123XYZ789LONGTOKEN',
  ]
  const text = [
    fixtures[0],
    fixtures[1],
    `contact ${fixtures[2]}`,
    `call ${fixtures[3]}`,
    `cloud ${fixtures[4]}`,
    `Authorization: Bearer ${fixtures[5]}`,
  ].join(' | ')
  const redacted = redactString(text)

  for (const fixture of fixtures) expect(redacted).not.toContain(fixture)
  expect(redacted).toContain(REDACTED)
})

test('redaction parity recursively handles arrays while preserving non-secret primitive values and input ownership', () => {
  const input = {
    count: 42,
    nested: { fresh: 'ok', token: 'secret' },
    ratio: 0.5,
    values: ['contact me at a@b.com'],
  }
  const output = redactPayload(input) as {
    readonly count: number
    readonly nested: { readonly fresh: string; readonly token: string }
    readonly ratio: number
    readonly values: readonly string[]
  }

  expect(output.count).toBe(42)
  expect(output.ratio).toBe(0.5)
  expect(output.nested).toEqual({ fresh: 'ok', token: REDACTED })
  expect(output.values[0]).not.toContain('@b.com')
  expect(input.nested.token).toBe('secret')
  expect(input.values[0]).toBe('contact me at a@b.com')
})

test('approval parity distinguishes absent, once, session, denied, and cleared decisions', () => {
  const store = new ApprovalStore()
  expect(store.check('run', 's1')).toBeUndefined()

  store.add({ argsHash: 'h1', granted: true, scope: ApprovalScope.ONCE, sessionId: 's1', toolName: 'run' })
  expect(store.check('run', 's1', 'h1')).toBeTrue()
  expect(store.check('run', 's1', 'h2')).toBeUndefined()
  store.add({ granted: false, scope: ApprovalScope.SESSION, sessionId: 's2', toolName: 'rm' })
  expect(store.check('rm', 's2')).toBeFalse()

  const sessions = new ApprovalStore()
  sessions.add({ granted: true, scope: ApprovalScope.SESSION, sessionId: 's1', toolName: 'a' })
  sessions.add({ granted: true, scope: ApprovalScope.SESSION, sessionId: 's2', toolName: 'b' })
  expect(sessions.clearSession('s1')).toBe(1)
  expect(sessions.check('a', 's1')).toBeUndefined()
  expect(sessions.check('b', 's2')).toBeTrue()
})
