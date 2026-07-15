// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { websocketRequestAuthorized } from '../src/daemon/websocketGateway.js'

test('gateway-auth parity disables blank tokens and rejects non-bearer credentials', () => {
  const request = new Request('http://localhost/rpc')

  expect(websocketRequestAuthorized(request, undefined)).toBeTrue()
  expect(websocketRequestAuthorized(request, '')).toBeTrue()
  expect(websocketRequestAuthorized(request, '   ')).toBeTrue()
  expect(websocketRequestAuthorized(new Request('http://localhost/rpc', {
    headers: { Authorization: 'Basic c2VjcmV0' },
  }), 'secret')).toBeFalse()
  expect(websocketRequestAuthorized(new Request('http://localhost/rpc', {
    headers: { Authorization: 'Bearer wrong' },
  }), 'secret')).toBeFalse()
})

test('gateway-auth parity finds an exact token among unrelated query parameters', () => {
  const expected = 'correct-token'
  const request = new Request(`http://localhost/rpc?foo=bar&token=${expected}&baz=qux`)

  expect(websocketRequestAuthorized(request, expected)).toBeTrue()
  expect(websocketRequestAuthorized(new Request('http://localhost/rpc?foo=bar&token=wrong&baz=qux'), expected)).toBeFalse()
})
