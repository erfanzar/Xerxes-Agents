// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'

import { providerErrorText } from '../src/streaming/providerErrorText.js'
import { runTurn } from '../src/streaming/loop.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'

test('Bun transport errors become what happened and what to check', () => {
  expect(providerErrorText(new Error('The socket connection was closed unexpectedly. For more information, pass `verbose: true` in the second argument to fetch()')))
    .toBe('The model server closed the connection mid-reply. It may have crashed or run out of memory')
  expect(providerErrorText(Object.assign(new Error('Unable to connect. Is the computer able to access the url?'), { code: 'ConnectionRefused' })))
    .toBe('Could not connect to the model server. It may not be running')
  // Any other Bun fetch message just loses the developer hint.
  expect(providerErrorText(new Error('TLS handshake failed. For more information, pass `verbose: true` in the second argument to fetch()')))
    .toBe('TLS handshake failed.')
  expect(providerErrorText(new Error('invalid api key'))).toBe('invalid api key')
})

test('a connection that keeps dropping ends with the plain message, not the fetch hint', async () => {
  const events: StreamEvent[] = []
  for await (const event of runTurn({ model: 'test', state: createAgentState(), userMessage: 'hello' }, {
    retryDelays: [],
    maxNetworkRetries: 1,
    delay: async () => {},
    llm: {
      async *stream() {
        yield { content: 'partial' }
        throw new Error('The socket connection was closed unexpectedly. For more information, pass `verbose: true` in the second argument to fetch()')
      },
    },
  })) events.push(event)
  const final = events.filter(event => event.type === 'text').map(event => event.type === 'text' ? event.text : '').join('')
  expect(final).toContain('[Error: The model server closed the connection mid-reply.')
  expect(final).not.toContain('verbose: true')
})
