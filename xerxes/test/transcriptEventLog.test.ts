// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  encodeTranscriptEvent,
  readTranscriptEventLines,
  transcriptMessageAppendedEvent,
  transcriptTurnCompletedEvent,
  transcriptTurnStartedEvent,
} from '../src/session/transcriptEventLog.js'

test('typed transcript events round-trip with stable JSONL field names', () => {
  const event = transcriptMessageAppendedEvent('facefeedfacefeed', 2, {
    role: 'assistant',
    content: 'done',
  }, { eventId: 'event-00000002', sequence: 7 })
  const encoded = encodeTranscriptEvent(event)

  expect(JSON.parse(encoded)).toEqual({
    event_schema_version: 1,
    type: 'message_appended',
    session_id: 'facefeedfacefeed',
    event_id: 'event-00000002',
    sequence: 7,
    index: 2,
    message: { role: 'assistant', content: 'done' },
  })
  expect(readTranscriptEventLines(encoded, 'facefeedfacefeed')).toEqual({
    events: [event],
    malformedLines: 0,
    partialTail: false,
  })
})

test('lifecycle events round-trip through the shared JSONL codec', () => {
  const events = [
    transcriptTurnStartedEvent('facefeedfacefeed', 'turn-1', { mode: 'code' }, {
      eventId: 'event-1', sequence: 1,
    }),
    transcriptTurnCompletedEvent('facefeedfacefeed', 'turn-1', { stopReason: 'completed' }, {
      eventId: 'event-2', sequence: 2,
    }),
  ]
  const decoded = readTranscriptEventLines(events.map(encodeTranscriptEvent).join(''), 'facefeedfacefeed')
  expect(decoded.events).toEqual(events)
  expect(decoded.malformedLines).toBe(0)
})

test('event reader accepts legacy journal rows and rejects cross-session events', () => {
  const text = [
    JSON.stringify({ index: 1, message: { role: 'assistant', content: 'legacy' } }),
    JSON.stringify({
      event_schema_version: 1,
      type: 'message_appended',
      session_id: 'other-session',
      index: 2,
      message: { role: 'assistant', content: 'wrong session' },
    }),
    '',
  ].join('\n')

  const result = readTranscriptEventLines(text, 'facefeedfacefeed')
  expect(result.events).toEqual([
    transcriptMessageAppendedEvent('facefeedfacefeed', 1, { role: 'assistant', content: 'legacy' }),
  ])
  expect(result.malformedLines).toBe(1)
  expect(result.partialTail).toBeFalse()
})

test('event reader preserves the valid prefix and reports a torn final line', () => {
  const first = encodeTranscriptEvent(transcriptMessageAppendedEvent(
    'facefeedfacefeed',
    0,
    { role: 'user', content: 'hello' },
  ))
  const result = readTranscriptEventLines(`${first}{"event_schema_version":1`, 'facefeedfacefeed')

  expect(result.events).toHaveLength(1)
  const event = result.events[0]
  expect(event?.type).toBe('message_appended')
  expect(event && 'index' in event ? event.index : -1).toBe(0)
  expect(result.malformedLines).toBe(0)
  expect(result.partialTail).toBeTrue()
})

test('event reader treats complete JSON without a newline delimiter as a partial tail', () => {
  const unterminated = encodeTranscriptEvent(transcriptMessageAppendedEvent(
    'facefeedfacefeed',
    0,
    { role: 'user', content: 'hello' },
  )).trimEnd()

  expect(readTranscriptEventLines(unterminated, 'facefeedfacefeed')).toEqual({
    events: [],
    malformedLines: 0,
    partialTail: true,
  })
})

test('current events require stable IDs and safe positive sequences', () => {
  const current = {
    event_schema_version: 1,
    type: 'message_appended',
    session_id: 'facefeedfacefeed',
    event_id: 'event-00000001',
    sequence: 1,
    index: 0,
    message: { role: 'user', content: 'hello' },
  }

  expect(readTranscriptEventLines(`${JSON.stringify(current)}\n`, 'facefeedfacefeed').events[0])
    .toMatchObject({ eventId: 'event-00000001', sequence: 1 })
  expect(readTranscriptEventLines(`${JSON.stringify({ ...current, event_id: '' })}\n`, 'facefeedfacefeed').malformedLines)
    .toBe(1)
  expect(readTranscriptEventLines(`${JSON.stringify({ ...current, sequence: 0 })}\n`, 'facefeedfacefeed').malformedLines)
    .toBe(1)
})

test('event factory rejects unsafe indexes', () => {
  expect(() => transcriptMessageAppendedEvent('facefeedfacefeed', -1, {})).toThrow(
    'invalid transcript message event',
  )
  expect(() => transcriptMessageAppendedEvent('facefeedfacefeed', Number.NaN, {})).toThrow(
    'invalid transcript message event',
  )
})
