// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { SSEParser, parseSseStream } from '../src/streaming/sse.js'

test('SSE parser incrementally assembles split multi-line events and tracks the latest id', () => {
  const parser = new SSEParser()

  parser.feed('id: event-42\r\nevent: tool_call\r\ndata: first')
  expect(parser.drain()).toEqual([])

  parser.feed(' line\r\ndata: second line\r\nretry: 1500\r\n\r\n')
  expect(parser.drain()).toEqual([{
    event: 'tool_call',
    data: 'first line\nsecond line',
    id: 'event-42',
    retry: 1500,
  }])
  expect(parser.lastEventId).toBe('event-42')
  expect(parser.drain()).toEqual([])
})

test('SSE parser ignores comments, resets invalid retry values, and does not leak record state', () => {
  const parser = new SSEParser()

  parser.feed(': keepalive\nretry: not-a-number\ndata: one\n\n')
  parser.feed('data: two\n\n')

  expect(parser.drain()).toEqual([
    { event: 'message', data: 'one', id: '', retry: undefined },
    { event: 'message', data: 'two', id: '', retry: undefined },
  ])
})

test('finite SSE parsing flushes an unterminated final record', () => {
  const events = [...parseSseStream(['event: notice\ndata: queued', ''])]

  expect(events).toEqual([{
    event: 'notice',
    data: 'queued',
    id: '',
    retry: undefined,
  }])
})

test('named events without data are never dispatched', () => {
  const parser = new SSEParser()

  parser.feed('event: ping\n\n')
  parser.feed('event: tool_call\nid: late-id\n\n')
  parser.feed('data: real\n\n')

  expect(parser.drain()).toEqual([
    { event: 'message', data: 'real', id: '', retry: undefined },
  ])
  expect(parser.lastEventId).toBe('')
})

test('only a single leading space is stripped from field values', () => {
  const parser = new SSEParser()

  parser.feed('data:  x\n\ndata:\t y\n\ndata:z\n\n')

  expect(parser.drain()).toEqual([
    { event: 'message', data: ' x', id: '', retry: undefined },
    { event: 'message', data: '\t y', id: '', retry: undefined },
    { event: 'message', data: 'z', id: '', retry: undefined },
  ])
})

test('bare carriage returns terminate lines without merging records', () => {
  const parser = new SSEParser()

  parser.feed('data: one\r\rdata: two\r')
  expect(parser.drain()).toEqual([{ event: 'message', data: 'one', id: '', retry: undefined }])

  // A trailing '\r' is held because it may be half of a chunk-split '\r\n'.
  parser.feed('\rdata: three\r\n\r\n')
  expect(parser.drain()).toEqual([
    { event: 'message', data: 'two', id: '', retry: undefined },
    { event: 'message', data: 'three', id: '', retry: undefined },
  ])
})

test('empty default records do not overwrite the last event id', () => {
  const parser = new SSEParser()

  parser.feed('id: saved\ndata: value\n\n')
  parser.feed('id: discarded\n\n')

  expect(parser.drain()).toHaveLength(1)
  expect(parser.lastEventId).toBe('saved')
})
