// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'
import { encodeTranscriptEvent, transcriptMessageAppendedEvent } from '../src/session/transcriptEventLog.js'
import { inspectTranscriptEventLog } from '../src/session/transcriptEventInspection.js'

const sessionId = 'facefeedfacefeed'

function event(index: number, sequence: number) {
  return transcriptMessageAppendedEvent(sessionId, index, { role: 'user', content: String(index) }, {
    eventId: `event-${sequence}`, sequence,
  })
}

test('inspection reports sequence gaps, duplicate IDs and sequences, malformed rows, and torn tails', () => {
  const text = [
    encodeTranscriptEvent(event(0, 1)),
    encodeTranscriptEvent({ ...event(1, 2), eventId: 'event-1' }),
    encodeTranscriptEvent(event(2, 4)),
    encodeTranscriptEvent({ ...event(3, 4), eventId: 'event-4b' }),
    '{bad json}\n',
    '{"event_schema_version":1',
  ].join('')

  expect(inspectTranscriptEventLog(new TextEncoder().encode(text), sessionId)).toEqual({
    duplicateEventIds: ['event-1'],
    duplicateSequences: [4],
    eventCount: 4,
    firstSequence: 1,
    gaps: [{ from: 3, to: 3 }],
    lastSequence: 4,
    malformedLines: 1,
    partialTail: true,
    sessionId,
  })
})

test('store inspection distinguishes a missing event log from an empty healthy log', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-event-inspection-'))
  try {
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    expect(await store.inspectEventLog(sessionId)).toEqual({ kind: 'missing', sessionId })
    await Bun.write(store.journalPathFor(sessionId), '')
    expect(await store.inspectEventLog(sessionId)).toMatchObject({
      kind: 'inspected',
      report: { eventCount: 0, malformedLines: 0, partialTail: false, sessionId },
    })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
