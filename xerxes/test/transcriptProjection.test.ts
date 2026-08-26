// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  transcriptContextComposedEvent,
  transcriptMessageAppendedEvent,
  transcriptPolicyDecidedEvent,
  transcriptRequestPreparedEvent,
  transcriptToolCompletedEvent,
  transcriptToolStartedEvent,
  transcriptTurnCompletedEvent,
  transcriptTurnStartedEvent,
} from '../src/session/transcriptEventLog.js'
import { projectTranscriptEvents } from '../src/session/transcriptProjection.js'

const sessionId = 'facefeedfacefeed'
const turnId = 'turn-1'

function identity(sequence: number) {
  return { eventId: `event-${sequence}`, sequence }
}

test('typed lifecycle events deterministically project turn, request, context, policy, and tool state', () => {
  const events = [
    transcriptTurnStartedEvent(sessionId, turnId, { mode: 'objective' }, identity(1)),
    transcriptRequestPreparedEvent(sessionId, turnId, {
      model: 'openai/gpt-test',
      provider: 'openai',
      ephemeralFields: ['provider_api_key'],
    }, identity(2)),
    transcriptContextComposedEvent(sessionId, turnId, {
      messageCount: 4,
      modelVisible: true,
      sources: ['transcript', 'project_context'],
    }, identity(3)),
    transcriptPolicyDecidedEvent(sessionId, turnId, {
      action: 'tool.execute',
      decision: 'allow',
      reason: 'approved capability',
    }, identity(4)),
    transcriptToolStartedEvent(sessionId, turnId, {
      callId: 'call-1',
      name: 'ReadFile',
      arguments: { file_path: 'README.md' },
    }, identity(5)),
    transcriptToolCompletedEvent(sessionId, turnId, {
      callId: 'call-1',
      name: 'ReadFile',
      ok: true,
      result: { bytes: 42 },
    }, identity(6)),
    transcriptMessageAppendedEvent(sessionId, 0, { role: 'assistant', content: 'done' }, identity(7)),
    transcriptTurnCompletedEvent(sessionId, turnId, {
      stopReason: 'completed',
      usage: { inputTokens: 10, outputTokens: 2 },
    }, identity(8)),
  ]

  const projection = projectTranscriptEvents(events)
  expect(projection).toEqual({
    lastSequence: 8,
    messages: [{ role: 'assistant', content: 'done' }],
    turns: [{
      context: { messageCount: 4, modelVisible: true, sources: ['transcript', 'project_context'] },
      ended: true,
      mode: 'objective',
      policies: [{ action: 'tool.execute', decision: 'allow', reason: 'approved capability' }],
      request: {
        model: 'openai/gpt-test',
        provider: 'openai',
        ephemeralFields: ['provider_api_key'],
      },
      stopReason: 'completed',
      tools: [{
        arguments: { file_path: 'README.md' },
        callId: 'call-1',
        name: 'ReadFile',
        ok: true,
        result: { bytes: 42 },
        status: 'completed',
      }],
      turnId,
      usage: { inputTokens: 10, outputTokens: 2 },
    }],
  })
  expect(projectTranscriptEvents([...events].reverse())).toEqual(projection)
})

test('projection rejects duplicate or gapped current event sequences', () => {
  const first = transcriptTurnStartedEvent(sessionId, turnId, {}, identity(1))
  const duplicate = transcriptTurnCompletedEvent(sessionId, turnId, { stopReason: 'completed' }, identity(1))
  const gap = transcriptTurnCompletedEvent(sessionId, turnId, { stopReason: 'completed' }, identity(3))

  expect(() => projectTranscriptEvents([first, duplicate])).toThrow('duplicate transcript event sequence 1')
  expect(() => projectTranscriptEvents([first, gap])).toThrow('transcript event sequence gap: expected 2, received 3')
})

test('projection marks incomplete turns and tools without inventing terminal success', () => {
  const projection = projectTranscriptEvents([
    transcriptTurnStartedEvent(sessionId, turnId, {}, identity(1)),
    transcriptToolStartedEvent(sessionId, turnId, {
      callId: 'call-1', name: 'ReadFile', arguments: {},
    }, identity(2)),
  ])

  expect(projection.turns[0]?.ended).toBeFalse()
  expect(projection.turns[0]?.stopReason).toBeUndefined()
  expect(projection.turns[0]?.tools[0]?.status).toBe('running')
})
