// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  BridgeSession,
  bridgeHistoryReplayRecords,
  type BridgeSessionStore,
} from '../src/bridge/session.js'
import { RESUME_REPLAY_SENTINEL } from '../src/session/resumeRepair.js'

class MemoryBridgeSessionStore implements BridgeSessionStore {
  readonly records = new Map<string, unknown>()
  readonly writes: Array<readonly [string, Readonly<Record<string, unknown>>]> = []

  read(sessionId: string): unknown | undefined {
    return this.records.get(sessionId)
  }

  write(sessionId: string, record: Readonly<Record<string, unknown>>): void {
    this.writes.push([sessionId, record])
    this.records.set(sessionId, record)
  }
}

test('bridge session saves a transcript-compatible record only through its injected store', async () => {
  const store = new MemoryBridgeSessionStore()
  const session = new BridgeSession({
    sessionId: 'a1b2c3d4',
    cwd: '/workspace',
    model: 'claude-sonnet-4-6',
    agentId: 'code',
    interactionMode: 'plan',
    planMode: true,
    workspace: 'workspace-a',
    clock: () => new Date('2026-07-13T10:00:00.000Z'),
    store,
  })
  session.appendMessage({ role: 'user', content: 'please inspect the bridge' })
  session.update({
    totalInputTokens: 12,
    totalOutputTokens: 7,
    turnCount: 1,
    thinkingContent: ['reasoned'],
    toolExecutions: [{ name: 'ReadFile' }],
  })

  const saved = await session.save()

  expect(store.writes).toHaveLength(1)
  expect(store.writes[0]).toEqual(['a1b2c3d4', saved])
  expect(saved).toMatchObject({
    format: 'xerxes-daemon-session',
    schema_version: 2,
    session_id: 'a1b2c3d4',
    key: 'a1b2c3d4',
    cwd: '/workspace',
    model: 'claude-sonnet-4-6',
    created_at: '2026-07-13T10:00:00.000Z',
    updated_at: '2026-07-13T10:00:00.000Z',
    agent_id: 'code',
    interaction_mode: 'plan',
    plan_mode: true,
    workspace: 'workspace-a',
    turn_count: 1,
    total_input_tokens: 12,
    total_output_tokens: 7,
  })
  expect(saved.messages).toEqual([{ role: 'user', content: 'please inspect the bridge' }])
})

test('bridge session resumes legacy records with descriptors but never automatically executes a tool', async () => {
  const store = new MemoryBridgeSessionStore()
  store.records.set('deadbeef', {
    session_id: 'deadbeef',
    model: 'gpt-4.1',
    created_at: '2026-07-01T00:00:00.000Z',
    cwd: '/saved-project',
    messages: [
      { role: 'user', content: 'read the file' },
      {
        role: 'assistant',
        content: 'Checking it. ASSISTANT_TOOL_CALLS: {"name":"hidden","input":{}}',
        tool_calls: [{ id: 'call-read', function: { name: 'ReadFile', arguments: '{"path":"README.md"}' } }],
      },
    ],
    turn_count: 1,
  })
  const session = new BridgeSession({
    sessionId: 'fresh000',
    cwd: '/current-project',
    model: 'fallback-model',
    clock: () => new Date('2026-07-13T11:00:00.000Z'),
    store,
  })

  const result = await session.resume('deadbeef')

  expect(result.status).toBe('resumed')
  expect(result.pendingResumeReplays).toEqual([
    { tool_call_id: 'call-read', name: 'ReadFile', arguments: '{"path":"README.md"}' },
  ])
  expect(session.snapshot.pendingResumeReplays).toEqual(result.pendingResumeReplays)
  expect(session.snapshot.messages).toEqual([
    { role: 'user', content: 'read the file' },
    {
      role: 'assistant',
      content: 'Checking it.',
      tool_calls: [{ id: 'call-read', function: { name: 'ReadFile', arguments: '{"path":"README.md"}' } }],
    },
    { role: 'tool', tool_call_id: 'call-read', content: RESUME_REPLAY_SENTINEL },
  ])
  expect(result.history).toEqual([
    { category: 'history', type: 'replay_user', severity: 'info', body: '✨ read the file' },
    { category: 'history', type: 'replay_assistant', severity: 'info', body: 'Checking it.' },
    {
      category: 'history',
      type: 'resumed',
      severity: 'info',
      body: '── resumed session deadbeef (2 messages) ──',
    },
  ])
  expect(store.writes).toEqual([])
})

test('bridge session leaves its state intact for missing or malformed persisted records', async () => {
  const store = new MemoryBridgeSessionStore()
  store.records.set('badrecord', { session_id: 'badrecord', messages: 'not an array' })
  const session = new BridgeSession({
    sessionId: 'aabbccdd',
    cwd: '/workspace',
    clock: () => new Date('2026-07-13T12:00:00.000Z'),
    store,
  })
  session.appendMessage({ role: 'user', content: 'keep me' })

  expect(await session.resume('missing')).toEqual({ status: 'missing', history: [], pendingResumeReplays: [] })
  expect(await session.resume('badrecord')).toEqual({ status: 'invalid', history: [], pendingResumeReplays: [] })
  expect(session.snapshot).toMatchObject({
    sessionId: 'aabbccdd',
    cwd: '/workspace',
    messages: [{ role: 'user', content: 'keep me' }],
  })
})

test('history projection mirrors bridge text handling and explicit repair remains host-controlled', () => {
  const records = bridgeHistoryReplayRecords([
    { role: 'user', content: [{ text: 'first' }, { type: 'image' }, 'second'] },
    { role: 'assistant', content: { text: 'answer ASSISTANT_TOOL_CALLS: {"name":"hidden","input":{}}' } },
    { role: 'tool', content: 'not a history record' },
  ], 'feedface')
  expect(records).toEqual([
    { category: 'history', type: 'replay_user', severity: 'info', body: '✨ first\nsecond' },
    { category: 'history', type: 'replay_assistant', severity: 'info', body: 'answer' },
    {
      category: 'history',
      type: 'resumed',
      severity: 'info',
      body: '── resumed session feedface (2 messages) ──',
    },
  ])

  const session = new BridgeSession({
    sessionId: 'feedface',
    cwd: '/workspace',
    clock: () => new Date('2026-07-13T13:00:00.000Z'),
    store: new MemoryBridgeSessionStore(),
  })
  session.appendMessage({ role: 'assistant', content: '', tool_calls: [{ id: 'call-a', name: 'ReadFile', input: {} }] })

  expect(session.snapshot.pendingResumeReplays).toEqual([])
  expect(session.repairInterruptedToolCalls()).toEqual([
    { tool_call_id: 'call-a', name: 'ReadFile', arguments: '{}' },
  ])
  expect(session.snapshot.messages.at(-1)).toEqual({
    role: 'tool',
    tool_call_id: 'call-a',
    content: RESUME_REPLAY_SENTINEL,
  })
})
