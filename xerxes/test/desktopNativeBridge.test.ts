// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { notificationFor, shouldNotify } from '../src/desktop/main/notify.js'

describe('native notification decisions', () => {
  test('turn_end, approval and question events earn pings; others stay quiet', () => {
    expect(notificationFor({ type: 'turn_end', payload: {} })).toMatchObject({ title: 'Task finished' })
    expect(notificationFor({ type: 'approval_request', payload: { description: 'rm -rf build' } }))
      .toMatchObject({ title: 'Approval needed', body: 'rm -rf build' })
    expect(notificationFor({
      type: 'question_request',
      payload: { questions: [{ question: 'Which database?' }] },
    })).toMatchObject({ title: 'Xerxes has a question', body: 'Which database?' })
    // Streaming noise never pings.
    expect(notificationFor({ type: 'text_part', payload: { text: 'partial' } })).toBeNull()
    expect(notificationFor({ type: 'status_update', payload: {} })).toBeNull()
  })

  test('an approval without a describable action stays quiet instead of pinging vaguely', () => {
    expect(notificationFor({ type: 'approval_request', payload: {} })).toBeNull()
    expect(notificationFor({ type: 'question_request', payload: { questions: [{}] } })).toBeNull()
  })

  test('long descriptions truncate with an ellipsis', () => {
    const ping = notificationFor({ type: 'approval_request', payload: { description: 'x'.repeat(300) } })
    expect(ping?.body.length).toBeLessThanOrEqual(140)
    expect(ping?.body.endsWith('…')).toBe(true)
  })

  test('the gate: preference first, then window focus', () => {
    const event = { type: 'turn_end', payload: {} }
    expect(shouldNotify(event, { enabled: false, anyWindowFocused: false })).toBeNull()
    expect(shouldNotify(event, { enabled: true, anyWindowFocused: true })).toBeNull()
    expect(shouldNotify(event, { enabled: true, anyWindowFocused: false })).toMatchObject({ title: 'Task finished' })
  })

  test('a saved artifact pings with a click target that opens it; its task is the target session', () => {
    const created = notificationFor({ type: 'tool_result', payload: { session_id: 's1', name: 'ArtifactTool', permitted: true, result: 'Created artifact "Plan (v2)" (id c2b490e7ea, version 1). It is in the Artifacts panel.' } })
    expect(created).toMatchObject({ title: 'New artifact', target: { sessionId: 's1', panel: 'docs', artifactId: 'c2b490e7ea' } })
    expect(created?.body).toContain('Plan (v2)')
    expect(notificationFor({ type: 'tool_result', payload: { name: 'ArtifactTool', result: 'Updated artifact "Plan" (id abc, version 3).' } }))
      .toMatchObject({ title: 'Artifact updated', body: expect.stringContaining('(v3)') })
    // Other tools, refused calls and failures stay quiet.
    expect(notificationFor({ type: 'tool_result', payload: { name: 'exec_command', result: 'Created artifact "x" (id a, version 1)' } })).toBeNull()
    expect(notificationFor({ type: 'tool_result', payload: { name: 'ArtifactTool', permitted: false, result: 'Created artifact "x" (id a, version 1)' } })).toBeNull()
    expect(notificationFor({ type: 'tool_result', payload: { name: 'ArtifactTool', result: 'Tool execution failed: no session' } })).toBeNull()
  })

  test('a finished, failed or waiting task carries its session as the click target', () => {
    expect(notificationFor({ type: 'turn_end', payload: { session_id: 's9' } })).toMatchObject({ title: 'Task finished', target: { sessionId: 's9' } })
    expect(notificationFor({ type: 'turn_end', payload: { session_id: 's9', error: 'provider 500' } })).toMatchObject({ title: 'Task failed', body: 'provider 500' })
    expect(notificationFor({ type: 'question_request', payload: { session_id: 's9', questions: [{ question: 'Which?' }] } })).toMatchObject({ target: { sessionId: 's9' } })
  })

  test('interrupted turns report stopped, unstarted turns stay quiet', () => {
    // Mirrors the TUI bell policy: a cancelled turn is not a completion, and
    // an unstarted submission has nothing to report at all.
    expect(notificationFor({ type: 'turn_end', payload: { cancelled: true } }))
      .toMatchObject({ title: 'Task stopped' })
    expect(notificationFor({ type: 'turn_end', payload: { cancelled: true, unstarted: true } })).toBeNull()
    expect(notificationFor({ type: 'turn_end', payload: { unstarted: true } })).toBeNull()
  })
})
