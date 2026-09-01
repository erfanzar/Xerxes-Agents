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
})
