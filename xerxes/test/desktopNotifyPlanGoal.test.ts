// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { goalPhaseNotification, notificationFor } from '../src/desktop/main/notify.js'

test('a plan exit request notifies as Plan ready, not as a tool approval', () => {
  const ping = notificationFor({ type: 'approval_request', payload: { session_id: 's1', tool_name: 'ExitPlanModeTool', description: 'ExitPlanModeTool {}' } })
  expect(ping).toMatchObject({ title: 'Plan ready', target: { sessionId: 's1' } })
  expect(notificationFor({ type: 'approval_request', payload: { tool_name: 'exec_command', description: 'rm x' } }))
    .toMatchObject({ title: 'Approval needed' })
})

test('a goal pings once when it becomes complete or blocked, never on a repeated phase', () => {
  expect(goalPhaseNotification('active', { session_id: 's1', goal: 'Ship it', goal_phase: 'complete' }))
    .toMatchObject({ title: 'Goal complete', body: 'Ship it', target: { sessionId: 's1' } })
  expect(goalPhaseNotification('active', { goal: 'Ship it', goal_phase: 'blocked' }))
    .toMatchObject({ title: 'Goal blocked — needs you' })
  expect(goalPhaseNotification('complete', { goal_phase: 'complete' })).toBeNull()
  expect(goalPhaseNotification(undefined, { goal_phase: 'active' })).toBeNull()
  expect(goalPhaseNotification('active', {})).toBeNull()
})
