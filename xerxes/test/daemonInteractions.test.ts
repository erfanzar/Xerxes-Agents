// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { DaemonInteractionBoard } from '../src/daemon/interactions.js'
import { ApprovalScope, ApprovalStore } from '../src/security/approvals.js'
import type { PermissionRequest } from '../src/streaming/events.js'

test('daemon interaction board persists approve-for-session only until the session is discarded', async () => {
  const board = new DaemonInteractionBoard()
  const sessionId = 'session-1'
  const request: PermissionRequest = {
    requestId: 'permission-1',
    description: 'write a file',
    inputs: {},
    toolCall: { id: 'call-1', type: 'function', function: { name: 'WriteFile', arguments: {} } },
  }
  const first = board.permissionBroker(sessionId).request(request)
  expect(board.respondPermission(request.requestId, 'approve_for_session')).toBe(true)
  await expect(first).resolves.toBe('approve_for_session')

  const repeated = await board.permissionBroker(sessionId).request({ ...request, requestId: 'permission-2' })
  expect(repeated).toBe('approve')
  expect(board.pendingPermissionIds()).toEqual([])

  board.cancelSession(sessionId)
  const third = board.permissionBroker(sessionId).request({ ...request, requestId: 'permission-3' })
  expect(board.pendingPermissionIds()).toEqual(['permission-3'])
  expect(board.respondPermission('permission-3', 'reject')).toBe(true)
  await expect(third).resolves.toBe('reject')
})

test('daemon interaction board validates choice-only questions and emits a v35 request', async () => {
  const board = new DaemonInteractionBoard()
  const events: Array<{ type: string; payload: Record<string, unknown> }> = []
  const release = board.bind('session-2', event => events.push(event))
  try {
    const pending = board.ask('session-2', { question: 'Pick one', options: ['yes', 'no'], allowFreeform: false })
    const requestId = String(events[0]?.payload.id)
    expect(events).toEqual([{
      type: 'question_request',
      payload: {
        id: requestId,
        tool_call_id: '',
        questions: [{ id: 'answer', question: 'Pick one', options: ['yes', 'no'], allow_free_form: false }],
      },
    }])
    expect(board.respondQuestion(requestId, { answer: 'other' })).toBe(false)
    expect(board.respondQuestion(requestId, { answer: 'yes' })).toBe(true)
    await expect(pending).resolves.toBe('yes')
  } finally {
    release()
  }
})

test('daemon interaction board can use an explicit persistent approval store without exposing call arguments', async () => {
  const store = new ApprovalStore()
  const board = new DaemonInteractionBoard({ approvalStore: store })
  const sessionId = 'persistent-session'
  const request: PermissionRequest = {
    requestId: 'persistent-1',
    description: 'write a file',
    inputs: { api_key: 'not-persisted' },
    toolCall: { id: 'call-1', type: 'function', function: { name: 'WriteFile', arguments: { api_key: 'not-persisted' } } },
  }
  const first = board.permissionBroker(sessionId).request(request)
  expect(board.respondPermission(request.requestId, 'always')).toBe(true)
  await expect(first).resolves.toBe('approve_for_session')
  expect(store.list()).toHaveLength(1)
  expect(store.list()[0]).toMatchObject({ toolName: 'WriteFile', scope: ApprovalScope.ALWAYS, granted: true })
  expect(JSON.stringify(store.list())).not.toContain('not-persisted')

  const repeated = await board.permissionBroker('fresh-session').request({ ...request, requestId: 'persistent-2' })
  expect(repeated).toBe('approve')
  expect(board.pendingPermissionIds()).toEqual([])
})
