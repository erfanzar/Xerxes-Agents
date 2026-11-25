// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { notificationDedupKey, notificationFor, shouldNotify, unwrapAttentionEvent } from '../src/desktop/main/notify.js'

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
    const foreground = { enabled: true, anyWindowFocused: true }
    expect(shouldNotify(event, foreground)?.title).toBe('Task finished')
    expect(shouldNotify({ type: 'approval_request', payload: { name: 'WriteFile' } }, foreground)?.title).toBe('Approval needed')
    expect(shouldNotify({ type: 'question_request', payload: { questions: [{ question: 'Which?' }] } }, foreground)?.body).toBe('Which?')
    const goal = { type: 'status_update', payload: { goal_phase: 'blocked' } }
    expect(shouldNotify(goal, foreground, 'active')?.title).toBe('Goal blocked — needs you')
    expect(shouldNotify(goal, foreground, 'blocked')).toBeNull()
    expect(shouldNotify(goal, { ...foreground, enabled: false }, 'active')).toBeNull()
    const artifact = { type: 'tool_result', payload: { name: 'ArtifactTool', result: 'Created artifact "Plan" (id a, version 1)' } }
    expect(shouldNotify(artifact, foreground)?.title).toBe('New artifact')
    expect(shouldNotify(artifact, { ...foreground, enabled: false })).toBeNull()
    expect(shouldNotify(artifact, { ...foreground, anyWindowFocused: false })?.title).toBe('New artifact')
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

  test.each([
    ['completed', 'Task finished'], ['objective_verified', 'Task finished'],
    ['provider_failed', 'Task failed'], ['turn_failed', 'Task failed'],
    ['aborted', 'Task stopped'], ['tool_budget_exhausted', 'Task stopped'],
  ])('daemon turn_end stop_reason %s reports %s in foreground', (stop_reason, title) => {
    expect(shouldNotify({ type: 'turn_end', payload: { session_id: 's1', stop_reason } }, { enabled: true, anyWindowFocused: true }))
      .toMatchObject({ title, target: { sessionId: 's1' } })
  })

  test('interrupted turns report stopped, unstarted turns stay quiet', () => {
    // Mirrors the TUI bell policy: a cancelled turn is not a completion, and
    // an unstarted submission has nothing to report at all.
    expect(notificationFor({ type: 'turn_end', payload: { cancelled: true } }))
      .toMatchObject({ title: 'Task stopped' })
    expect(notificationFor({ type: 'turn_end', payload: { cancelled: true, unstarted: true } })).toBeNull()
    expect(notificationFor({ type: 'turn_end', payload: { unstarted: true } })).toBeNull()
  })

  test('ArtifactTool return_value and external ACP artifact_id both notify with clickable artifact targets', () => {
    const nativeReturnVal = notificationFor({
      type: 'tool_result',
      payload: {
        session_id: 's2',
        name: 'ArtifactTool',
        permitted: true,
        return_value: 'Created artifact "Design Spec" (id spec123, version 1).',
      },
    })
    expect(nativeReturnVal).toMatchObject({
      title: 'New artifact',
      kind: 'artifact',
      target: { sessionId: 's2', panel: 'docs', artifactId: 'spec123' },
    })

    const agyCreated = notificationFor({
      type: 'tool_result',
      payload: {
        session_id: 's3',
        name: 'Write implementation_plan.md',
        title: 'Write implementation_plan.md',
        artifact_id: 'agy:implementation_plan.md',
        permitted: true,
      },
    })
    expect(agyCreated).toMatchObject({
      title: 'New artifact',
      kind: 'artifact',
      body: 'implementation plan — click to read it',
      target: { sessionId: 's3', panel: 'docs', artifactId: 'agy:implementation_plan.md' },
    })

    const agyEdited = notificationFor({
      type: 'tool_result',
      payload: {
        session_id: 's3',
        name: 'Edit walkthrough.md',
        artifact_id: 'agy:walkthrough.md',
        permitted: true,
      },
    })
    expect(agyEdited).toMatchObject({
      title: 'Artifact updated',
      kind: 'artifact',
      target: { sessionId: 's3', panel: 'docs', artifactId: 'agy:walkthrough.md' },
    })
  })

  test('session_attention unwraps background thread events, enriches titles, and dedups with direct events', () => {
    const wrappedFinished = {
      type: 'session_attention',
      payload: {
        session_id: 'bg-1',
        session_title: 'Refactor auth',
        event_type: 'turn_end',
        payload: { stop_reason: 'completed' },
      },
    }
    expect(unwrapAttentionEvent(wrappedFinished)).toEqual({
      type: 'turn_end',
      payload: { session_id: 'bg-1', session_title: 'Refactor auth', stop_reason: 'completed' },
    })
    expect(shouldNotify(wrappedFinished, { enabled: true, anyWindowFocused: true })).toMatchObject({
      title: 'Task finished',
      body: 'Refactor auth — Completed.',
      kind: 'finished',
      target: { sessionId: 'bg-1' },
    })

    const wrappedApproval = {
      type: 'session_attention',
      payload: {
        session_id: 'bg-2',
        session_title: 'Deploy fix',
        event_type: 'approval_request',
        payload: { id: 'req-9', title: 'Run bun test' },
      },
    }
    const approvalDecision = shouldNotify(wrappedApproval, { enabled: true, anyWindowFocused: true })
    expect(approvalDecision).toMatchObject({
      title: 'Approval needed',
      body: 'Deploy fix — Run bun test',
      kind: 'attention',
      urgent: true,
      target: { sessionId: 'bg-2' },
    })

    const directApproval = {
      type: 'approval_request',
      payload: { session_id: 'bg-2', id: 'req-9', title: 'Run bun test' },
    }
    const directDecision = shouldNotify(directApproval, { enabled: true, anyWindowFocused: true })!
    expect(notificationDedupKey(wrappedApproval, approvalDecision!)).toBe(
      notificationDedupKey(directApproval, directDecision),
    )

    const wrappedQuestion = {
      type: 'session_attention',
      payload: {
        session_id: 'bg-3',
        session_title: 'Schema migration',
        event_type: 'question_request',
        payload: { message: 'Drop legacy index?' },
      },
    }
    expect(shouldNotify(wrappedQuestion, { enabled: true, anyWindowFocused: true })).toMatchObject({
      title: 'Xerxes has a question',
      body: 'Schema migration — Drop legacy index?',
      kind: 'attention',
      urgent: true,
      target: { sessionId: 'bg-3' },
    })
  })

  test('desktop notifications use system OS notifications only (no in-app toast mirror) and expose System Settings IPC', async () => {
    const mainSrc = await Bun.file(new URL('../src/desktop/main.ts', import.meta.url)).text()
    const bridgeSrc = await Bun.file(new URL('../src/desktop/bridgeApi.ts', import.meta.url)).text()
    const appSrc = await Bun.file(new URL('../src/desktop/renderer/App.tsx', import.meta.url)).text()

    expect(mainSrc).not.toContain('sendToastToHost')
    expect(mainSrc).toContain("handle('native:notifications:open-settings'")
    expect(bridgeSrc).toContain('native:notifications:open-settings')
    expect(bridgeSrc).toContain('openNotificationSettings()')
    expect(appSrc).not.toContain('<NotificationToasts')
  })
})
