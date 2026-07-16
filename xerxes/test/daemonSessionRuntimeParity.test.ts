// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import {
  InMemoryDaemonRuntime,
  type DaemonEvent,
  type DaemonSession,
  type TurnRunner,
} from '../src/daemon/runtime.js'
import { claimSubagentConversation } from '../src/daemon/subagentConversations.js'

test('daemon runtime persists a project-scoped session and resumes only an explicit ID', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-daemon-runtime-parity-'))
  const projectDirectory = join(directory, 'project')
  const workspaceRoot = join(directory, 'agents')
  const sessionDirectory = join(directory, 'sessions')
  const runtime = new InMemoryDaemonRuntime(new RecordingRunner(), {
    currentProjectDirectory: projectDirectory,
    model: 'gpt-4o',
    sessionDirectory,
    workspaceRoot,
  })
  try {
    const session = await runtime.openSession('tui:default', 'researcher', { cwd: projectDirectory })
    await runtime.setSessionMode(session.sessionKey, 'researcher')
    const events: DaemonEvent[] = []

    await runtime.submitTurn(session.sessionKey, 'remember the native resume contract', event => events.push(event))

    expect(session.messages).toEqual([
      { role: 'user', content: 'remember the native resume contract' },
      { role: 'assistant', content: 'durable answer', thinking: 'inspect persisted state' },
    ])
    expect(session.thinkingContent).toEqual(['inspect persisted state'])
    expect(session.toolExecutions).toEqual([{ tool_call_id: 'tool-1', tool_name: 'ReadFile', output: 'README contents' }])
    expect(session.totalInputTokens).toBe(13)
    expect(session.totalOutputTokens).toBe(5)
    expect(events.map(event => event.type)).toEqual([
      'turn_begin',
      'think_part',
      'tool_result',
      'status_update',
      'text_part',
      'turn_end',
    ])

    const raw = JSON.parse(await readFile(join(sessionDirectory, `${session.id}.json`), 'utf8')) as Record<string, unknown>
    expect(raw).toMatchObject({
      session_id: session.id,
      key: 'tui:default',
      cwd: resolve(projectDirectory),
      workspace: resolve(workspaceRoot, 'researcher'),
      turn_count: 1,
      total_input_tokens: 13,
      total_output_tokens: 5,
      metadata: { title: 'remember the native resume contract' },
    })
    expect(await runtime.listSavedSessions()).toEqual([
      expect.objectContaining({
        id: session.id,
        key: 'tui:default',
        messageCount: 2,
        title: 'remember the native resume contract',
        turnCount: 1,
      }),
    ])
    await runtime.openSession('tui:empty')
    await runtime.flushSessions()
    expect(await runtime.listSavedSessions()).toHaveLength(1)

    const restarted = new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: projectDirectory,
      model: 'gpt-4o',
      sessionDirectory,
      workspaceRoot,
    })
    const freshSlot = await restarted.openSession('tui:default', 'researcher', { cwd: projectDirectory })
    expect(freshSlot.id).not.toBe(session.id)
    expect(freshSlot.messages).toEqual([])

    const resumed = await restarted.openSession(session.id, undefined, { cwd: projectDirectory, resume: true })
    expect(resumed).toMatchObject({
      id: session.id,
      sessionKey: session.id,
      agentId: 'researcher',
      cwd: resolve(projectDirectory),
      workspace: resolve(workspaceRoot, 'researcher'),
      interactionMode: 'researcher',
      turnCount: 1,
      totalInputTokens: 13,
      totalOutputTokens: 5,
      metadata: { title: 'remember the native resume contract' },
    })
    expect(resumed.messages).toEqual(session.messages)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('saved session listing exposes additive hierarchy metadata and limits roots after project and kind filters', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-daemon-session-hierarchy-'))
  const projectDirectory = join(directory, 'project-a')
  const otherProjectDirectory = join(directory, 'project-b')
  const sessionDirectory = join(directory, 'sessions')
  await mkdir(sessionDirectory, { recursive: true })

  const writeTranscript = async (
    sessionId: string,
    options: {
      readonly cwd: string
      readonly legacy?: boolean
      readonly metadata?: Record<string, unknown>
      readonly updatedAt: string
    },
  ) => {
    await writeFile(join(sessionDirectory, `${sessionId}.json`), JSON.stringify({
      ...(options.legacy ? {} : {
        format: 'xerxes-daemon-session',
        schema_version: 2,
      }),
      session_id: sessionId,
      key: sessionId,
      agent_id: 'default',
      cwd: options.cwd,
      workspace: '',
      updated_at: options.updatedAt,
      messages: [
        { role: 'user', content: `request ${sessionId}` },
        { role: 'assistant', content: `response ${sessionId}` },
      ],
      turn_count: 1,
      interaction_mode: 'code',
      plan_mode: false,
      total_input_tokens: 1,
      total_output_tokens: 1,
      metadata: options.metadata ?? {},
      thinking_content: [],
      tool_executions: [],
    }), 'utf8')
  }

  await writeTranscript('aaaabbbb0001', {
    cwd: projectDirectory,
    metadata: { model: 'root-model', project_root: projectDirectory, title: 'Root one' },
    updatedAt: '2026-07-17T00:05:00.000Z',
  })
  await writeTranscript('aaaabbbb0002', {
    cwd: projectDirectory,
    legacy: true,
    metadata: { title: 'Root two' },
    updatedAt: '2026-07-17T00:01:00.000Z',
  })
  await writeTranscript('aaaabbbb0003', {
    cwd: projectDirectory,
    metadata: {
      parent_session_id: 'aaaabbbb0001',
      project_root: projectDirectory,
      title: 'Regular branch',
    },
    updatedAt: '2026-07-17T00:00:00.000Z',
  })
  await writeTranscript('ccccdddd0001', {
    // A child may execute in a worktree while still belonging to its parent's project.
    cwd: join(directory, 'child-worktree'),
    metadata: {
      model: 'child-model',
      parent_session_id: 'aaaabbbb0001',
      project_root: projectDirectory,
      root_session_id: 'aaaabbbb0001',
      session_kind: 'subagent',
      status: 'running',
      subagent_id: 'subagent_child_one',
      title: 'Child one',
    },
    updatedAt: '2026-07-17T00:06:00.000Z',
  })
  await writeTranscript('eeeeffff0001', {
    cwd: otherProjectDirectory,
    metadata: { project_root: otherProjectDirectory, title: 'Other project' },
    updatedAt: '2026-07-17T00:07:00.000Z',
  })
  await writeTranscript('eeeeffff0002', {
    cwd: otherProjectDirectory,
    metadata: {
      parent_session_id: 'eeeeffff0001',
      project_root: otherProjectDirectory,
      root_session_id: 'eeeeffff0001',
      session_kind: 'subagent',
      status: 'completed',
      subagent_id: 'subagent_other_project',
      title: 'Other project child',
    },
    updatedAt: '2026-07-17T00:08:00.000Z',
  })

  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: projectDirectory,
    sessionDirectory,
  })
  try {
    const roots = await runtime.listSavedSessions(0, {
      kind: 'main',
      projectDirectory,
    })
    expect(roots.map(session => session.id)).toEqual(['aaaabbbb0001', 'aaaabbbb0002', 'aaaabbbb0003'])
    expect(roots[0]).toMatchObject({
      kind: 'main',
      model: 'root-model',
    })

    const children = await runtime.listSavedSessions(0, {
      kind: 'subagent',
      projectDirectory,
    })
    expect(children).toEqual([
      expect.objectContaining({
        id: 'ccccdddd0001',
        kind: 'subagent',
        model: 'child-model',
        parentSessionId: 'aaaabbbb0001',
        rootSessionId: 'aaaabbbb0001',
        resumable: true,
        status: 'interrupted',
        subagentId: 'subagent_child_one',
      }),
    ])

    const expandedRoot = await runtime.listSavedSessions(1, {
      includeSubagents: true,
      projectDirectory,
    })
    expect(expandedRoot.map(session => session.id)).toEqual([
      'ccccdddd0001',
      'aaaabbbb0001',
    ])
    expect((await runtime.listSavedSessions(1, { projectDirectory })).map(session => session.id)).toEqual([
      'aaaabbbb0001',
    ])
    expect((await runtime.listSavedSessions(0, {
      kind: 'all',
      projectDirectory,
    })).map(session => session.id)).toEqual([
      'ccccdddd0001',
      'aaaabbbb0001',
      'aaaabbbb0002',
      'aaaabbbb0003',
    ])

    const releaseChild = claimSubagentConversation('ccccdddd0001')
    try {
      expect(await runtime.listSavedSessions(0, {
        kind: 'subagent',
        projectDirectory,
      })).toEqual([
        expect.objectContaining({
          id: 'ccccdddd0001',
          resumable: false,
          status: 'running',
        }),
      ])
      await expect(runtime.openSession('ccccdddd0001', undefined, {
        cwd: projectDirectory,
        resume: true,
      })).rejects.toThrow('still owned by a running subagent')
    } finally {
      releaseChild()
    }
    await expect(runtime.openSession('ccccdddd0001', undefined, {
      cwd: otherProjectDirectory,
      resume: true,
    })).rejects.toThrow('different project')
    const resumedChild = await runtime.openSession('ccccdddd0001', undefined, {
      cwd: projectDirectory,
      resume: true,
    })
    expect(resumedChild.metadata.session_kind).toBe('subagent')
    expect(resumedChild.cwd).toBe(resolve(directory, 'child-worktree'))
    await expect(runtime.openSession('ccccdddd0001', undefined, {
      cwd: otherProjectDirectory,
      resume: true,
    })).rejects.toThrow('different project')
    expect(resumedChild.cwd).toBe(resolve(directory, 'child-worktree'))
    expect(() => claimSubagentConversation('ccccdddd0001')).toThrow('open as a direct session')
    runtime.evictSession('ccccdddd0001')
    const releasedNativeAfterEviction = claimSubagentConversation('ccccdddd0001')
    releasedNativeAfterEviction()
    const resumedOtherProjectChild = await runtime.openSession('eeeeffff0002', undefined, {
      cwd: otherProjectDirectory,
      resume: true,
    })
    expect(resumedOtherProjectChild.metadata).toMatchObject({
      project_root: otherProjectDirectory,
      session_kind: 'subagent',
      subagent_id: 'subagent_other_project',
    })
    runtime.evictSession('eeeeffff0002')
    expect(await runtime.listSavedSessions(0, {
      kind: 'main',
      projectDirectory: otherProjectDirectory,
    })).toEqual([
      expect.objectContaining({ id: 'eeeeffff0001', kind: 'main' }),
    ])
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('a corrupt explicit resume file creates an empty native session instead of crashing', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-daemon-corrupt-resume-parity-'))
  const sessionDirectory = join(directory, 'sessions')
  const sessionId = 'deadbeef'
  await mkdir(sessionDirectory, { recursive: true })
  await writeFile(join(sessionDirectory, `${sessionId}.json`), '{not valid json', 'utf8')
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory,
  })
  try {
    const session = await runtime.openSession(sessionId, undefined, { resume: true })

    expect(session).toMatchObject({ id: sessionId, sessionKey: sessionId, messages: [], turnCount: 0 })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('daemon cancellation drains queued steering, clears active state, and does not invent tool results', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-daemon-cancel-parity-'))
  const runner = new CancelAwareRunner()
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, 'sessions'),
  })
  try {
    const session = await runtime.openSession('tui:cancel')
    const events: DaemonEvent[] = []
    const turn = runtime.submitTurn(session.sessionKey, 'begin long task', event => events.push(event))
    await runner.waiting

    expect(runtime.cancelTurn(session.sessionKey)).toBe(true)
    expect(runtime.steerTurn(session.sessionKey, 'preserve the evidence')).toBe(true)
    expect(runtime.cancelTurn('missing')).toBe(false)
    await turn

    expect(session.status).toBe('idle')
    expect(session.activeTurnId).toBe('')
    expect(session.cancelRequested).toBe(true)
    expect(session.toolExecutions).toEqual([])
    expect(session.messages).toEqual([
      { role: 'user', content: 'begin long task' },
      { role: 'assistant', content: 'waiting for cancellation' },
      { role: 'user', content: '[steer from user saved for next turn]\npreserve the evidence' },
    ])
    expect(events.find(event => event.type === 'turn_end')?.payload).toMatchObject({
      session_id: session.id,
      cancelled: true,
    })
    expect(events.find(event => event.type === 'notification')?.payload.message).toBe(
      'Saved 1 steer for the next turn.',
    )
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('interaction mode is scoped to a daemon session rather than process-global runtime settings', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-daemon-mode-parity-'))
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, 'sessions'),
  })
  try {
    const first = await runtime.openSession('tui:first')
    const second = await runtime.openSession('tui:second')

    const changed = await runtime.setSessionMode(first.sessionKey, 'researcher')

    expect(changed).toMatchObject({ interactionMode: 'researcher', planMode: false })
    expect(runtime.sessionStatus(first.sessionKey)).toMatchObject({ interactionMode: 'researcher', planMode: false })
    expect(runtime.sessionStatus(second.sessionKey)).toMatchObject({ interactionMode: 'code', planMode: false })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

class RecordingRunner implements TurnRunner {
  async *run(_session: DaemonSession, _text: string, _signal: AbortSignal): AsyncGenerator<DaemonEvent> {
    yield { type: 'think_part', payload: { think: 'inspect persisted state' } }
    yield {
      type: 'tool_result',
      payload: { tool_call_id: 'tool-1', tool_name: 'ReadFile', output: 'README contents' },
    }
    yield { type: 'status_update', payload: { usage: { input_tokens: 13, output_tokens: 5 } } }
    yield { type: 'text_part', payload: { text: 'durable answer' } }
  }
}

class CancelAwareRunner implements TurnRunner {
  private releaseWaiting: (() => void) | undefined
  readonly waiting = new Promise<void>(resolve => { this.releaseWaiting = resolve })

  async *run(_session: DaemonSession, _text: string, signal: AbortSignal): AsyncGenerator<DaemonEvent> {
    yield { type: 'text_part', payload: { text: 'waiting for cancellation' } }
    await new Promise<void>(resolveAbort => {
      if (signal.aborted) {
        resolveAbort()
        return
      }
      signal.addEventListener('abort', () => resolveAbort(), { once: true })
      this.releaseWaiting?.()
    })
  }
}
