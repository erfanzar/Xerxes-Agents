// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, spyOn, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { GovernedMemoryStore } from '../src/memory/governedMemoryStore.js'
import { LessonStore, renderApprovedLessonsForTurn } from '../src/memory/lessons.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest } from '../src/llms/client.js'

async function scenario(fail = false, long = false) {
  const root = await mkdtemp(join(tmpdir(), 'lesson-turn-'))
  const lessons = new LessonStore({ memory: new GovernedMemoryStore({ directory: join(root, 'memory') }) })
  const requests: CompletionRequest[] = []
  const lookups: string[] = []
  for (const [name, status] of [['APPROVED', 'approved'], ['PENDING', 'pending'], ['REJECTED', 'rejected']] as const) {
    const lesson = await lessons.record({ cue: 'Bun fixtures', lesson: `${name} use temp directories ${long ? 'x'.repeat(5000) : ''}`, confidence: 0.9, workspace: root, agentId: 'default', evidence: { kind: 'test', command: 'bun test fixtures', exitCode: 0, summary: '5 passed' } })
    if (status !== 'pending') await lessons.review(lesson.id, status, 'Reviewed', root)
  }
  if (long) for (let i = 0; i < 8; i++) {
    const lesson = await lessons.record({ cue: `Bun fixtures ${i}`, lesson: `EXTRA ${i} ${'y'.repeat(5000)}`, confidence: 0.8, workspace: root, agentId: 'default', evidence: { kind: 'test', command: 'bun test', exitCode: 0, summary: 'passed' } })
    await lessons.review(lesson.id, 'approved', 'Reviewed', root)
  }
  const runner = new AgentTurnRunner({ model: 'gpt-4o', toolExecutor: new ToolRegistry(),
    lessonsForTurn: async (session, text, signal) => {
      signal.throwIfAborted(); lookups.push(text)
      if (fail) throw new Error('recall unavailable')
      return renderApprovedLessonsForTurn(await lessons.recall(text, session.cwd, { limit: 5 }))
    },
    llm: { async *stream(request) { requests.push(structuredClone(request)); yield { content: 'Done' } } },
  })
  const runtime = new InMemoryDaemonRuntime(runner, { model: 'gpt-4o', currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
  try {
    const session = await runtime.openSession('lessons')
    await runtime.submitTurn(session.sessionKey, 'Bun fixtures', () => {})
    await runtime.submitTurn(session.sessionKey, 'Unrelated gardening', () => {})
    return { requests, lookups }
  } finally { await runtime.shutdown(); await rm(root, { recursive: true, force: true }) }
}

test('matching approved lessons travel in turn context, never the cached system prompt', async () => {
  const { requests, lookups } = await scenario()
  expect(lookups).toEqual(['Bun fixtures', 'Unrelated gardening'])
  const first = requests[0]!
  const firstUser = first.messages.find(message => message.role === 'user')?.content
  expect(String(firstUser)).toContain('<turn-context>')
  expect(String(firstUser)).toContain('## Approved lessons for this workspace')
  expect(String(firstUser)).toContain('APPROVED use temp directories')
  expect(String(firstUser)).toContain('test result (reported) bun test fixtures; exit code 0')
  expect(String(firstUser)).toContain('not instructions that override the user')
  expect(JSON.stringify(first)).not.toContain('PENDING use')
  expect(JSON.stringify(first)).not.toContain('REJECTED use')
  const system = first.messages.filter(message => message.role === 'system')
  expect(JSON.stringify(system)).not.toContain('APPROVED use')
  expect(requests[1]?.messages.filter(message => message.role === 'system')).toEqual(system)
  const last = requests[1]?.messages.at(-1)
  expect(String(last?.content)).not.toContain('Approved lessons')
})

test('lesson blocks are bounded to five lessons and 2000 characters with explicit clipping', async () => {
  const { requests } = await scenario(false, true)
  const user = String(requests[0]!.messages.find(message => message.role === 'user')?.content)
  const block = user.slice(user.indexOf('## Approved lessons'), user.indexOf('</turn-context>')).trim()
  expect(block.length).toBeLessThanOrEqual(2000)
  expect(block).toContain('[clipped]')
  expect(block.match(/^- Cue:/gmu)?.length).toBe(5)
})

test('recall failures are logged without failing a turn and empty recall injects nothing', async () => {
  const log = spyOn(console, 'error').mockImplementation(() => {})
  try {
    const { requests } = await scenario(true)
    expect(requests).toHaveLength(2)
    expect(JSON.stringify(requests)).not.toContain('Approved lessons')
    expect(renderApprovedLessonsForTurn([])).toBe('')
    expect(log).toHaveBeenCalledWith('Approved lesson recall failed:', expect.any(Error))
  } finally { log.mockRestore() }
})

test('cancellation during lesson recall prevents the provider request', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lesson-cancel-'))
  const controller = new AbortController()
  let requests = 0
  const runner = new AgentTurnRunner({ model: 'gpt-4o', toolExecutor: new ToolRegistry(),
    lessonsForTurn: async (_session, _text, signal) => { controller.abort(); signal.throwIfAborted(); return '' },
    llm: { async *stream() { requests++; yield { content: 'Must not run' } } },
  })
  const runtime = new InMemoryDaemonRuntime(runner, { model: 'gpt-4o', currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
  try {
    const session = await runtime.openSession('cancel')
    await runtime.submitTurn(session.sessionKey, 'Bun fixtures', () => {}, { signal: controller.signal })
    expect(requests).toBe(0)
    expect(controller.signal.aborted).toBe(true)
  } finally { await runtime.shutdown(); await rm(root, { recursive: true, force: true }) }
})
