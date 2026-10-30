// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, spyOn, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { GovernedMemoryStore } from '../src/memory/governedMemoryStore.js'
import { LessonStore, lessonIdsInContext, renderApprovedLessonsForTurn } from '../src/memory/lessons.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest } from '../src/llms/client.js'

async function scenario(fail = false, long = false, texts: readonly string[] = ['Bun fixtures', 'Unrelated gardening']) {
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
      return renderApprovedLessonsForTurn(await lessons.recall(text, session.cwd, { limit: 30 }), { inContext: lessonIdsInContext(session.messages) })
    },
    llm: { async *stream(request) { requests.push(structuredClone(request)); yield { content: 'Done' } } },
  })
  const runtime = new InMemoryDaemonRuntime(runner, { model: 'gpt-4o', currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
  try {
    const session = await runtime.openSession('lessons')
    for (const text of texts) await runtime.submitTurn(session.sessionKey, text, () => {})
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

test('a turn carries at most five lessons, each with its full text and never clipped', async () => {
  const { requests } = await scenario(false, true)
  const user = String(requests[0]!.messages.find(message => message.role === 'user')?.content)
  const block = user.slice(user.indexOf('## Approved lessons'), user.indexOf('</turn-context>')).trim()
  expect(block).not.toContain('[clipped]')
  expect(block.match(/^- lesson-/gmu)?.length).toBe(5)
  expect(block).toContain(`APPROVED use temp directories ${'x'.repeat(5000)}`)
})

test('a lesson already in the context window is not injected again, and the earlier message is replayed byte-identically', async () => {
  const { requests } = await scenario(false, false, ['Bun fixtures', 'Bun fixtures again'])
  const first = requests[0]!.messages.find(message => message.role === 'user')!
  expect(String(first.content)).toContain('APPROVED use temp directories')
  // Prefix caching: the second request starts with the first user message exactly as sent.
  expect(requests[1]!.messages.find(message => message.role === 'user')).toEqual(first)
  const all = JSON.stringify(requests[1]!.messages)
  expect(all.split('APPROVED use temp directories').length - 1).toBe(1)
  expect(String(requests[1]!.messages.at(-1)?.content)).not.toContain('Approved lessons')
})

test('a duplicate-text twin of a lesson already in context is not injected either', () => {
  const base = { cue: 'c', confidence: 0.9, workspace: '/w', agentId: 'a', createdAt: 1, reviewStatus: 'approved' as const,
    evidence: { kind: 'test' as const, command: 'bun test', exitCode: 0, summary: 'ok' } }
  const shown = { ...base, id: 'lesson-00000000-0000-4000-8000-000000000001', lesson: 'Use temp dirs.' }
  const twin = { ...base, id: 'lesson-00000000-0000-4000-8000-000000000002', lesson: 'use  temp dirs.' }
  const other = { ...base, id: 'lesson-00000000-0000-4000-8000-000000000003', lesson: 'Clean up after.' }
  const inContext = lessonIdsInContext([{ content: `earlier turn context mentions ${shown.id}` }])
  const rendered = renderApprovedLessonsForTurn([shown, twin, other], { inContext })
  expect(rendered).not.toContain(shown.id)
  expect(rendered).not.toContain(twin.id)
  expect(rendered).toContain(other.id)
  expect(renderApprovedLessonsForTurn([shown, twin], { inContext })).toBe('')
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
