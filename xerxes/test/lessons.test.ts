// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, expect, test } from 'bun:test'
import { appendFile, mkdir, mkdtemp, readFile, realpath, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { GovernedMemoryStore } from '../src/memory/governedMemoryStore.js'
import { LessonError, LessonStore, type LessonErrorCode, type RecordLessonInput, type ResolveLessonToolCall } from '../src/memory/lessons.js'
import { registerLessonTools } from '../src/tools/lessonTools.js'
import type { JsonObject } from '../src/types/toolCalls.js'

let directory: string
let workspace: string
let otherWorkspace: string
let memory: GovernedMemoryStore
let store: LessonStore
const successfulCall: ResolveLessonToolCall = (_sessionId, id) => ({ id: id ?? 'call-latest', name: 'exec_command', ok: true, summary: 'bun test: 8 pass, 0 fail' })

beforeEach(async () => {
  directory = await mkdtemp(join(tmpdir(), 'xerxes-lessons-'))
  workspace = join(directory, 'workspace')
  otherWorkspace = join(directory, 'other')
  await mkdir(workspace)
  await mkdir(otherWorkspace)
  memory = new GovernedMemoryStore({ directory, now: () => 1_000 })
  store = new LessonStore({ memory, resolveToolCall: successfulCall })
})

afterEach(async () => {
  await rm(directory, { recursive: true, force: true })
})

function input(overrides: Partial<RecordLessonInput> = {}): RecordLessonInput {
  return {
    cue: 'Bun test fixtures', lesson: 'Create temporary directories before constructing fixtures.', confidence: 0.8,
    workspace, agentId: 'agent-a', sessionId: 'session-a', evidence: { kind: 'tool_call', toolCallId: 'call-1' },
    ...overrides,
  }
}

async function refusal(operation: Promise<unknown>, code: LessonErrorCode): Promise<void> {
  try {
    await operation
    throw new Error('expected lesson refusal')
  } catch (error) {
    expect(error).toBeInstanceOf(LessonError)
    expect((error as LessonError).code).toBe(code)
    expect((error as Error).message.length).toBeGreaterThan(20)
  }
}

test('lessons refuse missing evidence without persisting a record', async () => {
  const { evidence: _evidence, ...withoutEvidence } = input()
  await refusal(store.record(withoutEvidence), 'LESSON_EVIDENCE_REQUIRED')
  expect((await memory.load()).records.size).toBe(0)
})

test('tool evidence fails closed without a resolver or session', async () => {
  await refusal(new LessonStore({ memory }).record(input()), 'LESSON_EVIDENCE_UNAVAILABLE')
  const { sessionId: _sessionId, ...withoutSession } = input()
  await refusal(store.record(withoutSession), 'LESSON_EVIDENCE_UNAVAILABLE')
})

test('tool evidence refuses failed, missing, mismatched, and empty host results', async () => {
  const ports: ResolveLessonToolCall[] = [
    () => undefined,
    () => ({ id: 'call-1', name: 'exec_command', ok: false, summary: 'tests failed' }),
    () => ({ id: 'wrong-call', name: 'exec_command', ok: true, summary: 'tests passed' }),
    () => ({ id: 'call-1', name: 'exec_command', ok: true, summary: ' ' }),
  ]
  for (const resolveToolCall of ports) {
    await refusal(new LessonStore({ memory, resolveToolCall }).record(input()), 'LESSON_INVALID_EVIDENCE')
  }
  expect((await memory.load()).records.size).toBe(0)
})

test('tool evidence is populated from the session-scoped host, including latest-call resolution', async () => {
  const calls: unknown[] = []
  store = new LessonStore({ memory, resolveToolCall: async (sessionId, id) => {
    calls.push([sessionId, id])
    return { id: 'call-latest', name: 'exec_command', ok: true, summary: 'Verified eight passing tests' }
  } })
  const record = await store.record(input({ evidence: { kind: 'tool_call' } }))
  expect(calls).toEqual([['session-a', undefined]])
  expect(record.evidence).toEqual({ kind: 'tool_call', toolCallId: 'call-latest', toolName: 'exec_command', summary: 'Verified eight passing tests' })
  expect(record.workspace).toBe(await realpath(workspace))
  expect(record.reviewStatus).toBe('pending')
})

test('invalid cue, text, confidence, and empty evidence are actionable refusals', async () => {
  for (const overrides of [{ cue: ' ' }, { lesson: '' }, { confidence: -0.1 }, { confidence: 1.1 }, { confidence: Number.NaN }]) {
    await refusal(store.record(input(overrides)), 'LESSON_INVALID_INPUT')
  }
  await refusal(store.record(input({ evidence: { kind: 'tool_call', toolCallId: '' } })), 'LESSON_INVALID_EVIDENCE')
  await refusal(store.record(input({ evidence: { kind: 'test', command: '', exitCode: 0, summary: 'pass' } })), 'LESSON_INVALID_EVIDENCE')
  await refusal(store.record(input({ evidence: { kind: 'test', command: 'bun test', exitCode: 0, summary: ' ' } })), 'LESSON_INVALID_EVIDENCE')
  await refusal(store.record(input({ evidence: { kind: 'test', command: 'bun test', exitCode: Number.NaN, summary: 'pass' } })), 'LESSON_INVALID_EVIDENCE')
})

test('explicit test results persist evidence, including a failed test that taught a lesson', async () => {
  const evidence = { kind: 'test' as const, command: 'bun test fixture.test.ts', exitCode: 1, summary: 'The missing fixture directory causes ENOENT.' }
  const record = await new LessonStore({ memory }).record(input({ evidence }))
  expect(record.evidence).toEqual(evidence)
  expect((await new LessonStore({ memory }).recall('fixtures', workspace))[0]?.evidence).toEqual(evidence)
})

test('recall isolates canonical workspaces, including symlink aliases', async () => {
  const local = await store.record(input())
  await store.record(input({ workspace: otherWorkspace, lesson: 'Do not retrieve this other workspace lesson.' }))
  const alias = join(directory, 'workspace-alias')
  await symlink(workspace, alias)
  expect((await store.recall('BUN fixtures', alias)).map(lesson => lesson.id)).toEqual([local.id])
  expect((await store.recall('BUN fixtures', otherWorkspace))).toHaveLength(1)
})

test('recall ranks approved first, then cue overlap before lesson text, with deterministic ties and limits', async () => {
  const textMatch = await store.record(input({ cue: 'directories', lesson: 'Bun test fixtures should be temporary.' }))
  const partialCue = await store.record(input({ cue: 'Bun directories' }))
  const exactCue = await store.record(input({ cue: 'Bun test fixtures', confidence: 0.5 }))
  await store.review(textMatch.id, 'approved')
  const results = await store.recall('BUN test fixtures', workspace)
  expect(results.map(lesson => lesson.id)).toEqual([textMatch.id, exactCue.id, partialCue.id])
  expect((await store.recall('BUN test fixtures', workspace, { limit: 2 })).map(lesson => lesson.id)).toEqual([textMatch.id, exactCue.id])
  expect(await store.recall('no matching words', workspace)).toEqual([])
  const tied = await store.record(input({ cue: 'Bun test fixtures', confidence: 0.5 }))
  const expectedIds = [exactCue.id, tied.id].sort()
  expect((await store.recall('fixtures', workspace)).filter(lesson => expectedIds.includes(lesson.id)).map(lesson => lesson.id)).toEqual(expectedIds)
})

test('review, rejection, correction, and rollback are governed events and survive reload', async () => {
  const original = await store.record(input())
  const rejected = await store.record(input({ cue: 'Bun test rejected' }))
  await store.review(original.id, 'approved')
  await store.review(rejected.id, 'rejected')
  const corrected = await store.correct(original.id, 'Create and clean up temporary fixture directories.', 'Include cleanup verified by the same test.')
  expect(corrected.id).not.toBe(original.id)
  expect(corrected.correctionOf).toBe(original.id)
  expect(corrected.reviewStatus).toBe('pending')
  await store.review(corrected.id, 'approved')
  store = new LessonStore({ memory: new GovernedMemoryStore({ directory }) })
  expect((await store.recall('Bun test', workspace)).map(lesson => lesson.id)).toEqual([corrected.id])
  expect((await store.get(rejected.id, workspace))?.reviewStatus).toBe('rejected')
  await store.rollback(corrected.id, 'Correction generalized too far; restore the original.')
  store = new LessonStore({ memory: new GovernedMemoryStore({ directory }) })
  expect((await store.recall('Bun test', workspace)).map(lesson => lesson.id)).toEqual([original.id])
  const log = await readFile(memory.logPath, 'utf8')
  expect(log).toContain('"type":"reviewed"')
  expect(log).toContain('"correctionOf"')
  expect(log).toContain('"type":"expired"')
  expect(log).toContain('Correction generalized too far; restore the original.')
})

test('review and rollback refuse missing IDs and blank reasons without corrupting the log', async () => {
  const record = await store.record(input())
  await refusal(store.review('missing', 'approved'), 'LESSON_NOT_FOUND')
  await refusal(store.rollback(record.id, ' '), 'LESSON_INVALID_INPUT')
  await refusal(store.correct(record.id, ' ', 'fix'), 'LESSON_INVALID_INPUT')
  await refusal(store.correct(record.id, 'Updated text', ''), 'LESSON_INVALID_INPUT')
  expect((await store.recall('fixtures', workspace))).toHaveLength(1)
})

test('legacy governed records without lesson payloads still load and do not appear as lessons', async () => {
  await memory.record({ id: 'legacy', content: 'An older governed memory.', source: 'conversation', agentId: 'old-agent' })
  const lesson = await store.record(input())
  const reloadedMemory = new GovernedMemoryStore({ directory })
  expect((await reloadedMemory.load()).records.get('legacy')?.content).toBe('An older governed memory.')
  expect((await new LessonStore({ memory: reloadedMemory }).recall('fixtures', workspace)).map(item => item.id)).toEqual([lesson.id])
  const correction = await reloadedMemory.correct({ originalId: 'legacy', newId: 'legacy-correction', content: 'Corrected legacy memory.', reason: 'clarify' })
  expect(correction.payload).toBeUndefined()
})

test('lesson replay tolerates a torn tail and later appends repair it', async () => {
  const first = await store.record(input())
  await appendFile(memory.logPath, '{"type":"recorded","record":')
  expect((await store.recall('fixtures', workspace)).map(item => item.id)).toEqual([first.id])
  const second = await store.record(input({ cue: 'fixtures cleanup' }))
  const reloaded = new LessonStore({ memory: new GovernedMemoryStore({ directory }) })
  expect((await reloaded.recall('fixtures', workspace)).map(item => item.id).sort()).toEqual([first.id, second.id].sort())
})

test('render is a Markdown projection grouped by review status with evidence and without rejected lessons', async () => {
  const approved = await store.record(input({ cue: 'Approved fixtures' }))
  await store.review(approved.id, 'approved')
  await store.record(input({ cue: 'Pending fixtures', evidence: { kind: 'test', command: 'bun test', exitCode: 0, summary: '8 pass' } }))
  const rejected = await store.record(input({ cue: 'Rejected fixtures' }))
  await store.review(rejected.id, 'rejected')
  await store.record(input({ workspace: otherWorkspace, cue: 'Hidden foreign lesson' }))
  const markdown = await store.render(workspace)
  expect(markdown).toContain('## Approved')
  expect(markdown).toContain('## Pending')
  expect(markdown).toContain('Evidence: tool call')
  expect(markdown).toContain('exec\\_command')
  expect(markdown).toContain('call-1')
  expect(markdown).toContain('Evidence: test result (reported)')
  expect(markdown).toContain('bun test')
  expect(markdown).not.toContain('Rejected fixtures')
  expect(markdown).not.toContain('Hidden foreign lesson')
})

function tools(resolveToolCall?: ResolveLessonToolCall): ToolRegistry {
  const registry = new ToolRegistry()
  // The tool-level port must work even when the workspace store has no port.
  registerLessonTools(registry, { store: () => new LessonStore({ memory }), activeWorkspace: () => workspace, ...(resolveToolCall ? { resolveToolCall } : {}) })
  return registry
}

async function invoke(registry: ToolRegistry, name: string, args: JsonObject): Promise<JsonObject> {
  const result = await registry.get(name)!(args, { sessionId: 'session-a', agentId: 'model-a', metadata: {} })
  return result as JsonObject
}

test('model tools resolve latest verified evidence, recall by active workspace, and approve or roll back', async () => {
  const registry = tools(successfulCall)
  expect(registry.definitions().map(definition => definition.function.name)).toEqual(['lesson_record', 'lesson_recall', 'lesson_review'])
  const result = await invoke(registry, 'lesson_record', { cue: 'Bun fixtures', lesson: 'Use a temporary directory.', confidence: 0.9 })
  expect(result.ok).toBe(true)
  const saved = result.lesson as JsonObject
  expect(saved.agentId).toBe('model-a')
  expect((saved.evidence as JsonObject).toolCallId).toBe('call-latest')
  expect((await invoke(registry, 'lesson_recall', { cue: 'fixtures' })).ok).toBe(true)
  expect((await invoke(registry, 'lesson_review', { id: saved.id!, action: 'approve', reason: 'Checked the cited test output.' })).ok).toBe(true)
  expect((await store.get(String(saved.id), workspace))?.reviewStatus).toBe('approved')
  expect((await invoke(registry, 'lesson_review', { id: saved.id!, action: 'rollback', reason: 'No longer applies.' })).ok).toBe(true)
  expect(await store.recall('fixtures', workspace)).toEqual([])
})

test('automatic turns cannot approve lessons for future automatic injection', async () => {
  const lesson = await store.record(input())
  const registry = tools()
  expect(await registry.get('lesson_review')!({ id: lesson.id, action: 'approve', reason: 'Self approval' }, { metadata: { goal_turn_human: false } })).toMatchObject({ ok: false, code: 'LESSON_HUMAN_APPROVAL_REQUIRED' })
  expect((await store.get(lesson.id, workspace))?.reviewStatus).toBe('pending')
})

test('model tools return typed JSON refusals, including absent evidence ports and cross-workspace review', async () => {
  const registry = tools()
  const noPort = await invoke(registry, 'lesson_record', { cue: 'Bun fixtures', lesson: 'Create a directory.', confidence: 0.9 })
  expect(noPort).toMatchObject({ ok: false, code: 'LESSON_EVIDENCE_UNAVAILABLE' })
  expect(typeof noPort.error).toBe('string')
  const foreign = await store.record(input({ workspace: otherWorkspace }))
  expect(await invoke(registry, 'lesson_review', { id: foreign.id, action: 'reject', reason: 'wrong workspace' })).toMatchObject({ ok: false, code: 'LESSON_NOT_FOUND' })
  expect((await store.get(foreign.id, otherWorkspace))?.reviewStatus).toBe('pending')
  expect(await invoke(registry, 'lesson_recall', { cue: 'fixtures', limit: 0 })).toMatchObject({ ok: false, code: 'LESSON_INVALID_INPUT' })
  expect(await invoke(registry, 'lesson_review', { id: foreign.id, action: 'delete', reason: 'invalid' })).toMatchObject({ ok: false, code: 'LESSON_INVALID_INPUT' })
})

test('model tools accept explicit reported test evidence and refuse ambiguous evidence sources', async () => {
  const registry = tools()
  const evidence = { command: 'bun test', exit_code: 0, summary: '8 pass, 0 fail' }
  const result = await invoke(registry, 'lesson_record', { cue: 'Bun fixtures', lesson: 'Create a directory.', confidence: 0.9, test_evidence: evidence })
  expect(result.ok).toBe(true)
  expect(((result.lesson as JsonObject).evidence as JsonObject).kind).toBe('test')
  expect(await invoke(registry, 'lesson_record', { cue: 'Bun fixtures', lesson: 'Create a directory.', confidence: 0.9, evidence_tool_call_id: 'call-1', test_evidence: evidence })).toMatchObject({ ok: false, code: 'LESSON_INVALID_EVIDENCE' })
})

test('rejected corrections restore their predecessor and preserve its verified evidence', async () => {
  const original = await store.record(input())
  const corrected = await store.correct(original.id, 'Use temporary fixtures and clean them up.', 'Include cleanup.')
  expect(corrected.evidence).toEqual(original.evidence)
  await store.review(corrected.id, 'rejected', 'The added cleanup claim is not established.')
  const reloaded = new LessonStore({ memory: new GovernedMemoryStore({ directory }) })
  expect((await reloaded.recall('fixtures', workspace)).map(lesson => lesson.id)).toEqual([original.id])
  expect(await readFile(memory.logPath, 'utf8')).toContain('The added cleanup claim is not established.')
})

test('rolling back an intermediate correction does not resurrect a superseded ancestor', async () => {
  const original = await store.record(input())
  const middle = await store.correct(original.id, 'Use isolated temporary fixtures.', 'Clarify isolation.')
  const latest = await store.correct(middle.id, 'Use and clean up isolated temporary fixtures.', 'Include cleanup.')
  await store.rollback(middle.id, 'Withdraw this intermediate wording, not the later correction.')
  const reloaded = new LessonStore({ memory: new GovernedMemoryStore({ directory }) })
  expect((await reloaded.recall('fixtures', workspace)).map(lesson => lesson.id)).toEqual([latest.id])
})

test('tools preserve unexpected host failures rather than fabricating a refusal or success', async () => {
  const failure = new Error('session execution database is unavailable')
  const registry = tools(async () => { throw failure })
  await expect(invoke(registry, 'lesson_record', { cue: 'fixtures', lesson: 'Use a temp directory.', confidence: 0.8 })).rejects.toThrow(failure.message)
  expect((await memory.load()).records.size).toBe(0)
})

test('registered schemas execute through the registry and failed tool results become typed refusals', async () => {
  const registry = tools((_sessionId, id) => ({ id: id ?? 'failed-call', name: 'exec_command', ok: false, summary: 'Tests failed' }))
  const raw = await registry.execute({ id: 'record-call', type: 'function', function: {
    name: 'lesson_record', arguments: { cue: 'fixtures', lesson: 'Use temp directories.', confidence: 0.8, evidence_tool_call_id: 'failed-call' },
  } }, { sessionId: 'session-a', metadata: {} })
  expect(JSON.parse(raw)).toMatchObject({ ok: false, code: 'LESSON_INVALID_EVIDENCE' })
  const missing = tools(() => undefined)
  expect(await invoke(missing, 'lesson_record', { cue: 'fixtures', lesson: 'Use temp directories.', confidence: 0.8 })).toMatchObject({ ok: false, code: 'LESSON_INVALID_EVIDENCE' })
})
