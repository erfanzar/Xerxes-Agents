// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerLessonTools } from '../src/tools/lessonTools.js'
import { getActiveSession } from '../src/runtime/sessionContext.js'
import { reviewLearning } from '../src/daemon/xerxesCommand.js'
import { mkdir, mkdtemp, readFile, realpath, rm, symlink, writeFile } from 'node:fs/promises'
import { connect } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer, type DaemonServerOptions } from '../src/daemon/server.js'
import { renderTrajectory } from '../src/context/trajectory.js'
import type { TrajectoryMessage } from '../src/context/trajectory.js'
import type { TurnReflectionInput } from '../src/daemon/turnRunner.js'
import { GovernedMemoryStore } from '../src/memory/governedMemoryStore.js'
import { LessonStore } from '../src/memory/lessons.js'
import { createReflectionApplier } from '../src/runtime/reflectionApplier.js'
import { ReflectionProposalStore, type ReflectionApplier, type ReflectionProposal } from '../src/runtime/reflection.js'
import { SkillBundleStore } from '../src/tools/agentMetaTools.js'

const proposal: ReflectionProposal = { kind: 'skill_create', target: 'fixture-skill', rationale: '[OBSERVATION] Reuse this check', evidence: ['[OBSERVATION] check passed'], skill: { description: 'Reusable check', instructions: 'Check the result.' } }

async function fixture(run: (context: { root: string; server: DaemonServer; runtime: InMemoryDaemonRuntime; notifications: Record<string, unknown>[]; store: ReflectionProposalStore; lessons: LessonStore; calls: string[]; request: (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>> }) => Promise<void>, injected = true, reflectionClientFactory?: DaemonServerOptions['reflectionClientFactory']) {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-command-')))
  let now = Date.now()
  const store = new ReflectionProposalStore({ directory: join(root, 'reflection'), now: () => ++now })
  const lessons = new LessonStore({ memory: new GovernedMemoryStore({ directory: join(root, 'memory'), now: () => ++now }) })
  const calls: string[] = []
  const applier: ReflectionApplier = { applySkill: input => {
    expect(getActiveSession<{ cwd: string }>()?.cwd).toBe(root)
    calls.push(input.name); return { ok: true, applied: input.name }
  }, applyAgent: name => { calls.push(name); return { ok: true } } }
  const runtime = new InMemoryDaemonRuntime(undefined, { model: 'gpt-4o', currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
  const notifications: Record<string, unknown>[] = []
  const server = new DaemonServer({ socketPath: join(root, 'daemon.sock'), exitLogPath: join(root, 'exits.jsonl'), autoTitle: false,
    skillDirectories: [join(root, 'skills')], runtime,
    ...(reflectionClientFactory ? { reflectionClientFactory } : {}),
    ...(injected ? { learning: { store, lessons, applier } } : {}),
  })
  await server.start()
  const socket = connect({ path: join(root, 'daemon.sock') })
  await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) })
  socket.setEncoding('utf8')
  let buffer = ''; let id = 0
  const pending = new Map<number, (value: Record<string, unknown>) => void>()
  socket.on('data', chunk => {
    buffer += String(chunk)
    let newline: number
    while ((newline = buffer.indexOf('\n')) >= 0) {
      const frame = JSON.parse(buffer.slice(0, newline)) as { id?: number; result?: Record<string, unknown>; error?: unknown; params?: { type?: string; payload?: Record<string, unknown> } }
      buffer = buffer.slice(newline + 1)
      if (frame.params?.payload) notifications.push({ event: frame.params.type, ...frame.params.payload })
      if (frame.id !== undefined) { pending.get(frame.id)?.(frame.result ?? { error: frame.error }); pending.delete(frame.id) }
    }
  })
  const request = (method: string, params: Record<string, unknown>) => new Promise<Record<string, unknown>>(resolve => {
    const current = ++id; pending.set(current, resolve)
    socket.write(JSON.stringify({ jsonrpc: '2.0', id: current, method, params }) + '\n')
  })
  try { await request('initialize', { session_key: 'xerxes', project_dir: root }); await run({ root, server, runtime, notifications, store, lessons, calls, request }) }
  finally { socket.destroy(); await server.stop(); await rm(root, { recursive: true, force: true }) }
}

test('xerxes proposals/show/approve/reject are workspace scoped and approval applies exactly once', () => fixture(async ({ root, store, calls, request }) => {
  const first = await store.propose({ workspace: root, proposal })
  const second = await store.propose({ workspace: root, proposal: { ...proposal, target: 'second-skill' } })
  await mkdir(join(root, 'other'))
  const foreign = await store.propose({ workspace: join(root, 'other'), proposal })
  expect(await request('slash', { command: '/xerxes' })).toMatchObject({ ok: true, report: expect.stringContaining('/xerxes approve <id> [reason]') })
  expect(await request('slash', { command: '/xerxes proposals' })).toMatchObject({ ok: true, proposals: [expect.objectContaining({ id: second.id }), expect.objectContaining({ id: first.id })] })
  expect(await request('slash', { command: `/xerxes show ${first.id}` })).toMatchObject({ ok: true, proposal: expect.objectContaining({ proposal }) })
  expect(await request('slash', { command: `/xerxes show ${foreign.id}` })).toMatchObject({ ok: false })
  expect(await request('slash', { command: `/xerxes approve ${first.id} Reviewed` })).toMatchObject({ ok: true, proposal: expect.objectContaining({ status: 'applied', reason: 'Reviewed' }) })
  expect(await request('slash', { command: `/xerxes approve ${first.id}` })).toMatchObject({ ok: false })
  expect(calls).toEqual(['fixture-skill'])
  expect(await request('slash', { command: `/xerxes reject ${second.id}` })).toMatchObject({ ok: false })
  expect(await request('slash', { command: `/xerxes reject ${second.id} Not reusable` })).toMatchObject({ ok: true, proposal: expect.objectContaining({ status: 'rejected', reason: 'Not reusable' }) })
  expect(await request('complete', { text: '/xer' })).toMatchObject({ completions: expect.arrayContaining([expect.objectContaining({ label: 'xerxes' })]) })
  expect(await request('complete', { text: '/xerxes pr' })).toMatchObject({ completions: [expect.objectContaining({ value: '/xerxes proposals ' })] })
}))

test('xerxes lessons shares the store and plugin approval is only an acknowledgement', () => fixture(async ({ root, store, lessons, calls, request }) => {
  await lessons.record({ workspace: root, agentId: 'default', confidence: 0.9, cue: 'Bun fixtures', lesson: 'Use a private fixture', evidence: { kind: 'test', command: 'bun test', exitCode: 0, summary: 'Passed' } })
  expect(await request('slash', { command: '/xerxes lessons' })).toMatchObject({ ok: true, report: expect.stringContaining('Use a private fixture') })
  expect(await request('slash', { command: '/xerxes lessons Bun' })).toMatchObject({ ok: true, lessons: [expect.objectContaining({ cue: 'Bun fixtures' })] })
  const idea = await store.propose({ workspace: root, proposal: { kind: 'plugin_idea', target: 'checker', rationale: proposal.rationale, evidence: proposal.evidence, plugin: { description: 'Build a checker' } } })
  expect(await request('slash', { command: `/xerxes approve ${idea.id}` })).toMatchObject({ ok: true, report: expect.stringContaining('/plugin-creator'), proposal: expect.objectContaining({ status: 'acknowledged' }) })
  expect(calls).toEqual([])
}))

test('xerxes improve always previews the budget without importing its transport', () => fixture(async ({ root, request }) => {
  expect(await request('slash', { command: '/xerxes improve' })).toMatchObject({ ok: true, report: expect.stringContaining('playground:improve') })
  const transport = join(root, 'never-load.ts')
  await writeFile(transport, 'throw new Error("TRANSPORT MUST NEVER LOAD")')
  await writeFile(join(root, 'proposals.json'), JSON.stringify([{ id: 'one', scope: 'INV', target: 'prompt_addendum', description: 'Check results', apply: { path: 'AGENTS.md', content: 'Check.' } }]))
  const result = await request('slash', { command: `/xerxes improve --transport ${transport} --proposals proposals.json --max-candidates 1 --max-task-runs 32` })
  const report = String(result.report)
  expect(result.ok).toBe(true)
  expect(result.dry_run).toBe(true)
  expect(report).toContain('32')
  expect(report).toContain('bun run --cwd xerxes playground:improve -- --transport')
  expect(report).not.toContain('TRANSPORT MUST NEVER LOAD')
}))

test('xerxes review without injected host services is actionable, never a fake success', () => fixture(async ({ request }) => {
  expect(await request('slash', { command: '/xerxes proposals' })).toMatchObject({ ok: false, error: expect.stringContaining('not configured') })
  expect(await request('learning.list', {})).toMatchObject({ ok: false, error: expect.stringContaining('not configured') })
  expect(await request('learning.review', { target: 'proposal', id: 'missing', action: 'approve' })).toMatchObject({ ok: false, report: expect.any(String), error: expect.stringContaining('not configured') })
}, false))

test('learning.list is workspace-scoped, ordered, read-only and counts pending records', () => fixture(async ({ root, store, lessons, request }) => {
  await mkdir(join(root, 'foreign'))
  const foreign = join(root, 'foreign')
  const a = await store.propose({ workspace: root, proposal })
  const b = await store.propose({ workspace: root, proposal: { ...proposal, target: 'second' } })
  await store.reject(a.id, 'Not reusable', root)
  await store.propose({ workspace: foreign, proposal })
  const record = (cue: string, workspace = root) => lessons.record({ workspace, agentId: 'default', confidence: 0.9, cue, lesson: `Lesson ${cue}`, evidence: { kind: 'test', command: 'bun test', exitCode: 0, summary: 'Passed' } })
  const approved = await record('approved')
  await lessons.review(approved.id, 'approved', undefined, root)
  const pending = await record('pending')
  const rejected = await record('rejected'); await lessons.review(rejected.id, 'rejected', 'No', root)
  const expired = await record('expired'); await lessons.rollback(expired.id, 'Old', root)
  const superseded = await record('old'); const correction = await lessons.correct(superseded.id, 'New text', 'Correction', root)
  await record('foreign', foreign)
  const result = await request('learning.list', {})
  expect((result.lessons as { id: string }[]).map(row => row.id)).toEqual([approved.id, correction.id, pending.id])
  expect(result).toMatchObject({ ok: true, workspace: root, proposals: [expect.objectContaining({ id: b.id }), expect.objectContaining({ id: a.id, status: 'rejected' })], lessons: [expect.objectContaining({ id: approved.id }), expect.anything(), expect.anything()], counts: { pending_proposals: 1, pending_lessons: 2 } })
  const other = await request('session.open', { session_key: 'foreign', project_dir: foreign })
  expect(other.ok).toBe(true)
  expect(await request('learning.list', {})).toMatchObject({ ok: true, workspace: foreign, counts: { pending_proposals: 1, pending_lessons: 1 } })
  expect(await request('learning.list', { session_key: 'xerxes' })).toMatchObject({ workspace: root })
}))

test('learning.review shares slash approval, refuses repeat/invalid reviews and records lesson decisions', () => fixture(async ({ root, store, lessons, calls, request, notifications }) => {
  const first = await store.propose({ workspace: root, proposal })
  const second = await store.propose({ workspace: root, proposal })
  expect(await request('learning.review', { target: 'proposal', id: first.id, action: 'approve' })).toMatchObject({ ok: true, report: expect.any(String), proposal: { status: 'applied' } })
  expect(await request('slash', { command: `/xerxes approve ${first.id}` })).toMatchObject({ ok: false, report: expect.stringContaining('not pending') })
  expect(calls).toEqual(['fixture-skill'])
  for (const params of [
    { target: 'proposal', id: second.id, action: 'reject' },
    { target: 'proposal', id: second.id, action: 'rollback', reason: 'Undo' },
    { target: 'bad', id: second.id, action: 'approve' },
    { target: 'lesson', id: 'missing', action: 'bad' },
    { target: 'lesson', id: 'missing', action: 'reject', reason: ' ' },
    { target: 'lesson', id: 'missing', action: 'approve' },
  ]) expect(await request('learning.review', params)).toMatchObject({ ok: false, error: expect.any(String), report: expect.any(String) })
  expect(await request('learning.review', { target: 'proposal', id: second.id, action: 'reject', reason: 'Not general' })).toMatchObject({ ok: true, proposal: { status: 'rejected' } })
  const lesson = await lessons.record({ workspace: root, agentId: 'default', confidence: 0.8, cue: 'fixtures', lesson: 'Use temp dirs', evidence: { kind: 'test', command: 'bun test', exitCode: 0, summary: 'Passed' } })
  expect(await request('complete', { text: '/xerxes lessons ap' })).toMatchObject({ completions: [expect.objectContaining({ value: '/xerxes lessons approve ' })] })
  expect(await request('complete', { text: '/xerxes lessons approve ' })).toMatchObject({ completions: [expect.objectContaining({ label: lesson.id })] })
  expect(await request('slash', { command: `/xerxes lessons approve ${lesson.id}` })).toMatchObject({ ok: true, lesson: { reviewStatus: 'approved' } })
  expect(await request('learning.review', { target: 'lesson', id: lesson.id, action: 'approve' })).toMatchObject({ ok: false })
  expect(await request('learning.review', { target: 'lesson', id: lesson.id, action: 'reject' })).toMatchObject({ ok: false })
  expect(await request('slash', { command: `/xerxes lessons reject ${lesson.id} Not useful` })).toMatchObject({ ok: true, lesson: { reviewStatus: 'rejected' } })
  expect(await request('learning.review', { target: 'lesson', id: lesson.id, action: 'rollback' })).toMatchObject({ ok: false })
  expect(await request('learning.review', { target: 'lesson', id: lesson.id, action: 'rollback', reason: 'Undo' })).toMatchObject({ ok: true, lesson: { id: lesson.id }, report: expect.stringContaining('rolled back') })
  expect(await lessons.get(lesson.id, root)).toBeUndefined()
  expect(notifications).toContainEqual(expect.objectContaining({ event: 'learning_changed', workspace: root, counts: { pending_proposals: 0, pending_lessons: 0 } }))
}))

test('show is readable with complete proposal content and offers explicit JSON', () => fixture(async ({ root, store, request }) => {
  const full = { ...proposal, skill: { ...proposal.skill, files: { 'check.ts': 'check()\n' } } }
  const record = await store.propose({ workspace: root, proposal: full, trigger: 'goal_complete', objective: 'Check fixtures' })
  const shown = await request('slash', { command: `/xerxes show ${record.id}` })
  for (const text of ['skill_create fixture-skill', 'Status: pending', 'Trigger: goal_complete', 'Objective: Check fixtures', 'Rationale:', '> [OBSERVATION] check passed', 'Reusable check', 'Check the result.', 'check.ts (8 bytes)', 'check()']) expect(String(shown.report)).toContain(text)
  expect(String(shown.report).startsWith('{')).toBe(false)
  const json = await request('slash', { command: `/xerxes show ${record.id} --json` })
  expect(JSON.parse(String(json.report))).toMatchObject({ id: record.id, proposal: full })
}))

test('lesson tools invalidate learning for the shared store', () => fixture(async ({ root, lessons, request, notifications }) => {
  const registry = new ToolRegistry()
  registerLessonTools(registry, { store: () => lessons, activeWorkspace: () => root })
  const recorded = await registry.get('lesson_record')!({ cue: 'fixtures', lesson: 'Use temp dirs', confidence: 0.8, test_evidence: { command: 'bun test', exit_code: 0, summary: 'Passed' } }, { metadata: {} }) as { lesson: { id: string } }
  await request('learning.list', {})
  expect(notifications).toContainEqual(expect.objectContaining({ event: 'learning_changed', workspace: root, counts: { pending_proposals: 0, pending_lessons: 1 } }))
  await registry.get('lesson_review')!({ id: recorded.lesson.id, action: 'approve', reason: 'Reviewed' }, { metadata: {} })
  await request('learning.list', {})
  expect(notifications).toContainEqual(expect.objectContaining({ event: 'learning_changed', workspace: root, counts: { pending_proposals: 0, pending_lessons: 0 } }))
}))

test('a failed proposal apply remains observable through the shared review result', () => fixture(async ({ root, store, lessons, request }) => {
  const record = await store.propose({ workspace: root, proposal })
  let calls = 0
  const result = await reviewLearning({ target: 'proposal', id: record.id, action: 'approve' }, root, { store, lessons,
    applier: { applySkill: () => { calls++; return { ok: false, error: 'Lint refused this bundle' } }, applyAgent: () => ({ ok: false }) },
  })
  expect(result).toMatchObject({ ok: false, error: 'Lint refused this bundle', report: expect.stringContaining('failed'), proposal: { status: 'failed' } })
  expect(await request('learning.review', { target: 'proposal', id: record.id, action: 'approve' })).toMatchObject({ ok: false, error: expect.stringContaining('not pending') })
  expect(calls).toBe(1)
}))

test('learning_changed reaches every same-workspace client and excludes other workspaces', () => fixture(async ({ root, store, request, notifications }) => {
  const socket = connect({ path: join(root, 'daemon.sock') })
  const same: Record<string, unknown>[] = []
  const pending = new Map<number, (result: Record<string, unknown>) => void>()
  await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) })
  socket.setEncoding('utf8')
  let buffer = ''; let id = 0
  socket.on('data', chunk => {
    buffer += String(chunk)
    let index: number
    while ((index = buffer.indexOf('\n')) >= 0) {
      const frame = JSON.parse(buffer.slice(0, index))
      buffer = buffer.slice(index + 1)
      if (frame.params?.type === 'learning_changed') same.push(frame.params.payload)
      if (frame.id !== undefined) { pending.get(frame.id)?.(frame.result); pending.delete(frame.id) }
    }
  })
  const send = (method: string, params: Record<string, unknown>) => new Promise<Record<string, unknown>>(resolve => {
    const current = ++id; pending.set(current, resolve)
    socket.write(JSON.stringify({ jsonrpc: '2.0', id: current, method, params }) + '\n')
  })
  try {
    await send('initialize', { session_key: 'second', project_dir: root })
    const record = await store.propose({ workspace: root, proposal })
    await send('learning.list', {})
    expect(same).toHaveLength(1)
    expect(same[0]).toEqual({ workspace: root, counts: { pending_proposals: 1, pending_lessons: 0 } })
    expect(notifications.filter(row => row.event === 'learning_changed')).toHaveLength(1)
    await request('slash', { command: `/xerxes reject ${record.id} Reviewed` })
    await send('learning.list', {})
    expect(same.at(-1)).toEqual({ workspace: root, counts: { pending_proposals: 0, pending_lessons: 0 } })
    await mkdir(join(root, 'foreign'))
    await send('session.open', { session_key: 'foreign', project_dir: join(root, 'foreign') })
    const prior = same.length
    await store.propose({ workspace: root, proposal })
    await send('learning.list', {})
    expect(same).toHaveLength(prior)
  } finally { socket.destroy() }
}))

test('reflection applier preserves skill metadata, retains lint findings and validates contained atomic agent writes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'reflection-applier-'))
  try {
    await mkdir(join(root, 'skills'))
    const skills = new SkillBundleStore({ directory: join(root, 'skills') })
    await skills.manage({ action: 'create', name: 'fixture-skill', description: 'Old', instructions: 'Old instructions', tags: ['kept'], version: '2.3.4' })
    const applier = createReflectionApplier({ skills, activeWorkspace: () => root })
    expect(await applier.applySkill({ action: 'update', name: 'fixture-skill', description: 'New', instructions: 'New instructions' })).toMatchObject({ ok: true })
    expect((await skills.get('fixture-skill'))?.metadata).toMatchObject({ tags: ['kept'], version: '2.3.4' })
    expect(await applier.applySkill({ action: 'create', name: 'xerxes', description: 'Reserved', instructions: 'Review' })).toMatchObject({ ok: false, findings: expect.arrayContaining([expect.objectContaining({ code: 'builtin-command-conflict' })]) })
    const yaml = 'version: "1"\nagent:\n  name: reviewer\n  system_prompt: Review changes\n  tools: [ReadFile]\n'
    expect(await applier.applyAgent('../escape', yaml)).toMatchObject({ ok: false })
    expect(await applier.applyAgent('other', yaml)).toMatchObject({ ok: false })
    expect(await applier.applyAgent('reviewer', 'agent: [invalid]')).toMatchObject({ ok: false })
    expect(await applier.applyAgent('reviewer', yaml)).toMatchObject({ ok: true })
    const target = join(root, '.xerxes/agents/reviewer.yaml')
    expect(await readFile(target, 'utf8')).toBe(yaml)
    expect(await applier.applyAgent('reviewer', 'agent: [invalid]')).toMatchObject({ ok: false })
    expect(await readFile(target, 'utf8')).toBe(yaml)
    await writeFile(join(root, 'outside.yaml'), 'outside')
    await symlink(join(root, 'outside.yaml'), join(root, '.xerxes/agents/link.yaml'))
    expect(await applier.applyAgent('link', yaml.replace('reviewer', 'link'))).toMatchObject({ ok: false })
    await mkdir(join(root, 'other'), { recursive: true })
    await symlink(join(root, 'other'), join(root, '.xerxes/agents/alias'))
    expect(await readFile(join(root, 'outside.yaml'), 'utf8')).toBe('outside')
  } finally { await rm(root, { recursive: true, force: true }) }
})

const trajectory: readonly TrajectoryMessage[] = [
  { role: 'assistant', tool_calls: [{ id: 'proof', function: { name: 'Check', arguments: {} } }] },
  { role: 'tool', tool_call_id: 'proof', content: 'check passed' },
]
const evidence = renderTrajectory(trajectory).split('\n').find(line => line.startsWith('[OBSERVATION]'))!

test('background reflection stores only evidence-validated proposals and notifies the owning session', () => fixture(async ({ server, runtime, store, root, notifications }) => {
  const session = runtime.sessionStatus('xerxes')!
  const input: TurnReflectionInput = { session, messages: trajectory, trigger: 'goal_complete', signal: new AbortController().signal }
  const result = await server.reflectAfterTurn(input)
  expect(result?.summary).toBe('skill_create fixture-skill')
  const records = await store.list(root)
  expect(records).toHaveLength(1)
  expect(records[0]?.status).toBe('pending')
  // The notice is emitted asynchronously; wait for it instead of a fixed 5 ms.
  let notice = notifications.find(item => String(item.message).startsWith('Reflection proposal'))
  for (let attempt = 0; !notice && attempt < 400; attempt++) {
    await new Promise<void>(resolve => setTimeout(resolve, 5))
    notice = notifications.find(item => String(item.message).startsWith('Reflection proposal'))
  }
  expect(notice?.session_id).toBe(session.id)
  expect(notice).toMatchObject({ level: 'info', category: 'reflection', type: 'proposal', reflection: { id: result?.id, kind: 'skill_create', target: 'fixture-skill', rationale: proposal.rationale, evidence_count: 1 } })
  expect(notice?.message).toBe(`Reflection proposal ${result?.id}: skill_create fixture-skill. Nothing was changed. Review it with /xerxes.`)
  expect(notifications).toContainEqual(expect.objectContaining({ event: 'learning_changed', workspace: root, counts: { pending_proposals: 1, pending_lessons: 0 } }))
}, true, () => ({ async *stream(request) {
  expect(request.model).toBe('gpt-4o')
  expect(request.maxTokens).toBe(4096)
  yield { content: JSON.stringify({ ...proposal, evidence: [evidence] }) }
} })))

test('reflection obeys live settings and shutdown cancellation even after a turn has ended', async () => {
  const started = Promise.withResolvers<AbortSignal>()
  await fixture(async ({ server, runtime, store, root, request }) => {
    expect(runtime.status().reflection).toBe('all')
    expect(await request('runtime.reload', { reflection: 'off' })).toMatchObject({ reflection: 'off' })
    const input: TurnReflectionInput = { session: runtime.sessionStatus('xerxes')!, messages: trajectory, trigger: 'goal_complete', signal: new AbortController().signal }
    expect(await server.reflectAfterTurn(input)).toBeUndefined()
    expect((await store.list(root)).length).toBe(0)
    expect(await request('runtime.reload', { reflection: 'all' })).toMatchObject({ reflection: 'all' })
    const pending = server.reflectAfterTurn(input)
    const handled = pending.catch(() => undefined)
    const signal = await started.promise
    await server.stop()
    expect(signal.aborted).toBe(true)
    await handled
    expect((await store.list(root)).length).toBe(0)
  }, true, () => ({ async *stream(_request, signal) {
    if (!signal) throw new Error('Reflection requires lifetime cancellation')
    started.resolve(signal)
    await new Promise<void>(resolve => signal.addEventListener('abort', () => resolve(), { once: true }))
    signal.throwIfAborted()
    yield { content: 'none' }
  } }))
})
