// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { generateKeyPairSync, randomUUID } from 'node:crypto'
import { mkdir, mkdtemp, realpath, rm, writeFile } from 'node:fs/promises'
import { connect } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { CONCURRENT_DISPATCH_METHODS, DaemonServer } from '../src/daemon/server.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { sharedLessonsRpc } from '../src/daemon/lessonSync.js'
import { GovernedMemoryStore } from '../src/memory/governedMemoryStore.js'
import { LessonSharing } from '../src/memory/lessonSharing.js'
import { LessonStore } from '../src/memory/lessons.js'
import { createRepoIdentityResolver } from '../src/memory/repoIdentity.js'
import type { SharedLessonWire } from '../src/memory/sharedLessonWire.js'
import { ReflectionProposalStore } from '../src/runtime/reflection.js'
import { paramsHash, signHubAuth, type HubAuthClaims } from '../src/security/hubAuthEnvelope.js'
import { enrollHub, hubMethodPolicy } from '../src/security/hubIdentityVerifier.js'
import { AGENT_SCOPE_CEILING, HUMAN_ONLY_SCOPES, ROLE_SCOPES } from '../src/security/principal.js'

const repo = `git:${'a'.repeat(32)}`
function wire(): SharedLessonWire {
  const date = new Date().toISOString()
  return { id: `lesson-${randomUUID()}`, repo, status: 'active', cue: 'fixtures', lesson: 'Use temporary fixtures.', confidence: 0.9,
    evidence: 'bun test: 1 pass', supersedes: [], origin: { host: 'another-host', agent_id: 'agent' }, recorded_at: date, updated_at: date }
}
async function fixture(run: (r: {
  root: string; allowed: string; foreign: string; sharing: LessonSharing; lessons: LessonStore;
  request: (method: string, params?: Record<string, unknown>, scopes?: readonly string[]) => Promise<Record<string, unknown>>;
}) => Promise<void>, realGit = false): Promise<void> {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-daemon-lesson-')))
  const allowed = join(root, 'allowed'); const foreign = join(root, 'foreign')
  await mkdir(allowed); await mkdir(foreign)
  if (realGit) {
    for (const args of [['init', '--quiet'], ['remote', 'add', 'origin', 'git@github.com:fixture/lessons.git']]) {
      const p = Bun.spawn(['git', '-C', allowed, ...args], { stdout: 'ignore', stderr: 'pipe' })
      expect(await p.exited).toBe(0)
    }
    await writeFile(join(allowed, '.gitignore'), '.xerxes/\n')
  }
  const memory = new GovernedMemoryStore({ directory: join(root, 'memory') })
  const lessons = new LessonStore({ memory, resolveRepo: realGit ? createRepoIdentityResolver() : async () => repo })
  const sharing = new LessonSharing({ memory, lessons })
  const pair = generateKeyPairSync('ed25519'); const enrollment = join(root, 'enrollments.json')
  enrollHub(enrollment, { hub_id: 'hub-1', public_key_pem: pair.publicKey.export({ format: 'pem', type: 'spki' }).toString(), project_dirs: [allowed] })
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: foreign, model: 'fixture', sessionDirectory: join(root, 'sessions') })
  const server = new DaemonServer({ runtime, projectDirectory: foreign, hubEnrollmentPath: enrollment, socketPath: join(root, 'daemon.sock'),
    exitLogPath: join(root, 'exit.jsonl'), eventJournal: { directory: join(root, 'events') }, autoTitle: false, autoSnapshotTurns: false,
    learning: { lessons, sharing, store: new ReflectionProposalStore({ directory: join(root, 'reflection') }), applier: { applySkill: () => ({ ok: false }), applyAgent: () => ({ ok: false }) } } })
  await server.start()
  const socket = connect(join(root, 'daemon.sock')); socket.setEncoding('utf8')
  await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) })
  let buffer = ''; let nextId = 0
  const pending = new Map<number, (value: Record<string, unknown>) => void>()
  socket.on('data', data => {
    buffer += String(data)
    let newline: number
    while ((newline = buffer.indexOf('\n')) >= 0) {
      const frame = JSON.parse(buffer.slice(0, newline)) as { id?: number; result?: Record<string, unknown>; error?: unknown }
      buffer = buffer.slice(newline + 1)
      if (frame.id !== undefined) { pending.get(frame.id)?.(frame.result ?? { rpc_error: frame.error }); pending.delete(frame.id) }
    }
  })
  const request = (method: string, params: Record<string, unknown> = {}, scopes?: readonly string[]) => {
    if (scopes) {
      const claims: HubAuthClaims = { hub_id: 'hub-1', principal_kind: 'agent', principal_id: 'hub-lesson-sync', display_name: 'Hub lesson sync', workspace_id: 'ws-1',
        method, params_hash: paramsHash(params), request_id: randomUUID(), nonce: randomUUID(), issued_at: Date.now(), expires_at: Date.now() + 60_000, scopes }
      params = { ...params, hub_auth: signHubAuth(claims, pair.privateKey) }
    }
    return new Promise<Record<string, unknown>>((resolve, reject) => {
      const id = ++nextId; const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out: ${method}`)) }, 3_000)
      pending.set(id, value => { clearTimeout(timer); resolve(value) })
      socket.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
    })
  }
  try { await run({ root, allowed, foreign, sharing, lessons, request }) }
  finally { socket.destroy(); await server.stop(); sharing.close(); await rm(root, { recursive: true, force: true }) }
}

test('lesson RPC policies are explicit, concurrent and grant agents sync but no human approval power', () => {
  expect(hubMethodPolicy('lessons.shared.list', {})).toEqual({ kind: 'scope', scope: 'session.read' })
  expect(hubMethodPolicy('lessons.shared.apply', {})).toEqual({ kind: 'scope', scope: 'lessons.sync' })
  expect(AGENT_SCOPE_CEILING.has('lessons.sync')).toBe(true)
  expect(ROLE_SCOPES.owner.has('lessons.sync')).toBe(true)
  expect(HUMAN_ONLY_SCOPES.has('lessons.sync')).toBe(false)
  expect(AGENT_SCOPE_CEILING.has('approvals.manage')).toBe(false)
  for (const method of ['lessons.shared.list', 'lessons.shared.apply']) expect(CONCURRENT_DISPATCH_METHODS.has(method)).toBe(true)
})

test('RPCs work without a session, auto-approve imports and advertise both capability surfaces', () => fixture(async ({ allowed, lessons, request }) => {
  const incoming = wire()
  expect(await request('lessons.shared.list', { project_dir: allowed })).toEqual({ ok: true, repo, lessons: [], next_after_id: null })
  expect(await request('lessons.shared.apply', { project_dir: allowed, repo, lessons: [incoming] })).toEqual({ ok: true, applied: 1, retired: 0, skipped: [] })
  expect((await lessons.get(incoming.id, allowed))?.reviewStatus).toBe('approved')
  expect(await request('lessons.shared.list', { project_dir: allowed })).toEqual({ ok: true, repo, lessons: [incoming], next_after_id: null })
  expect(await request('runtime.status')).toMatchObject({ lessons_sync_v1_supported: true })
  expect(await request('initialize', { project_dir: allowed, session_key: 'fixture' })).toMatchObject({ lessons_sync_v1_supported: true })
}))

test('signed sync uses project_dir enrollment instead of the attached foreign session and checks scopes', () => fixture(async ({ allowed, foreign, request }) => {
  await request('initialize', { project_dir: foreign, session_key: 'foreign' })
  const incoming = wire()
  expect(await request('lessons.shared.list', { project_dir: allowed }, ['session.read'])).toMatchObject({ ok: true, repo })
  expect(await request('lessons.shared.apply', { project_dir: allowed, repo, lessons: [incoming] }, ['lessons.sync'])).toMatchObject({ ok: true, applied: 1 })
  expect(await request('lessons.shared.list', { project_dir: foreign }, ['session.read'])).toMatchObject({ rpc_error: { code: -32001, message: expect.stringContaining('hub_not_enrolled_for_project') } })
  expect(await request('lessons.shared.apply', { project_dir: allowed, repo, lessons: [] }, ['session.read'])).toMatchObject({ rpc_error: { code: -32001, message: expect.stringContaining('lessons.sync') } })
  expect(await request('lessons.shared.list', { project_dir: allowed }, ['lessons.sync'])).toMatchObject({ rpc_error: { code: -32001, message: expect.stringContaining('session.read') } })
}))

test('RPC validation refuses unknown fields, invalid records, >200 rows, >1MiB and repository mismatch', () => fixture(async ({ allowed, request }) => {
  const incoming = wire()
  for (const params of [ {}, { project_dir: 'relative' }, { project_dir: allowed, extra: true } ]) expect(await request('lessons.shared.list', params)).toMatchObject({ ok: false, code: 'invalid_request' })
  for (const params of [ { project_dir: allowed, repo, lessons: [{}] }, { project_dir: allowed, repo, lessons: Array(201).fill(incoming) }, { project_dir: allowed, repo, lessons: Array(200).fill({ ...incoming, lesson: 'x'.repeat(8000) }) } ]) {
    expect(await request('lessons.shared.apply', params)).toMatchObject({ ok: false, code: 'invalid_request' })
  }
  expect(await request('lessons.shared.apply', { project_dir: allowed, repo: `git:${'b'.repeat(32)}`, lessons: [] })).toMatchObject({ ok: false, code: 'repo_mismatch' })
  const previous = process.env.XERXES_LESSON_SYNC
  try { process.env.XERXES_LESSON_SYNC = 'off'; expect(await request('lessons.shared.list', { project_dir: allowed })).toMatchObject({ ok: false, code: 'disabled' }) }
  finally { if (previous === undefined) delete process.env.XERXES_LESSON_SYNC; else process.env.XERXES_LESSON_SYNC = previous }
}))

test('share/sync commands and completions work, and share warns when Git ignores the folder', () => fixture(async ({ allowed, lessons, request }) => {
  await request('initialize', { project_dir: allowed, session_key: 'commands' })
  const lesson = await lessons.record({ workspace: allowed, cue: 'fixtures', lesson: 'Use temp dirs.', confidence: 0.8, agentId: 'fixture', evidence: { kind: 'test', command: 'bun test', exitCode: 0, summary: 'Passed' } })
  await lessons.review(lesson.id, 'approved')
  expect(await request('slash', { command: '/xerxes lessons share' })).toMatchObject({ ok: true, exported: 1, ignored: true, report: expect.stringContaining('Git ignores') })
  expect(await request('slash', { command: '/xerxes lessons sync' })).toMatchObject({ ok: true, applied: 0, retired: 0, skipped: [] })
  expect(await request('complete', { text: '/xerxes lessons s' })).toMatchObject({ completions: [expect.objectContaining({ label: 'share' }), expect.objectContaining({ label: 'sync' })] })
}, true))

test('list replies page every id in ascending order under both the 1000-row and 1 MiB ceilings', () => fixture(async ({ allowed, sharing, request }) => {
  const original = sharing.snapshot.bind(sharing)
  try {
    for (const rows of [Array.from({ length: 1001 }, () => wire()), Array.from({ length: 200 }, () => ({ ...wire(), lesson: 'x'.repeat(8000) }))]) {
      sharing.snapshot = async () => ({ repo, lessons: rows })
      const ids: string[] = []
      let cursor: string | undefined
      do {
        const page = await request('lessons.shared.list', { project_dir: allowed, ...(cursor ? { after_id: cursor } : {}) }, ['session.read'])
        expect(page.ok).toBe(true)
        const lessons = page.lessons as SharedLessonWire[]
        expect(lessons.length).toBeLessThanOrEqual(1000)
        expect(Buffer.byteLength(JSON.stringify(page))).toBeLessThanOrEqual(1024 * 1024)
        ids.push(...lessons.map(row => row.id))
        expect(page.next_after_id === null || page.next_after_id === lessons.at(-1)?.id).toBe(true)
        cursor = typeof page.next_after_id === 'string' ? page.next_after_id : undefined
      } while (cursor)
      expect(ids).toEqual(rows.map(row => row.id).sort())
      expect(new Set(ids).size).toBe(rows.length)
    }
    expect(await request('lessons.shared.list', { project_dir: allowed, after_id: 'not-a-lesson-id' })).toMatchObject({ ok: false, code: 'invalid_request' })
    sharing.snapshot = async () => ({ repo, lessons: [] })
    expect(await request('lessons.shared.list', { project_dir: allowed, after_id: wire().id })).toMatchObject({ lessons: [], next_after_id: null })
  } finally { sharing.snapshot = original }
}))

test('apply checks repository identity without reading or exporting a full snapshot', () => fixture(async ({ allowed, sharing, request }) => {
  sharing.snapshot = async () => { throw new Error('apply must not call snapshot') }
  expect(await request('lessons.shared.apply', { project_dir: allowed, repo, lessons: [wire()] }, ['lessons.sync'])).toMatchObject({ ok: true, applied: 1 })
}))

test('unconfigured hosts refuse rather than claiming sync succeeded', async () => {
  expect(await sharedLessonsRpc('lessons.shared.list', { project_dir: '/tmp' })).toMatchObject({ ok: false, code: 'disabled' })
})
