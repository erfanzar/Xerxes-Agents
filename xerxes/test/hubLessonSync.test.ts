// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { join } from 'node:path'

import { HubLessonStore, HubLessonSync, HUB_LESSONS_PER_REPO, type HubLessonBackend } from '../src/hub/lessonSync.js'
import { ensureHubDirectories, hubPaths } from '../src/hub/paths.js'
import { HubRuntimePool, type PoolCallResult, type PoolLessonWorkspace, type PoolWorkspace } from '../src/hub/runtimePool.js'
import { createHubService } from '../src/hub/service.js'
import { WorkspaceRegistry } from '../src/hub/workspaceRegistry.js'
import { mergeSharedLesson, parseSharedLessonWire, SHARED_LESSON_LIMITS, type SharedLessonWire } from '../src/memory/sharedLessonWire.js'
import { verifyHubAuth } from '../src/security/hubAuthEnvelope.js'
import { disposeAll, registerCleanup, tempDir, testSigner, until } from './hubHarness.js'

afterEach(disposeAll)
const REPO = `git:${'1'.repeat(32)}`
const OTHER_REPO = `git:${'2'.repeat(32)}`
const WHEN = '2026-01-01T00:00:00.000Z'
function lesson(index = 1, overrides: Partial<SharedLessonWire> = {}): SharedLessonWire {
  return {
    id: `lesson-00000000-0000-4000-8000-${index.toString(16).padStart(12, '0')}`,
    repo: REPO, status: 'active', cue: 'Run the focused check', lesson: 'Verify the changed path before reporting success.', confidence: 0.9,
    evidence: 'bun test: 1 pass', supersedes: [], origin: { host: 'machine-a', agent_id: 'agent-a' }, recorded_at: WHEN, updated_at: WHEN,
    ...overrides,
  }
}
function project(id: string, runtimeId = id, projectDir = `/${id}`): PoolLessonWorkspace { return { id, runtimeId, projectDir } }
interface Call { readonly workspaceId: string; readonly method: string; readonly params: Readonly<Record<string, unknown>> }
function fakeBackend(workspaces = [project('a'), project('b')]) {
  const calls: Call[] = []
  const copies = new Map<string, { repo: string | null; lessons: unknown[] }>(workspaces.map(row => [row.id, { repo: REPO, lessons: [] }]))
  let handler: ((call: Call, signal?: AbortSignal) => Promise<PoolCallResult | undefined> | PoolCallResult | undefined) | undefined
  const backend: HubLessonBackend = {
    lessonWorkspaces: () => workspaces,
    async callLessons(workspaceId, method, params, options = {}) {
      const call = { workspaceId, method, params }; calls.push(call)
      expect(options.timeoutMs).toBe(30_000)
      expect(options.signal).toBeInstanceOf(AbortSignal)
      const overridden = await handler?.(call, options.signal)
      if (overridden !== undefined) return overridden
      const copy = copies.get(workspaceId)!
      if (method === 'lessons.shared.list') return { state: 'accepted', result: { ok: true, repo: copy.repo, lessons: [...copy.lessons] } }
      let applied = 0, retired = 0
      for (const row of params.lessons as SharedLessonWire[]) {
        const index = copy.lessons.findIndex(value => typeof value === 'object' && value !== null && 'id' in value && value.id === row.id)
        const existing = index < 0 ? undefined : parseSharedLessonWire(copy.lessons[index])
        const current = existing?.ok === true ? existing.lesson : undefined
        const next = mergeSharedLesson(current, row)
        if (next === current) continue
        if (index < 0) copy.lessons.push(next); else copy.lessons[index] = next
        if (next.status === 'retired') retired++; else applied++
      }
      return { state: 'accepted', result: { ok: true, applied, retired, skipped: [] } }
    },
  }
  return { backend, calls, copies, respond(value: typeof handler) { handler = value } }
}
async function fixture(workspaces?: PoolLessonWorkspace[], environment: Readonly<Record<string, string | undefined>> = {}) {
  const path = join(await tempDir('xh-lessons-'), 'hub.sqlite')
  const fake = fakeBackend(workspaces)
  const signer = testSigner()
  const errors: { message: string; context: string }[] = []
  const sync = new HubLessonSync({ databasePath: path, signer, backend: fake.backend, environment,
    onError: (error, context) => { errors.push({ message: String(error), context }) } })
  registerCleanup(() => sync.close())
  return { ...fake, path, signer, sync, errors }
}

function pagedReply(rows: readonly SharedLessonWire[], after: unknown, pageSize = 1000, repo = REPO): PoolCallResult {
  const remaining = [...rows].sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0)
    .filter(row => typeof after !== 'string' || row.id > after)
  const page: SharedLessonWire[] = []
  let bytes = 256
  for (const row of remaining) {
    const size = Buffer.byteLength(JSON.stringify(row)) + 1
    if (page.length >= pageSize || bytes + size > SHARED_LESSON_LIMITS.payloadBytes) break
    page.push(row); bytes += size
  }
  return { state: 'accepted', result: { ok: true, repo, lessons: page, next_after_id: page.length < remaining.length ? page.at(-1)!.id : null } }
}

// Real Unix transport, fake daemon. No user runtime, provider, credentials or SSH.
type Frame = { id: number; method: string; params: Record<string, unknown> }
async function fakeDaemon(options: { runtimeId?: string; supported?: boolean; signed?: boolean; verified?: boolean } = {}) {
  const dir = await tempDir('xh-lesson-daemon-')
  const socketPath = join(dir, 'd.sock')
  const requests: Frame[] = []
  let lessons: unknown[] = [], repo: string | null = REPO, silent = false
  const sockets = new Set<Bun.Socket<{ buffer: string }>>()
  const server = Bun.listen<{ buffer: string }>({ unix: socketPath, socket: {
    open(socket) { sockets.add(socket); socket.data = { buffer: '' } },
    close(socket) { sockets.delete(socket) },
    data(socket, bytes) {
      socket.data.buffer += bytes.toString()
      for (;;) {
        const newline = socket.data.buffer.indexOf('\n'); if (newline < 0) return
        const frame = JSON.parse(socket.data.buffer.slice(0, newline)) as Frame
        socket.data.buffer = socket.data.buffer.slice(newline + 1); requests.push(frame)
        let result: Record<string, unknown> = { ok: true }
        if (frame.method === 'initialize') result = { ok: true, daemon_protocol: 35 }
        if (frame.method === 'runtime.status') result = { ok: true, daemon_protocol: 35,
          ...(options.verified === false ? {} : { runtime_id: options.runtimeId ?? 'runtime' }),
          principal_auth_v1_supported: options.signed !== false, lessons_sync_v1_supported: options.supported !== false }
        if (frame.method === 'session.active_list') result = { ok: true, sessions: [] }
        if (frame.method === 'lessons.shared.list') {
          if (silent) continue
          result = { ok: true, repo, lessons }
        }
        if (frame.method === 'lessons.shared.apply') {
          let applied = 0, retired = 0
          for (const incoming of frame.params.lessons as SharedLessonWire[]) {
            const index = lessons.findIndex(row => typeof row === 'object' && row !== null && 'id' in row && row.id === incoming.id)
            const existing = index < 0 ? undefined : parseSharedLessonWire(lessons[index])
            const current = existing?.ok === true ? existing.lesson : undefined
            const merged = mergeSharedLesson(current, incoming)
            if (merged === current) continue
            if (index < 0) lessons.push(merged); else lessons[index] = merged
            if (merged.status === 'retired') retired++; else applied++
          }
          result = { ok: true, applied, retired, skipped: [] }
        }
        socket.write(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }) + '\n')
      }
    },
  } })
  registerCleanup(() => { for (const socket of sockets) socket.terminate(); server.stop(true) })
  return { socketPath, requests, setLessons(rows: unknown[]) { lessons = rows }, get lessons() { return lessons }, setRepo(value: string | null) { repo = value },
    silence() { silent = true }, drop() { for (const socket of sockets) socket.terminate() } }
}
function workspace(id: string, socketPath: string, projectDir = `/${id}`): PoolWorkspace {
  return { id, label: id, projectDir, autoConnect: true, endpoint: { kind: 'local', socketPath } }
}
async function poolFixture(workspaces: PoolWorkspace[]) {
  const pool = new HubRuntimePool({ sink: { ingest() {}, upstreamCursor() { return undefined } },
    clientOptions: { retryBaseMs: 1000, retryMaxMs: 1000, requestTimeoutMs: 1000 }, pollIntervalMs: 60_000 })
  registerCleanup(() => pool.stop())
  await pool.start(workspaces)
  return pool
}

test('real pool syncs A to B, then a B retirement reaches A in the same pass; signatures bind project_dir and scopes', async () => {
  const a = await fakeDaemon({ runtimeId: 'machine-a' }), b = await fakeDaemon({ runtimeId: 'machine-b' })
  const active = lesson(); a.setLessons([active])
  const pool = await poolFixture([workspace('a', a.socketPath, '/clone/a'), workspace('b', b.socketPath, '/clone/b'), workspace('alias', a.socketPath, '/clone/a')])
  const signer = testSigner(), path = join(await tempDir(), 'hub.sqlite')
  const errors: unknown[] = []
  const sync = new HubLessonSync({ databasePath: path, signer, backend: pool, environment: {}, onError: error => { errors.push(error) } })
  registerCleanup(() => sync.close())
  await sync.syncNow()
  expect(b.lessons).toEqual([active])
  expect(a.requests.filter(row => row.method === 'lessons.shared.list')).toHaveLength(1)
  const retired = lesson(1, { status: 'retired', updated_at: '2026-01-02T00:00:00.000Z', retired_reason: 'Superseded by new evidence' })
  b.setLessons([retired])
  await sync.syncNow()
  expect(a.lessons).toEqual([retired])
  expect(sync.status()).toMatchObject({ enabled: true, running: false, last_run: expect.any(String), repos: [{ repo: REPO, lessons: 1, active: 0, retired: 1 }] })
  for (const [daemon, id, projectDir] of [[a, 'a', '/clone/a'], [b, 'b', '/clone/b']] as const) {
    for (const request of daemon.requests.filter(row => row.method.startsWith('lessons.shared.'))) {
      const { hub_auth, ...params } = request.params
      expect(params.project_dir).toBe(projectDir)
      const checked = verifyHubAuth(hub_auth, { publicKey: signer.publicKeyPem, method: request.method, params })
      expect(checked.ok).toBe(true)
      if (checked.ok) expect(checked.claims).toMatchObject({ principal_kind: 'agent', principal_id: 'hub-lesson-sync', display_name: 'Hub lesson sync',
        hub_id: signer.hubId, workspace_id: id, scopes: request.method === 'lessons.shared.list' ? ['session.read'] : ['lessons.sync'] })
    }
  }
  const count = [...a.requests, ...b.requests].filter(row => row.method === 'lessons.shared.apply').length
  await sync.syncNow()
  expect([...a.requests, ...b.requests].filter(row => row.method === 'lessons.shared.apply')).toHaveLength(count)
  expect(errors).toEqual([])
})

test('invalid and wrong-repo rows are dropped, and duplicate ids merge monotonically', async () => {
  const t = await fixture()
  const active = lesson(), retired = lesson(1, { status: 'retired', updated_at: '2026-01-02T00:00:00.000Z' })
  t.copies.get('a')!.lessons = [null, { ...lesson(2), unknown: true }, lesson(3, { repo: OTHER_REPO }), active, retired, active]
  await t.sync.syncNow()
  expect(t.copies.get('b')!.lessons).toEqual([retired])
  expect(t.sync.status().repos).toEqual([{ repo: REPO, lessons: 1, active: 0, retired: 1 }])
  expect(t.errors).toMatchObject([{ message: expect.stringContaining('Dropped 3'), context: 'lesson sync workspace a' }])
})

test('pool skips unsupported, read-only, unverified, discovery-entry and offline lanes before any sync RPC', async () => {
  const old = await fakeDaemon({ supported: false, runtimeId: 'old' })
  const readOnly = await fakeDaemon({ signed: false, runtimeId: 'readonly' })
  const unverified = await fakeDaemon({ verified: false })
  const online = await fakeDaemon({ runtimeId: 'online' })
  const pool = await poolFixture([workspace('old', old.socketPath), workspace('readonly', readOnly.socketPath), workspace('unverified', unverified.socketPath),
    workspace('online', online.socketPath), { ...workspace('machine', online.socketPath, '/'), discover: true }, { ...workspace('offline', online.socketPath), autoConnect: false }])
  expect(pool.lessonWorkspaces()).toEqual([project('online')])
  const t = await fixture()
  const sync = new HubLessonSync({ databasePath: join(await tempDir(), 'hub.sqlite'), signer: t.signer, backend: pool, environment: {} })
  registerCleanup(() => sync.close())
  await sync.syncNow()
  for (const daemon of [old, readOnly, unverified]) expect(daemon.requests.some(row => row.method.startsWith('lessons.'))).toBe(false)
  for (const [id, code] of [['old', 'unsupported'], ['readonly', 'read_only'], ['unverified', 'unavailable'], ['offline', 'unavailable'], ['machine', 'unavailable']] as const) {
    expect(await pool.callLessons(id, 'lessons.shared.list', {})).toMatchObject({ state: 'rejected', error: { code } })
  }
  online.drop()
  await until(() => pool.health('online')?.state !== 'online')
  expect(pool.lessonWorkspaces()).toEqual([])
})

test('apply batches are capped at 200 and every signature matches the sent bytes', async () => {
  const t = await fixture([project('a'), project('b')])
  t.copies.get('a')!.lessons = Array.from({ length: 450 }, (_, i) => lesson(i + 1))
  await t.sync.syncNow()
  const applies = t.calls.filter(row => row.method === 'lessons.shared.apply')
  expect(applies.map(row => (row.params.lessons as unknown[]).length)).toEqual([200, 200, 50])
  expect(t.copies.get('b')!.lessons).toHaveLength(450)
  for (const call of applies) {
    const { hub_auth, ...params } = call.params
    expect(verifyHubAuth(hub_auth, { publicKey: t.signer.publicKeyPem, method: call.method, params }).ok).toBe(true)
    expect(Buffer.byteLength(JSON.stringify(call.params))).toBeLessThanOrEqual(SHARED_LESSON_LIMITS.payloadBytes)
  }
  expect(t.errors).toEqual([])
})

test('UTF-8 byte limits split large lessons before the 1 MiB ceiling, even below 200 items', async () => {
  const t = await fixture([project('a')])
  const store = new HubLessonStore(t.path)
  const large = Array.from({ length: 200 }, (_, i) => lesson(i + 1, { lesson: '🧪'.repeat(4000) }))
  store.merge(REPO, large); store.close()
  await t.sync.syncNow()
  const applies = t.calls.filter(row => row.method === 'lessons.shared.apply')
  expect(applies.length).toBeGreaterThan(1)
  expect(t.copies.get('a')!.lessons).toHaveLength(200)
  for (const call of applies) expect(Buffer.byteLength(JSON.stringify(call.params))).toBeLessThanOrEqual(SHARED_LESSON_LIMITS.payloadBytes)
  expect(t.errors).toEqual([])
})

test('store persists active lessons and absorbing retirements across reopen', async () => {
  const path = join(await tempDir(), 'hub.sqlite'), active = lesson()
  const first = new HubLessonStore(path)
  first.merge(REPO, [active]); first.close()
  const second = new HubLessonStore(path)
  expect(second.lessons(REPO)).toEqual([active])
  const retired = lesson(1, { status: 'retired', updated_at: '2026-01-02T00:00:00.000Z' })
  second.merge(REPO, [retired]); second.close()
  const third = new HubLessonStore(path)
  expect(third.merge(REPO, [active])).toEqual({ changed: 0, capped: 0, conflicts: [] })
  expect(third.lessons(REPO)).toEqual([retired]); third.close()
})

test('off disables explicit, periodic and online-triggered sync; interval is clamped or defaults safely', async () => {
  const t = await fixture(undefined, { XERXES_HUB_LESSON_SYNC: 'off', XERXES_HUB_LESSON_SYNC_MS: '1' })
  t.copies.get('a')!.lessons = [lesson()]
  t.sync.start(); t.sync.workspaceOnline(); await t.sync.syncNow()
  expect(t.calls).toEqual([])
  expect(t.sync.status()).toEqual({ enabled: false, running: false, interval_ms: 10_000, last_run: null, repos: [],
    conflicts: 0, suppressed_conflicts: 0, conflict_limit_reached: false, incomplete_listings: 0 })
  for (const value of ['NaN', '0', '-1', 'Infinity', '0.5', '2147483648', '9007199254740991']) {
    const other = await fixture([], { XERXES_HUB_LESSON_SYNC_MS: value })
    expect(other.sync.status().interval_ms).toBe(60_000)
  }
})

test('per-repo cap never evicts a tombstone and still permits retirement updates at capacity', async () => {
  const path = join(await tempDir(), 'hub.sqlite'), store = new HubLessonStore(path)
  for (let offset = 0; offset < HUB_LESSONS_PER_REPO; offset += 1000) {
    expect(store.merge(REPO, Array.from({ length: 1000 }, (_, i) => lesson(offset + i + 1))).changed).toBe(1000)
  }
  expect(store.merge(REPO, [lesson(6000)])).toEqual({ changed: 0, capped: 1, conflicts: [] })
  const retired = lesson(1, { status: 'retired', updated_at: '2026-01-02T00:00:00.000Z' })
  expect(store.merge(REPO, [retired])).toEqual({ changed: 1, capped: 0, conflicts: [] })
  expect(store.merge(REPO, [lesson(6000), lesson()])).toEqual({ changed: 0, capped: 1, conflicts: [] })
  expect(store.merge(OTHER_REPO, [lesson(6000, { repo: OTHER_REPO })])).toEqual({ changed: 1, capped: 0, conflicts: [] })
  expect(store.counts()).toEqual([{ repo: REPO, lessons: 5000, active: 4999, retired: 1 }, { repo: OTHER_REPO, lessons: 1, active: 1, retired: 0 }])
  store.close()
})

test('store boundary validates rows, rejects cross-repo writes and rolls a failed merge back atomically', async () => {
  const store = new HubLessonStore(join(await tempDir(), 'hub.sqlite'))
  expect(() => store.merge(REPO, [lesson(), lesson(2, { repo: OTHER_REPO })])).toThrow('Invalid shared lesson')
  expect(store.counts()).toEqual([])
  expect(() => store.merge('not-a-repo', [lesson()])).toThrow('Invalid shared lesson repo')
  expect(() => store.merge(REPO, Array.from({ length: 1001 }, () => lesson()))).toThrow('Too many')
  store.close()
})

test('failed list and failed apply are reported per workspace without stopping healthy peers', async () => {
  const t = await fixture([project('bad-list'), project('source'), project('bad-apply'), project('target')])
  t.copies.get('source')!.lessons = [lesson()]
  t.respond(call => {
    if (call.workspaceId === 'bad-list') throw new Error('offline')
    if (call.workspaceId === 'bad-apply' && call.method === 'lessons.shared.apply') return { state: 'rejected', error: { code: 'disabled', message: 'disabled locally' } }
    return undefined
  })
  await t.sync.syncNow()
  expect(t.copies.get('target')!.lessons).toEqual([lesson()])
  expect(t.errors.map(row => row.context).sort()).toEqual(['lesson sync workspace bad-apply', 'lesson sync workspace bad-list'])
})

test('malformed, oversized, and over-count list replies fail closed; a null repo is unavailable for sharing', async () => {
  for (const reply of [
    { ok: true, repo: 'not-a-repo', lessons: [lesson()] },
    { ok: true, repo: REPO, lessons: Array.from({ length: 1001 }, () => lesson()) },
    { ok: true, repo: REPO, lessons: [lesson()], extra: 'x'.repeat(SHARED_LESSON_LIMITS.payloadBytes) },
    { ok: false, repo: REPO, lessons: [lesson()] },
  ]) {
    const t = await fixture([project('a')]); t.respond(() => ({ state: 'accepted', result: reply }))
    await t.sync.syncNow()
    expect(t.sync.status().repos).toEqual([])
    expect(t.calls.filter(row => row.method === 'lessons.shared.apply')).toEqual([])
    expect(t.errors).toHaveLength(1)
  }
  const t = await fixture([project('a')]); t.copies.get('a')!.repo = null
  await t.sync.syncNow(); expect(t.errors).toEqual([]); expect(t.sync.status().repos).toEqual([])
})

test('malformed apply acknowledgements are reported and do not prevent a healthy workspace from receiving lessons', async () => {
  for (const result of [
    { ok: true, applied: 2, retired: 0, skipped: [] },
    { ok: true, applied: -1, retired: 0, skipped: [] },
    { ok: true, applied: 0, retired: 0, skipped: [{ id: lesson(99).id, reason: 'not in the batch' }] },
    { ok: true, applied: 0, retired: 0, skipped: [{ id: lesson().id, reason: 'x'.repeat(501) }] },
    { ok: false, applied: 0, retired: 0, skipped: [] },
  ]) {
    const t = await fixture([project('source'), project('bad'), project('good')])
    t.copies.get('source')!.lessons = [lesson()]
    t.respond(call => call.workspaceId === 'bad' && call.method === 'lessons.shared.apply' ? { state: 'accepted', result } : undefined)
    await t.sync.syncNow()
    expect(t.copies.get('good')!.lessons).toEqual([lesson()])
    expect(t.errors).toMatchObject([{ message: expect.stringContaining('Invalid shared lesson apply reply'), context: 'lesson sync workspace bad' }])
  }
})

test('concurrent syncNow calls share a pass; online arrivals coalesce into one serialized follow-up', async () => {
  const t = await fixture([project('a')])
  let release!: () => void, waiting = true, active = 0, maxActive = 0
  const gate = new Promise<void>(resolve => { release = resolve })
  t.respond(async () => {
    active++; maxActive = Math.max(maxActive, active)
    if (waiting) await gate
    active--; return undefined
  })
  const first = t.sync.syncNow(), second = t.sync.syncNow()
  expect(first).toBe(second)
  await until(() => t.calls.length > 0)
  t.sync.workspaceOnline(); t.sync.workspaceOnline()
  expect(t.calls).toHaveLength(1)
  waiting = false; release(); await first
  await until(() => t.calls.length === 2 && !t.sync.status().running)
  expect(maxActive).toBe(1)
})

test('close aborts pending calls, stops future passes and never writes an apply after cancellation', async () => {
  const t = await fixture([project('a')])
  t.respond((_call, signal) => new Promise((_resolve, reject) => {
    signal!.addEventListener('abort', () => reject(new Error('cancelled')), { once: true })
  }))
  const running = t.sync.syncNow()
  await until(() => t.calls.length > 0)
  await t.sync.close(); await running
  await t.sync.syncNow(); t.sync.workspaceOnline(); t.sync.start()
  expect(t.calls).toHaveLength(1)
  expect(t.errors).toEqual([])
})

test('lesson call admission rejects generic RPCs, unsigned or mis-scoped requests, excessive batches/bytes, timeout and pre-cancel', async () => {
  const daemon = await fakeDaemon(), pool = await poolFixture([workspace('a', daemon.socketPath)])
  const params = { project_dir: '/a', hub_auth: {} }
  expect(await pool.callLessons('a', 'shutdown', params)).toMatchObject({ state: 'rejected', error: { code: 'forbidden' } })
  for (const bad of [{}, { ...params, project_dir: '/other' }, { project_dir: '/a' }]) {
    expect(await pool.callLessons('a', 'lessons.shared.list', bad)).toMatchObject({ state: 'rejected', error: { code: 'invalid_request' } })
  }
  expect(await pool.callLessons('a', 'lessons.shared.apply', { ...params, repo: REPO, lessons: Array.from({ length: 201 }, () => lesson()) })).toMatchObject({ state: 'rejected', error: { code: 'invalid_request' } })
  expect(await pool.callLessons('a', 'lessons.shared.list', { ...params, extra: 'x'.repeat(SHARED_LESSON_LIMITS.payloadBytes) })).toMatchObject({ state: 'rejected', error: { code: 'invalid_request' } })
  expect(daemon.requests.some(row => row.method.startsWith('lessons.'))).toBe(false)
  expect(await pool.callLessons('a', 'lessons.shared.list', params, { signal: AbortSignal.abort() })).toMatchObject({ state: 'rejected' })
  expect(daemon.requests.some(row => row.method.startsWith('lessons.'))).toBe(false)
  daemon.silence()
  expect(await pool.callLessons('a', 'lessons.shared.list', params, { timeoutMs: 15 })).toMatchObject({ state: 'rejected', error: { code: 'unavailable' } })
  expect(daemon.requests.filter(row => row.method === 'lessons.shared.list')).toHaveLength(1)
})

test('store refuses changes to every immutable field, including retired-copy rewrites, but permits lifecycle updates', async () => {
  const store = new HubLessonStore(join(await tempDir(), 'hub.sqlite'))
  const original = lesson()
  store.merge(REPO, [original])
  for (const changed of [
    { cue: 'Different cue' }, { lesson: 'Different lesson' }, { confidence: 0.2 }, { evidence: 'Different evidence' },
    { supersedes: [lesson(2).id] }, { origin: { host: 'other', agent_id: 'agent-a' } },
    { origin: { host: 'machine-a', agent_id: 'other' } }, { recorded_at: '2026-01-02T00:00:00.000Z' },
  ]) {
    expect(store.merge(REPO, [lesson(1, { ...changed, status: 'retired', updated_at: '2026-01-03T00:00:00.000Z' })])).toMatchObject({ changed: 0, conflicts: [original.id] })
    expect(store.lessons(REPO)).toEqual([original])
  }
  const retired = lesson(1, { status: 'retired', updated_at: '2026-01-03T00:00:00.000Z', retired_reason: 'new evidence' })
  expect(store.merge(REPO, [retired])).toMatchObject({ changed: 1, conflicts: [] })
  const earlier = { ...retired, updated_at: '2026-01-02T00:00:00.000Z', retired_reason: 'earlier retirement' }
  expect(store.merge(REPO, [earlier])).toMatchObject({ changed: 1, conflicts: [] })
  expect(store.lessons(REPO)).toEqual([earlier]); store.close()
})

test('conflicting retired list rows cannot rewrite Hub content or be pushed; conflicts are counted once per runtime and id', async () => {
  const t = await fixture([project('bad'), project('good')])
  const original = lesson(), store = new HubLessonStore(t.path)
  store.merge(REPO, [original]); store.close()
  t.copies.get('bad')!.lessons = [lesson(1, { lesson: 'Changed content', status: 'retired', updated_at: '2026-01-02T00:00:00.000Z' })]
  await t.sync.syncNow(); await t.sync.syncNow()
  const reopened = new HubLessonStore(t.path)
  expect(reopened.lessons(REPO)).toEqual([original]); reopened.close()
  expect(t.copies.get('good')!.lessons).toEqual([original])
  expect(t.calls.some(call => call.workspaceId === 'bad' && call.method === 'lessons.shared.apply')).toBe(false)
  expect(t.sync.status()).toMatchObject({ conflicts: 1, suppressed_conflicts: 1 })
  expect(t.errors.some(error => /conflict/i.test(error.message))).toBe(true)
})

test('conflicting duplicate rows within one daemon listing do not replace the first immutable body', async () => {
  const t = await fixture()
  t.copies.get('a')!.lessons = [lesson(), lesson(1, { cue: 'Rewritten cue', status: 'retired' })]
  await t.sync.syncNow()
  expect(t.copies.get('b')!.lessons).toEqual([lesson()])
  expect(t.calls.some(call => call.workspaceId === 'a' && call.method === 'lessons.shared.apply')).toBe(false)
  expect(t.sync.status()).toMatchObject({ conflicts: 1 })
})

test('pulls every page beyond 1000 lessons, signs after_id, and does not re-push already-present later ids', async () => {
  const t = await fixture([project('source'), project('target')])
  const rows = Array.from({ length: 1500 }, (_, i) => lesson(i + 1))
  t.copies.get('source')!.lessons = [...rows]
  t.respond(call => {
    if (call.method !== 'lessons.shared.list') return undefined
    return pagedReply(t.copies.get(call.workspaceId)!.lessons as SharedLessonWire[], call.params.after_id)
  })
  await t.sync.syncNow()
  expect(t.copies.get('target')!.lessons).toHaveLength(1500)
  const applies = t.calls.filter(call => call.method === 'lessons.shared.apply').length
  await t.sync.syncNow()
  expect(t.calls.filter(call => call.method === 'lessons.shared.apply')).toHaveLength(applies)
  expect(t.calls.some(call => call.workspaceId === 'source' && call.method === 'lessons.shared.apply')).toBe(false)
  const pages = t.calls.filter(call => call.method === 'lessons.shared.list' && call.params.after_id !== undefined)
  expect(pages.length).toBeGreaterThan(0)
  for (const call of pages) {
    const { hub_auth, ...params } = call.params
    const checked = verifyHubAuth(hub_auth, { publicKey: t.signer.publicKeyPem, method: call.method, params })
    expect(checked.ok).toBe(true)
    expect(verifyHubAuth(hub_auth, { publicKey: t.signer.publicKeyPem, method: call.method, params: { ...params, after_id: lesson(9999).id } }).ok).toBe(false)
  }
  expect(t.sync.status().repos).toEqual([{ repo: REPO, lessons: 1500, active: 1500, retired: 0 }])
  expect(t.errors).toEqual([])
})

test('byte-limited pages are fully read without re-sending the unseen tail every pass', async () => {
  const t = await fixture([project('a')])
  const rows = Array.from({ length: 180 }, (_, i) => lesson(i + 1, { lesson: '🧪'.repeat(4000) }))
  t.copies.get('a')!.lessons = rows
  t.respond(call => call.method === 'lessons.shared.list' ? pagedReply(rows, call.params.after_id) : undefined)
  await t.sync.syncNow(); await t.sync.syncNow()
  expect(t.calls.filter(call => call.method === 'lessons.shared.list').length).toBeGreaterThan(2)
  expect(t.calls.filter(call => call.method === 'lessons.shared.apply')).toHaveLength(0)
  expect(t.sync.status().repos[0]?.lessons).toBe(180)
  expect(t.errors).toEqual([])
})

test('page bound reports incomplete listings and only pushes ids inside the observed prefix', async () => {
  const t = await fixture([project('a')])
  const rows = Array.from({ length: 11 }, (_, i) => lesson((i + 1) * 2))
  t.copies.get('a')!.lessons = rows
  const retired = lesson(2, { status: 'retired', updated_at: '2026-01-02T00:00:00.000Z' })
  const store = new HubLessonStore(t.path)
  store.merge(REPO, [lesson(1), retired, lesson(21), lesson(22), lesson(23)]); store.close()
  t.respond(call => call.method === 'lessons.shared.list' ? pagedReply(rows, call.params.after_id, 1) : undefined)
  await t.sync.syncNow()
  expect(t.calls.filter(call => call.method === 'lessons.shared.list')).toHaveLength(10)
  const pushed = t.calls.filter(call => call.method === 'lessons.shared.apply').flatMap(call => call.params.lessons as SharedLessonWire[])
  expect(pushed.map(row => row.id)).toEqual([lesson(1).id, lesson(2).id])
  expect(t.sync.status()).toMatchObject({ incomplete_listings: 1 })
  expect(t.errors.some(error => /page.*bound/i.test(error.message))).toBe(true)
})

test('bad pagination cursors, ordering and changing repos fail closed before pushing', async () => {
  for (const variant of ['malformed', 'repeat', 'wrong-last', 'unordered', 'changed-repo']) {
    const t = await fixture([project('a')])
    t.respond(call => {
      if (call.method !== 'lessons.shared.list') return undefined
      if (variant === 'malformed') return { state: 'accepted', result: { ok: true, repo: REPO, lessons: [lesson()], next_after_id: 'bad' } }
      if (variant === 'wrong-last') return { state: 'accepted', result: { ok: true, repo: REPO, lessons: [lesson()], next_after_id: lesson(9).id } }
      if (variant === 'unordered') return { state: 'accepted', result: { ok: true, repo: REPO, lessons: [lesson(2), lesson()], next_after_id: lesson().id } }
      if (call.params.after_id === undefined) return { state: 'accepted', result: { ok: true, repo: REPO, lessons: [lesson()], next_after_id: lesson().id } }
      return variant === 'repeat'
        ? { state: 'accepted', result: { ok: true, repo: REPO, lessons: [lesson()], next_after_id: lesson().id } }
        : { state: 'accepted', result: { ok: true, repo: OTHER_REPO, lessons: [lesson(2, { repo: OTHER_REPO })], next_after_id: null } }
    })
    await t.sync.syncNow()
    expect(t.calls.filter(call => call.method === 'lessons.shared.apply')).toHaveLength(0)
    expect(t.errors).toHaveLength(1)
  }
})

test('daemon push conflict replies suppress the same repo/runtime/id across passes and workspace aliases for this Hub run', async () => {
  const workspaces = [project('a', 'runtime')]
  const t = await fixture(workspaces), store = new HubLessonStore(t.path)
  store.merge(REPO, [lesson()]); store.merge(OTHER_REPO, [lesson(1, { repo: OTHER_REPO })]); store.close()
  t.respond(call => call.method === 'lessons.shared.apply'
    ? { state: 'accepted', result: { ok: true, applied: 0, retired: 0, skipped: [{ id: lesson().id, reason: 'conflict' }] } } : undefined)
  await t.sync.syncNow()
  workspaces.push(project('alias', 'runtime', '/other-clone')); t.copies.set('alias', { repo: REPO, lessons: [] })
  await t.sync.syncNow(); await t.sync.syncNow()
  expect(t.calls.filter(call => call.method === 'lessons.shared.apply')).toHaveLength(1)
  expect(t.sync.status()).toMatchObject({ conflicts: 1, suppressed_conflicts: 1 })
  workspaces.push(project('other-runtime', 'runtime-2')); t.copies.set('other-runtime', { repo: REPO, lessons: [] })
  await t.sync.syncNow()
  expect(t.calls.filter(call => call.method === 'lessons.shared.apply')).toHaveLength(2)
  t.copies.get('alias')!.repo = OTHER_REPO
  await t.sync.syncNow()
  expect(t.calls.filter(call => call.method === 'lessons.shared.apply')).toHaveLength(3)
  expect(t.sync.status()).toMatchObject({ conflicts: 3, suppressed_conflicts: 3 })
  await t.sync.close()
  const reopened = new HubLessonSync({ databasePath: t.path, signer: t.signer, backend: t.backend, environment: {}, onError() {} })
  registerCleanup(() => reopened.close())
  await reopened.syncNow()
  expect(t.calls.filter(call => call.method === 'lessons.shared.apply').length).toBeGreaterThan(3)
})

test('non-conflict skipped replies remain retryable and are not suppressed', async () => {
  const t = await fixture([project('a')]), store = new HubLessonStore(t.path)
  store.merge(REPO, [lesson()]); store.close()
  t.respond(call => call.method === 'lessons.shared.apply'
    ? { state: 'accepted', result: { ok: true, applied: 0, retired: 0, skipped: [{ id: lesson().id, reason: 'temporarily unavailable' }] } } : undefined)
  await t.sync.syncNow(); await t.sync.syncNow()
  expect(t.calls.filter(call => call.method === 'lessons.shared.apply')).toHaveLength(2)
  expect(t.sync.status()).toMatchObject({ conflicts: 0, suppressed_conflicts: 0 })
})

test('the tenth page can complete the listing and permit pushes beyond its last returned id', async () => {
  const t = await fixture([project('a')]), store = new HubLessonStore(t.path)
  const rows = Array.from({ length: 10 }, (_, i) => lesson(i + 1))
  store.merge(REPO, [lesson(11)]); store.close()
  t.respond(call => call.method === 'lessons.shared.list' ? pagedReply(rows, call.params.after_id, 1) : undefined)
  await t.sync.syncNow()
  expect(t.calls.filter(call => call.method === 'lessons.shared.list')).toHaveLength(10)
  expect(t.copies.get('a')!.lessons).toEqual([lesson(11)])
  expect(t.sync.status().incomplete_listings).toBe(0)
  expect(t.errors).toEqual([])
})

test('service starts sync, triggers it on workspace online, exposes status and closes it without daemon shutdown', async () => {
  const home = await tempDir('xh-lesson-service-'), paths = hubPaths({ XERXES_HOME: home })
  await ensureHubDirectories(paths)
  const daemon = await fakeDaemon(); daemon.setLessons([lesson()])
  const registry = new WorkspaceRegistry(paths.workspacesPath, { generateId: () => 'a' })
  await registry.add({ label: 'A', projectDir: '/a', endpoint: { kind: 'local', socketPath: daemon.socketPath }, connection: { autoConnect: true }, consent: { bootstrap: false, update: false } }, 0)
  const errors: unknown[] = []
  const service = createHubService({ paths, registry, signer: testSigner(), exposure: { port: 0 }, onError: error => { errors.push(error) } })
  registerCleanup(() => service.stop())
  await service.start(); await service.workspacesSettled
  await until(() => service.lessonSync?.status().repos.length === 1)
  expect(service.lessonSync?.status().repos).toEqual([{ repo: REPO, lessons: 1, active: 1, retired: 0 }])
  const sync = service.lessonSync!
  await service.stop(); await sync.syncNow()
  expect(daemon.requests.some(row => row.method === 'shutdown' || row.method === 'turn.cancel')).toBe(false)
  expect(errors).toEqual([])
})
