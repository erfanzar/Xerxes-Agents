// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { Database } from 'bun:sqlite'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { Embedder } from '../src/memory/index.js'
import {
  AgentTransitionRecord,
  FileSessionStore,
  InMemorySessionStore,
  ReplayView,
  SessionFTSIndex,
  SessionIndex,
  SessionManager,
  SessionRecord,
  SessionSummarizer,
  SnapshotManager,
  SQLITE_SESSION_SCHEMA_VERSION,
  SQLiteSessionStore,
  ToolCallRecord,
  TurnRecord,
  branchSession,
  cloneSessionRecord,
  diffAgainstSnapshot,
  migrateSessionRecord,
  registerMigration,
  sessionLineage,
  unregisterMigration,
} from '../src/session/index.js'

function turn(turnId = 'turn-1', prompt = 'How do I deploy to Kubernetes?'): TurnRecord {
  return new TurnRecord({
    turnId,
    agentId: 'coder',
    prompt,
    responseContent: 'Use kubectl apply -f deployment.yaml',
    startedAt: '2026-01-01T00:00:00.000Z',
    endedAt: '2026-01-01T00:00:01.000Z',
    toolCalls: [new ToolCallRecord({
      callId: `${turnId}-tool`,
      toolName: 'exec_command',
      arguments: { cmd: 'kubectl apply -f deployment.yaml' },
      durationMs: 42,
    })],
  })
}

test('session records round-trip their nested shape and unknown fields', () => {
  const record = new SessionRecord({
    sessionId: 'session-1',
    workspaceId: 'workspace-1',
    agentId: 'coder',
    turns: [turn()],
    agentTransitions: [new AgentTransitionRecord({ fromAgent: 'coder', toAgent: 'reviewer', timestamp: 'now' })],
    metadata: { title: 'Deploy' },
    extra: { future_field: true },
  })
  const restored = SessionRecord.fromRecord(record.toRecord())

  expect(restored.sessionId).toBe('session-1')
  expect(restored.turns[0]?.toolCalls[0]?.toolName).toBe('exec_command')
  expect(restored.agentTransitions[0]?.toAgent).toBe('reviewer')
  expect(restored.toRecord().future_field).toBe(true)
})

test('migration registry upgrades only in the forward direction', () => {
  try {
    registerMigration(2, record => ({ ...record, migrated: true }))
    const migrated = migrateSessionRecord({ session_id: 'legacy', schema_version: 1 }, 2)
    expect(migrated.schema_version).toBe(2)
    expect(migrated.migrated).toBe(true)
    const future = { session_id: 'future', schema_version: 3 }
    expect(migrateSessionRecord(future, 2)).toBe(future)
  } finally {
    unregisterMigration(2)
  }
})

test('SQLite store migrates old records once and preserves migration fields', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-session-migrate-'))
  const database = join(directory, 'sessions.db')
  try {
    const first = new SQLiteSessionStore({ dbPath: database, schemaVersion: 1 })
    first.saveSession(new SessionRecord({ sessionId: 'legacy', schemaVersion: 1 }))
    first.close()
    registerMigration(2, record => {
      record.migration_marker = 'kept'
      return record
    })
    const upgraded = new SQLiteSessionStore({ dbPath: database, schemaVersion: 2 })
    const session = upgraded.loadSession('legacy')
    expect(session?.schemaVersion).toBe(2)
    expect(session?.extra.migration_marker).toBe('kept')
    upgraded.close()
  } finally {
    unregisterMigration(2)
    rmSync(directory, { recursive: true, force: true })
  }
})

test('SQLite session store persists, indexes turns, branches, and replays', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-session-'))
  const database = join(directory, 'sessions.db')
  let store: SQLiteSessionStore | undefined
  let restoredStore: SQLiteSessionStore | undefined
  try {
    store = new SQLiteSessionStore({ dbPath: database })
    const manager = new SessionManager(store)
    const session = manager.startSession({ sessionId: 'root', workspaceId: 'workspace', agentId: 'coder' })
    await manager.recordTurn(session.sessionId, turn())
    await manager.recordAgentTransition(session.sessionId, new AgentTransitionRecord({
      fromAgent: 'coder',
      toAgent: 'reviewer',
      turnId: 'turn-1',
      timestamp: '2026-01-01T00:00:01.000Z',
    }))

    expect(store.search('Kubernetes').map(hit => hit.turnId)).toEqual(['turn-1'])
    const child = branchSession(store, { sourceSessionId: 'root', newSessionId: 'child', title: 'Experiment' })
    child.turns[0]!.prompt = 'A different prompt'
    store.saveSession(child)
    expect(store.loadSession('root')?.turns[0]?.prompt).toContain('Kubernetes')
    expect(sessionLineage(store, 'child')).toEqual(['child', 'root'])

    const replay = new ReplayView(store.loadSession('root')!)
    expect(replay.getTimeline().map(event => event.eventType)).toContain('tool_call')
    expect(replay.toMarkdown()).toContain('exec_command')
    store.close()
    store = undefined

    restoredStore = new SQLiteSessionStore({ dbPath: database })
    expect(restoredStore.loadSession('root')?.turns).toHaveLength(1)
    expect(restoredStore.listSessions('workspace')).toContain('root')
  } finally {
    store?.close()
    restoredStore?.close()
    rmSync(directory, { recursive: true, force: true })
  }
})

test('file and in-memory session stores retain the shared SessionStore behavior', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-session-file-'))
  try {
    const fileStore = new FileSessionStore(directory)
    const session = new SessionRecord({ sessionId: 'file-session', workspaceId: 'workspace', turns: [turn()] })
    fileStore.saveSession(session)
    expect(existsSync(join(directory, 'workspace', 'file-session.json'))).toBe(true)
    expect(fileStore.loadSession('file-session')?.turns[0]?.turnId).toBe('turn-1')

    const memoryStore = new InMemorySessionStore()
    memoryStore.saveSession(session)
    expect(memoryStore.search('Kubernetes')).toHaveLength(1)
    expect(memoryStore.deleteSession('file-session')).toBe(true)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('cloneSessionRecord deep-copies nested metadata, arguments, and results', () => {
  const source = new SessionRecord({
    sessionId: 'source',
    metadata: { nested: { count: 1 } },
    turns: [new TurnRecord({
      turnId: 'turn-1',
      prompt: 'deploy',
      metadata: { tags: ['one'] },
      toolCalls: [new ToolCallRecord({
        callId: 'call-1',
        toolName: 'exec_command',
        arguments: { options: { flags: ['--force'] } },
        result: { data: { ids: [1, 2] } },
      })],
    })],
  })

  const clone = cloneSessionRecord(source)
  ;(clone.metadata.nested as { count: number }).count = 99
  ;(clone.turns[0]!.metadata.tags as string[]).push('two')
  ;(clone.turns[0]!.toolCalls[0]!.arguments.options as { flags: string[] }).flags.push('--dry-run')
  ;(clone.turns[0]!.toolCalls[0]!.result as { data: { ids: number[] } }).data.ids.push(3)

  expect((source.metadata.nested as { count: number }).count).toBe(1)
  expect(source.turns[0]!.metadata.tags).toEqual(['one'])
  expect((source.turns[0]!.toolCalls[0]!.arguments.options as { flags: string[] }).flags).toEqual(['--force'])
  expect((source.turns[0]!.toolCalls[0]!.result as { data: { ids: number[] } }).data.ids).toEqual([1, 2])
})

test('SQLite store saves rows and turn index on one connection and indexes appended turns incrementally', async () => {
  let embeddings = 0
  const embedder: Embedder = {
    dimension: 4,
    name: 'counting',
    embed: () => {
      embeddings += 1
      return [1, 0, 0, 0]
    },
    embedBatch: texts => texts.map(() => [1, 0, 0, 0]),
  }
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-session-incremental-'))
  const database = join(directory, 'sessions.db')
  let store: SQLiteSessionStore | undefined
  try {
    store = new SQLiteSessionStore({ dbPath: database, embedder })
    const manager = new SessionManager(store)
    const session = manager.startSession({ sessionId: 'incremental' })
    await manager.recordTurn(session.sessionId, turn('turn-1'))
    expect(embeddings).toBe(1)
    await manager.recordTurn(session.sessionId, turn('turn-2', 'Second prompt'))
    // Appending one turn embeds only that turn instead of re-embedding both.
    expect(embeddings).toBe(2)
    // Re-saving unchanged content indexes nothing new.
    store.saveSession(manager.getSession('incremental')!)
    expect(embeddings).toBe(2)
    expect(store.search('Kubernetes').map(hit => hit.turnId)).toEqual(['turn-1'])

    // The session row and its indexed turns commit together on one connection.
    const raw = new Database(database, { readonly: true })
    try {
      const sessionRows = raw.query('SELECT COUNT(*) AS count FROM sessions').get() as { count: number }
      const turnRows = raw.query('SELECT COUNT(*) AS count FROM session_turns WHERE session_id = ?').get('incremental') as { count: number }
      expect(sessionRows.count).toBe(1)
      expect(turnRows.count).toBe(2)
    } finally {
      raw.close()
    }
  } finally {
    store?.close()
    rmSync(directory, { recursive: true, force: true })
  }
})

test('SessionManager serializes concurrent mutations so no turn or transition is lost', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-session-race-'))
  const database = join(directory, 'sessions.db')
  let store: SQLiteSessionStore | undefined
  try {
    store = new SQLiteSessionStore({ dbPath: database })
    const manager = new SessionManager(store)
    const session = manager.startSession({ sessionId: 'race', workspaceId: 'workspace' })
    // Deliberately unsequenced: every writer must observe the others' saves.
    await Promise.all([
      manager.recordTurn(session.sessionId, turn('turn-a')),
      manager.recordTurn(session.sessionId, turn('turn-b')),
      manager.recordAgentTransition(session.sessionId, new AgentTransitionRecord({
        fromAgent: 'coder',
        toAgent: 'reviewer',
        turnId: 'turn-a',
        timestamp: '2026-01-01T00:00:02.000Z',
      })),
      manager.endSession(session.sessionId),
    ])
    const saved = store.loadSession('race')!
    expect(saved.turns.map(entry => entry.turnId).sort()).toEqual(['turn-a', 'turn-b'])
    expect(saved.agentTransitions.map(entry => entry.toAgent)).toEqual(['reviewer'])
    expect(saved.metadata.ended).toBe(true)
  } finally {
    store?.close()
    rmSync(directory, { recursive: true, force: true })
  }
})

test('SessionManager drains the per-session lock map after mutations settle, including failures', async () => {
  const store = new InMemorySessionStore()
  const manager = new SessionManager(store)
  const locks = (manager as unknown as { sessionLocks: Map<string, Promise<void>> }).sessionLocks

  const session = manager.startSession({ sessionId: 'drain' })
  const pending = Promise.all([
    manager.recordTurn(session.sessionId, turn('one')),
    manager.recordTurn(session.sessionId, turn('two')),
    manager.endSession(session.sessionId),
  ])
  // The lock chain exists while writers are queued...
  expect(locks.size).toBeGreaterThan(0)
  await pending
  // ...and is drained once the tail settles with no queued successor.
  expect(locks.size).toBe(0)

  // A failed mutation rejects but releases the chain instead of poisoning it.
  await expect(manager.recordTurn('missing', turn('missing'))).rejects.toThrow('Session not found')
  expect(locks.size).toBe(0)
  await manager.recordTurn(session.sessionId, turn('three'))
  expect(manager.getSession('drain')?.turns.map(entry => entry.turnId)).toEqual(['one', 'two', 'three'])
  expect(locks.size).toBe(0)
})

test('SQLite session store applies user_version migrations once and idempotently', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-session-userversion-'))
  const database = join(directory, 'sessions.db')
  let store: SQLiteSessionStore | undefined
  let reopened: SQLiteSessionStore | undefined
  try {
    // Simulate a pre-migration database: the v1 table exists but no
    // user_version was ever recorded.
    const legacy = new Database(database)
    try {
      legacy.run(`
        CREATE TABLE IF NOT EXISTS sessions (
          session_id TEXT PRIMARY KEY,
          workspace_id TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          agent_id TEXT,
          parent_session_id TEXT,
          schema_version INTEGER NOT NULL,
          metadata TEXT NOT NULL,
          record TEXT NOT NULL
        )
      `)
      legacy.query(`
        INSERT INTO sessions
          (session_id, workspace_id, created_at, updated_at, agent_id, parent_session_id, schema_version, metadata, record)
        VALUES ('legacy-row', NULL, 'now', 'now', NULL, NULL, 1, '{}', '{"session_id":"legacy-row"}')
      `).run()
      expect((legacy.query('PRAGMA user_version').get() as { user_version: number }).user_version).toBe(0)
    } finally {
      legacy.close()
    }

    // Opening upgrades the legacy database exactly once, preserving rows.
    store = new SQLiteSessionStore({ dbPath: database })
    const raw = new Database(database, { readonly: true })
    try {
      expect((raw.query('PRAGMA user_version').get() as { user_version: number }).user_version)
        .toBe(SQLITE_SESSION_SCHEMA_VERSION)
    } finally {
      raw.close()
    }
    expect(store.loadSession('legacy-row')?.sessionId).toBe('legacy-row')
    store.saveSession(new SessionRecord({ sessionId: 'fresh', turns: [turn()] }))
    store.close()
    store = undefined

    // Reopening is idempotent: the version stays put and no data is touched.
    reopened = new SQLiteSessionStore({ dbPath: database })
    const check = new Database(database, { readonly: true })
    try {
      expect((check.query('PRAGMA user_version').get() as { user_version: number }).user_version)
        .toBe(SQLITE_SESSION_SCHEMA_VERSION)
    } finally {
      check.close()
    }
    expect(reopened.loadSession('legacy-row')?.sessionId).toBe('legacy-row')
    expect(reopened.loadSession('fresh')?.turns).toHaveLength(1)
  } finally {
    store?.close()
    reopened?.close()
    rmSync(directory, { recursive: true, force: true })
  }
})

test('SQLite store skips corrupt records in list, search, and rebuild paths', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-session-corrupt-'))
  const database = join(directory, 'sessions.db')
  let store: SQLiteSessionStore | undefined
  try {
    store = new SQLiteSessionStore({ dbPath: database })
    store.saveSession(new SessionRecord({ sessionId: 'good', turns: [turn()] }))
    const raw = new Database(database)
    try {
      raw.query(`
        INSERT INTO sessions
          (session_id, workspace_id, created_at, updated_at, agent_id, parent_session_id, schema_version, metadata, record)
        VALUES ('corrupt', NULL, 'now', 'now', NULL, NULL, 1, '{}', 'this is not json')
      `).run()
    } finally {
      raw.close()
    }

    expect(store.listSessionRecords().map(session => session.sessionId)).toEqual(['good'])
    // Zero-hit indexed search falls back to scanning records; the corrupt row is skipped.
    expect(store.search('no-such-term-anywhere')).toEqual([])
    expect(store.search('Kubernetes').map(hit => hit.sessionId)).toEqual(['good'])
    expect(store.rebuildSearchIndex()).toBe(1)
    // Direct single-record loads still surface the corruption instead of hiding it.
    expect(() => store!.loadSession('corrupt')).toThrow()
    store.close()
    store = undefined

    const fileStore = new FileSessionStore(join(directory, 'files'))
    fileStore.saveSession(new SessionRecord({ sessionId: 'fine', turns: [turn()] }))
    writeFileSync(join(directory, 'files', 'broken.json'), 'not json', 'utf8')
    expect(fileStore.listSessionRecords().map(session => session.sessionId)).toEqual(['fine'])
    expect(fileStore.search('no-such-term-anywhere')).toEqual([])
  } finally {
    store?.close()
    rmSync(directory, { recursive: true, force: true })
  }
})

test('session search escapes LIKE wildcards in fallback queries', () => {
  const index = new SessionIndex({ dbPath: ':memory:' })
  try {
    index.indexSession(new SessionRecord({ sessionId: 'percent', turns: [turn('one', 'progress reached 100% today')] }))
    index.indexSession(new SessionRecord({ sessionId: 'wildcard', turns: [turn('two', 'progress reached 100x today')] }))
    // '%' must match literally instead of acting as a LIKE wildcard.
    expect(index.search('100%').map(hit => hit.sessionId)).toEqual(['percent'])
  } finally {
    index.close()
  }

  const fts = new SessionFTSIndex(':memory:')
  try {
    if (!fts.ftsAvailable) return
    fts.indexSession(new SessionRecord({ sessionId: 'percent', turns: [turn('one', 'deploy at 100% capacity')] }))
    fts.indexSession(new SessionRecord({ sessionId: 'wildcard', turns: [turn('two', 'deploy at 100x capacity')] }))
    expect(fts.search('100%').map(hit => hit.sessionId)).toEqual(['percent'])
  } finally {
    fts.close()
  }
})

test('FTS search only falls back to LIKE on query syntax errors', () => {
  const index = new SessionFTSIndex(':memory:')
  index.indexSession(new SessionRecord({ sessionId: 'fts', turns: [turn()] }))
  index.close()
  // A closed database is a storage failure, not a syntax error: it must surface.
  expect(() => index.search('Kubernetes')).toThrow()
})

test('standalone FTS index replaces stale session rows', () => {
  const index = new SessionFTSIndex(':memory:')
  try {
    if (!index.ftsAvailable) return
    const session = new SessionRecord({ sessionId: 'fts', turns: [turn('one', 'first version')] })
    index.indexSession(session)
    session.turns[0]!.prompt = 'second version'
    index.indexSession(session)
    expect(index.search('second')).toHaveLength(1)
    expect(index.search('first')).toEqual([])
  } finally {
    index.close()
  }
})

test('summaries derive actions and safely use the heuristic fallback', () => {
  const session = new SessionRecord({
    sessionId: 'summary',
    turns: [turn('one'), new TurnRecord({ turnId: 'two', agentId: 'reviewer', prompt: 'review', status: 'error' })],
  })
  const summary = new SessionSummarizer().summarize(session)
  expect(summary.outcome).toBe('mixed')
  expect(summary.keyActions).toEqual(['exec_command'])
  expect(summary.agentIds).toEqual(['coder', 'reviewer'])
})

test('shadow git snapshots restore files and produce a textual diff', async () => {
  if (!Bun.which('git')) return
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-snapshot-'))
  const workspace = join(directory, 'workspace')
  const shadow = join(directory, 'shadow')
  try {
    mkdirSync(workspace)
    writeFileSync(join(workspace, 'a.txt'), 'first version', 'utf8')
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    const snapshot = await snapshots.snapshot('first')
    writeFileSync(join(workspace, 'a.txt'), 'changed version', 'utf8')
    const diff = await diffAgainstSnapshot(snapshots, snapshot.id)
    expect(diff.fileCount).toBe(1)
    expect(diff.added).toBeGreaterThan(0)
    await snapshots.rollback(snapshot.id)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('first version')
    snapshots.reset()
    expect(snapshots.list()).toEqual([])
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('shadow git snapshots exclude secrets, stay private, and rollback removes post-snapshot files', async () => {
  if (!Bun.which('git')) return
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-snapshot-secrets-'))
  const workspace = join(directory, 'workspace')
  const shadow = join(directory, 'shadow')
  try {
    mkdirSync(workspace)
    writeFileSync(join(workspace, 'a.txt'), 'first version', 'utf8')
    writeFileSync(join(workspace, '.env'), 'TOKEN=secret', 'utf8')
    writeFileSync(join(workspace, 'id_rsa'), 'private-key', 'utf8')
    writeFileSync(join(workspace, 'service.key'), 'key-material', 'utf8')
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    const snapshot = await snapshots.snapshot('base')

    const tracked = await snapshots.runGit(['ls-files'])
    expect(tracked).toContain('a.txt')
    expect(tracked).not.toContain('.env')
    expect(tracked).not.toContain('id_rsa')
    expect(tracked).not.toContain('service.key')
    expect(statSync(snapshots.shadowDirectory).mode & 0o777).toBe(0o700)

    writeFileSync(join(workspace, 'a.txt'), 'changed version', 'utf8')
    writeFileSync(join(workspace, 'created-later.txt'), 'new file', 'utf8')
    await snapshots.rollback(snapshot.id)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('first version')
    // Full-tree restore removes files created after the snapshot...
    expect(existsSync(join(workspace, 'created-later.txt'))).toBe(false)
    // ...but never deletes excluded secret files.
    expect(readFileSync(join(workspace, '.env'), 'utf8')).toBe('TOKEN=secret')
    expect(existsSync(join(workspace, 'id_rsa'))).toBe(true)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('shadow git prune rewrites retained history and collects pruned commits', async () => {
  if (!Bun.which('git')) return
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-snapshot-prune-'))
  const workspace = join(directory, 'workspace')
  const shadow = join(directory, 'shadow')
  try {
    mkdirSync(workspace)
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    writeFileSync(join(workspace, 'a.txt'), 'one', 'utf8')
    const first = await snapshots.snapshot('first')
    writeFileSync(join(workspace, 'a.txt'), 'two', 'utf8')
    await snapshots.snapshot('second')
    writeFileSync(join(workspace, 'a.txt'), 'three', 'utf8')
    const third = await snapshots.snapshot('third')

    expect(await snapshots.prune({ keep: 1 })).toBe(2)
    const retained = snapshots.list()
    expect(retained).toHaveLength(1)
    expect(retained[0]?.id).toBe(third.id)
    // Pruned commits are no longer reachable in the shadow repository.
    await expect(snapshots.runGit(['cat-file', '-e', first.commitSha])).rejects.toThrow()
    // The retained snapshot still rolls back to its full tree.
    writeFileSync(join(workspace, 'a.txt'), 'dirty', 'utf8')
    await snapshots.rollback(third.id)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('three')
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})
