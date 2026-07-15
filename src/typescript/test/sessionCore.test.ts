// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  AgentTransitionRecord,
  FileSessionStore,
  InMemorySessionStore,
  ReplayView,
  SessionFTSIndex,
  SessionManager,
  SessionRecord,
  SessionSummarizer,
  SnapshotManager,
  SQLiteSessionStore,
  ToolCallRecord,
  TurnRecord,
  branchSession,
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

test('SQLite session store persists, indexes turns, branches, and replays', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-session-'))
  const database = join(directory, 'sessions.db')
  let store: SQLiteSessionStore | undefined
  let restoredStore: SQLiteSessionStore | undefined
  try {
    store = new SQLiteSessionStore({ dbPath: database })
    const manager = new SessionManager(store)
    const session = manager.startSession({ sessionId: 'root', workspaceId: 'workspace', agentId: 'coder' })
    manager.recordTurn(session.sessionId, turn())
    manager.recordAgentTransition(session.sessionId, new AgentTransitionRecord({
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

test('shadow git snapshots restore files and produce a textual diff', () => {
  if (!Bun.which('git')) return
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-snapshot-'))
  const workspace = join(directory, 'workspace')
  const shadow = join(directory, 'shadow')
  try {
    mkdirSync(workspace)
    writeFileSync(join(workspace, 'a.txt'), 'first version', 'utf8')
    const snapshots = new SnapshotManager(workspace, { shadowRoot: shadow })
    const snapshot = snapshots.snapshot('first')
    writeFileSync(join(workspace, 'a.txt'), 'changed version', 'utf8')
    const diff = diffAgainstSnapshot(snapshots, snapshot.id)
    expect(diff.fileCount).toBe(1)
    expect(diff.added).toBeGreaterThan(0)
    snapshots.rollback(snapshot.id)
    expect(readFileSync(join(workspace, 'a.txt'), 'utf8')).toBe('first version')
    snapshots.reset()
    expect(snapshots.list()).toEqual([])
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})
