// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { HookRunner } from '../src/extensions/hooks.js'
import {
  HashEmbedder,
  LongTermMemory,
  type Memory,
  SimpleStorage,
  makeMemoryProvider,
  makeTurnIndexerHook,
} from '../src/memory/index.js'
import {
  AgentTransitionRecord,
  CURRENT_SESSION_SCHEMA_VERSION,
  FileSessionStore,
  InMemorySessionStore,
  ReplayView,
  SessionFTSIndex,
  SessionIndex,
  SessionManager,
  SessionRecord,
  SessionReplay,
  SessionSummarizer,
  SnapshotManager,
  ToolCallRecord,
  TurnRecord,
  branchSession,
  migrateSessionRecord,
  registerMigration,
  sessionLineage,
  unregisterMigration,
} from '../src/session/index.js'
import { SearchHistoryTool } from '../src/tools/history.js'

function toolCall(callId: string, toolName = 'search'): ToolCallRecord {
  return new ToolCallRecord({
    callId,
    toolName,
    arguments: { query: 'hello' },
    result: 'found it',
    durationMs: 42.5,
  })
}

function turn(
  turnId: string,
  options: {
    readonly agentId?: string
    readonly endedAt?: string
    readonly prompt?: string
    readonly responseContent?: string
    readonly status?: string
    readonly toolCalls?: readonly ToolCallRecord[]
  } = {},
): TurnRecord {
  return new TurnRecord({
    turnId,
    agentId: options.agentId ?? 'agent-a',
    prompt: options.prompt ?? `prompt ${turnId}`,
    responseContent: options.responseContent ?? `response ${turnId}`,
    startedAt: `2026-01-01T00:00:0${turnId.at(-1) ?? '0'}.000Z`,
    endedAt: options.endedAt ?? `2026-01-01T00:01:0${turnId.at(-1) ?? '0'}.000Z`,
    status: options.status ?? 'success',
    toolCalls: options.toolCalls ?? [],
  })
}

test('session models, stores, and manager retain the Python persistence lifecycle contracts', () => {
  const defaults = ToolCallRecord.fromRecord({ call_id: 'call', tool_name: 'search', arguments: {} })
  expect(defaults.status).toBe('success')
  expect(defaults.error).toBeNull()
  expect(defaults.toRecord()).toEqual({
    call_id: 'call',
    tool_name: 'search',
    arguments: {},
    result: null,
    status: 'success',
    error: null,
    duration_ms: null,
    sandbox_context: null,
    metadata: {},
  })

  const emptyRecord = new SessionRecord({ sessionId: 'empty' })
  expect(emptyRecord.toRecord().schema_version).toBe(CURRENT_SESSION_SCHEMA_VERSION)
  const empty = SessionRecord.fromRecord(emptyRecord.toRecord())
  expect(empty.schemaVersion).toBe(CURRENT_SESSION_SCHEMA_VERSION)
  expect(empty.turns).toEqual([])
  expect(empty.agentTransitions).toEqual([])
  expect(SessionRecord.fromRecord({ session_id: 'legacy' }).schemaVersion).toBe(CURRENT_SESSION_SCHEMA_VERSION)
  const childRecord = new SessionRecord({ sessionId: 'child', parentSessionId: 'parent' })
  expect(SessionRecord.fromRecord(childRecord.toRecord()).parentSessionId).toBe('parent')

  const memoryStore = new InMemorySessionStore()
  memoryStore.saveSession(new SessionRecord({ sessionId: 's1', workspaceId: 'w1', metadata: { version: 1 } }))
  memoryStore.saveSession(new SessionRecord({ sessionId: 's2', workspaceId: 'w2' }))
  memoryStore.saveSession(new SessionRecord({ sessionId: 's3', workspaceId: 'w1' }))
  memoryStore.saveSession(new SessionRecord({ sessionId: 's1', workspaceId: 'w1', metadata: { version: 2 } }))
  expect(new Set(memoryStore.listSessions())).toEqual(new Set(['s1', 's2', 's3']))
  expect(new Set(memoryStore.listSessions('w1'))).toEqual(new Set(['s1', 's3']))
  expect(memoryStore.loadSession('s1')?.metadata.version).toBe(2)
  expect(memoryStore.loadSession('missing')).toBeUndefined()
  expect(memoryStore.deleteSession('s2')).toBeTrue()
  expect(memoryStore.deleteSession('missing')).toBeFalse()

  const manager = new SessionManager(memoryStore)
  const session = manager.startSession({ sessionId: 'explicit', workspaceId: 'workspace', agentId: 'agent-a' })
  manager.recordTurn(session.sessionId, turn('t1'))
  manager.recordAgentTransition(session.sessionId, new AgentTransitionRecord({
    fromAgent: 'agent-a',
    toAgent: 'agent-b',
    turnId: 't1',
    timestamp: '2026-01-01T00:00:05.000Z',
  }))
  manager.endSession(session.sessionId)
  expect(manager.getSession(session.sessionId)).toMatchObject({
    workspaceId: 'workspace',
    agentId: 'agent-a',
    metadata: { ended: true },
  })
  expect(manager.getSession(session.sessionId)?.turns).toHaveLength(1)
  expect(manager.getSession(session.sessionId)?.agentTransitions).toHaveLength(1)
  expect(() => manager.recordTurn('missing', turn('missing'))).toThrow('Session not found')
  expect(() => manager.recordAgentTransition('missing', new AgentTransitionRecord({ toAgent: 'agent-b' }))).toThrow(
    'Session not found',
  )
  expect(() => manager.endSession('missing')).toThrow('Session not found')
})

test('file sessions retain flat and workspace layouts while applying forward migrations atomically', () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-session-python-parity-'))
  const migrationTarget = 831
  try {
    const store = new FileSessionStore(directory)
    store.saveSession(new SessionRecord({ sessionId: 'flat' }))
    store.saveSession(new SessionRecord({ sessionId: 'nested', workspaceId: 'workspace', turns: [turn('t1')] }))
    expect(existsSync(join(directory, 'flat.json'))).toBeTrue()
    expect(existsSync(join(directory, 'workspace', 'nested.json'))).toBeTrue()
    expect(store.loadSession('missing')).toBeUndefined()
    expect(new Set(store.listSessions())).toEqual(new Set(['flat', 'nested']))
    expect(store.listSessions('workspace')).toEqual(['nested'])
    expect(store.deleteSession('flat')).toBeTrue()
    expect(store.deleteSession('flat')).toBeFalse()

    writeFileSync(join(directory, 'legacy.json'), JSON.stringify({ session_id: 'legacy', schema_version: migrationTarget - 1 }))
    registerMigration(migrationTarget, record => ({ ...record, migrated_by_native_bun: true }))
    const migratingStore = new FileSessionStore({ baseDirectory: directory, schemaVersion: migrationTarget })
    const legacy = migratingStore.loadSession('legacy')
    expect(legacy?.schemaVersion).toBe(migrationTarget)
    expect(legacy?.extra.migrated_by_native_bun).toBe(true)
    const persisted = JSON.parse(readFileSync(join(directory, 'legacy.json'), 'utf8')) as Record<string, unknown>
    expect(persisted).toMatchObject({ schema_version: migrationTarget, migrated_by_native_bun: true })
    expect(readdirSync(directory).filter(name => name.endsWith('.tmp'))).toEqual([])
  } finally {
    unregisterMigration(migrationTarget)
    rmSync(directory, { recursive: true, force: true })
  }
})

test('the migration registry chains upgrades, rejects duplicate and missing steps, and never downgrades', () => {
  const first = 841
  const second = 842
  const duplicate = 843
  try {
    registerMigration(first, record => ({ ...record, first: true }))
    registerMigration(second, record => ({ ...record, second: record.first === true }))
    expect(migrateSessionRecord({ session_id: 'legacy', schema_version: first - 1 }, second)).toMatchObject({
      schema_version: second,
      first: true,
      second: true,
    })
    const future = { session_id: 'future', schema_version: second + 1 }
    expect(migrateSessionRecord(future, second)).toBe(future)

    registerMigration(duplicate, record => record)
    expect(() => registerMigration(duplicate, record => record)).toThrow('Duplicate migration')
    expect(() => migrateSessionRecord({ session_id: 'missing', schema_version: duplicate + 1 }, duplicate + 2)).toThrow(
      'No migration registered',
    )
  } finally {
    unregisterMigration(first)
    unregisterMigration(second)
    unregisterMigration(duplicate)
  }
})

test('branches and replay views preserve full copied history, projections, and timeline detail', () => {
  const store = new InMemorySessionStore()
  const source = new SessionRecord({
    sessionId: 'root',
    agentId: 'agent-a',
    turns: [
      turn('t1', { agentId: 'agent-a', toolCalls: [toolCall('tc1', 'search'), toolCall('tc2', 'calc')] }),
      turn('t2', { agentId: 'agent-b', toolCalls: [toolCall('tc3', 'fetch')] }),
      turn('t3', { agentId: 'agent-a' }),
    ],
    agentTransitions: [
      new AgentTransitionRecord({ fromAgent: 'agent-a', toAgent: 'agent-b', reason: 'capability', turnId: 't1', timestamp: '2026-01-01T00:00:05.000Z' }),
      new AgentTransitionRecord({ fromAgent: 'agent-b', toAgent: 'agent-a', reason: 'review', turnId: 't2', timestamp: '2026-01-01T00:01:05.000Z' }),
    ],
  })
  store.saveSession(source)

  const child = branchSession(store, { sourceSessionId: 'root', newSessionId: 'child', title: 'experiment' })
  child.turns[0]!.prompt = 'mutated child prompt'
  store.saveSession(child)
  expect(child.parentSessionId).toBe('root')
  expect(child.metadata).toMatchObject({ forked_from: 'root', title: 'experiment' })
  expect(child.turns.map(record => record.turnId)).toEqual(['t1', 't2', 't3'])
  expect(store.loadSession('root')?.turns[0]?.prompt).toBe('prompt t1')
  expect(sessionLineage(store, 'child')).toEqual(['child', 'root'])
  expect(() => branchSession(store, { sourceSessionId: 'missing' })).toThrow('unknown source session')

  const view = SessionReplay.load(source)
  expect(view.session).toBe(source)
  expect(new ReplayView(source, [source.turns[0]!]).turns).toHaveLength(1)
  expect(view.getTurn(0)?.turnId).toBe('t1')
  expect(view.getTurn('t2')?.agentId).toBe('agent-b')
  expect(view.getTurn(-1)).toBeUndefined()
  expect(view.getTurn(99)).toBeUndefined()
  expect(view.getTurn('missing')).toBeUndefined()
  expect(view.getToolCalls().map(call => call.toolName)).toEqual(['search', 'calc', 'fetch'])
  expect(view.getAgentTransitions().map(transition => transition.toAgent)).toEqual(['agent-b', 'agent-a'])

  const timeline = view.getTimeline()
  expect(timeline.map(event => event.timestamp)).toEqual([...timeline.map(event => event.timestamp)].sort())
  expect(timeline.filter(event => event.eventType === 'turn_start')).toHaveLength(3)
  expect(timeline.filter(event => event.eventType === 'turn_end')).toHaveLength(3)
  expect(timeline.filter(event => event.eventType === 'tool_call')).toHaveLength(3)
  expect(timeline.filter(event => event.eventType === 'agent_transition')).toHaveLength(2)

  const filtered = view.filterByAgent('agent-a')
  expect(filtered.session).toBe(source)
  expect(filtered.turns.map(record => record.turnId)).toEqual(['t1', 't3'])
  expect(filtered.getToolCalls().map(call => call.toolName)).toEqual(['search', 'calc'])
  expect(view.filterByAgent('missing').turns).toEqual([])
  const markdown = view.toMarkdown()
  expect(markdown).toStartWith('# Session root')
  expect(markdown).toContain('search')
  expect(markdown).toContain('agent-a -> agent-b')
})

test('native FTS, indexed history, and session search retain filters, replacement, and deletion semantics', () => {
  const fts = new SessionFTSIndex(':memory:')
  const index = new SessionIndex({ embedder: new HashEmbedder(64) })
  try {
    if (fts.ftsAvailable) {
      fts.indexSession(new SessionRecord({
        sessionId: 'fts-1',
        turns: [turn('t1', { prompt: 'How do I deploy to Kubernetes?' })],
      }))
      fts.indexSession(new SessionRecord({
        sessionId: 'fts-2',
        turns: [turn('t2', { prompt: 'What about Kubernetes networking?' })],
      }))
      expect(new Set(fts.search('Kubernetes').map(hit => hit.sessionId))).toEqual(new Set(['fts-1', 'fts-2']))
      expect(fts.search('Kubernetes', { sessionId: 'fts-1' }).map(hit => hit.sessionId)).toEqual(['fts-1'])
      fts.deleteSession('fts-1')
      expect(fts.search('deploy')).toEqual([])
      expect(fts.search('   ')).toEqual([])
    }

    const indexed = new SessionRecord({
      sessionId: 'indexed',
      turns: [
        turn('t1', { agentId: 'coder', prompt: 'set up continuous integration with github actions' }),
        turn('t2', { agentId: 'reviewer', prompt: 'make me a sandwich' }),
      ],
    })
    index.indexSession(indexed)
    expect(index.search('github actions', { k: 2 }).at(0)?.turnId).toBe('t1')
    expect(index.search('github actions', { k: 2 }).at(0)?.score).toBeGreaterThan(0)
    expect(index.search('continuous', { agentId: 'coder' }).map(hit => hit.turnId)).toEqual(['t1'])
    expect(index.search('continuous', { agentId: 'reviewer' })).toEqual([])
    expect(index.search('continuous', { sessionId: 'indexed' }).map(hit => hit.sessionId)).toEqual(['indexed'])
    indexed.turns[0]!.prompt = 'second version'
    index.indexSession(indexed)
    expect(index.search('second').map(hit => hit.turnId)).toEqual(['t1'])
    expect(index.search('continuous')).toEqual([])
    expect(new SearchHistoryTool({ index }).search('second')).toMatchObject({
      count: 1,
      hits: [{ session_id: 'indexed', turn_id: 't1' }],
    })
    expect(index.removeSession('indexed')).toBe(2)
    expect(index.search('second')).toEqual([])
    expect(index.search('')).toEqual([])

    const store = new InMemorySessionStore()
    store.saveSession(new SessionRecord({
      sessionId: 'history',
      turns: [turn('history-turn', { agentId: 'coder', prompt: 'set up CI with github actions' })],
    }))
    const history = new SearchHistoryTool({ store })
    expect(history.search('CI', { agentId: 'coder' })).toMatchObject({
      count: 1,
      hits: [{ session_id: 'history', turn_id: 'history-turn', agent_id: 'coder' }],
    })
    expect(history.search('')).toEqual({ query: '', count: 0, hits: [] })
  } finally {
    fts.close()
    index.close()
  }
})

test('summaries and snapshots preserve deterministic history inspection behavior', () => {
  const summarizer = new SessionSummarizer()
  const empty = summarizer.summarize(new SessionRecord({ sessionId: 'empty' }))
  expect(empty.title).toStartWith('Session ')
  expect(empty.synopsis).toContain('Empty')
  expect(empty.outcome).toBe('unknown')

  const longPrompt = Array.from({ length: 50 }, () => 'word').join(' ')
  const summary = summarizer.summarize(new SessionRecord({
    sessionId: 'summary',
    turns: [
      turn('t1', { prompt: longPrompt, agentId: 'coder', toolCalls: [toolCall('read', 'Read'), toolCall('bash', 'Bash')] }),
      turn('t2', { agentId: 'reviewer', status: 'error', toolCalls: [toolCall('bash-2', 'Bash'), toolCall('edit', 'Edit')] }),
    ],
  }))
  expect(summary.title).toContain('…')
  expect(summary.outcome).toBe('mixed')
  expect(summary.keyActions).toEqual(['Read', 'Bash', 'Edit'])
  expect(summary.agentIds).toEqual(['coder', 'reviewer'])
  expect(summarizer.summarize(new SessionRecord({
    sessionId: 'title',
    turns: [turn('t1', { prompt: 'configure github actions for my project' })],
  })).title.toLowerCase()).toContain('github actions')
  expect(summarizer.summarize(new SessionRecord({ sessionId: 'ok', turns: [turn('t1')] })).outcome).toBe('success')
  expect(summarizer.summarize(new SessionRecord({ sessionId: 'bad', turns: [turn('t1', { status: 'error' })] })).outcome).toBe('failure')

  let refinements = 0
  const refined = new SessionSummarizer({
    llmClient: () => {
      refinements += 1
      return 'Refined native synopsis.'
    },
  }).summarize(new SessionRecord({ sessionId: 'refined', turns: [turn('t1')] }))
  expect(refinements).toBe(1)
  expect(refined.synopsis).toBe('Refined native synopsis.')
  const fallback = new SessionSummarizer({
    llmClient: () => {
      throw new Error('unavailable')
    },
  }).summarize(new SessionRecord({ sessionId: 'fallback', turns: [turn('t1')] }))
  expect(fallback.synopsis).toContain('User asked')

  if (!Bun.which('git')) return
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-snapshot-python-parity-'))
  try {
    const workspace = join(directory, 'workspace')
    const manager = new SnapshotManager(workspace, { shadowRoot: join(directory, 'shadow') })
    mkdirSync(workspace)
    writeFileSync(join(workspace, 'note.txt'), 'version one', 'utf8')
    const first = manager.snapshot('first')
    const second = manager.snapshot('second')
    expect(manager.get(first.id)).toEqual(first)
    expect(manager.get('second')).toEqual(second)
    expect(manager.get(first.commitSha.slice(0, 7))).toEqual(first)
    expect(() => manager.rollback('missing')).toThrow('snapshot not found')
    manager.snapshot('third')
    manager.snapshot('fourth')
    manager.snapshot('fifth')
    expect(manager.prune({ keep: 2 })).toBe(3)
    expect(manager.list()).toHaveLength(2)
    manager.reset()
    expect(manager.list()).toEqual([])
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('turn indexing normalizes responses, participates in hooks, and cannot abort a turn on storage failure', () => {
  class ContentResponse {
    content = 'A useful attribute response'
  }

  const memory = new LongTermMemory({ storage: new SimpleStorage(), enableEmbeddings: false })
  const hook = makeTurnIndexerHook(memory, { minChars: 1 })
  hook({ agentId: 'agent-a', response: 'A useful string response' })
  hook({ agentId: 'agent-a', response: { content: 'A useful object response' } })
  hook({ agentId: 'agent-a', response: { content: [{ text: 'A useful block response' }] } })
  hook({ agentId: 'agent-a', response: new ContentResponse() })
  hook({ agentId: 'agent-a', response: undefined })
  expect(memory.size).toBe(4)
  expect(memory.search('attribute', 1).at(0)?.metadata.source).toBe('turn_indexer')
  expect(memory.search('attribute', 1).at(0)?.agentId).toBe('agent-a')
  expect(makeMemoryProvider(memory, false)('useful', 5)).toEqual(expect.arrayContaining([
    'A useful string response',
    'A useful attribute response',
  ]))
  makeTurnIndexerHook(memory, { minChars: 20 })({ response: 'short' })
  expect(memory.size).toBe(4)

  const runner = new HookRunner()
  runner.register('on_turn_end', hook)
  runner.run('on_turn_end', { agentId: 'agent-b', response: 'A useful hook response' })
  expect(memory.size).toBe(5)

  const brokenMemory = {
    save() {
      throw new Error('disk full')
    },
  } as unknown as Memory
  expect(() => makeTurnIndexerHook(brokenMemory, { minChars: 1 })({ response: 'A useful broken response' })).not.toThrow()

  const brokenSearchMemory = {
    search() {
      throw new Error('search unavailable')
    },
  } as unknown as Memory
  expect(makeMemoryProvider(brokenSearchMemory)('agent-a', 5)).toEqual([])
})
