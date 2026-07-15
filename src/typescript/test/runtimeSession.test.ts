// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  MAX_HISTORY_TOOL_RESULT_CHARS,
  MAX_PERSISTED_STREAM_EVENTS,
  MAX_TOOL_EXECUTION_RESULT_CHARS,
  RuntimeContext,
  RuntimeSession,
  runtimeSessionPath,
  type RuntimeContextHost,
  type RuntimeSessionFileSystem,
} from '../src/runtime/session.js'

test('runtime sessions capture injected Bun context and resolve their model provider', () => {
  const session = RuntimeSession.create({
    contextHost: fixtureHost(),
    idFactory: () => 'runtime-session-1',
    model: 'anthropic/claude-sonnet-4-6',
    prompt: 'Review the native runtime.',
  })

  expect(session.sessionId).toBe('runtime-session-1')
  expect(session.context).toMatchObject({
    cwd: '/workspace/xerxes',
    runtimeVersion: 'Bun 1.3.0',
    platformName: 'darwin',
    gitBranch: 'main',
    model: 'anthropic/claude-sonnet-4-6',
    provider: 'anthropic',
    timestamp: '2026-07-13T12:00:00.000Z',
  })
  expect(session.context.toRecord()).toEqual({
    cwd: '/workspace/xerxes',
    runtime_version: 'Bun 1.3.0',
    platform_name: 'darwin',
    git_branch: 'main',
    model: 'anthropic/claude-sonnet-4-6',
    provider: 'anthropic',
    timestamp: '2026-07-13T12:00:00.000Z',
  })

  const unavailableGit = RuntimeContext.capture({
    host: { ...fixtureHost(), gitBranch: () => { throw new Error('not a repository') } },
    model: 'gpt-4o',
  })
  expect(unavailableGit.gitBranch).toBe('')
})

test('runtime sessions compose transcript, history, costs, and bounded raw records', () => {
  const session = RuntimeSession.create({
    contextHost: fixtureHost(),
    sessionId: 'runtime-session-2',
    model: 'openai/gpt-4o',
  })
  session.transcript.append('user', 'Please inspect the implementation.', { source: 'test' })
  const execution = session.recordToolExecution(
    'exec_command',
    { cmd: 'bun test' },
    'x'.repeat(MAX_TOOL_EXECUTION_RESULT_CHARS + 20),
    12.5,
    false,
  )
  for (let index = 0; index <= MAX_PERSISTED_STREAM_EVENTS; index += 1) {
    session.recordStreamEvent('text_chunk', { index })
  }
  session.recordTurn('openai/gpt-4o', 1_000, 500)

  expect(execution.result).toHaveLength(MAX_TOOL_EXECUTION_RESULT_CHARS)
  expect(session.history.events[0]?.detail).toHaveLength(MAX_HISTORY_TOOL_RESULT_CHARS)
  expect(session.history.events[0]).toMatchObject({ kind: 'tool_call', title: 'exec_command' })
  expect(session.costTracker.eventCount).toBe(1)
  expect(session.costTracker.events[0]?.sessionId).toBe('runtime-session-2')
  expect(session.streamEvents).toHaveLength(MAX_PERSISTED_STREAM_EVENTS + 1)

  const record = session.toRecord()
  expect(record.stream_events).toHaveLength(MAX_PERSISTED_STREAM_EVENTS)
  expect(record.stream_events[0]).toMatchObject({ type: 'text_chunk', index: 0 })
  expect(record.stream_events.at(-1)).toMatchObject({ index: MAX_PERSISTED_STREAM_EVENTS - 1 })
  expect(session.asMarkdown()).toContain('- `exec_command` [DENIED] (12.5ms)')
  expect(session.asMarkdown()).toContain('Total events: ' + (MAX_PERSISTED_STREAM_EVENTS + 1))
  expect(session.asMarkdown()).toContain('# Cost Summary')
  expect(session.asMarkdown()).toContain('# Transcript')
})

test('runtime session JSON snapshots round-trip through explicit filesystem ports', () => {
  const fileSystem = new MemoryFileSystem()
  const session = RuntimeSession.create({
    contextHost: fixtureHost(),
    sessionId: 'runtime-session-3',
    model: 'gpt-4o',
    prompt: 'Persist this session.',
    metadata: { title: 'Runtime snapshot', pinned: true },
  })
  session.transcript.append('user', 'Hello', { request: 'one' })
  session.transcript.append('assistant', 'Hi there')
  session.recordToolExecution('ReadFile', { path: '/workspace/xerxes/package.json' }, 'ok', 2.25)
  session.recordStreamEvent('tool_end', { call_id: 'call-1' })
  session.recordTurn('gpt-4o', 200, 50)

  const path = session.save({ directory: '/sessions', fileSystem })
  const snapshot = JSON.parse(fileSystem.readFile(path)) as Record<string, unknown>
  const restored = RuntimeSession.load({
    fileSystem,
    path,
    now: () => new Date('2026-07-14T00:00:00.000Z'),
  })

  expect(path).toBe(runtimeSessionPath('/sessions', 'runtime-session-3'))
  expect(fileSystem.directories).toEqual(['/sessions'])
  expect(session.transcript.flushed).toBe(true)
  expect(snapshot).toMatchObject({
    session_id: 'runtime-session-3',
    prompt: 'Persist this session.',
    metadata: { title: 'Runtime snapshot', pinned: true },
  })
  expect(restored.context).toEqual(session.context)
  expect(restored.metadata).toEqual(session.metadata)
  expect(restored.transcript.toMessages()).toEqual(session.transcript.toMessages())
  expect(restored.history.asDicts()).toEqual(session.history.asDicts())
  expect(restored.costTracker.asRecords()).toEqual(session.costTracker.asRecords())
  expect(restored.toolExecutions).toEqual(session.toolExecutions)
  expect(restored.streamEvents).toEqual(session.streamEvents)
  expect(restored.transcript.flushed).toBe(false)
})

test('runtime session parsing rejects malformed snapshots and unsafe file names', () => {
  const fileSystem = new MemoryFileSystem()
  fileSystem.writeFile('/sessions/bad.json', JSON.stringify({ session_id: 'bad' }))

  expect(() => RuntimeSession.load({ fileSystem, path: '/sessions/bad.json' })).toThrow('context must be an object')
  expect(() => runtimeSessionPath('/sessions', '../escape')).toThrow('single non-empty path segment')
  expect(() => new RuntimeContext({ timestamp: 'not-a-date' })).toThrow('timestamp')
})

function fixtureHost(): RuntimeContextHost {
  return {
    cwd: () => '/workspace/xerxes',
    gitBranch: cwd => cwd === '/workspace/xerxes' ? 'main' : '',
    now: () => new Date('2026-07-13T12:00:00.000Z'),
    platform: () => 'darwin',
    runtimeVersion: () => 'Bun 1.3.0',
  }
}

class MemoryFileSystem implements RuntimeSessionFileSystem {
  readonly directories: string[] = []
  private readonly files = new Map<string, string>()

  makeDirectory(path: string): void {
    if (!this.directories.includes(path)) this.directories.push(path)
  }

  readFile(path: string): string {
    const value = this.files.get(path)
    if (value === undefined) throw new Error('Missing file: ' + path)
    return value
  }

  writeFile(path: string, contents: string): void {
    this.files.set(path, contents)
  }
}
