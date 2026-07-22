// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { AcpEventKind, toAcpEvent } from '../src/acp/events.js'
import { AcpPermissionBoard, routePermission } from '../src/acp/permissions.js'
import { ACP_REGISTRY_METADATA, writeAcpRegistryFile } from '../src/acp/registry.js'
import { AcpServer, ServerCapabilities } from '../src/acp/server.js'
import { AcpSessionConflictError, AcpSessionStore } from '../src/acp/session.js'

test('ACP registry manifest preserves IDE discovery metadata', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-acp-registry-'))
  try {
    const output = await writeAcpRegistryFile(directory)
    expect(output).toBe(join(directory, 'xerxes', 'agent.json'))
    expect(JSON.parse(await readFile(output, 'utf8'))).toEqual(ACP_REGISTRY_METADATA)
    expect(ACP_REGISTRY_METADATA.distribution).toEqual({ type: 'command', command: 'xerxes-acp', args: [] })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('ACP event conversion preserves streaming event semantics', () => {
  expect(toAcpEvent({ type: 'text', text: 'hello' }).toWire()).toEqual({ kind: AcpEventKind.TEXT_DELTA, text: 'hello' })
  expect(toAcpEvent({ type: 'thinking', text: 'plan' }).toWire()).toEqual({ kind: AcpEventKind.THINKING_DELTA, text: 'plan' })
  expect(toAcpEvent({
    type: 'tool_start',
    call: { id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } },
  }).toWire()).toEqual({
    kind: AcpEventKind.TOOL_CALL_START,
    name: 'ReadFile',
    inputs: { path: 'README.md' },
    tool_call_id: 'call-1',
  })
  expect(toAcpEvent({
    type: 'turn_done',
    model: 'gpt-4o',
    toolCallsCount: 2,
    usage: { inputTokens: 3, outputTokens: 5, cacheReadTokens: 7, cacheCreationTokens: 11 },
  }).toWire()).toEqual({
    kind: AcpEventKind.TURN_END,
    input_tokens: 3,
    output_tokens: 5,
    tool_calls_count: 2,
    model: 'gpt-4o',
    cache_read_tokens: 7,
    cache_creation_tokens: 11,
  })
  expect(toAcpEvent({ type: 'provider_retry', attempt: 1 }).kind).toBe(AcpEventKind.UNKNOWN)
})

test('ACP session store creates, mutates, and removes isolated sessions', () => {
  const sessions = new AcpSessionStore()
  const session = sessions.create('/workspace', { model: 'gpt-4o', title: 'Investigate tests' })

  expect(sessions.get(session.sessionId)).toBe(session)
  expect(sessions.setModel(session.sessionId, 'claude-sonnet-4-5')).toBe(true)
  expect(session.modelOverride).toBe('claude-sonnet-4-5')
  expect(sessions.cancel(session.sessionId)).toBe(true)
  expect(session.cancelled).toBe(true)
  expect(sessions.drop(session.sessionId)).toBe(true)
  expect(sessions.get(session.sessionId)).toBeUndefined()
})

test('ACP session store rejects duplicate attachExisting instead of dropping state', () => {
  const sessions = new AcpSessionStore()
  const session = sessions.create('/workspace', { model: 'gpt-4o', title: 'Keep me' })
  sessions.cancel(session.sessionId)

  expect(() => sessions.attachExisting({ sessionId: session.sessionId }, '/elsewhere'))
    .toThrow(AcpSessionConflictError)
  const retained = sessions.get(session.sessionId)
  expect(retained).toBe(session)
  expect(retained?.title).toBe('Keep me')
  expect(retained?.modelOverride).toBe('gpt-4o')
  expect(retained?.cancelled).toBe(true)

  const attached = sessions.attachExisting({ sessionId: 'external-session-1' }, '/elsewhere')
  expect(attached.sessionId).toBe('external-session-1')
  expect(sessions.get('external-session-1')).toBe(attached)
})

test('ACP permission board resolves pending decisions and cleans up aborted waits', async () => {
  const board = new AcpPermissionBoard()
  const request = routePermission({ sessionId: 'session-1', toolName: 'WriteFile', description: 'Write a file', inputs: {} })
  const decision = board.submit(request)

  expect(board.snapshotPending()).toEqual([request])
  expect(board.resolve(request.id, true)).toBe(true)
  await expect(decision).resolves.toBe(true)
  expect(board.resolve(request.id, false)).toBe(false)
  expect(board.snapshotPending()).toEqual([])

  const controller = new AbortController()
  const cancelled = routePermission({ sessionId: 'session-1', toolName: 'Bash', description: 'Run a command', inputs: {} })
  const waiting = board.awaitDecision(cancelled, controller.signal)
  controller.abort()
  await expect(waiting).resolves.toBe(false)
  expect(board.get(cancelled.id)).toBeUndefined()
})

test('ACP server exposes capabilities, live sessions, prompts, and approval responses', async () => {
  let received: { readonly sessionId: string; readonly text: string } | undefined
  const closedSessions: string[] = []
  const server = new AcpServer({
    capabilities: new ServerCapabilities({ protocolVersion: '0.9-test', fork: false }),
    promptHandler: async ({ session, text }) => {
      received = { sessionId: session.sessionId, text }
      return { ok: true, echo: text }
    },
    toolListProvider: () => [{
      type: 'function',
      function: { name: 'echo', description: 'Echo text.', parameters: { type: 'object' } },
    }],
    modelListProvider: () => [{ id: 'gpt-4o', name: 'GPT-4o' }],
    onSessionClose: sessionId => closedSessions.push(sessionId),
  })

  expect(server.initialize()).toEqual({
    server_name: 'xerxes',
    capabilities: { protocol_version: '0.9-test', streaming: true, tools: true, permissions: true, fork: false },
  })
  expect(server.listTools()[0]?.function.name).toBe('echo')
  expect(server.listModels()).toEqual([{ id: 'gpt-4o', name: 'GPT-4o' }])

  const opened = server.openSession('/workspace', { model: 'gpt-4o', title: 'Demo' })
  const sessionId = String(opened.session_id)
  await expect(server.prompt(sessionId, 'hello ACP')).resolves.toEqual({ ok: true, echo: 'hello ACP' })
  expect(received).toEqual({ sessionId, text: 'hello ACP' })
  expect(server.listSessions()).toEqual([{
    session_id: sessionId,
    cwd: '/workspace',
    model: 'gpt-4o',
    title: 'Demo',
    cancelled: false,
  }])

  const permissionId = String(server.requestPermission({
    sessionId,
    toolName: 'WriteFile',
    description: 'Write a file',
    inputs: {},
  }).permission_id)
  expect(server.pendingPermissions()[0]).toMatchObject({ id: permissionId, session_id: sessionId })
  expect(server.respondPermission(permissionId, true)).toEqual({ ok: true })
  expect(server.pendingPermissions()).toEqual([])
  expect(server.cancel(sessionId)).toEqual({ ok: true })
  // Cancellation keeps the session alive and promptable: onSessionClose must
  // fire only from closeSession or shutdown.
  expect(closedSessions).toEqual([])
  expect(server.closeSession(sessionId)).toEqual({ ok: true })
  expect(closedSessions).toEqual([sessionId])
})

test('ACP server serializes prompts per session while keeping sessions concurrent', async () => {
  const order: string[] = []
  const gates = new Map<string, () => void>()
  const server = new AcpServer({
    promptHandler: async ({ session, text }) => {
      order.push(`start:${session.sessionId}:${text}`)
      await new Promise<void>(resolve => gates.set(`${session.sessionId}:${text}`, resolve))
      order.push(`end:${session.sessionId}:${text}`)
      return { ok: true }
    },
  })
  const firstSession = String(server.openSession('/workspace').session_id)
  const secondSession = String(server.openSession('/workspace').session_id)

  const first = server.prompt(firstSession, 'one')
  const queued = server.prompt(firstSession, 'two')
  const otherSession = server.prompt(secondSession, 'parallel')
  await Bun.sleep(20)

  // The second prompt for the same session waits, but another session runs.
  expect(order).toEqual([
    `start:${firstSession}:one`,
    `start:${secondSession}:parallel`,
  ])
  gates.get(`${secondSession}:parallel`)?.()
  await otherSession
  gates.get(`${firstSession}:one`)?.()
  await first
  await Bun.sleep(20)
  expect(order).toContain(`start:${firstSession}:two`)
  gates.get(`${firstSession}:two`)?.()
  await queued
  expect(order).toEqual([
    `start:${firstSession}:one`,
    `start:${secondSession}:parallel`,
    `end:${secondSession}:parallel`,
    `end:${firstSession}:one`,
    `start:${firstSession}:two`,
    `end:${firstSession}:two`,
  ])

  // A rejected prompt does not wedge the per-session queue.
  const failing = new AcpServer({
    promptHandler: ({ text }) => {
      if (text === 'fail') {
        throw new Error('handler exploded')
      }
      return { ok: true }
    },
  })
  const failingSession = String(failing.openSession('/workspace').session_id)
  await expect(failing.prompt(failingSession, 'fail')).rejects.toThrow('handler exploded')
  await expect(failing.prompt(failingSession, 'after')).resolves.toEqual({ ok: true })
})
