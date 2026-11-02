// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { SUBAGENT_HISTORY_PAGE_BYTES, SUBAGENT_HISTORY_TEXT_CHARS } from '../src/daemon/subagentHistory.js'
import { JobStore } from '../src/cron/jobs.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'

type Frame = { id?: number; result?: Record<string, unknown>; error?: unknown }

/** Newline-delimited JSON-RPC over the daemon socket; replies matched by id. */
class Client {
  private buffer = ''
  private readonly waiting = new Map<number, (frame: Frame) => void>()
  private id = 0
  private constructor(private readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', chunk => {
      this.buffer += String(chunk)
      let newline: number
      while ((newline = this.buffer.indexOf('\n')) >= 0) {
        const line = this.buffer.slice(0, newline)
        this.buffer = this.buffer.slice(newline + 1)
        if (!line.trim()) continue
        const frame = JSON.parse(line) as Frame
        if (typeof frame.id === 'number') this.waiting.get(frame.id)?.(frame)
      }
    })
  }
  static async connect(path: string): Promise<Client> {
    const socket = connect({ path })
    await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) })
    return new Client(socket)
  }
  call(method: string, params: Record<string, unknown> = {}): Promise<Frame> {
    const id = ++this.id
    return new Promise(resolve => {
      this.waiting.set(id, resolve)
      this.socket.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
    })
  }
  close(): void { this.socket.destroy() }
}

const CHILD_HISTORY = 'abcdef0123456789'

async function saveChild(store: DaemonTranscriptStore, directory: string, parentSessionId: string, messages: Record<string, unknown>[], owner = 'child'): Promise<void> {
  await store.save({
    agentId: 'reviewer', cwd: directory, extra: {}, format: 'bun-v2', interactionMode: 'code', key: CHILD_HISTORY,
    messages,
    metadata: { session_kind: 'subagent', subagent_id: owner, subagent_handle_id: owner, parent_session_id: parentSessionId, history_session_id: CHILD_HISTORY },
    pendingResumeReplays: [], planMode: false, schemaVersion: undefined, sessionId: CHILD_HISTORY,
    thinkingContent: [], toolExecutions: [{ toolCallId: 'run-1', name: 'exec_command', durationMs: 1200, result: 'secret-duplicate', permitted: true }],
    totalInputTokens: 0, totalOutputTokens: 0, turnCount: 1, updatedAt: '2026-11-02T00:00:00.000Z', workspace: '',
  }, { mode: 'rewrite', expectedGeneration: (await store.load(CHILD_HISTORY))?.generation ?? 0 })
}

test('subagent.inspect pages the child conversation additively, bounded, and only for its own parent', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-agent-history-'))
  const sessions = join(directory, 'sessions')
  const socketPath = join(directory, 'rpc.sock')
  const runtime = new InMemoryDaemonRuntime(undefined, { model: 'fixture', currentProjectDirectory: directory, sessionDirectory: sessions })
  const server = new DaemonServer({ runtime, socketPath, projectDirectory: directory, sessionArchiveDirectory: sessions, cronStoreFactory: () => new JobStore(join(directory, 'cron/jobs.json')) })
  await server.start()
  const client = await Client.connect(socketPath)
  try {
    await client.call('initialize', { session_key: 'parent' })
    const parent = runtime.sessionStatus('parent')!
    const store = new DaemonTranscriptStore({ directory: sessions, currentProjectDirectory: directory })
    const longResult = 'x'.repeat(SUBAGENT_HISTORY_TEXT_CHARS + 500)
    await saveChild(store, directory, parent.id, [
      { role: 'user', content: 'Review the change' },
      { role: 'assistant', content: 'Reading files first.', thinking: 'Plan the review.' },
      { role: 'assistant', content: '', tool_calls: [{ id: 'run-1', type: 'function', function: { name: 'exec_command', arguments: '{"cmd":"bun","args":["test"]}' } }] },
      { role: 'tool', tool_call_id: 'run-1', content: longResult },
      // Still in flight at the checkpoint: no result yet.
      { role: 'assistant', content: '', tool_calls: [{ id: 'run-2', type: 'function', function: { name: 'exec_command', arguments: '{"cmd":"ls"}' } }] },
    ])

    // Old clients (no history_limit) receive exactly the previous response.
    parent.metadata.xerxes_subagent_snapshots_v1 = [{ id: 'child', status: 'running', agent_id: 'reviewer', last_input: 'Review the change', last_output: 'Reading', history_session_id: CHILD_HISTORY }]
    const plain = (await client.call('subagent.inspect', { task: 'child' })).result!
    expect(plain.ok).toBe(true)
    expect(plain).not.toHaveProperty('history')
    expect(plain.agent).toMatchObject({ id: 'child', prompt: 'Review the change', output: 'Reading' })

    const paged = (await client.call('subagent.inspect', { task: 'child', history_limit: 100 })).result!
    expect(paged).toMatchObject({ ok: true, history_session_id: CHILD_HISTORY, agent: { id: 'child', output: 'Reading' } })
    const history = paged.history as { actions: { id: string; messages: Record<string, unknown>[]; executions: Record<string, unknown>[] }[]; has_more: boolean; clipped?: boolean }
    expect(history.has_more).toBe(false)
    expect(history.clipped).toBe(true)
    const messages = history.actions.flatMap(action => action.messages)
    expect(messages.some(message => message.content === 'Reading files first.')).toBe(true)
    const result = messages.find(message => message.role === 'tool' && message.tool_call_id === 'run-1')!
    expect(String(result.content).length).toBeLessThan(longResult.length)
    expect(String(result.content)).toContain('more characters not shown')
    // The in-flight call stays unanswered: no resume sentinel reaches the client.
    expect(JSON.stringify(history)).not.toContain('pending replay')
    expect(messages.some(message => message.role === 'tool' && message.tool_call_id === 'run-2')).toBe(false)
    // Execution records keep timing; the result is carried once, by the tool message.
    expect(JSON.stringify(history)).not.toContain('secret-duplicate')
    expect(history.actions.flatMap(action => action.executions)[0]).toMatchObject({ toolCallId: 'run-1', durationMs: 1200 })

    // Paging with a cursor, as session.history does.
    const first = (await client.call('subagent.inspect', { task: 'child', history_limit: 2 })).result!.history as { actions: { id: string }[]; before: string | null; has_more: boolean }
    expect(first.has_more).toBe(true)
    const older = (await client.call('subagent.inspect', { task: 'child', history_limit: 100, before: first.before })).result!.history as { actions: { id: string }[]; has_more: boolean }
    expect(older.has_more).toBe(false)
    expect([...older.actions, ...first.actions].map(action => action.id)).toEqual(history.actions.map(action => action.id))

    // Byte cap: a page of large replies shrinks until it fits.
    await saveChild(store, directory, parent.id, Array.from({ length: 40 }, (_, index) => ({ role: 'assistant', content: `${index}:${'y'.repeat(30_000)}` })))
    const big = (await client.call('subagent.inspect', { task: 'child', history_limit: 100 })).result!.history as { actions: unknown[]; has_more: boolean }
    expect(big.has_more).toBe(true)
    expect(big.actions.length).toBeLessThan(40)
    expect(Buffer.byteLength(JSON.stringify(big))).toBeLessThanOrEqual(SUBAGENT_HISTORY_PAGE_BYTES)

    // A file that names another child is never returned for this one.
    await saveChild(store, directory, parent.id, [{ role: 'assistant', content: 'someone else' }], 'other-child')
    const foreign = (await client.call('subagent.inspect', { task: 'child', history_limit: 10 })).result!
    expect(foreign).toMatchObject({ ok: true, history: null })
    expect(JSON.stringify(foreign)).not.toContain('someone else')

    // No saved conversation: a note, and the rest of the inspection.
    parent.metadata.xerxes_subagent_snapshots_v1 = [{ id: 'child', status: 'completed', agent_id: 'reviewer', last_output: 'Done' }]
    const missing = (await client.call('subagent.inspect', { task: 'child', history_limit: 10 })).result!
    expect(missing).toMatchObject({ ok: true, history: null, agent: { output: 'Done' } })
    expect(typeof missing.history_note).toBe('string')

    // Invalid limits are rejected, never silently clamped.
    const invalid = await client.call('subagent.inspect', { task: 'child', history_limit: 0 })
    expect(invalid.result?.ok === true).toBe(false)

    // A foreign parent session cannot reach the child at all.
    await client.call('initialize', { session_key: 'foreign' })
    expect((await client.call('subagent.inspect', { task: 'child', history_limit: 10 })).result?.ok).toBe(false)
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }) }
})
