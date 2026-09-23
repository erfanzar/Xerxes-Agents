// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { PtySessionManager } from '../src/operators/pty.js'
import { registerPtyTools } from '../src/tools/ptyTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

function registryWithPty(manager: PtySessionManager): ToolRegistry {
  const registry = new ToolRegistry()
  registerPtyTools(registry, manager)
  return registry
}

const OWNER = { metadata: {}, sessionId: 'session-a' }
const OTHER = { metadata: {}, sessionId: 'session-b' }

// registry.execute serializes tool results; unwrap for assertions.
function unwrap<T>(serialized: string): T {
  return JSON.parse(serialized) as T
}

test('pty_open keeps a shell alive and pty_write runs further commands on it', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-pty-tools-'))
  const manager = new PtySessionManager({ workspaceRoot: root })
  const registry = registryWithPty(manager)
  try {
    // One persistent shell; the marker proves state survives between calls.
    const opened = unwrap(await registry.execute(call('pty_open', { command: 'export XERXES_MARK=alive; exec bash' }), OWNER))
    const sessionId = String((opened as { sessionId: string }).sessionId)
    expect(sessionId.startsWith('pty_')).toBe(true)

    const written = unwrap(await registry.execute(
      call('pty_write', { session_id: sessionId, chars: 'echo mark-$XERXES_MARK\n', yield_time_ms: 2_000 }),
      OWNER,
    )) as { stdout: string; running: boolean }
    expect(written.stdout).toContain('mark-alive')
    expect(written.running).toBe(true)

    const listed = unwrap(await registry.execute(call('pty_list', {}), OWNER)) as { sessions: { sessionId: string }[] }
    expect(listed.sessions.map(session => session.sessionId)).toContain(sessionId)

    const closed = unwrap(await registry.execute(call('pty_close', { session_id: sessionId }), OWNER)) as { closed: boolean }
    expect(closed.closed).toBe(true)
    const after = unwrap(await registry.execute(call('pty_list', {}), OWNER)) as { sessions: unknown[] }
    expect(after.sessions).toHaveLength(0)
  } finally {
    await manager.closeAll()
    await rm(root, { force: true, recursive: true })
  }
}, 20_000)

test('pty sessions are owner-scoped: another session cannot see, write, or close them', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-pty-tools-'))
  const manager = new PtySessionManager({ workspaceRoot: root })
  const registry = registryWithPty(manager)
  try {
    const opened = unwrap(await registry.execute(call('pty_open', { command: 'exec sleep 30' }), OWNER))
    const sessionId = String((opened as { sessionId: string }).sessionId)

    const foreignList = unwrap(await registry.execute(call('pty_list', {}), OTHER)) as { sessions: unknown[] }
    expect(foreignList.sessions).toHaveLength(0)
    // Same error for unknown and foreign ids — no existence oracle.
    await expect(registry.execute(call('pty_write', { session_id: sessionId, chars: '' }), OTHER))
      .rejects.toThrow(/PTY session not found/)
    await expect(registry.execute(call('pty_close', { session_id: sessionId }), OTHER))
      .rejects.toThrow(/PTY session not found/)
  } finally {
    await manager.closeAll()
    await rm(root, { force: true, recursive: true })
  }
}, 20_000)

test('disposeOwner closes only that owner\'s sessions at teardown', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-pty-tools-'))
  const manager = new PtySessionManager({ workspaceRoot: root })
  try {
    const a = await manager.createSession('exec sleep 30', { ownerSessionId: 'session-a' })
    const b = await manager.createSession('exec sleep 30', { ownerSessionId: 'session-b' })
    await manager.disposeOwner('session-a')
    expect(manager.listForOwner('session-a')).toHaveLength(0)
    expect(manager.listForOwner('session-b').map(session => session.sessionId)).toEqual([b.sessionId])
    await manager.disposeAll()
    expect(manager.listSessions()).toHaveLength(0)
    void a
  } finally {
    await manager.closeAll()
    await rm(root, { force: true, recursive: true })
  }
}, 20_000)

test('closing an interactive shell is prompt: it hangs up instead of waiting out a SIGTERM the shell ignores', async () => {
  // The desktop Terminal tab opens a bare login shell. Interactive shells
  // ignore SIGTERM, so close() used to sit out its full 2s grace period and
  // then SIGKILL — the tab felt laggy to close.
  const manager = new PtySessionManager({ maxPendingOutputChars: 100_000 })
  try {
    const opened = await manager.createSession('', { yieldTimeMs: 200 })
    let seen = opened.stdout
    for (let tries = 0; tries < 40 && !seen.includes('ready-7'); tries += 1) {
      seen += (await manager.write(opened.sessionId, { chars: tries === 0 ? 'echo ready-$((3+4))\r' : '', yieldTimeMs: 150 })).stdout
    }
    expect(seen).toContain('ready-7')
    const started = performance.now()
    const closed = await manager.close(opened.sessionId)
    expect(closed.closed).toBe(true)
    expect(performance.now() - started).toBeLessThan(1000)
    expect(closed.exitCode).not.toBe(137) // not force-killed
  } finally {
    await manager.closeAll()
  }
})

test('the shell owns its terminal: a resize reaches it and Ctrl-C stops the running command', async () => {
  // Bun's PTY never made the terminal the shell's *controlling* terminal, so
  // the kernel had nowhere to send SIGWINCH/SIGINT: zsh kept its old width
  // (prompt drawn short after the desktop pane grew) and Ctrl-C did nothing.
  if (process.platform === 'win32') return
  const manager = new PtySessionManager({ maxPendingOutputChars: 200_000 })
  const drain = async (sessionId: string, chars: string, until: RegExp, tries = 40): Promise<string> => {
    let seen = ''
    for (let i = 0; i < tries && !until.test(seen); i += 1) {
      seen += (await manager.writeForOwner('owner', sessionId, { chars: i === 0 ? chars : '', yieldTimeMs: 150 })).stdout
    }
    return seen
  }
  try {
    const opened = await manager.createSession('', { ownerSessionId: 'owner', cols: 80, rows: 24, yieldTimeMs: 300 })
    await drain(opened.sessionId, 'echo up-$((1+1))\r', /up-2/)
    manager.resizeForOwner('owner', opened.sessionId, 150, 40)
    await Bun.sleep(300)
    expect(await drain(opened.sessionId, 'echo cols-$COLUMNS\r', /cols-\d+/)).toMatch(/cols-150/)

    await drain(opened.sessionId, 'sleep 30; echo NOT-INTERRUPTED\r', /sleep 30/, 3)
    await Bun.sleep(300)
    const after = await drain(opened.sessionId, '\x03', /\^C/, 5) + await drain(opened.sessionId, 'echo alive-$((2+2))\r', /alive-4\r?\n/)
    expect(after).toMatch(/alive-4\r?\n/)
    expect(after).not.toContain('NOT-INTERRUPTED')
  } finally {
    await manager.closeAll()
  }
})
