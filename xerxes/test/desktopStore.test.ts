// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, describe, expect, spyOn, test } from 'bun:test'

import { daemonCompatibilityWarning } from '../src/desktop/renderer/buildInfo.js'
import { Store, type XerxesLike } from '../src/desktop/renderer/store.js'
import type { DaemonEvent, SessionRow } from '../src/desktop/renderer/types.js'

/**
 * Store behavior against a scripted fake bridge: no sockets, no React. Each
 * test drives wire events the way the daemon would and asserts what the
 * snapshot folds — steering queue, changes accumulator, plan capture, failed
 * turns, and connection recovery.
 */

type Responder = (method: string, params: Record<string, unknown>) => Record<string, unknown> | Promise<Record<string, unknown>>

class FakeBridge implements XerxesLike {
  readonly calls: Array<{ method: string; params: Record<string, unknown> }> = []
  private handlers = new Set<(event: DaemonEvent) => void>()
  private respond: Responder

  constructor(respond: Responder = () => ({})) {
    this.respond = respond
  }

  /** Swap the responder mid-test (per-test wire shapes over per-bridge ones). */
  respondWith(respond: Responder): void {
    this.respond = respond
  }

  call(method: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    this.calls.push({ method, params })
    return Promise.resolve(this.respond(method, params))
  }

  push(type: string, payload: Record<string, unknown> = {}): void {
    for (const handler of this.handlers) handler({ type, payload })
  }

  // Test-side stand-in for window.xerxes.onEvent.
  subscribe(handler: (event: DaemonEvent) => void): () => void {
    this.handlers.add(handler)
    return () => this.handlers.delete(handler)
  }
}

function withWindow(bridge: FakeBridge): void {
  ;(globalThis as { window?: unknown }).window = {
    xerxes: { onEvent: (handler: (event: DaemonEvent) => void) => bridge.subscribe(handler) },
  }
}

const initializeResult = {
  session_id: 'aa19f402',
  session: { id: 'aa19f402', title: '', cwd: '/repo', plan_mode: false },
  model: 'kimi-for-coding',
  agent_name: 'default',
  cwd: '/repo',
  branch: 'main',
  context_limit: 262_000,
  daemon_protocol: 35,
  daemon_version: '0.4.5',
  daemon_build_id: 'current-build',
}

const editCall = (path: string, oldString: string, newString: string, id = 't1'): DaemonEvent => ({
  type: 'tool_call',
  payload: { id, name: 'FileEditTool', arguments: JSON.stringify({ file_path: path, old_string: oldString, new_string: newString }) },
})

describe('Store workspace folds', () => {
  let bridge: FakeBridge
  let store: Store

  beforeEach(() => {
    bridge = new FakeBridge(method => {
      if (method === 'initialize') return initializeResult
      if (method === 'runtime.status') return { ok: true, permission_mode: 'auto' }
      if (method === 'session.goal') return { ok: true, text: 'No goal is currently set.' }
      if (method === 'session.list' || method === 'session.active_list') return { ok: true, sessions: [] }
      if (method === 'turn.steer') return { ok: true }
      return { ok: true }
    })
    store = new Store()
    withWindow(bridge)
    store.start(bridge)
  })

  test('slow submission remains visible and a preparation warning does not duplicate its echo', async () => {
    await Bun.sleep(0)
    const reply = Promise.withResolvers<Record<string, unknown>>()
    bridge.respondWith(method => method === 'turn.submit' ? reply.promise : { ok: true })
    const sending = store.submit('review this')
    expect(store.getSnapshot().submissionPending).toBe(true)
    await store.submit('second request')
    expect(bridge.calls.filter(call => call.method === 'turn.submit')).toHaveLength(1)
    bridge.push('notification', { severity: 'warning', message: 'Could not snapshot before this work' })
    bridge.push('turn_begin', { text: 'review this' })
    expect(store.getSnapshot().submissionPending).toBe(false)
    expect(store.getSnapshot().blocks.filter(block => block.kind === 'user' && block.text === 'review this')).toHaveLength(1)
    expect(store.getSnapshot().blocks.some(block => block.kind === 'notice' && block.text.includes('Could not snapshot'))).toBe(true)
    reply.resolve({ ok: true }); await sending
    bridge.push('turn_end', {})
    bridge.push('turn_begin', { text: 'review this' })
    expect(store.getSnapshot().blocks.filter(block => block.kind === 'user' && block.text === 'review this')).toHaveLength(2)
  })

  test('missing session previews have bounded concurrency and retries without empty redraws', async () => {
    await Bun.sleep(0)
    const rows: SessionRow[] = Array.from({ length: 60 }, (_, i) => ({ id: `preview-${i}`, key: `preview-${i}`, title: '', status: '', age: '', current: false, kind: 'main', turns: 1, messages: 2, cwd: '/repo', untitled: true }))
    const pending: Array<ReturnType<typeof Promise.withResolvers<Record<string, unknown>>>> = []
    bridge.respondWith(method => {
      if (method !== 'session.status') return { ok: true }
      const reply = Promise.withResolvers<Record<string, unknown>>()
      pending.push(reply)
      return reply.promise
    })
    const loader = store as unknown as { enrichUntitled(rows: readonly SessionRow[]): void }
    let redraws = 0
    const unsub = store.subscribe(() => redraws++)
    loader.enrichUntitled(rows)
    loader.enrichUntitled(rows)
    expect(pending).toHaveLength(4)
    pending[0]!.reject(new Error('Unavailable'))
    for (const reply of pending.slice(1)) reply.resolve({ ok: false })
    await Bun.sleep(0)
    expect(redraws).toBe(0)
    loader.enrichUntitled(rows.slice(0, 4))
    expect(pending).toHaveLength(4)
    const clock = spyOn(Date, 'now').mockReturnValue(Date.now() + 60_001)
    try {
      loader.enrichUntitled(rows.slice(0, 4))
      expect(pending).toHaveLength(8)
      for (const reply of pending.slice(4)) reply.resolve({ session: { preview: 'Useful task preview' } })
      await Bun.sleep(0)
      expect(store.getSnapshot().snippets['preview-0']).toBe('Useful task preview')
      expect(redraws).toBe(4)
      loader.enrichUntitled(rows.slice(0, 4))
      expect(pending).toHaveLength(8)
    } finally { clock.mockRestore(); unsub() }
  })

  test('rejected preparation clears sending and rolls back only its optimistic message', async () => {
    await Bun.sleep(0)
    const reply = Promise.withResolvers<Record<string, unknown>>()
    bridge.respondWith(method => method === 'turn.submit' ? reply.promise : { ok: true })
    const sending = store.submit('not admitted')
    bridge.push('notification', { severity: 'warning', message: 'Preparation warning' })
    reply.reject(new Error('Provider configuration missing')); await sending
    expect(store.getSnapshot().submissionPending).toBe(false)
    expect(store.getSnapshot().blocks.some(block => block.kind === 'user' && block.text === 'not admitted')).toBe(false)
    expect(store.getSnapshot().error).toContain('Provider configuration missing')
  })

  test('RPC acceptance keeps sending visible until the daemon begins or ends preparation', async () => {
    await Bun.sleep(0)
    await store.submit('accepted before preparation')
    expect(store.getSnapshot().submissionPending).toBe(true)
    bridge.push('notification', { severity: 'warning', message: 'Snapshot unavailable' })
    expect(store.getSnapshot().submissionPending).toBe(true)
    bridge.push('turn_end', {})
    expect(store.getSnapshot().submissionPending).toBe(false)
  })

  test.each(['turn.submit', 'slash', 'set_plan_mode', 'turn.steer'])('composer admission reports a refused %s instead of clearing its draft', async (method) => {
    await Bun.sleep(0)
    bridge.respondWith(name => name === method ? { ok: false, error: 'Audit refusal' } : { ok: true })
    const input = method === 'slash' ? '/missing argument' : method === 'set_plan_mode' ? '/plan' : 'Keep this\n  draft'
    if (method === 'turn.steer') bridge.push('turn_begin', { text: 'existing task' })
    expect(await store.submit(input)).toBe(false)
    expect(store.getSnapshot().error || store.getSnapshot().blocks.map(block => block.kind === 'notice' ? block.text : '').join(' ')).toContain('Audit refusal')
  })

  test.each([false, true])('slash rejection renders one notice when the event accompanies the RPC result (throws=%s)', async (throws) => {
    await Bun.sleep(0)
    bridge.respondWith(method => {
      if (method !== 'slash') return { ok: true }
      bridge.push('notification', { category: 'slash', type: 'result', severity: 'warning', body: 'Unknown command: /absent (type /help).' })
      if (throws) throw new Error('Unknown slash command: /absent')
      return { ok: false, error: 'Unknown slash command: /absent' }
    })
    expect(await store.submit('/absent argument')).toBe(false)
    expect(store.getSnapshot().blocks.filter(block => block.kind === 'notice')).toHaveLength(1)
  })

  test('slash success omits redundant ok but preserves a distinct rejection', async () => {
    await Bun.sleep(0)
    bridge.respondWith(method => {
      if (method !== 'slash') return { ok: true }
      bridge.push('notification', { category: 'slash', type: 'result', severity: 'info', body: 'Configuration loaded' })
      return { ok: true }
    })
    expect(await store.submit('/settings inspect')).toBe(true)
    expect(store.getSnapshot().blocks.filter(block => block.kind === 'notice')).toHaveLength(1)
    bridge.respondWith(method => {
      if (method !== 'slash') return { ok: true }
      bridge.push('notification', { category: 'slash', type: 'result', severity: 'warning', body: 'Configuration loaded with warnings' })
      return { ok: false, error: 'Authentication required' }
    })
    expect(await store.submit('/settings inspect')).toBe(false)
    expect(store.getSnapshot().blocks.some(block => block.kind === 'notice' && block.text === 'Authentication required')).toBe(true)
  })

  test('late command rejection cannot add an error to another session', async () => {
    await Bun.sleep(0)
    const reply = Promise.withResolvers<Record<string, unknown>>()
    bridge.respondWith(method => method === 'slash' ? reply.promise : method === 'initialize'
      ? { ...initializeResult, session_id: 'command-other', session: { id: 'command-other', key: 'command-other', cwd: '/repo' }, messages: [] }
      : { ok: true, sessions: [] })
    const command = store.submit('/settings inspect')
    await store.openSession('command-other')
    const before = store.getSnapshot()
    reply.reject(new Error('Old command failed'))
    expect(await command).toBe(false)
    expect(store.getSnapshot().blocks).toEqual(before.blocks)
    expect(store.getSnapshot().error).toBe(before.error)
  })

  test('a rejected submission from another session cannot alter the current draft transcript or connection', async () => {
    await Bun.sleep(0)
    const reply = Promise.withResolvers<Record<string, unknown>>()
    bridge.respondWith((method, params) => method === 'turn.submit' ? reply.promise : method === 'initialize'
      ? { ...initializeResult, session_id: 'other-session', session: { id: 'other-session', key: 'other-session', cwd: '/repo' }, messages: [] }
      : { ok: true, sessions: [] })
    const sending = store.submit('old session request')
    await store.openSession('other-session')
    expect(store.getSnapshot().submissionPending).toBe(false)
    const before = store.getSnapshot()
    reply.reject(new Error('Old connection closed')); await sending
    expect(store.getSnapshot().currentId).toBe('other-session')
    expect(store.getSnapshot().blocks).toEqual(before.blocks)
    expect(store.getSnapshot().connection).toBe(before.connection)
    expect(store.getSnapshot().error).toBe(before.error)
  })

  test('workspace errors stay separate from the task and overlapping host requests are suppressed', async () => {
    await new Promise(resolve => setTimeout(resolve, 0))
    const before = store.getSnapshot()
    let reject!: (error: Error) => void
    let calls = 0
    Object.assign(bridge, { useWorkspace: async () => {
      calls++
      return new Promise((_resolve, fail) => { reject = fail })
    } })
    const pending = store.enterWorkspace('/inaccessible')
    expect(store.getSnapshot().workspaceBusy).toBe(true)
    await store.enterWorkspace('/another')
    expect(calls).toBe(1)
    reject(new Error('Folder access denied'))
    await pending
    expect(store.getSnapshot().workspaceError).toBe('Folder access denied')
    expect(store.getSnapshot().workspaceBusy).toBe(false)
    expect(store.getSnapshot().failed).toBe(before.failed)
    expect(store.getSnapshot().sessionKey).toBe(before.sessionKey)
    expect(store.getSnapshot().cwd).toBe(before.cwd)
    Object.assign(bridge, { chooseWorkspace: async () => null })
    await store.chooseWorkspace()
    expect(store.getSnapshot().workspaceError).toBeNull()
    expect(store.getSnapshot().cwd).toBe(before.cwd)
    Object.assign(bridge, { useWorkspace: async () => true })
    await store.enterWorkspace('/available')
    expect(store.getSnapshot().workspaceError).toBeNull()
  })

  test('workspace host capability failures are visible and dismissible', async () => {
    await store.chooseWorkspace()
    expect(store.getSnapshot().workspaceError).toContain('cannot open')
    store.clearWorkspaceError()
    expect(store.getSnapshot().workspaceError).toBeNull()
    await store.enterWorkspace('/available')
    expect(store.getSnapshot().workspaceError).toContain('cannot switch')
  })

  test('terminal control awaits acceptance, preserves line input and propagates rejection', async () => {
    let answer!: (result: Record<string, unknown>) => void
    bridge.respondWith(() => new Promise(resolve => { answer = resolve }))
    let settled = false
    const pending = store.controlTerminal('shell-1', 'write', '  bun test\n').finally(() => { settled = true })
    await Promise.resolve()
    expect(settled).toBe(false)
    expect(bridge.calls.at(-1)?.params.chars).toBe('  bun test\n')
    answer({ ok: false, error: 'Terminal input denied' })
    await expect(pending).rejects.toThrow('Terminal input denied')
    expect(store.getSnapshot().failed).toBeNull()
  })

  test('terminal list failure retains known rows and reports an inline error', async () => {
    bridge.respondWith(() => ({ ok: true, terminals: [{ id: 'shell-1', running: true }] }))
    store.loadTerminals()
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(store.getSnapshot().terminals).toHaveLength(1)
    bridge.respondWith(() => ({ ok: false, error: 'Terminal registry unavailable' }))
    store.loadTerminals()
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(store.getSnapshot().terminals).toHaveLength(1)
    expect(store.getSnapshot().terminalsError).toBe('Terminal registry unavailable')
    expect(store.getSnapshot().terminalsLoading).toBe(false)
    expect(store.getSnapshot().failed).toBeNull()
  })

  test('late terminal refresh cannot replace a newer exited state', async () => {
    const answers: Array<(result: Record<string, unknown>) => void> = []
    bridge.respondWith(() => new Promise(resolve => { answers.push(resolve) }))
    store.loadTerminals()
    store.loadTerminals()
    answers[1]!({ ok: true, terminals: [{ id: 'shell-1', running: false, exitCode: 130 }] })
    await new Promise(resolve => setTimeout(resolve, 0))
    answers[0]!({ ok: true, terminals: [{ id: 'shell-1', running: true }] })
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(store.getSnapshot().terminals[0]).toMatchObject({ running: false, exitCode: 130 })
    expect(store.getSnapshot().terminalsLoading).toBe(false)
  })

  test('channel mutation stays pending until the daemon answers and reports rejection', async () => {
    let answer!: (result: Record<string, unknown>) => void
    bridge.respondWith(method => method === 'channel.enable'
      ? new Promise(resolve => { answer = resolve }) : { ok: true })
    let finished = false
    const pending = store.setChannelEnabled('test-gateway', true).finally(() => { finished = true })
    await Promise.resolve()
    expect(finished).toBe(false)
    expect(bridge.calls.at(-1)).toEqual({ method: 'channel.enable', params: { name: 'test-gateway' } })
    answer({ ok: false, error: 'Gateway credentials unavailable' })
    await expect(pending).rejects.toThrow('Gateway credentials unavailable')
    expect(finished).toBe(true)
    expect(store.getSnapshot().channels).toEqual([])
  })

  test('MCP status and reload reject typed errors without replacing known server state', async () => {
    bridge.respondWith(() => ({ session: { mcp_status: { local: { connected: true, tools: 3 } } } }))
    await store.refreshMcpStatus()
    expect(store.getSnapshot().mcpStatus.local?.tools).toBe(3)
    bridge.respondWith(() => ({ ok: false, error: 'MCP configuration is unreadable' }))
    await expect(store.refreshMcpStatus()).rejects.toThrow('MCP configuration is unreadable')
    await expect(store.reloadMcp()).rejects.toThrow('MCP reload failed and current server status could not be refreshed.')
    expect(store.getSnapshot().mcpStatus.local?.connected).toBe(true)
    bridge.respondWith(() => ({ session: {} }))
    await expect(store.refreshMcpStatus()).rejects.toThrow('MCP status is unavailable')
    bridge.respondWith(() => ({ session: { mcp_status: {} } }))
    await store.refreshMcpStatus()
    expect(store.getSnapshot().mcpStatus).toEqual({})
  })

  test('failed MCP reload refreshes disconnected status while preserving the failure', async () => {
    bridge.respondWith(() => ({ session: { mcp_status: { local: { connected: true, tools: 1 } } } }))
    await store.refreshMcpStatus()
    bridge.respondWith(method => method === 'slash'
      ? { ok: false, error: 'Server failed to reconnect' }
      : { session: { mcp_status: { local: { connected: false, tools: 0, lastError: 'Process exited' } } } })
    await expect(store.reloadMcp()).rejects.toThrow('Server failed to reconnect')
    expect(store.getSnapshot().mcpStatus.local).toMatchObject({ connected: false, tools: 0, lastError: 'Process exited' })
  })

  test("slash search combines command and skill results", async () => {
    bridge.respondWith((method, params) => method === "complete" ? {completions: params.text === "/deep" ? [{value:"/deep-command",category:"session"}] : [{value:"/skill deep-research",label:"deep-research"}]} : {ok:true})
    const items = await store.completeText("/deep")
    expect(items.map(item => item.value)).toEqual(["/deep-command", "/skill deep-research"])
    expect(items.map(item => item.kind)).toEqual(["command", "skill"])
    expect(bridge.calls.filter(call => call.method === "complete").map(call => call.params.text)).toEqual(["/deep", "/skill deep"])
  })

  test('provider loading exposes errors and a successful retry clears them', async () => {
    bridge.respondWith(method => method === 'provider_list' ? { ok: false, error: 'Profile file is unreadable' } : { ok: true })
    await store.loadProviders()
    expect(store.getSnapshot().providerError).toBe('Profile file is unreadable')
    bridge.respondWith(method => method === 'provider_list' ? { ok: true, profiles: [{ name: 'local', provider: 'openai', model: 'configured-model', active: true }] } : { ok: true })
    await store.loadProviders()
    expect(store.getSnapshot().providerError).toBe('')
    expect(store.getSnapshot().providers[0]?.name).toBe('local')
  })

  test('manual reconnect resumes the current session and keeps the failure actionable', async () => {
    await new Promise(resolve => setTimeout(resolve, 20))
    const id = store.getSnapshot().currentId
    bridge.respondWith(async method => { if (method === 'initialize') throw new Error('Authentication failed: configure provider credentials'); return {ok:true} })
    store.retryConnection()
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(bridge.calls.filter(call => call.method === 'initialize').at(-1)?.params.resume_session_id).toBe(id)
    expect(store.getSnapshot().connection).toBe('offline')
    expect(store.getSnapshot().error).toContain('Authentication failed')
    expect(store.getSnapshot().currentId).toBe(id)
  })

  test('heartbeat preserves initialization errors until the session reconnects', async () => {
    await new Promise(resolve => setTimeout(resolve, 20))
    const id = store.getSnapshot().currentId
    const heartbeat = store as unknown as { beat(): Promise<void>; wentOffline(error: unknown): void }
    heartbeat.wentOffline(new Error('Socket closed'))
    bridge.respondWith(async method => {
      if (method === 'initialize') throw new Error('Authentication failed: configure provider credentials')
      return { ok: true }
    })
    await heartbeat.beat()
    expect(store.getSnapshot().connection).toBe('offline')
    expect(store.getSnapshot().error).toContain('Authentication failed')
    expect(store.getSnapshot().currentId).toBe(id)
    expect(bridge.calls.filter(call => call.method === 'initialize').at(-1)?.params.resume_session_id).toBe(id)
  })

  afterEach(() => {
    delete (globalThis as { window?: unknown }).window
  })

  test('opening global history switches project and carries the resume id without calling the old daemon', async () => {
    const switches: unknown[][] = []
    ;(bridge as XerxesLike).useWorkspace = async (...args) => { switches.push(args) }
    bridge.respondWith(method => {
      if (method === 'initialize') return initializeResult
      if (method === 'session.list') return { ok: true, sessions: [{ id: 'other-session', title: 'Other project', cwd: '/other-project', message_count: 4 }] }
      if (method === 'session.active_list') return { ok: true, sessions: [] }
      return { ok: true }
    })
    await new Promise(resolve => setTimeout(resolve, 10))
    await store.renameSession('aa19f402', 'Current project')
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.getSnapshot().sessions.some(row => row.id === 'other-session')).toBe(true)
    const callsBefore = bridge.calls.filter(call => call.method === 'initialize').length
    await store.openSession('other-session')
    expect(switches).toEqual([['/other-project', 'other-session']])
    expect(bridge.calls.filter(call => call.method === 'initialize')).toHaveLength(callsBefore)
    expect(store.getSnapshot().currentId).toBe('aa19f402')
    ;(bridge as XerxesLike).useWorkspace = async () => { throw new Error('Folder is unavailable') }
    await store.openSession('other-session')
    expect(store.getSnapshot().error).toBe('Folder is unavailable')
    expect(store.getSnapshot().currentId).toBe('aa19f402')
  })

  test('explicit resume signals a fresh tail position even for the same session, but failed resume does not', async () => {
    const revision = store.getSnapshot().sessionOpenRevision
    await store.openSession('aa19f402')
    expect(store.getSnapshot().sessionOpenRevision).toBe(revision + 1)
    await store.openSession('aa19f402')
    expect(store.getSnapshot().sessionOpenRevision).toBe(revision + 2)
    bridge.respondWith(method => { if (method === 'initialize') throw new Error('Resume unavailable'); return { ok: true } })
    await store.openSession('aa19f402')
    expect(store.getSnapshot().sessionOpenRevision).toBe(revision + 2)
  })

  test('runtime update reports waiting and errors and resumes the original session', async () => {
    await new Promise(resolve => setTimeout(resolve, 10))
    let outcome: Record<string, unknown> = { ok: false, busy: true }
    bridge.respondWith(method => method === "initialize" ? initializeResult : method === "desktop.restartRuntime" ? outcome : { ok: true })
    await store.restartDaemon()
    expect(store.getSnapshot().runtimeUpdate).toBe("waiting")
    expect(store.getSnapshot().currentId).toBe("aa19f402")
    outcome = { ok: false, error: "Update unavailable" }
    await store.restartDaemon()
    expect(store.getSnapshot().runtimeUpdate).toBe("failed")
    expect(store.getSnapshot().runtimeUpdateMessage).toBe("Update unavailable")
    outcome = { ok: true }
    await store.restartDaemon()
    expect(store.getSnapshot().runtimeUpdate).toBeUndefined()
    expect(bridge.calls.filter(call => call.method === "initialize").at(-1)?.params.resume_session_id).toBe("aa19f402")
    expect(bridge.calls.some(call => call.method === "slash" && call.params.command === "/restart")).toBe(false)
  })

  test('a build mismatch automatically requests a safe idle update and stops retrying errors', async () => {
    await new Promise(resolve => setTimeout(resolve, 10))
    let attempts = 0
    bridge.respondWith(method => {
      if (method === "initialize") return { ...initializeResult, daemon_protocol: 34 }
      if (method === "desktop.restartRuntime") { attempts += 1; return { ok: false, error: "Older runtime" } }
      return { ok: true }
    })
    await store.openSession("aa19f402")
    await (store as unknown as { beat(): Promise<void> }).beat()
    expect(attempts).toBe(1)
    expect(store.getSnapshot().daemonWarning).toBeTruthy()
    await (store as unknown as { beat(): Promise<void> }).beat()
    expect(attempts).toBe(1)
    expect(store.getSnapshot().runtimeUpdate).toBe("failed")
  })

  test('automatic update retries busy work then resumes the same session after idle', async () => {
    await new Promise(resolve => setTimeout(resolve, 10))
    let updated = false
    let busy = true
    bridge.respondWith(method => {
      if (method === 'initialize') return { ...initializeResult, daemon_protocol: updated ? 35 : 34 }
      if (method === 'desktop.restartRuntime') {
        if (busy) return { ok: false, busy: true }
        updated = true
        return { ok: true }
      }
      return { ok: true }
    })
    await store.openSession('aa19f402')
    const beat = () => (store as unknown as { beat(): Promise<void> }).beat()
    await beat()
    expect(store.getSnapshot().runtimeUpdate).toBe('waiting')
    expect(updated).toBe(false)
    busy = false
    await beat()
    expect(store.getSnapshot().runtimeUpdate).toBeUndefined()
    expect(store.getSnapshot().daemonWarning).toBeNull()
    expect(store.getSnapshot().currentId).toBe('aa19f402')
    expect(bridge.calls.filter(call => call.method === 'desktop.restartRuntime').every(call => call.params.allow_legacy === false)).toBe(true)
  })

  test('starts online after initialize and sends the desktop handshake', async () => {
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.getSnapshot().connection).toBe('online')
    expect(store.getSnapshot().model).toBe('kimi-for-coding')
    const initialize = bridge.calls.find(call => call.method === 'initialize')
    expect(initialize?.params).toMatchObject({ client_protocol: 35, client_version: '0.4.5' })
    expect(store.getSnapshot().daemonWarning).toBeNull()
  })

  test('the handshake catches an older or same-version stale daemon build', () => {
    const desktop = { version: '0.3.0', protocol: 35, expectedDaemonBuildId: 'new-build' }
    expect(daemonCompatibilityWarning({ daemon_protocol: 34, daemon_version: '0.3.0' }, desktop)).toBe(
      'Daemon is older than the app — restart it.',
    )
    expect(daemonCompatibilityWarning({
      daemon_protocol: 35,
      daemon_version: '0.3.0',
      daemon_build_id: 'old-build',
    }, desktop)).toBe('App and daemon builds differ — restart the daemon, and quit and relaunch the app.')
    expect(daemonCompatibilityWarning({
      daemon_protocol: 35,
      daemon_version: '0.3.0',
      daemon_build_id: 'new-build',
    }, desktop)).toBeNull()
    expect(daemonCompatibilityWarning({ daemon_protocol: 35, daemon_version: '0.3.0',
      daemon_build_id: 'remote-release', desktop_expected_daemon_build_id: 'remote-release',
    }, desktop)).toBeNull()
    expect(daemonCompatibilityWarning({ daemon_protocol: 36, daemon_version: '0.3.0',
      daemon_build_id: 'remote-release', desktop_expected_daemon_build_id: 'remote-release',
    }, desktop)).toContain('app is older')
  })

  test('steering while acting queues visibly and clears on turn end', async () => {
    bridge.push('turn_begin', { user_input: 'map the repo' })
    await store.submit('also cover the replay path')
    const queued = store.getSnapshot().queue
    expect(queued).toHaveLength(1)
    expect(queued[0]!.text).toBe('also cover the replay path')
    const steer = bridge.calls.find(call => call.method === 'turn.steer')
    expect(steer?.params.content).toBe('also cover the replay path')

    // The acceptance echo must NOT drop the mirror — it means "queued
    // daemon-side", not "consumed". Only turn_end clears it.
    bridge.push('steer_input', { content: 'also cover the replay path' })
    expect(store.getSnapshot().queue).toHaveLength(1)

    bridge.push('turn_end', {})
    expect(store.getSnapshot().queue).toHaveLength(0)
  })

  test('the session menu renames through session.title with the row key', async () => {
    store.openSessionMenu({ id: 'aa19f402', key: 'aa19f402', title: 'Old title' }, 40, 60)
    expect(store.getSnapshot().sessionMenu).toMatchObject({ id: 'aa19f402', title: 'Old title' })
    await store.renameSession('aa19f402', '  Ship cancel-safe loop  ')
    const call = bridge.calls.find(c => c.method === 'session.title')
    expect(call).toMatchObject({ params: { session_key: 'aa19f402', title: 'Ship cancel-safe loop' } })
    expect(store.getSnapshot().sessionMenu).toBeNull()
  })

  test('an empty rename clears the menu without touching the wire', async () => {
    store.openSessionMenu({ id: 'aa19f402', key: 'aa19f402', title: 'Old title' }, 40, 60)
    await store.renameSession('aa19f402', '   ')
    expect(bridge.calls.some(c => c.method === 'session.title')).toBe(false)
    expect(store.getSnapshot().sessionMenu).toBeNull()
  })

  test('rename rejection retains the attempted title and a retry clears the error', async () => {
    bridge.respondWith(method => method === 'session.title' ? { ok: false, error: 'Session is read-only' } : { ok: true, sessions: [] })
    store.openSessionMenu({ id: 'aa19f402', key: 'aa19f402', title: 'Old title' }, 40, 60)
    await store.renameSession('aa19f402', 'New title')
    expect(store.getSnapshot().sessionMenu).toMatchObject({ title: 'New title', renaming: true, pending: false, error: 'Session is read-only' })
    bridge.respondWith(() => ({ ok: true, sessions: [] }))
    await store.renameSession('aa19f402', 'New title')
    expect(store.getSnapshot().sessionMenu).toBeNull()
  })

  test('a rejected transcript read never produces an empty successful download', async () => {
    const host = globalThis as { document?: unknown }
    const previous = host.document
    let downloads = 0
    host.document = { createElement: () => { downloads += 1; throw new Error('Unexpected download') } }
    bridge.respondWith(method => method === 'session.status' ? { ok: false, error: 'Transcript access denied' } : { ok: true, sessions: [] })
    try {
      await expect(store.downloadSessionTranscript('aa19f402')).rejects.toThrow('Transcript access denied')
      expect(downloads).toBe(0)
      await store.exportSessionTranscript('aa19f402')
      expect(downloads).toBe(0)
      expect(store.getSnapshot().error).toContain('Transcript access denied')
    } finally {
      if (previous === undefined) delete host.document
      else host.document = previous
    }
  })

  test('dismissing a pending rename does not reopen its menu on failure', async () => {
    let finish: ((value: Record<string, unknown>) => void) | undefined
    bridge.respondWith(method => method === 'session.title' ? new Promise(resolve => { finish = resolve }) : { ok: true, sessions: [] })
    store.openSessionMenu({ id: 'aa19f402', key: 'aa19f402', title: 'Old title' }, 40, 60)
    const saving = store.renameSession('aa19f402', 'New title')
    expect(store.getSnapshot().sessionMenu?.pending).toBe(true)
    store.closeSessionMenu()
    finish?.({ ok: false, error: 'Disconnected' })
    await saving
    expect(store.getSnapshot().sessionMenu).toBeNull()
  })

  // ── New-task modal (mockup 18) ────────────────────────────────────────

  test('task startup waits for the selected model and does not submit on rejection', async () => {
    await new Promise(resolve => setTimeout(resolve, 10))
    let finishModel: ((value: Record<string, unknown>) => void) | undefined
    const previousCall = bridge.call.bind(bridge)
    bridge.call = (method, params = {}) => {
      if (method === 'slash' && String(params.command).startsWith('/model ')) {
        bridge.calls.push({ method, params })
        return new Promise(resolve => { finishModel = resolve })
      }
      return previousCall(method, params)
    }
    store.openTaskModal()
    const start = store.startTask('Use the chosen model', false, 'default', 'gpt-5.6-sol')
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(bridge.calls.some(call => call.method === 'turn.submit')).toBe(false)
    expect(bridge.calls.find(call => call.method === 'slash')).toMatchObject({ params: { command: '/model gpt-5.6-sol', session_key: store.getSnapshot().sessionKey } })
    finishModel?.({ ok: true, model: 'gpt-5.6-sol' })
    await start
    expect(store.getSnapshot().model).toBe('gpt-5.6-sol')
    expect(bridge.calls.filter(call => call.method === 'turn.submit')).toHaveLength(1)
    store.openTaskModal()
    const rejected = store.startTask('Do not send', false, 'default', 'missing-model')
    await new Promise(resolve => setTimeout(resolve, 10))
    finishModel?.({ ok: false, error: 'Model unavailable' })
    await rejected
    expect(bridge.calls.filter(call => call.method === 'turn.submit')).toHaveLength(1)
    expect(store.getSnapshot()).toMatchObject({ taskModalOpen: true, error: 'Model unavailable' })
  })

  test('⌘N opens the task modal; startTask re-keys, arms the ceiling, submits in order', async () => {
    const before = store.getSnapshot().sessionKey
    expect(store.getSnapshot().taskModalOpen).toBe(false)
    store.openTaskModal()
    expect(store.getSnapshot().taskModalOpen).toBe(true)

    await store.startTask('  Add rate limiting  ', true)
    expect(store.getSnapshot().taskModalOpen).toBe(false)
    // A fresh session is bound before the objective rides it.
    expect(store.getSnapshot().sessionKey).not.toBe(before)
    const init = bridge.calls.find(c => c.method === 'initialize')
    const plan = bridge.calls.find(c => c.method === 'set_plan_mode')
    const submit = bridge.calls.find(c => c.method === 'turn.submit')
    expect(init).toBeDefined()
    expect(plan).toMatchObject({ params: { enabled: true } })
    expect(submit).toMatchObject({ params: { text: 'Add rate limiting' } })
    const order = bridge.calls.map(c => c.method)
    expect(order.indexOf('initialize')).toBeLessThan(order.indexOf('set_plan_mode'))
    expect(order.indexOf('set_plan_mode')).toBeLessThan(order.indexOf('turn.submit'))
  })

  test('startTask without the plan switch leaves the daemon ceiling alone', async () => {
    store.openTaskModal()
    await store.startTask('ship it', false)
    expect(bridge.calls.some(c => c.method === 'set_plan_mode')).toBe(false)
    expect(bridge.calls.some(c => c.method === 'turn.submit')).toBe(true)
  })

  test('startTask with an empty objective just opens a fresh session', async () => {
    const before = store.getSnapshot().sessionKey
    store.openTaskModal()
    await store.startTask('   ', true)
    expect(store.getSnapshot().sessionKey).not.toBe(before)
    expect(bridge.calls.some(c => c.method === 'turn.submit')).toBe(false)
    expect(store.getSnapshot().taskModalOpen).toBe(false)
  })

  test('preset folder opening reports a refused host response and permits retry', async () => {
    await new Promise(resolve => setTimeout(resolve, 0))
    bridge.respondWith(() => ({ ok: true, path: '/fixture/presets/reviewer' }))
    const opened: string[] = []
    Object.assign(window.xerxes, { openPath: async (path: string) => { opened.push(path); return false } })
    await store.openAgentPresetLocation('reviewer')
    expect(opened).toEqual(['/fixture/presets/reviewer'])
    expect(store.getSnapshot().agentPresetsError).toContain('file manager could not open')
    expect(store.getSnapshot().failed).toBeNull()
    Object.assign(window.xerxes, { openPath: async () => true })
    await store.openAgentPresetLocation('reviewer')
    expect(store.getSnapshot().agentPresetsError).toBeNull()
  })

  test('preset mutations report local errors without replacing the task failure state', async () => {
    await new Promise(resolve => setTimeout(resolve, 0))
    bridge.respondWith(() => ({ ok: false, error: 'Preset validation rejected the draft' }))
    expect(await store.copyAgentPreset('default', 'review-copy')).toBe(false)
    expect(store.getSnapshot().agentPresetsError).toBe('Preset validation rejected the draft')
    expect(await store.writeAgentPreset('review-copy', 'invalid: [')).toBe(false)
    expect(store.getSnapshot().failed).toBeNull()
    const selected = store.getSnapshot().currentAgentPreset
    expect(await store.selectAgentPreset('review-copy')).toBe(false)
    expect(store.getSnapshot().currentAgentPreset).toBe(selected)
    bridge.respondWith(method => method === 'agentPreset.list' ? { ok: true, presets: [] } : { ok: true })
    expect(await store.writeAgentPreset('review-copy', 'name: reviewer')).toBe(true)
    expect(store.getSnapshot().agentPresetsError).toBeNull()
    expect(store.getSnapshot().failed).toBeNull()
  })

  test('new tasks stage an agent preset and the roster mirrors daemon management RPCs', async () => {
    bridge.respondWith((method) => {
      if (method === 'initialize') return { ...initializeResult, agent_name: 'creator', session: { ...initializeResult.session, agent_id: 'creator' } }
      if (method === 'agentPreset.list') return {
        ok: true,
        presets: [
          { id: 'default', name: 'Default', description: 'Coding', trust: 'system', is_default: true, manageable: false },
          { id: 'creator', name: 'Creator', description: 'Authors presets', trust: 'system', is_default: false, manageable: false },
        ],
      }
      return { ok: true }
    })
    await store.loadAgentPresets()
    expect(store.getSnapshot().agentPresets.map(row => row.id)).toEqual(['default', 'creator'])
    await store.startTask('', false, 'creator')
    expect(bridge.calls.filter(call => call.method === 'initialize').at(-1)?.params).toMatchObject({ agent_id: 'creator' })
    expect(store.getSnapshot().currentAgentPreset).toBe('creator')
    bridge.push('agent_preset_selected', { agent_preset: 'default' })
    expect(store.getSnapshot().currentAgentPreset).toBe('default')
  })

  test('the direct Creator mode launch opens a blank creator session', async () => {
    store.startCreatorMode()
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(bridge.calls.filter(call => call.method === 'initialize').at(-1)?.params).toMatchObject({ agent_id: 'creator' })
    expect(bridge.calls.some(call => call.method === 'turn.submit')).toBe(false)
  })

  test('the task modal refuses to open while a turn runs', async () => {
    bridge.push('turn_begin', { user_input: 'busy' })
    expect(store.getSnapshot().turnActive).toBe(true)
    store.openTaskModal()
    expect(store.getSnapshot().taskModalOpen).toBe(false)
  })

  test('skill telemetry restores from RPC and adopts live structured suggestions', async () => {
    bridge.respondWith(method => method === 'skill_suggestions'
      ? {
          ok: true,
          suggestions: [{
            skill_name: 'release-checklist',
            description: 'Repeat the verified release sequence.',
            version: '0.1.0',
            source_path: '/tmp/release/SKILL.md',
            tool_count: 4,
            unique_tools: ['Read', 'Bash'],
          }],
        }
      : method === 'initialize' ? initializeResult : { ok: true })
    store.loadSkillSuggestions()
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.getSnapshot().skillSuggestions[0]).toMatchObject({
      skillName: 'release-checklist',
      toolCount: 4,
      uniqueTools: ['Read', 'Bash'],
    })

    bridge.push('notification', {
      level: 'info',
      message: 'Skill suggestion: test-loop',
      skill: {
        skillName: 'test-loop',
        description: 'Reuse the test loop.',
        version: '0.1.0',
        sourcePath: '/tmp/test/SKILL.md',
        toolCount: 2,
        uniqueTools: ['Bash'],
      },
    })
    expect(store.getSnapshot().skillSuggestions.map(row => row.skillName)).toEqual([
      'release-checklist',
      'test-loop',
    ])
  })

  test('creator mode restores its declarative policy trace from the daemon', async () => {
    bridge.respondWith(method => method === 'creator_trace'
      ? {
          ok: true,
          trace: [{
            action: 'define',
            name: 'briefing',
            version: '0.1.0',
            status: 'ok',
            detail: '',
            at: '2026-03-24T10:00:00.000Z',
          }],
        }
      : method === 'initialize' ? initializeResult : { ok: true })
    store.loadCreatorTrace()
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.getSnapshot().creatorTrace).toEqual([{
      action: 'define',
      name: 'briefing',
      version: '0.1.0',
      status: 'ok',
      detail: '',
      at: '2026-03-24T10:00:00.000Z',
    }])
  })

  test('the status echo carries cost and provider telemetry into the snapshot', () => {
    expect(store.getSnapshot().costUsd).toBeNull()
    bridge.push('status_update', {
      cost_usd: 0.41,
      context_tokens: 12_000,
      max_context: 262_144,
      llm_duration_ms: 2_100,
      ttft_ms: 640,
      tokens_per_second: 37.25,
      cache_hit_rate: 0.75,
      total_input_tokens: 8_000,
      total_output_tokens: 400,
    })
    expect(store.getSnapshot()).toMatchObject({
      costUsd: 0.41,
      contextTokens: 12_000,
      contextMax: 262_144,
      llmDurationMs: 2_100,
      llmSteps: 1,
      ttftMs: 640,
      tokensPerSecond: 37.25,
      cacheHitRate: 0.75,
      inputTokens: 8_000,
      outputTokens: 400,
    })
    bridge.push('status_update', {
      llm_duration_ms: 900,
      ttft_ms: 1_360,
      total_input_tokens: 9_000,
      total_output_tokens: 500,
    })
    bridge.push('tool_call', { tool_call_id: 'metric-tool', name: 'ReadFile', arguments: '{}' })
    bridge.push('tool_result', { tool_call_id: 'metric-tool', name: 'ReadFile', duration_ms: 500, return_value: 'ok' })
    expect(store.getSnapshot()).toMatchObject({
      llmDurationMs: 3_000,
      llmSteps: 2,
      toolDurationMs: 500,
      toolSteps: 1,
      ttftMs: 1_000,
      inputTokens: 9_000,
      outputTokens: 500,
    })
    // A status echo without a cost or telemetry leaves the last known values alone.
    bridge.push('status_update', { context_tokens: 13_000 })
    expect(store.getSnapshot()).toMatchObject({ contextMax: 262_144, costUsd: 0.41, ttftMs: 1_000 })
    // An authoritative zero clears a previous profile's capacity to unknown.
    bridge.push('status_update', { max_context: 0 })
    expect(store.getSnapshot().contextMax).toBeNull()
  })

  test('initialize adopts the workspace git branch from the daemon', async () => {
    store.openTaskModal()
    await store.startTask('', false) // fresh initialize without a submit
    expect(store.getSnapshot().branch).toBe('main')
  })

  test('a daemon drop mid-turn folds live runs so carets cannot resurrect', async () => {
    // Turn one streams a partial reply…
    bridge.push('turn_begin', { user_input: 'hello' })
    bridge.push('text_part', { text: 'partial reply' })
    // Only agent/thinking variants carry `streaming` on the Block union.
    const streamingBlocks = (): Array<{ kind: string; text?: string }> =>
      store.getSnapshot().blocks
        .filter(block => (block.kind === 'agent' || block.kind === 'thinking') && block.streaming)
        .map(block => (block.kind === 'agent' || block.kind === 'thinking'
          ? { kind: block.kind, text: block.text }
          : { kind: block.kind }))
    expect(streamingBlocks()).toEqual([{ kind: 'agent', text: 'partial reply' }])

    // …then the connection dies before turn_end: the fold must commit the
    // runs as CLOSED blocks, or they come back as blinking carets riding
    // the next turn.
    bridge.respondWith(() => {
      throw new Error('socket closed')
    })
    store.retryConnection()
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(store.getSnapshot().turnActive).toBe(false)
    expect(streamingBlocks()).toEqual([])
    expect(store.getSnapshot().blocks.some(block => block.kind === 'agent' && block.text === 'partial reply')).toBe(true)

    // A new turn after reconnect: only ITS text streams; the old reply is a
    // settled block exactly once.
    bridge.respondWith(method => (method === 'initialize'
      ? { ok: true, session: { id: 's1' } }
      : { ok: true }))
    bridge.push('turn_begin', { user_input: 'again' })
    bridge.push('text_part', { text: 'fresh' })
    expect(streamingBlocks()).toEqual([{ kind: 'agent', text: 'fresh' }])
    expect(store.getSnapshot().blocks.filter(block => block.kind === 'agent' && block.text === 'partial reply')).toHaveLength(1)
    bridge.push('turn_end', {})
    expect(streamingBlocks()).toEqual([])
  })

  test('initialize reconciles a stuck acting badge with daemon turn truth', async () => {
    bridge.push('turn_begin', { user_input: 'hello' })
    bridge.push('text_part', { text: 'unterminated' })
    expect(store.getSnapshot().turnActive).toBe(true)

    // The daemon reports NO active turn (active_turn_id absent) — a fold
    // that keeps believing it is acting would blink carets forever.
    bridge.respondWith(method => (method === 'initialize'
      ? { ok: true, session: { id: 's1' } }
      : { ok: true }))
    await store.retryConnection()
    expect(store.getSnapshot().turnActive).toBe(false)
    expect(store.getSnapshot().blocks.some(block => (block.kind === 'agent' || block.kind === 'thinking') && block.streaming)).toBe(false)
    expect(store.getSnapshot().blocks.some(block => block.kind === 'agent' && block.text === 'unterminated')).toBe(true)
  })

  test('undoChanges drops undone files from the review list and reports refusals', async () => {
    bridge.push('turn_begin', { user_input: 'edit' })
    const first = editCall('src/a.ts', 'one', 'two', 'e1')
    const second = editCall('src/b.ts', 'three', 'four', 'e2')
    bridge.push(first.type, first.payload)
    bridge.push(second.type, second.payload)
    expect(store.getSnapshot().changes.map(change => change.path)).toEqual(['src/a.ts', 'src/b.ts'])

    bridge.respondWith(method => {
      if (method === 'changes.undo') {
        return { ok: true, reverted: 1, results: [{ path: 'src/a.ts', ok: true, reverted: 1 }, { path: 'src/b.ts', ok: false, error: 'file changed since edit 1 of 1 — refusing to undo blindly' }] }
      }
      return { ok: true }
    })
    await store.undoChanges(null)
    expect(store.getSnapshot().changes.map(change => change.path)).toEqual(['src/b.ts'])
    const notice = store.getSnapshot().blocks.at(-1)
    expect(notice?.kind).toBe('notice')
  })

  test('createWorktree switches the shell into the created worktree', async () => {
    // A list, not a `let` — control-flow analysis would pin the variable to
    // its `null` initializer across the async capture and mistype toBe().
    const usedWorkspaces: string[] = []
    ;(bridge as unknown as { useWorkspace?: (dir: string) => Promise<unknown> }).useWorkspace = async (dir: string) => {
      usedWorkspaces.push(dir)
      return dir
    }
    bridge.respondWith(method => {
      if (method === 'workspace.worktree') return { ok: true, path: '/repo/repo-feat-x', branch: 'feat-x' }
      return { ok: true }
    })
    store.openTaskModal()
    await store.createWorktree('feat x')
    expect(bridge.calls.some(call => call.method === 'workspace.worktree')).toBe(true)
    expect(usedWorkspaces).toEqual(['/repo/repo-feat-x'])
    expect(store.getSnapshot().taskModalOpen).toBe(false)
  })

  test('edits fold into per-file changes with real hunks and stats', () => {
    bridge.push('turn_begin', { user_input: 'fix it' })
    const call = editCall('src/a.ts', 'old line\nold two', 'new line\nnew two\nnew three')
    bridge.push(call.type, call.payload)
    const changes = store.getSnapshot().changes
    expect(changes).toHaveLength(1)
    expect(changes[0]).toMatchObject({ path: 'src/a.ts', adds: 3, dels: 2 })
    expect(changes[0]!.hunks.filter(line => line.kind === 'del')).toHaveLength(2)
    expect(changes[0]!.hunks.filter(line => line.kind === 'add')).toHaveLength(3)
  })

  test('an error notification during a turn marks it failed; retry resubmits', async () => {
    bridge.push('turn_begin', { user_input: 'make it pass' })
    bridge.push('notification', { severity: 'error', body: 'provider 429: quota exceeded' })
    bridge.push('turn_end', {})
    const failed = store.getSnapshot().failed
    expect(failed).toMatchObject({ error: 'provider 429: quota exceeded', turn: 1, lastUser: 'make it pass' })
    expect(store.getSnapshot().turnFailed).toBe(true)

    store.retryFailed()
    await new Promise(resolve => setTimeout(resolve, 5))
    expect(store.getSnapshot().failed).toBeNull()
    const resubmit = bridge.calls.filter(call => call.method === 'turn.submit')
    expect(resubmit.at(-1)?.params.text).toBe('make it pass')
  })

  test('compaction recovery targets the session, retains rejected failures, and never resubmits the goal', async () => {
    bridge.push('turn_begin', { user_input: 'continue goal' })
    bridge.push('notification', { severity: 'error', body: 'Automatic context compaction failed: 400' })
    bridge.push('turn_end', {})
    bridge.respondWith(() => ({ ok: false, error: 'Provider rejected compaction' }))
    await store.retryCompaction()
    expect(store.getSnapshot().failed?.error).toContain('Provider rejected compaction')
    expect(bridge.calls.at(-1)).toEqual({ method: 'slash', params: { session_key: store.getSnapshot().sessionKey, command: '/compact' } })
    bridge.respondWith(() => ({ ok: true, output: 'Compacted successfully' }))
    await store.retryCompaction()
    expect(store.getSnapshot().failed).toBeNull()
    expect(bridge.calls.filter(call => call.method === 'turn.submit')).toHaveLength(0)
  })

  test('a clean turn ends without failure and clears the queue', () => {
    bridge.push('turn_begin', { user_input: 'hello' })
    bridge.push('turn_end', {})
    expect(store.getSnapshot().failed).toBeNull()
    expect(store.getSnapshot().turnFailed).toBe(false)
  })

  test('plan mode captures agent markdown as the working plan', () => {
    bridge.push('status_update', { plan_mode: true })
    bridge.push('turn_begin', { user_input: 'plan the fix' })
    bridge.push('text_part', { text: '# Plan\n- [ ] reproduce\n- [ ] fix' })
    bridge.push('turn_end', {})
    const plan = store.getSnapshot().plan
    expect(plan?.items).toHaveLength(2)
    expect(plan?.items[0]).toMatchObject({ done: false, text: 'reproduce' })
    expect(store.getSnapshot().planMode).toBe(true)
  })

  test('a plan-review question captures the proposal and renders approval-shaped', () => {
    bridge.push('status_update', { plan_mode: true })
    bridge.push('turn_begin', { user_input: 'plan it' })
    bridge.push('text_part', { text: '## Approach\n- [ ] step one' })
    bridge.push('question_request', {
      id: 'q1',
      tool_call_id: '',
      questions: [{ id: 'answer', question: 'Do you approve this plan?', options: ['Approve plan and proceed', 'Keep planning'], allow_free_form: true }],
    })
    const snap = store.getSnapshot()
    expect(snap.question?.requestId).toBe('q1')
    expect(snap.plan?.markdown).toContain('## Approach')

    store.answerQuestion('q1', { answer: 'Approve plan and proceed' })
    const answered = bridge.calls.find(call => call.method === 'question_response')
    expect(answered?.params.answers).toEqual({ answer: 'Approve plan and proceed' })
  })

  test('approvals map onto the daemon vocabulary exactly', () => {
    bridge.push('approval_request', { id: 'ap1', tool_call_id: 't9', tool_name: 'bash', action: 'bash', description: 'rm -rf dist' })
    expect(store.getSnapshot().approval?.id).toBe('ap1')
    store.approve('ap1', 'allow_once')
    store.approve('ap1', 'allow_session')
    store.approve('ap1', 'deny')
    const responses = bridge.calls.filter(call => call.method === 'permission_response').map(call => call.params.response)
    expect(responses).toEqual(['approve', 'approve_for_session', 'reject'])
  })

  test('selecting a provider switches the profile and adopts its model', async () => {
    const profiles = [
      { name: 'kimi-code', provider: 'moonshot', model: 'kimi-for-coding', active: true },
      { name: 'zai', provider: 'z-ai', model: 'glm-5.2', active: false },
    ]
    const original = (bridge as unknown as { respond: Responder }).respond
    ;(bridge as unknown as { respond: Responder }).respond = (method: string) =>
      method === 'provider_list' ? { ok: true, profiles } : original(method, {})
    await store.loadProviders()
    // loadProviders patches from a floating promise; give it a tick.
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.getSnapshot().providers).toHaveLength(2)

    const initializeCount = bridge.calls.filter(call => call.method === 'initialize').length
    store.selectProvider('zai')
    await new Promise(resolve => setTimeout(resolve, 20))
    const select = bridge.calls.find(call => call.method === 'provider_select')
    expect(select?.params.name).toBe('zai')
    expect(bridge.calls.filter(call => call.method === 'initialize')).toHaveLength(initializeCount)
    expect(store.getSnapshot().model).toBe('glm-5.2')

    // No-ops: the already-active profile and unknown names stay silent.
    const callsBefore = bridge.calls.length
    store.selectProvider('kimi-code')
    store.selectProvider('nope')
    expect(bridge.calls.length).toBe(callsBefore)

    // Refused mid-turn: a running request is riding the live credentials.
    bridge.push('turn_begin', { user_input: 'hold on' })
    store.selectProvider('zai')
    const selects = bridge.calls.filter(call => call.method === 'provider_select').length
    expect(selects).toBe(1) // only the first, pre-turn switch
  })

  test('initialize failure flips the snapshot offline; recovery comes back', async () => {
    const dead = new FakeBridge(() => {
      throw new Error('daemon not connected')
    })
    const offlineStore = new Store()
    withWindow(dead)
    offlineStore.start(dead)
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(offlineStore.getSnapshot().connection).toBe('offline')
    delete (globalThis as { window?: unknown }).window
  })

  test('resuming a session adopts the daemon-bound key for every later call', async () => {
    const original = (bridge as unknown as { respond: Responder }).respond
    ;(bridge as unknown as { respond: Responder }).respond = (method: string, params: Record<string, unknown>) => {
      if (method === 'initialize' && params.resume_session_id === 'bb229911') {
        return {
          ok: true,
          session_id: 'bb229911',
          session: {
            id: 'bb229911', key: 'bb229911', title: 'resumed task', plan_mode: false, messages: [],
            turn_count: 39, calls: 1_200, llm_duration_ms: 30_947_000, llm_steps: 1_200,
            tool_duration_ms: 120_100, tool_steps: 906, input_tokens: 142_000, output_tokens: 8_000,
            ttft_samples: 100, ttft_total_ms: 850_000, ttft_avg_ms: 8_500,
            cache_hit_rate: 0.98, tokens_per_second: 43,
          },
        }
      }
      return original(method, params)
    }
    await store.openSession('bb229911')
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.getSnapshot()).toMatchObject({
      turnCount: 39,
      llmDurationMs: 30_947_000,
      llmSteps: 1_200,
      toolDurationMs: 120_100,
      toolSteps: 906,
      inputTokens: 142_000,
      outputTokens: 8_000,
      ttftMs: 8_500,
      cacheHitRate: 0.98,
      tokensPerSecond: 43,
    })
    // The daemon keys a resumed session by its id — our synthesized key would
    // have silently addressed a fresh, context-free session.
    await store.submit('continue this')
    const submitted = bridge.calls.find(call => call.method === 'turn.submit')
    expect(submitted?.params.session_key).toBe('bb229911')
  })

  test('model picks ride the daemon /model slash and are refused mid-turn', async () => {
    bridge.push('turn_begin', { user_input: 'busy' })
    store.pickModel('glm-5.2')
    expect(bridge.calls.some(call => call.method === 'slash' && String(call.params.command).includes('glm-5.2'))).toBe(false)

    bridge.push('turn_end', {})
    store.pickModel('glm-5.2')
    await new Promise(resolve => setTimeout(resolve, 10))
    // The /model slash pins the live session AND persists the choice to the
    // active profile — the old initialize+model path reverted on every
    // daemon restart.
    const pick = bridge.calls.find(call => call.method === 'slash' && call.params.command === '/model glm-5.2')
    expect(pick).toBeTruthy()
    expect(bridge.calls.some(call => call.method === 'initialize' && call.params.model === 'glm-5.2')).toBe(false)
  })

  test('status_update carries the session permission mode and foreign titles stay out', () => {
    bridge.push('status_update', { model: 'glm-5.2', context_tokens: 900, permission_mode: 'manual' })
    expect(store.getSnapshot().permissionMode).toBe('manual')

    // session_title is a broadcast: only our session may retitle the header.
    bridge.push('session_title', { session_id: 'someoneelse', title: 'Other Client Task' })
    expect(store.getSnapshot().currentTitle).toBe('')
    bridge.push('session_title', { session_id: initializeResult.session_id, title: 'Ours' })
    expect(store.getSnapshot().currentTitle).toBe('Ours')
  })

  test('a refused plan-mode flip never arms a local ceiling', async () => {
    const original = (bridge as unknown as { respond: Responder }).respond
    ;(bridge as unknown as { respond: Responder }).respond = (method: string) =>
      method === 'set_plan_mode' ? { ok: false, error: 'no active session' } : original(method, {})
    store.setPlanMode(true)
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.getSnapshot().planMode).toBe(false)
    expect(store.getSnapshot().blocks.at(-1)?.kind).toBe('notice')
  })

  test('provider model catalogs load by exact profile without switching it', async () => {
    bridge.respondWith((method, params) => {
      if (method === 'provider_models') {
        expect(params.profile_name).toBe('zai')
        return {
          ok: true,
          profile: 'zai',
          models: ['glm-5.3-flash', 'glm-5.2'],
          catalog: [
            { id: 'glm-5.3-flash', context_limit: 262_144, max_output_tokens: 65_536, context_source: 'provider' },
            { id: 'glm-5.2', context_limit: 1_000_000, max_output_tokens: 131_072, context_source: 'catalog', output_source: 'catalog' },
          ],
        }
      }
      return method === 'initialize' ? initializeResult : { ok: true }
    })
    store.loadProviderModels('zai')
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.getSnapshot().providerModels.zai).toEqual([
      {
        id: 'glm-5.3-flash',
        contextLimit: 262_144,
        contextSource: 'provider',
        maxOutputTokens: 65_536,
        overridden: false,
      },
      {
        id: 'glm-5.2',
        contextLimit: 1_000_000,
        contextSource: 'catalog',
        maxOutputTokens: 131_072,
        outputSource: 'catalog',
        overridden: false,
      },
    ])
    expect(store.getSnapshot().providerModelLoading).toEqual([])
    expect(bridge.calls.find(call => call.method === 'provider_models')?.params).toEqual({ profile_name: 'zai' })
  })

  test('model capacity edits send exact per-profile override fields', async () => {
    bridge.respondWith((method) => method === 'provider_model_override' ? { ok: true } : { ok: true, models: [] })
    store.saveModelCapabilities(' zai ', ' glm-5.2 ', 1_000_000, 131_072)
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(bridge.calls.find(call => call.method === 'provider_model_override')?.params).toEqual({
      profile_name: 'zai',
      model: 'glm-5.2',
      context_limit: 1_000_000,
      max_output_tokens: 131_072,
    })
    expect(bridge.calls.some(call => call.method === 'provider_models')).toBe(true)
  })

  test('provider save returns daemon refusal to keep the editor open for retry', async () => {
    const initializations = bridge.calls.filter(call => call.method === 'initialize').length
    bridge.respondWith(method => method === 'provider_save' ? { ok: false, error: 'Profile path is read-only' } : {})
    expect(await store.saveProvider({name:'local',baseUrl:'https://example.invalid/v1',model:'test'})).toBe('Profile path is read-only')
    expect(bridge.calls.filter(call => call.method === 'initialize')).toHaveLength(initializations)
    expect(await store.saveProvider({name:'',baseUrl:'',model:''})).toContain('required')
  })

  test('saving a provider persists via provider_save with exact wire fields', async () => {
    store.saveProvider({ name: ' openrouter ', baseUrl: ' https://api.local/v1 ', model: ' glm-5.2 ', provider: '', apiKey: '  ' })
    await new Promise(resolve => setTimeout(resolve, 10))
    const saved = bridge.calls.find(call => call.method === 'provider_save')
    // Blank optionals must be omitted entirely, and values trimmed.
    expect(saved?.params).toEqual({ name: 'openrouter', base_url: 'https://api.local/v1', model: 'glm-5.2' })

    // Required-field gaps surface as a notice instead of a dead call.
    const calls = bridge.calls.length
    store.saveProvider({ name: '', baseUrl: '', model: '' })
    expect(bridge.calls.length).toBe(calls)
    expect(store.getSnapshot().blocks.at(-1)?.kind).toBe('notice')

    // Refused mid-turn, same as switching.
    bridge.push('turn_begin', { user_input: 'busy' })
    store.saveProvider({ name: 'x', baseUrl: 'y', model: 'z' })
    expect(bridge.calls.filter(call => call.method === 'provider_save').length).toBe(1)
  })

  test('deleting a provider refuses the active profile and the daemon vocabulary', async () => {
    const profiles = [
      { name: 'kimi-code', provider: 'kimi', model: 'kimi-for-coding', active: true },
      { name: 'zai', provider: 'z-ai', model: 'glm-5.2', active: false },
    ]
    const original = (bridge as unknown as { respond: Responder }).respond
    ;(bridge as unknown as { respond: Responder }).respond = (method: string) =>
      method === 'provider_list' ? { ok: true, profiles } : original(method, {})
    await store.loadProviders()
    await new Promise(resolve => setTimeout(resolve, 10))

    store.deleteProvider('kimi-code')
    expect(bridge.calls.some(call => call.method === 'provider_delete')).toBe(false)

    store.deleteProvider('zai')
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(bridge.calls.find(call => call.method === 'provider_delete')?.params).toEqual({ name: 'zai' })
  })

  test('the daemon slash catalog folds into palette commands', async () => {
    const original = (bridge as unknown as { respond: Responder }).respond
    ;(bridge as unknown as { respond: Responder }).respond = (method: string) =>
      method === 'commands.catalog'
        ? { ok: true, pairs: [['/status', 'runtime status'], ['/undo', 'revert the last turn'], ['/bogus', '']] }
        : original(method, {})
    store.loadCommands()
    await new Promise(resolve => setTimeout(resolve, 10))
    const commands = store.getSnapshot().commands
    expect(commands.map(command => command.name)).toEqual(['status', 'undo', 'bogus'])
    expect(commands[0]?.description).toBe('runtime status')
  })

  test('a fast retry completion supersedes its optimistic acknowledgement', async () => {
    bridge.push('subagent_event', { agent_id: 'retry-child', title: 'Retry review', event: { type: 'turn_begin', payload: {} } })
    const completedAt = new Date().toISOString()
    bridge.respondWith(method => method === 'subagent.retry' ? { ok: true } : {
      session: { subagent_snapshots: [{ id: 'retry-child', title: 'Retry review', status: 'completed', summary: 'Finished before acknowledgement', updated_at: completedAt }] },
    })
    await store.controlAgent('retry-child', 'retry', 'Review again')
    await new Promise(resolve => setTimeout(resolve, 0))
    const child = store.getSnapshot().fleet.find(row => row.id === 'retry-child')
    expect(child?.status).toBe('completed')
    expect(child?.agentDetails?.summary).toBe('Finished before acknowledgement')
    expect(bridge.calls.find(call => call.method === 'subagent.retry')?.params.message).toBe('Review again')
  })

  test('nested agent events stay out of the main transcript and controls expose runtime rejection', async () => {
    const before = store.getSnapshot().blocks
    bridge.push('subagent_event', { agent_id: 'live-child', title: 'Live review', event: { type: 'turn_begin', payload: {} } })
    bridge.push('subagent_event', { agent_id: 'live-child', event: { type: 'tool_call', payload: { id: 'child-call', name: 'read_file', arguments: 'src/test.ts' } } })
    bridge.push('subagent_event', { agent_id: 'foreign-child', session_id: 'another-session', event: { type: 'turn_begin', payload: {} } })
    expect(store.getSnapshot().fleet.map(row=>row.id)).toContain('live-child')
    expect(store.getSnapshot().fleet.map(row=>row.id)).not.toContain('foreign-child')
    expect(store.getSnapshot().fleet.find(row=>row.id==='live-child')?.agentDetails?.toolCalls?.[0]?.arg).toBe('src/test.ts')
    expect(store.getSnapshot().blocks.filter(block=>block.kind==='tools')).toEqual(before.filter(block=>block.kind==='tools'))
    bridge.respondWith(method => method === 'subagent.interrupt' ? { found: false } : { ok: false, error: 'Retry denied' })
    await expect(store.controlAgent('live-child','stop')).rejects.toThrow('did not confirm')
    await expect(store.controlAgent('live-child','retry')).rejects.toThrow('Retry denied')
    expect(store.getSnapshot().fleet.find(row=>row.id==='live-child')?.status).toBe('running')
  })

  test('agent tool calls mid-turn refresh the fleet rail from subagent snapshots', async () => {
    // The manifest only exists once the spawn persists inside tool execution,
    // so model the race: session.status answers empty until the tool_call
    // arrives, then reports two running children.
    let spawned = false
    const snapshots = [
      { id: 'sub-one', title: 'Analyze libs/eyvan', status: 'working', summary: 'Reviewing cancellation handling', model: 'test-model', tool_count: 12, input_tokens: 4500, files_read: ['src/recovery.ts', 42], files_written: [], error: '' },
      { id: 'sub-two', title: 'Analyze the OCI release pipeline', status: 'working' },
    ]
    bridge.respondWith(method => {
      if (method === 'initialize') return initializeResult
      if (method === 'runtime.status') return { ok: true, permission_mode: 'auto' }
      if (method === 'session.goal') return { ok: true, text: 'No goal is currently set.' }
      if (method === 'session.list' || method === 'session.active_list') return { ok: true, sessions: [] }
      if (method === 'session.status') {
        return { ok: true, session: { id: 'aa19f402', ...(spawned ? { subagent_snapshots: snapshots } : {}) } }
      }
      return { ok: true }
    })

    bridge.push('turn_begin', { user_input: 'read and analyze the changes' })
    await new Promise(resolve => setTimeout(resolve, 10))
    // turn_begin refresh runs before any spawn — the panel is honestly empty.
    expect(store.getSnapshot().fleet).toHaveLength(0)

    spawned = true
    bridge.push('tool_call', { id: 't1', tool_call_id: 't1', name: 'AgentTool', arguments: '{"prompt":"analyze"}' })
    await new Promise(resolve => setTimeout(resolve, 10))
    const fleet = store.getSnapshot().fleet
    expect(fleet.map(row => row.title)).toEqual(['Analyze libs/eyvan', 'Analyze the OCI release pipeline'])
    expect(fleet.every(row => row.kind === 'subagent' && row.status === 'working')).toBe(true)
    expect(fleet[0]?.agentDetails).toMatchObject({ summary: 'Reviewing cancellation handling', model: 'test-model', toolCount: 12, inputTokens: 4500, filesRead: ['src/recovery.ts'] })

    // A non-agent tool call must not pay a status fetch: the snapshots it
    // would read cannot have moved.
    const statusCalls = bridge.calls.filter(call => call.method === 'session.status').length
    bridge.push('tool_call', { id: 't2', tool_call_id: 't2', name: 'bash', arguments: '{"command":"ls"}' })
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(bridge.calls.filter(call => call.method === 'session.status').length).toBe(statusCalls)

    // The foreground agent's terminal status lands with its result.
    spawned = false
    bridge.push('tool_result', { tool_call_id: 't1', name: 'AgentTool', result: 'done', permitted: true })
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.getSnapshot().fleet).toHaveLength(0)

    bridge.push('tool_result', { tool_call_id: 't2', name: 'bash', result: '', permitted: true })
    bridge.push('turn_end', {})
    expect(store.getSnapshot().turnActive).toBe(false)
  })

  test('a spawn batch opens an in-chat agents card even when snapshots never arrive', async () => {
    // Provider-outage shape: the daemon answers empty manifests forever; the
    // card must still appear from the spawn call itself, then mark failed
    // from the spawn's own error result.
    bridge.respondWith(method => {
      if (method === 'initialize') return initializeResult
      if (method === 'session.goal') return { ok: true, text: 'No goal is currently set.' }
      if (method === 'session.list' || method === 'session.active_list') return { ok: true, sessions: [] }
      if (method === 'session.status') return { ok: true, session: { id: 'aa19f402' } }
      return { ok: true }
    })
    bridge.push('turn_begin', { user_input: 'map the repo' })
    bridge.push('tool_call', {
      id: 't1',
      tool_call_id: 't1',
      name: 'SpawnAgents',
      arguments: JSON.stringify({ agents: [{ title: 'Map entry points' }, { title: 'Map hot paths' }] }),
    })
    await new Promise(resolve => setTimeout(resolve, 10))
    const card = store.getSnapshot().blocks.find(b => b.kind === 'agents')
    expect(card && card.kind === 'agents' ? card.members.map(m => `${m.title}:${m.status}`) : []).toEqual([
      'Map entry points:working',
      'Map hot paths:working',
    ])
    // The rail stays honest too: no fabricated fleet rows.
    expect(store.getSnapshot().fleet).toHaveLength(0)

    // The spawn dies inside the daemon — the card marks the batch failed.
    bridge.push('tool_result', { tool_call_id: 't1', name: 'SpawnAgents', error: 'provider unavailable', permitted: true })
    await new Promise(resolve => setTimeout(resolve, 10))
    const failed = store.getSnapshot().blocks.find(b => b.kind === 'agents')
    expect(failed && failed.kind === 'agents' ? failed.members.every(m => m.status === 'failed') : false).toBe(true)
    bridge.push('turn_end', {})
  })

  test('background turn events drive the jobs chip and never touch the fold', async () => {
    bridge.respondWith(method => {
      if (method === 'initialize') return initializeResult
      if (method === 'session.goal') return { ok: true, text: 'No goal is currently set.' }
      if (method === 'session.list' || method === 'session.active_list') return { ok: true, sessions: [] }
      if (method === 'session.status') return { ok: true, session: { id: 'aa19f402' } }
      return { ok: true }
    })
    await new Promise(resolve => setTimeout(resolve, 10))
    // A daemon-backgrounded turn starts on the same pipe.
    bridge.push('turn_begin', { session_id: 'bg-77', user_input: 'summarize the week' })
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.getSnapshot().backgroundJobs).toEqual([{ id: 'bg-77', title: 'summarize the week', status: 'working' }])
    // Its deltas must not open a foreground turn or blocks.
    bridge.push('text_part', { session_id: 'bg-77', text: 'bg output' })
    expect(store.getSnapshot().turnActive).toBe(false)
    expect(store.getSnapshot().blocks.filter(b => b.kind === 'agent')).toHaveLength(0)
    // Settled work leaves the chip.
    bridge.push('turn_end', { session_id: 'bg-77' })
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.getSnapshot().backgroundJobs).toHaveLength(0)
  })
})

 test('native reconnect startup replays the resumed transcript and binds its key', async () => {
  const bridge = new FakeBridge(method => method === 'initialize' ? {
    ...initializeResult, session_id: 'remote-resume',
    session: { id: 'remote-resume', key: 'remote-resume', title: 'Recovered', active_turn_id: 'still-running', messages: [{role:'user',content:'Keep the important work'}] },
  } : {ok:true})
  Object.assign(bridge, {getResumeSession: async () => 'remote-resume'})
  withWindow(bridge)
  const resumed = new Store()
  resumed.start(bridge)
  await new Promise(resolve => setTimeout(resolve, 20))
  expect(bridge.calls.find(call => call.method === 'initialize')?.params.resume_session_id).toBe('remote-resume')
  expect(resumed.getSnapshot().sessionKey).toBe('remote-resume')
  expect(resumed.getSnapshot().turnActive).toBe(true)
  bridge.push('turn_end', {})
  expect(JSON.stringify(resumed.getSnapshot().blocks)).toContain('Keep the important work')
  delete (globalThis as {window?:unknown}).window
})

test('desktop selection resumes per workspace after reload and explicit navigation takes precedence', async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  const values = new Map<string, string>()
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
  } })
  const bridge = new FakeBridge((method, params) => method === 'initialize' ? {
    ...initializeResult, session_id: params.resume_session_id || 'fresh',
    session: { id: params.resume_session_id || 'fresh', key: params.resume_session_id || 'fresh', cwd: '/repo', title: 'Selected task' },
  } : { ok: true })
  Object.assign(bridge, { getWorkspace: async () => '/repo' })
  withWindow(bridge)
  try {
    const first = new Store()
    first.start(bridge)
    await new Promise(resolve => setTimeout(resolve, 20))
    await first.openSession('chosen-task')
    bridge.calls.length = 0
    new Store().start(bridge)
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(bridge.calls.find(call => call.method === 'initialize')?.params.resume_session_id).toBe('chosen-task')
    bridge.calls.length = 0
    Object.assign(bridge, { getResumeSession: async () => 'explicit-task' })
    new Store().start(bridge)
    await new Promise(resolve => setTimeout(resolve, 20))
    expect(bridge.calls.find(call => call.method === 'initialize')?.params.resume_session_id).toBe('explicit-task')
  } finally {
    if (original) Object.defineProperty(globalThis, 'localStorage', original)
    else delete (globalThis as { localStorage?: unknown }).localStorage
    delete (globalThis as { window?: unknown }).window
  }
})

test('opening a second workspace window preserves the current session and surfaces host failure', async () => {
  const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : { ok: true })
  const paths: Array<string | undefined> = []
  let fail = true
  Object.assign(bridge, { openWorkspaceWindow: async (path?: string) => { paths.push(path); if (fail) throw new Error('Window unavailable') } })
  withWindow(bridge)
  const current = new Store()
  current.start(bridge)
  await new Promise(resolve => setTimeout(resolve, 20))
  const key = current.getSnapshot().sessionKey
  await current.openWorkspaceWindow('/second')
  expect(current.getSnapshot().workspaceError).toBe('Window unavailable')
  expect(current.getSnapshot().sessionKey).toBe(key)
  fail = false
  await current.openWorkspaceWindow('/second')
  expect(paths).toEqual(['/second', '/second'])
  expect(current.getSnapshot().workspaceError).toBeNull()
  expect(current.getSnapshot().workspaceBusy).toBe(false)
  expect(current.getSnapshot().sessionKey).toBe(key)
  delete (globalThis as { window?: unknown }).window
})

test('todos restore, update from confirmed tool results, and remain isolated from failed or background writes', async () => {
  const bridge = new FakeBridge(method => method === 'initialize' ? {...initializeResult,session:{...initializeResult.session,todos:[{id:'one',content:'Saved task',status:'pending'}]}} : {ok:true})
  withWindow(bridge)
  const current=new Store();current.start(bridge)
  await new Promise(resolve=>setTimeout(resolve,20))
  expect(current.getSnapshot().todos?.[0]?.content).toBe('Saved task')
  const result={name:'TodoWriteTool',permitted:true,display_blocks:[{type:'todo',items:[{id:'one',content:'Saved task',status:'in_progress'}]}]}
  bridge.push('tool_result',{...result,session_id:'foreign'})
  expect(current.getSnapshot().todos?.[0]?.status).toBe('pending')
  bridge.push('tool_result',{...result,error:'Denied'})
  expect(current.getSnapshot().todos?.[0]?.status).toBe('pending')
  bridge.push('tool_result',result)
  expect(current.getSnapshot().todos?.[0]?.status).toBe('in_progress')
  bridge.push('tool_result',{name:'TodoWriteTool',permitted:true,return_value:'1. [x] Saved task'})
  expect(current.getSnapshot().todos?.[0]?.status).toBe('completed')
  bridge.push('tool_result',{name:'TodoWriteTool',permitted:true,display_blocks:[{type:'todo',items:[]}]})
  expect(current.getSnapshot().todos).toEqual([])
  bridge.respondWith(method => method === 'initialize' ? { ...initializeResult, session_id:'different', session:{id:'different'} } : {ok:true})
  await current.openSession('different')
  expect(current.getSnapshot().todos).toBeNull()
  delete (globalThis as {window?:unknown}).window
})

test('stale cross-workspace startup selection opens a fresh local session without resuming foreign history', async () => {
  const bridge = new FakeBridge((method, params) => {
    if (method !== 'initialize') return { ok: true }
    if (params.resume_session_id) throw new Error('rpc -32000: Validation error for session_id: belongs to a main session from a different project')
    return initializeResult
  })
  Object.assign(bridge, { getWorkspace: async () => '/repo', getResumeSession: async () => 'foreign-session' })
  withWindow(bridge)
  try {
    const current = new Store()
    current.start(bridge)
    await new Promise(resolve => setTimeout(resolve, 20))
    const calls = bridge.calls.filter(call => call.method === 'initialize')
    expect(calls).toHaveLength(2)
    expect(calls[0]?.params.resume_session_id).toBe('foreign-session')
    expect(calls[1]?.params.resume_session_id).toBeUndefined()
    expect(calls[1]?.params.session_key).not.toBe(calls[0]?.params.session_key)
    expect(current.getSnapshot().connection).toBe('online')
    expect(current.getSnapshot().cwd).toBe('/repo')
    expect(JSON.stringify(current.getSnapshot().blocks)).toContain('original conversation is unchanged')
  } finally { delete (globalThis as {window?:unknown}).window }
})

test('startup authentication rejection stays actionable and is not automatically retried or replaced', async () => {
  const bridge = new FakeBridge(method => {
    if (method === 'initialize') throw new Error('Authentication failed: configure credentials')
    return { ok: true }
  })
  Object.assign(bridge, { getWorkspace: async () => '/repo', getResumeSession: async () => 'saved-session' })
  withWindow(bridge)
  try {
    const current = new Store()
    current.start(bridge)
    await new Promise(resolve => setTimeout(resolve, 20))
    await (current as unknown as { beat(): Promise<void> }).beat()
    expect(bridge.calls.filter(call => call.method === 'initialize')).toHaveLength(1)
    expect(current.getSnapshot().cwd).toBe('/repo')
    expect(current.getSnapshot().error).toContain('Authentication failed')
  } finally { delete (globalThis as {window?:unknown}).window }
})

test('desktop opens only the tail, retries older pages, and ignores pages from a previous session', async()=>{
 const action=(i:number)=>({id:String(i),messages:[{role:'user',content:`History ${i}`}],executions:[],thinking:[]})
 const tail={actions:Array.from({length:100},(_,i)=>action(i+100)),has_more:true,before:'page-100'}
 let rejectPage=true,release:((v:Record<string,unknown>)=>void)|undefined
 const bridge=new FakeBridge((method,params)=>{
   if(method==='initialize')return {...initializeResult,session:{...initializeResult.session,history:tail}}
   if(method==='session.history'){
     if(rejectPage)throw Error('Temporary read failure')
     return new Promise(resolve=>{release=resolve})
   }
   return {ok:true}
 })
 withWindow(bridge)
 try{
   const current=new Store();current.start(bridge);await new Promise(r=>setTimeout(r,20))
   expect(bridge.calls.find(c=>c.method==='initialize')?.params.history_limit).toBe(100)
   expect(current.getSnapshot().blocks).toHaveLength(100)
   expect(JSON.stringify(current.getSnapshot().blocks)).not.toContain('History 99"')
   await current.loadOlderHistory()
   expect(current.getSnapshot().historyError).toBe('Temporary read failure')
   expect(current.getSnapshot().historyMore).toBe(true)
   rejectPage=false
   const pending=current.loadOlderHistory()
   await current.loadOlderHistory()
   expect(bridge.calls.filter(c=>c.method==='session.history')).toHaveLength(2)
   release!({ok:true,history:{actions:Array.from({length:100},(_,i)=>action(i)),has_more:false,before:null}})
   await pending
   expect(current.getSnapshot().blocks).toHaveLength(200)
   expect(current.getSnapshot().historyMore).toBe(false)
   await current.openSession('next')
   const stale=current.loadOlderHistory()
   await current.openSession('third')
   release!({ok:true,history:{actions:[action(0)],has_more:false,before:null}})
   await stale
   expect(current.getSnapshot().blocks).toHaveLength(100)
   expect(current.getSnapshot().historyMore).toBe(true)
 }finally{delete (globalThis as {window?:unknown}).window}
})

test('opening another project during a turn preserves the running session and opens the exact selected session', async () => {
  const opened: unknown[][] = []
  let reject = false
  const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : method === 'session.list' ? { ok: true, sessions: [{ id: 'other-project-session', title: 'Other', cwd: '/another-project', message_count: 2 }] } : { ok: true, sessions: [] })
  Object.assign(bridge, { openWorkspaceWindow: async (...args: unknown[]) => { if (reject) throw Error('Window unavailable'); opened.push(args) } })
  withWindow(bridge)
  try {
    const current = new Store(); current.start(bridge)
    await new Promise(resolve => setTimeout(resolve, 20))
    bridge.push('turn_begin', { turn_id: 'active-turn', text: 'Keep working' })
    bridge.push('text_part', { text: 'Live response' })
    const before = current.getSnapshot()
    const calls = bridge.calls.length
    await current.openSession('other-project-session')
    expect(opened).toEqual([['/another-project', 'other-project-session']])
    expect(current.getSnapshot().currentId).toBe(before.currentId)
    expect(current.getSnapshot().sessionKey).toBe(before.sessionKey)
    expect(current.getSnapshot().blocks).toEqual(before.blocks)
    expect(current.getSnapshot().turnActive).toBe(true)
    expect(bridge.calls.slice(calls).some(call => call.method === 'initialize' || call.method === 'turn.cancel')).toBe(false)
    reject = true
    await current.openSession('other-project-session')
    expect(current.getSnapshot().error).toBe('Window unavailable')
    expect(current.getSnapshot().turnActive).toBe(true)
    bridge.push('text_part', { text: ' continues' })
    expect(JSON.stringify(current.getSnapshot().blocks)).toContain('Live response continues')
  } finally { delete (globalThis as { window?: unknown }).window }
})

test('online sidebar refresh reflects work started and completed in another workspace', async () => {
  let status='idle'
  const bridge=new FakeBridge(method=>method==='initialize'?initializeResult:method==='session.active_list'?{sessions:[{id:'other',cwd:'/other',title:'Other task',status,turn_count:1}]}:{sessions:[]})
  withWindow(bridge)
  try {
    const current=new Store();current.start(bridge)
    await new Promise(resolve=>setTimeout(resolve,20))
    expect(current.getSnapshot().live.find(row=>row.id==='other')?.status).toBe('idle')
    status='working'
    await new Promise(resolve=>setTimeout(resolve,5100))
    expect(current.getSnapshot().live.find(row=>row.id==='other')?.status).toBe('working')
    status='idle'
    await new Promise(resolve=>setTimeout(resolve,5100))
    expect(current.getSnapshot().live.find(row=>row.id==='other')?.status).toBe('idle')
  }finally{delete (globalThis as {window?:unknown}).window}
}, 15000)

describe('independent endpoint navigation', () => {
  afterEach(() => { delete (globalThis as { window?: unknown }).window })
  test('failed remote activation preserves the current session and displays the error', async () => {
    const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : { ok: true, sessions: [] })
    const selected: unknown[] = []
    Object.assign(bridge, { activateContext: async (...args: unknown[]) => { selected.push(args); throw new Error('SSH unavailable') } })
    withWindow(bridge)
    const store = new Store(); store.start(bridge)
    await Bun.sleep(10)
    const id = store.getSnapshot().currentId
    await store.activateContext(12, 'remote-task')
    expect(selected).toEqual([[12, 'remote-task']])
    expect(store.getSnapshot().currentId).toBe(id)
    expect(store.getSnapshot().connection).toBe('online')
    expect(store.getSnapshot().error).toContain('SSH unavailable')
    expect(bridge.calls.filter(row => row.method === 'initialize')).toHaveLength(1)
    expect(bridge.calls.some(row => row.method === 'turn.cancel')).toBe(false)
  })
  test('an unconnected SSH view retains local navigation and endpoint-specific storage', async () => {
    const bridge = new FakeBridge()
    const contexts = [{ id: 1, label: 'This Mac', workspace: '/repo', remote: false, sessions: [{ id: 'local', cwd: '/repo', title: 'Local task', status: 'working' }] }]
    Object.assign(bridge, { getWorkspace: async () => null, getContextScope: async () => 'ssh:worker', getContexts: async () => contexts })
    withWindow(bridge)
    const store = new Store(); store.start(bridge)
    await Bun.sleep(10)
    expect(store.getSnapshot().contexts).toEqual(contexts)
    expect(store.getSnapshot().storageScope).toBe('ssh:worker:')
    expect(store.getSnapshot().noWorkspace).toBe(true)
    expect(bridge.calls).toHaveLength(0)
  })
})

describe('transport restoration', () => {
  afterEach(() => { delete (globalThis as { window?: unknown }).window })
  test('leased reconnect preserves streamed text and restores missed deltas and unanswered prompts', async () => {
    const bridge = new FakeBridge(method => method === 'initialize' ? { ...initializeResult, connection_lease_supported: true } : { ok: true, sessions: [] })
    withWindow(bridge)
    const store = new Store(); store.start(bridge); await Bun.sleep(10)
    bridge.push('turn_begin', { text: 'Keep my task', turn_id: 'running' })
    bridge.push('text_part', { text: 'Before disconnect. ' })
    bridge.respondWith(method => method === 'initialize' ? {
      ...initializeResult,
      connection_lease_supported: true,
      session: { ...initializeResult.session, active_turn_id: 'running' },
      reconnect_events: [{ method: 'event', params: { type: 'text_part', payload: { text: 'During disconnect.' } } }],
      pending_interactions: [{ type: 'approval_request', payload: { id: 'recovered-permission', tool_name: 'Exec', arguments: {} } }],
    } : { ok: true, sessions: [] })
    bridge.push('desktop_connection', { online: false })
    await store.retryConnection()
    expect(JSON.stringify(store.getSnapshot().blocks)).toContain('Keep my task')
    expect(store.getSnapshot().blocks.some(block => block.kind === 'agent' && block.text === 'Before disconnect. During disconnect.')).toBe(true)
    expect(store.getSnapshot().approval).not.toBeNull()
    expect(store.getSnapshot().turnActive).toBe(true)
    bridge.push('turn_end', {})
  })
  test('socket recovery resumes the same session once despite repeated Retry clicks', async () => {
    const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : { ok: true, sessions: [] })
    withWindow(bridge)
    const store = new Store(); store.start(bridge); await Bun.sleep(10)
    let complete!: (result: Record<string, unknown>) => void
    bridge.respondWith(method => method === 'initialize' ? new Promise(resolve => { complete = resolve }) : { ok: true, sessions: [] })
    bridge.push('desktop_connection', { online: false })
    expect(store.getSnapshot().connection).toBe('offline')
    bridge.push('desktop_connection', { online: true })
    store.retryConnection(); store.retryConnection()
    expect(bridge.calls.filter(row => row.method === 'initialize')).toHaveLength(2)
    expect(bridge.calls.filter(row => row.method === 'initialize').at(-1)?.params.resume_session_id).toBe(initializeResult.session_id)
    complete({ ...initializeResult, session: { ...initializeResult.session, active_turn_id: 'still-running' } })
    await Bun.sleep(10)
    expect(store.getSnapshot().currentId).toBe(initializeResult.session_id)
    expect(store.getSnapshot().connection).toBe('online')
    expect(store.getSnapshot().turnActive).toBe(true)
    bridge.push('text_part', { session_id: initializeResult.session_id, text: 'After reconnect' })
    expect(store.getSnapshot().blocks.some(block => block.kind === 'agent' && block.text === 'After reconnect')).toBe(true)
    bridge.push('turn_end', {})
  })
})

describe('failed session navigation', () => {
  afterEach(() => { delete (globalThis as { window?: unknown }).window })
  test('a refused session open keeps the existing transcript and selection', async () => {
    const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : { ok: true, sessions: [] })
    withWindow(bridge)
    const store = new Store(); store.start(bridge); await Bun.sleep(10)
    bridge.push('turn_begin', { text: 'Existing task' })
    bridge.push('text_part', { text: 'Existing response' })
    bridge.push('turn_end', {})
    const before = store.getSnapshot()
    bridge.respondWith(method => method === 'initialize' ? { ok: false, error: 'Requested agent preset is missing' } : { ok: true, sessions: [] })
    await store.openSession('rejected-session')
    expect(store.getSnapshot().currentId).toBe(before.currentId)
    expect(store.getSnapshot().sessionKey).toBe(before.sessionKey)
    expect(JSON.stringify(store.getSnapshot().blocks)).toContain('Existing response')
    expect(store.getSnapshot().error).toBe('Requested agent preset is missing')
  })
  test('a new chat commits its identity only after initialization succeeds', async () => {
    const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : { ok: true, sessions: [] })
    withWindow(bridge)
    const store = new Store(); store.start(bridge); await Bun.sleep(10)
    bridge.push('turn_begin', { text: 'Existing task' })
    bridge.push('text_part', { text: 'Keep this response' })
    bridge.push('turn_end', {})
    const before = store.getSnapshot()
    let settle!: (value: Record<string, unknown>) => void
    bridge.respondWith(method => method === 'initialize' ? new Promise(resolve => { settle = resolve }) : { ok: true, sessions: [] })
    store.newChat()
    expect(store.getSnapshot().currentId).toBe(before.currentId)
    expect(store.getSnapshot().connection).toBe('connecting')
    expect(JSON.stringify(store.getSnapshot().blocks)).toContain('Keep this response')
    await store.retryConnection()
    expect(bridge.calls.filter(call => call.method === 'initialize')).toHaveLength(2)
    settle({ ok: false, error: 'New session refused' })
    await Bun.sleep(10)
    expect(store.getSnapshot().currentId).toBe(before.currentId)
    expect(store.getSnapshot().sessionKey).toBe(before.sessionKey)
    expect(store.getSnapshot().connection).toBe('online')
    expect(store.getSnapshot().error).toBe('New session refused')
    store.newChat()
    const key = String(bridge.calls.filter(call => call.method === 'initialize').at(-1)!.params.session_key)
    settle({ ...initializeResult, session_id: 'new-session', session: { ...initializeResult.session, id: 'new-session', key, messages: [] } })
    await Bun.sleep(10)
    expect(store.getSnapshot().currentId).toBe('new-session')
    expect(store.getSnapshot().sessionKey).toBe(key)
    expect(store.getSnapshot().blocks).toEqual([])
    expect(store.getSnapshot().error).toBeNull()
  })
})

describe('interaction response ownership', () => {
  afterEach(() => { delete (globalThis as { window?: unknown }).window })
  for (const kind of ['approval', 'question'] as const) {
    for (const outcome of ['success', 'refusal', 'error'] as const) {
      for (const navigate of [false, true]) {
        test(`${navigate ? 'session switch' : 'next request'}: late ${kind} ${outcome} does not replace the next request`, async () => {
          const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : { ok: true, sessions: [] })
          withWindow(bridge)
          const store = new Store(); store.start(bridge); await Bun.sleep(10)
          const request = (id: string) => bridge.push(kind === 'approval' ? 'approval_request' : 'question_request', kind === 'approval'
            ? { id, tool_call_id: id, tool_name: 'bash', action: 'bash', description: id }
            : { id, questions: [{ id: 'answer', question: id, options: ['yes', 'no'] }] })
          request('old')
          let resolve!: (value: Record<string, unknown>) => void
          let reject!: (error: Error) => void
          bridge.respondWith(method => method === 'initialize'
            ? { ...initializeResult, session_id: 'different-session', session: { ...initializeResult.session, id: 'different-session', key: 'different-key' } }
            : method === 'permission_response' || method === 'question_response'
              ? new Promise((done, fail) => { resolve = done; reject = fail }) : { ok: true, sessions: [] })
          if (kind === 'approval') store.approve('old', 'allow_once')
          else store.answerQuestion('old', { answer: 'yes' })
          if (navigate) await store.openSession('different-session')
          request(navigate ? 'old' : 'new')
          const before = store.getSnapshot()
          if (outcome === 'error') reject(new Error('old transport failure'))
          else resolve(outcome === 'refusal' ? { ok: false, error: 'old request refused' } : { ok: true })
          await Bun.sleep(10)
          expect(store.getSnapshot()[kind]).toEqual(before[kind])
          expect(store.getSnapshot().error).toBe(before.error)
          expect(store.getSnapshot().blocks).toEqual(before.blocks)
        })
      }
    }
  }
})

describe('rapid session navigation', () => {
  afterEach(() => { delete (globalThis as { window?: unknown }).window })
  for (const refused of [false, true]) test(`serializes daemon binding; final selection refused=${refused}`, async () => {
    const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : { ok: true, sessions: [] })
    withWindow(bridge)
    const store = new Store(); store.start(bridge); await Bun.sleep(10)
    const before = store.getSnapshot().currentId
    const pending: Array<{ id: string; resolve: (value: Record<string, unknown>) => void }> = []
    bridge.respondWith((method, params) => method === 'initialize'
      ? new Promise(resolve => pending.push({ id: String(params.resume_session_id), resolve })) : { ok: true, sessions: [] })
    const first = store.openSession('first')
    await Bun.sleep(0)
    const second = store.openSession('second')
    await Bun.sleep(0)
    expect(pending.map(row => row.id)).toEqual(['first'])
    expect(store.getSnapshot().connection).toBe('connecting')
    pending[0]!.resolve({ ...initializeResult, session_id: 'first', session: { id: 'first', key: 'first' } })
    await first
    await Bun.sleep(0)
    expect(store.getSnapshot().currentId).toBe(before)
    expect(pending.map(row => row.id)).toEqual(['first', 'second'])
    pending[1]!.resolve(refused ? { ok: false, error: 'Cannot open second' } : { ...initializeResult, session_id: 'second', session: { id: 'second', key: 'second' } })
    if (refused) {
      await Bun.sleep(0)
      expect(pending[2]?.id).toBe(before)
      pending[2]!.resolve(initializeResult)
    }
    await second
    expect(store.getSnapshot().currentId).toBe(refused ? before : 'second')
    expect(store.getSnapshot().sessionKey).toBe(refused ? before : 'second')
    expect(store.getSnapshot().connection).toBe('online')
  })
})

describe('session controls follow selection', () => {
  afterEach(() => { delete (globalThis as { window?: unknown }).window })
  test('resume adopts the selected model, effort, permissions, plan and live state', async () => {
    const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : { ok: true, sessions: [] })
    withWindow(bridge)
    const store = new Store(); store.start(bridge); await Bun.sleep(10)
    bridge.respondWith(method => method === 'initialize' ? {
      ...initializeResult, session_id: 'selected', model: 'selected-model', reasoning_effort: 'high', permission_mode: 'ask',
      branch: 'selected-branch', context_limit: 100000,
      session: { id: 'selected', key: 'selected', cwd: '/repo', model: 'selected-model', plan_mode: true, active_turn_id: 'running' },
    } : { ok: true, sessions: [] })
    await store.openSession('selected')
    expect(store.getSnapshot()).toMatchObject({ currentId: 'selected', model: 'selected-model', reasoningEffort: 'high',
      permissionMode: 'ask', planMode: true, branch: 'selected-branch', contextMax: 100000, turnActive: true })
    bridge.push('turn_end', {})
  })
})

describe('reasoning controls stay with their session', () => {
  afterEach(() => { delete (globalThis as { window?: unknown }).window })
  for (const action of ['load', 'select'] as const) for (const outcome of ['success', 'refusal', 'error'] as const) {
    test(`${action} ${outcome} from the previous session cannot change the selected chat`, async () => {
      let resolve!: (result: Record<string, unknown>) => void
      let reject!: (error: Error) => void
      const pending = new Promise<Record<string, unknown>>((yes, no) => { resolve = yes; reject = no })
      const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : { ok: true, sessions: [] })
      withWindow(bridge)
      const store = new Store(); store.start(bridge); await Bun.sleep(10)
      bridge.respondWith(method => method === 'initialize' ? {
        ...initializeResult, session_id: 'other', reasoning_effort: 'low', session: { id: 'other', key: 'other', cwd: '/repo' },
      } : method === (action === 'load' ? 'reasoning_levels' : 'slash') ? pending : { ok: true, sessions: [] })
      if (action === 'load') store.openReasoningPicker(); else store.pickReasoning('high')
      await store.openSession('other')
      const before = store.getSnapshot()
      if (outcome === 'error') reject(new Error('Old request failed'))
      else resolve(outcome === 'refusal' ? { ok: false, error: 'Old request refused' } : { ok: true, current: 'high', reasoning_effort: 'high', levels: [{ effort: 'high' }] })
      await Bun.sleep(1)
      expect(store.getSnapshot().reasoningEffort).toBe('low')
      expect(store.getSnapshot().error).toBe(before.error)
      expect(store.getSnapshot().blocks).toEqual(before.blocks)
      expect(store.getSnapshot().reasoningPickerOpen).toBe(false)
      expect(store.getSnapshot().reasoningLoading).toBe(false)
    })
  }
})

describe('session controls ignore responses from a previous selection', () => {
  afterEach(() => { delete (globalThis as { window?: unknown }).window })
  for (const action of ['model', 'plan', 'permissions', 'context'] as const) for (const outcome of ['success', 'refusal', 'error'] as const) {
    test(`${action}: late ${outcome} stays out of the selected chat`, async () => {
      let resolve!: (result: Record<string, unknown>) => void
      let reject!: (error: Error) => void
      const pending = new Promise<Record<string, unknown>>((yes, no) => { resolve = yes; reject = no })
      const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : { ok: true, sessions: [] })
      withWindow(bridge)
      const store = new Store(); store.start(bridge); await Bun.sleep(10)
      const rpc = action === 'context' ? 'context_breakdown' : action === 'plan' ? 'set_plan_mode' : 'slash'
      bridge.respondWith(method => method === 'initialize' ? {
        ...initializeResult, session_id: 'other', model: 'other-model', permission_mode: 'ask',
        session: { id: 'other', key: 'other', cwd: '/repo', plan_mode: false },
      } : method === rpc ? pending : { ok: true, sessions: [] })
      if (action === 'model') store.pickModel('old-model')
      if (action === 'plan') store.setPlanMode(true)
      if (action === 'permissions') store.setPermissionMode('auto')
      if (action === 'context') store.toggleContextMenu()
      await store.openSession('other')
      const before = store.getSnapshot()
      if (outcome === 'error') reject(new Error('Old control failed'))
      else resolve(outcome === 'refusal' ? { ok: false, error: 'Old control refused' } : {
        ok: true, model: 'old-model', permission_mode: 'auto', total_tokens: 9999, context_limit: 10000,
      })
      await Bun.sleep(1)
      const after = store.getSnapshot()
      expect(after.model).toBe('other-model')
      expect(after.planMode).toBe(false)
      expect(after.permissionMode).toBe('ask')
      expect(after.contextBreakdown).toBeNull()
      expect(after.contextMenuOpen).toBe(false)
      expect(after.contextBreakdownLoading).toBe(false)
      expect(after.error).toBe(before.error)
      expect(after.blocks).toEqual(before.blocks)
    })
  }
})

describe('permission control feedback', () => {
  afterEach(() => { delete (globalThis as { window?: unknown }).window })
  for (const outcome of ['success', 'refusal', 'error'] as const) {
    test(`${outcome} reports the result without pretending the permission changed`, async () => {
      let resolve!: (result: Record<string, unknown>) => void
      let reject!: (error: Error) => void
      const pending = new Promise<Record<string, unknown>>((yes, no) => { resolve = yes; reject = no })
      const bridge = new FakeBridge(method => method === 'initialize' ? { ...initializeResult, permission_mode: 'manual' } : { ok: true, sessions: [] })
      withWindow(bridge)
      const store = new Store(); store.start(bridge); await Bun.sleep(10)
      bridge.respondWith(method => method === 'slash' ? pending : { ok: true, sessions: [] })
      store.setPermissionMode('auto'); store.setPermissionMode('auto')
      expect(bridge.calls.filter(call => call.method === 'slash')).toHaveLength(1)
      expect(store.getSnapshot().permissionUpdating).toBe(true)
      expect(store.getSnapshot().permissionMode).toBe('manual')
      if (outcome === 'error') reject(new Error('Policy service rejected the request'))
      else resolve(outcome === 'refusal' ? { ok: false, error: 'Policy service rejected the request' } : { ok: true, permission_mode: 'auto' })
      await Bun.sleep(1)
      expect(store.getSnapshot().permissionUpdating).toBe(false)
      expect(store.getSnapshot().permissionMode).toBe(outcome === 'success' ? 'auto' : 'manual')
      expect(store.getSnapshot().permissionError).toBe(outcome === 'success' ? null : 'Policy service rejected the request')
    })
  }
})

describe('provider switching feedback and navigation', () => {
  afterEach(() => { delete (globalThis as { window?: unknown }).window })
  for (const navigate of [false, true]) for (const outcome of ['success', 'refusal', 'error'] as const) {
    test(`${outcome}, navigate=${navigate}: switch does not rebind the displayed chat`, async () => {
      let resolve!: (result: Record<string, unknown>) => void
      let reject!: (error: Error) => void
      const pending = new Promise<Record<string, unknown>>((yes, no) => { resolve = yes; reject = no })
      const profiles = [{ name: 'next', provider: 'fixture', model: 'next-model', active: false }]
      const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : method === 'provider_list' ? { ok: true, profiles } : { ok: true, sessions: [] })
      withWindow(bridge)
      const store = new Store(); store.start(bridge); await Bun.sleep(10); await store.loadProviders()
      bridge.respondWith(method => method === 'provider_select' ? pending : method === 'provider_list' ? { ok: true, profiles } : method === 'initialize' ? {
        ...initializeResult, session_id: 'other', model: 'other-model', session: { id: 'other', key: 'other', cwd: '/repo' },
      } : { ok: true, sessions: [] })
      store.selectProvider('next'); store.selectProvider('next')
      expect(bridge.calls.filter(c => c.method === 'provider_select')).toHaveLength(1)
      expect(store.getSnapshot().providerSwitching).toBe('next')
      if (navigate) await store.openSession('other')
      const initializes = bridge.calls.filter(c => c.method === 'initialize').length
      if (outcome === 'error') reject(new Error('Profile credentials rejected'))
      else resolve(outcome === 'refusal' ? { ok: false, error: 'Profile credentials rejected' } : { ok: true })
      await Bun.sleep(5)
      expect(bridge.calls.filter(c => c.method === 'initialize')).toHaveLength(initializes)
      expect(store.getSnapshot().currentId).toBe(navigate ? 'other' : initializeResult.session_id)
      expect(store.getSnapshot().model).toBe(navigate ? 'other-model' : outcome === 'success' ? 'next-model' : initializeResult.model)
      expect(store.getSnapshot().providerSwitching).toBeNull()
      expect(store.getSnapshot().providerSwitchError).toBe(outcome === 'success' ? null : 'Profile credentials rejected')
    })
  }
})

describe('saving providers preserves chat identity', () => {
  afterEach(() => { delete (globalThis as { window?: unknown }).window })
  for (const navigate of [false, true]) for (const apply of ['success', 'refusal', 'error'] as const) {
    test(`save then ${apply}, navigate=${navigate}`, async () => {
      let saved!: (result: Record<string, unknown>) => void
      const saveResult = new Promise<Record<string, unknown>>(resolve => { saved = resolve })
      const bridge = new FakeBridge(method => method === 'initialize' ? initializeResult : { ok: true, sessions: [] })
      withWindow(bridge)
      const store = new Store(); store.start(bridge); await Bun.sleep(10)
      const originalKey = store.getSnapshot().sessionKey
      bridge.respondWith(method => {
        if (method === 'provider_save') return saveResult
        if (method === 'set_model') {
          if (apply === 'error') return Promise.reject(new Error('Model apply failed'))
          return apply === 'refusal' ? { ok: false, error: 'Model apply failed' } : { ok: true, model: 'saved-model' }
        }
        if (method === 'initialize') return { ...initializeResult, session_id: 'other', model: 'other-model', session: { id: 'other', key: 'other', cwd: '/repo' } }
        return { ok: true, sessions: [] }
      })
      const pending = store.saveProvider({ name: 'saved', model: 'saved-model', baseUrl: 'http://localhost:9999/v1' })
      if (navigate) await store.openSession('other')
      const initializeCount = bridge.calls.filter(c => c.method === 'initialize').length
      saved({ ok: true, profile: { model: 'saved-model' } })
      const failure = await pending
      expect(bridge.calls.filter(c => c.method === 'initialize')).toHaveLength(initializeCount)
      expect(bridge.calls.find(c => c.method === 'set_model')?.params).toEqual({ session_key: originalKey, model: 'saved-model', provider_profile: 'saved' })
      expect(failure).toBe(apply === 'success' ? null : 'Provider saved, but its model could not be applied to the original chat: Model apply failed')
      expect(store.getSnapshot().currentId).toBe(navigate ? 'other' : initializeResult.session_id)
      expect(store.getSnapshot().model).toBe(navigate ? 'other-model' : apply === 'success' ? 'saved-model' : initializeResult.model)
    })
  }
})
