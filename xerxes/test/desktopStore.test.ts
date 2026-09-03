// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, describe, expect, test } from 'bun:test'

import { daemonCompatibilityWarning } from '../src/desktop/renderer/buildInfo.js'
import { Store, type XerxesLike } from '../src/desktop/renderer/store.js'
import type { DaemonEvent } from '../src/desktop/renderer/types.js'

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
  daemon_version: '0.4.0',
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

  afterEach(() => {
    delete (globalThis as { window?: unknown }).window
  })

  test('starts online after initialize and sends the desktop handshake', async () => {
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.getSnapshot().connection).toBe('online')
    expect(store.getSnapshot().model).toBe('kimi-for-coding')
    const initialize = bridge.calls.find(call => call.method === 'initialize')
    expect(initialize?.params).toMatchObject({ client_protocol: 35, client_version: '0.4.0' })
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

  // ── New-task modal (mockup 18) ────────────────────────────────────────

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

    store.selectProvider('zai')
    await new Promise(resolve => setTimeout(resolve, 20))
    const select = bridge.calls.find(call => call.method === 'provider_select')
    expect(select?.params.name).toBe('zai')
    const reinit = bridge.calls.filter(call => call.method === 'initialize').at(-1)
    expect(reinit?.params.model).toBe('glm-5.2')

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

  test('agent tool calls mid-turn refresh the fleet rail from subagent snapshots', async () => {
    // The manifest only exists once the spawn persists inside tool execution,
    // so model the race: session.status answers empty until the tool_call
    // arrives, then reports two running children.
    let spawned = false
    const snapshots = [
      { id: 'sub-one', title: 'Analyze libs/eyvan', status: 'working' },
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
