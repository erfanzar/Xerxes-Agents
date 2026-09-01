// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

import { Shell } from '../src/desktop/renderer/App.js'
import type { Snapshot } from '../src/desktop/renderer/store.js'

/**
 * Renders the presentational shell across connectivity states (SSR, no
 * browser) and guards the composition root: the preload, the security flags,
 * and the store's production start must all be wired, per
 * compositionRoot.test.ts precedent — injected-socket tests prove mechanism,
 * not wiring.
 */

const snapshot = (overrides: Partial<Snapshot>): Snapshot => ({
  connection: 'online',
  cwd: '/repo',
  model: 'kimi',
  currentAgentPreset: 'default',
  agentPresets: [],
  models: [],
  contextTokens: null,
  contextMax: null,
  ttftMs: null,
  tokensPerSecond: null,
  llmDurationMs: 0,
  llmSteps: 0,
  toolDurationMs: 0,
  toolSteps: 0,
  inputTokens: 0,
  outputTokens: 0,
  metricPhase: null,
  metricPhaseStartedAt: null,
  cacheHitRate: null,
  sessions: [],
  live: [],
  fleet: [],
  skillSuggestions: [],
  creatorTrace: [],
  backgroundJobs: [],
  currentId: '',
  currentTitle: '',
  sessionKey: '',
  branch: '',
  daemonWarning: null,
  costUsd: null,
  mcpStatus: {},
  goal: '',
  approval: null,
  question: null,
  planMode: false,
  turnActive: false,
  turnFailed: false,
  turnSeconds: 0,
  blocks: [],
  error: null,
  tab: 'activity',
  turnCount: 0,
  queue: [],
  changes: [],
  plan: null,
  log: [],
  failed: null,
  settingsOpen: false,
  settingsTab: 'general',
  paletteOpen: false,
  commands: [],
  pickerOpen: false,
  wsMenuOpen: false,
  sessionMenu: null,
  streamThinking: true,
  taskModalOpen: false,
  providers: [],
  providerModels: {},
  providerModelLoading: [],
  providerModelWarnings: {},
  providerTypes: [],
  permissionMode: '',
  snippets: {},
  ...overrides,
})

const render = (snap: Snapshot): string =>
  renderToStaticMarkup(createElement(Shell, { snap }))

test('the shell renders the right state in every connectivity mode', () => {
  const cases: Array<[string, Snapshot, string[]]> = [
    ['offline', snapshot({ connection: 'offline' }), ['daemon offline', 'New task']],
    ['connecting', snapshot({ connection: 'connecting' }), ['connecting']],
    ['online-empty', snapshot({}), ['Describe the outcome']],
    [
      'online-turn',
      snapshot({
        turnActive: true,
        turnSeconds: 12,
        blocks: [
          { kind: 'user', id: 1, text: 'map the repo' },
          { kind: 'tools', id: 2, running: true, items: [{ id: 't', verb: 'grep', arg: 'x', dur: '', state: 'working' }] },
          { kind: 'agent', id: 3, text: 'here it is', streaming: true },
        ],
      }),
      ['acting · 12s', 'Stop', 'map the repo', 'here it is', 'Acting… 12s', 'Grep'],
    ],
  ]
  for (const [label, snap, needles] of cases) {
    const html = render(snap)
    for (const needle of needles) {
      expect({ label, needle, found: html.includes(needle) }).toEqual({ label, needle, found: true })
    }
  }
})

test('a stale daemon handshake renders the actionable restart warning', () => {
  const html = render(snapshot({ daemonWarning: 'Daemon is older than the app — restart it.' }))
  expect(html).toContain('Daemon is older than the app — restart it.')
  expect(html).toContain('Restart the project daemon')
})

test('a running turn shows live state and a Stop affordance', () => {
  const html = render(snapshot({ turnActive: true, turnSeconds: 8 }))
  expect(html).toContain('acting · 8s')
  expect(html).toContain('Stop')
})

test('an approval request renders the three explicit responses', () => {
  const html = render(
    snapshot({ approval: { id: 'a1', action: 'bash', description: 'rm -rf tmp/' } }),
  )
  for (const needle of ['bash — approval required', 'Allow once', 'This session', 'Deny']) {
    expect(html).toContain(needle)
  }
})

test('a mid-turn approval flips the header to needs-input even while acting', () => {
  // The card lives in the Activity stream; on Changes/Plan/Log it is not
  // mounted, so the header badge is the only cross-tab signal.
  const html = render(
    snapshot({
      turnActive: true,
      approval: { id: 'a1', action: 'bash', description: 'rm -rf tmp/' },
    }),
  )
  expect(html).toContain('needs input · acting paused')
  expect(html).toContain('Stop')
})

test('the statusline reflects mode, model, context and connection', () => {
  const html = render(
    snapshot({
      planMode: true,
      model: 'kimi-for-coding',
      contextTokens: 61_000,
      contextMax: 262_000,
      turnCount: 39,
      llmDurationMs: 515 * 60_000 + 47_000,
      llmSteps: 1_200,
      toolDurationMs: 120_100,
      toolSteps: 906,
      inputTokens: 142_000,
      ttftMs: 8_500,
      tokensPerSecond: 43,
      cacheHitRate: 0.98,
    }),
  )
  for (const needle of ['⏸ plan', '39 turns · 2106 steps', 'LLM 515m47s · Tool calls 2m', 'TTFT avg 8.5s', '43.0 tok/s', 'Cache hit 98%', 'Input 142K', 'kimi-for-coding', 'ctx 61k/262k', 'connected']) {
    expect(html).toContain(needle)
  }
  const acting = render(snapshot({ turnActive: true }))
  expect(acting).toContain('▶ act')
  expect(acting).toContain('Cache hit —')
})

test('the statusline shows the git branch and cost only when the wire reports them', () => {
  const bare = render(snapshot({}))
  expect(bare).not.toContain('⎇')
  expect(bare).not.toContain('$')
  const full = render(snapshot({ branch: 'feat/cancel-safe', costUsd: 0.4123 }))
  expect(full).toContain('⎇ feat/cancel-safe')
  expect(full).toContain('$0.41')
  // Sub-cent costs keep four decimals; nothing is shown for a free run.
  const tiny = render(snapshot({ costUsd: 0.0041 }))
  expect(tiny).toContain('$0.0041')
  const free = render(snapshot({ costUsd: 0 }))
  expect(free).not.toContain('$')
})

test('no colour is hard-coded past the generated palette', () => {
  const html = render(
    snapshot({
      blocks: [
        { kind: 'user', id: 1, text: 'x' },
        { kind: 'tools', id: 2, running: false, items: [{ id: 't', verb: 'read', arg: 'f', dur: '1s', state: 'done' }] },
      ],
    }),
  )
  expect(html.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
})

// ── Workspace tabs ──────────────────────────────────────────────────────

test('the tabbed workspace renders every surface with live counts', () => {
  const changes = [
    { path: 'src/a.ts', adds: 12, dels: 4, isNew: false, hunks: [{ kind: 'add' as const, text: '+ new' }], turn: 1 },
  ]
  const tabs = render(
    snapshot({
      changes,
      plan: { markdown: '# Plan\n- [ ] one\n- [x] two', items: [{ text: 'one', done: false }, { text: 'two', done: true }], turn: 1 },
      log: [{ id: 1, turn: 1, type: 'text_part', summary: 'text=hi' }],
      tab: 'changes',
    }),
  )
  for (const needle of ['Activity', 'Changes', 'Plan', 'Log', 'src/a.ts', 'Keep all']) {
    expect(tabs).toContain(needle)
  }
  expect(tabs).toContain('+12')
  expect(tabs).toContain('1/2')

  const plan = render(snapshot({ tab: 'plan', planMode: true, plan: { markdown: '- [ ] one', items: [{ text: 'one', done: false }], turn: 1 } }))
  expect(plan).toContain('Working plan')
  expect(plan).toContain('⏸ plan mode')

  const log = render(snapshot({ tab: 'log', log: [{ id: 1, turn: 2, type: 'tool_call', summary: 'name=read' }] }))
  expect(log).toContain('tool_call')
  expect(log).toContain('name=read')
})

test('a failed turn renders the error body with retry and resolve affordances', () => {
  const html = render(
    snapshot({ failed: { error: 'provider 429: quota exceeded', turn: 3, lastUser: 'fix it' }, turnFailed: true }),
  )
  for (const needle of ['Turn 3 failed', 'provider 429: quota exceeded', 'Retry', 'Mark resolved']) {
    expect(html).toContain(needle)
  }
})

test('offline renders retry state and the launch command', () => {
  const html = render(snapshot({ connection: 'offline', cwd: '/Users/erfan/Documents/Projects/Xerxes-Agents' }))
  for (const needle of ['Daemon offline', 'Retry now', 'bun xerxes daemon --project-dir']) {
    expect(html).toContain(needle)
  }
})

test('the palette surfaces the daemon slash catalog with descriptions', () => {
  const html = render(
    snapshot({
      paletteOpen: true,
      commands: [
        { name: 'undo', description: 'revert the last turn' },
        { name: 'usage', description: 'token and cost usage' },
      ],
    }),
  )
  expect(html).toContain('/undo')
  expect(html).toContain('revert the last turn')
  expect(html).toContain('/usage')
  expect(html).toContain('token and cost usage')
})

test('a steering queue renders visibly above the composer', () => {
  const html = render(snapshot({ turnActive: true, queue: [{ id: 1, text: 'also cover the replay path' }] }))
  expect(html).toContain('queued 1')
  expect(html).toContain('also cover the replay path')
})

test('the settings modal opens on its cards and reads daemon state', () => {
  const general = render(snapshot({ settingsOpen: true, settingsTab: 'general' }))
  for (const needle of ['Settings', 'General', 'Theme', 'Models &amp; Providers', 'Permissions']) {
    expect(general).toContain(needle)
  }
  const models = render(
    snapshot({
      settingsOpen: true,
      settingsTab: 'models',
      model: 'kimi-for-coding',
      models: [{ id: 'kimi-for-coding', provider: 'kimi' }, { id: 'z-ai/glm-5.2', provider: 'z-ai' }],
      providers: [
        { name: 'kimi', provider: 'kimi', model: 'kimi-for-coding', active: true, baseUrl: 'https://api.moonshot.cn/v1' },
        { name: 'zai', provider: 'z-ai', model: 'glm-5.2', active: false, baseUrl: 'https://api.z.ai/api/coding/paas/v4' },
      ],
    }),
  )
  expect(models).toContain('kimi-for-coding')
  expect(models).toContain('Discovered models · 2')
  expect(models).toContain('z-ai/glm-5.2')
  // Provider rows are live switches, not a static list.
  expect(models).toContain('Providers · click to switch')
  expect(models).toContain('switch ▸')
  expect(models).toContain('title="make zai the active profile"')
  // TUI parity: the profile CRUD surface the /provider flow offers.
  expect(models).toContain('＋ Add provider')
  expect(models).toContain('title="delete zai"')
  // The active profile offers Edit but no Delete — switch away first.
  expect(models).toContain('title="edit kimi"')
  expect(models).not.toContain('title="delete kimi"')
  const permissions = render(snapshot({ settingsOpen: true, settingsTab: 'permissions', permissionMode: 'auto' }))
  expect(permissions).toContain('daemon reports: auto')
  expect(permissions).toContain('accept-all')
})

test('agent preset settings mirror the DSH roster and creator entry point', () => {
  const html = render(snapshot({
    settingsOpen: true,
    settingsTab: 'agents',
    agentPresets: [
      { id: 'default', name: 'Default', description: 'Full coding agent', trust: 'system', isDefault: true, manageable: false },
      { id: 'creator', name: 'Creator', description: 'Authors presets', trust: 'system', isDefault: false, manageable: false },
      { id: 'my-agent', name: 'My Agent', description: 'Custom', trust: 'user', isDefault: false, manageable: true },
    ],
  }))
  for (const needle of ['Agent presets', 'Draft a custom preset with Creator mode', 'Built-in', 'Custom', 'Set default', 'Duplicate', 'Open folder', 'Delete']) {
    expect(html).toContain(needle)
  }
})

test('a plan-review question renders markdown with approve / keep-planning', () => {
  const html = render(
    snapshot({
      planMode: true,
      plan: { markdown: '# Cancel-safe loop\n- [ ] reproduce', items: [{ text: 'reproduce', done: false }], turn: 1 },
      question: {
        requestId: 'q1',
        toolCallId: '',
        items: [{ id: 'answer', question: 'Review the plan before acting on it', options: ['Approve plan — start acting', 'Keep planning'], allowFreeform: true }],
      },
    }),
  )
  for (const needle of ['plan review', 'Cancel-safe loop', 'Approve plan — start acting', 'Keep planning', 'feedback for the next revision']) {
    expect(html).toContain(needle)
  }
})

test('checkpoint markers and edit stats render in the activity feed', () => {
  const html = render(
    snapshot({
      blocks: [
        {
          kind: 'tools',
          id: 1,
          running: false,
          items: [{ id: 't1', verb: 'edit', arg: 'src/a.ts', dur: '0.7s', state: 'done', path: 'src/a.ts', diff: { adds: 21, dels: 6 } }],
        },
        { kind: 'checkpoint', id: 2, turn: 1, adds: 21, dels: 6 },
      ],
    }),
  )
  expect(html).toContain('+21')
  // Flat row: label and detail live in separate spans — assert the parts.
  expect(html).toContain('Checkpoint')
  expect(html).toContain('turn 1 end')
})

test('the feed renders think and tool activity as flat one-line rows', () => {
  const html = render(
    snapshot({
      turnActive: true,
      turnSeconds: 75,
      blocks: [
        {
          kind: 'thinking' as const,
          id: 1,
          text: 'Let me trace how the cursor is rendered\nand cleared, end to end.',
          streaming: true,
        },
        {
          kind: 'tools' as const,
          id: 2,
          running: false,
          items: [{ id: 't1', verb: 'bash', arg: 'grep -rn caret src/', dur: '0.4s', state: 'done' as const }],
        },
        { kind: 'agent' as const, id: 3, text: 'The caret tails only the live run now.', streaming: false },
      ],
    }),
  )
  // Think row tails the latest streamed line; the full trail stays expandable.
  expect(html).toContain('Think')
  expect(html).toContain('<span class="frow__excerpt">and cleared, end to end. …</span>')
  // Tool row: title-cased label, arg summary, duration — no fold header.
  expect(html).toContain('Bash')
  expect(html).toContain('grep -rn caret src/')
  expect(html).toContain('0.4s')
  expect(html).not.toContain('tools · 1')
  // The live status line carries the turn clock at the end of the feed.
  expect(html).toContain('Acting… 1m 15s')
})

// ── Workspace switcher (mockup 16) ──────────────────────────────────────

test('streaming thinking clips from the live tail instead of freezing its prefix', () => {
  const stalePrefix = 'old-prefix-'.repeat(14)
  const html = render(snapshot({
    blocks: [{ kind: 'thinking' as const, id: 1, text: `${stalePrefix}LATEST_REASONING_TAIL`, streaming: true }],
  }))
  const excerpt = html.match(/<span class="frow__excerpt">([^<]+)<\/span>/)?.[1] ?? ''
  expect(excerpt).toContain('LATEST_REASONING_TAIL')
  expect(excerpt.startsWith('…')).toBe(true)
  expect(excerpt.length).toBeLessThanOrEqual(113)
})

test('the topbar renders the workspace as a switcher chip, not a dead label', () => {
  const html = render(snapshot({ cwd: '/Users/erfan/Documents/Projects/EasyDeL' }))
  expect(html).toContain('wschip')
  expect(html).toContain('Switch workspace')
  expect(html).toContain('EasyDeL')
})

test('the workspace menu lists known folders current-first with an add row', () => {
  const html = render(
    snapshot({
      cwd: '/repo',
      wsMenuOpen: true,
      sessions: [
        { id: 's1', key: 'k1', title: 'a', cwd: '/other', status: 'idle', turns: 1, messages: 2, age: '2h', untitled: false, current: false, kind: 'main' },
        { id: 's2', key: 'k2', title: 'b', cwd: '/repo', status: 'idle', turns: 2, messages: 4, age: '1h', untitled: false, current: true, kind: 'main' },
      ],
    }),
  )
  for (const needle of ['wsmenu', 'Workspaces', 'Add workspace…']) {
    expect(html).toContain(needle)
  }
  // Current workspace is marked, and the home group leads the list.
  expect(html).toContain('● current')
  expect(html.indexOf('repo')).toBeLessThan(html.indexOf('other'))
  expect(html).toContain('✓')
})

// ── Right rail run list (mockup 01) ─────────────────────────────────────

test('the feed renders the plan as a live to-dos card', () => {
  const html = render(
    snapshot({
      turnActive: true,
      plan: {
        markdown: '- [x] one\n- [ ] two\n- [ ] three',
        items: [{ text: 'one', done: true }, { text: 'two', done: false }, { text: 'three', done: false }],
        turn: 2,
      },
    }),
  )
  // Header counts: done / in-progress (first open while acting) / pending.
  expect(html).toContain('To-dos')
  expect(html).toContain('1 completed')
  expect(html).toContain('1 in progress')
  expect(html).toContain('1 pending')
  // Row states ride the icon's data-state: green check, spinner, dashed.
  expect(html).toContain('data-state="done"')
  expect(html).toContain('data-state="cur"')
  expect(html).toContain('data-state="todo"')
  // The rail no longer duplicates the list — the feed owns it.
  expect(html).not.toContain('This run')
})

test('the header fleet chip opens the live subagent roster', () => {
  const fleet = [
    { id: 'f1', key: 'f1', title: 'Analyze libs/eyvan', status: 'working', age: '', current: false, kind: 'subagent' as const, turns: 0, messages: 0, cwd: '', untitled: false },
    { id: 'f2', key: 'f2', title: 'Analyze the OCI pipeline', status: 'completed', age: '', current: false, kind: 'subagent' as const, turns: 0, messages: 0, cwd: '', untitled: false },
  ]
  const html = render(snapshot({ currentId: 'c1', currentTitle: 'T', fleet }))
  expect(html).toContain('2 subagents')
  // SSR renders local state closed; the roster mounts on click — but the
  // rail lists the same rows either way.
  expect(html).toContain('Analyze libs/eyvan')
  expect(html).toContain('Analyze the OCI pipeline')
})

test('the rail surfaces skill suggestions with observed tool telemetry', () => {
  const html = render(snapshot({
    skillSuggestions: [{
      skillName: 'release-checklist',
      description: 'Repeat the verified release sequence.',
      version: '0.1.0',
      sourcePath: '/tmp/release/SKILL.md',
      toolCount: 4,
      uniqueTools: ['Read', 'Bash'],
    }],
  }))
  expect(html).toContain('Skill suggestions · 1')
  expect(html).toContain('release-checklist')
  expect(html).toContain('4 tool calls · Read, Bash')
})

test('the rail separates legacy template-forge traces from Creator mode', () => {
  const html = render(snapshot({
    creatorTrace: [{
      action: 'define',
      name: 'briefing',
      version: '0.1.0',
      status: 'ok',
      detail: '',
      at: '2026-03-24T10:00:00.000Z',
    }],
  }))
  expect(html).toContain('Template forge · legacy')
  expect(html).toContain('define · briefing@0.1.0')
  expect(html).toContain('data-state="ok"')
})

test('spawned subagents render as an in-chat card; background jobs ride the header', () => {
  const html = render(
    snapshot({
      currentId: 'c1',
      currentTitle: 'T',
      backgroundJobs: [{ id: 'bg-1', title: 'summarize the week', status: 'working' }],
      blocks: [
        { kind: 'tools' as const, id: 1, running: false, items: [{ id: 't1', verb: 'spawn agents', arg: '2 agents', dur: '', state: 'done' as const }] },
        {
          kind: 'agents' as const,
          id: 2,
          members: [
            { key: 't1:0', title: 'Map entry points', status: 'working' },
            { key: 't1:1', title: 'Map hot paths', status: 'failed' },
          ],
        },
      ],
    }),
  )
  expect(html).toContain('1 background job running')
  expect(html).toContain('Subagents')
  expect(html).toContain('2 agents')
  expect(html).toContain('1 working')
  expect(html).toContain('1 failed')
  expect(html).toContain('Map entry points')
  expect(html).toContain('Map hot paths')
})

// ── Sidebar keeps the current task in its group (mockup 07) ─────────────

test('the current session stays in the sidebar with live status, not excluded', () => {
  const html = render(
    snapshot({
      cwd: '/repo',
      currentId: 'cur123',
      currentTitle: 'Ship cancel-safe loop',
      turnActive: true,
      turnSeconds: 9,
      turnCount: 3,
      sessions: [
        { id: 's1', key: 'k1', title: 'Old chat', cwd: '/repo', status: 'idle', turns: 4, messages: 8, age: '1d', untitled: false, current: false, kind: 'main' },
      ],
    }),
  )
  // The open session renders as a row with its live acting state…
  expect(html).toContain('Ship cancel-safe loop')
  expect(html).toContain('acting · 9s')
  expect(html).toContain('is-current')
  // …inside the current workspace group, next to the folder's history.
  expect(html).toContain('Old chat')
})

test('an acting current row shows acting; an idle one shows the turn count', () => {
  const acting = render(snapshot({ cwd: '/repo', currentId: 'c1', currentTitle: 'T', turnActive: true, turnSeconds: 4, turnCount: 1 }))
  // The status subline and the right-aligned age ride separate spans.
  expect(acting).toContain('acting')
  expect(acting).toContain('sess__age')
  const idle = render(snapshot({ cwd: '/repo', currentId: 'c1', currentTitle: 'T', turnActive: false, turnCount: 2 }))
  expect(idle).toContain('2 turns')
})

// ── Session context menu (mockup 08) ────────────────────────────────────

test('the session context menu offers open, rename and copy id', () => {
  const html = render(
    snapshot({ sessionMenu: { id: 'aa19f402', key: 'aa19f402', title: 'Ship loop', x: 40, y: 60 } }),
  )
  for (const needle of ['Session actions', 'Open', 'Rename…', 'Copy id']) {
    expect(html).toContain(needle)
  }
  // Items with no wire capability are deliberately absent.
  expect(html).not.toContain('Delete')
  expect(html).not.toContain('Move to worktree')
})

test('the session context menu rename mode renders an inline field', () => {
  const html = render(
    snapshot({ sessionMenu: { id: 'aa19f402', key: 'aa19f402', title: 'Ship loop', x: 40, y: 60 } }),
  )
  // SSR renders the closed state; the rename input mounts on user action.
  expect(html).not.toContain('menu__rename')
})

// ── Stream thinking switch (mockup 10) ──────────────────────────────────

test('the general card renders creator policy and the stream-thinking switch', () => {
  const html = render(snapshot({ settingsOpen: true }))
  expect(html).toContain('Creator mode')
  expect(html).toContain('DSH-style agent preset authoring · runtime inspection · changes apply to future sessions')
  expect(html).toContain('Stream thinking')
  expect(html).toContain('switch is-on')
})

test('streamThinking off hides thinking blocks but keeps the tool runs', () => {
  const blocks = [
    { kind: 'thinking' as const, id: 1, text: 'secret plan', streaming: false },
    { kind: 'tools' as const, id: 2, running: false, items: [{ id: 't1', verb: 'grep', arg: 'x', dur: '', state: 'done' }] },
    { kind: 'agent' as const, id: 3, text: 'answer' },
  ]
  const on = render(snapshot({ blocks }))
  expect(on).toContain('secret plan')
  const off = render(snapshot({ blocks, streamThinking: false }))
  expect(off).not.toContain('secret plan')
  expect(off).toContain('Grep')
  expect(off).toContain('answer')
})

// ── New-task modal (mockup 18) ──────────────────────────────────────────

test('Creator mode is a direct visible desktop launch, not hidden in settings', () => {
  const html = render(snapshot({ currentAgentPreset: 'creator' }))
  expect(html).toContain('◈ Creator mode')
  expect(html).toContain('Start a fresh DSH-style Creator mode session')
})

test('the new-task modal renders the mockup-18 grammar', () => {
  const html = render(snapshot({ taskModalOpen: true, permissionMode: 'auto' }))
  expect(html).toContain('New task')
  expect(html).toContain('Objective')
  expect(html).toContain('Review plan before changes')
  expect(html).toContain('Start task ⏎')
  expect(html).toContain('Cancel')
  expect(html).toContain('switch is-on')
  expect(html).toContain('approvals in this workspace: auto')
  // No fabricated worktree slots: the current folder, the folder picker,
  // and the worktree creator only — never a pretend list of slots.
  expect(html).not.toContain('· main')
  expect(html).toContain('new worktree…')
})

test('the task modal stays closed by default and keeps worktree wording out', () => {
  const html = render(snapshot({}))
  expect(html).not.toContain('Start task')
})

// ── Changes undo (mockup 03) ────────────────────────────────────────────

test('the changes tab offers per-file and whole-list undo', () => {
  const html = render(
    snapshot({
      tab: 'changes',
      changes: [{ path: 'src/a.ts', adds: 2, dels: 1, isNew: false, hunks: [{ kind: 'add', text: 'new' }] }],
    }),
  )
  expect(html).toContain('Undo all')
  expect(html).toContain('>undo<')
  expect(html).toContain('Reverse this session&#x27;s recorded edits to this file')
})

// ── MCP settings card (mockup 19) ───────────────────────────────────────

test('the MCP card renders connected and failed servers from the daemon', () => {
  const html = render(
    snapshot({
      settingsOpen: true,
      settingsTab: 'mcp',
      mcpStatus: {
        filesystem: { connected: true, tools: 11, resources: 2, prompts: 1 },
        sqlite: { connected: false, tools: 0, resources: 0, prompts: 0, lastError: 'connect ECONNREFUSED' },
      },
    }),
  )
  expect(html).toContain('MCP Servers')
  expect(html).toContain('filesystem')
  expect(html).toContain('connected · 11 tools · 2 resources · 1 prompts')
  expect(html).toContain('not connected — connect ECONNREFUSED')
  expect(html).toContain('Reload servers')
  expect(html).toContain('~/.xerxes/mcp.json')
})

test('the MCP card explains where configuration lives when no servers exist', () => {
  const html = render(snapshot({ settingsOpen: true, settingsTab: 'mcp' }))
  expect(html).toContain('No MCP servers configured')
  expect(html).toContain('~/.xerxes/mcp.json')
})

test('the composer goal badge follows parseGoal, not raw goal-text truthiness', () => {
  // The daemon answers an empty /goal query with prose and the store keeps
  // that string verbatim — truthiness alone lit the badge with no goal set.
  const none = render(snapshot({ goal: 'No goal is currently set.\nUsage: /goal [<objective>|clear|edit <objective>|pause|resume]' }))
  expect(none).toContain('No goal set')
  expect(none).not.toContain('◎ goal set')

  const armed = render(snapshot({ goal: 'Goal created\nStatus: active\nObjective: ship the release\nRounds: 0/20\nActivation: armed' }))
  expect(armed).toContain('◎ goal set')
  expect(armed).toContain('ship the release')
})

// ── Composition-root wiring ─────────────────────────────────────────────

test('feed rows are scoped (.frow) and provider rows stack name over detail', async () => {
  // The feed's flat-row classes must not collide with the settings lists'
  // `.row` grammar — an unscoped `.row` restyle ran provider names and their
  // detail lines together on one baseline.
  const [css, app] = await Promise.all([read('renderer/app.css'), read('renderer/App.tsx')])
  expect(css).toMatch(/^\.frow \{/m)
  expect(app).toContain('frow frow--tool')
  expect(app).not.toContain('className="row row--tool"')

  const models = render(
    snapshot({
      settingsOpen: true,
      settingsTab: 'models',
      providers: [{ name: 'zai', provider: 'zhipu', model: 'glm-5.2', active: true, baseUrl: 'https://api.z.ai/api/coding/paas/v4' }],
    }),
  )
  // Provider cards (Codex grammar): stacked name/sub, Edit action, no
  // Delete on the active profile.
  expect(models).toContain('pcard__main')
  expect(models).toContain('zai')
  expect(models).toContain('in use')
  expect(models).toContain('Edit')
  expect(models).not.toContain('pcard__del')
})

test('the provider form is wire-fed: registry dropdown, env fallback, default endpoint', async () => {
  // The sheet mounts on local state (not SSR-visible), so guard the wiring
  // at the source: the dropdown reads providerTypes, the key placeholder
  // names the registry's env var, and Base URL shows the registry default.
  const overlays = await read('renderer/Overlays.tsx')
  expect(overlays).toContain('snap.providerTypes')
  expect(overlays).toContain('<select')
  expect(overlays).toContain('leave blank to use $')
  expect(overlays).toContain('Provider default —')
  expect(overlays).toContain('leave blank to keep the stored key')

  // Rendered for real: the dropdown lists the daemon's registry, and the
  // env fallback + registry default ride the selected type.
  const { ProviderForm } = await import('../src/desktop/renderer/Overlays.js')
  const form = renderToStaticMarkup(
    createElement(ProviderForm, {
      snap: snapshot({
        providerTypes: [
          { name: 'zhipu', baseUrl: 'https://api.z.ai/api/coding/paas/v4', apiKeyEnv: 'ZHIPU_API_KEY' },
          { name: 'custom', baseUrl: '', apiKeyEnv: 'CUSTOM_API_KEY' },
        ],
      }),
      editing: null,
      onCancel: () => {},
    }),
  )
  expect(form).toContain('<select')
  expect(form).toContain('>zhipu</option>')
  expect(form).toContain('>custom</option>')
  expect(form).toContain('Customized settings')

  const editing = { name: 'zai', provider: 'zhipu', model: 'glm-5.2', active: true, baseUrl: '' }
  const editForm = renderToStaticMarkup(
    createElement(ProviderForm, {
      snap: snapshot({
        providers: [editing],
        providerModels: { zai: ['glm-5.3-flash', 'glm-5.2'] },
      }),
      editing,
      onCancel: () => {},
    }),
  )
  expect(editForm).toContain('Models · 2 discovered')
  expect(editForm).toContain('glm-5.3-flash')
  expect(editForm).toContain('Fetch this provider’s models')
})

const DESKTOP = join(import.meta.dir, '..', 'src', 'desktop')
const read = (relative: string): Promise<string> => readFile(join(DESKTOP, relative), 'utf8')

test('main.ts wires the preload through an intact security posture', async () => {
  const [main, ipc] = await Promise.all([read('main.ts'), read('main/ipc.ts')])
  expect(main).toContain('preload:')
  expect(main).toContain('registerDaemonBridge')
  // The single call channel lives on the bridge surface.
  expect(ipc).toContain("'daemon:call'")
  expect(ipc).toContain("'daemon:event'")
  expect(main).toContain('contextIsolation: true')
  expect(main).toContain('nodeIntegration: false')
  expect(main).toContain('sandbox: true')
  expect(main).toContain("const APP_NAME = 'Xerxes Agents'")
  expect(main).toContain('app.setName(APP_NAME)')
  expect(main).toContain('title: APP_NAME')
})

test('desktop launch metadata names the product instead of the Electron runtime', async () => {
  const [builder, packager, runner, html, packageJson] = await Promise.all([
    readFile(join(DESKTOP, '..', '..', 'scripts', 'buildDesktop.ts'), 'utf8'),
    readFile(join(DESKTOP, '..', '..', 'scripts', 'packageDesktopMac.ts'), 'utf8'),
    readFile(join(DESKTOP, '..', '..', 'scripts', 'runDesktop.ts'), 'utf8'),
    read('renderer/index.html'),
    readFile(join(DESKTOP, '..', '..', 'package.json'), 'utf8'),
  ])
  expect(builder).toContain("productName: 'Xerxes Agents'")
  expect(builder).toContain("main: 'main.js'")
  expect(packager).toContain("replacePlistString(plist, 'CFBundleDisplayName', productName)")
  expect(packager).toContain("replacePlistString(plist, 'CFBundleExecutable', productName)")
  expect(packager).toContain("replacePlistString(plist, 'CFBundleIdentifier', 'dev.xerxes.agents')")
  expect(packager).toContain('brandHelperBundles')
  expect(packager).toContain('`${productName} Helper${suffix}`')
  expect(runner).toContain("'Xerxes Agents.app'")
  expect(html).toContain('<title>Xerxes Agents</title>')
  expect(packageJson).toContain('bun scripts/runDesktop.ts')
  expect(packageJson).not.toContain('bunx electron dist/desktop')
})

test('the event channel ships one frame object the preload can parse', async () => {
  // ipc sends; preload's cleanEvent validates a single {type, payload} frame.
  // Sending (type, payload) as two args once made cleanEvent parse the type
  // STRING as a frame and silently drop every daemon event.
  const [ipc, preload] = await Promise.all([read('main/ipc.ts'), read('preload.ts')])
  expect(ipc).toMatch(/send\('daemon:event',\s*\{\s*type,\s*payload\s*\}\)/)
  expect(preload).toContain('const { type, payload } = frame as')
})

test('the preload exposes narrow validated capabilities, never raw ipcRenderer', async () => {
  const [preload, main] = await Promise.all([read('preload.ts'), read('main.ts')])
  expect(preload).toContain("contextBridge.exposeInMainWorld('xerxes'")
  expect(preload).toContain('call(')
  expect(preload).toContain('onEvent(')
  expect(preload).toContain('openPath(path: unknown)')
  expect(main).toContain("ipcMain.handle('native:preset:open-path'")
  expect(main).toContain("relative(root, candidate)")
  expect(preload).not.toMatch(/exposeInMainWorld\(\s*'xerxes'\s*,\s*ipcRenderer/)
})

test('the renderer actually starts the store in production', async () => {
  const app = await read('renderer/App.tsx')
  expect(app).toContain('store.start()')
})
