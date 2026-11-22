// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { FilesChangedCard, Shell, Sidebar } from '../src/desktop/renderer/App.js'
import { T3ModelPicker, compactModelTriggerName, reasoningLevelLabel } from '../src/desktop/renderer/T3ModelPicker.js'
import { T3Sidebar, groupSectionsByMachine } from '../src/desktop/renderer/T3Sidebar.js'
import { ThreadDetailsCard } from '../src/desktop/renderer/ThreadDetailsCard.js'
import { RightPanelLauncher, XERXES_SURFACES } from '../src/desktop/renderer/RightPanelLauncher.js'
import { DesktopPage, DesktopRail, RAIL_PANELS } from '../src/desktop/renderer/DesktopPanels.js'
import { SubagentsSection, subagentRoleIcon } from '../src/desktop/renderer/RailSections.js'
import { ConnectHub } from '../src/desktop/renderer/SupervisorPage.js'
import { railT3IconOf } from '../src/desktop/renderer/T3Icons.js'
import {
  loadDesktopWorkspaces,
  rememberDesktopWorkspace,
  removeDesktopWorkspace,
  saveDesktopWorkspace,
} from '../src/desktop/main/workspaceSettings.js'
import {
  TerminalDrawer,
  TERMINAL_DRAWER_DEFAULT_HEIGHT,
  TERMINAL_DRAWER_MAX_HEIGHT,
  TERMINAL_DRAWER_MIN_HEIGHT,
  clampTerminalDrawerHeight,
} from '../src/desktop/renderer/TerminalDrawer.js'
import { ProjectMonogram, setProjectIconCacheForTests } from '../src/desktop/renderer/ProjectMonogram.js'
import {
  clearProjectIconCacheForTests,
  extractDeclaredIconHref,
  resolveProjectIconDataUrl,
} from '../src/desktop/main/projectFavicon.js'
import { PALETTES } from '../src/desktop/renderer/palettes.js'
import { Store, type Snapshot } from '../src/desktop/renderer/store.js'
import {
  formatSnoozeRemaining,
  hideWorkspaceInState,
  isSnoozedActive,
  monogramForProject,
  parseThreadInboxState,
  partitionInboxThreads,
  resetThreadInboxCacheForTests,
  setSettledInState,
  setSnoozedInState,
  snoozePresets,
  togglePinInState,
  unhideWorkspaceInState,
} from '../src/desktop/renderer/threadInbox.js'

const HEX_RE = /#[0-9a-fA-F]{6}\b/g
const holder = globalThis as { document?: unknown }
const originalDocument = holder.document

function stampStyle(style: string | null): void {
  holder.document = {
    documentElement: {
      dataset: {},
      getAttribute: (name: string) => (name === 'data-style' ? style : null),
    },
  }
}

afterEach(() => {
  holder.document = originalDocument
  resetThreadInboxCacheForTests()
  setProjectIconCacheForTests()
  clearProjectIconCacheForTests()
})

function makeSnapshot(overrides: Partial<Snapshot> = {}): Snapshot {
  return {
    ...new Store().getSnapshot(),
    connection: 'online',
    noWorkspace: false,
    cwd: '/workspace/Xerxes-Agents',
    branch: 'main',
    model: 'claude-opus-4-6',
    currentId: 'thr-1',
    currentTitle: 'Match T3 Code UI',
    sessionKey: 'key-1',
    ...overrides,
  }
}

describe('T3 Code Sidebar V2 inbox logic', () => {
  test('partitions threads into pinned, active, snoozed and settled while keeping working threads visible', () => {
    const now = 1_700_000_000_000
    let state = parseThreadInboxState(null)
    state = togglePinInState(state, 't-pin')
    state = setSnoozedInState(state, 't-snooze', now + 60 * 60_000)
    state = setSnoozedInState(state, 't-expired', now - 1_000)
    state = setSettledInState(state, 't-settled', true)
    state = setSettledInState(state, 't-settled-working', true)

    const rows = [
      { id: 't-active', status: 'idle' },
      { id: 't-pin', status: 'idle' },
      { id: 't-snooze', status: 'idle' },
      { id: 't-expired', status: 'idle' },
      { id: 't-settled', status: 'idle' },
      { id: 't-settled-working', status: 'working' },
    ]
    const result = partitionInboxThreads(rows, state, now)
    expect(result.pinned.map(row => row.id)).toEqual(['t-pin'])
    expect(result.active.map(row => row.id)).toEqual(['t-active', 't-expired', 't-settled-working'])
    expect(result.snoozed.map(row => row.id)).toEqual(['t-snooze'])
    expect(result.settled.map(row => row.id)).toEqual(['t-settled'])
    expect(isSnoozedActive(state.snoozed, 't-snooze', now)).toBe(true)
    expect(isSnoozedActive(state.snoozed, 't-expired', now)).toBe(false)
  })

  test('snooze presets and remaining time formatter produce readable labels', () => {
    const now = new Date('2026-11-22T14:00:00Z').getTime()
    const presets = snoozePresets(now)
    expect(presets.map(preset => preset.id)).toEqual(['1h', '3h', 'tomorrow', 'week'])
    expect(formatSnoozeRemaining(now + 45 * 60_000, now)).toBe('45m')
    expect(formatSnoozeRemaining(now + 3 * 60 * 60_000, now)).toBe('3h')
    expect(formatSnoozeRemaining(now + 48 * 60 * 60_000, now)).toBe('2d')
  })

  test('ProjectMonogram shows real discovered project logos and renders nothing (pure text) when absent', async () => {
    expect(monogramForProject('Xerxes-Agents').text).toBe('XA')
    expect(monogramForProject('t3code').text).toBe('T3')

    // No cached/discovered logo -> renders nothing (zero 2-letter placeholder box)
    expect(renderToStaticMarkup(createElement(ProjectMonogram, { name: 'prismcore', seed: '/workspace/prismcore' }))).toBe('')

    // Cached real project logo -> renders crisp <img> favicon with zero inline hexes
    setProjectIconCacheForTests({ '/workspace/Xerxes-Agents': 'data:image/png;base64,iVBORw0KGgo=' })
    const logoHtml = renderToStaticMarkup(createElement(ProjectMonogram, { name: 'Xerxes-Agents', seed: '/workspace/Xerxes-Agents', size: 16 }))
    expect(logoHtml).toContain('data-slot="project-favicon"')
    expect(logoHtml).toContain('src="data:image/png;base64,iVBORw0KGgo="')
    expect(logoHtml.match(HEX_RE) ?? []).toEqual([])

    // Real filesystem discovery: assets/logo.svg, index.html <link rel="icon">, symlink escape refusal, and missing -> null
    expect(extractDeclaredIconHref('<link rel="icon" href="/brand/mark.svg?v=2">')).toBe('/brand/mark.svg')

    const temp = await mkdtemp(join(tmpdir(), 'xerxes-project-icon-'))
    try {
      const emptyDir = join(temp, 'plain-repo')
      await mkdir(emptyDir, { recursive: true })
      expect(await resolveProjectIconDataUrl(emptyDir)).toBeNull()

      const svgDir = join(temp, 'svg-repo')
      await mkdir(join(svgDir, 'assets'), { recursive: true })
      await writeFile(join(svgDir, 'assets', 'logo.svg'), '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"></svg>')
      const svgUrl = await resolveProjectIconDataUrl(svgDir)
      expect(svgUrl).toStartWith('data:image/svg+xml;base64,')

      const htmlDir = join(temp, 'html-repo')
      await mkdir(join(htmlDir, 'public', 'brand'), { recursive: true })
      await writeFile(join(htmlDir, 'index.html'), '<head><link rel="shortcut icon" href="/brand/app.svg"></head>')
      await writeFile(join(htmlDir, 'public', 'brand', 'app.svg'), '<svg xmlns="http://www.w3.org/2000/svg"></svg>')
      const htmlUrl = await resolveProjectIconDataUrl(htmlDir)
      expect(htmlUrl).toStartWith('data:image/svg+xml;base64,')

      const escapeDir = join(temp, 'escape-repo')
      const outsideSvg = join(temp, 'secret.svg')
      await mkdir(escapeDir, { recursive: true })
      await writeFile(outsideSvg, '<svg xmlns="http://www.w3.org/2000/svg"></svg>')
      await symlink(outsideSvg, join(escapeDir, 'favicon.svg'))
      expect(await resolveProjectIconDataUrl(escapeDir)).toBeNull()
    } finally {
      await rm(temp, { recursive: true, force: true })
    }
  })
})

describe('T3 Code Shell, Sidebar V2, Breadcrumb Header & Panels', () => {
  test('T3Sidebar renders Inbox cards, Pinned/Snoozed/Settled shelves, Undo banner, and icon footer', () => {
    const now = Date.now()
    const snap = makeSnapshot()
    const inbox = parseThreadInboxState({
      viewMode: 'inbox',
      pinnedIds: ['t-pin'],
      settledIds: ['t-done'],
      snoozed: { 't-later': now + 3_600_000 },
      snoozedOpen: true,
      settledOpen: true,
    })
    const html = renderToStaticMarkup(createElement(T3Sidebar, {
      snap,
      page: null,
      sections: [],
      allRows: [
        { id: 't-pin', key: 'k-pin', title: 'Pinned architecture spec', status: 'idle', age: '5m', current: false, turns: 2, cwd: '/workspace/Xerxes-Agents' },
        { id: 'thr-1', key: 'key-1', title: 'Match T3 Code UI', status: 'working', age: '12s', current: true, turns: 1, cwd: '/workspace/Xerxes-Agents' },
        { id: 't-later', key: 'k-later', title: 'Follow up tomorrow', status: 'idle', age: '2h', current: false, turns: 1, cwd: '/workspace/Xerxes-Agents' },
        { id: 't-done', key: 'k-done', title: 'Shipped release', status: 'idle', age: '1d', current: false, turns: 4, cwd: '/workspace/Xerxes-Agents' },
      ],
      filter: '',
      setFilter: () => {},
      inbox,
      undo: { threadId: 't-done', title: 'Shipped release', verb: 'Settled', previous: inbox },
      onUndo: () => {},
      onDismissUndo: () => {},
      onSetViewMode: () => {},
      onToggleProjectScope: () => {},
      onClearProjectScope: () => {},
      onTogglePin: () => {},
      onSettle: () => {},
      onSnooze: () => {},
      onToggleShelf: () => {},
      canGoBack: true,
      canGoForward: false,
      onGoBack: () => {},
      onGoForward: () => {},
      onOpen: () => {},
      sidebarToggle: createElement('button', { type: 'button' }, 'Toggle'),
      runtimeStatus: createElement('span', null, 'Connected'),
    }))

    for (const needle of [
      'data-app-sidebar="true"',
      'data-sidebar-view="inbox"',
      'Xerxes',
      'Code',
      'Pinned architecture spec',
      'Match T3 Code UI',
      'Follow up tomorrow',
      'Shipped release',
      'Snoozed',
      'Settled',
      'Settled “Shipped release”',
      'Undo',
      'aria-label="XERXES Supervisor"',
      'aria-label="Settings"',
    ]) {
      expect(html).toContain(needle)
    }
    expect(html.match(HEX_RE) ?? []).toEqual([])
  })

  test('T3Sidebar defaults to Machine + Project folder grouping across local and SSH hosts', () => {
    const snap = makeSnapshot()
    const inbox = parseThreadInboxState(null)
    expect(inbox.viewMode).toBe('folders')

    const sections = [
      {
        kind: 'here' as const,
        key: 'This Mac\u0000/workspace/Xerxes-Agents',
        host: 'This Mac',
        folder: 'Xerxes-Agents',
        label: 'This Mac · Xerxes-Agents',
        cwd: '/workspace/Xerxes-Agents',
        rows: [
          { id: 'thr-1', key: 'key-1', title: 'Match T3 Code UI', status: 'idle', age: 'now', current: true, kind: 'main' as const, turns: 2, messages: 4, cwd: '/workspace/Xerxes-Agents', untitled: false },
          { id: 'thr-2', key: 'key-2', title: 'Fix mac notifications', status: 'idle', age: '2d', current: false, kind: 'main' as const, turns: 1, messages: 2, cwd: '/workspace/Xerxes-Agents', untitled: false },
        ],
      },
      {
        kind: 'context' as const,
        key: 'v5p-128-dev\u0000/srv/prismcore',
        host: 'v5p-128-dev',
        folder: 'prismcore',
        label: 'v5p-128-dev · prismcore',
        cwd: '/srv/prismcore',
        context: { id: 9, label: 'v5p-128-dev · prismcore', workspace: '/srv/prismcore', workspacePath: '/srv/prismcore', remote: true, alias: 'v5p-128-dev', sessions: [] } as never,
        rows: [
          { id: 'rem-1', title: 'Swarm Bug Review', status: 'working', cwd: '/srv/prismcore' },
        ],
      },
    ]

    const grouped = groupSectionsByMachine(sections)
    expect(grouped.map(item => item.host)).toEqual(['This Mac', 'v5p-128-dev'])
    expect(grouped[0]!.isLocal).toBe(true)
    expect(grouped[1]!.isLocal).toBe(false)

    const html = renderToStaticMarkup(createElement(T3Sidebar, {
      snap,
      page: null,
      sections,
      allRows: [],
      filter: '',
      setFilter: () => {},
      inbox,
      undo: null,
      onUndo: () => {},
      onDismissUndo: () => {},
      onSetViewMode: () => {},
      onToggleProjectScope: () => {},
      onClearProjectScope: () => {},
      onTogglePin: () => {},
      onSettle: () => {},
      onSnooze: () => {},
      onToggleShelf: () => {},
      canGoBack: false,
      canGoForward: false,
      onGoBack: () => {},
      onGoForward: () => {},
      onOpen: () => {},
      sidebarToggle: createElement('button', { type: 'button' }, 'Toggle'),
      runtimeStatus: createElement('span', null, 'Connected'),
    }))

    for (const needle of [
      'data-sidebar-view="folders"',
      'data-machine-grouped="true"',
      'data-machine-group="This Mac"',
      'data-machine-group="v5p-128-dev"',
      'data-project-folder="/workspace/Xerxes-Agents"',
      'data-project-folder="/srv/prismcore"',
      'Match T3 Code UI',
      'Swarm Bug Review',
      './logo-128.png',
    ]) {
      expect(html).toContain(needle)
    }
    expect(html.match(HEX_RE) ?? []).toEqual([])
  })

  test('Shell under data-style="vibes" renders T3 Breadcrumb Header, PanelLayoutControls, restored Xerxes Welcome, unified Model Trigger and inline ThreadDetailsCard without bottom RepoBar', () => {
    stampStyle('vibes')
    const snap = makeSnapshot({
      blocks: [],
      turnActive: false,
      sessions: [
        { id: 'prev-1', key: 'prev-1', title: 'Earlier refactor', cwd: '/workspace/Xerxes-Agents', status: 'idle', turns: 2, messages: 4, age: '1h', untitled: false, current: false, kind: 'main' },
      ],
    })
    const html = renderToStaticMarkup(createElement(Shell, { snap }))

    for (const needle of [
      'data-app-sidebar="true"',
      'data-workspace-header="true"',
      'aria-label="Workspace breadcrumb"',
      'data-panel-layout-controls="true"',
      'aria-label="Toggle thread details"',
      'aria-label="Toggle terminal drawer"',
      'aria-label="Toggle right panel"',
      'welcome__wordmark',
      '>XERXES<',
      './logo-128.png',
      'Pick up where you left off',
      'data-t3-model-trigger="true"',
      'data-thread-details-card="true"',
      'New worktree',
    ]) {
      expect(html).toContain(needle)
    }
    expect(html).not.toContain('data-context-strip="true"')
    expect(html.match(HEX_RE) ?? []).toEqual([])
  })

  test('T3ModelPicker renders real Xerxes logo, compact trigger, and clickable 3-pane popover with zero inline hexes', async () => {
    expect(compactModelTriggerName('Gemini 4 Neon')).toBe('4 Neon')
    expect(compactModelTriggerName('Claude Sonnet 4.6')).toBe('Sonnet 4.6')
    expect(reasoningLevelLabel('xhigh')).toBe('Extra high')
    expect(reasoningLevelLabel('max')).toBe('Max')

    const snap = makeSnapshot({
      model: 'claude-sonnet-4-6',
      reasoningEffort: 'xhigh',
      reasoningLevels: [
        { effort: 'low', description: 'Quick answers' },
        { effort: 'medium', description: 'Balanced' },
        { effort: 'high', description: 'Deep reasoning' },
        { effort: 'xhigh', description: 'Extra high reasoning' },
        { effort: 'max', description: 'Maximum reasoning' },
      ],
      permissionMode: 'accept-all',
      models: [
        { id: 'claude-sonnet-4-6', label: 'Claude Sonnet 4.6', provider: 'Anthropic' },
        { id: 'claude-opus-4-6', label: 'Claude Opus 4.6', provider: 'Anthropic' },
      ],
    })

    const html = renderToStaticMarkup(createElement(T3ModelPicker, { snap, defaultOpen: true }))
    for (const needle of [
      'data-t3-model-trigger="true"',
      'data-t3-model-picker="true"',
      './logo-128.png',
      'aria-label="Favorites"',
      'placeholder="Search models..."',
      '⌘1',
      'Thinking Level',
      'Extra high',
      'High',
      'Max',
      'Claude Sonnet 4.6',
      'Full access',
    ]) {
      expect(html).toContain(needle)
    }
    expect(html.match(HEX_RE) ?? []).toEqual([])

    const pickerSrc = await Bun.file(new URL('../src/desktop/renderer/T3ModelPicker.tsx', import.meta.url)).text()
    expect(pickerSrc).toContain('store.loadReasoningLevels')
    const storeSrc = await Bun.file(new URL('../src/desktop/renderer/store.ts', import.meta.url)).text()
    expect(storeSrc).toContain('loadReasoningLevels(')

    const atelier = await Bun.file(new URL('../src/desktop/renderer/atelier.css', import.meta.url)).text()
    expect(atelier).toContain('.picker-layer>.acp-pop,.picker-layer>.acp-optpop,.picker-layer>.t3-mpop{pointer-events:auto;z-index:1}')
  })

  test('FilesChangedCard under data-style="vibes" renders T3 collapsible folder-tree card', () => {
    stampStyle('vibes')
    const html = renderToStaticMarkup(
      createElement(FilesChangedCard, {
        group: [
          { kind: 'checkpoint', id: 'cp-1', turn: 1, files: ['xerxes/src/a.ts', 'docs/readme.md'], createdAt: '2026-11-22T00:00:00Z' },
        ],
        changes: [
          { path: 'xerxes/src/a.ts', adds: 40, dels: 6, isNew: false, hunks: [], turn: 1 },
          { path: 'docs/readme.md', adds: 12, dels: 2, isNew: false, hunks: [], turn: 1 },
        ],
      }),
    )
    for (const needle of [
      't3-changed-card',
      '2 changed files',
      '+52',
      '-8',
      'Open diff',
      '>xerxes<',
      '>docs<',
    ]) {
      expect(html).toContain(needle)
    }
    expect(html.match(HEX_RE) ?? []).toEqual([])
  })

  test('Sidebar keeps classic unstyled fallback only when data-style is null (VS Code high contrast)', () => {
    stampStyle('vibes')
    const t3 = renderToStaticMarkup(createElement(Sidebar, { snap: makeSnapshot(), page: null }))
    expect(t3).toContain('data-app-sidebar="true"')
    expect(t3).toContain('t3-side__toolbar')

    stampStyle(null)
    const fallback = renderToStaticMarkup(createElement(Sidebar, { snap: makeSnapshot(), page: null }))
    expect(fallback).not.toContain('t3-side__toolbar')
    expect(fallback).toContain('sidebar-wordmark')
  })

  test('ThreadDetailsCard and TerminalDrawer render complete controls with zero hard-coded hexes', () => {
    const snap = makeSnapshot({
      changes: [{ path: 'xerxes/src/desktop/renderer/App.tsx', adds: 42, dels: 7, isNew: false, hunks: [], turn: 1 }],
      goal: 'Objective: Ship 1:1 T3 Code UI parity\nStatus: active',
      contextTokens: 50_000,
      contextMax: 200_000,
      turnCount: 3,
      costUsd: 0.14,
    })

    const detailsHtml = renderToStaticMarkup(createElement(ThreadDetailsCard, {
      snap,
      mode: 'inline',
      onClose: () => {},
      onNavigate: () => {},
      onOpenTerminal: () => {},
    }))
    for (const needle of [
      'data-thread-details-card="true"',
      'data-thread-details-mode="inline"',
      'This Mac',
      'Open in Files',
      'New worktree',
      'main',
      'Commit &amp; push',
      'Changes',
      '+42',
      '-7',
      'Ship 1:1 T3 Code UI parity',
    ]) {
      expect(detailsHtml).toContain(needle)
    }
    expect(detailsHtml.match(HEX_RE) ?? []).toEqual([])

    expect(clampTerminalDrawerHeight(undefined)).toBe(TERMINAL_DRAWER_DEFAULT_HEIGHT)
    expect(clampTerminalDrawerHeight(40)).toBe(TERMINAL_DRAWER_MIN_HEIGHT)
    expect(clampTerminalDrawerHeight(9999)).toBe(TERMINAL_DRAWER_MAX_HEIGHT)
    expect(clampTerminalDrawerHeight(320)).toBe(320)

    const drawerHtml = renderToStaticMarkup(createElement(TerminalDrawer, {
      snap,
      height: 300,
      onResize: () => {},
      onMoveToRightPanel: () => {},
      onClose: () => {},
    }))
    for (const needle of [
      'data-thread-terminal-drawer="true"',
      'aria-orientation="horizontal"',
      'aria-valuenow="300"',
      'Move terminal to right panel',
      'Close terminal drawer',
    ]) {
      expect(drawerHtml).toContain(needle)
    }
    expect(drawerHtml.match(HEX_RE) ?? []).toEqual([])
  })

  test('RightPanelLauncher renders Open a surface with all Xerxes surfaces and single-letter shortcuts', () => {
    const snap = makeSnapshot({
      changes: [{ path: 'xerxes/src/a.ts', adds: 10, dels: 2, isNew: false, hunks: [], turn: 1 }],
    })
    const html = renderToStaticMarkup(createElement(RightPanelLauncher, {
      snap,
      onPick: () => {},
      onClose: () => {},
    }))
    for (const needle of [
      'data-right-panel-launcher="true"',
      'Open a surface',
      'Browser',
      'Terminal',
      'Files',
      'Changes',
      'Side question',
      'Subagents',
      'Immortals',
      'Artifacts',
      'Background',
      'Outputs',
      'Activity',
      'Usage',
    ]) {
      expect(html).toContain(needle)
    }
    const shortcuts = new Set(XERXES_SURFACES.map(item => item.shortcut))
    expect(shortcuts.size).toBe(XERXES_SURFACES.length)
    expect(html.match(HEX_RE) ?? []).toEqual([])
  })

  test('built-in T3 Code themes ship in PALETTES and vibes.css defines T3 obsidian/paper tokens', async () => {
    const names = PALETTES.map(palette => palette.name)
    for (const id of ['t3-code', 't3-chat', 'grove', 'ocean', 't3-ember', 'iris']) {
      expect(names).toContain(id)
    }
    const css = await Bun.file(new URL('../src/desktop/renderer/vibes.css', import.meta.url)).text()
    expect(css).toContain('--x-screen:#000000')
    expect(css).toContain('--x-accent:#346bf1')
    expect(css).toContain('--x-screen:#fcfcfc')
    expect(css).toContain('--x-accent:#1b4ed8')
    expect(css).toContain('height:52px;min-height:52px')
    expect(css).toContain('@keyframes t3-live-shine')
    expect(css).toContain('.t3-term-drawer{')
    expect(css).toContain('.t3-details-card{')
    expect(css).toContain('border-radius:999px;background:var(--x-accent)')
    expect(css).toContain('button.desktop-rail__tabclose{position:absolute;top:-4px;right:-4px;z-index:2;display:inline-grid;place-items:center;flex:none;width:14px;min-width:14px;max-width:14px;height:14px;min-height:14px;max-height:14px')
  })

  test('icon-only Right Panel tabs & rehauled Subagents T3 Lucide icons render with zero inline hexes', () => {
    stampStyle('vibes')
    for (const panel of RAIL_PANELS) {
      expect(railT3IconOf(panel)).toBeTruthy()
    }
    expect(subagentRoleIcon({ id: '1', key: '1', title: 'Codebase Researcher', status: 'done', age: '1m', current: false, kind: 'subagent', turns: 1, messages: 2, cwd: '', untitled: false })).toBe('compass')
    expect(subagentRoleIcon({ id: '2', key: '2', title: 'Security Reviewer', status: 'done', age: '1m', current: false, kind: 'subagent', turns: 1, messages: 2, cwd: '', untitled: false })).toBe('shieldCheck')
    expect(subagentRoleIcon({ id: '3', key: '3', title: 'Architecture Planner', status: 'done', age: '1m', current: false, kind: 'subagent', turns: 1, messages: 2, cwd: '', untitled: false })).toBe('sparkles')
    expect(subagentRoleIcon({ id: '4', key: '4', title: 'Worker Alpha', status: 'done', age: '1m', current: false, kind: 'subagent', turns: 1, messages: 2, cwd: '', untitled: false })).toBe('bot')

    const snap = makeSnapshot()
    const railHtml = renderToStaticMarkup(createElement(DesktopRail, {
      panel: 'subagents',
      activeKey: 'subagents',
      tabs: [{ panel: 'terminal' }, { panel: 'background' }, { panel: 'files' }, { panel: 'subagents' }],
      snap,
      close: () => {},
      activityDetails: null,
    }))
    expect(railHtml).toContain('t3-icon')
    expect(railHtml).toContain('aria-label="Close Subagents"')
    expect(railHtml.match(HEX_RE) ?? []).toEqual([])

    const subagentsHtml = renderToStaticMarkup(createElement(SubagentsSection, {
      rows: [
        { id: 'sa-1', key: 'sa-1', title: 'Explore architecture', status: 'working', age: '12s', current: false, kind: 'subagent', turns: 1, messages: 1, cwd: '', untitled: false },
        { id: 'sa-2', key: 'sa-2', title: 'Verify tests', status: 'completed', age: '1m', current: false, kind: 'subagent', turns: 2, messages: 4, cwd: '', untitled: false },
        { id: 'sa-3', key: 'sa-3', title: 'Build failure check', status: 'failed', age: '2m', current: false, kind: 'subagent', turns: 1, messages: 2, cwd: '', untitled: false },
      ],
      sessionKey: 'key-1',
      online: true,
    }))
    expect(subagentsHtml).toContain('rsec__row--subagent')
    expect(subagentsHtml).toContain('rsec__agent-ico')
    expect(subagentsHtml).toContain('rsec__done-ico')
    expect(subagentsHtml).toContain('t3-icon')
    expect(subagentsHtml.match(HEX_RE) ?? []).toEqual([])
  })

  test('workspace removal deletes remembered entries without touching disk and filters hidden workspaces in T3Sidebar', async () => {
    const temp = await mkdtemp(join(tmpdir(), 'xerxes-ws-remove-'))
    try {
      const wsA = join(temp, 'xerxes')
      const wsB = join(temp, 'Xerxes-Agents')
      await mkdir(wsA, { recursive: true })
      await mkdir(wsB, { recursive: true })
      const file = join(temp, 'desktop.json')

      saveDesktopWorkspace(file, wsA)
      saveDesktopWorkspace(file, wsB)
      expect(loadDesktopWorkspaces(file)).toHaveLength(2)

      const afterRemove = removeDesktopWorkspace(file, wsA)
      expect(afterRemove).toHaveLength(1)
      expect(afterRemove[0]).toEndWith('Xerxes-Agents')

      const afterRestore = rememberDesktopWorkspace(file, wsA)
      expect(afterRestore).toHaveLength(2)
    } finally {
      await rm(temp, { recursive: true, force: true })
    }

    let state = parseThreadInboxState(null)
    state = hideWorkspaceInState(state, '/workspace/xerxes/')
    expect(state.hiddenWorkspaces).toEqual(['/workspace/xerxes'])

    const sections = [
      {
        kind: 'here' as const,
        key: 'This Mac\u0000/workspace/xerxes',
        host: 'This Mac',
        folder: 'xerxes',
        label: 'This Mac · xerxes',
        cwd: '/workspace/xerxes',
        rows: [],
      },
      {
        kind: 'here' as const,
        key: 'This Mac\u0000/workspace/Xerxes-Agents',
        host: 'This Mac',
        folder: 'Xerxes-Agents',
        label: 'This Mac · Xerxes-Agents',
        cwd: '/workspace/Xerxes-Agents',
        rows: [],
      },
    ]

    const html = renderToStaticMarkup(createElement(T3Sidebar, {
      snap: makeSnapshot({ workspaceDirectories: ['/workspace/xerxes', '/workspace/Xerxes-Agents'] }),
      page: null,
      sections,
      allRows: [],
      filter: '',
      setFilter: () => {},
      inbox: state,
      undo: null,
      onUndo: () => {},
      onDismissUndo: () => {},
      onSetViewMode: () => {},
      onToggleProjectScope: () => {},
      onClearProjectScope: () => {},
      onTogglePin: () => {},
      onSettle: () => {},
      onSnooze: () => {},
      onToggleShelf: () => {},
      canGoBack: false,
      canGoForward: false,
      onGoBack: () => {},
      onGoForward: () => {},
      onOpen: () => {},
      sidebarToggle: createElement('button', { type: 'button' }, 'Toggle'),
      runtimeStatus: createElement('span', null, 'Connected'),
    }))
    expect(html).not.toContain('data-project-folder="/workspace/xerxes"')
    expect(html).toContain('data-project-folder="/workspace/Xerxes-Agents"')
    expect(html).toContain('aria-haspopup="menu"')

    const unhidden = unhideWorkspaceInState(state, '/workspace/xerxes')
    expect(unhidden.hiddenWorkspaces).toEqual([])
  })

  test('every desktop page (Agents, Skills & tools, Scheduled jobs, Artifacts, XERXES Supervisor) renders cleanly and fixes header & empty-state layout in Vibes and Neo', async () => {
    stampStyle('neo')
    const snap = makeSnapshot()
    for (const page of ['agents', 'extensions', 'schedules', 'artifacts'] as const) {
      const html = renderToStaticMarkup(createElement(DesktopPage, { panel: page, snap }))
      expect(html).toContain('class="desktop-page__bar"')
      expect(html.match(HEX_RE) ?? []).toEqual([])
    }
    const supervisorConnectHtml = renderToStaticMarkup(createElement(ConnectHub, { onOpenSettings: () => {} }))
    expect(supervisorConnectHtml).toContain('Connect a Hub')
    expect(supervisorConnectHtml.match(HEX_RE) ?? []).toEqual([])

    const [vibesCss, neoCss] = await Promise.all([
      Bun.file(new URL('../src/desktop/renderer/vibes.css', import.meta.url)).text(),
      Bun.file(new URL('../src/desktop/renderer/neo.css', import.meta.url)).text(),
    ])
    // Page breadcrumb titles must never flip before the project logo via atelier.css's order:-1
    expect(vibesCss).toContain('.t3-breadcrumb .chat__title{order:0;')
    // Agents page empty card & actions must center cleanly with a flat background
    expect(vibesCss).toContain('.desktop-page .specialists .agents-empty-layout{width:100%;max-width:560px;margin:36px auto 0;')
    expect(vibesCss).toContain('.desktop-page .specialists>.studio-actions.agents-empty-actions{max-width:560px;margin:16px auto 0;')
    // Neo Supervisor dock has no leading + button, so its ❯ prompt aligns cleanly at 14px
    expect(neoCss).toContain('.composer-wrap.supervisor__dock>.composer-dock>.composer::before{padding-left:14px}')
  })
})
