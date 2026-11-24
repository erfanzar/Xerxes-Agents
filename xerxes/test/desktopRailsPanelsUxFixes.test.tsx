// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { BROWSER_OVERLAY_SELECTOR } from '../src/desktop/renderer/BrowserPanel.js'
import { DesktopRail, FilesPanel, WorkspacePanel } from '../src/desktop/renderer/DesktopPanels.js'
import { FileActions } from '../src/desktop/renderer/FilePreview.js'
import { primaryAction } from '../src/desktop/renderer/GitPanel.js'
import { GoalInspector } from '../src/desktop/renderer/GoalInspector.js'
import { ArtifactsSection, SubagentsSection, type ArtifactRow } from '../src/desktop/renderer/RailSections.js'
import {
  RightPanelLauncher,
  isEditableEventTarget,
} from '../src/desktop/renderer/RightPanelLauncher.js'
import {
  SIDE_QUESTION_SCROLL_PIN_THRESHOLD,
  isSideQuestionNearBottom,
} from '../src/desktop/renderer/SideQuestion.js'
import type { Snapshot } from '../src/desktop/renderer/store.js'
import {
  SwarmBoard,
  shouldEnableSwarmFeedScroll,
  swarmLiveBadgeState,
} from '../src/desktop/renderer/SwarmBoard.js'
import { SwarmGraph } from '../src/desktop/renderer/SwarmGraph.js'
import type { SwarmBoardMessage, SwarmStatusSnapshot } from '../src/desktop/renderer/swarmState.js'

const rendererDir = join(import.meta.dir, '../src/desktop/renderer')

function makeSnap(overrides: Partial<Snapshot> = {}): Snapshot {
  return {
    connection: 'online',
    noWorkspace: false,
    cwd: '/repo/demo',
    model: 'claude-opus-4-6',
    currentAgentPreset: 'default',
    agentPresets: [],
    agentPresetsError: null,
    channels: [],
    channelsAvailable: false,
    channelsConfigured: false,
    terminals: [],
    terminalsLoading: false,
    terminalsError: null,
    searchOpen: false,
    searchResults: [],
    searchStats: null,
    searchSearching: false,
    searchError: null,
    branch: 'main',
    daemonWarning: null,
    costUsd: null,
    costBasis: null,
    mcpStatus: {},
    models: [
      { id: 'claude-opus-4-6', provider: 'anthropic', label: 'Claude Opus 4.6' },
      { id: 'claude-sonnet-4-6', provider: 'anthropic', label: 'Claude Sonnet 4.6' },
    ],
    contextTokens: null,
    computerUseWanted: false,
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
    cacheReadTokens: null,
    cacheWriteTokens: null,
    sessions: [],
    live: [],
    fleet: [],
    skillSuggestions: [],
    creatorTrace: [],
    backgroundJobs: [],
    currentId: 'sess-ux-1',
    sessionOpenRevision: 0,
    currentTitle: 'UX Fixes',
    sessionKey: 'desktop-ux-1',
    goal: '',
    approval: null,
    question: null,
    interactionHidden: false,
    planMode: false,
    turnActive: false,
    turnFailed: false,
    turnSeconds: 0,
    blocks: [],
    error: null,
    tab: 'activity',
    turnCount: 1,
    queue: [],
    changes: [],
    changesKept: false,
    todos: null,
    plan: null,
    log: [],
    failed: null,
    stopArmed: false,
    settingsOpen: false,
    settingsTab: 'general',
    paletteOpen: false,
    pickerOpen: false,
    modelMenuOpen: false,
    appUpdate: null,
    providerRelay: null,
    delegationMode: 'auto',
    delegationMenuOpen: false,
    contextMenuOpen: false,
    contextBreakdown: null,
    contextBreakdownLoading: false,
    reasoningPickerOpen: false,
    reasoningEffort: '',
    reasoningLevels: [],
    reasoningDefault: '',
    reasoningNote: '',
    reasoningLoading: false,
    wsMenuOpen: false,
    workspaceBusy: false,
    workspaceError: null,
    sessionMenu: null,
    streamThinking: true,
    taskModalOpen: false,
    providers: [],
    providerError: '',
    navigating: null,
    providersLoading: false,
    modelsLoading: false,
    modelsError: null,
    providerSwitching: null,
    providerSwitchError: null,
    runMode: null,
    runModeError: '',
    runModeSaving: false,
    localSetup: { status: null, job: null, error: '', loading: false },
    providerModels: {},
    providerModelLoading: [],
    providerModelWarnings: {},
    providerTypes: [],
    commands: [],
    permissionMode: 'auto',
    permissionUpdating: false,
    permissionError: null,
    snippets: {},
    swarm: null,
    swarmMessages: [],
    swarmLoading: false,
    swarmError: null,
    swarmOpen: false,
    swarmOpenRevision: 0,
    ...overrides,
  }
}

const sampleSwarmStatus: SwarmStatusSnapshot = {
  active: true,
  session_id: 'sess-ux-1',
  title: 'Parallel Audit',
  board: 'local',
  open_board: '',
  head: 3,
  overview: null,
  channels: [{ name: 'main', topic: '', created_by: 'coordinator', archived: false, subscriber_count: 3 }],
  members: [
    {
      name: 'coordinator',
      role: 'coordinator',
      model: 'claude-opus-4-6',
      model_display_name: 'Claude Opus 4.6',
      workspace: 'inherit',
      channels: ['*'],
      state: 'waiting',
      killed: false,
      last_read_seq: 3,
      last_post_seq: 1,
    },
    {
      name: 'worker-a',
      group: 'backend',
      role: 'worker',
      added_by: 'coordinator',
      model: 'claude-sonnet-4-6',
      model_display_name: 'Claude Sonnet 4.6',
      workspace: 'share',
      channels: ['main'],
      state: 'working',
      killed: false,
      last_read_seq: 3,
      last_post_seq: 2,
      current_activity: 'Editing api.ts',
    },
    {
      name: 'worker-b',
      group: 'frontend',
      role: 'worker',
      added_by: 'coordinator',
      model: 'claude-sonnet-4-6',
      model_display_name: 'Claude Sonnet 4.6',
      workspace: 'share',
      channels: ['main'],
      state: 'waiting',
      killed: false,
      last_read_seq: 3,
      last_post_seq: 3,
    },
  ],
}

const sampleSwarmMessages: readonly SwarmBoardMessage[] = [
  {
    seq: 1,
    agent: 'coordinator',
    channel: 'main',
    text: '@worker-a Own api.ts, @worker-b Own ui.tsx',
    mentions: ['worker-a', 'worker-b'],
    channel_mentions: [],
    broadcast: false,
    time: '10:00:00',
  },
  {
    seq: 2,
    agent: 'worker-a',
    channel: 'main',
    text: '@coordinator Working on api.ts',
    mentions: ['coordinator'],
    channel_mentions: [],
    broadcast: false,
    time: '10:01:00',
  },
  {
    seq: 3,
    agent: 'worker-b',
    channel: 'main',
    text: '@coordinator [done] Updated ui.tsx',
    mentions: ['coordinator'],
    channel_mentions: [],
    broadcast: false,
    time: '10:02:00',
  },
]

describe('Desktop Right-Rail, Workspace Panels, Git, Files, Browser, Swarm & SideQuestion UX fixes', () => {
  test('1 & 2. WorkspacePanel removes alien ws-gemini/Material-3 overrides and eliminates re-entrant SSH saveMachine deadlock', () => {
    const panelsSrc = readFileSync(join(rendererDir, 'DesktopPanels.tsx'), 'utf8')
    expect(panelsSrc).not.toContain('ws-gemini')
    expect(panelsSrc).not.toContain('ws-btn-primary')
    expect(panelsSrc).not.toContain('ws-btn-tonal')
    expect(panelsSrc).not.toContain('ws-btn-ghost')

    // Unwrapped connectNow helper prevents request.run -> saveMachine(true) -> connect -> request.run deadlock
    const connectNowIdx = panelsSrc.indexOf('const connectNow = async (machine: RpcRecord)')
    const connectIdx = panelsSrc.indexOf('const connect = (machine: RpcRecord) => request.run(() => connectNow(machine))')
    const saveIdx = panelsSrc.indexOf('const saveMachine = async (andConnect: boolean)')
    expect(connectNowIdx).toBeGreaterThan(-1)
    expect(connectIdx).toBeGreaterThan(connectNowIdx)
    expect(saveIdx).toBeGreaterThan(connectIdx)
    const saveBody = panelsSrc.slice(saveIdx, saveIdx + 700)
    expect(saveBody).toContain('await connectNow(machine)')
    expect(saveBody).not.toContain('await connect(machine)')

    // Rendered WorkspacePanel uses standard Xerxes .seg, .btn, and .workspace-panel without ws-gemini
    const html = renderToStaticMarkup(createElement(WorkspacePanel, { snap: makeSnap() }))
    expect(html).toContain('class="studio-form workspace-panel"')
    expect(html).toContain('class="seg ws-tabs"')
    expect(html).toContain('class="btn"')
    expect(html).not.toContain('ws-gemini')
    expect(html).not.toContain('ws-btn-primary')
  })

  test('3. DesktopRail marks item tabs (files:*, docs:*) with data-item-tab and data-has-label for crisp tab identity', () => {
    const html = renderToStaticMarkup(
      createElement(DesktopRail, {
        panel: 'files',
        snap: makeSnap(),
        close: () => undefined,
        activityDetails: null,
        activeKey: 'files:src/index.ts',
        tabs: [
          { panel: 'activity' },
          { panel: 'files', item: 'src/index.ts', title: 'index.ts' },
          { panel: 'docs', item: 'art-1', title: 'Design Spec' },
        ],
      }),
    )
    expect(html).toContain('data-item-tab="true"')
    expect(html).toContain('data-has-label="true"')
    expect(html).toContain('class="desktop-rail__tablabel">index.ts</span>')
    expect(html).toContain('class="desktop-rail__tablabel">Design Spec</span>')
    expect(html).toContain('aria-label="Close index.ts"')
    expect(html).toContain('aria-label="Close Design Spec"')
  })

  test('4. FilesPanel fixes narrow tree trap, keeps selected file during search, dismisses ··· menu safely, and CodeEditor stops Escape propagation', () => {
    const panelsSrc = readFileSync(join(rendererDir, 'DesktopPanels.tsx'), 'utf8')
    const codeEditorSrc = readFileSync(join(rendererDir, 'CodeEditor.tsx'), 'utf8')
    const filePreviewSrc = readFileSync(join(rendererDir, 'FilePreview.tsx'), 'utf8')

    // pick() always resets treeToggled even when path === selected
    const pickIdx = panelsSrc.indexOf('const pick = (path: string): void => {')
    expect(pickIdx).toBeGreaterThan(-1)
    expect(panelsSrc.slice(pickIdx, pickIdx + 180)).toContain('setTreeToggled(false)')
    expect(panelsSrc).toContain('Back to {selectedName}')

    // Search input no longer nukes selected or remounts tree per keystroke
    expect(panelsSrc).not.toContain("setSelected(''); setNeedle(")
    expect(panelsSrc).not.toContain('key={snap.cwd + snap.sessionKey + needle}')
    expect(panelsSrc).toContain("key={snap.cwd + snap.sessionKey + ':search'}")

    // ··· menu has Escape + outside-click dismissal and lifts FileActions errors out of menu unmount
    expect(panelsSrc).toContain("document.addEventListener('mousedown', onOutside)")
    expect(panelsSrc).toContain("window.addEventListener('keydown', onEscape, true)")
    expect(panelsSrc).toContain('onError={setActionError}')
    expect(filePreviewSrc).toContain('onError?: (message: string) => void')

    // CodeEditor stops Escape propagation (both when dismissing completions and when focused)
    expect(codeEditorSrc).toContain("if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); return }")
    expect(codeEditorSrc).toContain('// Never let Escape inside the code editor bubble up to GlobalKeys turn-stop.')

    // SSR check for FilesPanel & FileActions
    const filesHtml = renderToStaticMarkup(createElement(FilesPanel, { snap: makeSnap(), close: () => undefined }))
    expect(filesHtml).toContain('aria-label="Filter workspace files by path"')
    const actionsHtml = renderToStaticMarkup(createElement(FileActions, { path: 'a.ts', cwd: '/repo', remote: false, onAction: () => undefined }))
    expect(actionsHtml).toContain('role="menuitem"')
  })

  test('5. GitPanel commits staged-only when staged > 0, expands narrow diff on file select, dismisses BranchList cleanly, and avoids window.confirm', () => {
    const repo = { upstream: 'origin/main', ahead: 0, behind: 0, branch: 'main', hasHead: true, detached: false }
    expect(primaryAction({ ...repo, counts: { staged: 3 } }, 7)).toMatchObject({
      kind: 'commit',
      stagedOnly: true,
      label: 'Commit staged (3)',
    })
    expect(primaryAction({ ...repo, counts: { staged: 0 } }, 4)).toMatchObject({
      kind: 'commit',
      stagedOnly: false,
      label: 'Commit all',
    })

    const gitSrc = readFileSync(join(rendererDir, 'GitPanel.tsx'), 'utf8')
    // Never hardcodes { message, all: true } when staged changes exist
    expect(gitSrc).not.toContain("git.commit', { message, all: true }")
    expect(gitSrc).toContain("git.commit', { message, all: commitAll }")
    expect(gitSrc).toContain('hasBothStagedAndUnstaged')
    // Narrow diff expansion in choose()
    expect(gitSrc).toContain('if (diffHiddenByStyle || narrowContainer) onExpand()')
    // BranchList Escape, outside-click, and explicit Close button
    expect(gitSrc).toContain('aria-label="Close branches"')
    expect(gitSrc).toContain("window.addEventListener('keydown', onEscape, true)")
    expect(gitSrc).toContain("document.addEventListener('mousedown', onOutside)")
    // Non-blocking inline discard confirm instead of window.confirm
    expect(gitSrc).not.toContain('window.confirm(')
    expect(gitSrc).toContain('aria-label="Confirm discard changes"')
  })

  test('6. SideQuestion uses 48px near-bottom pin detection so streaming does not yank scrolled-up users', () => {
    expect(SIDE_QUESTION_SCROLL_PIN_THRESHOLD).toBe(48)
    expect(isSideQuestionNearBottom({ scrollTop: 0, clientHeight: 300, scrollHeight: 280 })).toBe(true)
    expect(isSideQuestionNearBottom({ scrollTop: 552, clientHeight: 400, scrollHeight: 1000 })).toBe(true)
    expect(isSideQuestionNearBottom({ scrollTop: 551, clientHeight: 400, scrollHeight: 1000 })).toBe(false)
    expect(isSideQuestionNearBottom({ scrollTop: 120, clientHeight: 400, scrollHeight: 1000 })).toBe(false)

    const sideSrc = readFileSync(join(rendererDir, 'SideQuestion.tsx'), 'utf8')
    expect(sideSrc).toContain('isSideQuestionNearBottom(el, SIDE_QUESTION_SCROLL_PIN_THRESHOLD)')
    expect(sideSrc).toContain('Jump to latest side question output')
  })

  test('7. RightPanelLauncher guards shortcuts behind overlays/inputs, handles Escape cleanly, and omits maximize button', () => {
    const html = renderToStaticMarkup(
      createElement(RightPanelLauncher, {
        snap: makeSnap(),
        onPick: () => undefined,
        onClose: () => undefined,
        filesExpanded: false,
        toggleFilesExpanded: () => undefined,
      }),
    )
    expect(html).toContain('aria-label="Close right panel"')
    expect(html).not.toContain('aria-label="Expand right panel"')
    expect(html).not.toContain('aria-label="Restore conversation"')

    expect(isEditableEventTarget({ tagName: 'INPUT', isContentEditable: false } as unknown as EventTarget)).toBe(true)
    expect(isEditableEventTarget({ tagName: 'TEXTAREA', isContentEditable: false } as unknown as EventTarget)).toBe(true)
    expect(isEditableEventTarget({ tagName: 'DIV', isContentEditable: true } as unknown as EventTarget)).toBe(true)
    expect(isEditableEventTarget({ tagName: 'BUTTON', isContentEditable: false } as unknown as EventTarget)).toBe(false)

    const launcherSrc = readFileSync(join(rendererDir, 'RightPanelLauncher.tsx'), 'utf8')
    expect(launcherSrc).toContain('hasBlockingLauncherOverlay(rootRef.current)')
    expect(launcherSrc).toContain("if (event.key === 'Escape'")
    expect(launcherSrc).toContain('event.stopPropagation()')
  })

  test('8. SwarmBoard confirms permanent stops, fixes split-view feedScroll & modelAnchorRef, reports real Live/Paused/Offline/Idle state, deduplicates Overview, and SwarmGraph stays legible with clean tooltip/legend', () => {
    // Live badge reflects real swarm & connection state
    expect(swarmLiveBadgeState(true, sampleSwarmStatus, { total: 3, working: 1, waiting: 2, stopped: 0, killed: 0 })).toMatchObject({ label: 'Live', live: true })
    expect(swarmLiveBadgeState(false, sampleSwarmStatus, { total: 3, working: 1, waiting: 2, stopped: 0, killed: 0 })).toMatchObject({ label: 'Offline', live: false })
    expect(swarmLiveBadgeState(true, { active: false }, { total: 2, working: 0, waiting: 0, stopped: 2, killed: 0 })).toMatchObject({ label: 'Paused', live: false })
    expect(swarmLiveBadgeState(true, null, { total: 0, working: 0, waiting: 0, stopped: 0, killed: 0 })).toMatchObject({ label: 'Idle', live: false })

    // Split-view feedScroll remains enabled when Members tab is selected at >=780px
    expect(shouldEnableSwarmFeedScroll('members', false, false)).toBe(false)
    expect(shouldEnableSwarmFeedScroll('members', true, false)).toBe(true)
    expect(shouldEnableSwarmFeedScroll('board', true, false)).toBe(true)
    expect(shouldEnableSwarmFeedScroll('board', true, true)).toBe(false)

    // Armed inline confirmation renders Confirm stop? / Cancel
    const armedAllHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleSwarmStatus,
        messages: sampleSwarmMessages,
        initialArmedKillTarget: 'all',
      }),
    )
    expect(armedAllHtml).toContain('aria-label="Confirm stop all Immortals workers"')
    expect(armedAllHtml).toContain('Confirm stop?')
    expect(armedAllHtml).toContain('aria-label="Cancel stop all Immortals workers"')

    const armedWorkerHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleSwarmStatus,
        messages: sampleSwarmMessages,
        initialTab: 'members',
        initialArmedKillTarget: 'worker-a',
      }),
    )
    expect(armedWorkerHtml).toContain('aria-label="Confirm stop worker-a"')
    expect(armedWorkerHtml).toContain('aria-label="Cancel stop worker-a"')

    // Single model switcher per waiting member (no duplicate Switch model button clobbering modelAnchorRef)
    const switchMatches = armedWorkerHtml.match(/aria-label="Switch model for worker-b"/g) ?? []
    expect(switchMatches).toHaveLength(1)

    // Offline / Idle badge text is never hardcoded as Live
    const idleHtml = renderToStaticMarkup(
      createElement(SwarmBoard, { status: null, messages: [], online: false }),
    )
    expect(idleHtml).toContain('>Offline</span>')
    expect(idleHtml).not.toContain('>Live</span>')

    // Split view (>= 780px) unifies Board & Members tab cleanly
    const splitHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleSwarmStatus,
        messages: sampleSwarmMessages,
        initialSplitView: true,
        initialTab: 'members',
      }),
    )
    expect(splitHtml).toContain('data-split="true"')
    expect(splitHtml).toContain('>Board &amp; Members</span>')

    // Overview deduplication: running worker-a appears once as a task row (in Running),
    // and completed worker-b appears once as a task row (in Threads, not duplicated in Progress card)
    const overviewHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleSwarmStatus,
        messages: sampleSwarmMessages,
        initialTab: 'overview',
      }),
    )
    const overviewStart = overviewHtml.indexOf('aria-label="Narrator overview"')
    const overviewEnd = overviewHtml.indexOf('class="swarm-board__main"')
    const overviewSlice = overviewHtml.slice(overviewStart, overviewEnd)
    expect(overviewSlice.match(/class="swarm-ov-task-row(?:"| )/g) ?? []).toHaveLength(2)

    // SwarmGraph: readable min-width floor, Mentions tooltip copy ("Click to trace"), and grey SVG dashed swatch (no ASCII "-->")
    const mapHtml = renderToStaticMarkup(
      createElement(SwarmGraph, {
        status: sampleSwarmStatus,
        messages: sampleSwarmMessages,
        initialMode: 'battlefield',
      }),
    )
    expect(mapHtml).toContain('min-width:560px')
    expect(mapHtml).not.toContain('--&gt;')
    expect(mapHtml).toContain('stroke="var(--x-meta)"')

    const graphSrc = readFileSync(join(rendererDir, 'SwarmGraph.tsx'), 'utf8')
    expect(graphSrc).toContain("Click to trace")
  })

  test('9. BrowserPanel tracks editing in a ref and expands BROWSER_OVERLAY_SELECTOR for all floating surfaces', () => {
    for (const selector of [
      '.findbar',
      '.artifact-toast',
      '.t3-details-card',
      '.notice-stack',
      '.workspace-notice',
      '.acp-pop',
      '.combo-pop',
      '.swarm-model-picker',
    ]) {
      expect(BROWSER_OVERLAY_SELECTOR).toContain(selector)
    }

    const browserSrc = readFileSync(join(rendererDir, 'BrowserPanel.tsx'), 'utf8')
    expect(browserSrc).toContain('const editingRef = useRef(false)')
    expect(browserSrc).toContain('editingRef.current ? current : displayUrl(next.url)')
  })

  test('10. RailSections omits duplicate markdown bar chips, guards dirty edits & empty search; SearchPanel scrolls active row; GoalInspector flattens criteria with .btn controls', () => {
    const mdRow: ArtifactRow = {
      id: 'art-md-1',
      title: 'Architecture Notes',
      kind: 'markdown',
      size: 2048,
      createdAt: '2026-11-24T00:00:00.000Z',
      updatedAt: '2026-11-24T01:00:00.000Z',
      pinned: false,
      version: 2,
    }
    const mdDetailHtml = renderToStaticMarkup(
      createElement(ArtifactsSection, {
        sessionKey: 'sess-1',
        online: true,
        target: { id: mdRow.id, count: 1 },
        initialDocument: { row: mdRow, content: '# Architecture Notes\n\nClean body.' },
      }),
    )
    // Duplicate .rsec__title / .rsec__kind-pill / .rsec__ver-pill omitted from .rsec__bar while .artdoc__head is visible
    expect(mdDetailHtml).not.toContain('class="rsec__title"')
    expect(mdDetailHtml).not.toContain('class="rsec__kind-pill"')
    expect(mdDetailHtml).not.toContain('class="rsec__ver-pill"')
    expect(mdDetailHtml).toContain('<h1 class="artdoc__title">Architecture Notes</h1>')

    const railSectionsSrc = readFileSync(join(rendererDir, 'RailSections.tsx'), 'utf8')
    expect(railSectionsSrc).toContain('Discard unsaved changes to this artifact?')
    expect(railSectionsSrc).toContain('No matching artifacts')
    expect(railSectionsSrc).toContain('No matching subagents')
    expect(railSectionsSrc).toContain('Clear filter')

    const subHtml = renderToStaticMarkup(createElement(SubagentsSection, { rows: [] }))
    expect(subHtml).toContain('No subagents in this task yet.')

    const searchSrc = readFileSync(join(rendererDir, 'SearchPanel.tsx'), 'utf8')
    expect(searchSrc).toContain("querySelector<HTMLElement>('.srow.is-sel')")
    expect(searchSrc).toContain("scrollIntoView?.({ block: 'nearest' })")

    const goalHtml = renderToStaticMarkup(createElement(GoalInspector, { snap: makeSnap() }))
    expect(goalHtml).toContain('class="btn"')
    expect(goalHtml).not.toContain('<br/>')
    const goalSrc = readFileSync(join(rendererDir, 'GoalInspector.tsx'), 'utf8')
    expect(goalSrc).not.toContain('<br/>')
    expect(goalSrc).not.toContain('<details className="goal-criterion"')
    expect(goalSrc).toContain('<ul className="goal-criteria">')
    expect(goalSrc).toContain('className="btn btn--solid"')
  })
})
