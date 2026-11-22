// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { renderToStaticMarkup } from 'react-dom/server'
import { createSupervisorSlackClient } from '../src/client/supervisorSlack.js'
import { VibesSideNav } from '../src/desktop/renderer/App.js'
import { DesktopRail, RAIL_PANELS } from '../src/desktop/renderer/DesktopPanels.js'
import { PANEL_SIZES } from '../src/desktop/renderer/layout.js'
import { SideQuestionPanel } from '../src/desktop/renderer/SideQuestion.js'
import type { Snapshot } from '../src/desktop/renderer/store.js'
import { SwarmBoard } from '../src/desktop/renderer/SwarmBoard.js'
import { SlackAppSetup, SupervisorSlack } from '../src/desktop/renderer/SupervisorSlack.js'
import { sidebarSections, type GroupableRow, type WorkspaceGroup } from '../src/desktop/renderer/workspaceGroups.js'
import type { WorkspaceContext } from '../src/desktop/main/contextNavigation.js'

const rendererDir = join(import.meta.dir, '../src/desktop/renderer')
const atelierCss = readFileSync(join(rendererDir, 'atelier.css'), 'utf8')
const layoutCss = readFileSync(join(rendererDir, 'layout.css'), 'utf8')
const vibesCss = readFileSync(join(rendererDir, 'vibes.css'), 'utf8')
const neoCss = readFileSync(join(rendererDir, 'neo.css'), 'utf8')

const minimalSnapshot: Snapshot = {
  connection: 'online',
  noWorkspace: false,
  cwd: '/workspace/xerxes',
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
  models: [{ id: 'claude-opus-4-6', provider: 'Anthropic', label: 'Claude Opus 4.6' }],
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
  currentId: 'sess-audit-1',
  sessionOpenRevision: 0,
  currentTitle: 'Desktop Audit',
  sessionKey: 'audit-key-1',
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
  turnCount: 0,
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
}

describe('Desktop GUI visual and CSS regression audit', () => {
  test('SwarmBoard buttons and interactive tokens outrank .studio-sheet button reset', () => {
    expect(atelierCss).toContain('.studio-sheet button{')

    const requiredScopedSelectors = [
      '.swarm-board :is(.swarm-btn,button.swarm-btn)',
      '.swarm-board :is(.swarm-btn--primary,button.swarm-btn--primary)',
      '.swarm-board :is(.swarm-btn--danger,button.swarm-btn--danger)',
      '.swarm-board :is(.swarm-btn--ghost,button.swarm-btn--ghost)',
      '.swarm-board :is(.swarm-btn--xs,button.swarm-btn--xs)',
      '.swarm-board :is(.swarm-tab,button.swarm-tab)',
      '.swarm-board :is(.swarm-channel,button.swarm-channel)',
      '.swarm-board :is(.swarm-member__name,button.swarm-member__name)',
      '.swarm-board button.swarm-msg__seq',
      '.swarm-board button.swarm-msg__author',
      '.swarm-board button.swarm-msg__channel',
      '.swarm-board button.swarm-token--mention',
      '.swarm-board button.swarm-token--channel',
      '.swarm-board button.swarm-token--seq',
      '.swarm-board :is(.swarm-mentions-pop__row,button.swarm-mentions-pop__row)',
    ]

    for (const selector of requiredScopedSelectors) {
      expect(atelierCss).toContain(selector)
    }
  })

  test('SwarmBoard stays single-column in the side rail (<780px) and splits to 2 columns when expanded (>=780px)', () => {
    expect(PANEL_SIZES.swarm.width).toBeLessThan(780)
    expect(PANEL_SIZES.btw).toEqual({ width: 420, min: 320 })
    expect(atelierCss).toContain('@container (min-width:780px){')
    expect(atelierCss).toContain('.desktop-rail>.studio-sheet-content>.swarm-board')
    expect(atelierCss).toContain('.desktop-rail>.studio-sheet-content>.side-question')
  })

  test('SupervisorSlack uses valid flex layout for token fields and preserves 100% of setup UX', () => {
    expect(atelierCss).not.toContain('repeat(auto-fit,minmax(min(220px,100%),1fr)) auto')
    expect(atelierCss).toContain('.xslack__fields{display:flex;flex-wrap:wrap;gap:6px;width:min(640px,100%);align-items:center}')
    expect(atelierCss).toContain('.xslack__fields input{flex:1 1 180px;min-width:0;width:auto}')

    const redirectUrl = 'https://hub.xerxes.ts.net/api/hub/v1/slack/oauth/callback'
    const setupHtml = renderToStaticMarkup(
      <SlackAppSetup
        redirectUrl={redirectUrl}
        disabled={false}
        openApp={() => {}}
        openTokenPage={() => {}}
        onCreate={() => {}}
        onSave={() => {}}
      />,
    )
    expect(setupHtml).toContain('Open Slack token page')
    expect(setupHtml).toContain('Slack configuration token')
    expect(setupHtml).toContain('Set up automatically')
    expect(setupHtml).toContain('Set it up by hand instead')

    const client = createSupervisorSlackClient({
      allowed: () => true,
      request: async () => ({
        configured: true,
        browser_sign_in: false,
        app: 'none',
        redirect_url: redirectUrl,
        connection: null,
        contacts: [],
      }),
    })
    const panelHtml = renderToStaticMarkup(
      <SupervisorSlack client={client} openOAuth={async () => false} />,
    )
    expect(panelHtml).toContain('Slack')
  })

  test('opened tool command line reserves left gutter so $ prompt never overlaps command text', () => {
    expect(layoutCss).toContain(':root:not(#layout) .atelier .activity-group__body .execution__command{margin:0 0 6px;padding:0 0 0 16px;border:0;font:400 12px/1.6 var(--mono)}')
    expect(layoutCss).toContain(':root:not(#layout) .atelier .activity-group__body .execution__command::before{left:0}')
  })

  test('Neo composer prompt clears the + attach button and preserves round Swarm/Sections status dots', () => {
    expect(neoCss).toContain(':root[data-style=\'neo\']:not(#neo) .atelier .composer::before{content:\'❯\';font:700 14px/1.45 var(--mono);color:var(--x-accent)}')
    expect(neoCss).toContain(':root[data-style=\'neo\']:not(#neo) .atelier .composer-wrap:not(.agent-dock)>.composer-dock>.composer::before{grid-column:1;grid-row:5;align-self:start;justify-self:start;padding:11px 0 0 38px;pointer-events:none;z-index:1}')
    expect(neoCss).toContain(':root[data-style=\'neo\']:not(#neo) .atelier .composer-wrap:not(.agent-dock) .composer__input{padding-left:54px}')
    expect(neoCss).toContain('.swarm-board__dot,.swarm-member__dot,.sections-menu__dot')
  })

  test('Vibes and Neo right-window tabs stay icon-only with accessible hover-dismiss × close buttons and More menu surfaces Swarm & Side question', () => {
    expect(vibesCss).toMatch(/@media\s*\(min-width:761px\)\s*\{\s*:root:is\(\[data-style='vibes'\],\[data-style='neo'\]\):not\(#vibes\)\s+\.atelier\s+\.app__body:has\(>\.floatrail\)\s+\.chat__head\{padding-right:268px\}/)
    expect(vibesCss).not.toContain('button.desktop-rail__tabclose{display:none}')
    expect(vibesCss).toContain('.desktop-rail__tab>button.desktop-rail__tabmain>span{position:absolute;width:1px;height:1px;')
    expect(vibesCss).toContain('.desktop-rail__tab:is(:hover,:focus-within)>button.desktop-rail__tabclose{opacity:1;pointer-events:auto}')

    expect(RAIL_PANELS).toContain('swarm')
    expect(RAIL_PANELS).toContain('btw')

    const railHtml = renderToStaticMarkup(
      <DesktopRail
        panel="swarm"
        activeKey="swarm"
        tabs={[
          { panel: 'activity' },
          { panel: 'files' },
          { panel: 'subagents' },
          { panel: 'swarm' },
          { panel: 'btw' },
        ]}
        snap={minimalSnapshot}
        close={() => {}}
        activityDetails={null}
      />,
    )
    expect(railHtml).toContain('class="desktop-rail__tabclose"')
    expect(railHtml).toContain('aria-label="Close Activity"')
    expect(railHtml).toContain('aria-label="Close Files"')
    expect(railHtml).toContain('aria-label="Close Subagents"')
    expect(railHtml).toContain('aria-label="Close Immortals"')
    expect(railHtml).toContain('aria-label="Close Side question"')

    const moreNavHtml = renderToStaticMarkup(
      <VibesSideNav
        snap={minimalSnapshot}
        page={null}
        filter=""
        setFilter={() => {}}
        defaultMoreOpen
      />,
    )
    expect(moreNavHtml).toContain('<span>Immortals</span>')
    expect(moreNavHtml).toContain('<span>Side question</span>')

    const sideQuestionHtml = renderToStaticMarkup(
      <SideQuestionPanel
        snap={{
          ...minimalSnapshot,
          turnActive: true,
          sideQuestionTurns: [
            {
              requestId: 'req-1',
              question: 'What does sanitizeForkedMessages do?',
              answer: 'It clones messages and pairs pending tool calls.',
              model: 'claude-opus-4-6',
              status: 'done',
            },
          ],
        }}
      />,
    )
    expect(sideQuestionHtml).toContain('Ephemeral · read-only')
    expect(sideQuestionHtml).toContain('Main task continues')
    expect(sideQuestionHtml).toContain('It clones messages and pairs pending tool calls.')
  })

  test('sidebarSections tolerates malformed context entries without sessions array and SwarmBoard renders cleanly', () => {
    const groups: readonly WorkspaceGroup<GroupableRow>[] = [{ cwd: '/workspace/xerxes', name: 'xerxes', rows: [] }]
    const malformedContext = {
      id: 'ctx-malformed',
      remote: true,
      alias: 'devbox',
      label: 'devbox · xerxes',
      workspace: '/srv/xerxes',
      workspacePath: '/srv/xerxes',
    } as unknown as WorkspaceContext

    const sections = sidebarSections(groups, [malformedContext], 'This Mac')
    expect(sections).toHaveLength(2)
    expect(sections[1]?.rows).toEqual([])

    const boardHtml = renderToStaticMarkup(
      <SwarmBoard
        online
        loading={false}
        error={null}
        models={[{ id: 'claude-opus-4-6', provider: 'Anthropic', label: 'Claude Opus 4.6' }]}
        status={{
          active: true,
          session_id: 'sess-1',
          title: 'Desktop Audit Swarm',
          board: 'local',
          open_board: '[Open the Swarm UI](sidecar://swarm/swarm_board/?auto_open=false)',
          head: 2,
          overview: {
            markdown: '### Live Overview\n- Checked button specificity',
            summary: 'Checked button specificity',
            updated_at: '18:00:00',
            author: 'narrator',
          },
          channels: [{ name: 'main', topic: 'Main board', created_by: 'coordinator', archived: false, subscriber_count: 2 }],
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
              last_read_seq: 2,
            },
            {
              name: 'ui-worker',
              role: 'worker',
              added_by: 'coordinator',
              model: 'claude-sonnet-4-6',
              model_display_name: 'Claude Sonnet 4.6',
              workspace: 'share',
              channels: ['main'],
              state: 'working',
              killed: false,
              last_read_seq: 1,
              current_activity: 'Reading vibes.css',
            },
          ],
        }}
        messages={[
          { seq: 1, agent: 'coordinator', channel: 'main', text: '@ui-worker Audit SwarmBoard CSS', mentions: ['ui-worker'], channel_mentions: [], broadcast: false, time: '18:00:01' },
          { seq: 2, agent: 'ui-worker', channel: 'main', text: '@coordinator [done] Fixed button specificity (#1)', mentions: ['coordinator'], channel_mentions: [], broadcast: false, time: '18:00:10' },
        ]}
      />,
    )
    expect(boardHtml).toContain('Desktop Audit Swarm')
    expect(boardHtml).toContain('Reading vibes.css')
    expect(boardHtml).toContain('swarm-badge--done')
    expect(boardHtml).toContain('swarm-board__composer')
    expect(boardHtml).not.toContain('Wake:')
    expect(boardHtml).not.toContain('aria-label="Immortals channels"')
    expect(atelierCss).toContain('.swarm-model-picker{')
    expect(atelierCss).toContain('.picker-layer>.modelpop,.picker-layer>.swarm-model-picker{z-index:1}')
  })
})
