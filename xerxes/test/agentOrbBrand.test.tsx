// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { initialHubState, initialSupervisorConfig, type HubState } from '../src/client/hubStore.js'
import { AgentOrb, type AgentOrbState } from '../src/desktop/renderer/AgentOrb.js'
import { AgentsCard } from '../src/desktop/renderer/AgentsCard.js'
import { RailStatus } from '../src/desktop/renderer/RailStatus.js'
import { ConnectHub, SignInFailed, SupervisorView } from '../src/desktop/renderer/SupervisorPage.js'
import { LiveSteps, ThreadMessages } from '../src/desktop/renderer/SupervisorThread.js'
import {
  XerxesStatusMark,
  resolveXerxesStatusTone,
  type XerxesStatusTone,
} from '../src/desktop/renderer/XerxesSprites.js'
import type { Snapshot } from '../src/desktop/renderer/store.js'
import type { AgentMember } from '../src/desktop/renderer/types.js'

const HEX_COLOR_RE = /#[0-9a-fA-F]{6}\b/g
const atelierCss = readFileSync(
  join(import.meta.dir, '../src/desktop/renderer/atelier.css'),
  'utf8',
)

const ALL_STATES: ReadonlyArray<readonly [AgentOrbState, XerxesStatusTone]> = [
  ['working', 'active'],
  ['thinking', 'active'],
  ['reasoning', 'active'],
  ['solving', 'active'],
  ['searching', 'active'],
  ['writing', 'active'],
  ['composing', 'active'],
  ['shaping', 'active'],
  ['weaving', 'active'],
  ['listening', 'active'],
  ['idle', 'idle'],
  ['breathing', 'idle'],
  ['ready', 'idle'],
  ['connecting', 'waiting'],
  ['waiting', 'waiting'],
  ['approval', 'waiting'],
  ['needs', 'waiting'],
  ['error', 'error'],
  ['failed', 'error'],
  ['done', 'done'],
  ['success', 'done'],
]

function makeSnapshot(overrides: Partial<Snapshot> = {}): Snapshot {
  return {
    connection: 'online',
    cwd: '/repo',
    model: 'claude-opus-4-6',
    currentAgentPreset: 'default',
    agentPresets: [],
    models: [],
    reasoningLevels: [],
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
    currentId: 'sess-1',
    currentTitle: 'Cosmic Simurgh Brand',
    sessionKey: 'k1',
    branch: 'main',
    daemonWarning: null,
    costUsd: null,
    mcpStatus: {},
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
    permissionMode: 'auto',
    snippets: {},
    ...overrides,
  } as unknown as Snapshot
}

describe('Cosmic Simurgh AgentOrb & XerxesStatusMark', () => {
  test('AgentOrb renders bespoke Cosmic Simurgh SVG without any legacy thinking-orbs canvas', () => {
    for (const [state, expectedTone] of ALL_STATES) {
      expect(resolveXerxesStatusTone(state)).toBe(expectedTone)

      const compactHtml = renderToStaticMarkup(
        createElement(AgentOrb, { state, size: 20, live: true }),
      )
      expect(compactHtml).toContain(`class="agent-orb" data-state="${state}"`)
      expect(compactHtml).toContain(`data-tone="${expectedTone}"`)
      expect(compactHtml).toContain('data-live="true"')
      expect(compactHtml).toContain('<svg')
      expect(compactHtml).toContain('viewBox="0 0 24 24"')
      expect(compactHtml).toContain('data-emblem="xerxes-status"')
      expect(compactHtml).not.toContain('<canvas')
      expect(compactHtml.match(HEX_COLOR_RE) ?? []).toEqual([])

      const regalHtml = renderToStaticMarkup(
        createElement(AgentOrb, { state, size: 64, live: true }),
      )
      expect(regalHtml).toContain('<svg')
      expect(regalHtml).toContain('width="64"')
      expect(regalHtml).toContain(`data-xerxes-status-tone="${expectedTone}"`)
      expect(regalHtml).not.toContain('<canvas')
      expect(regalHtml.match(HEX_COLOR_RE) ?? []).toEqual([])
    }
  })

  test('each semantic tone produces a distinct visual SVG silhouette and color signature', () => {
    const representativeStates: readonly AgentOrbState[] = [
      'working',
      'idle',
      'needs',
      'failed',
      'done',
    ]

    const compactOutputs = new Set<string>()
    const regalOutputs = new Set<string>()
    for (const state of representativeStates) {
      compactOutputs.add(
        renderToStaticMarkup(createElement(XerxesStatusMark, { state, size: 20, live: true })),
      )
      regalOutputs.add(
        renderToStaticMarkup(createElement(XerxesStatusMark, { state, size: 48, live: true })),
      )
    }

    expect(compactOutputs.size).toBe(5)
    expect(regalOutputs.size).toBe(5)

    // Active: clean 2×2 chasing 4-dot grid loader
    const activeHtml = renderToStaticMarkup(
      createElement(AgentOrb, { state: 'working', size: 20, live: true }),
    )
    expect(activeHtml).toContain('xerxes-mark-spin')
    expect(activeHtml).toContain('xerxes-dot-grid')
    expect(activeHtml).toContain('var(--x-title')
    expect(activeHtml.match(/<circle /g)?.length).toBe(4)
    const stillActive = renderToStaticMarkup(
      createElement(AgentOrb, { state: 'working', size: 20, live: false }),
    )
    expect(stillActive).not.toContain('xerxes-mark-spin')
    expect(stillActive).toContain('xerxes-dot-grid')

    // Idle: quiet 3.5px solid dot
    const idleHtml = renderToStaticMarkup(
      createElement(AgentOrb, { state: 'idle', size: 20, live: false }),
    )
    expect(idleHtml).toContain('var(--x-meta')
    expect(idleHtml).toContain('data-live="false"')
    expect(idleHtml).toContain('r="3.5"')
    expect(idleHtml).not.toContain('<path')
    expect(idleHtml).not.toContain('xerxes-mark-spin')

    // Waiting: quiet 1.75px hollow circle (r=4.5)
    const waitingHtml = renderToStaticMarkup(
      createElement(AgentOrb, { state: 'approval', size: 20, live: true }),
    )
    expect(waitingHtml).toContain('var(--x-needs')
    expect(waitingHtml).toContain('r="4.5"')
    expect(waitingHtml).toContain('fill="none"')
    expect(waitingHtml).not.toContain('xerxes-mark-spin')

    // Error: minimal 1.75px cross stroke
    const errorHtml = renderToStaticMarkup(
      createElement(AgentOrb, { state: 'failed', size: 20, live: true }),
    )
    expect(errorHtml).toContain('var(--x-failed')
    expect(errorHtml).toContain('<path')
    expect(errorHtml).toContain('stroke-width="1.75"')
    expect(errorHtml).not.toContain('<polygon')
    expect(errorHtml).not.toContain('xerxes-mark-spin')

    // Done: minimal 1.75px check stroke
    const doneHtml = renderToStaticMarkup(
      createElement(AgentOrb, { state: 'done', size: 20, live: false }),
    )
    expect(doneHtml).toContain('var(--x-done')
    expect(doneHtml).toContain('<path')
    expect(doneHtml).toContain('stroke-width="1.75"')
    expect(doneHtml).not.toContain('<circle')

    // Pure GPU CSS motion (no SMIL <animate> elements)
    for (const html of [...compactOutputs, ...regalOutputs]) {
      expect(html).not.toContain('<animate')
      expect(html).not.toContain('kaviani')
    }
  })

  test('AgentOrb handles accessibility labels and pauses motion when live=false or environment is paused', () => {
    const decorative = renderToStaticMarkup(createElement(AgentOrb, { state: 'working', size: 20 }))
    expect(decorative).toContain('aria-hidden="true"')
    expect(decorative).not.toContain('role="img"')

    const labelled = renderToStaticMarkup(
      createElement(AgentOrb, { state: 'working', size: 36, label: 'Xerxes is thinking' }),
    )
    expect(labelled).toContain('role="img"')
    expect(labelled).toContain('aria-label="Xerxes is thinking"')

    const still = renderToStaticMarkup(
      createElement(AgentOrb, { state: 'working', size: 20, live: false }),
    )
    expect(still).toContain('data-live="false"')
    expect(still).not.toContain('xerxes-mark-spin')

    // The working ring pauses when the window is unfocused, occluded, perf-paused, or under reduced motion.
    expect(atelierCss).toContain('@keyframes xerxes-mark-spin')
    expect(atelierCss).toContain(':root[data-window-unfocused] .xerxes-mark-spin')
    expect(atelierCss).toContain(':root[data-occluded] .xerxes-mark-spin')
    expect(atelierCss).toContain("@media (prefers-reduced-motion:reduce){.xerxes-mark-spin{animation:none}}")

    // Zero decorative / idle animations remain in atelier.css
    for (const removed of [
      'xerxes-astrolabe-ring',
      'xerxes-simurgh-orbit',
      'xerxes-glyph-spin',
      'xerxes-emblem-breath',
      '@keyframes activity-sheen',
      '@keyframes working-breathe',
      '@keyframes phrase-in',
      '@keyframes phrase-out',
      '@keyframes studio-enter',
      '@keyframes relay-pulse',
      '@keyframes acard-pulse',
      '@keyframes usage-spin',
      '@keyframes fleetchip-spin',
    ]) {
      expect(atelierCss).not.toContain(removed)
    }
  })
})

describe('App-wide Cosmic Simurgh cohesion across RailStatus, AgentsCard, and Supervisor', () => {
  test('RailStatus uses Cosmic Simurgh AgentOrb across all tones with zero gray dots or hex literals', () => {
    const cases: Array<readonly [Partial<Snapshot>, string]> = [
      [{ turnActive: true }, 'active'],
      [{ approval: { id: 'a1', action: 'bash', description: 'run' } }, 'waiting'],
      [{ failed: { error: 'boom', turn: 1, lastUser: 'go' } }, 'error'],
      [{ planMode: true }, 'idle'],
      [{}, 'idle'],
    ]

    for (const [overrides, expectedTone] of cases) {
      const html = renderToStaticMarkup(
        createElement(RailStatus, { snap: makeSnapshot(overrides) }),
      )
      expect(html).toContain('class="agent-orb railstatus__orb"')
      expect(html).toContain(`data-tone="${expectedTone}"`)
      expect(html).toContain('xerxes-status-mark')
      expect(html).not.toContain('railstatus__dot')
      expect(html).not.toContain('<canvas')
      expect(html.match(HEX_COLOR_RE) ?? []).toEqual([])
    }
  })

  test('AgentsCard uses Cosmic Simurgh AgentOrb and role-mapped SimurghSprite on active agents', () => {
    const members: AgentMember[] = [
      {
        key: 'a1',
        runtimeId: 'a1',
        title: 'graph-verifier',
        status: 'working',
        group: { id: 'wf1', label: 'Polish Wave', phase: 'Verify' },
      },
      {
        key: 'a2',
        runtimeId: 'a2',
        title: 'brand-builder',
        status: 'working',
        group: { id: 'wf1', label: 'Polish Wave', phase: 'Verify' },
      },
      {
        key: 'a3',
        runtimeId: 'a3',
        title: 'doc-writer',
        status: 'completed',
        group: { id: 'wf1', label: 'Polish Wave', phase: 'Docs' },
      },
    ]

    const html = renderToStaticMarkup(createElement(AgentsCard, { members }))
    expect(html).toContain('class="agent-orb"')
    expect(html).toContain('xerxes-status-mark')
    expect(html).toContain('data-simurgh-role="verifier"')
    expect(html).toContain('data-simurgh-role="worker"')
    expect(html).not.toContain('<canvas')
    expect(html.match(HEX_COLOR_RE) ?? []).toEqual([])

    const settledHtml = renderToStaticMarkup(
      createElement(AgentsCard, {
        members: [{ key: 'done1', runtimeId: 'done1', title: 'worker-1', status: 'completed' }],
      }),
    )
    expect(settledHtml).toContain('data-state="done"')
    expect(settledHtml).toContain('xerxes-status-mark')
    expect(settledHtml.match(HEX_COLOR_RE) ?? []).toEqual([])
  })

  test('SupervisorPage and SupervisorThread render Xerxes real phoenix logo (no split-X) and animated status mark', () => {
    const noop = (): void => {}
    const state: HubState = {
      ...initialHubState(),
      connection: { state: 'live' },
      supervisor: { phase: 'ready', busy: false, messages: [], confirmations: [] },
      supervisorConfig: {
        ...initialSupervisorConfig(),
        phase: 'ready',
        status: 'configured',
        canEdit: true,
        config: { machineId: 'm1', machineLabel: 'Mac', profile: 'work', model: 'claude-opus-4-6', effort: 'high' },
      },
    }

    const emptyHtml = renderToStaticMarkup(
      createElement(SupervisorView, {
        state,
        hubLabel: 'hub.local',
        now: Date.now(),
        onSend: async () => true,
        onRespond: noop,
        onReopen: noop,
        onOpenSettings: noop,
        chooserOpen: false,
        onChooserChange: noop,
        onChooserShown: noop,
        onOptions: noop,
        onSaveModel: async () => true,
      }),
    )
    expect(emptyHtml).toContain('xerxes-brand-emblem')
    expect(emptyHtml).toContain('src="./logo-128.png"')
    expect(emptyHtml).not.toContain('data-emblem="xerxes-mark"')
    expect(emptyHtml).not.toContain('simurgh-mark')
    expect(emptyHtml.match(HEX_COLOR_RE) ?? []).toEqual([])

    const connectHtml = renderToStaticMarkup(createElement(ConnectHub, { onOpenSettings: noop }))
    expect(connectHtml).toContain('xerxes-brand-emblem')
    expect(connectHtml).toContain('src="./logo-128.png"')
    expect(connectHtml).not.toContain('data-emblem="xerxes-mark"')
    expect(connectHtml.match(HEX_COLOR_RE) ?? []).toEqual([])

    const failHtml = renderToStaticMarkup(
      createElement(SignInFailed, {
        message: 'timeout',
        denied: false,
        onRetry: noop,
        onSignInAgain: noop,
        onOpenSettings: noop,
      }),
    )
    expect(failHtml).toContain('data-state="failed"')
    expect(failHtml).toContain('xerxes-status-mark')
    expect(failHtml.match(HEX_COLOR_RE) ?? []).toEqual([])

    const threadHtml = renderToStaticMarkup(
      createElement(ThreadMessages, {
        messages: [
          { id: 'x1', role: 'xerxes', text: 'All systems nominal.', at: '2026-11-22T00:00:00.000Z', source: 'text' },
        ],
        now: Date.parse('2026-11-22T00:01:00.000Z'),
        label: 'Conversation',
      }),
    )
    expect(threadHtml).toContain('xmsg__crest')
    expect(threadHtml).toContain('xerxes-brand-emblem')
    expect(threadHtml).toContain('src="./logo-128.png"')
    expect(threadHtml).not.toContain('data-emblem="xerxes-mark"')
    expect(threadHtml).toContain('aria-label="Copy reply"')
    expect(threadHtml.match(HEX_COLOR_RE) ?? []).toEqual([])

    const liveHtml = renderToStaticMarkup(
      createElement(LiveSteps, {
        steps: [
          { id: 's1', turnId: 't1', kind: 'tool', text: 'Inspecting swarm', state: 'running', at: '2026-11-22T00:00:00.000Z' },
        ],
        seconds: 3,
      }),
    )
    expect(liveHtml).toContain('class="agent-orb"')
    expect(liveHtml).toContain('xerxes-status-mark')
    expect(liveHtml).not.toContain('<canvas')
    expect(liveHtml.match(HEX_COLOR_RE) ?? []).toEqual([])
  })
})

describe('Customize .learned-badge flat tabular-nums polish', () => {
  test('.learned-badge renders as a clean borderless tabular-nums count with zero circular orb or pill fill', () => {
    const badgeRuleMatch = /^\.learned-badge\{([^}]+)\}/m.exec(atelierCss)
    expect(badgeRuleMatch).not.toBeNull()
    const badgeRule = badgeRuleMatch?.[1] ?? ''

    expect(badgeRule).toContain('background:transparent')
    expect(badgeRule).toContain('border:0')
    expect(badgeRule).toContain('border-radius:0')
    expect(badgeRule).toContain('color:var(--x-meta)')
    expect(badgeRule).toContain('font-variant-numeric:tabular-nums')
    expect(badgeRule).not.toContain('border-radius:9px')
    expect(badgeRule).not.toContain('border-radius:50%')
    expect(badgeRule).not.toContain('var(--x-accent-soft)')

    const tabsBadgeRuleMatch = /^\.learned-panel__tabs \.learned-badge\{([^}]+)\}/m.exec(atelierCss)
    expect(tabsBadgeRuleMatch).not.toBeNull()
    const tabsBadgeRule = tabsBadgeRuleMatch?.[1] ?? ''
    expect(tabsBadgeRule).toContain('margin-left:6px')
    expect(tabsBadgeRule).toContain('opacity:.75')
  })
})
