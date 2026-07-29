// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { testRender } from '@opentui/react/test-utils'
import { act, Profiler } from 'react'
import { describe, expect, it } from 'vitest'

import type { SpawnSnapshot } from '../app/spawnHistoryStore.js'
import { agentContentWidth, agentSidebarWidth, shouldMountAgentSidebar } from '../domain/agentPanelLayout.js'
import {
  AgentPanel,
  AgentPanelHotkey,
  AgentPanelOverlay,
  collectAgentPanelRecords,
  shortAgentTitle,
  shouldShowAgentSidebar
} from '../opentui/agentPanel.js'
import { DEFAULT_THEME } from '../theme.js'
import type { SubagentProgress } from '../types.js'

const agent = (overrides: Partial<SubagentProgress> = {}): SubagentProgress => ({
  agentType: 'researcher',
  depth: 0,
  goal: 'Audit authentication policy boundaries',
  id: 'agent-1',
  index: 0,
  notes: [],
  parentId: null,
  status: 'completed',
  taskCount: 1,
  thinking: [],
  toolCount: 3,
  tools: ['ReadFile', 'Grep', 'ReadFile'],
  ...overrides
})

const snapshot = (subagents: SubagentProgress[]): SpawnSnapshot => ({
  finishedAt: 20,
  id: 'snapshot-1',
  label: 'authentication audit',
  sessionId: 'session-1',
  startedAt: 10,
  subagents
})

describe('agent panel model', () => {
  it('keeps explicit agent titles short and strips runtime id suffixes', () => {
    const title = shortAgentTitle(agent({ title: 'runtime-policy-audit#019f5f030000' }))

    expect(title).toBe('Runtime Policy Audit')
    expect(title.length).toBeLessThanOrEqual(24)
    expect(
      shortAgentTitle(agent({ goal: 'Review provider routing and authentication boundaries', title: undefined }))
    ).toBe('Review Provider Routing…')
  })

  it('combines live and archived agents once and resolves creator hierarchy', () => {
    const parent = agent({ id: 'parent', name: 'runtime-audit', status: 'running' })
    const child = agent({ creatorId: 'parent', depth: 1, id: 'child', name: 'policy-review', parentId: 'parent' })
    const rows = collectAgentPanelRecords([parent], [snapshot([parent, child])])

    expect(rows).toHaveLength(2)
    expect(rows[0]).toMatchObject({ archived: false, childCount: 1, creatorTitle: 'Xerxes', title: 'Runtime Audit' })
    expect(rows[1]).toMatchObject({ archived: true, creatorTitle: 'Runtime Audit', title: 'Policy Review' })
  })

  it('unmounts the rail while the overlay is open so agents are not listed twice', () => {
    expect(shouldMountAgentSidebar(160, 4, false)).toBe(true)
    expect(shouldMountAgentSidebar(160, 4, true)).toBe(false)
    // The "does it fit" answer the footer hint reads must not change with the
    // overlay, or the hint flips while the overlay covers it and flips back.
    expect(shouldShowAgentSidebar(160, 4)).toBe(true)
  })

  it('keeps the sidebar at zero width until an agent is actually tracked', () => {
    expect(shouldShowAgentSidebar(117, 4)).toBe(false)
    expect(shouldShowAgentSidebar(118, 0)).toBe(false)
    expect(shouldShowAgentSidebar(118, 1)).toBe(true)
    expect(agentSidebarWidth(118)).toBe(38)
    expect(agentContentWidth(118, 0)).toBe(118)
    expect(agentContentWidth(118, 1)).toBe(80)
    expect(agentContentWidth(100, 4)).toBe(100)
  })
})

const auditAgent = (overrides: Partial<SubagentProgress> = {}): SubagentProgress =>
  agent({
    apiCalls: 2,
    durationSeconds: 12,
    filesRead: ['src/auth/session.ts'],
    filesWritten: ['src/auth/policy.ts'],
    inputTokens: 1200,
    model: 'grok-code-fast',
    outputTokens: 340,
    reasoningTokens: 90,
    rules: ['read-only audit', 'no network'],
    summary: 'Found and documented the missing policy guard.',
    title: 'Policy audit',
    toolsets: ['ReadFile', 'Grep'],
    ...overrides
  })

describe('OpenTUI agent panel', () => {
  it('reduces each row to what the agent cost, not what it is saying', async () => {
    // The row answers "which agent, how much, how long" and nothing else.
    // Policy, files, and live commentary moved into the inspector, where they
    // do not compete with the numbers you compare agents by.
    const setup = await testRender(
      <box height="100%" width="100%">
        <AgentPanel history={[snapshot([auditAgent()])]} liveAgents={[]} t={DEFAULT_THEME} />
      </box>,
      { height: 24, width: 72 }
    )

    try {
      await setup.flush()
      const frame = setup.captureCharFrame()

      expect(frame).toContain('Policy Audit')
      expect(frame).toContain('1.5k tok · 12s · 3 tools')
      expect(frame).toContain('task · Audit authentication policy boundaries')
      expect(frame).toContain('↳ Xerxes · researcher · grok-code-fast')
      expect(frame).not.toContain('Found and documented the missing policy guard.')
      expect(frame).not.toContain('policy · read-only audit')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('inspects one agent on Enter and returns to the list on Esc', async () => {
    const started = Date.now() - 10_000
    const parent = auditAgent({
      status: 'running',
      summary: undefined,
      toolCalls: [
        { endedAt: started + 2_400, id: 'call-1', name: 'ReadFile', ok: true, preview: 'src/auth/policy.ts', startedAt: started },
        { id: 'call-2', name: 'Grep', startedAt: Date.now() - 1_000 }
      ]
    })
    let closed = false
    const setup = await testRender(
      <AgentPanelOverlay
        history={[]}
        liveAgents={[parent]}
        onClose={() => {
          closed = true
        }}
        t={DEFAULT_THEME}
      />,
      { height: 34, width: 96 }
    )

    try {
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('1.5k tok')

      act(() => setup.mockInput.pressEnter())
      await setup.flush()
      const detail = setup.captureCharFrame()

      expect(detail).toContain('◆ Agent')
      expect(detail).toContain('1.2k in · 340 out · 90 reasoning · 2 API calls')
      expect(detail).toContain('tool calls (2)')
      // The finished call reports how long it took; the live one says it is
      // still going rather than reporting a duration it does not have.
      expect(detail).toContain('ReadFile · 2.4s')
      expect(detail).toContain('so far')

      // Esc backs out to the list before it closes the panel.
      act(() => setup.mockInput.pressEscape())
      // The renderer holds a bare ESC briefly to disambiguate escape sequences.
      await act(async () => {
        await Bun.sleep(50)
      })
      await setup.flush()
      expect(closed).toBe(false)
      expect(setup.captureCharFrame()).toContain('◆ Agents')

      act(() => setup.mockInput.pressEscape())
      // The renderer holds a bare ESC briefly to disambiguate escape sequences.
      await act(async () => {
        await Bun.sleep(50)
      })
      await setup.flush()
      expect(closed).toBe(true)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('opens straight into the agent a rail click named', async () => {
    const setup = await testRender(
      <AgentPanelOverlay
        history={[]}
        initialInspectId="agent-1"
        liveAgents={[auditAgent({ status: 'running', summary: undefined })]}
        onClose={() => undefined}
        t={DEFAULT_THEME}
      />,
      { height: 30, width: 96 }
    )

    try {
      await setup.flush()
      const frame = setup.captureCharFrame()

      expect(frame).toContain('◆ Agent ')
      expect(frame).toContain('tool calls')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('scrolls the inspector to the policy and files it keeps below the fold', async () => {
    const setup = await testRender(
      <AgentPanelOverlay history={[]} liveAgents={[auditAgent()]} onClose={() => undefined} t={DEFAULT_THEME} />,
      { height: 34, width: 96 }
    )

    try {
      await setup.flush()
      act(() => setup.mockInput.pressEnter())
      await setup.flush()
      act(() => setup.mockInput.pressKey('END'))
      await setup.flush()
      const bottom = setup.captureCharFrame()

      expect(bottom).toContain('1 wrote · 1 read · +policy.ts, session.ts')
      expect(bottom).toContain('rules · read-only audit, no network')
      expect(bottom).toContain('access · ReadFile, Grep')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('renders no sidebar surface before any agent activity', async () => {
    const setup = await testRender(
      <box height="100%" width="100%">
        <AgentPanel history={[]} liveAgents={[]} t={DEFAULT_THEME} />
        <text>full-width workspace</text>
      </box>,
      { height: 8, width: 72 }
    )

    try {
      await setup.flush()
      const frame = setup.captureCharFrame()

      expect(frame).toContain('full-width workspace')
      expect(frame).not.toContain('Agents')
      expect(frame).not.toContain('No agents yet')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('reports an archived interrupted row as done rather than live', async () => {
    const setup = await testRender(
      <box height="100%" width="100%">
        <AgentPanel history={[snapshot([agent({ status: 'interrupted' })])]} liveAgents={[]} t={DEFAULT_THEME} />
      </box>,
      { height: 12, width: 72 }
    )

    try {
      await setup.flush()
      const frame = setup.captureCharFrame()

      expect(frame).toContain('1 done')
      expect(frame).not.toContain('1 live')
      expect(frame).toContain('interrupted')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('does not schedule periodic commits while live-agent props stay stable', async () => {
    let commits = 0
    const liveAgents = [agent({ status: 'running' })]
    const setup = await testRender(
      <Profiler id="stable-agent-panel" onRender={() => commits++}>
        <AgentPanel history={[]} liveAgents={liveAgents} t={DEFAULT_THEME} />
      </Profiler>,
      { height: 18, width: 72 }
    )

    try {
      await setup.flush()
      const initialCommits = commits

      await Bun.sleep(650)
      await setup.flush()

      expect(initialCommits).toBeGreaterThan(0)
      expect(commits).toBe(initialCommits)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('toggles the keyboard-accessible panel with F6', async () => {
    const transitions: boolean[] = []
    const setup = await testRender(
      <box>
        <AgentPanelHotkey disabled={false} onToggle={open => transitions.push(open)} open={false} />
        <text>ready</text>
      </box>,
      { height: 4, width: 30 }
    )

    try {
      setup.mockInput.pressKey('F6')
      await setup.flush()
      expect(transitions).toEqual([true])
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('keeps the overlay inside the terminal and its footer on screen for a long agent list', async () => {
    // Ten agents is far more than fits: the frame previously grew to its content
    // height and ran off the bottom of the terminal, taking the footer — the only
    // place the close key is advertised — with it.
    const many = Array.from({ length: 10 }, (_, index) =>
      agent({ id: `agent-${index}`, status: 'running', title: `Deep Read Subsystem ${index}` })
    )
    const setup = await testRender(
      <AgentPanelOverlay history={[]} liveAgents={many} onClose={() => {}} t={DEFAULT_THEME} />,
      { height: 40, width: 120 }
    )

    try {
      await setup.flush()
      const rows = setup.captureCharFrame().split('\n')

      expect(rows.join('\n')).toContain('10 live')
      expect(rows.join('\n')).toContain('F6/Esc close')
      // The frame must not spill past the viewport it was given.
      expect(rows.filter(row => row.trim()).length).toBeLessThanOrEqual(40)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('scrolls the default selection into view when it sits below the fold', async () => {
    // The failed agent is last, so it is both the default selection and far
    // outside the initial viewport. A bounded frame that does not scroll would
    // open on the first rows and leave the row the user came for invisible.
    const many = [
      ...Array.from({ length: 9 }, (_, index) =>
        agent({ id: `ok-${index}`, status: 'completed', title: `Healthy Worker ${index}` })
      ),
      agent({ id: 'boom', status: 'failed', title: 'Crashed Worker' })
    ]
    const setup = await testRender(
      <AgentPanelOverlay history={[]} liveAgents={many} onClose={() => {}} t={DEFAULT_THEME} />,
      { height: 30, width: 90 }
    )

    try {
      // One extra settle frame: cards report no position until Yoga lays them out.
      await setup.flush()
      await act(async () => {
        await Bun.sleep(0)
      })
      await setup.flush()

      expect(setup.captureCharFrame()).toContain('Crashed Worker')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('closes the narrow overlay with its advertised F6 key', async () => {
    let closed = 0
    const setup = await testRender(
      <AgentPanelOverlay history={[]} liveAgents={[]} onClose={() => closed++} t={DEFAULT_THEME} />,
      { height: 10, width: 20 }
    )

    try {
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('Agents')
      expect(setup.captureCharFrame()).toContain('No agents yet')
      setup.mockInput.pressKey('F6')
      await setup.flush()
      expect(closed).toBe(1)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
