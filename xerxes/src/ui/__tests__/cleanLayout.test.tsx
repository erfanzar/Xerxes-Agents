// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { AppLayoutProps } from '../app/interfaces.js'
import { GatewayProvider } from '../app/gatewayContext.js'
import { findSlashCommand } from '../app/slash/registry.js'
import { patchOverlayState, resetOverlayState, getOverlayState, resetFlowOverlays } from '../app/overlayStore.js'
import { $agentRailVisible, resetPanelWidth } from '../app/panelSizeStore.js'
import { resetToolRunVisibility } from '../app/toolRunStore.js'
import { resetTurnState, patchTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { DERAFSH_KAVIANI_ART, DERAFSH_KAVIANI_COMPACT_ART } from '../banner.js'
import { buildToolTrailLine } from '../lib/text.js'
import { SpawnFleetRoster } from '../opentui/messageLine.js'
import { AppLayout, Composer, StartupWelcome } from '../opentui/appLayout.js'
import { AgentPanelHotkey } from '../opentui/agentPanel.js'
import { EMPTY_PULSE } from '../lib/repoPulse.js'
import { DARK_THEME, LIGHT_THEME } from '../theme.js'

const noop = () => undefined
const props = (cols: number): AppLayoutProps =>
  ({
    actions: new Proxy({}, { get: () => noop }),
    composer: {
      cols,
      compIdx: 0,
      compReplace: false,
      completions: [],
      empty: false,
      handleTextPaste: async () => null,
      input: 'Keep cancellation behavior unchanged.',
      inputBuf: [],
      queueEditIdx: -1,
      queuedDisplay: [],
      submit: noop,
      updateInput: noop
    },
    progress: { showProgressArea: false },
    status: { cwdLabel: '/repo', statusColor: DARK_THEME.color.muted },
    transcript: {
      historyItems: [],
      scrollRef: { current: null },
      virtualRows: [],
      virtualHistory: {
        start: 0,
        end: 0,
        topSpacer: 0,
        bottomSpacer: 0,
        totalHeight: 0,
        measureRef: () => noop,
        setScrollHandle: noop
      }
    }
  }) as unknown as AppLayoutProps

const reset = () => {
  resetUiState()
  resetOverlayState()
  resetTurnState()
  resetToolRunVisibility()
  resetPanelWidth()
  $agentRailVisible.set(true)
}

beforeEach(() => {
  reset()
  patchUiState({ info: { cwd: '/repo', model: 'sonnet-4.6', mode: 'code', permission_mode: 'default' } as never })
})

it.each([220, 80, 40])('shows current reasoning effort beside the model and updates it at %i columns', async width => {
  patchUiState({ info: { cwd: '/repo', model: 'gpt-6-astra', mode: 'code', reasoning_effort: 'high', permission_mode: 'default' } as never })
  const screen = await testRender(<Composer composer={props(width).composer} />, { width, height: 16 })
  try {
    await screen.flush()
    expect(screen.captureCharFrame().replace(/\s+/g, ' ')).toContain('reasoning: high')
    act(() => patchUiState(state => ({ ...state, info: { ...state.info!, reasoning_effort: 'low' } })))
    await screen.flush()
    expect(screen.captureCharFrame().replace(/\s+/g, ' ')).toContain('reasoning: low')
    expect(screen.captureCharFrame()).not.toContain('reasoning: high')
    act(() => patchUiState(state => ({ ...state, info: { ...state.info!, reasoning_effort: undefined } })))
    await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('reasoning:')
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([
  ['runs', 'Runs · Workspace', 'run.list'],
  ['monitors', 'Monitors · 0 watching', 'monitor.list'],
  ['schedules', 'Schedules · current workspace', 'schedule.list'],
  ['loop', 'Follow-ups · this conversation', 'schedule.list'],
])('opens /%s through its registered handler in the full layout and returns to the draft', async (command, title, method) => {
  patchUiState({ sid: 'discovery-session' })
  const rpc = vi.fn(async () => ({ ok: true, runs: [], monitors: [], jobs: [] }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as never}><AppLayout {...props(220)} /></GatewayProvider>, { width: 220, height: 65, kittyKeyboard: true })
  try {
    await screen.flush()
    act(() => findSlashCommand(command)!.run('', {} as never, '/' + command))
    await screen.flush()
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith(method, expect.any(Object)))
    expect(screen.captureCharFrame()).toContain(title)
    act(() => screen.mockInput.pressKey('ESCAPE'))
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Keep cancellation behavior unchanged.')
    expect(screen.captureCharFrame()).not.toContain(title)
  } finally { act(() => screen.renderer.destroy()) }
})
afterEach(reset)

describe('clean terminal layout', () => {
  it.each([
    [150, 40],
    [80, 24],
    [60, 20],
    [40, 16]
  ])('keeps draft, identity, policy and send hints inside %ix%i', async (width, height) => {
    patchUiState({ busy: true })
    const p = props(width)
    const s = await testRender(<AppLayout {...p} />, { width, height })
    try {
      await s.flush()
      const frame = s.captureCharFrame()
      expect(frame).toContain('cancellation behavior')
      expect(frame).toContain('code mode')
      expect(frame).toContain('sonnet-4.6')
      expect(frame).toContain('Enter')
      expect(frame).toContain('Esc')
      expect(frame).toContain('writes')
      expect(frame).not.toContain('╭─')
    } finally {
      act(() => s.renderer.destroy())
    }
  })

  it('keeps the real Derafsh artwork on a roomy welcome and hides artwork on short terminals', async () => {
    const wide = await testRender(
      <StartupWelcome cols={120} composer={props(120).composer} pulse={EMPTY_PULSE} rows={40} />,
      { width: 120, height: 40 }
    )
    const short = await testRender(
      <StartupWelcome cols={80} composer={props(80).composer} pulse={EMPTY_PULSE} rows={24} />,
      { width: 80, height: 24 }
    )
    const tall = await testRender(
      <StartupWelcome cols={120} composer={props(120).composer} pulse={EMPTY_PULSE} rows={80} />,
      { width: 120, height: 80 }
    )
    try {
      await wide.flush()
      await short.flush()
      await tall.flush()
      const tallFrame = tall.captureCharFrame()
      for (const line of DERAFSH_KAVIANI_ART) expect(tallFrame).toContain(line)
      const heading = tallFrame.split('\n').find(line => line.includes('What are we working on?'))!
      expect(heading.indexOf('What are we working on?')).toBe(6)
      expect(tallFrame).toContain('map this repo')
      expect(tallFrame).toContain('Explore capabilities · /features')
      expect(wide.captureCharFrame()).toContain(DERAFSH_KAVIANI_COMPACT_ART[0]!)
      expect(short.captureCharFrame()).not.toContain(DERAFSH_KAVIANI_COMPACT_ART[0]!)
      expect(short.captureCharFrame()).toContain('map this repo')
      expect(short.captureCharFrame()).toContain('/features')
      expect(short.captureCharFrame()).toContain('XERXES')
    } finally {
      act(() => {
        wide.renderer.destroy()
        short.renderer.destroy()
        tall.renderer.destroy()
      })
    }
  })

  it.each([[240, 64], [150, 40], [80, 24], [40, 16]])(
    "bounds and centers the welcome column at %ix%i",
    async (width, height) => {
      const p = props(width)
      p.composer.empty = true
      p.composer.input = ""
      const s = await testRender(<AppLayout {...p} />, { width, height })
      try {
        await s.flush()
        const lines = s.captureCharFrame().split("\n")
        const titleRow = lines.findIndex(line => line.includes("XERXES"))
        const rule = lines.find(line => line.includes("─".repeat(20)) && line.trim().length === Math.min(120, width - 4))!
        expect(titleRow).toBeGreaterThan(0)
        expect(titleRow).toBeLessThan(height / 2)
        expect(rule.trim().length).toBeLessThanOrEqual(120)
        expect(rule.indexOf("─")).toBeGreaterThanOrEqual(Math.floor((width - Math.min(120, width - 4)) / 2))
        expect(lines.join("\n")).toContain("sonnet-4.6")
      } finally {
        act(() => s.renderer.destroy())
      }
    }
  )

  it.each([[240, 64], [150, 40], [80, 24], [40, 16]])(
    'shows and scrolls goals and long todo lists inside the app at %ix%i',
    async (width, height) => {
      const p = props(width)
      patchUiState({ info: { cwd: '/repo', model: 'sonnet-4.6', goal: 'Audit every terminal feature', goal_phase: 'paused' } as never })
      patchTurnState({ todos: Array.from({length: 40}, (_, i) => ({id: String(i), content: 'Verify feature ' + String(i).padStart(2, '0'), status: i === 0 ? 'completed' : i === 1 ? 'in_progress' : 'pending'})) })
      patchOverlayState({goal: true})
      const s = await testRender(<AppLayout {...p} />, {width, height, kittyKeyboard: true})
      try {
        await s.flush()
        expect(s.captureCharFrame()).toContain('Goal & Todos')
        expect(s.captureCharFrame()).toContain('Esc close')
        expect(s.captureCharFrame()).toContain('paused')
        act(() => s.mockInput.pressKey('END'))
        await s.flush()
        expect(s.captureCharFrame()).toContain('Verify feature 39')
        act(() => resetFlowOverlays())
        expect(getOverlayState().goal).toBe(true)
        act(() => s.mockInput.pressKey('ESCAPE'))
        await s.flush()
        expect(getOverlayState().goal).toBe(false)
        expect(s.captureCharFrame()).toContain('cancellation behavior')
      } finally { act(() => s.renderer.destroy()) }
    }
  )

  it.each([[240, 64], [150, 40], [80, 24], [40, 16]])(
    'keeps a busy fleet, goal, todos and draft usable at %ix%i', async (width, height) => {
      const p = props(width)
      p.progress.showProgressArea = true
      patchUiState({busy: true, info: {cwd: '/repo', model: 'sonnet-4.6', goal: 'Audit all features', goal_phase: 'active'} as never})
      patchTurnState({
        subagents: [
          {id:'a1', index:0, name:'reviewer', title:'Reviewer', agentType:'researcher', goal:'Review changes', status:'running', depth:0, parentId:null, taskCount:1, notes:[],thinking:[], tools:[],toolCount:0},
          {id:'a2', index:1, name:'tester', title:'Tester', agentType:'researcher', goal:'Run tests', status:'failed', depth:0, parentId:null, taskCount:1, notes:['Provider unavailable'],thinking:[], tools:[],toolCount:0}
        ],
        todos: [{id:'todo1',content:'Verify fleet rendering',status:'in_progress'}]
      })
      const s = await testRender(<AppLayout {...p} />, {width,height,kittyKeyboard:true})
      try {
        await s.flush()
        expect(s.captureCharFrame()).toContain('cancellation behavior')
        expect(s.captureCharFrame()).toContain('Audit all features')
        expect(s.captureCharFrame()).toContain('Tasks 0/1')
        act(() => patchUiState(state => ({
          ...state, info: { ...state.info!, goal: 'Optimize TPU kernels', goal_phase: 'active' }
        })))
        await s.flush()
        expect(s.captureCharFrame()).toContain('Optimize TPU kernels')
        expect(s.captureCharFrame()).not.toContain('Audit all features')
        act(() => s.mockInput.pressKey('F10'))
        await s.flush()
        expect(s.captureCharFrame()).toContain('Verify fleet rendering')
        act(() => s.mockInput.pressKey('F10'))
        await s.flush()
        expect(getOverlayState().goal).toBe(false)
        act(() => s.mockInput.pressKey('F6'))
        await s.flush()
        expect(getOverlayState().agents).toBe(true)
        expect(s.captureCharFrame()).toContain('Tester')
        act(() => s.mockInput.pressKey('F6'))
        await s.flush()
        expect(getOverlayState().agents).toBe(false)
        expect(s.captureCharFrame()).toContain('cancellation behavior')
      } finally {act(() => s.renderer.destroy())}
    }
  )

  it('keeps nonempty tasks pinned and preserves the durable F10 plan', async () => {
    const p = props(120)
    p.progress.showProgressArea = true
    patchUiState({ busy: true, info: { cwd: '/repo', model: 'sonnet-4.6', goal: 'Old completed goal', goal_phase: 'active' } as never })
    patchTurnState({ todos: [{ id: 'done', content: 'Finished task', status: 'completed' }] })
    const s = await testRender(<AppLayout {...p} />, { width: 120, height: 30, kittyKeyboard: true })
    try {
      await s.flush()
      expect(s.captureCharFrame()).toContain('Old completed goal')
      expect(s.captureCharFrame()).toContain('Tasks 1/1')

      act(() => patchUiState(state => ({ ...state, info: { ...state.info!, goal_phase: 'complete' } })))
      await s.flush()
      const liveFrame = s.captureCharFrame()
      expect(liveFrame.split('\n').filter(line => line.includes('Old completed goal'))).toHaveLength(1)
      expect(liveFrame).toContain('Tasks 1/1')

      act(() => s.mockInput.pressKey('F10'))
      await s.flush()
      expect(s.captureCharFrame()).toContain('Goal & Todos')
      expect(s.captureCharFrame()).toContain('Old completed goal')
      act(() => s.mockInput.pressKey('F10'))
      await s.flush()

      act(() => patchTurnState({ todos: [
        { id: 'done', content: 'Finished task', status: 'completed' },
        { id: 'next', content: 'Next unfinished task', status: 'pending' }
      ] }))
      await s.flush()
      expect(s.captureCharFrame()).toContain('Tasks 1/2')
      expect(s.captureCharFrame()).toContain('Next unfinished task')
      expect(s.captureCharFrame().split('\n').filter(line => line.includes('Old completed goal'))).toHaveLength(1)
    } finally { act(() => s.renderer.destroy()) }
  })

  it.each(['skillsHub', 'pluginsHub'] as const)('closes %s without erasing the draft', async kind => {
    patchOverlayState({[kind]: true})
    const s = await testRender(<AppLayout {...props(80)} />, {width:80,height:24,kittyKeyboard:true})
    try {
      await s.flush()
      expect(s.captureCharFrame()).toContain(kind === 'skillsHub' ? 'Native skills' : 'Native plugins')
      expect(s.captureCharFrame()).toContain(kind === 'skillsHub' ? 'Discover' : 'Inspect')
      act(() => s.mockInput.pressKey('END'))
      await s.flush()
      expect(s.captureCharFrame()).toContain(kind === 'skillsHub' ? 'Use it' : 'Manage')
      act(() => s.mockInput.pressKey('F10'))
      expect(getOverlayState().goal).toBe(false)
      act(() => s.mockInput.pressKey('q'))
      await s.flush()
      expect(getOverlayState()[kind]).toBe(false)
      expect(s.captureCharFrame()).toContain('Keep cancellation behavior unchanged.')
    } finally {act(() => s.renderer.destroy())}
  })

  it('expands successful runs with F9, keeps failures visible, and blocks F9 during a prompt', async () => {
    const p = props(100)
    p.transcript.virtualRows = [
      {
        key: 'tools',
        index: 0,
        leadGap: false,
        rail: 'none',
        turnSeconds: 0,
        turnTools: 0,
        msg: {
          kind: 'trail',
          role: 'system',
          text: '',
          tools: [
            ...['one.ts', 'two.ts', 'three.ts'].map((file) => buildToolTrailLine('read_file', file, false, '', 0.1)),
            buildToolTrailLine('bash', 'bun test', true, 'cancelled by user', 0.2)
          ]
        }
      }
    ]
    p.transcript.virtualHistory.end = 1
    const s = await testRender(<AppLayout {...p} />, { width: 100, height: 32 })
    try {
      await s.flush()
      expect(s.captureCharFrame()).toContain('3 tools')
      expect(s.captureCharFrame()).not.toContain('one.ts')
      expect(s.captureCharFrame()).toContain('cancelled by user')
      act(() => patchOverlayState({ confirm: { title: 'Confirm action', onConfirm: noop } }))
      act(() => s.mockInput.pressKey('F9'))
      await s.flush()
      expect(s.captureCharFrame()).not.toContain('one.ts')
      act(() => resetOverlayState())
      act(() => s.mockInput.pressKey('F9'))
      await s.flush()
      expect(s.captureCharFrame()).toContain('one.ts')
      expect(s.captureCharFrame()).not.toContain('····')
      act(() => s.mockInput.pressKey('F9'))
      await s.flush()
      expect(s.captureCharFrame()).not.toContain('one.ts')
      expect(s.captureCharFrame()).toContain('cancelled by user')
    } finally {
      act(() => s.renderer.destroy())
    }
  })

  it('preserves a draft across approval denial and keeps short-screen approval choices visible', async () => {
    const p = props(80)
    p.actions.answerApproval = () => patchOverlayState({ approval: null })
    // Replace the generic fixture proxy for the action exercised here.
    p.actions = { answerApproval: () => patchOverlayState({ approval: null }) } as AppLayoutProps['actions']
    const s = await testRender(<AppLayout {...p} />, { width: 80, height: 24 })
    try {
      act(() =>
        patchOverlayState({ approval: { command: 'bun test', requestId: 'approval', description: 'Focused tests' } })
      )
      await s.flush()
      expect(s.captureCharFrame()).toContain('bun test')
      expect(s.captureCharFrame()).toContain('n deny')
      expect(s.captureCharFrame()).toContain('Esc deny')
      act(() => s.mockInput.pressKey('n'))
      await s.flush()
      expect(s.captureCharFrame()).not.toContain('Approval needed')
      expect(s.captureCharFrame()).toContain('Keep cancellation behavior unchanged.')
    } finally {
      act(() => s.renderer.destroy())
    }
  })

  it('Ctrl+F6 toggles the rail without opening the inspector', async () => {
    const toggle = vi.fn()
    const inspect = vi.fn()
    const s = await testRender(
      <AgentPanelHotkey disabled={false} open={false} onToggle={inspect} onToggleRail={toggle} />,
      { width: 80, height: 10, kittyKeyboard: true }
    )
    try {
      act(() => s.mockInput.pressKey('F6', { ctrl: true }))
      await s.flush()
      expect(toggle).toHaveBeenCalledOnce()
      expect(inspect).not.toHaveBeenCalled()
    } finally {
      act(() => s.renderer.destroy())
    }
  })

  it.each([280, 281])('expands conversation input to the pane width at %i columns', async width => {
    const p = props(width)
    p.composer.empty = true
    const setup = await testRender(<AppLayout {...p} />, {width, height: 80})
    try {
      await setup.flush()
      const before = setup.captureCharFrame().split('\n')
      const firstRule = before.find(line => line.trim().startsWith('─') && line.trim().length === 120)!
      act(() => patchUiState({busy: true}))
      await setup.flush()
      const after = setup.captureCharFrame().split('\n')
      const rules = after.filter(line => line.trim().startsWith('─') && line.trim().length === width - 4)
      expect(rules).toHaveLength(2)
      expect(rules[0]!.trim()).toHaveLength(width - 4)
      expect(rules[0]!.indexOf('─')).toBe(2)
      expect(after.indexOf(rules[0]!)).toBe(before.indexOf(firstRule))
      expect(after.join('\n')).toContain('Keep cancellation behavior unchanged.')
    } finally {act(() => setup.renderer.destroy())}
  })

  it('opens the spawned agent inspector from its transcript status row', async () => {
    patchTurnState({ subagents: [{id:'inspect-me',index:0,name:'reviewer',title:'Reviewer',agentType:'researcher',goal:'Review changes',status:'running',depth:0,parentId:null,taskCount:1,notes:['raw tool output should stay in the inspector'],thinking:[],tools:[],toolCount:3}] })
    const setup = await testRender(<SpawnFleetRoster names={['reviewer']} t={DARK_THEME} />, { width:100,height:10 })
    try {
      await setup.flush()
      const lines = setup.captureCharFrame().split('\n')
      const y = lines.findIndex(line => line.includes('reviewer'))
      expect(lines.join('\n')).not.toContain('raw tool output')
      await act(async () => setup.mockMouse.click(lines[y]!.indexOf('reviewer') + 1, y))
      await setup.flush()
      expect(getOverlayState().agentsInspectId).toBe('inspect-me')
      expect(getOverlayState().agents).toBe(true)
    } finally { act(() => setup.renderer.destroy()) }
  })

  it('opens the goal inspector from the clickable workspace header', async () => {
    const setup = await testRender(<AppLayout {...props(240)} />, {width:240,height:64})
    try {
      await setup.flush()
      const lines = setup.captureCharFrame().split('\n')
      const y = lines.findIndex(line => line.includes('F10 goals'))
      const x = lines[y]!.indexOf('F10 goals')
      await act(async () => setup.mockMouse.click(x + 2, y))
      await setup.flush()
      expect(getOverlayState().goal).toBe(true)
      expect(setup.captureCharFrame()).toContain('Goal & Todos')
    } finally {act(() => setup.renderer.destroy())}
  })

  it.each([DARK_THEME, LIGHT_THEME])('renders legible composer text in both appearances', async (theme) => {
    patchUiState({ theme })
    const s = await testRender(<Composer composer={props(80).composer} />, { width: 80, height: 12 })
    try {
      await s.flush()
      expect(s.captureCharFrame()).toContain('sonnet-4.6')
      const luminance = (hex: string) => {
        const channels = [1, 3, 5]
          .map((i) => Number.parseInt(hex.slice(i, i + 2), 16) / 255)
          .map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4))
        return channels[0]! * 0.2126 + channels[1]! * 0.7152 + channels[2]! * 0.0722
      }
      for (const foreground of [theme.ds.meta, theme.ds.caption, theme.ds.numeric]) {
        const values = [luminance(foreground), luminance(theme.ds.screen)].sort((a, b) => b - a)
        expect((values[0]! + 0.05) / (values[1]! + 0.05)).toBeGreaterThanOrEqual(4.5)
      }
    } finally {
      act(() => s.renderer.destroy())
    }
  })
})
