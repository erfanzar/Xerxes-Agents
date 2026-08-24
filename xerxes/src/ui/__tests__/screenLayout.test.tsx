/** @jsxImportSource @opentui/react */
//
// Whole-screen layout, assembled.
//
// Every other UI test renders one component. That is why a run of visual
// regressions shipped green: the transcript stranded at the top of a tall
// terminal, agent titles eaten by their own goal line, and a per-side border
// painting itself through the composer's identity row are all properties of
// the ASSEMBLED screen, invisible to a component rendered on its own.
//
// Set XERXES_SCREEN_DUMP=1 to print the frames while working on a screen.
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'

import type { AppLayoutProps } from '../app/interfaces.js'
import { $uiState, $uiTheme } from '../app/uiStore.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { AppLayout } from '../opentui/appLayout.js'
import { buildToolTrailLine } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const DIFF = {
  kind: 'diff' as const,
  deletions: 6,
  files: 2,
  insertions: 21,
  truncated: false,
  untracked: [],
  lines: [
    { kind: 'file' as const, text: 'packages/runtime/scheduler.ts' },
    { kind: 'hunk' as const, text: '@@ -118,7 +118,9 @@ class Scheduler {' },
    { kind: 'context' as const, newLine: 118, oldLine: 118, text: '   private onWorkerExit(w: Worker, code: number) {' },
    { kind: 'del' as const, oldLine: 119, text: '-    if (code !== 0) this.requeue(w.task)' },
    { kind: 'add' as const, newLine: 119, text: '+    const lease = this.leases.release(w.id)' },
    { kind: 'add' as const, newLine: 120, text: '+    if (lease?.inFlight) this.requeue(lease.task, { reason: "worker-exit" })' },
    { kind: 'context' as const, newLine: 121, oldLine: 120, text: '     this.pool.drop(w.id)' },
    { kind: 'file' as const, text: 'packages/runtime/lease.ts' },
    { kind: 'hunk' as const, text: '@@ -12,3 +12,4 @@' },
    { kind: 'add' as const, newLine: 12, text: '+  inFlight = true' }
  ]
}


const noop = () => undefined

const props = (cols: number): AppLayoutProps =>
  ({
    actions: new Proxy({}, { get: () => noop }),
    composer: {
      cols,
      compIdx: 0,
      compReplace: 0,
      completions: [],
      empty: true,
      handleTextPaste: async () => null,
      input: '',
      inputBuf: [],
      pagerPageSize: 20,
      queueEditIdx: null,
      queuedDisplay: [],
      submit: noop,
      updateInput: noop,
      voiceRecordKey: { ctrl: true, name: 'r' }
    },
    mouseTracking: 'off',
    progress: { showProgressArea: false },
    status: {
      cwdLabel: '~/src/xerxes',
      goodVibesTick: 0,
      lastTurnDurationMs: null,
      lastTurnEndedAt: null,
      sessionStartedAt: null,
      showStickyPrompt: false,
      statusColor: DEFAULT_THEME.color.muted,
      stickyPrompt: '',
      turnStartedAt: null,
      voiceLabel: ''
    },
    transcript: {
      historyItems: [],
      scrollRef: { current: null },
      virtualHistory: {
        start: 0,
        end: 0,
        topSpacer: 0,
        bottomSpacer: 0,
        totalHeight: 0,
        measureRef: () => noop,
        setScrollHandle: noop
      },
      virtualRows: []
    }
  }) as unknown as AppLayoutProps

const row = (msg: Record<string, unknown>, over: Record<string, unknown> = {}) =>
  ({ index: 0, key: String(Math.random()), leadGap: false, msg, rail: 'none', turnSeconds: 0, turnTools: 0, ...over }) as never

const agent = (over: Record<string, unknown> = {}) =>
  ({ id: 'a1', index: 0, name: 'structure-analyzer', title: 'Structure Analyzer', goal: 'map the repo',
     status: 'running', depth: 0, notes: ['reading src/index.ts'], thinking: [], tools: [], toolCount: 1,
     startedAt: Date.now() - 8000, inputTokens: 1200, outputTokens: 300, ...over }) as never

const dump = (label: string, frame: string) => {
  if (process.env.XERXES_SCREEN_DUMP) {
    console.log(`\n── ${label} ──\n${frame}`)
  }
}

const savedRow = (title: string, msgs: number, ago: number) =>
  ({ group: 'saved', id: title, kind: 'saved',
     item: { id: title, title, message_count: msgs, last_message_at: Date.now() / 1000 - ago, started_at: 0 } }) as never

const liveRow = (title: string, status: string, activity: string) =>
  ({ group: status === 'working' ? 'working' : 'review', id: title, kind: 'live',
     item: { id: title, title, status, activity, current: status === 'working',
             last_active: Date.now() / 1000 - 60, started_at: 0, model: 'k3-256k' } }) as never

describe('assembled screens', () => {
  it('diff rows at 120 cols', async () => {
    $uiTheme.set(DEFAULT_THEME)
    const { DiffRow } = await import('../opentui/diffPanel.js')
    const { intraLineWordRanges } = await import('../lib/wordDiff.js')
    // The panel derives these the same way: one map keyed by row index.
    const ranges = intraLineWordRanges(DIFF.lines)
    const s = await testRender(
      <box flexDirection="column" height="100%" width="100%">
        {DIFF.lines.map((line, i) => {
          const words = ranges.get(i)

          return <DiffRow key={i} line={line} t={DEFAULT_THEME} {...(words ? { words } : {})} />
        })}
      </box>,
      { height: 14, width: 120 }
    )
    await s.flush()
    const frame = s.captureCharFrame()
    dump('diff rows', frame)

    // Cyan hunk headers, and code that keeps the ramp's prose step while the
    // +/− sign carries the state.
    expect(frame).toContain('@@ -118,7 +118,9 @@')
    expect(frame).toContain('▾ packages/runtime/scheduler.ts')
    act(() => s.renderer.destroy())
  })

  it('model picker keeps every row inside the frame', async () => {
    $uiTheme.set(DEFAULT_THEME)
    const { ModelPicker } = await import('../opentui/modelPicker.js')
    const { GatewayProvider } = await import('../app/gatewayContext.js')
    const providers = ['cc', 'codex', 'v5pk', 'Minimax', 'kimi-code', 'zai', 'opr', 'kir', 'Qwen-Plan']
    const request = vi.fn(async (method: string) =>
      method === 'model.options'
        ? {
            model: 'kimi-for-coding',
            providers: providers.map((slug, i) => ({
              slug, name: slug, is_current: slug === 'kimi-code', provider_type: 'custom', auth_type: 'env', index: i
            }))
          }
        : {
            // Deliberately interleaved, the way a ranked provider list
            // arrives: vendors must be gathered, not captioned every time
            // the vendor happens to change.
            models: [
              'meta/muse-spark-1.2-contributor',
              'deepseek/deepseek-v4-flash-vision-exp',
              'stealth/ox-alpha',
              'tencent/hy-mt2-1.8b',
              '~z-ai/glm-latest',
              'tencent/hy-mt2-7b',
              'z-ai/glm-5.3',
              'deepseek/deepseek-v4-pro-0813',
              'meta/llama-3.3-70b'
            ],
            source: 'remote'
          })
    const services = { gw: { request }, rpc: vi.fn() } as never

    const s = await testRender(
      <GatewayProvider value={services}>
        <ModelPicker onSelect={vi.fn()} sessionId="live" t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 40, width: 150 }
    )
    await act(async () => {
      await Bun.sleep(10)
    })
    await s.flush()
    const frame = s.captureCharFrame()
    dump('model picker', frame)
    const lines = frame.split('\n')

    // Nothing may paint outside the frame: every non-blank row between the
    // panel's top and bottom border must start and end with the border.
    const top = lines.findIndex(line => line.includes('╭'))
    const bottom = lines.findIndex(line => line.includes('╰'))

    expect(top).toBeGreaterThanOrEqual(0)
    expect(bottom).toBeGreaterThan(top)
    for (const line of lines.slice(top + 1, bottom)) {
      if (line.trim().length > 0) {
        expect(line.trim().startsWith('│')).toBe(true)
      }
    }
    // …and the last provider is inside it, not stranded under the footer.
    expect(lines.slice(top, bottom).some(line => line.includes('Qwen-Plan'))).toBe(true)
    // A family of one must not print a caption naming the single row under
    // it: `kimi-for-coding · 1` above `kimi-for-coding` cost a row per model
    // and is why a four-model provider showed two.
    expect(frame).not.toMatch(/· 1\b/)
    // Vendor-prefixed ids group by vendor, which is the only grouping that
    // means anything across a large catalogue.
    expect(frame).toContain('meta · 2')
    // Each family appears ONCE, with all of its members under it — an
    // unsorted list printed `deepseek · 14` three times over two rows.
    const captionLines = lines.filter(line => /\b(meta|deepseek|tencent|z-ai) · \d/.test(line))

    expect(captionLines).toHaveLength(new Set(captionLines.map(line => line.trim())).size)
    act(() => s.renderer.destroy())
  })

  it('terminals at 150x40', async () => {
    resetOverlayState()
    $uiTheme.set(DEFAULT_THEME)
    const { TerminalPanelOverlay } = await import('../opentui/terminalPanel.js')
    const { GatewayProvider } = await import('../app/gatewayContext.js')
    const now = Date.now()
    const rpc = vi.fn(async (method: string) =>
      method === 'terminal.list'
        ? { terminals: [
            { id: 't1', label: 'flake-hunter', command: 'bun test --filter runtime --repeat 50', cwd: '/repo', kind: 'background', running: true, exitCode: null, startedAt: now - 134_000, outputChars: 12_400, pid: 48377, canKill: true, canWrite: false, canInterrupt: true },
            { id: 't2', label: 'pdf-export', command: 'bun run build:pdf', cwd: '/repo', kind: 'background', running: false, exitCode: 137, startedAt: now - 900_000, endedAt: now - 480_000, outputChars: 900, pid: 41999, canKill: false, canWrite: false, canInterrupt: false },
            { id: 't3', label: 'you', command: 'zsh', cwd: '/etc', kind: 'pty', running: true, exitCode: null, startedAt: now - 13_260_000, outputChars: 40, pid: 39004, canKill: true, canWrite: true, canInterrupt: true },
            { id: 't4', label: 'session 4a91', command: 'bun test packages/runtime', cwd: '/repo', kind: 'background', running: false, exitCode: 0, startedAt: now - 20_000, endedAt: now - 9_800, outputChars: 300, pid: 41220, canKill: false, canWrite: false, canInterrupt: false }
          ] }
        : {})
    const services = { gw: { request: rpc }, rpc } as never
    const s = await testRender(
      <GatewayProvider value={services}>
        <TerminalPanelOverlay onClose={() => undefined} t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 40, width: 150 }
    )
    for (let pass = 0; pass < 6; pass += 1) {
      await Promise.resolve()
      await s.flush()
    }
    const frame = s.captureCharFrame()
    dump('terminals', frame)
    expect(frame).toContain('RUNNING')
    act(() => s.renderer.destroy())
  })

  it('agent view fills the screen and pairs the list with an inspector', async () => {
    $uiTheme.set(DEFAULT_THEME)
    const { SessionPicker } = await import('../opentui/sessionPicker.js')
    const { GatewayProvider } = await import('../app/gatewayContext.js')
    const now = Math.floor(Date.now() / 1000)
    const request = vi.fn(async (method: string) =>
      method === 'session.active'
        ? { sessions: [{ id: 'live-1', title: 'map this repo', status: 'working', activity: 'reading src/index.ts', current: true, last_active: now - 60, started_at: now - 600, model: 'anthropic/k3-256k' }] }
        : method === 'session.list'
          ? { sessions: Array.from({ length: 8 }, (_, i) => ({ id: `s${i}`, title: `Saved chat ${i}`, message_count: 4 + i, last_message_at: now - 3600 * (i + 1), started_at: 0 })) }
          : {})
    const services = { gw: { request }, rpc: async () => null } as never

    const s = await testRender(
      <GatewayProvider value={services}>
        <SessionPicker actions={{ activateLiveSession: () => undefined, resumeById: () => undefined } as never} t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 34, width: 150 }
    )
    // Drain the picker's two RPCs by flushing microtasks, not by sleeping:
    // a real timer here lengthens the whole suite and shifts the timing of
    // every neighbouring test.
    for (let pass = 0; pass < 6; pass += 1) {
      await Promise.resolve()
      await s.flush()
    }
    const frame = s.captureCharFrame()
    dump('agent view', frame)
    const lines = frame.split('\n')

    // Screen 03's body: list and inspector on screen together.
    expect(frame).toContain('SAVED CHATS')
    expect(frame).toContain('LATEST')
    // …and the panel FILLS the terminal: the footer is on the last painted
    // row, not stranded mid-screen with dead space under it.
    const footer = lines.findIndex(line => line.includes('Esc exit'))
    const lastPainted = lines.map(line => line.trim().length > 0).lastIndexOf(true)
    expect(footer).toBeGreaterThan(0)
    expect(lastPainted - footer).toBeLessThanOrEqual(1)
    act(() => s.renderer.destroy())
  })

  it('agent view rows at 150 cols', async () => {
    $uiTheme.set(DEFAULT_THEME)
    const { SessionListRow } = await import('../opentui/sessionPicker.js')
    const counts = { 'needs-input': 0, review: 0, saved: 3, working: 1 } as never
    const rows = [
      { row: liveRow('map this repo', 'working', 'reading src/index.ts'), first: true },
      { row: savedRow('Greeting and Introduction', 34, 240), first: true },
      { row: savedRow('DeepScan Project Analysis', 24, 21600), first: false },
      { row: savedRow('—', 286, 172800), first: false }
    ]
    const s = await testRender(
      <box flexDirection="column" height="100%" width="100%">
        {rows.map((r, i) => (
          <SessionListRow counts={counts} firstInGroup={r.first} key={i} maxLabelWidth={140} row={r.row} selected={false} t={DEFAULT_THEME} />
        ))}
      </box>,
      { height: 16, width: 150 }
    )
    await s.flush()
    const frame = s.captureCharFrame()
    dump('agent view', frame)

    // A saved chat is history, not an agent waiting to be reviewed.
    expect(frame).toContain('SAVED CHATS · 3')
    expect(frame).not.toContain('READY TO REVIEW · 3')
    // Quantities hang right on a leader, so a long list stacks into a column.
    expect(frame).toMatch(/Greeting and Introduction ·+ +34 msgs/)
    act(() => s.renderer.destroy())
  })

  it('F6 overlay pairs the list with an inspector', async () => {
    resetOverlayState()
    $uiTheme.set(DEFAULT_THEME)
    const { AgentPanelOverlay } = await import('../opentui/agentPanel.js')
    const s = await testRender(
      <AgentPanelOverlay
        history={[]}
        liveAgents={[
          agent({ id: 'a0', title: 'Auth Migration', status: 'queued', goal: 'move sessions table to the new schema', notes: ['overwrite packages/auth/schema.sql?'] }),
          agent(),
          agent({ id: 'a2', title: 'Docs Sweep', status: 'completed', summary: 'README and CLI help are back in sync.' })
        ]}
        onClose={() => undefined}
        t={DEFAULT_THEME}
      />,
      { height: 34, width: 150 }
    )
    await s.flush()
    const frame = s.captureCharFrame()
    dump('F6 overlay', frame)

    // Both panes on screen: the list keeps its groups while the inspector
    // shows the selected agent's detail.
    expect(frame).toContain('WORKING')
    expect(frame).toContain('TOOL CALLS')
    act(() => s.renderer.destroy())
  })

  it('rail at 150x40', async () => {
    resetOverlayState()
    $uiTheme.set(DEFAULT_THEME)
    const { AgentPanel } = await import('../opentui/agentPanel.js')
    const s = await testRender(
      <box flexDirection="row" height="100%" width="100%">
        <box flexGrow={1} />
        <box flexShrink={0} width={44}>
          <AgentPanel history={[]} liveAgents={[agent(), agent({ id: 'a2', title: 'Docs Sweep', status: 'completed', summary: 'README and CLI help are back in sync.' }), agent({ id: 'a3', title: 'Pdf Export', status: 'failed' })]} t={DEFAULT_THEME} />
        </box>
      </box>,
      { height: 40, width: 150 }
    )
    await s.flush()
    const frame = s.captureCharFrame()
    dump('rail', frame)

    // The title is the handle; a goal crammed beside it at 40 columns turned
    // every card into `● Structur...p the repo`.
    expect(frame).toContain('● Structure Analyzer')
    expect(frame).toContain('● Docs Sweep')
    // The rail drops goals but KEEPS its one live activity line — tying those
    // two together silenced the rail completely.
    expect(frame).toContain('reading src/index.ts')
    // A failed run collapses to one dim line and sorts last.
    expect(frame.indexOf('FAILED')).toBeGreaterThan(frame.indexOf('WORKING'))
    act(() => s.renderer.destroy())
  })

  it('session at 150x40', async () => {
    resetOverlayState()
    $uiTheme.set(DEFAULT_THEME)
    $uiState.set({
      ...$uiState.get(),
      busy: false,
      info: { cwd: '/repo', mode: 'code', model: 'claude-sonnet-4.6', version: '0.9.4' } as never,
      sessionTitle: 'scheduler drops tasks on worker death',
      sid: 'sess-4a91',
      usage: { context_max: 200000, context_used: 18200 } as never,
      sessionTabs: []
    })

    const rows = [
      row({ role: 'user', text: "the scheduler drops tasks when a worker dies mid-flight. find out why and fix it — don't touch the public API." }),
      row(
        { role: 'assistant', text: 'Three suspects: the requeue path, the heartbeat timeout, and the in-flight lease. Reading the scheduler first.',
          thinking: 'worker lifecycle and requeue semantics', thinkingTokens: 1400 },
        { rail: 'mid' }
      ),
      row(
        { kind: 'trail', role: 'system', text: '', tools: [
          buildToolTrailLine('read_file', 'packages/runtime/scheduler.ts', false, '', 0.2),
          buildToolTrailLine('grep', 'heartbeat|lease', false, '', 0.1),
          buildToolTrailLine('edit', 'packages/runtime/scheduler.ts', false, '+21 −6', 0.4)
        ] },
        { rail: 'end', turnSeconds: 11.4, turnTools: 6 }
      )
    ]

    const base = props(150)
    const p2 = {
      ...base,
      composer: { ...base.composer, empty: false },
      transcript: {
        ...base.transcript,
        virtualRows: rows,
        virtualHistory: { ...base.transcript.virtualHistory, end: rows.length, totalHeight: 12 }
      }
    } as never

    const s = await testRender(<AppLayout {...(p2 as AppLayoutProps)} />, { height: 40, width: 150 })
    await s.flush()
    const frame = s.captureCharFrame()
    dump('session', frame)
    const lines = frame.split('\n')
    const at = (needle: string) => lines.findIndex(line => line.includes(needle))

    // Bottom-anchored: a short conversation sits just above the composer, not
    // stranded at the top of a tall terminal with a dead gap underneath.
    const ledger = at('6 tools · 11.4s')
    const composerTop = lines.findIndex(line => line.includes('╭─'))
    expect(ledger).toBeGreaterThan(0)
    expect(composerTop - ledger).toBeLessThanOrEqual(2)

    // The user's own words carry the prompt glyph on a filled band.
    expect(frame).toContain('❯ the scheduler drops tasks')
    // The composer frame is intact — a per-side border inside it used to
    // paint the edge through the text (`╰─◆─code─mode─·─…─╯`).
    expect(frame).toContain('│ ◆ code mode')
    act(() => s.renderer.destroy())
  })

  it('home at 150x40', async () => {
    resetOverlayState()
    $uiTheme.set(DEFAULT_THEME)
    $uiState.set({
      ...$uiState.get(),
      busy: false,
      info: { cwd: '/repo', mode: 'plan', model: 'claude-sonnet-4.6', version: '0.9.4' } as never,
      usage: { context_max: 200000, context_used: 18200 } as never,
      sessionTabs: []
    })

    const s = await testRender(<AppLayout {...props(150)} />, { height: 40, width: 150 })
    await s.flush()
    const frame = s.captureCharFrame()
    dump('home', frame)

    // The wordmark dominates, as it does on the canvas: block letters, not a
    // letter-spaced word at body size.
    expect(frame).toContain('██╗  ██╗███████╗██████╗')
    expect(frame).toContain('Many agents, one terminal.')
    // Chips read as chips.
    expect(frame).toContain('START WITH')
    expect(frame).toContain('│ 1 ⏺ map this repo')
    // The composer never degrades and never loses its frame.
    expect(frame).toContain('│ ❯ describe a task, paste a stack trace, or press / for commands')
    act(() => s.renderer.destroy())
  })
})
