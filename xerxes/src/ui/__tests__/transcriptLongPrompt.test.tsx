// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// Bug A reproduction: a long multi-paragraph user prompt must not make the
// following assistant response collapse to ~one line / become invisible.
// The harness mirrors appLayout.tsx's transcript region: a sticky scrollbox,
// useVirtualHistory spacers, estimatedMsgHeight seeding, MessageLine rows,
// a live streaming block below the virtual rows, and live-tail auto-scroll.
import type { ScrollBoxRenderable } from '@opentui/core'
import { testRender } from '@opentui/react/test-utils'
import { act, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { describe, expect, it } from 'vitest'

import { useVirtualHistory } from '../hooks/useVirtualHistory.js'
import type { ScrollBoxHandle } from '../lib/terminalTypes.js'
import { estimatedMsgHeight, wrappedLines } from '../lib/virtualHeights.js'
import { MessageLine } from '../opentui/messageLine.js'
import { DEFAULT_THEME, themeForMode } from '../theme.js'
import type { Msg } from '../types.js'

const theme = themeForMode(DEFAULT_THEME, 'code')
const COLS = 80

const paragraph = (seed: string, sentences: number) =>
  Array.from({ length: sentences }, (_, i) => `${seed} sentence ${i + 1} wraps across the terminal width.`).join(' ')

const longUserText = [
  paragraph('First paragraph of the long user prompt', 6),
  paragraph('Second paragraph with more context for the agent', 6),
  paragraph('Third paragraph spelling out constraints', 6),
  paragraph('Fourth paragraph with the actual ask', 6)
].join('\n\n')

const longAssistantText = [
  '# Answer',
  '',
  paragraph('The assistant opens with a summary of the plan', 8),
  '',
  paragraph('Then it goes into implementation detail', 8),
  '',
  'ASSISTANT_TAIL_MARKER: the response ends here.'
].join('\n')

const scrollAdapter = (scrollbox: ScrollBoxRenderable): ScrollBoxHandle => ({
  getFreshScrollHeight: () => scrollbox.scrollHeight,
  getLastManualScrollAt: () => 0,
  getPendingDelta: () => 0,
  getScrollHeight: () => scrollbox.scrollHeight,
  getScrollTop: () => scrollbox.scrollTop,
  getViewportHeight: () => scrollbox.viewport.height,
  getViewportTop: () => scrollbox.scrollTop,
  isSticky: () => scrollbox.scrollTop >= Math.max(0, scrollbox.scrollHeight - scrollbox.viewport.height - 1),
  scrollBy: delta => scrollbox.scrollTo(Math.max(0, scrollbox.scrollTop + delta)),
  scrollTo: y => scrollbox.scrollTo(Math.max(0, y)),
  scrollToBottom: () => scrollbox.scrollTo(Math.max(0, scrollbox.scrollHeight - scrollbox.viewport.height)),
  scrollToElement: () => {},
  setClampBounds: () => {},
  subscribe: listener => {
    const notify = () => listener()

    scrollbox.verticalScrollBar.on('change', notify)
    scrollbox.on('layout-changed', notify)

    return () => {
      scrollbox.verticalScrollBar.off('change', notify)
      scrollbox.off('layout-changed', notify)
    }
  }
})

interface Turn {
  live: string
  liveSegments?: Msg[]
  msgs: Msg[]
}

function Transcript({ composerRows = 3, turn }: { composerRows?: number; turn: Turn }) {
  const scrollRef = useRef<ScrollBoxHandle | null>(null)
  const rows = useMemo(() => turn.msgs.map((msg, index) => ({ index, key: `k${index}:c${COLS}`, msg })), [turn.msgs])
  const estimateRowHeight = useCallback(
    (index: number) =>
      estimatedMsgHeight(rows[index]!.msg, COLS, {
        compact: false,
        details: true,
        thinkingVisible: true,
        toolsVisible: true
      }),
    [rows]
  )
  const virtual = useVirtualHistory(scrollRef, rows, COLS, { estimateHeight: estimateRowHeight })
  const scrollboxRef = useCallback(
    (scrollbox: ScrollBoxRenderable | null) => virtual.setScrollHandle(scrollbox ? scrollAdapter(scrollbox) : null),
    [virtual.setScrollHandle]
  )
  const visible = rows.slice(virtual.start, virtual.end)

  // LiveTailFollower: keep the tail pinned while the turn streams.
  useEffect(() => {
    if (turn.live) {
      queueMicrotask(() => scrollRef.current?.scrollToBottom())
    }
  })

  return (
    <box flexDirection="column" flexGrow={1} gap={1} minHeight={0} paddingX={2} paddingY={1}>
      <scrollbox
        ref={scrollboxRef}
        stickyScroll
        stickyStart="bottom"
        style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }}
        viewportCulling
      >
        <box flexDirection="column">
          {virtual.topSpacer > 0 ? <box flexShrink={0} height={virtual.topSpacer} /> : null}
          {visible.map(row => (
            <box flexDirection="column" flexShrink={0} key={row.key} ref={virtual.measureRef(row.key)}>
              <MessageLine msg={row.msg} t={theme} />
            </box>
          ))}
          {(turn.liveSegments ?? []).map((segment, index) => (
            <MessageLine key={`live-segment:${index}`} msg={segment} t={theme} />
          ))}
          {turn.live ? <MessageLine msg={{ role: 'assistant', text: turn.live }} t={theme} /> : null}
          {virtual.bottomSpacer > 0 ? <box flexShrink={0} height={virtual.bottomSpacer} /> : null}
        </box>
      </scrollbox>
      {/* Composer stand-in: tall while the long prompt is typed, shrinks on submit. */}
      <box flexDirection="column" flexShrink={0} height={composerRows}>
        <text>composer</text>
      </box>
    </box>
  )
}

const flushFrames = async (setup: Awaited<ReturnType<typeof testRender>>, passes = 12) => {
  for (let i = 0; i < passes; i++) {
    await act(async () => {
      await Bun.sleep(10)
      await setup.flush()
    })
  }
}

describe('transcript after a long multi-paragraph user prompt', () => {
  it('keeps the full assistant response visible after live streaming completes', async () => {
    let advance = (_turn: Turn | ((current: Turn) => Turn)) => {}

    function Harness() {
      const [turn, setTurn] = useState<Turn>({ live: '', msgs: [{ role: 'user', text: longUserText }] })

      advance = setTurn

      return <Transcript turn={turn} />
    }

    const setup = await testRender(<Harness />, { height: 24, width: COLS })

    try {
      await flushFrames(setup)
      expect(setup.captureCharFrame()).toContain('Fourth paragraph')

      // Live phase: the response streams below the virtual rows while the
      // live-tail follower keeps the scrollbox pinned to the bottom.
      for (const chunk of longAssistantText.split('\n')) {
        await act(async () => {
          advance(current => ({
            live: current.live ? `${current.live}\n${chunk}` : chunk,
            msgs: current.msgs
          }))
          await Bun.sleep(5)
          await setup.flush()
        })
      }

      // Turn completes: the streamed text becomes a persisted virtual row.
      act(() =>
        advance({
          live: '',
          msgs: [
            { role: 'user', text: longUserText },
            { role: 'assistant', text: longAssistantText }
          ]
        })
      )
      await flushFrames(setup)
      const after = setup.captureCharFrame()

      expect(after).toContain('ASSISTANT_TAIL_MARKER')
      // The response must fill the viewport, not collapse to one line: its
      // body paragraphs should be on screen together with the tail.
      expect(after).toContain('implementation detail')
    } finally {
      await setup.waitForVisualIdle()
      act(() => setup.renderer.destroy())
    }
  })

  it('keeps the response visible on a later turn with a thinking trail and prior history', async () => {
    let advance = (_turn: Turn | ((current: Turn) => Turn)) => {}

    const priorTurn: Msg[] = [
      { role: 'user', text: 'earlier short question' },
      { role: 'assistant', text: 'earlier short answer' }
    ]
    const thinkingTrail: Msg = {
      kind: 'trail',
      role: 'system',
      text: '',
      thinking: paragraph('Reasoning trace line', 20),
      thinkingTokens: 500
    }

    function Harness() {
      const [turn, setTurn] = useState<Turn>({
        live: '',
        msgs: [...priorTurn, { role: 'user', text: longUserText }]
      })

      advance = setTurn

      return <Transcript turn={turn} />
    }

    const setup = await testRender(<Harness />, { height: 24, width: COLS })

    try {
      await flushFrames(setup)
      expect(setup.captureCharFrame()).toContain('Fourth paragraph')

      // Live phase: the reasoning trail + response stream below the rows.
      act(() => advance(current => ({ ...current, liveSegments: [thinkingTrail] })))
      for (const chunk of longAssistantText.split('\n')) {
        await act(async () => {
          advance(current => ({
            ...current,
            live: current.live ? `${current.live}\n${chunk}` : chunk
          }))
          await Bun.sleep(5)
          await setup.flush()
        })
      }

      // Turn completes: reasoning trail and response become virtual rows.
      act(() =>
        advance({
          live: '',
          msgs: [...priorTurn, { role: 'user', text: longUserText }, thinkingTrail, { role: 'assistant', text: longAssistantText }]
        })
      )
      await flushFrames(setup)
      const after = setup.captureCharFrame()

      expect(after).toContain('ASSISTANT_TAIL_MARKER')
      expect(after).toContain('implementation detail')
    } finally {
      await setup.waitForVisualIdle()
      act(() => setup.renderer.destroy())
    }
  })

  it('keeps a very long markdown response measurable after a long prompt', async () => {
    let advance = (_turn: Turn | ((current: Turn) => Turn)) => {}

    const hugeAssistantText = [
      '# Plan',
      '',
      ...Array.from({ length: 12 }, (_, i) => paragraph(`Section ${i + 1} explains step ${i + 1}`, 8)),
      '',
      '```ts',
      ...Array.from({ length: 20 }, (_, i) => `const step${i} = compute(${i}) // implementation line ${i}`),
      '```',
      '',
      paragraph('Closing summary of the whole answer', 8),
      '',
      'HUGE_TAIL_MARKER: done.'
    ].join('\n\n')

    function Harness() {
      const [turn, setTurn] = useState<Turn>({ live: '', msgs: [{ role: 'user', text: longUserText }] })

      advance = setTurn

      return <Transcript turn={turn} />
    }

    const setup = await testRender(<Harness />, { height: 24, width: COLS })

    try {
      await flushFrames(setup)

      act(() =>
        advance({
          live: '',
          msgs: [
            { role: 'user', text: longUserText },
            { role: 'assistant', text: hugeAssistantText }
          ]
        })
      )
      await flushFrames(setup, 30)
      const after = setup.captureCharFrame()

      expect(after).toContain('HUGE_TAIL_MARKER')
      expect(after).toContain('Closing summary')
    } finally {
      await setup.waitForVisualIdle()
      act(() => setup.renderer.destroy())
    }
  })

  it('estimates CJK/emoji text by display cells, not UTF-16 units', () => {
    // 46 CJK characters occupy 92 display cells (2 each) but only 46 UTF-16
    // units — at width 40 the truth is 3 rows; the old unit-counting
    // estimate said 2 and under-measured every CJK row by ~2×.
    const cjk = '漢字仮名交じり文テスト文字列入力幅計算確認用文章'.repeat(2)
    const emoji = '👨‍👩‍👧‍👦🎉🚀✨🔥💡🎯📌'.repeat(4)

    expect(wrappedLines(cjk, 40)).toBe(3)
    expect(wrappedLines(emoji, 8)).toBe(8)
  })

  it('keeps the response visible after a long CJK user prompt', async () => {
    let advance = (_turn: Turn | ((current: Turn) => Turn)) => {}

    const cjkUserText = Array.from({ length: 4 }, (_, i) =>
      `第${i + 1}段落です。`.concat('日本語の長い文章をここに書いて端末幅での折り返しを確認します。'.repeat(4))
    ).join('\n\n')

    function Harness() {
      const [turn, setTurn] = useState<Turn>({ live: '', msgs: [{ role: 'user', text: cjkUserText }] })

      advance = setTurn

      return <Transcript turn={turn} />
    }

    const setup = await testRender(<Harness />, { height: 24, width: COLS })

    try {
      await flushFrames(setup)

      act(() =>
        advance({
          live: '',
          msgs: [
            { role: 'user', text: cjkUserText },
            { role: 'assistant', text: longAssistantText }
          ]
        })
      )
      await flushFrames(setup)
      const after = setup.captureCharFrame()

      expect(after).toContain('ASSISTANT_TAIL_MARKER')
    } finally {
      await setup.waitForVisualIdle()
      act(() => setup.renderer.destroy())
    }
  })

  it('keeps the response visible when the composer shrinks after submitting the long prompt', async () => {
    let advance = (_turn: Turn | ((current: Turn) => Turn)) => {}
    let setRows = (_rows: number) => {}

    function Harness() {
      const [turn, setTurn] = useState<Turn>({ live: '', msgs: [] })
      const [composerRows, setComposerRows] = useState(10)

      advance = setTurn
      setRows = setComposerRows

      return <Transcript composerRows={composerRows} turn={turn} />
    }

    const setup = await testRender(<Harness />, { height: 30, width: COLS })

    try {
      await flushFrames(setup)

      // Submit: the tall composer collapses and the long prompt lands.
      act(() => {
        setRows(3)
        advance({ live: '', msgs: [{ role: 'user', text: longUserText }] })
      })
      await flushFrames(setup)

      // Stream the response in chunks.
      for (const chunk of longAssistantText.split('\n')) {
        await act(async () => {
          advance(current => ({
            ...current,
            live: current.live ? `${current.live}\n${chunk}` : chunk
          }))
          await Bun.sleep(5)
          await setup.flush()
        })
      }

      act(() =>
        advance({
          live: '',
          msgs: [
            { role: 'user', text: longUserText },
            { role: 'assistant', text: longAssistantText }
          ]
        })
      )
      await flushFrames(setup)
      const after = setup.captureCharFrame()

      expect(after).toContain('ASSISTANT_TAIL_MARKER')
      expect(after).toContain('implementation detail')
    } finally {
      await setup.waitForVisualIdle()
      act(() => setup.renderer.destroy())
    }
  })

  it('reports sane heights for a long user + long assistant pair', () => {
    const userHeight = estimatedMsgHeight({ role: 'user', text: longUserText }, COLS, {
      compact: false,
      details: true
    })
    const assistantHeight = estimatedMsgHeight({ role: 'assistant', text: longAssistantText }, COLS, {
      compact: false,
      details: true
    })

    // The estimator must not collapse a long message to a single row or
    // inflate it beyond the cap; both messages are dozens of rows at 80 cols.
    expect(userHeight).toBeGreaterThan(10)
    expect(userHeight).toBeLessThan(200)
    expect(assistantHeight).toBeGreaterThan(10)
    expect(assistantHeight).toBeLessThan(200)
  })
})
