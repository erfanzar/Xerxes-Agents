// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// F7 diff viewer: hotkey toggles the overlay, the overlay renders parsed diff
// rows (with word-level highlights on paired edits), closes on F7/Esc/q,
// Shift+Ctrl+←/→ resizes the shared panel width, and the file index carries
// per-file +/- badges whose selection follows the viewport.
import { parseColor, type CapturedFrame, type CapturedLine } from '@opentui/core'
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getPanelWidthDelta, resetPanelWidth } from '../app/panelSizeStore.js'
import type { GitDiffResult } from '../lib/gitDiff.js'
import { blendColors, WORD_TINT_ADD, WORD_TINT_DEL } from '../lib/wordDiff.js'
import { DiffPanelHotkey, DiffPanelOverlay, toRenderableColor } from '../opentui/diffPanel.js'
import { DEFAULT_THEME } from '../theme.js'

const DIFF_RESULT = {
  kind: 'ok' as const,
  diff: {
    deletions: 1,
    files: 1,
    insertions: 1,
    lines: [
      { kind: 'file' as const, text: 'src/a.ts' },
      { kind: 'hunk' as const, text: '@@ -1,1 +1,1 @@' },
      { kind: 'del' as const, oldLine: 1, text: '-old' },
      { kind: 'add' as const, newLine: 1, text: '+new' }
    ],
    truncated: false,
    untracked: ['draft.ts'],
    untrackedTruncated: false
  }
}

interface ContextRow {
  kind: 'context'
  newLine: number
  oldLine: number
  text: string
}

const contextRows = (startLine: number, count: number): ContextRow[] =>
  Array.from({ length: count }, (_, offset) => ({
    kind: 'context' as const,
    newLine: startLine + offset,
    oldLine: startLine + offset,
    text: ` padding row ${offset}`
  }))

// Three files tall enough to overflow the viewport at the test render size,
// so scroll keys can actually move the top visible row across file sections.
const MULTI_FILE_DIFF: GitDiffResult = {
  kind: 'ok',
  diff: {
    deletions: 2,
    files: 3,
    insertions: 3,
    lines: [
      { kind: 'file', text: 'src/auth/middleware.ts' },
      { kind: 'hunk', text: '@@ -1,2 +1,2 @@' },
      { kind: 'del', oldLine: 1, text: '-const token = raw.split(" ")[1]' },
      { kind: 'add', newLine: 1, text: '+const token = parseTokenHeader(raw)' },
      { kind: 'context', newLine: 2, oldLine: 2, text: ' return verify(token)' },
      { kind: 'file', text: 'test/auth.test.ts' },
      { kind: 'hunk', text: '@@ -9,0 +10,1 @@' },
      { kind: 'add', newLine: 10, text: '+it("expires the token")' },
      ...contextRows(11, 14),
      { kind: 'file', text: 'docs/guide.md' },
      { kind: 'hunk', text: '@@ -2,1 +2,1 @@' },
      { kind: 'del', oldLine: 2, text: '-old docs line' },
      { kind: 'add', newLine: 2, text: '+new docs line' },
      ...contextRows(3, 26)
    ],
    truncated: false,
    untracked: [],
    untrackedTruncated: false
  }
}

let activeDiffResult: GitDiffResult = DIFF_RESULT

vi.mock('../lib/gitDiff.js', async importOriginal => {
  const original = await importOriginal<typeof import('../lib/gitDiff.js')>()
  return {
    ...original,
    collectGitDiff: vi.fn(async () => activeDiffResult)
  }
})

afterEach(() => {
  activeDiffResult = DIFF_RESULT
})

describe('DiffPanelHotkey', () => {
  afterEach(() => {
    resetPanelWidth()
  })

  it('toggles with F7', async () => {
    const transitions: boolean[] = []
    const setup = await testRender(
      <box>
        <DiffPanelHotkey disabled={false} onToggle={open => transitions.push(open)} open={false} />
        <text>ready</text>
      </box>,
      { height: 4, width: 30 }
    )

    try {
      setup.mockInput.pressKey('F7')
      await setup.flush()
      expect(transitions).toEqual([true])
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('ignores F7 when disabled', async () => {
    const transitions: boolean[] = []
    const setup = await testRender(
      <box>
        <DiffPanelHotkey disabled onToggle={open => transitions.push(open)} open={false} />
        <text>ready</text>
      </box>,
      { height: 4, width: 30 }
    )

    try {
      setup.mockInput.pressKey('F7')
      await setup.flush()
      expect(transitions).toEqual([])
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})

describe('DiffPanelOverlay', () => {
  afterEach(() => {
    resetPanelWidth()
  })

  it('renders the diff rows and closes with F7', async () => {
    let closed = 0
    const setup = await testRender(<DiffPanelOverlay onClose={() => closed++} t={DEFAULT_THEME} />, {
      height: 24,
      width: 80
    })

    await act(async () => {
      await Bun.sleep(10)
    })
    await setup.flush()

    try {
      const frame = setup.captureCharFrame()
      // The header names the base and what it points at — "12 files changed"
      // means nothing until you know changed against what.
      expect(frame).toContain('working tree')
      expect(frame).toContain('vs HEAD')
      expect(frame).toContain('+1')
      expect(frame).toContain('−1')
      expect(frame).toContain('1 file')
      expect(frame).toContain('src/a.ts')
      expect(frame).toContain('OLD')
      expect(frame).toContain('NEW')
      expect(frame).toContain('UNTRACKED FILES')

      setup.mockInput.pressKey('F7')
      await act(async () => {
        await setup.flush()
      })
      expect(closed).toBe(1)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('resizes the shared panel width with Shift+Ctrl+Left/Right', async () => {
    const setup = await testRender(<DiffPanelOverlay onClose={() => {}} t={DEFAULT_THEME} />, {
      height: 24,
      width: 80
    })

    await act(async () => {
      await Bun.sleep(10)
    })
    await setup.flush()

    try {
      expect(getPanelWidthDelta()).toBe(0)
      await act(async () => {
        setup.mockInput.pressArrow('right', { ctrl: true, shift: true })
      })
      await setup.flush()
      expect(getPanelWidthDelta()).toBe(4)
      await act(async () => {
        setup.mockInput.pressArrow('left', { ctrl: true, shift: true })
        setup.mockInput.pressArrow('left', { ctrl: true, shift: true })
      })
      await setup.flush()
      expect(getPanelWidthDelta()).toBe(-4)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})

// ── Mockup 07: file-index badges, word highlights, viewport follow ──────

const channelEq = (a: number[], b: number[]): boolean => a[0] === b[0] && a[1] === b[1] && a[2] === b[2]

const rgb = (color: string): number[] => {
  // Theme roles may be rgb()/ansi256(); coerce the same way the panel does
  // so expectations describe what OpenTUI is actually asked to render.
  const native = toRenderableColor(color)
  const ints = (typeof native === 'string' ? parseColor(native) : native).toInts()
  return [ints[0]!, ints[1]!, ints[2]!]
}

const hasFg = (span: { fg: { toInts(): number[] } }, color: string): boolean =>
  channelEq(span.fg.toInts(), rgb(color))

const hasBg = (span: { bg: { toInts(): number[] } }, color: string): boolean =>
  channelEq(span.bg.toInts(), rgb(color))

const hasFgBg = (
  span: { bg: { toInts(): number[] }; fg: { toInts(): number[] } },
  fg: string,
  bg: string
): boolean => hasFg(span, fg) && hasBg(span, bg)

const lineWith = (frame: CapturedFrame, needle: string): CapturedLine | undefined =>
  frame.lines.find(line => line.spans.some(span => span.text.includes(needle)))

const lineText = (line: CapturedLine): string => line.spans.map(span => span.text).join('')

describe('DiffPanel file index and word highlights (mockup 07)', () => {
  afterEach(() => {
    resetPanelWidth()
    activeDiffResult = DIFF_RESULT
  })

  const renderWide = async () => {
    // Wide enough for the index pane (>96) and tall enough that ~50 diff
    // rows overflow, so scroll keys can move the top visible row.
    const setup = await testRender(<DiffPanelOverlay onClose={() => {}} t={DEFAULT_THEME} />, {
      height: 40,
      width: 120
    })
    await act(async () => {
      await Bun.sleep(10)
    })
    await setup.flush()

    return setup
  }

  it('renders per-file +/− badge chips with word roles on diff tints', async () => {
    activeDiffResult = MULTI_FILE_DIFF
    const setup = await renderWide()

    try {
      const frame = setup.captureCharFrame()
      expect(frame).toContain('FILE INDEX · 3')

      const caps = setup.captureSpans()
      const t = DEFAULT_THEME

      // middleware.ts has +1 −1: both totals, in the done/failed voices.
      // They are plain text now rather than tinted chips — the canvas keeps
      // filled grounds for state cards, and a per-file count is not a state.
      const middleware = lineWith(caps, 'middleware.ts')
      expect(middleware).toBeDefined()
      expect(lineText(middleware!)).toMatch(/\+1/)
      expect(lineText(middleware!)).toMatch(/−1/)
      expect(middleware!.spans.some(span => hasFg(span, t.ds.done))).toBe(true)
      expect(middleware!.spans.some(span => hasFg(span, t.ds.failed))).toBe(true)

      // auth.test.ts only adds — no − badge (the mockup's +38-only row).
      const testFile = lineWith(caps, 'auth.test.ts')
      expect(testFile).toBeDefined()
      expect(lineText(testFile!)).toMatch(/\+1/)
      expect(lineText(testFile!)).not.toMatch(/−\d/)

      // Counts are per file: three +1 chips total, zero-counts never shown.
      const allSpans = caps.lines.flatMap(line => line.spans)
      const addBadges = allSpans.filter(
        span => /^\s*\+1\s*$/.test(span.text) && hasFg(span, t.ds.done)
      )
      expect(addBadges).toHaveLength(3)
      expect(allSpans.filter(span => /[-−]\s*0$/.test(span.text.trim()))).toHaveLength(0)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('highlights exactly the changed words within paired del/add lines', async () => {
    activeDiffResult = MULTI_FILE_DIFF
    const setup = await renderWide()

    try {
      const caps = setup.captureSpans()
      const t = DEFAULT_THEME
      // Pre-composited token pairs rather than a blend computed at paint
      // time, so the highlight's ground and its text can never drift apart.
      const addWordTint = t.color.diffAddedWordBg
      const delWordTint = t.color.diffRemovedWordBg

      // Add side: the rewritten call carries the word role on its tint…
      const addLine = lineWith(caps, 'parseTokenHeader')
      expect(addLine).toBeDefined()
      const addWord = addLine!.spans.find(span => span.text.includes('parseTokenHeader'))
      expect(addWord).toBeDefined()
      expect(hasFg(addWord!, t.color.diffAddedWord)).toBe(true)
      expect(hasBg(addWord!, addWordTint)).toBe(true)

      // …while the kept prefix stays on the ramp's prose step: the row tint
      // says the line changed, the word highlight says which part.
      const keptAdd = addLine!.spans.find(span => span.text.includes('const token ='))
      expect(keptAdd).toBeDefined()
      expect(hasFg(keptAdd!, t.ds.prose)).toBe(true)
      expect(hasBg(keptAdd!, addWordTint)).toBe(false)

      // Del side mirrors with the removed-word role on its own tint.
      const delLine = lineWith(caps, 'raw.split')
      expect(delLine).toBeDefined()
      const delWord = delLine!.spans.find(span => span.text.includes('raw.split'))
      expect(delWord).toBeDefined()
      expect(hasFg(delWord!, t.color.diffRemovedWord)).toBe(true)
      expect(hasBg(delWord!, delWordTint)).toBe(true)

      // The second rewrite pair highlights just 'old'/'new' at line start.
      // Word spans split the row, so match on the joined line text.
      const docsDelLine = caps.lines.find(line => /−\s*old docs/.test(lineText(line)))
      expect(docsDelLine).toBeDefined()
      const docsWord = docsDelLine!.spans.find(span => span.text === 'old')
      expect(docsWord).toBeDefined()
      expect(hasFgBg(docsWord!, t.color.diffRemovedWord, delWordTint)).toBe(true)

      // Hunk headers stay cyan.
      const hunkLine = lineWith(caps, '@@ -1,2')
      expect(hunkLine).toBeDefined()
      expect(hunkLine!.spans.some(span => hasFg(span, t.color.diffHunk))).toBe(true)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('follows the viewport with the selection while [ ] jumps stay authoritative', async () => {
    activeDiffResult = MULTI_FILE_DIFF
    const setup = await renderWide()

    try {
      const selected = (): Record<string, boolean> => {
        const caps = setup.captureSpans()
        const state: Record<string, boolean> = {}

        // Only index-pane rows carry a ▸ marker; the scrollbox's own file
        // headers share the basenames and must not count.
        const paneRow = (base: string): CapturedLine | undefined =>
          caps.lines.find(
            line => line.spans.some(span => span.text.trim() === '▸') && line.spans.some(span => span.text.includes(base))
          )

        for (const base of ['middleware.ts', 'auth.test.ts', 'guide.md']) {
          const line = paneRow(base)
          state[base] = Boolean(
            line && line.spans.some(span => span.text.trim() === '▸' && hasFg(span, DEFAULT_THEME.color.accent))
          )
        }

        return state
      }

      // Keyboard-driven selection updates must run inside act.
      const sendKey = async (key: () => void) => {
        await act(async () => {
          key()
        })
        await setup.flush()
      }

      // The initial viewport sits in the first file's section.
      expect(selected()).toEqual({ 'auth.test.ts': false, 'guide.md': false, 'middleware.ts': true })

      // 23 arrow-downs push the top visible row into docs/guide.md's section.
      await sendKey(() => {
        for (let i = 0; i < 23; i += 1) setup.mockInput.pressArrow('down')
      })
      expect(selected()).toEqual({ 'auth.test.ts': false, 'guide.md': true, 'middleware.ts': false })

      // Home returns the follow to the first section…
      await sendKey(() => setup.mockInput.pressKey('HOME'))
      expect(selected()['middleware.ts']).toBe(true)

      // …while explicit [ ] jumps remain authoritative.
      await sendKey(() => setup.mockInput.pressKey(']'))
      expect(selected()['auth.test.ts']).toBe(true)

      await sendKey(() => setup.mockInput.pressKey('['))
      expect(selected()['middleware.ts']).toBe(true)
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
