// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { RGBA, type ScrollBoxRenderable } from '@opentui/core'
import { useStore } from '@nanostores/react'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { workspaceFileDiff } from '../lib/workspaceDiffPreview.js'
import { useOptionalGateway } from '../app/gatewayContext.js'

import {
  $panelWidthDelta,
  adjustPanelWidth,
  PANEL_WIDTH_STEP,
  withPanelWidthDelta
} from '../app/panelSizeStore.js'

import { OVERLAY_PANEL_SPECS, overlayPanelSize } from './overlayLayout.js'
import { fileIndexFollowingRow, indexDiffFiles } from '../lib/diffFiles.js'
import { collectGitDiff, type DiffLine, type GitDiffResult } from '../lib/gitDiff.js'
import { intraLineWordRanges, type WordRange } from '../lib/wordDiff.js'
import type { Theme } from '../theme.js'

import { clipPath, GLYPH } from '../domain/nocturne.js'

import { isPageDownKey, isPageUpKey, PAGE_KEY_HINT } from '../lib/pageKeys.js'

import { GroupCaption } from './nocturne.js'
import { Box, Span, Text } from './primitives.js'

/**
 * F7 git diff viewer: shows what the worktree changed (staged + unstaged vs
 * HEAD) with per-file sections, colored +/- rows with word-level highlights
 * on the paired lines that actually changed, a file index whose badges count
 * each file's edits and whose selection follows the viewport, and the
 * untracked-file list. Mirrors the agents overlay's layout and key
 * discipline: F7/Esc/q close, arrows/PageUp/PageDown scroll,
 * [/] jump between files, r re-runs the diff.
 */

type KeyEvent = Parameters<Parameters<typeof useKeyboard>[0]>[0]

const consumeKey = (event: KeyEvent) => {
  event.preventDefault()
  event.stopPropagation()
}

const RGB_FUNCTION = /^rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$/i
const ANSI_INDEXED = /^ansi256\((\d+)\)$/i

/**
 * OpenTUI's string color parser only understands hex and CSS names, while
 * theme roles legitimately arrive as `rgb(...)` (the diff word roles) or
 * `ansi256(n)` (ANSI-normalized light terminals). Coerce those to a native
 * RGBA so raw spans/text render the role instead of the parser's magenta
 * error fallback. Exported for the color assertions in the overlay tests.
 */
export const toRenderableColor = (color: string): RGBA | string => {
  const fn = RGB_FUNCTION.exec(color.trim())

  if (fn) {
    return RGBA.fromInts(Number(fn[1]), Number(fn[2]), Number(fn[3]))
  }

  const indexed = ANSI_INDEXED.exec(color.trim())

  if (indexed) {
    return RGBA.fromIndex(Number(indexed[1]))
  }

  return color
}

export function DiffPanelHotkey({
  disabled,
  open,
  onToggle
}: {
  disabled: boolean
  open: boolean
  onToggle: (open: boolean) => void
}) {
  useKeyboard(event => {
    if (disabled || event.name !== 'f7') return
    consumeKey(event)
    onToggle(!open)
  })

  return null
}

/** Enough for a basename plus the selection marker. */
const FILE_PANE_WIDTH = 36
/** Below this the index would be taking columns the hunks need more. */
const FILE_PANE_MIN_PANEL_WIDTH = 104

/** Files are shown by basename; the full path is in the hunk header. */
const baseName = (path: string) => path.split('/').at(-1) || path

const GUTTER_WIDTH = 6

/**
 * Content rows the scrollbox shows above diff.lines[0] (the OLD/NEW gutter
 * header). Scroll positions must shed this offset before they index into
 * the parsed rows.
 */
const DIFF_HEADER_ROWS = 1

/** Wheel scrolling bypasses the keyboard path; this keeps following it cheap. */
const SCROLL_FOLLOW_INTERVAL_MS = 200

const lineNumber = (value: number | undefined): string =>
  value === undefined ? ' '.repeat(GUTTER_WIDTH) : String(value).padStart(GUTTER_WIDTH)

const codeText = (line: DiffLine): string => line.text.slice(1) || ' '

function DiffCodeRow({ line, t, words }: { line: DiffLine; t: Theme; words?: WordRange }) {
  const added = line.kind === 'add'
  const removed = line.kind === 'del'
  // The CODE keeps its ramp step and the +/− SIGN carries the state. Painting
  // the whole line green or red made a diff read as two blocks of coloured
  // text rather than as code with marks on it, and left nothing louder for
  // the word-level highlight to be.
  const marker = added ? '+' : removed ? '−' : ' '
  const markerColor = added ? t.ds.done : removed ? t.ds.failed : t.ds.separator
  const foreground = added || removed ? t.ds.prose : t.ds.diffContext
  const background = added ? t.color.diffAddedBg : removed ? t.color.diffRemovedBg : undefined
  const code = codeText(line)

  // "Word-level highlight marks the substring that actually moved; the line
  // tint alone makes you re-read a whole line to find one renamed symbol."
  // The tokens are pre-composited pairs — a stronger ground plus a lighter
  // foreground — rather than a blend computed here, so the two halves of the
  // highlight can never drift apart.
  const wordHighlight =
    words && (added || removed)
      ? {
          bg: added ? t.color.diffAddedWordBg : t.color.diffRemovedWordBg,
          end: Math.min(words.end, code.length),
          fg: toRenderableColor(added ? t.color.diffAddedWord : t.color.diffRemovedWord),
          start: Math.min(words.start, code.length)
        }
      : null

  return (
    <Box backgroundColor={background} flexDirection="row">
      <Text color={t.ds.separator}>{lineNumber(line.oldLine)} </Text>
      <Text color={t.ds.separator}>{lineNumber(line.newLine)} </Text>
      <Text color={foreground} wrap="truncate-end">
        <Span bold={added || removed} color={markerColor}>{marker} </Span>
        {wordHighlight ? (
          <>
            {code.slice(0, wordHighlight.start)}
            <span bg={wordHighlight.bg} fg={wordHighlight.fg}>
              {code.slice(wordHighlight.start, wordHighlight.end)}
            </span>
            {code.slice(wordHighlight.end)}
          </>
        ) : (
          code
        )}
      </Text>
    </Box>
  )
}

/**
 * Panel resize chord: Shift plus an action modifier (Cmd where the terminal
 * forwards Super, Ctrl/Option as the portable fallback) + Left/Right.
 */
export const isPanelResizeKey = (event: KeyEvent): boolean =>
  (event.name === 'left' || event.name === 'right') &&
  event.shift === true &&
  (event.super === true || event.ctrl === true || event.meta === true)

/** Exported for the assembled-screen test; the panel is its only user. */
export function DiffRow({ line, t, words }: { line: DiffLine; t: Theme; words?: WordRange }) {
  switch (line.kind) {
    case 'file':
      return (
        <Box backgroundColor={t.ds.diffFoldBg} flexDirection="row" paddingX={1}>
          <Text wrap="truncate-end">
            <Span color={t.color.accent}>{`${GLYPH.expanded} `}</Span>
            <Span color={t.ds.title}>{line.text}</Span>
          </Text>
        </Box>
      )
    case 'hunk':
      // Cyan is this line and nothing else in the whole product, so a fold
      // boundary can never be mistaken for a state dot.
      return (
        <Box backgroundColor={t.ds.diffHunkBg} flexDirection="row">
          <Text color={t.ds.separator}>{' '.repeat((GUTTER_WIDTH + 1) * 2)}</Text>
          <Text color={t.color.diffHunk} wrap="truncate-end">  {line.text}</Text>
        </Box>
      )
    case 'add':
    case 'del':
    case 'context':
      return <DiffCodeRow line={line} t={t} words={words} />
    case 'meta':
      return <Text color={t.ds.caption} italic wrap="truncate-end">  {line.text}</Text>
  }
}

export function DiffPanelOverlay({
  cwd,
  onClose,
  t
}: {
  cwd?: string | undefined
  onClose: () => void
  t: Theme
}) {
  const gateway = useOptionalGateway()
  const scrollRef = useRef<ScrollBoxRenderable | null>(null)
  const indexScrollRef = useRef<ScrollBoxRenderable | null>(null)
  const [result, setResult] = useState<GitDiffResult | null>(null)
  const [allResult, setAllResult] = useState<GitDiffResult | null>(null)
  const [detailPath, setDetailPath] = useState('')
  const overviewScroll = useRef(0)
  const [loading, setLoading] = useState(true)
  const [fileIdx, setFileIdx] = useState(0)
  const generation = useRef(0)
  const untrackedLimit = useRef(50)
  const [listingError, setListingError] = useState('')
  const { height: terminalHeight, width: terminalWidth } = useTerminalDimensions()
  useStore($panelWidthDelta)
  const { height: panelHeight, width: fittedWidth } = overlayPanelSize(
    { height: terminalHeight, width: terminalWidth },
    OVERLAY_PANEL_SPECS.diff
  )
  const panelWidth = withPanelWidthDelta(fittedWidth, terminalWidth)

  const reload = useCallback((path?: string, untracked = false) => {
    setDetailPath(path ?? '')
    const gen = ++generation.current
    setLoading(true)
    void (gateway ? (path ? workspaceFileDiff((method, params) => gateway.rpc(method, params), path, untracked) : gateway.rpc('workspace.diff', {untracked_limit: untrackedLimit.current})).then(value => {
      if (!value || typeof value !== 'object' || !('kind' in value) || !['ok', 'clean', 'error'].includes(String(value.kind))) throw new Error('Invalid workspace diff response')
      return value as GitDiffResult
    }) : collectGitDiff({ cwd: cwd ?? '', includeUntracked: true, maxUntracked: untrackedLimit.current, ...(path ? { path } : {}) }))
      .then(next => {
        if (generation.current === gen) {
          setResult(next)
          if (!path) setAllResult(next)
          else scrollRef.current?.scrollTo(0)
          setLoading(false)
        }
      })
      .catch((error: unknown) => {
        if (generation.current === gen) {
          setResult({ kind: 'error', message: error instanceof Error ? error.message : String(error) })
          setLoading(false)
        }
      })
  }, [cwd, gateway])

  useEffect(() => {
    reload()
    return () => { generation.current += 1 }
  }, [reload])

  const page = 10

  useKeyboard(event => {
    if (isPanelResizeKey(event)) {
      consumeKey(event)
      adjustPanelWidth(event.name === 'right' ? PANEL_WIDTH_STEP : -PANEL_WIDTH_STEP)
    } else if (event.name === 'escape' || event.name === 'f7' || event.sequence === 'q') {
      consumeKey(event)
      onClose()
    } else if (event.name === 'backspace' && detailPath) {
      consumeKey(event); returnToOverview()
    } else if (event.sequence === 'm' && overview?.untrackedTruncated) {
      consumeKey(event); loadMoreUntracked()
    } else if (event.sequence === 'r') {
      consumeKey(event)
      reload(detailPath || undefined, overview?.untracked.includes(detailPath) ?? false)
    } else if (event.name === 'left' || event.name === 'right') {
      consumeKey(event)
      scrollRef.current?.scrollBy({ x: event.name === 'right' ? 12 : -12, y: 0 })
    } else if (event.name === 'up') {
      consumeKey(event)
      scrollRef.current?.scrollBy(-1)
      syncSelectionFromScroll()
    } else if (event.name === 'down') {
      consumeKey(event)
      scrollRef.current?.scrollBy(1)
      syncSelectionFromScroll()
    } else if (isPageUpKey(event)) {
      consumeKey(event)
      scrollRef.current?.scrollBy(-page)
      syncSelectionFromScroll()
    } else if (isPageDownKey(event)) {
      consumeKey(event)
      scrollRef.current?.scrollBy(page)
      syncSelectionFromScroll()
    } else if (event.sequence === ']' || event.sequence === '[') {
      // Jump between files. The parsed diff knows the row each file starts
      // at, so this is a seek rather than a page-until-you-find-it. Explicit
      // selection stays authoritative here; the viewport-follow below lands
      // on the same file anyway because content rows sit one gutter header
      // below the parsed rows.
      if (files.length) {
        const next = Math.max(0, Math.min(files.length - 1, fileIdx + (event.sequence === ']' ? 1 : -1)))

        openFile(next)
      }
    } else if (event.name === 'home') {
      consumeKey(event)
      scrollRef.current?.scrollTo(0)
      syncSelectionFromScroll()
    } else if (event.name === 'end') {
      consumeKey(event)
      scrollRef.current?.scrollTo(Number.MAX_SAFE_INTEGER)
      syncSelectionFromScroll()
    }
  })

  const diff = result?.kind === 'ok' ? result.diff : null
  const overview = allResult?.kind === 'ok' ? allResult.diff : diff
  const files = useMemo(() => {
    const indexed = overview ? indexDiffFiles(overview.lines) : []
    for (const name of overview?.untracked ?? []) if (!indexed.some(file => file.name === name)) indexed.push({ name, line: -1, insertions: 0, deletions: 0 })
    return indexed
  }, [overview])
  function openFile(index: number) {
    const file = files[index]
    if (!file) return
    if (!overview?.truncated && file.line >= 0) {
      setFileIdx(index)
      if (detailPath) { generation.current += 1; setDetailPath(''); setResult(allResult); setLoading(false) }
      setTimeout(() => scrollRef.current?.scrollTo(file.line + DIFF_HEADER_ROWS), 0)
      return
    }
    if (!detailPath) overviewScroll.current = scrollRef.current?.scrollTop ?? 0
    setFileIdx(index); reload(file.name, overview?.untracked.includes(file.name) ?? false)
  }
  function returnToOverview() {
    generation.current += 1
    setDetailPath(''); setResult(allResult); setLoading(false)
    setTimeout(() => scrollRef.current?.scrollTo(overviewScroll.current), 0)
  }
  function loadMoreUntracked() {
    if (loading || listingError) return
    if (untrackedLimit.current >= 10000) { setListingError('New-file lists are limited to 10,000 entries.'); return }
    untrackedLimit.current = Math.min(10000, untrackedLimit.current + 100)
    reload()
  }
  // Mockup 07: "word highlights answer what exactly changed in this line".
  // Recomputed per refresh, keyed by row index into diff.lines.
  const wordRanges = useMemo(
    () => (diff ? intraLineWordRanges(diff.lines) : new Map<number, WordRange>()),
    [diff]
  )
  // Only when the panel is wide enough that the index is not stealing the
  // columns the hunks need.
  const showFilePane = files.length > 1 && panelWidth >= FILE_PANE_MIN_PANEL_WIDTH
  useEffect(() => {
    // Wait for newly loaded index rows to participate in terminal layout.
    const timer = setTimeout(() => indexScrollRef.current?.scrollTo(Math.max(0, fileIdx - 3)), 0)
    return () => clearTimeout(timer)
  }, [fileIdx, files.length, showFilePane])
  const codeWidth = Math.max(panelWidth - (showFilePane ? FILE_PANE_WIDTH + 1 : 0) - 8,
    ...((diff?.lines ?? []).map(line => Bun.stringWidth(line.text) + (GUTTER_WIDTH + 1) * 2 + 4)))
  const location = cwd?.trim() || 'current workspace'

  // The file index is a where-am-I index, so its selection mirrors the
  // viewport: whichever file's section holds the top visible row is the one
  // highlighted. The functional update returns `previous` unchanged until
  // the top row crosses a file boundary, so poll ticks cost one scrollTop
  // read and never re-render on their own.
  const syncSelectionFromScroll = useCallback(() => {
    const box = scrollRef.current

    if (detailPath || !box || files.length === 0) {
      return
    }

    setFileIdx(previous =>
      fileIndexFollowingRow(files, Math.max(0, Math.floor(box.scrollTop) - DIFF_HEADER_ROWS), previous)
    )
  }, [files, detailPath])

  useEffect(() => {
    // A reload can land with the viewport already deep inside some file;
    // follow it immediately instead of waiting for a scroll or poll tick.
    syncSelectionFromScroll()
  }, [syncSelectionFromScroll])

  useEffect(() => {
    if (!showFilePane) return undefined
    // Mouse-wheel scrolling bypasses the keyboard path above; a slow poll
    // keeps wheel users on the same follow without hooking renderer frames.
    const timer = setInterval(syncSelectionFromScroll, SCROLL_FOLLOW_INTERVAL_MS)

    return () => clearInterval(timer)
  }, [showFilePane, syncSelectionFromScroll])

  return (
    <box
      alignItems="center"
      backgroundColor="#000000cc"
      flexDirection="column"
      height="100%"
      justifyContent="center"
      left={0}
      position="absolute"
      top={0}
      width="100%"
      zIndex={185}
    >
      <Box
        backgroundColor={t.color.completionBg}
        borderColor={t.color.border}
        borderSides={['top', 'bottom']}
        flexDirection="column"
        height={panelHeight}
        paddingX={2}
        paddingY={1}
        width={panelWidth}
      >
        {/* The header names the base AND what it points at: "12 files
            changed" means nothing until you know changed against what. */}
        <Box flexDirection="row" flexShrink={0} justifyContent="space-between">
          <Box flexDirection="row" flexShrink={1} minWidth={0} overflow="hidden">
            <Text wrap="truncate-end">
              <Span color={t.color.accent}>{`${GLYPH.brand} `}</Span>
              <Span bold color={t.ds.title}>Working changes</Span>
              <Span color={t.ds.caption}> vs </Span>
              <Span color={t.ds.secondary}>HEAD</Span>
              <Span color={t.ds.separator}>{`  ${GLYPH.separator} `}</Span>
              <Span color={t.ds.meta}>{location}</Span>
            </Text>
          </Box>
          <Box flexShrink={0}>
            <Text wrap="truncate-end">
              {diff ? (
                <>
                  <Span color={t.ds.done}>{`+${diff.insertions}`}</Span>
                  <Span color={t.ds.failed}>{` −${diff.deletions}`}</Span>
                  <Span color={t.ds.separator}>{`  ${GLYPH.separator} `}</Span>
                  <Span color={t.ds.meta}>
                    {`${diff.files} file${diff.files === 1 ? '' : 's'}`}
                    {diff.truncated ? ' · truncated' : ''}
                  </Span>
                </>
              ) : null}
            </Text>
          </Box>
        </Box>
        {detailPath && <Box onClick={returnToOverview}><Text color={t.color.accent}>← All changes (Backspace)</Text><Text color={t.color.muted}>  {detailPath}</Text></Box>}
        <Box flexDirection="row" flexGrow={1} flexShrink={1} minHeight={0} marginTop={1}>
        {/* A change set of forty-five files is not a document you read top to
            bottom — it is an index you navigate. The pane appears only when
            there is width to spare for it. */}
        {showFilePane ? (
          <Box
            flexDirection="column"
            flexShrink={0}
            height="100%"
            minHeight={0}
            marginRight={1}
            overflow="hidden"
            width={FILE_PANE_WIDTH}
          >
            <GroupCaption count={files.length} label="FILE INDEX" t={t} width={FILE_PANE_WIDTH} />
            <scrollbox key={files.length} ref={indexScrollRef} flexGrow={1} flexShrink={1} minHeight={0} scrollY scrollX={false}>
            <Box flexDirection="column" flexShrink={0} height={files.length}>
            {files.map((file, index) => (
              <Box
                flexShrink={0}
                backgroundColor={index === fileIdx ? t.ds.selected : undefined}
                flexDirection="row"
                key={file.name}
                onClick={() => openFile(index)}
              >
                <Box flexGrow={1} minWidth={0} overflow="hidden">
                  {/* Paths clip from the LEFT — the filename is the part you
                      are looking for. The budget is what is left after the
                      marker and the widest +/− pair the pane can carry.
                      (The canvas puts the directory on a second row under the
                      name; at 24 columns there is no directory worth showing,
                      and screen 09 makes side panels the first thing to go,
                      so a cramped pane must not grow to earn one.) */}
                  <Text wrap="truncate-end">
                    <Span color={index === fileIdx ? t.color.accent : t.ds.separator}>
                      {`${GLYPH.collapsed} `}
                    </Span>
                    <Span color={index === fileIdx ? t.ds.title : t.ds.secondary}>
                      {clipPath(baseName(file.name), FILE_PANE_WIDTH - 10)}
                    </Span>
                  </Text>
                </Box>
                {/* Mockup 07 file index: per-file +/− chips right-aligned,
                    word-role colors on their diff background tints. Zero
                    counts stay hidden, like the mockup's +38-only row. */}
                <Box flexShrink={0}>
                  {file.insertions > 0 ? (
                    <text fg={toRenderableColor(t.ds.done)}>{` +${file.insertions}`}</text>
                  ) : null}
                  {file.deletions > 0 ? (
                    <text fg={toRenderableColor(t.ds.failed)}>{` −${file.deletions}`}</text>
                  ) : null}
                </Box>
              </Box>
            ))}
            </Box>
            </scrollbox>
          </Box>
        ) : null}
        <scrollbox ref={scrollRef} scrollX style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} viewportCulling>
          <Box flexDirection="column" width={codeWidth}>
            {loading ? <Text color={t.color.muted}>◌ Refreshing worktree changes…</Text> : null}
            {!loading && result?.kind === 'clean' ? (
              <Text color={t.color.muted}>working tree clean — nothing removed or replaced</Text>
            ) : null}
            {!loading && result?.kind === 'error' ? (
              <Text color={t.color.warn}>{result.message}</Text>
            ) : null}
            {diff && diff.lines.length > 0 ? (
              <Box backgroundColor={t.ds.chrome} flexDirection="row">
                <Text color={t.ds.caption}>{'OLD'.padStart(GUTTER_WIDTH)} {'NEW'.padStart(GUTTER_WIDTH)}  CODE</Text>
              </Box>
            ) : null}
            {diff
              ? diff.lines.map((line, index) => (
                  <DiffRow key={index} line={line} t={t} words={wordRanges.get(index)} />
                ))
              : null}
            {diff && diff.untracked.length > 0 ? (
              <Box flexDirection="column" marginTop={1}>
                <Box backgroundColor={t.color.completionBg} flexDirection="row" justifyContent="space-between" paddingX={1}>
                  <Text bold color={t.color.label}>UNTRACKED FILES</Text>
                  <Text color={t.color.muted}>{diff.untracked.length}{diff.untrackedTruncated ? '+' : ''}</Text>
                </Box>
                {diff.untracked.map(name => (
                  <Box key={name} onClick={() => { const index = files.findIndex(file => file.name === name); if (index >= 0) openFile(index) }}><Text color={t.color.accent} wrap="truncate-end">{'  U  '}{name}</Text></Box>
                ))}
              </Box>
            ) : null}
          </Box>
        </scrollbox>
        </Box>
        {overview?.untrackedTruncated && <Box flexShrink={0} onClick={loadMoreUntracked}><Text color={t.color.accent}>{listingError || 'M · Load more new files (requires current workspace runtime)'}</Text></Box>}
        <Box
          backgroundColor={t.color.completionBg}
          flexDirection="row"
          flexShrink={0}
          justifyContent="space-between"
          marginTop={1}
          paddingX={1}
        >
          <Text color={t.color.muted}>{`↑↓ scroll · ←→ code · [ ] files   ${PAGE_KEY_HINT}   R Refresh   ⇧⌘/⌃←/→ Resize`}</Text>
          <Text color={t.color.muted}>Esc / Q Close</Text>
        </Box>
      </Box>
    </box>
  )
}
