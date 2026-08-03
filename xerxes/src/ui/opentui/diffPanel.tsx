// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import type { ScrollBoxRenderable } from '@opentui/core'
import { useStore } from '@nanostores/react'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import {
  $panelWidthDelta,
  adjustPanelWidth,
  PANEL_WIDTH_STEP,
  withPanelWidthDelta
} from '../app/panelSizeStore.js'
import { collectGitDiff, type DiffLine, type GitDiffResult } from '../lib/gitDiff.js'
import type { Theme } from '../theme.js'

import { Box, Span, Text } from './primitives.js'

/**
 * F7 git diff viewer: shows what the worktree changed (staged + unstaged vs
 * HEAD) with per-file sections, colored +/- rows, and the untracked-file
 * list. Mirrors the agents overlay's layout and key discipline: F7/Esc/q
 * close, arrows/PageUp/PageDown scroll, r re-runs the diff.
 */

type KeyEvent = Parameters<Parameters<typeof useKeyboard>[0]>[0]

const consumeKey = (event: KeyEvent) => {
  event.preventDefault()
  event.stopPropagation()
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

const HUNK_COLOR = '#56c2d6'
const GUTTER_WIDTH = 6

const lineNumber = (value: number | undefined): string =>
  value === undefined ? ' '.repeat(GUTTER_WIDTH) : String(value).padStart(GUTTER_WIDTH)

const codeText = (line: DiffLine): string => line.text.slice(1) || ' '

function DiffCodeRow({ line, t }: { line: DiffLine; t: Theme }) {
  const added = line.kind === 'add'
  const removed = line.kind === 'del'
  const foreground = added ? t.color.ok : removed ? t.color.error : t.color.text
  const background = added ? '#14251b' : removed ? '#2a171b' : undefined
  const marker = added ? '+' : removed ? '−' : ' '

  return (
    <Box backgroundColor={background} flexDirection="row">
      <Text color={t.color.muted}>{lineNumber(line.oldLine)} </Text>
      <Text color={t.color.muted}>{lineNumber(line.newLine)} </Text>
      <Text color={foreground} wrap="truncate-end">
        <Span bold={added || removed} color={foreground}>{marker} </Span>
        {codeText(line)}
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

function DiffRow({ line, t }: { line: DiffLine; t: Theme }) {
  switch (line.kind) {
    case 'file':
      return (
        <Box backgroundColor={t.color.completionCurrentBg} flexDirection="row" paddingX={1}>
          <Text bold color={t.color.primary} wrap="truncate-end">▾ {line.text}</Text>
        </Box>
      )
    case 'hunk':
      return (
        <Box backgroundColor={t.color.completionMetaBg} flexDirection="row">
          <Text color={t.color.muted}>{' '.repeat((GUTTER_WIDTH + 1) * 2)}</Text>
          <Text color={HUNK_COLOR} wrap="truncate-end">  {line.text}</Text>
        </Box>
      )
    case 'add':
    case 'del':
    case 'context':
      return <DiffCodeRow line={line} t={t} />
    case 'meta':
      return <Text color={t.color.muted} italic wrap="truncate-end">  {line.text}</Text>
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
  const scrollRef = useRef<ScrollBoxRenderable | null>(null)
  const [result, setResult] = useState<GitDiffResult | null>(null)
  const [loading, setLoading] = useState(true)
  const generation = useRef(0)
  const { width: terminalWidth } = useTerminalDimensions()
  useStore($panelWidthDelta)
  const panelWidth = withPanelWidthDelta(Math.min(120, Math.floor(terminalWidth * 0.9)), terminalWidth)

  const reload = useCallback(() => {
    const gen = ++generation.current
    setLoading(true)
    void collectGitDiff({ cwd: cwd ?? '' })
      .then(next => {
        if (generation.current === gen) {
          setResult(next)
          setLoading(false)
        }
      })
      .catch((error: unknown) => {
        if (generation.current === gen) {
          setResult({ kind: 'error', message: error instanceof Error ? error.message : String(error) })
          setLoading(false)
        }
      })
  }, [cwd])

  useEffect(() => {
    reload()
  }, [reload])

  const page = 10

  useKeyboard(event => {
    if (isPanelResizeKey(event)) {
      consumeKey(event)
      adjustPanelWidth(event.name === 'right' ? PANEL_WIDTH_STEP : -PANEL_WIDTH_STEP)
    } else if (event.name === 'escape' || event.name === 'f7' || event.sequence === 'q') {
      consumeKey(event)
      onClose()
    } else if (event.sequence === 'r') {
      consumeKey(event)
      reload()
    } else if (event.name === 'up') {
      consumeKey(event)
      scrollRef.current?.scrollBy(-1)
    } else if (event.name === 'down') {
      consumeKey(event)
      scrollRef.current?.scrollBy(1)
    } else if (event.name === 'pageup') {
      consumeKey(event)
      scrollRef.current?.scrollBy(-page)
    } else if (event.name === 'pagedown') {
      consumeKey(event)
      scrollRef.current?.scrollBy(page)
    } else if (event.name === 'home') {
      consumeKey(event)
      scrollRef.current?.scrollTo(0)
    } else if (event.name === 'end') {
      consumeKey(event)
      scrollRef.current?.scrollTo(Number.MAX_SAFE_INTEGER)
    }
  })

  const diff = result?.kind === 'ok' ? result.diff : null
  const title = diff
    ? `${diff.files} file${diff.files === 1 ? '' : 's'} changed`
    : 'Working tree'
  const location = cwd?.trim() || 'current workspace'

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
        backgroundColor={t.color.statusBg}
        borderColor={t.color.accent}
        borderStyle="single"
        flexDirection="column"
        height="80%"
        maxWidth={150}
        paddingX={2}
        paddingY={1}
        width={panelWidth}
      >
        <Box flexDirection="row" flexShrink={0} justifyContent="space-between">
          <Box flexDirection="column">
            <Text bold color={t.color.primary}>SOURCE CONTROL</Text>
            <Text color={t.color.muted} wrap="truncate-end">{location}</Text>
          </Box>
          <Text color={t.color.muted}>F7  ×</Text>
        </Box>
        <Box
          backgroundColor={t.color.completionBg}
          flexDirection="row"
          flexShrink={0}
          justifyContent="space-between"
          marginBottom={1}
          marginTop={1}
          paddingX={1}
        >
          <Text bold color={t.color.label}>CHANGES  ·  {title}</Text>
          {diff ? (
            <Text>
              <Span bold color={t.color.ok}>+{diff.insertions}</Span>
              <Span color={t.color.muted}>  </Span>
              <Span bold color={t.color.error}>−{diff.deletions}</Span>
              {diff.truncated ? <Span color={t.color.warn}>  truncated</Span> : null}
            </Text>
          ) : null}
        </Box>
        <scrollbox ref={scrollRef} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} viewportCulling>
          <Box flexDirection="column">
            {loading ? <Text color={t.color.muted}>◌ Refreshing worktree changes…</Text> : null}
            {!loading && result?.kind === 'clean' ? (
              <Text color={t.color.muted}>working tree clean — nothing removed or replaced</Text>
            ) : null}
            {!loading && result?.kind === 'error' ? (
              <Text color={t.color.warn}>{result.message}</Text>
            ) : null}
            {diff && diff.lines.length > 0 ? (
              <Box backgroundColor={t.color.completionMetaBg} flexDirection="row">
                <Text bold color={t.color.muted}>{'OLD'.padStart(GUTTER_WIDTH)} {'NEW'.padStart(GUTTER_WIDTH)}  CODE</Text>
              </Box>
            ) : null}
            {diff
              ? diff.lines.map((line, index) => <DiffRow key={index} line={line} t={t} />)
              : null}
            {diff && diff.untracked.length > 0 ? (
              <Box flexDirection="column" marginTop={1}>
                <Box backgroundColor={t.color.completionBg} flexDirection="row" justifyContent="space-between" paddingX={1}>
                  <Text bold color={t.color.label}>UNTRACKED FILES</Text>
                  <Text color={t.color.muted}>{diff.untracked.length}{diff.untrackedTruncated ? '+' : ''}</Text>
                </Box>
                {diff.untracked.map(name => (
                  <Text key={name} color={t.color.text} wrap="truncate-end">
                    {'  U  '}{name}
                  </Text>
                ))}
              </Box>
            ) : null}
          </Box>
        </scrollbox>
        <Box
          backgroundColor={t.color.completionBg}
          flexDirection="row"
          flexShrink={0}
          justifyContent="space-between"
          marginTop={1}
          paddingX={1}
        >
          <Text color={t.color.muted}>↑↓ Navigate   PgUp/PgDn Page   R Refresh   ⇧⌘/⌃←/→ Resize</Text>
          <Text color={t.color.muted}>Esc / Q Close</Text>
        </Box>
      </Box>
    </box>
  )
}
