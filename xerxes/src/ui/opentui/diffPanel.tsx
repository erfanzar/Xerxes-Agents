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

import { Box, Text } from './primitives.js'

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

const ADD_COLOR = '#5fd75f'
const DEL_COLOR = '#ff5f5f'
const HUNK_COLOR = '#56c2d6'

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
      return <Text bold color={t.color.accent} wrap="truncate-end">{line.text}</Text>
    case 'hunk':
      return <Text color={HUNK_COLOR} wrap="truncate-end">{line.text}</Text>
    case 'add':
      return <Text color={ADD_COLOR}>{line.text}</Text>
    case 'del':
      return <Text color={DEL_COLOR}>{line.text}</Text>
    case 'meta':
      return <Text color={t.color.muted} wrap="truncate-end">{line.text}</Text>
    default:
      return <Text color={t.color.muted}>{line.text || ' '}</Text>
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
      scrollRef.current?.scrollBy(-1)
    } else if (event.name === 'down') {
      scrollRef.current?.scrollBy(1)
    } else if (event.name === 'pageup') {
      scrollRef.current?.scrollBy(-page)
    } else if (event.name === 'pagedown') {
      scrollRef.current?.scrollBy(page)
    } else if (event.name === 'home') {
      scrollRef.current?.scrollTo(0)
    } else if (event.name === 'end') {
      scrollRef.current?.scrollTo(Number.MAX_SAFE_INTEGER)
    }
  })

  const diff = result?.kind === 'ok' ? result.diff : null
  const title = diff
    ? `Git diff — ${diff.files} file${diff.files === 1 ? '' : 's'}, +${diff.insertions} −${diff.deletions}${diff.truncated ? ' (truncated)' : ''}`
    : 'Git diff'

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
        <Box flexDirection="row" flexShrink={0} justifyContent="space-between" marginBottom={1}>
          <Text bold color={t.color.accent}>
            {title}
          </Text>
          <Text color={t.color.muted}>F7 close</Text>
        </Box>
        <scrollbox ref={scrollRef} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }}>
          <Box flexDirection="column">
            {loading ? <Text color={t.color.muted}>running git diff…</Text> : null}
            {!loading && result?.kind === 'clean' ? (
              <Text color={t.color.muted}>working tree clean — nothing removed or replaced</Text>
            ) : null}
            {!loading && result?.kind === 'error' ? (
              <Text color={t.color.warn}>{result.message}</Text>
            ) : null}
            {diff
              ? diff.lines.map((line, index) => <DiffRow key={index} line={line} t={t} />)
              : null}
            {diff && diff.untracked.length > 0 ? (
              <>
                <Text bold color={t.color.accent}>
                  ▌ untracked ({diff.untracked.length}{diff.untrackedTruncated ? '+' : ''})
                </Text>
                {diff.untracked.map(name => (
                  <Text key={name} color={t.color.muted} wrap="truncate-end">
                    {'  '}{name}
                  </Text>
                ))}
              </>
            ) : null}
          </Box>
        </scrollbox>
        <Box flexDirection="row" flexShrink={0} justifyContent="space-between" marginTop={1}>
          <Text color={t.color.muted}>↑↓ scroll · PgUp/PgDn · r refresh · ⇧⌘/⌃←/→ width</Text>
          <Text color={t.color.muted}>esc / q close</Text>
        </Box>
      </Box>
    </box>
  )
}
