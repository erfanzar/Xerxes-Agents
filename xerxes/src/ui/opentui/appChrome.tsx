// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// Quiet session chrome: mode/title above the transcript and workspace below.
// Context and model metadata live with the composer, where they are actionable.
import type { SessionTab } from '../app/interfaces.js'
import type { LiveSessionStatus } from '../gatewayTypes.js'
import type { Theme } from '../theme.js'

import { Box, Span, Text } from './primitives.js'

const TAB_STATUS_GLYPH: Record<LiveSessionStatus, string> = {
  idle: '✓',
  starting: '…',
  waiting: '?',
  working: '◆'
}

const truncate = (value: string, max: number) =>
  value.length > max ? `${value.slice(0, Math.max(1, max - 1))}…` : value

/**
 * Longest tab label, and the basis of the strip's fit budget. One constant so
 * the two cannot drift: they used to disagree (12 vs 18), which made the strip
 * claim to fit and then clip silently.
 */
const TAB_LABEL_MAX = 14

export const displayModeLabel = (mode?: string): string => {
  const value = (mode || 'code').trim()

  return value ? value[0]!.toUpperCase() + value.slice(1) : 'Code'
}

export function SessionHeader({
  mode,
  sessionId,
  sessionTitle,
  t
}: {
  mode?: string
  sessionId?: null | string
  sessionTitle?: null | string
  t: Theme
}) {
  const title = sessionTitle?.trim()
  const id = sessionId?.trim()

  return (
    <Box flexDirection="row" flexShrink={0} paddingX={2} paddingY={1} width="100%">
      <Box flexGrow={1} flexShrink={1} overflow="hidden">
        <Text bold wrap="truncate-end">
          <Span bold color={t.color.accent}>
            {displayModeLabel(mode)}
          </Span>
          {title ? <Span color={t.color.text}>: {title}</Span> : null}
        </Text>
      </Box>
      {id ? (
        <Text color={t.color.muted} wrap="truncate-end">
          {id}
        </Text>
      ) : null}
    </Box>
  )
}

/**
 * Row of live-session tabs. Hidden for a single session — one tab is just the
 * header restated. When the strip cannot fit the terminal it collapses to a
 * `‹ n/total ›` position indicator instead of truncated titles.
 */
export function SessionTabStrip({
  activeId,
  tabs,
  t,
  width
}: {
  activeId: null | string
  tabs: SessionTab[]
  t: Theme
  width: number
}) {
  if (tabs.length < 2) {
    return null
  }

  const activeIndex = tabs.findIndex(tab => tab.id === activeId)
  // Each tab costs `space + glyph + space + label + space`, so derive the
  // budget from the label cap instead of guessing. The previous estimate of
  // 12 against an 18-char truncation under-counted by 5 per tab, so the strip
  // claimed to fit and then silently clipped against overflow="hidden".
  const perTab = TAB_LABEL_MAX + 5
  const fits = tabs.length * perTab + 4 <= width

  if (!fits) {
    const current = activeIndex >= 0 ? activeIndex + 1 : 1
    return (
      <Box flexDirection="row" flexShrink={0} paddingX={2} width="100%">
        <Text color={t.color.muted}>{`‹ ${current}/${tabs.length} ›`}</Text>
      </Box>
    )
  }

  return (
    <Box flexDirection="row" flexShrink={0} overflow="hidden" paddingX={2} width="100%">
      {tabs.map((tab, index) => {
        const active = tab.id === activeId
        const glyph = TAB_STATUS_GLYPH[tab.status] ?? '·'
        // An unnamed session shows its position instead. A tab needs a stable
        // handle more than it needs a name, and a row of identical
        // placeholders is no handle at all.
        const label = tab.title.trim() ? truncate(tab.title, TAB_LABEL_MAX) : String(index + 1)
        return (
          <Box
            backgroundColor={active ? t.color.selectionBg : undefined}
            flexShrink={0}
            key={tab.id}
          >
            <Text wrap="truncate-end">
              <Span bold={active} color={active ? t.color.accent : t.color.muted}>
                {` ${glyph} ${label} `}
              </Span>
            </Text>
          </Box>
        )
      })}
    </Box>
  )
}

export function WorkspaceFooter({ cwdLabel, rightLabel, t }: { cwdLabel: string; rightLabel?: string; t: Theme }) {  if (!cwdLabel && !rightLabel) {
    return null
  }

  return (
    <Box
      flexDirection="row"
      flexShrink={0}
      // Same gutter rule as the status row: `space-between` alone let the cwd
      // butt straight into the hotkey hints ("sandbox (main)F6 agents"), and
      // an un-truncated right label wrapped onto a second row.
      gap={2}
      justifyContent="space-between"
      overflow="hidden"
      paddingBottom={1}
      paddingX={2}
      width="100%"
    >
      <Box flexDirection="row" flexShrink={1} minWidth={0} overflow="hidden">
        <Text color={t.color.muted} wrap="truncate-end">
          {cwdLabel}
        </Text>
      </Box>
      {rightLabel ? (
        <Text color={t.color.muted} wrap="truncate-end">
          {rightLabel}
        </Text>
      ) : null}
    </Box>
  )
}
