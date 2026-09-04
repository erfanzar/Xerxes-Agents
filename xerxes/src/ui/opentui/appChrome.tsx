// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// Quiet session chrome: mode/title above the transcript and workspace below.
// Context and model metadata live with the composer, where they are actionable.
import type { SessionTab } from '../app/interfaces.js'
import type { LiveSessionStatus } from '../gatewayTypes.js'
import { GLYPH } from '../domain/nocturne.js'
import { ctxBarColor, ctxMeterBar } from '../domain/statusFormat.js'
import { fmtK } from '../lib/text.js'
import { branchLabel, type RepoPulse } from '../lib/repoPulse.js'
import type { Theme } from '../theme.js'

import { Box, Span, Text } from './primitives.js'

const TAB_STATUS_GLYPH: Record<LiveSessionStatus, string> = {
  // ○ reads as "nothing pending" — the ring is the v2 idle shape. The other
  // states keep their informative glyphs.
  idle: '○',
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

/**
 * The session header — "where am I", answered in one row.
 *
 * Left names the thing and its handle; right names its state and how much of
 * the context window it has spent. The context read-out lives here rather
 * than beside the composer because it answers "what is happening", and the
 * composer row answers "what will ⏎ do" — two different questions that were
 * previously sharing one line and colliding on narrow terminals.
 */
export function SessionHeader({
  busy,
  contextMax,
  contextUsed,
  goal,
  goalPhase,
  mode,
  sessionId,
  sessionTitle,
  t
}: {
  /** Whether a turn is in flight; drives the state dot and its verb. */
  busy?: boolean
  contextMax?: number
  contextUsed?: number
  /** Current goal objective, when one is set. */
  goal?: string
  /** Goal phase: active, paused, blocked, complete. */
  goalPhase?: string
  mode?: string
  sessionId?: null | string
  sessionTitle?: null | string
  t: Theme
}) {
  const title = sessionTitle?.trim()
  const id = sessionId?.trim()
  // A short handle, not the whole id: enough to tell two chats apart at a
  // glance, small enough that it never competes with the title.
  const shortId = id ? id.slice(-4) : ''
  const max = contextMax ?? 0
  const used = contextUsed ?? 0
  const pressure = max > 0 ? Math.min(100, (used / max) * 100) : undefined

  return (
    <Box
      flexDirection="column"
      flexShrink={0}
      overflow="hidden"
      paddingX={2}
      paddingY={1}
      width="100%"
    >
      <Box flexDirection="row" flexShrink={0} gap={2} justifyContent="space-between" overflow="hidden" width="100%">
        <Box flexDirection="row" flexShrink={1} minWidth={0} overflow="hidden">
          <Text wrap="truncate-end">
            <Span color={t.color.brandGold}>{`${GLYPH.brand} `}</Span>
            {/* The mode is stated once — in the composer's chip. Repeating it
                here made the header read like a second status bar. An untitled
                session still falls back to it as a name. */}
            <Span color={t.ds.title}>{title || displayModeLabel(mode)}</Span>
            {shortId ? (
              <>
                <Span color={t.ds.separator}>{` ${GLYPH.separator} `}</Span>
                <Span color={t.ds.meta}>{`session ${shortId}`}</Span>
              </>
            ) : null}
          </Text>
        </Box>
        <Box flexDirection="row" flexShrink={0} minWidth={0} overflow="hidden">
          <Text wrap="truncate-end">
            <Span color={busy ? t.ds.working : t.ds.done}>{`${GLYPH.state} `}</Span>
            <Span color={t.ds.meta}>{busy ? 'working' : 'idle'}</Span>
            <Span color={t.ds.rule}>{` ${GLYPH.sectionBreak} `}</Span>
            {max > 0 ? (
              <>
                {/* One answer to "how much context is left", in one place: the
                    bar for the glance, the numbers for the decision. It used
                    to be rendered beside the composer as well, on the row that
                    answers what ⏎ will do — two places, two truncation rules,
                    and a collision at 80 columns. */}
                <Span color={ctxBarColor(pressure, t)}>{`${ctxMeterBar(pressure)} `}</Span>
                <Span color={ctxBarColor(pressure, t)}>{fmtK(used)}</Span>
                <Span color={t.ds.meta}>{` / ${fmtK(max)} ctx`}</Span>
              </>
            ) : (
              <Span color={t.ds.meta}>ctx unknown</Span>
            )}
          </Text>
        </Box>
      </Box>
      {goal ? (
        <Box flexDirection="row" flexShrink={0} marginTop={1} overflow="hidden" width="100%">
          <Text wrap="truncate-end">
            <Span color={t.ds.caption}>{'goal '}</Span>
            <Span color={t.ds.meta}>{goal}</Span>
            {goalPhase && goalPhase !== 'active' ? (
              <Span color={t.color.muted}>{` (${goalPhase})`}</Span>
            ) : null}
          </Text>
        </Box>
      ) : null}
    </Box>
  )
}

/**
 * Cumulative session telemetry, one quiet line under the header — the same
 * counters the desktop stats bar shows (turns, steps, LLM/tool time, TTFT,
 * throughput, cache, tokens). Hidden for a fresh session: a bar of zeros is
 * noise, and the caller passes '' then.
 */
export function SessionTelemetryRow({ line, t }: { line: string; t: Theme }) {
  if (!line) return null

  return (
    <Box
      flexDirection="row"
      flexShrink={0}
      justifyContent="flex-end"
      overflow="hidden"
      paddingX={2}
      width="100%"
    >
      <Text color={t.ds.meta} wrap="truncate-end">
        {line}
      </Text>
    </Box>
  )
}

const NEW_TAB_LABEL = ' +'
// Mockup 04's gesture, advertised where the tabs live: Left walks one tab
// left, and Left from the leftmost tab backgrounds the session into the
// agent view. Right stays the picker's re-enter key, so it is not promised
// here.
const HINTS_LABEL = '← switch · ←← agent view'

/**
 * Row of live-session tabs on mockup 02's darker band. Hidden for a single
 * session — one tab is just the header restated. When the strip cannot fit
 * the terminal it collapses to a `‹ n/total ›` position indicator instead of
 * truncated titles.
 */
export function SessionTabStrip({
  activeId,
  onNewTab,
  onSelect,
  tabs,
  t,
  width
}: {
  activeId: null | string
  /** Click target for the faint "+" affordance; absent renders it inert. */
  onNewTab?: () => void
  /** Click target for switching to a tab; absent renders tabs inert. */
  onSelect?: (id: string) => void
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
  const baseCost = tabs.length * perTab + 4
  // Mockup 02: a faint "+" affordance closes the strip and, when the width
  // allows it, a faint right-aligned key hint. Both are disposable — the "+"
  // spends its budget first, and the hints drop before any tab label does.
  const NEW_TAB_CELL = 2
  const fits = baseCost + NEW_TAB_CELL <= width
  const showHints = fits && baseCost + NEW_TAB_CELL + 1 + HINTS_LABEL.length <= width

  if (!fits) {
    const current = activeIndex >= 0 ? activeIndex + 1 : 1
    return (
      <Box backgroundColor={t.color.completionBg} flexDirection="row" flexShrink={0} paddingX={2} width="100%">
        <Text color={t.color.muted}>{`‹ ${current}/${tabs.length} ›`}</Text>
      </Box>
    )
  }

  return (
    <Box
      backgroundColor={t.color.completionBg}
      flexDirection="row"
      flexShrink={0}
      height={2}
      overflow="hidden"
      paddingX={2}
      width="100%"
    >
      {tabs.map((tab, index) => {
        const active = tab.id === activeId
        // An unnamed session shows its position instead. A tab needs a stable
        // handle more than it needs a name, and a row of identical
        // placeholders is no handle at all.
        const label = tab.title.trim() ? truncate(tab.title, TAB_LABEL_MAX) : String(index + 1)

        if (active) {
          // The filled dot + brand colour is the whole active treatment; a
          // selection block behind one tab made the strip read as buttons.
          // Mockup 02 adds an underline: a one-row ─ segment drawn as the
          // second row of the active tab's own column, so the renderer never
          // has to mix a border into the band background.
          const content = `● ${label}`
          const cellWidth = content.length + 2

          return (
            <Box
              flexDirection="column"
              flexShrink={0}
              key={tab.id}
              onClick={onSelect ? () => onSelect(tab.id) : undefined}
              width={cellWidth}
            >
              <Text wrap="truncate-end">
                <Span bold color={t.color.brandGold}>{` ${content} `}</Span>
              </Text>
              <Text color={t.color.brandGold}>{'━'.repeat(cellWidth)}</Text>
            </Box>
          )
        }

        const glyph = TAB_STATUS_GLYPH[tab.status] ?? '·'
        const content = `${glyph} ${label}`

        return (
          <Box
            flexDirection="column"
            flexShrink={0}
            key={tab.id}
            onClick={onSelect ? () => onSelect(tab.id) : undefined}
            width={content.length + 2}
          >
            <Text wrap="truncate-end">
              <Span color={t.color.muted}>{` ${content} `}</Span>
            </Text>
            {/* Keeps every tab column two rows tall, so the strip reads as
                one band instead of labels floating over loose rows. */}
            <Text>{' '.repeat(content.length + 2)}</Text>
          </Box>
        )
      })}
      <Box flexDirection="column" flexShrink={0} onClick={onNewTab}>
        <Text color={t.color.muted}>{NEW_TAB_LABEL}</Text>
        <Text>{' '}</Text>
      </Box>
      {showHints ? (
        <Box flexGrow={1} flexShrink={1} justifyContent="flex-end" minWidth={0} overflow="hidden">
          <Text color={t.color.muted} wrap="truncate-end">
            {HINTS_LABEL}
          </Text>
        </Box>
      ) : null}
    </Box>
  )
}

/**
 * The statusbar — identical on every screen, which is what makes it readable
 * as a fixed place rather than as more content.
 *
 * Left answers "which machine am I on": workspace, branch, and whether the
 * tree owes you anything. Right answers "what can I do": the view keys and
 * the version. `│` separates the two regions and is quieter than the `·`
 * that separates facts inside one region — it divides regions of the same
 * bar, not facts.
 */
export function WorkspaceFooter({
  cwdLabel,
  providerModel,
  pulse,
  rightLabel,
  t
}: {
  cwdLabel: string
  /** Provider health input; undefined hides the segment (info not loaded yet). */
  providerModel?: string
  /** Working-tree state. Omitted before the first `git` answer lands. */
  pulse?: RepoPulse
  rightLabel?: string
  t: Theme
}) {
  const showProvider = providerModel !== undefined
  const modelConfigured = Boolean(providerModel?.trim())
  const branch = pulse ? branchLabel(pulse) : ''
  // Amber appears here for exactly one reason and nowhere else on the bar:
  // uncommitted work is unreviewed work, and unreviewed is the same "a human
  // is required" the agents screen means by it. A clean tree earns the done
  // green instead, so the bar is never amber when nothing is outstanding.
  const tree = pulse?.dirty ? `● ${pulse.dirty} dirty` : branch ? '✓ clean' : ''
  const treeColor = pulse?.dirty ? t.color.warn : t.color.statusGood

  if (!cwdLabel && !rightLabel && !showProvider) {
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
      {/* Shrink priority: the workspace path is the identity of this screen
          and yields LAST. The hotkey hints are disposable, so their box
          carries the shrink and truncates first; the path only truncates
          when the terminal itself is genuinely too narrow for it. */}
      <Box flexDirection="row" flexShrink={0} minWidth={0} overflow="hidden">
        {cwdLabel ? (
          <Text color={t.ds.secondary} wrap="truncate-end">
            <Span color={t.color.brandGold}>{`${GLYPH.brand} `}</Span>
            {cwdLabel}
            {branch ? (
              <>
                <Span color={t.ds.rule}>{` ${GLYPH.sectionBreak} `}</Span>
                <Span color={t.ds.caption}>{branch}</Span>
              </>
            ) : null}
            {tree ? <Span color={treeColor}>{` ${tree}`}</Span> : null}
          </Text>
        ) : null}
      </Box>
      {rightLabel || showProvider ? (
        <Box flexDirection="row" flexShrink={1} gap={1} justifyContent="flex-end" minWidth={0} overflow="hidden">
          {rightLabel ? (
            <Text color={t.ds.caption} wrap="truncate-end">
              {rightLabel}
            </Text>
          ) : null}
          {showProvider ? (
            modelConfigured ? (
              // A green dot vouches that the next ⏎ will actually reach a model.
              <Text wrap="truncate-end">
                <Span color={t.ds.rule}>{`${GLYPH.sectionBreak} `}</Span>
                <Span color={t.color.statusGood}>{`${GLYPH.state} `}</Span>
                <Span color={t.ds.caption}>provider ready</Span>
              </Text>
            ) : (
              <Text wrap="truncate-end">
                <Span color={t.ds.rule}>{`${GLYPH.sectionBreak} `}</Span>
                <Span color={t.color.warn}>{`${GLYPH.state} no model · /provider`}</Span>
              </Text>
            )
          ) : null}
        </Box>
      ) : null}
    </Box>
  )
}
