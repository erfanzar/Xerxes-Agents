// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import type { KeyEvent, ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { type MutableRefObject, memo, useEffect, useMemo, useRef, useState } from 'react'

import { useOptionalGateway } from '../app/gatewayContext.js'
import { adjustPanelWidth, PANEL_WIDTH_STEP, withPanelWidthDelta } from '../app/panelSizeStore.js'
import type { SpawnSnapshot } from '../app/spawnHistoryStore.js'
export { AGENT_SIDEBAR_BREAKPOINT, shouldShowAgentSidebar } from '../domain/agentPanelLayout.js'
import { retrySubagent, subagentFailed, subagentRetryable } from '../lib/agentRetry.js'
import { subagentElapsedSeconds } from '../lib/subagentElapsed.js'
import { fmtDuration, fmtTokens } from '../lib/subagentTree.js'
import type { Theme } from '../theme.js'
import type { SubagentProgress } from '../types.js'

import { isPanelResizeKey } from './diffPanel.js'
import { Box, Span, Text } from './primitives.js'

export const AGENT_TITLE_MAX_LENGTH = 24

const TERMINAL_STATUSES = new Set<SubagentProgress['status']>([
  'completed',
  'error',
  'failed',
  'interrupted',
  'timeout'
])

export interface AgentPanelRecord {
  archived: boolean
  childCount: number
  creatorTitle: string
  item: SubagentProgress
  snapshotLabel?: string
  title: string
}

interface AgentPanelProps {
  history: readonly SpawnSnapshot[]
  liveAgents: readonly SubagentProgress[]
  /** Show retry affordances; false when no daemon gateway is connected. */
  retryEnabled?: boolean
  /** Per-agent retry feedback keyed by agent id. */
  retryNotes?: ReadonlyMap<string, string>
  /** Keyboard-selected agent id (overlay only). */
  selectedId?: null | string
  t: Theme
  variant: 'overlay' | 'sidebar'
}

interface AgentPanelOverlayProps extends Omit<AgentPanelProps, 'variant'> {
  /** Open straight into one agent's inspector — set by clicking a rail card. */
  initialInspectId?: null | string
  onClose: () => void
}

const compactLine = (value: string, max: number): string => {
  const line = value.replace(/\s+/g, ' ').trim()

  return line.length > max ? `${line.slice(0, Math.max(1, max - 1)).trimEnd()}…` : line
}

const titleCase = (value: string): string => value.replace(/\b[a-z]/g, letter => letter.toUpperCase())

/**
 * Always provide a concise human label, including for older daemon events
 * that predate the required `title` field. Explicit titles win; a goal-based
 * fallback is more useful than a generic role such as "researcher".
 */
export function shortAgentTitle(agent: SubagentProgress, max = AGENT_TITLE_MAX_LENGTH): string {
  const explicit = agent.title?.trim() || agent.name?.trim()
  const fallback = agent.goal?.trim() || agent.agentType?.trim() || agent.model?.trim() || 'Agent task'
  const withoutRuntimeSuffix = (explicit || fallback).split('#', 1)[0] ?? fallback
  const normalized = titleCase(
    withoutRuntimeSuffix
      .replace(/^\/?root\//i, '')
      .replace(/[-_]+/g, ' ')
      .trim()
  )

  return compactLine(normalized || 'Agent task', Math.max(8, max))
}

const titleForId = (agents: readonly SubagentProgress[]): Map<string, string> => {
  const titles = new Map<string, string>()

  for (const agent of agents) {
    if (!titles.has(agent.id)) {
      titles.set(agent.id, shortAgentTitle(agent))
    }
  }

  return titles
}

/** Live rows come first. Archived snapshots then supply every unique prior agent. */
export function collectAgentPanelRecords(
  liveAgents: readonly SubagentProgress[],
  history: readonly Pick<SpawnSnapshot, 'label' | 'subagents'>[]
): AgentPanelRecord[] {
  const all = [...liveAgents, ...history.flatMap(snapshot => snapshot.subagents)]
  const titles = titleForId(all)
  const seen = new Set<string>()
  const rows: Array<{ archived: boolean; item: SubagentProgress; snapshotLabel?: string }> = []

  for (const item of liveAgents) {
    if (!seen.has(item.id)) {
      rows.push({ archived: false, item })
      seen.add(item.id)
    }
  }

  for (const snapshot of history) {
    for (const item of snapshot.subagents) {
      if (seen.has(item.id)) continue
      rows.push({ archived: true, item, snapshotLabel: snapshot.label })
      seen.add(item.id)
    }
  }

  const childCounts = new Map<string, number>()
  for (const { item } of rows) {
    if (item.parentId) {
      childCounts.set(item.parentId, (childCounts.get(item.parentId) ?? 0) + 1)
    }
  }

  return rows.map(row => ({
    ...row,
    childCount: childCounts.get(row.item.id) ?? 0,
    creatorTitle:
      (row.item.creatorId && titles.get(row.item.creatorId)) ||
      (row.item.parentId && titles.get(row.item.parentId)) ||
      'Xerxes',
    title: shortAgentTitle(row.item)
  }))
}

function statusPresentation(status: SubagentProgress['status'], t: Theme): { color: string; glyph: string } {
  if (status === 'running') return { color: t.color.accent, glyph: '●' }
  if (status === 'queued') return { color: t.color.muted, glyph: '○' }
  if (status === 'completed') return { color: t.color.ok, glyph: '✓' }
  if (status === 'interrupted') return { color: t.color.warn, glyph: '■' }
  if (status === 'timeout') return { color: t.color.warn, glyph: '⌛' }

  return { color: t.color.error, glyph: '✗' }
}

const basename = (path: string): string => path.replaceAll('\\', '/').split('/').filter(Boolean).at(-1) || path

function activitySummary(item: SubagentProgress): string {
  if (TERMINAL_STATUSES.has(item.status) && item.summary?.trim()) return item.summary.trim()
  if (item.notes.at(-1)?.trim()) return item.notes.at(-1)!.trim()
  if (item.tools.at(-1)?.trim()) return item.tools.at(-1)!.trim()
  if (item.thinking.at(-1)?.trim()) return item.thinking.at(-1)!.trim()
  if (item.summary?.trim()) return item.summary.trim()

  return item.status === 'queued' ? 'Waiting to start' : item.status === 'running' ? 'Working' : 'No summary reported'
}

const agentToolCount = (item: SubagentProgress): number =>
  Math.max(item.toolCount, item.toolCalls?.length ?? 0, item.tools.length, item.outputTail?.length ?? 0)

/**
 * The row's one metrics line: spend, time, work done.
 *
 * Tokens lead because that is the number the list exists to answer — what an
 * agent cost. What it is saying right now belongs in the inspector, not here:
 * a rail of live chatter re-rendered every delta and still told you nothing
 * you could compare between agents.
 */
function metricLine(item: SubagentProgress, now: number): string {
  const tokens = (item.inputTokens ?? 0) + (item.outputTokens ?? 0)
  const elapsed = subagentElapsedSeconds(item, now)
  const toolCount = agentToolCount(item)
  const parts = [tokens > 0 ? `${fmtTokens(tokens)} tok` : 'no tokens yet']

  if (elapsed != null) parts.push(fmtDuration(elapsed))
  parts.push(`${toolCount} tool${toolCount === 1 ? '' : 's'}`)

  return parts.join(' · ')
}

/** Row-width token summary: in/out, with cached appended when it is earning its place. */
function tokenSummary(item: SubagentProgress): string {
  const input = item.inputTokens ?? 0
  const output = item.outputTokens ?? 0
  const cached = item.cacheReadTokens ?? 0
  if (!input && !output && !cached) {
    return "no tokens yet"
  }
  const parts = [`${fmtTokens(input)} in`, `${fmtTokens(output)} out`]
  if (cached > 0) parts.push(`${fmtTokens(cached)} cached`)
  return parts.join("/")
}

/** Full token breakdown, shown only in the inspector where there is room. */
function tokenDetail(item: SubagentProgress): string {
  const parts: string[] = []
  if ((item.inputTokens ?? 0) > 0) parts.push(`${fmtTokens(item.inputTokens!)} in`)
  if ((item.outputTokens ?? 0) > 0) parts.push(`${fmtTokens(item.outputTokens!)} out`)
  if ((item.cacheReadTokens ?? 0) > 0) parts.push(`${fmtTokens(item.cacheReadTokens!)} cached`)
  if ((item.cacheCreationTokens ?? 0) > 0) parts.push(`${fmtTokens(item.cacheCreationTokens!)} cache write`)
  if ((item.reasoningTokens ?? 0) > 0) parts.push(`${fmtTokens(item.reasoningTokens!)} reasoning`)
  if ((item.apiCalls ?? 0) > 0) parts.push(`${item.apiCalls} API call${item.apiCalls === 1 ? '' : 's'}`)
  if (typeof item.costUsd === 'number' && item.costUsd > 0) parts.push(`$${item.costUsd.toFixed(4)}`)

  return parts.length ? parts.join(' · ') : 'no usage reported yet'
}

/** Milliseconds as "820ms" / "4.2s" / "1m 12s"; the tool-call list's whole point. */
function fmtMillis(ms: number): string {
  if (ms < 1_000) return `${Math.max(0, Math.round(ms))}ms`
  if (ms < 60_000) return `${(ms / 1_000).toFixed(1)}s`

  return fmtDuration(ms / 1_000)
}

/**
 * Renderable id for one card, so the overlay can scroll its selection into view.
 *
 * The overlay frame is bounded by the terminal, so ←/→ can move the selection
 * to a row below the fold; without this the selected agent — and the retry hint
 * that only renders on it — was simply invisible until the user scrolled by hand.
 */
export const agentCardRenderableId = (agentId: string): string => `agent-card:${agentId}`

function AgentCardView({
  now,
  onOpen,
  record,
  retryNote,
  selected,
  t
}: {
  now: number
  onOpen?: (agentId: string) => void
  record: AgentPanelRecord
  retryNote?: string
  selected?: boolean
  t: Theme
}) {
  const { item } = record
  const status = statusPresentation(item.status, t)
  const role = item.agentType?.trim() || 'agent'
  const model = item.model?.trim()
  const depth = Math.min(4, Math.max(0, item.depth))
  const task = compactLine(item.goal?.trim() || '', 120)
  // The title is derived from the goal when nothing better exists, so repeating
  // it as a task line would just be the same words twice.
  const showTask = Boolean(task) && !record.title.toLowerCase().startsWith(task.slice(0, 12).toLowerCase())

  return (
    <Box
      backgroundColor={selected ? t.color.selectionBg : t.color.completionCurrentBg}
      flexDirection="row"
      flexShrink={0}
      id={agentCardRenderableId(item.id)}
      marginBottom={1}
      marginLeft={depth}
      {...(onOpen ? { onClick: () => onOpen(item.id) } : {})}
      paddingRight={1}
      paddingY={1}
    >
      <Box backgroundColor={selected ? t.color.accent : status.color} flexShrink={0} width={1} />
      <Box flexDirection="column" flexGrow={1} flexShrink={1} paddingLeft={1}>
        <Text color={t.color.text} wrap="truncate-end">
          <Span color={status.color}>{status.glyph} </Span>
          <Span bold color={t.color.text}>
            {record.title}
          </Span>
          <Span color={t.color.muted}> · {item.status}</Span>
        </Text>
        <Text color={t.color.accent} wrap="truncate-end">
          {metricLine(item, now)}
          {record.childCount ? ` · ${record.childCount} child${record.childCount === 1 ? '' : 'ren'}` : ''}
        </Text>
        {showTask ? (
          <Text color={t.color.muted} wrap="truncate-end">
            task · {task}
          </Text>
        ) : null}
        <Text color={t.color.muted} dimColor wrap="truncate-end">
          ↳ {record.creatorTitle} · {role}
          {model ? ` · ${model}` : ''}
          {record.archived && record.snapshotLabel ? ` · ${record.snapshotLabel}` : ''}
        </Text>
        {retryNote ? (
          <Text color={retryNote.startsWith('↻') ? t.color.accent : t.color.error} wrap="truncate-end">
            {compactLine(retryNote, 140)}
          </Text>
        ) : selected && subagentRetryable(item.status) ? (
          // An affordance, not commentary: it appears on the selected row only,
          // and it is the one thing the list can act on directly.
          <Text color={t.color.accent} dimColor wrap="truncate-end">
            {subagentFailed(item.status) ? '↻ press r to retry this agent' : '↻ press r to run this agent again'}
          </Text>
        ) : null}
      </Box>
    </Box>
  )
}

const AgentCard = memo(
  AgentCardView,
  (previous, next) =>
    previous.t === next.t &&
    previous.now === next.now &&
    previous.onOpen === next.onOpen &&
    previous.retryNote === next.retryNote &&
    previous.selected === next.selected &&
    previous.record.item === next.record.item &&
    previous.record.archived === next.record.archived &&
    previous.record.childCount === next.record.childCount &&
    previous.record.creatorTitle === next.record.creatorTitle &&
    previous.record.snapshotLabel === next.record.snapshotLabel &&
    previous.record.title === next.record.title
)

/**
 * The inspector: one agent, in full.
 *
 * Everything the row deliberately drops lives here — what it is doing right
 * now, every tool call it made and how long each took, the files it touched,
 * and the policy it runs under.
 */
function AgentDetailView({
  now,
  record,
  retryNote,
  scrollRef,
  t
}: {
  now: number
  record: AgentPanelRecord
  retryNote?: string
  scrollRef?: MutableRefObject<ScrollBoxRenderable | null>
  t: Theme
}) {
  const { item } = record
  const status = statusPresentation(item.status, t)
  const elapsed = subagentElapsedSeconds(item, now)
  const rules = item.rules?.length ? item.rules.join(', ') : 'inherited defaults'
  const toolsets = item.toolsets?.length ? item.toolsets.join(', ') : 'runtime policy'
  const read = item.filesRead ?? []
  const written = item.filesWritten ?? []
  const calls = item.toolCalls ?? []
  // Newest first: on a long run the call you want is the one happening now.
  const ordered = [...calls].reverse()
  const thinking = item.thinking.at(-1)?.trim()
  const notes = item.notes.slice(-3).filter(note => note.trim())

  return (
    <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0}>
      <Box flexDirection="column" flexShrink={0} marginBottom={1}>
        <Text color={t.color.text} wrap="truncate-end">
          <Span color={status.color}>{status.glyph} </Span>
          <Span bold color={t.color.text}>
            {record.title}
          </Span>
          <Span color={t.color.muted}> · {item.status}</Span>
          {elapsed == null ? null : <Span color={t.color.muted}> · {fmtDuration(elapsed)}</Span>}
        </Text>
        <Text color={t.color.accent} wrap="truncate-end">
          {tokenDetail(item)}
        </Text>
        <Text color={t.color.muted} wrap="truncate-end">
          ↳ {record.creatorTitle} · {item.agentType?.trim() || 'agent'}
          {item.model?.trim() ? ` · ${item.model.trim()}` : ''}
          {record.childCount ? ` · ${record.childCount} child${record.childCount === 1 ? '' : 'ren'}` : ''}
        </Text>
      </Box>
      <scrollbox ref={scrollRef} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} viewportCulling>
        <Box flexDirection="column" flexShrink={0}>
          <Text bold color={t.color.text}>
            task
          </Text>
          <Text color={t.color.muted} wrap="wrap">
            {item.goal?.trim() || 'no task recorded'}
          </Text>

          {TERMINAL_STATUSES.has(item.status) && item.summary?.trim() ? (
            <>
              <Text bold color={t.color.text}>
                result
              </Text>
              <Text color={t.color.text} wrap="wrap">
                {compactLine(item.summary.trim(), 600)}
              </Text>
            </>
          ) : (
            <>
              <Box flexShrink={0} marginTop={1}>
                  <Text bold color={t.color.text}>
                    doing now
                  </Text>
                </Box>
              <Text color={t.color.text} wrap="wrap">
                {compactLine(activitySummary(item), 400)}
              </Text>
              {thinking ? (
                <Text color={t.color.muted} dimColor wrap="wrap">
                  thinking · {compactLine(thinking, 300)}
                </Text>
              ) : null}
            </>
          )}

          <Box flexShrink={0} marginTop={1}>
              <Text bold color={t.color.text}>
                tool calls ({calls.length ? calls.length : agentToolCount(item)})
              </Text>
            </Box>
          {ordered.length ? (
            ordered.map(call => {
              const finished = call.endedAt !== undefined
              const took = finished ? fmtMillis(call.endedAt! - call.startedAt) : fmtMillis(now - call.startedAt)
              const glyph = !finished ? '▸' : call.ok === false ? '✗' : '✓'
              const color = !finished ? t.color.accent : call.ok === false ? t.color.error : t.color.muted

              return (
                <Box flexDirection="column" flexShrink={0} key={call.id}>
                  <Text color={t.color.text} wrap="truncate-end">
                    <Span color={color}>{glyph} </Span>
                    {call.name}
                    <Span color={color}> · {took}{finished ? '' : ' so far'}</Span>
                  </Text>
                  {call.preview?.trim() ? (
                    <Text color={t.color.muted} dimColor wrap="truncate-end">
                      {'   '}
                      {compactLine(call.preview, 160)}
                    </Text>
                  ) : null}
                </Box>
              )
            })
          ) : (
            <Text color={t.color.muted} dimColor wrap="wrap">
              {agentToolCount(item)
                ? 'this agent ran before the inspector was recording call timings'
                : 'no tool calls yet'}
            </Text>
          )}

          {notes.length ? (
            <>
              <Box flexShrink={0} marginTop={1}>
                  <Text bold color={t.color.text}>
                    activity
                  </Text>
                </Box>
              {notes.map((note, index) => (
                <Text color={t.color.muted} key={index} wrap="truncate-end">
                  {compactLine(note, 200)}
                </Text>
              ))}
            </>
          ) : null}

          <Box flexShrink={0} marginTop={1}>
              <Text bold color={t.color.text}>
                files
              </Text>
            </Box>
          <Text color={t.color.muted} wrap="wrap">
            {written.length || read.length
              ? `${written.length} wrote · ${read.length} read${
                  written.length || read.length
                    ? ` · ${[...written.map(path => `+${basename(path)}`), ...read.map(basename)]
                        .slice(0, 12)
                        .join(', ')}`
                    : ''
                }`
              : 'none touched'}
          </Text>

          <Box flexShrink={0} marginTop={1}>
              <Text bold color={t.color.text}>
                policy
              </Text>
            </Box>
          <Text color={t.color.muted} wrap="wrap">
            rules · {rules}
          </Text>
          <Text color={t.color.muted} wrap="wrap">
            access · {toolsets}
          </Text>

          {retryNote ? (
            <Box flexShrink={0} marginTop={1}>
              <Text color={retryNote.startsWith('↻') ? t.color.accent : t.color.error} wrap="wrap">
                {compactLine(retryNote, 300)}
              </Text>
            </Box>
          ) : subagentRetryable(item.status) ? (
            <Box flexShrink={0} marginTop={1}>
              <Text color={t.color.accent} dimColor wrap="truncate-end">
                {subagentFailed(item.status) ? '↻ press r to retry this agent' : '↻ press r to run this agent again'}
              </Text>
            </Box>
          ) : null}
        </Box>
      </scrollbox>
    </Box>
  )
}

function AgentPanelBody({
  history,
  liveAgents,
  now,
  onOpen,
  openRecord,
  retryEnabled,
  retryNotes,
  scrollRef,
  selectedId,
  t,
  variant
}: AgentPanelProps & {
  now?: number
  onOpen?: (agentId: string) => void
  openRecord?: AgentPanelRecord | undefined
  scrollRef?: MutableRefObject<ScrollBoxRenderable | null>
}) {
  const records = useMemo(() => collectAgentPanelRecords(liveAgents, history), [history, liveAgents])
  const activeCount = records.filter(
    record => record.item.status === 'running' || record.item.status === 'queued'
  ).length
  const tick = now ?? Date.now()
  const footer = openRecord
    ? retryEnabled
      ? '↑↓ scroll · r retry · Esc back to the list'
      : '↑↓ scroll · PgUp/PgDn · Esc back to the list'
    : variant === 'overlay'
      ? retryEnabled
        ? '↑↓ select · Enter inspect · r retry dead agent · F6/Esc close'
        : '↑↓ select · Enter inspect · PgUp/PgDn · F6/Esc close'
      : 'F6 inspect · /agents'

  return (
    <Box
      backgroundColor={t.color.completionBg}
      borderColor={variant === 'overlay' ? t.color.border : undefined}
      borderStyle={variant === 'overlay' ? 'round' : undefined}
      flexDirection="column"
      // Both variants fill a parent that already owns the height (the rail
      // column, or the overlay's `panelHeight`). The overlay previously grew
      // from its content with `flexShrink: 0` and no height, so a long agent
      // list pushed the frame past the bottom of the terminal and the last card
      // plus the footer were clipped off-screen. Filling the parent instead
      // hands the overflow to the scrollbox below, which is what scrolls.
      flexShrink={1}
      height="100%"
      minHeight={0}
      paddingX={variant === 'sidebar' ? 2 : 1}
      paddingY={1}
      width="100%"
    >
      <Box flexDirection="row" flexShrink={0} justifyContent="space-between" marginBottom={1}>
        <Text bold color={t.color.text}>
          <Span color={t.color.accent}>◆ </Span>
          {openRecord ? 'Agent' : 'Agents'}
        </Text>
        <Text color={activeCount ? t.color.accent : t.color.muted}>
          {activeCount ? `${activeCount} live` : `${records.length} done`}
        </Text>
      </Box>
      {openRecord ? (
        <AgentDetailView
          now={tick}
          record={openRecord}
          {...(retryNotes?.get(openRecord.item.id) ? { retryNote: retryNotes.get(openRecord.item.id) } : {})}
          scrollRef={scrollRef}
          t={t}
        />
      ) : (
      <scrollbox ref={scrollRef} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} viewportCulling>
        <Box flexDirection="column" flexShrink={0}>
          {records.length ? (
            records.map(record => (
              <AgentCard
                key={`${record.archived ? 'past' : 'live'}:${record.item.id}`}
                now={tick}
                {...(onOpen ? { onOpen } : {})}
                record={record}
                {...(retryNotes?.get(record.item.id) ? { retryNote: retryNotes.get(record.item.id) } : {})}
                selected={record.item.id === selectedId}
                t={t}
              />
            ))
          ) : (
            // Deliberately not vertically centred inside a reserved block: the
            // frame is bounded by its parent now, so a 5-row centred placeholder
            // on a short terminal put its first visible row in the blank padding
            // above the text and the message read as an empty panel.
            <Box alignItems="center" flexDirection="column" flexShrink={0}>
              <Text color={t.color.muted}>No agents yet</Text>
              <Text color={t.color.muted} dimColor>
                Delegated work appears here.
              </Text>
            </Box>
          )}
        </Box>
      </scrollbox>
      )}
      {/* Truncate rather than wrap: on a narrow panel this hint wrapped to two
          rows and, now that the frame is bounded, those rows came out of the
          agent list rather than out of the terminal. */}
      <Text color={t.color.muted} dimColor wrap="truncate-end">
        {footer}
      </Text>
    </Box>
  )
}

export function AgentPanel({
  onInspect,
  ...props
}: Omit<AgentPanelProps, 'variant'> & {
  /** Clicking a rail card opens the overlay straight into that agent. */
  onInspect?: (agentId: string) => void
}) {
  if (!collectAgentPanelRecords(props.liveAgents, props.history).length) return null

  return <AgentPanelBody {...props} {...(onInspect ? { onOpen: onInspect } : {})} variant="sidebar" />
}

const consumeKey = (event: KeyEvent) => {
  event.preventDefault()
  event.stopPropagation()
}

export function AgentPanelHotkey({
  disabled,
  open,
  onToggle,
  resizeEnabled = false
}: {
  disabled: boolean
  open: boolean
  onToggle: (open: boolean) => void
  /** Allow Shift+Cmd/Ctrl/Option+←/→ panel-width chords (sidebar or overlay visible). */
  resizeEnabled?: boolean
}) {
  useKeyboard(event => {
    if (!disabled && resizeEnabled && isPanelResizeKey(event)) {
      adjustPanelWidth(event.name === 'right' ? PANEL_WIDTH_STEP : -PANEL_WIDTH_STEP)
      consumeKey(event)
      return
    }
    if (disabled || event.name !== 'f6') return
    onToggle(!open)
    consumeKey(event)
  })

  return null
}

export function AgentPanelOverlay({
  history,
  initialInspectId,
  liveAgents,
  onClose,
  t
}: AgentPanelOverlayProps) {
  const scrollRef = useRef<ScrollBoxRenderable | null>(null)
  const { height, width } = useTerminalDimensions()
  const gateway = useOptionalGateway()
  // Preferred vertical breathing room for the overlay (user request: 30 rows
  // top and bottom). On short terminals the margin shrinks toward the old 1-2
  // rows so the panel keeps ~20 usable rows instead of collapsing.
  const marginY = Math.min(30, Math.max(1, Math.floor((height - 20) / 2)))
  const panelHeight = Math.max(1, height - marginY * 2)
  const page = Math.max(4, panelHeight - 8)
  const panelWidth = withPanelWidthDelta(Math.max(1, Math.min(96, width - 2)), width)
  const records = useMemo(() => collectAgentPanelRecords(liveAgents, history), [history, liveAgents])
  const [selectedId, setSelectedId] = useState<null | string>(null)
  const [openId, setOpenId] = useState<null | string>(initialInspectId ?? null)
  const [retryNotes, setRetryNotes] = useState<ReadonlyMap<string, string>>(new Map())
  const pendingRetries = useRef(new Set<string>())
  // Elapsed time has to advance on its own. A queued agent publishes no events
  // at all, and a thinking one can go a minute between them — without a clock
  // its "running for 4s" sat frozen at 4s and read as a hung agent.
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1_000)
    return () => clearInterval(timer)
  }, [])

  // Default the selection to the first retryable (dead) agent — the rows a
  // user most likely opened the overlay to act on — else the first row.
  const selectedRecord = useMemo(() => {
    if (!records.length) return undefined
    const explicit = selectedId ? records.find(record => record.item.id === selectedId) : undefined
    return explicit ?? records.find(record => subagentFailed(record.item.status)) ?? records[0]
  }, [records, selectedId])

  // Keep the selected card on screen now that the frame is bounded by the
  // terminal instead of overflowing it.
  //
  // The scroll is attempted twice on purpose. On the commit that first mounts a
  // card, Yoga has not positioned it yet — every renderable still reports y=0
  // and height=0, so scrollChildIntoView computes a zero delta and silently
  // does nothing. The deferred call runs after that layout pass, one frame
  // later, which is imperceptible interactively. The list length is a dependency
  // because adding or removing an agent shifts every offset below it.
  const selectedCardId = selectedRecord?.item.id
  const openRecord = openId ? records.find(record => record.item.id === openId) : undefined
  useEffect(() => {
    // The detail view replaces the list, so there is no card to scroll to and
    // the scrollbox belongs to the inspector's own content.
    if (!selectedCardId || openId) return
    const target = agentCardRenderableId(selectedCardId)
    scrollRef.current?.scrollChildIntoView(target)
    const settle = setTimeout(() => scrollRef.current?.scrollChildIntoView(target), 0)
    return () => clearTimeout(settle)
  }, [openId, records.length, selectedCardId])

  const setRetryNote = (id: string, note: string) => {
    setRetryNotes(previous => {
      const next = new Map(previous)
      next.set(id, note)
      return next
    })
  }

  const moveSelection = (delta: number) => {
    if (!records.length) return
    const currentIndex = selectedRecord ? records.indexOf(selectedRecord) : 0
    const nextIndex = Math.min(records.length - 1, Math.max(0, currentIndex + delta))
    const next = records[nextIndex]
    if (next) setSelectedId(next.item.id)
  }

  const retrySelected = () => {
    const record = openRecord ?? selectedRecord
    if (!record) return
    const item = record.item
    if (!gateway) {
      setRetryNote(item.id, 'retry unavailable: not connected to the daemon')
      return
    }
    if (!subagentRetryable(item.status)) {
      setRetryNote(item.id, `cannot retry: agent is still ${item.status} — wait or stop it first`)
      return
    }
    // One in-flight retry per agent; the daemon additionally deduplicates a
    // retry of an already-running task, so a double-press never double-spawns.
    if (pendingRetries.current.has(item.id)) return
    pendingRetries.current.add(item.id)
    setRetryNote(item.id, '↻ retry requested — resuming agent with its prior conversation…')
    retrySubagent(gateway.rpc, item.name?.trim() || item.id)
      .then(response => {
        if (response.ok === false) {
          setRetryNote(item.id, `retry failed: ${response.error?.trim() || 'the daemon rejected the retry'}`)
          return
        }
        const status = response.agent?.status?.trim() || 'running'
        setRetryNote(item.id, `↻ retry accepted — same agent resumed (${status}); watch for live progress`)
      })
      .catch((error: unknown) => {
        setRetryNote(item.id, `retry failed: ${error instanceof Error && error.message ? error.message : 'request failed'}`)
      })
      .finally(() => {
        pendingRetries.current.delete(item.id)
      })
  }

  useKeyboard(event => {
    const isEnter = event.name === 'return' || event.name === 'enter' || event.name === 'kpenter'

    if (isPanelResizeKey(event)) {
      adjustPanelWidth(event.name === 'right' ? PANEL_WIDTH_STEP : -PANEL_WIDTH_STEP)
    } else if (event.name === 'escape' || event.name === 'f6' || event.sequence === 'q') {
      // Esc steps back to the list before it closes the panel: the inspector is
      // a place inside /agents, not a modal stacked on top of it. Only from the
      // list does Esc return you to the main agent.
      if (openId) setOpenId(null)
      else onClose()
    } else if (isEnter || (event.name === 'right' && !openId)) {
      if (!openId && selectedRecord) setOpenId(selectedRecord.item.id)
    } else if (event.name === 'left') {
      if (openId) setOpenId(null)
      else moveSelection(-1)
    } else if (event.name === 'right') {
      moveSelection(1)
    } else if (event.sequence === 'r') {
      retrySelected()
    } else if (event.name === 'up') {
      if (openId) scrollRef.current?.scrollBy(-1)
      else moveSelection(-1)
    } else if (event.name === 'down') {
      if (openId) scrollRef.current?.scrollBy(1)
      else moveSelection(1)
    } else if (event.name === 'pageup') {
      scrollRef.current?.scrollBy(-page)
    } else if (event.name === 'pagedown') {
      scrollRef.current?.scrollBy(page)
    } else if (event.name === 'home') {
      scrollRef.current?.scrollTo(0)
    } else if (event.name === 'end') {
      scrollRef.current?.scrollTo(Number.MAX_SAFE_INTEGER)
    } else {
      return
    }

    consumeKey(event)
  })

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
      zIndex={180}
    >
      {/* `panelWidth` is already clamped by withPanelWidthDelta. A maxWidth of
          96 alongside it only contradicted the resize chord — min-width wins
          over max-width, so the cap never applied once a user widened. */}
      <Box flexDirection="column" flexShrink={0} height={panelHeight} width={panelWidth}>
        <AgentPanelBody
          history={history}
          liveAgents={liveAgents}
          now={now}
          onOpen={setOpenId}
          openRecord={openRecord}
          retryEnabled={Boolean(gateway)}
          retryNotes={retryNotes}
          scrollRef={scrollRef}
          selectedId={selectedRecord?.item.id ?? null}
          t={t}
          variant="overlay"
        />
      </Box>
    </box>
  )
}
