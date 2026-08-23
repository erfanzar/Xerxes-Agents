// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import type { KeyEvent, ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { type MutableRefObject, memo, useEffect, useMemo, useRef, useState } from 'react'

import { useOptionalGateway } from '../app/gatewayContext.js'
import { adjustPanelWidth, PANEL_WIDTH_STEP, withPanelWidthDelta } from '../app/panelSizeStore.js'

import {
  AGENT_GROUP_LABEL,
  AGENT_GROUP_STATE,
  agentGroup,
  agentHeading,
  orderAgentRecords,
  type AgentGroup
} from '../lib/agentGroups.js'
import {
  densityFor,
  GLYPH,
  RAIL_DENSITY,
  type NocturneDensity,
  type StateSkin,
  stateSkin
} from '../domain/nocturne.js'

import { agentSidebarWidth } from '../domain/agentPanelLayout.js'

import { OVERLAY_PANEL_SPECS, overlayPanelSize, overlayPanelWidth } from './overlayLayout.js'
import type { SpawnSnapshot } from '../app/spawnHistoryStore.js'
import type { SubagentInterruptResponse } from '../gatewayTypes.js'
export { AGENT_SIDEBAR_BREAKPOINT, shouldShowAgentSidebar } from '../domain/agentPanelLayout.js'
import { retrySubagent, subagentFailed, subagentRetryable } from '../lib/agentRetry.js'
import { fmtDuration, fmtTokens, subagentElapsedSeconds } from '../lib/subagentElapsed.js'
import type { Theme } from '../theme.js'
import type { SubagentProgress } from '../types.js'

import { isPanelResizeKey } from './diffPanel.js'
import { isPageDownKey, isPageUpKey, PAGE_KEY_HINT } from '../lib/pageKeys.js'

import { GroupCaption, LeaderRow } from './nocturne.js'
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
  /** Drop the header counts — panels too narrow to fit them legibly. */
  compactHeader?: boolean
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

  // Ordered here, at the single place records are built, so the overlay, the
  // sidebar, keyboard selection and the inspector all walk the same sequence.
  return orderAgentRecords(rows).map(row => ({
    ...row,
    childCount: childCounts.get(row.item.id) ?? 0,
    creatorTitle:
      (row.item.creatorId && titles.get(row.item.creatorId)) ||
      (row.item.parentId && titles.get(row.item.parentId)) ||
      'Xerxes',
    title: shortAgentTitle(row.item)
  }))
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

/**
 * The card's voice. One lookup through the Nocturne state table, which also
 * hands back the tinted ground and edge the card is drawn on, so "needs you"
 * and "went wrong" differ in surface as well as in dot.
 */
function agentCardVoice(item: SubagentProgress, t: Theme): { color: string; group: AgentGroup; skin: StateSkin } {
  const group = agentGroup(item.status)
  const skin = stateSkin(AGENT_GROUP_STATE[group], t.ds)

  return { color: skin.dot, group, skin }
}

/**
 * Right-aligned budget for the card's first row: elapsed time plus tokens.
 *
 * The same shape on every card, so a runaway agent shows up as a wide right
 * column. `withTokens` is the first thing screen 09 gives up as the terminal
 * narrows — the elapsed clock answers "is this stuck", which is the question
 * you are scanning for; the token count answers "what did it cost", which
 * can wait for a wider window.
 */
function cardBudget(item: SubagentProgress, now: number, withTokens = true): string {
  const tokens = (item.inputTokens ?? 0) + (item.outputTokens ?? 0)
  const elapsed = subagentElapsedSeconds(item, now)
  const parts = [elapsed != null ? fmtDuration(elapsed) : item.status]

  if (withTokens && tokens > 0) {
    parts.push(`${fmtTokens(tokens)} tok`)
  }

  return parts.join(' · ')
}

/** The one line of substance under the title, per action group. */
function cardSecondLine(item: SubagentProgress): { color: keyof Theme['color'] | undefined; text: string } | null {
  const group = agentGroup(item.status)

  if (group === 'working') {
    const activity = activitySummary(item)

    return activity ? { color: 'thinking', text: `└ ${compactLine(activity, 140)}` } : null
  }

  if (group === 'review') {
    const summary = item.summary?.trim()

    return summary ? { color: 'label', text: compactLine(summary, 140) } : null
  }

  // Needs input: the agent's last word is what you are walking into.
  const note = item.notes.at(-1)?.trim() || item.summary?.trim()

  return note
    ? { color: 'label', text: compactLine(note, 140) }
    : { color: 'muted', text: `waiting · ${item.status}` }
}

function AgentCardView({
  collapseCards,
  density,
  framed,
  now,
  onOpen,
  record,
  retryNote,
  selected,
  t
}: {
  /**
   * Collapse every card except the blocked one to its single-line form.
   *
   * Deliberately NOT `!density.goals`: the rail drops goals because a goal
   * does not fit beside a title at 40 columns, but it still wants the one
   * violet activity line that says what the agent is doing. Tying the two
   * together silenced the rail entirely.
   */
  collapseCards: boolean
  /** What this width can afford. See `densityFor` for the order of sacrifice. */
  density: NocturneDensity
  /** Overlay cards draw the mockup's rounded outline; the rail stays flat. */
  framed?: boolean
  now: number
  onOpen?: (agentId: string) => void
  record: AgentPanelRecord
  retryNote?: string
  selected?: boolean
  t: Theme
}) {
  const { item } = record
  const voice = agentCardVoice(item, t)
  const role = item.agentType?.trim() || 'agent'
  const model = item.model?.trim()
  const goal = item.goal?.trim() || ''
  const depth = Math.min(4, Math.max(0, item.depth))
  // The title derives from the goal when nothing better exists, so repeating
  // it as a task line would just be the same words twice.
  const showGoal = Boolean(goal) && !record.title.toLowerCase().startsWith(goal.slice(0, 12).toLowerCase())
  const second = cardSecondLine(item)
  const secondColor = second?.color ? t.color[second.color] : undefined

  // Mockup 04's accent edge: every card carries its voice colour on the
  // left, and the card asking for attention — keyboard-selected or sitting in
  // NEEDS INPUT — gets the thicker edge the design draws at 3px.
  const emphasized = Boolean(selected) || voice.group === 'input'

  // Two reasons a card collapses to one line.
  //
  // A failed run has already spent its money and does not get to spend your
  // attention too, so it stays one dim line until you select it. And on a
  // narrow terminal every card except the one actually blocked collapses,
  // because six one-line agents beat three legible ones when you are
  // scanning for the amber dot — cards shrink rather than reflowing taller.
  const collapsed = (voice.group === 'failed' || (collapseCards && voice.group !== 'input')) && !selected

  if (collapsed) {
    return (
      <Box
        flexDirection="row"
        flexShrink={0}
        id={agentCardRenderableId(item.id)}
        marginLeft={depth}
        {...(onOpen ? { onClick: () => onOpen(item.id) } : {})}
        paddingRight={1}
        width="100%"
      >
        <Box flexGrow={1} flexShrink={1} minWidth={0} overflow="hidden">
          <Text wrap="truncate-end">
            <Span color={voice.color}>{`${GLYPH.state} `}</Span>
            <Span color={t.ds.secondary}>{record.title}</Span>
            {density.goals && second ? (
              <Span color={t.ds.meta}>{`  ${compactLine(second.text, 90)}`}</Span>
            ) : null}
          </Text>
        </Box>
        <Box flexShrink={0}>
          <Text color={t.ds.numeric}>{` ${cardBudget(item, now, density.cardBudget)}`}</Text>
        </Box>
      </Box>
    )
  }

  return (
    <Box
      backgroundColor={voice.skin.ground}
      {...(framed ? { borderColor: selected ? voice.skin.dot : voice.skin.border, borderStyle: 'round' as const } : {})}
      flexDirection="row"
      flexShrink={0}
      id={agentCardRenderableId(item.id)}
      marginBottom={1}
      marginLeft={depth}
      {...(onOpen ? { onClick: () => onOpen(item.id) } : {})}
      paddingRight={1}
      paddingY={framed ? 0 : 1}
    >
      <Box backgroundColor={selected ? t.color.accent : voice.color} flexShrink={0} width={emphasized ? 2 : 1} />
      <Box flexDirection="column" flexGrow={1} flexShrink={1} paddingLeft={1}>
        {/* Row one: status dot, voice-coloured title, muted goal — with the
            elapsed/token budget hanging off the right edge. */}
        <Box flexDirection="row" flexShrink={0}>
          <Box flexGrow={1} flexShrink={1} minWidth={0} overflow="hidden">
            {/* The DOT carries the state and the TITLE stays on the ramp.
                Colouring both said the same thing twice and left a board of
                four amber words when one amber dot was the whole message. */}
            <Text wrap="truncate-end">
              <Span color={voice.color}>{`${GLYPH.state} `}</Span>
              <Span bold color={t.ds.title}>
                {record.title}
              </Span>
              {showGoal && density.goals ? <Span color={t.ds.meta}>{`  ${compactLine(goal, 90)}`}</Span> : null}
            </Text>
          </Box>
          {/* Same shape on every card — `elapsed · tokens`, right-aligned —
              so a runaway agent shows up as a wide right column rather than
              as a number you have to go looking for. */}
          <Box flexShrink={0}>
            <Text color={t.ds.numeric}>{` ${cardBudget(item, now, density.cardBudget)}`}</Text>
          </Box>
        </Box>
        {/* Row two: the single line of substance — violet while it works,
            its result once it has one, its last word when it needs you. */}
        {second ? (
          <Text color={secondColor} wrap="truncate-end">
            {second.text}
          </Text>
        ) : (
          <Text color={t.color.muted}> </Text>
        )}
        <Text color={t.ds.caption} wrap="truncate-end">
          {GLYPH.wrap} {record.creatorTitle} · {role}
          {model ? ` · ${model}` : ''}
          {record.childCount ? ` · ${record.childCount} child${record.childCount === 1 ? '' : 'ren'}` : ''}
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
    previous.collapseCards === next.collapseCards &&
    previous.density === next.density &&
    previous.framed === next.framed &&
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
  paneWidth,
  record,
  retryNote,
  scrollRef,
  t
}: {
  now: number
  /** Columns this pane owns; absent means it has the whole overlay. */
  paneWidth?: number
  record: AgentPanelRecord
  retryNote?: string
  scrollRef?: MutableRefObject<ScrollBoxRenderable | null>
  t: Theme
}) {
  const { item } = record
  const voice = agentCardVoice(item, t)
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
  // The inspector renders only inside the overlay, whose width this hook
  // tracks; leaders size against it minus the frame's padding and borders.
  const { width } = useTerminalDimensions()

  /** Columns a leader row may run to inside the inspector's frame. */
  const rowWidth = Math.max(24, (paneWidth ?? width) - 6)

  return (
    <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0}>
      {/* Panel head: dot + title on their own row, the goal under it.
          Sharing one row with the goal AND a right-aligned elapsed crushed
          all three the moment this became a side pane — the title came out
          as `● Auth Migration...to the new schema8s elapsed`. The title is
          the handle and gets a row to itself. */}
      <Box flexDirection="column" flexShrink={0} marginBottom={1}>
        <Box flexDirection="row" flexShrink={0}>
          <Box flexGrow={1} flexShrink={1} minWidth={0} overflow="hidden">
            <Text wrap="truncate-end">
              <Span color={voice.color}>{`${GLYPH.state} `}</Span>
              <Span bold color={t.ds.title}>
                {record.title}
              </Span>
            </Text>
          </Box>
          <Box flexShrink={0}>
            <Text color={t.ds.numeric}>
              {elapsed == null ? item.status : ` ${fmtDuration(elapsed)}`}
            </Text>
          </Box>
        </Box>
        {item.goal?.trim() ? (
          <Text color={t.ds.meta} wrap="truncate-end">
            {compactLine(item.goal.trim(), Math.max(12, rowWidth))}
          </Text>
        ) : null}
        {/* Policy is what you are actually trusting while an agent runs, so
            identity rides directly under the title as chips. */}
        <Text wrap="truncate-end">
          <Span color={t.color.accent}>{`[${item.agentType?.trim() || 'agent'}]`}</Span>
          <Span color={t.color.muted}>
            {' '}
            {item.model?.trim() ? `[${item.model.trim()}] ` : ''}
            {`${agentToolCount(item)} tools`}
          </Span>
        </Text>
      </Box>
      <scrollbox ref={scrollRef} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} viewportCulling>
        <Box flexDirection="column" flexShrink={0}>
          {TERMINAL_STATUSES.has(item.status) && item.summary?.trim() ? (
            <>
              <Text color={t.color.muted} dimColor>
                RESULT
              </Text>
              <Text color={t.color.text} wrap="wrap">
                {compactLine(item.summary.trim(), 600)}
              </Text>
            </>
          ) : (
            <>
              <Text color={t.color.muted} dimColor>
                LIVE
              </Text>
              <Text color={t.color.text} wrap="wrap">
                {compactLine(activitySummary(item), 400)}
              </Text>
              {thinking ? (
                <Text color={t.color.muted} dimColor wrap="wrap">
                  {`└ ${compactLine(thinking, 300)}`}
                </Text>
              ) : null}
            </>
          )}

          <Box flexShrink={0} marginTop={1}>
              <GroupCaption
                count={calls.length ? calls.length : agentToolCount(item)}
                label="TOOL CALLS"
                t={t}
                width={rowWidth}
              />
            </Box>
          {ordered.length ? (
            ordered.map(call => {
              const finished = call.endedAt !== undefined
              const took = finished ? fmtMillis(call.endedAt! - call.startedAt) : fmtMillis(now - call.startedAt)
              const failedCall = call.ok === false
              // Mockup toolrow: ⏺ glyph, bold name, muted summary, dotted
              // leader, right-aligned duration — `running…` while live.
              const duration = finished ? took : 'running…'
              const preview = call.preview?.trim() ? compactLine(call.preview, 60) : ''

              // The shared leader row: glyph, verb, target, dotted leader,
              // right-aligned duration. Same component the transcript and the
              // diff review stats use, so durations line up into one readable
              // column wherever they appear.
              return (
                <LeaderRow
                  glyph={GLYPH.tool}
                  glyphColor={failedCall ? t.ds.failed : finished ? t.ds.done : t.ds.working}
                  key={call.id}
                  label={call.name}
                  quiet={!finished}
                  right={duration}
                  rightColor={finished ? t.ds.numeric : t.color.accent}
                  t={t}
                  {...(preview ? { target: preview } : {})}
                  width={rowWidth}
                />
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
                  <GroupCaption label="ACTIVITY" t={t} width={rowWidth} />
                </Box>
              {notes.map((note, index) => (
                <Text color={t.color.muted} key={index} wrap="truncate-end">
                  {compactLine(note, 200)}
                </Text>
              ))}
            </>
          ) : null}

          <Box flexShrink={0} marginTop={1}>
            <GroupCaption label="FILES TOUCHED" t={t} width={rowWidth} />
          </Box>
          {written.length || read.length ? (
            <>
              {[...written.map(path => `+${basename(path)}`), ...read.map(basename)]
                .slice(0, 8)
                .map((entry, index) => (
                  <Text color={t.color.label} key={index} wrap="truncate-end">
                    {entry}
                  </Text>
                ))}
              {written.length + read.length > 8 ? (
                <Text color={t.color.muted} dimColor wrap="truncate-end">
                  {`+${written.length + read.length - 8} more`}
                </Text>
              ) : null}
            </>
          ) : (
            <Text color={t.color.muted} wrap="wrap">
              none touched
            </Text>
          )}

          <Box flexShrink={0} marginTop={1}>
            <Text color={t.color.muted} dimColor>
              COST
            </Text>
          </Box>
          {/* tokenDetail already carries the dollar figure when the daemon
              reports one; appending it again here printed the cost twice. */}
          <Text color={t.color.muted} wrap="truncate-end">
            {tokenDetail(item)}
            {` · parent: ${record.creatorTitle}`}
          </Text>

          <Box flexShrink={0} marginTop={1}>
              <Text color={t.color.muted} dimColor>
                POLICY
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
  compactHeader,
  history,
  liveAgents,
  now,
  onOpen,
  openRecord,
  retryEnabled,
  retryNotes,
  scrollRef,
  selectedId,
  shortFrame,
  t,
  variant
}: AgentPanelProps & {
  now?: number
  onOpen?: (agentId: string) => void
  openRecord?: AgentPanelRecord | undefined
  scrollRef?: MutableRefObject<ScrollBoxRenderable | null>
  /** Too few rows for the header/footer hairlines — spend them on content. */
  shortFrame?: boolean
}) {
  const records = useMemo(() => collectAgentPanelRecords(liveAgents, history), [history, liveAgents])
  // Columns a caption's rule may run to. The rail and the overlay have very
  // different widths, so a constant here would either overshoot in the rail
  // or stop short in the overlay.
  const { width: terminalWidth } = useTerminalDimensions()
  const panelWidth = Math.max(
    24,
    (variant === 'sidebar' ? agentSidebarWidth(terminalWidth) : overlayPanelWidth(terminalWidth, OVERLAY_PANEL_SPECS.agents)) - 6
  )
  // Density is measured against the PANEL, not the terminal: a 200-column
  // terminal showing a 40-column overlay must degrade the overlay, not decide
  // it is roomy because the screen is. The rail opts out entirely — it is a
  // summary at every size, so its shape is a design decision rather than a
  // sacrifice, and screen 09's order does not apply to it.
  const density = useMemo(
    () => (variant === 'overlay' ? densityFor(panelWidth) : RAIL_DENSITY),
    [panelWidth, variant]
  )
  // Screen 03 puts the list and the inspector on screen together. A view that
  // REPLACES the list with a detail makes you remember which agent you were
  // on and press Esc to check — which is exactly what Enter used to do here.
  // The threshold matches the agent view's: the inspector needs ~40 columns
  // of its own, and taking those from a 100-column overlay leaves a list too
  // narrow to read the titles it exists to show.
  const twoPane = variant === 'overlay' && terminalWidth >= 120
  const inspectorWidth = twoPane ? Math.max(34, Math.min(56, Math.floor(panelWidth * 0.42))) : 0
  const listWidth = twoPane ? Math.max(24, panelWidth - inspectorWidth - 1) : panelWidth
  // The inspector follows the SELECTION, so it is never empty and never
  // disagrees with the row the arrow keys are on. An explicit `openRecord`
  // (clicking a rail card) still wins, so F6-from-the-rail lands where you
  // clicked.
  const inspectRecord =
    openRecord ?? records.find(record => record.item.id === selectedId) ?? records[0]
  const activeCount = records.filter(
    record => record.item.status === 'running' || record.item.status === 'queued'
  ).length
  const tick = now ?? Date.now()
  const footer = openRecord
    ? retryEnabled
      // Mockup 05 inspector foot: peek/cancel join retry. `space` stays
      // unadvertised until a transcript-peek surface exists for agents.
      ? '↑↓ scroll · r retry · c cancel · Esc back to the list'
      : `↑↓ scroll · ${PAGE_KEY_HINT} · Esc back to the list`
    : variant === 'overlay'
      ? retryEnabled
        ? '↑↓ select · Enter inspect · r retry dead agent · F6/Esc close'
        : // "Enter inspect" is a promise about a pane that is already on
          // screen once the overlay is wide enough to show both, so the wide
          // footer says what Enter actually adds: pinning it there.
          twoPane
          ? `↑↓ select · Enter pin · ${PAGE_KEY_HINT} · F6/Esc close`
          : `↑↓ select · Enter inspect · ${PAGE_KEY_HINT} · F6/Esc close`
      : 'F6 inspect · /agents'

  // Mockup 04/05 panel chrome: the header bar and the footer hint row are
  // ruled off from the content by hairlines. The sidebar rail keeps its flat
  // treatment, and a frame too short for rules spends those rows on content.
  const hairlines = variant === 'overlay' && !shortFrame

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
      paddingY={variant === 'overlay' && shortFrame ? 0 : 1}
      width="100%"
    >
      <box
        {...(hairlines ? { border: ['bottom' as const], borderColor: t.color.border } : {})}
        flexDirection="row"
        flexShrink={0}
        justifyContent="space-between"
        marginBottom={1}
      >
        {/* Title and counts live in separate clipped boxes rather than one
            truncating Text: span-heavy truncate at narrow widths can blank
            the whole run, while flex clipping degrades gracefully. */}
        <Box flexDirection="row" flexShrink={1} minWidth={0} overflow="hidden">
          <Text bold color={t.color.text} wrap="truncate-end">
            <Span color={t.color.accent}>✦ </Span>
            {openRecord ? 'Agent' : 'Agent View'}
          </Text>
          {openRecord || compactHeader ? null : (
            <Text color={t.color.muted} wrap="truncate-end">
              {`  ${records.length} chat${records.length === 1 ? '' : 's'} · ${activeCount} working`}
            </Text>
          )}
        </Box>
        {openRecord ? (
          <Box flexShrink={0}>
            <Text color={t.color.muted} wrap="truncate-end">
              {openRecord.item.status}
            </Text>
          </Box>
        ) : (
          <Box flexShrink={0}>
            <Text color={activeCount ? t.color.accent : t.color.muted}>{activeCount ? 'live' : 'idle'}</Text>
          </Box>
        )}
      </box>
      {openRecord && !twoPane ? (
        <AgentDetailView
          now={tick}
          record={openRecord}
          {...(retryNotes?.get(openRecord.item.id) ? { retryNote: retryNotes.get(openRecord.item.id) } : {})}
          scrollRef={scrollRef}
          t={t}
        />
      ) : records.length ? (
      <Box flexDirection="row" flexGrow={1} minHeight={0} width="100%">
      <Box flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0} minWidth={0}>
      <scrollbox ref={scrollRef} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} viewportCulling>
        <Box flexDirection="column" flexShrink={0}>
          {records.map((record, index) => {
            const heading = agentHeading(records, index)

            return (
              <Box
                flexDirection="column"
                flexShrink={0}
                key={`${record.archived ? 'past' : 'live'}:${record.item.id}`}
              >
                {heading ? (
                  // The action-group caption wears its group's voice, so
                  // NEEDS INPUT is amber from across the room and a board
                  // with nothing blocked carries no amber at all.
                  <GroupCaption
                    count={records.filter(row => agentGroup(row.item.status) === agentGroup(record.item.status)).length}
                    label={AGENT_GROUP_LABEL[agentGroup(record.item.status)]}
                    t={t}
                    tone={stateSkin(AGENT_GROUP_STATE[agentGroup(record.item.status)], t.ds).dot}
                    width={listWidth}
                  />
                ) : null}
                <AgentCard
                  collapseCards={variant === 'overlay' && !density.goals}
                  density={density}
                  framed={variant === 'overlay'}
                  now={tick}
                  {...(onOpen ? { onOpen } : {})}
                  record={record}
                  {...(retryNotes?.get(record.item.id) ? { retryNote: retryNotes.get(record.item.id) } : {})}
                  selected={record.item.id === selectedId}
                  t={t}
                />
              </Box>
            )
          })}
        </Box>
      </scrollbox>
      </Box>
      {twoPane && inspectRecord ? (
        <>
          {/* A filled column rather than a per-side border: OpenTUI paints an
              edge THROUGH the text when a bordered child sits inside a framed
              parent, and this panel has a frame. */}
          <Box backgroundColor={t.ds.hairline} flexShrink={0} width={1} />
          <Box flexDirection="column" flexShrink={0} minHeight={0} paddingLeft={1} width={inspectorWidth}>
            <AgentDetailView
              now={tick}
              paneWidth={inspectorWidth}
              record={inspectRecord}
              {...(retryNotes?.get(inspectRecord.item.id) ? { retryNote: retryNotes.get(inspectRecord.item.id) } : {})}
              t={t}
            />
          </Box>
        </>
      ) : null}
      </Box>
      ) : (
        // The panel keeps its full size when empty, so the placeholder
        // centers inside it — a tiny box shrink-wrapped around two lines
        // was the rejected look.
        <Box alignItems="center" flexDirection="column" flexGrow={1} flexShrink={1} justifyContent="center" minHeight={0}>
          <Text color={t.color.muted}>No agents yet</Text>
          <Text color={t.color.muted} dimColor>
            Delegated work appears here.
          </Text>
        </Box>
      )}
      {/* Truncate rather than wrap: on a narrow panel this hint wrapped to two
          rows and, now that the frame is bounded, those rows came out of the
          agent list rather than out of the terminal. */}
      <box
        {...(hairlines ? { border: ['top' as const], borderColor: t.color.border } : {})}
        flexDirection="column"
        flexShrink={0}
      >
        <Text color={t.color.muted} dimColor wrap="truncate-end">
          {footer}
        </Text>
      </box>
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
  const records = useMemo(() => collectAgentPanelRecords(liveAgents, history), [history, liveAgents])
  // Shared with F7/F8 so the three overlays stop diverging. Mockup 04 sizes
  // the agent view as a LARGE bounded panel — full height minus the standard
  // gutter, diff-width — even when it is empty: the empty state centers
  // inside the frame instead of collapsing the frame around itself.
  const { height: panelHeight, width: fittedWidth } = overlayPanelSize(
    { height, width },
    OVERLAY_PANEL_SPECS.agents
  )
  const page = Math.max(4, panelHeight - 8)
  const panelWidth = withPanelWidthDelta(fittedWidth, width)
  const [selectedId, setSelectedId] = useState<null | string>(null)
  const [openId, setOpenId] = useState<null | string>(initialInspectId ?? null)
  const [retryNotes, setRetryNotes] = useState<ReadonlyMap<string, string>>(new Map())
  const pendingRetries = useRef(new Set<string>())
  const pendingCancels = useRef(new Set<string>())
  // Elapsed time has to advance on its own. A queued agent publishes no events
  // at all, and a thinking one can go a minute between them — without a clock
  // its "running for 4s" sat frozen at 4s and read as a hung agent.
  const [now, setNow] = useState(() => Date.now())
  const hasRunningClock = records.some(record => record.item.status === 'running' || record.item.status === 'queued')
  useEffect(() => {
    if (!hasRunningClock) return
    const timer = setInterval(() => setNow(Date.now()), 1_000)
    return () => clearInterval(timer)
  }, [hasRunningClock])

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

  // Mockup 05: `c` cancels the inspected agent. The only existing
  // agent-cancel surface is the documented `subagent.interrupt` RPC, so the
  // request routes through it and its typed failure is surfaced verbatim in
  // the same note channel retry uses — never a fabricated success. When the
  // daemon grows real cancellation, this binding starts working unchanged.
  const cancelSelected = () => {
    const record = openRecord ?? selectedRecord
    if (!record) return
    const item = record.item
    if (!gateway) {
      setRetryNote(item.id, 'cancel unavailable: not connected to the daemon')
      return
    }
    if (subagentRetryable(item.status)) {
      setRetryNote(item.id, `cannot cancel: agent already ${item.status} — nothing to stop`)
      return
    }
    if (pendingCancels.current.has(item.id)) return
    pendingCancels.current.add(item.id)
    setRetryNote(item.id, 'cancel requested…')
    gateway.rpc<SubagentInterruptResponse>('subagent.interrupt', { task: item.name?.trim() || item.id })
      .then(response => {
        if (response?.found === false) {
          setRetryNote(item.id, 'cancel failed: the daemon no longer tracks that agent')
          return
        }
        setRetryNote(item.id, '✕ cancel accepted — the daemon was asked to stop the agent')
      })
      .catch((error: unknown) => {
        setRetryNote(item.id, `cancel unavailable: ${error instanceof Error && error.message ? error.message : 'request failed'}`)
      })
      .finally(() => {
        pendingCancels.current.delete(item.id)
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
    } else if (event.sequence === 'c') {
      // Scoped to the inspector: a stray press while browsing the list
      // must never stop an agent the user was only reading about.
      if (openId) cancelSelected()
      else return
    } else if (event.name === 'up') {
      if (openId) scrollRef.current?.scrollBy(-1)
      else moveSelection(-1)
    } else if (event.name === 'down') {
      if (openId) scrollRef.current?.scrollBy(1)
      else moveSelection(1)
    } else if (isPageUpKey(event)) {
      scrollRef.current?.scrollBy(-page)
    } else if (isPageDownKey(event)) {
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
          compactHeader={panelWidth < 56}
          history={history}
          liveAgents={liveAgents}
          now={now}
          onOpen={setOpenId}
          openRecord={openRecord}
          retryEnabled={Boolean(gateway)}
          retryNotes={retryNotes}
          scrollRef={scrollRef}
          selectedId={selectedRecord?.item.id ?? null}
          // Below a dozen rows the hairlines and vertical padding cost more
          // than the content they frame; a degenerate terminal keeps the
          // list, the empty state, and the footer keys instead.
          shortFrame={panelHeight < 12}
          t={t}
          variant="overlay"
        />
      </Box>
    </box>
  )
}
