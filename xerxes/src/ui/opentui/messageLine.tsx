// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { turnOutcomeLabel, type TurnOutcomeReason } from '../../types/turnOutcome.js'
// OpenTUI message renderer. One Msg becomes one flat transcript block.
// Assistant text renders through OpenTUI's native <markdown> (tables, code,
// emphasis). Tool-call trail lines stay compact like Grok's transcript: the
// call is always visible, while diagnostic and diff detail remains available
// when it carries information the one-line result cannot safely summarize.
import { computed } from 'nanostores'
import { useStore } from '@nanostores/react'
import { memo, type ReactNode, useEffect, useMemo, useRef, useState } from 'react'

import {
  $thinkingVisibility,
  thinkingRowExpanded,
  toggleThinkingRow
} from '../app/thinkingVisibilityStore.js'
import { patchOverlayState } from '../app/overlayStore.js'
import { $spawnHistory, spawnHistoryForSession } from '../app/spawnHistoryStore.js'
import { $toolRunVisibility, toggleToolRun, toolRunExpanded } from '../app/toolRunStore.js'
import { $toolStepVisibility, toggleToolStep, toolStepExpanded } from '../app/toolStepStore.js'
import { $uiDetailVisibility, $uiState } from '../app/uiStore.js'
import { useTurnSelector } from '../app/turnStore.js'
import { sectionMode } from '../domain/details.js'
import { GLYPH } from '../domain/nocturne.js'
import { VOICE } from '../domain/roles.js'
import { messageHasVisibleDetails, trailHasRenderableContent } from '../lib/liveProgress.js'
import { reconcileSpawnRoster, spawnRosterFromLine } from '../lib/toolStartDisplay.js'
import { subagentCardAccent, subagentCardModel } from '../lib/subagentCards.js'
import { fmtDuration, subagentElapsedSeconds } from '../lib/subagentElapsed.js'
import { groupToolRun, toolRunSpawnRoster, type ToolRunGroup } from '../lib/toolRun.js'
import { estimateTokensRough, fmtK, parseToolTrailResultLine, toolTrailParts } from '../lib/text.js'
import { splitStreamingRender, STREAMING_CHUNKS_EMPTY, type StreamingChunks } from '../lib/streamingMarkdown.js'
import type { Theme } from '../theme.js'
import type { Msg, SubagentProgress } from '../types.js'

import { PanelMessage } from './panelView.js'
import { Box, Span, Text } from './primitives.js'
import { getSyntaxStyle } from './syntax.js'

const ERROR_DETAIL_MAX_LINES = 16
const DIFF_DETAIL_MAX_LINES = 20
const TOOL_RESULT_MAX_LINES = 12
const TOOL_RESULT_COMPACT_CHARS = 180

const TABLE_OPTIONS = {
  borderStyle: 'rounded' as const,
  borders: true,
  cellPadding: 1,
  columnFitter: 'balanced' as const,
  outerBorder: true,
  widthMode: 'full' as const,
  wrapMode: 'word' as const
}

function Markdown({ content, t }: { content: string; t: Theme }) {
  return (
    <markdown
      conceal
      content={content}
      flexShrink={0}
      selectable
      syntaxStyle={getSyntaxStyle(t)}
      tableOptions={{ ...TABLE_OPTIONS, borderColor: t.color.border }}
    />
  )
}

const MarkdownChunk = memo(Markdown)

/**
 * Live-streaming assistant text. Re-parsing the whole growing buffer into one
 * native <markdown> element on every delta is O(total) per delta, so the
 * fence-aware chunker (lib/streamingMarkdown) freezes completed blocks into
 * memoized chunks that never re-parse; only the unstable tail does.
 *
 * Visual parity with a single whole-buffer <markdown>: a lone document
 * inserts one blank row between top-level blocks, so chunk elements are
 * interleaved with a one-row spacer and each chunk is stripped of its
 * trailing blank line by the chunker (OpenTUI preserves a trailing "\n\n"
 * after heading blocks, which would otherwise double the spacing).
 */
export function StreamingMarkdown({ text, t }: { text: string; t: Theme }) {
  const stateRef = useRef<StreamingChunks>(STREAMING_CHUNKS_EMPTY)
  const render = splitStreamingRender(text, stateRef.current)
  stateRef.current = render.state

  return (
    <>
      {render.chunks.map((chunk, index) => (
        <Box flexDirection="column" flexShrink={0} key={index}>
          <MarkdownChunk content={chunk} t={t} />
          <Box flexShrink={0} height={1} />
        </Box>
      ))}
      {render.tail ? <Markdown content={render.tail} t={t} /> : null}
    </>
  )
}

function UserMessage({ msg, t }: { msg: Msg; t: Theme }) {
  return (
    // One blank row above the band, none below: the next block adds its own
    // lead gap when the voice changes. This used to paint 2 rows above and 3
    // below every turn, and the estimator only ever predicted 2.
    // The canvas marks the user's own words with the prompt glyph on a filled
    // band, not with a left bar: `❯ …` is what you typed, and the band is
    // what makes turn starts findable by scrolling alone. The bar was the
    // pre-canvas treatment and put two markers on one row.
    <Box flexDirection="row" flexShrink={0} marginTop={1}>
      <Box
        backgroundColor={t.color.userBandBg}
        flexDirection="row"
        flexGrow={1}
        paddingLeft={1}
        paddingRight={1}
        paddingY={1}
      >
        <Box flexShrink={0}>
          <Text color={VOICE.user(t).bar}>{`${GLYPH.prompt} `}</Text>
        </Box>
        <Box flexDirection="column" flexGrow={1} minWidth={0}>
          <Text color={VOICE.user(t).body} wrap="wrap">
            {msg.text}
          </Text>
        </Box>
      </Box>
    </Box>
  )
}

/** Shared live/settled message layout, matching the design's author and body rows. */
export function AssistantFrame({children, t}: {children: ReactNode; leadGap?: boolean; t: Theme}) {
  return (
    <Box flexDirection="column" flexShrink={0} marginTop={1}>
      <Text bold color={VOICE.assistant(t).glyphColor}>✦ Xerxes</Text>
      <Box flexDirection="column" marginTop={1} paddingLeft={3} minWidth={0}>{children}</Box>
    </Box>
  )
}

function AssistantMessage({leadGap, msg, t}: {leadGap?: boolean; msg: Msg; rail?: TurnRail; t: Theme}) {
  return <AssistantFrame leadGap={leadGap} t={t}><Markdown content={msg.text} t={t} /></AssistantFrame>
}

export type TurnRail = 'end' | 'mid' | 'none'

/**
 * One column: the turn's rail, or the blank it replaced.
 *
 * A filled box rather than a `│` glyph — a glyph is a single row of text and
 * would mark only the first line of a multi-line answer, leaving the rest of
 * the turn unrailed. The fill spans whatever height the message takes.
 */
function RailGutter({ rail, t }: { rail?: TurnRail; t: Theme }) {
  const inTurn = rail === 'mid' || rail === 'end'

  return <Box backgroundColor={inTurn ? t.color.turnRail : undefined} flexShrink={0} width={1} />
}

/**
 * Closes a turn — the receipt for the rows above it.
 *
 * The canvas writes it `└ 6 tools · 11.4s · 18.2k tok · $0.11`. Two of those
 * four are on the wire today; per-turn token spend and cost are not, and
 * inventing them would make the receipt a decoration. So the row states what
 * it can and stays the same shape for the day the other two land.
 */
function TurnLedger({ seconds, tools, t, outcome }: { seconds: number; tools: number; t: Theme; outcome?: TurnOutcomeReason }) {
  const facts = [
    turnOutcomeLabel(outcome),
    tools > 0 ? `${tools} tool${tools === 1 ? '' : 's'}` : '',
    seconds >= 0.05 ? `${seconds.toFixed(1)}s` : ''
  ].filter(Boolean)

  return (
    <Box flexDirection="row" flexShrink={0}>
      <Box flexShrink={0} width={1}>
        <Text color={t.ds.caption}>{GLYPH.ledger}</Text>
      </Box>
      <Text color={t.ds.caption} wrap="truncate-end">
        {facts.map((fact, index) => (
          <Span key={fact}>
            {index ? <Span color={t.ds.rule}>{` ${GLYPH.separator} `}</Span> : '  '}
            <Span color={t.ds.caption}>{fact}</Span>
          </Span>
        ))}
      </Text>
    </Box>
  )
}

function SystemMessage({ msg, t }: { msg: Msg; t: Theme }) {
  return (
    <Box flexShrink={0} paddingLeft={2}>
      <Text color={VOICE.system(t).body} wrap="wrap">
        <Span color={VOICE.system(t).glyphColor}>{VOICE.system(t).glyph} </Span>
        {msg.text}
      </Text>
    </Box>
  )
}

function ToolResultMessage({ msg, t }: { msg: Msg; t: Theme }) {
  const text = msg.text.trim()
  const lines = text.split('\n')
  const first = lines.find(line => line.trim())?.trim() ?? ''
  const diagnostic = /^(?:error|exception|failed|failure|denied|fatal)(?:\b|:)/i.test(first)
  const diff = looksLikeDiff(lines)

  // Successful tool return values are already represented by their compact
  // chronological tool row. Rendering the protocol payload as a second block
  // is what produced the screenfuls of Args/Result JSON in the old view.
  if (!diagnostic && !diff) {
    return null
  }

  const maxLines = diff ? DIFF_DETAIL_MAX_LINES : diagnostic ? TOOL_RESULT_MAX_LINES : 1
  const shown = lines.slice(0, maxLines)
  let preview = shown.join('\n')
  if (!diff && !diagnostic && preview.length > TOOL_RESULT_COMPACT_CHARS) {
    preview = `${preview.slice(0, TOOL_RESULT_COMPACT_CHARS)}…`
  } else if (lines.length > shown.length) {
    preview += `\n… +${lines.length - shown.length} more line${lines.length - shown.length === 1 ? '' : 's'}`
  }

  return (
    <Box flexDirection="column" flexShrink={0} paddingLeft={3}>
      {preview ? (
        preview.split('\n').map((line, i) => (
          <Text
            color={toolDetailColor(line, diagnostic, t)}
            dimColor={!diagnostic && !isDiffLine(line)}
            key={i}
            wrap="wrap"
          >
            {i === 0 ? '→ ' : '  '}
            {line || ' '}
          </Text>
        ))
      ) : (
        <Text color={t.color.muted} dimColor>
          → (empty tool result)
        </Text>
      )}
    </Box>
  )
}

function isDiffLine(line: string): boolean {
  return /^(?:diff --git |index |@@ |--- |\+\+\+ |[-+](?![-+]))/.test(line)
}

function looksLikeDiff(lines: readonly string[]): boolean {
  return lines.some(isDiffLine)
}

function usefulToolDetail(detail: string, failed: boolean): { diagnostic: boolean; lines: string[]; overflow: number } {
  if (!detail) {
    return { diagnostic: failed, lines: [], overflow: 0 }
  }

  const all = detail.split('\n')
  if (failed) {
    const errorStart = all.findIndex(line => /^Error:\s*$/.test(line))
    const relevant = errorStart >= 0 ? all.slice(errorStart) : all
    const lines = relevant.slice(0, ERROR_DETAIL_MAX_LINES)

    return { diagnostic: true, lines, overflow: relevant.length - lines.length }
  }

  const diffStart = all.findIndex(isDiffLine)
  if (diffStart >= 0) {
    const relevant = all.slice(diffStart)
    const lines = relevant.slice(0, DIFF_DETAIL_MAX_LINES)

    return { diagnostic: false, lines, overflow: relevant.length - lines.length }
  }

  return { diagnostic: false, lines: [], overflow: 0 }
}

function toolDetailColor(line: string, diagnostic: boolean, t: Theme): string {
  if (/^\+(?!\+\+)/.test(line)) {
    return t.color.diffAddedWord
  }
  if (/^-(?!---)/.test(line)) {
    return t.color.diffRemovedWord
  }
  if (diagnostic && !/^Args:\s*$/.test(line)) {
    return t.color.error
  }

  return t.color.muted
}

/**
 * Quiet, read-only calls tint their outcome glyph faint instead of ok-green
 * (anatomy element ④: dim=read, green=ok, red=fail) so a wall of reads never
 * masquerades as a wall of wins. Classified from the call's leading verb —
 * the transcript only ever sees the display name ('Read File', 'Glob').
 */
const QUIET_TOOL_VERBS = new Set(['cat', 'find', 'glob', 'grep', 'head', 'list', 'ls', 'read', 'search', 'tail', 'view'])

export function isQuietToolName(name: string): boolean {
  const verb = name
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean)[0]

  return verb !== undefined && QUIET_TOOL_VERBS.has(verb)
}


/**
 * Braille's dot matrix is the closest one-cell terminal analogue to DSH's
 * animated 3×3 agent cube. Rotating the filled edge reads as motion without
 * changing row width or making the agent name jump.
 */
const FLEET_CUBE_FRAMES = ['⡿', '⣿', '⢿', '⣻', '⣽', '⣾', '⣷', '⣯'] as const
const FLEET_CUBE_TICK_MS = 120

type FleetRowState = 'done' | 'failed' | 'missing' | 'working'

const fleetRowState = (entry: SubagentProgress | undefined): FleetRowState => {
  if (!entry) return 'missing'
  if (entry.status === 'completed') return 'done'
  if (entry.status === 'error' || entry.status === 'failed' || entry.status === 'interrupted' || entry.status === 'timeout') return 'failed'
  return 'working'
}

/**
 * Live per-agent status roster under a Spawn Agents transcript row (DSH-style
 * fleet visibility): a rolling cyan cube while the agent works, a still green
 * cube when it completes, red when it fails or dies. Each row pairs the stable
 * agent name with its latest activity and elapsed time; missing agents retain a
 * muted dot-matrix cube so the requested roster never disappears.
 */
export function SpawnFleetRoster({
  archived = [],
  names,
  t
}: {
  archived?: readonly SubagentProgress[]
  names: readonly string[]
  t: Theme
}) {
  const live = useTurnSelector(state => state.subagents)
  const history = useStore($spawnHistory)
  const sessionId = useStore($uiState).sid
  const fleet = useMemo(() => {
    const byName = new Map<string, SubagentProgress>()
    const remember = (agent: SubagentProgress) => {
      for (const alias of [agent.name, agent.title]) {
        const key = alias?.trim().toLowerCase()
        if (key) byName.set(key, agent)
      }
    }
    // Oldest first, so the most recent snapshot wins for a reused title.
    for (const snapshot of [...spawnHistoryForSession(history, sessionId)].reverse()) {
      for (const agent of snapshot.subagents) remember(agent)
    }
    for (const agent of archived) remember(agent)
    // Live entries win: their status is fresher than any archived snapshot.
    for (const agent of live) remember(agent)
    return byName
  }, [archived, history, live, sessionId])
  const resolve = (name: string) => fleet.get(name.trim().toLowerCase())
  const rosterNames = reconcileSpawnRoster(names, fleet)
  const [frame, setFrame] = useState(0)
  const anyWorking = rosterNames.some(name => fleetRowState(resolve(name)) === 'working')

  useEffect(() => {
    if (!anyWorking) return
    const timer = setInterval(() => setFrame(value => value + 1), FLEET_CUBE_TICK_MS)
    timer.unref?.()
    return () => clearInterval(timer)
  }, [anyWorking])

  return (
    <Box flexDirection="column" flexShrink={0}>
      {rosterNames.map(name => {
        const entry = resolve(name)
        const stableName = entry?.name?.trim()
        const label = stableName && stableName !== 'subagent' ? stableName : entry?.title?.trim() || stableName || name
        const state = fleetRowState(entry)
        const glyph = state === 'working'
          ? FLEET_CUBE_FRAMES[frame % FLEET_CUBE_FRAMES.length]
          : state === 'missing'
            ? '⠿'
            : '⣿'
        const color =
          state === 'working'
            ? '#6487ff'
            : state === 'done'
              ? t.color.ok
              : state === 'failed'
                ? t.color.error
                : t.color.muted
        const elapsed = entry ? subagentElapsedSeconds(entry) : null
        const tokens = entry && entry.inputTokens !== undefined
          ? fmtK(entry.inputTokens + (entry.outputTokens ?? 0))
          : ''
        const timing = elapsed === null ? '' : ` [${fmtDuration(elapsed)}]`
        const tokenText = tokens ? ` · ${tokens} tok` : ''
        const activity = entry?.status ?? 'status unavailable'
        return (
          <Box key={name} backgroundColor={t.color.statusBg} borderSides={['left']} borderColor={color}
            onClick={entry ? () => patchOverlayState({ agents: true, agentsInspectId: entry.id }) : undefined}>
          <Text wrap="truncate-end">
            <Span color={color}>{`  ${glyph} `}</Span>
            <Span bold color={t.ds.title}>{label}</Span>
            <Span color={t.color.muted}>{`  · ${activity}${timing}${tokenText} · ${entry?.toolCount ?? 0} tools`}</Span>
          </Text>
          </Box>
        )
      })}
    </Box>
  )
}

function ToolStep({
  archived,
  saved,
  cols,
  line,
  msgKey,
  t
}: {
  archived?: readonly SubagentProgress[]
  saved?: Msg
  cols?: number
  line: string
  msgKey?: string
  t: Theme
}) {
  const parsed = parseToolTrailResultLine(line)

  const voice = VOICE.tool(t)

  const roster = spawnRosterFromLine(line)

  if (!parsed) {
    // In-flight / transient call line ("drafting …", a bare tool name). No
    // mark and a muted glyph, so "still running" reads differently from a
    // settled row at a glance.
    return (
      <Box flexDirection="column" flexShrink={0}>
        <Text color={voice.body} wrap="truncate-end">
          <Span color={t.color.muted}>{voice.glyph} </Span>
          {line}
        </Text>
        {roster ? (
          <>
            <SpawnFleetRoster archived={archived ?? []} names={roster.names} t={t} />
            {roster.extra > 0 ? (
              <Text color={t.color.muted} wrap="truncate-end">{`    … +${roster.extra} more in the agents panel (F6)`}</Text>
            ) : null}
          </>
        ) : null}
      </Box>
    )
  }

  const failed = parsed.mark === '✗'
  const markColor = failed ? t.color.error : t.color.ok
  const detail = usefulToolDetail(parsed.detail, failed)
  const { args, duration, name } = toolTrailParts(parsed.call)
  // Outcome glyph FIRST (anatomy element ④): faint for quiet read-only calls,
  // ok-green on success, error-red on failure — the verdict lands before the
  // eye reaches any words. The name stays lapis either way.
  const quiet = !failed && isQuietToolName(name)
  const outcomeColor = failed ? t.color.error : quiet ? t.color.muted : t.color.ok
  const mark = parsed.mark

  // Expandable detail: the step id is stable for the message's lifetime so
  // scrolling away and back does not collapse it. The record lookup goes
  // through the trail-line → tool-id map because the daemon's tool id is not
  // part of the rendered line.
  const stepId = msgKey ? `${msgKey}:${line}` : line
  const expanded = toolStepExpanded(useStore($toolStepVisibility), stepId)
  const record = useTurnSelector(state => {
    const savedId = saved?.toolLineToId?.[line]
    if (saved?.toolRecords) return savedId ? saved.toolRecords[savedId] : undefined
    const toolId = state.toolLineToId[line]
    return toolId ? state.toolRecords[toolId] : undefined
  })

  return (
    <Box flexDirection="column" flexShrink={0}>
      <Box flexShrink={0} onClick={() => toggleToolStep(stepId)}>
        <Text color={voice.body} wrap="truncate-end">
          <Span color={t.color.muted}>{expanded ? '▾' : '▸'} </Span>
          <Span color={outcomeColor}>{voice.glyph} </Span>
          <Span color={t.color.toolName}>{name}</Span>
          {args ? <Span color={t.ds.title}>{`  ${args}`}</Span> : null}
          {/* Paint the tick too. Rendering only '✗' left success visually
              identical to a call that is still running. */}
          {mark ? (
            <>
              {' '}
              <Span color={markColor} dimColor={!failed}>
                {mark}
              </Span>
            </>
          ) : null}

          {duration ? <Span color={t.ds.numeric}>{`  ${duration}`}</Span> : null}
        </Text>
      </Box>
      {detail.lines.map((d, i) => (
        <Text
          color={toolDetailColor(d, detail.diagnostic, t)}
          dimColor={!detail.diagnostic && !isDiffLine(d)}
          key={i}
          wrap="wrap"
        >
          {'  '}
          {d || ' '}
        </Text>
      ))}
      {expanded && record ? (
        <Box flexDirection="column" flexShrink={0} paddingLeft={2}>
          {record.reasoning ? (
            <Text color={t.color.thinking} wrap="wrap">
              {'  '}reasoning: {record.reasoning}
            </Text>
          ) : null}
          {record.args ? (
            <Text color={t.color.muted} wrap="wrap">
              {'  '}call: {record.args}
            </Text>
          ) : null}
          {record.result ? (
            <Box flexDirection="column" flexShrink={0}>
              <Text color={t.color.muted}>Output</Text>
              {record.result.split('\n').map((line, index) => <Text key={index} color={t.color.muted} wrap="wrap">{line || ' '}</Text>)}
            </Box>
          ) : null}
          {record.error ? (
            <Text color={t.color.error} wrap="wrap">
              {'  '}error: {record.error}
            </Text>
          ) : null}
        </Box>
      ) : null}
      {roster ? (
        <>
          <SpawnFleetRoster archived={archived ?? []} names={roster.names} t={t} />
          {roster.extra > 0 ? (
            <Text color={t.color.muted} wrap="truncate-end">{`    … +${roster.extra} more in the agents panel (F6)`}</Text>
          ) : null}
        </>
      ) : null}
      {detail.lines.length > 0 && detail.overflow > 0 ? (
        <Text color={t.color.muted} dimColor>
          {'  '}… +{detail.overflow} more line{detail.overflow === 1 ? '' : 's'}
        </Text>
      ) : null}
    </Box>
  )
}

interface DetailVisibility {
  subagents: boolean
  thinking: boolean
  tools: boolean
}

const detailVisibility = (snapshot: string): DetailVisibility => {
  const [thinking, tools, subagents] = snapshot.split(':')

  return { subagents: subagents === 'true', thinking: thinking === 'true', tools: tools === 'true' }
}

// Inline agent cards (mockups 02 / 03⑥) key off the same /details machinery
// as thinking and tools. `$uiDetailVisibility` still pins `subagents` to
// false from the era when spawn trees lived only in the agent panel, so the
// transcript resolves its own section mode here; useMainApp derives the same
// value from the same inputs when filtering rows and estimating heights.
const $subagentCardsVisible = computed($uiState, state =>
  sectionMode('subagents', state.detailsMode, state.sections, state.detailsModeCommandOverride) !== 'hidden'
)

// Stable per-message identity for thinking toggles when the caller has no
// row key (direct MessageLine consumers). Settled transcript rows should
// pass their virtual row key so the toggle survives virtualization; live
// stream segments pass their segment key so it survives per-delta Msg
// object replacement.
const fallbackRowIds = new WeakMap<Msg, string>()
let fallbackRowIdSeq = 0

const thinkingRowId = (msg: Msg, msgKey?: string): string => {
  if (msgKey) {
    return msgKey
  }

  let id = fallbackRowIds.get(msg)

  if (!id) {
    id = `thinking:${++fallbackRowIdSeq}`
    fallbackRowIds.set(msg, id)
  }

  return id
}

/**
 * Collapsed thinking header.
 *
 * Thinking is a single row by default: it is evidence, not content — one
 * line and the token count it cost. You expand it only if
 * you doubt the answer.
 *
 * The row leads with the disclosure triangle alone. It used to carry a ✻ as
 * well, which put two marks on a row whose whole point is to be one line, and
 * spent a glyph the vocabulary reserves for the live progress pill.
 */
export function thinkingHeaderLabel({
  durationSeconds,
  expanded
}: {
  durationSeconds?: number
  expanded: boolean
}): string {
  const arrow = expanded ? '▾' : '▸'
  const subject =
    typeof durationSeconds === 'number' && Number.isFinite(durationSeconds)
      ? `thought for ${fmtDuration(Math.max(0, durationSeconds))}`
      : 'thinking'

  return `${arrow} ${subject}`
}

function ThinkingBlock({ cols, msg, rowId, t }: { cols?: number; msg: Msg; rowId: string; t: Theme }) {
  const visibility = useStore($thinkingVisibility)
  const expanded = thinkingRowExpanded(visibility, rowId)
  const thinking = msg.thinking?.trim() ?? ''
  const tokens = msg.thinkingTokens && msg.thinkingTokens > 0 ? msg.thinkingTokens : estimateTokensRough(thinking)
  const tokenLabel = tokens > 0 ? `~${fmtK(tokens)} tok` : ''
  const label = thinkingHeaderLabel({ expanded })

  return (
    <Box flexDirection="column" flexShrink={0}>
      {/* Header carries the violet; no dimColor on top of it, or it drops
          below readable on terminals that dim aggressively. The expanded
          trace below stays muted — a long trace must not be violet. */}
      <Box flexShrink={0} onClick={() => toggleThinkingRow(rowId)}>
        <Text wrap="truncate-end">
          <Span color={t.ds.caption}>{label.slice(0, 1)}</Span>
          <Span color={t.color.thinking}>{label.slice(1)}</Span>

          {tokenLabel ? <Span color={t.ds.numeric}>{` ${tokenLabel}`}</Span> : null}
        </Text>
      </Box>
      {expanded
        ? thinking.split('\n').map((line, i) => (
            <Text color={t.color.muted} dimColor key={i} wrap="wrap">
              {'  '}
              {line || ' '}
            </Text>
          ))
        : null}
    </Box>
  )
}

function ToolRun({
  archived,
  saved,
  cols,
  group,
  runId,
  t
}: {
  archived: readonly SubagentProgress[]
  saved?: Msg
  cols?: number
  group: Extract<ToolRunGroup, { kind: 'run' }>
  runId: string
  t: Theme
}) {
  const visibility = useStore($toolRunVisibility)
  const expanded = toolRunExpanded(visibility, runId)
  const { duration, tally, total } = group.summary
  const roster = toolRunSpawnRoster(group.lines)

  if (expanded) {
    return (
      <Box flexDirection="column" flexShrink={0}>
        <Box flexShrink={0} onClick={() => toggleToolRun(runId)}>
          <Text color={t.color.muted} wrap="truncate-end">
            {'▾ '}
            <Span bold color={t.color.toolName}>
              {total} tools
            </Span>
          </Text>
        </Box>
        {group.lines.map((line, i) => (
          <ToolStep archived={archived} saved={saved} cols={cols} key={i} line={line} msgKey={runId} t={t} />
        ))}
      </Box>
    )
  }

  return (
    <Box flexDirection="column" flexShrink={0}>
      <Box flexShrink={0} onClick={() => toggleToolRun(runId)}>
        <Text color={t.color.muted} wrap="truncate-end">
          {'▸ '}
          <Span bold color={t.color.toolName}>
            {total} tools
          </Span>
          {duration > 0 ? <Span>{`  · ${duration.toFixed(1)}s`}</Span> : null}
          {tally ? <Span color={t.color.muted}>{`   ${tally}`}</Span> : null}
        </Text>
      </Box>
      {roster ? (
        <>
          <SpawnFleetRoster archived={archived} names={roster.names} t={t} />
          {roster.extra > 0 ? (
            <Text color={t.color.muted} wrap="truncate-end">{`    … +${roster.extra} more in the agents panel (F6)`}</Text>
          ) : null}
        </>
      ) : null}
    </Box>
  )
}

/**
 * Compact stateless agent cards on a trail row (mockups 02/03⑥). One ≤3-line
 * block per archived agent: tinted status dot, name, task summary, then a
 * violet latest-activity line while it works or its result sentence once
 * settled, with a dim token/tool budget when the wire reported one. No local
 * state and no clock — live elapsed time stays in the agent panel — so a
 * virtualized row can remount freely.
 */
function SubagentTrailCards({ items, t }: { items: readonly SubagentProgress[]; t: Theme }) {
  return (
    <Box flexDirection="column" flexShrink={0}>
      {items.map(item => {
        const model = subagentCardModel(item)
        const accent = subagentCardAccent(item.status, t)

        return (
          <Box flexDirection="column" flexShrink={0} key={item.id}>
            <Text wrap="truncate-end">
              <Span color={accent}>{'● '}</Span>
              <Span bold color={t.ds.title}>{model.headline}</Span>
              {model.summary ? <Span color={t.color.muted}>{` — ${model.summary}`}</Span> : null}
            </Text>
            {model.activity ? (
              <Text color={t.color.thinking} wrap="truncate-end">
                {`└ ${model.activity}`}
              </Text>
            ) : null}
            {model.result ? (
              <Text color={t.color.label} wrap="truncate-end">
                {model.result}
              </Text>
            ) : null}
            {model.budget ? (
              <Text color={t.color.muted} dimColor wrap="truncate-end">
                {model.budget}
              </Text>
            ) : null}
          </Box>
        )
      })}
    </Box>
  )
}

function ToolTrail({
  cols,
  leadGap,
  msg,
  msgKey,
  rail,
  subagentsVisible,
  t,
  visibility
}: {
  cols?: number
  leadGap?: boolean
  msg: Msg
  msgKey?: string
  rail?: TurnRail
  subagentsVisible?: boolean
  t: Theme
  visibility: DetailVisibility
}) {
  const thinking = msg.thinking?.trim()
  const tools = msg.tools ?? []
  const cards = subagentsVisible && msg.subagents?.length ? msg.subagents : null

  return (
    <Box flexDirection="row" flexShrink={0} marginTop={leadGap ? 1 : 0}>
      <RailGutter rail={rail} t={t} />
      <Box flexDirection="column" flexGrow={1} minWidth={0} paddingLeft={2}>
      {thinking && visibility.thinking ? (
        <ThinkingBlock cols={cols} msg={msg} rowId={thinkingRowId(msg, msgKey)} t={t} />
      ) : null}

      {visibility.tools
        ? groupToolRun(tools).map((group, i) =>
            group.kind === 'row' ? (
              <ToolStep archived={msg.subagents} saved={msg} cols={cols} key={i} line={group.line} msgKey={msgKey} t={t} />
            ) : (
              <ToolRun archived={msg.subagents ?? []} saved={msg} cols={cols} group={group} key={i} runId={`${thinkingRowId(msg, msgKey)}:run${i}`} t={t} />
            )
          )
        : null}

      {/* Archived spawn tree (mockup element ⑥): one compact card per agent,
          painted after the turn's tool rows. Height is mirrored by the
          estimator via lib/subagentCards.subagentCardRows. */}
      {cards ? <SubagentTrailCards items={cards} t={t} /> : null}
      </Box>
    </Box>
  )
}

/** Append the closing ledger when this row is the last of its turn. */
function withLedger(
  body: ReactNode,
  rail: TurnRail | undefined,
  tools: number,
  seconds: number,
  t: Theme
) {
  if (rail !== 'end') {
    return body
  }

  return (
    <Box flexDirection="column" flexShrink={0}>
      {body}
      <TurnLedger seconds={seconds} t={t} tools={tools} />
    </Box>
  )
}

function MessageLineView({
  cols,
  leadGap,
  msg,
  msgKey,
  rail,
  t,
  turnSeconds = 0,
  turnTools = 0
}: {
  cols?: number
  leadGap?: boolean
  msg: Msg
  msgKey?: string
  rail?: TurnRail
  t: Theme
  turnSeconds?: number
  turnTools?: number
}) {
  const visibility = detailVisibility(useStore($uiDetailVisibility))
  // Agent cards ride the same /details gate as the other trail sections, but
  // resolve it through $subagentCardsVisible (see above) and count as visible
  // detail on their own — a trail carrying only a spawn tree must paint.
  const subagentsVisible = useStore($subagentCardsVisible)
  const hasSubagentCards = subagentsVisible && Boolean(msg.subagents?.length)
  const hasVisibleDetails = hasSubagentCards || messageHasVisibleDetails(msg, visibility)

  if (msg.kind === 'outcome') return <TurnLedger outcome={msg.outcome} seconds={turnSeconds} tools={turnTools} t={t} />

  if (msg.kind === 'intro') {
    return null
  }

  if (msg.kind === 'panel' && msg.panelData) {
    return <PanelMessage cols={cols} panel={msg.panelData} t={t} />
  }

  if (msg.kind === 'trail') {
    // trailHasRenderableContent predates inline agent cards and cannot see
    // `subagents`; cards alone keep the row alive.
    if ((!trailHasRenderableContent(msg) && !hasSubagentCards) || !hasVisibleDetails) {
      return null
    }

    return withLedger(
      <ToolTrail
        cols={cols}
        leadGap={leadGap}
        msg={msg}
        msgKey={msgKey}
        rail={rail}
        subagentsVisible={subagentsVisible}
        t={t}
        visibility={visibility}
      />,
      rail,
      turnTools,
      turnSeconds,
      t
    )
  }

  if (msg.role === 'user') {
    return <UserMessage msg={msg} t={t} />
  }

  if (msg.role === 'assistant') {
    return withLedger(hasVisibleDetails ? (
      <Box flexDirection="column" flexShrink={0}>
        <ToolTrail
          cols={cols}
          leadGap={leadGap}
          msg={msg}
          msgKey={msgKey}
          rail={rail}
          subagentsVisible={subagentsVisible}
          t={t}
          visibility={visibility}
        />
        {/* The speaker label keeps its own breathing room after the trail. */}
        {msg.text ? <AssistantMessage msg={msg} rail={rail} t={t} /> : null}
      </Box>
    ) : (
      <AssistantMessage leadGap={leadGap} msg={msg} rail={rail} t={t} />
    ), rail, turnTools, turnSeconds, t)
  }

  if (msg.role === 'tool') {
    return <ToolResultMessage msg={msg} t={t} />
  }

  return hasVisibleDetails ? (
    <Box flexDirection="column" flexShrink={0}>
      <ToolTrail msg={msg} msgKey={msgKey} subagentsVisible={subagentsVisible} t={t} visibility={visibility} />
      {msg.text ? <SystemMessage msg={msg} t={t} /> : null}
    </Box>
  ) : (
    <SystemMessage msg={msg} t={t} />
  )
}

export const MessageLine = memo(MessageLineView)
