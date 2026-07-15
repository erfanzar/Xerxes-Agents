// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import type { KeyEvent, ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { type MutableRefObject, useEffect, useMemo, useRef, useState } from 'react'

import type { SpawnSnapshot } from '../app/spawnHistoryStore.js'
export { AGENT_SIDEBAR_BREAKPOINT, shouldShowAgentSidebar } from '../domain/agentPanelLayout.js'
import { subagentElapsedSeconds } from '../lib/subagentElapsed.js'
import { fmtDuration, fmtTokens } from '../lib/subagentTree.js'
import type { Theme } from '../theme.js'
import type { SubagentProgress } from '../types.js'

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
  t: Theme
  variant: 'overlay' | 'sidebar'
}

interface AgentPanelOverlayProps extends Omit<AgentPanelProps, 'variant'> {
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

function metricLine(item: SubagentProgress, now: number): string {
  const toolCount = Math.max(item.toolCount, item.tools.length, item.outputTail?.length ?? 0)
  const parts = [`${toolCount} tool${toolCount === 1 ? '' : 's'}`]
  const input = item.inputTokens ?? 0
  const output = item.outputTokens ?? 0
  const tokens = input + output
  const elapsed = subagentElapsedSeconds(item, now)

  if (tokens > 0) parts.push(`${fmtTokens(tokens)} tok`)
  if ((item.reasoningTokens ?? 0) > 0) parts.push(`${fmtTokens(item.reasoningTokens!)} reasoning`)
  if (elapsed != null) parts.push(fmtDuration(elapsed))
  if ((item.apiCalls ?? 0) > 0) parts.push(`${item.apiCalls} API`)

  return parts.join(' · ')
}

function AgentCard({ now, record, t }: { now: number; record: AgentPanelRecord; t: Theme }) {
  const { item } = record
  const status = statusPresentation(item.status, t)
  const role = item.agentType?.trim() || 'agent'
  const model = item.model?.trim()
  const rules = item.rules?.length ? item.rules.join(', ') : 'inherited defaults'
  const toolsets = item.toolsets?.length ? item.toolsets.join(', ') : 'runtime policy'
  const read = item.filesRead ?? []
  const written = item.filesWritten ?? []
  const filePreview = [...written.map(path => `+${basename(path)}`), ...read.map(path => basename(path))].slice(0, 4)
  const depth = Math.min(4, Math.max(0, item.depth))

  return (
    <Box
      backgroundColor={t.color.completionCurrentBg}
      flexDirection="row"
      flexShrink={0}
      marginBottom={1}
      marginLeft={depth}
      paddingRight={1}
      paddingY={1}
    >
      <Box backgroundColor={status.color} flexShrink={0} width={1} />
      <Box flexDirection="column" flexGrow={1} flexShrink={1} paddingLeft={1}>
        <Text color={t.color.text} wrap="truncate-end">
          <Span color={status.color}>{status.glyph} </Span>
          <Span bold color={t.color.text}>
            {record.title}
          </Span>
          <Span color={t.color.muted}> · {item.status}</Span>
        </Text>
        <Text color={t.color.muted} wrap="truncate-end">
          ↳ {record.creatorTitle} · {role}
          {model ? ` · ${model}` : ''}
        </Text>
        <Text color={TERMINAL_STATUSES.has(item.status) ? t.color.text : t.color.muted} wrap="wrap">
          {compactLine(activitySummary(item), 180)}
        </Text>
        <Text color={t.color.accent} wrap="truncate-end">
          {metricLine(item, now)}
          {record.childCount ? ` · ${record.childCount} child${record.childCount === 1 ? '' : 'ren'}` : ''}
        </Text>
        <Text color={t.color.muted} wrap="truncate-end">
          policy · {compactLine(rules, 54)}
        </Text>
        <Text color={t.color.muted} wrap="truncate-end">
          access · {compactLine(toolsets, 54)}
        </Text>
        {read.length || written.length ? (
          <Text color={t.color.muted} wrap="truncate-end">
            files · {written.length} wrote · {read.length} read
            {filePreview.length ? ` · ${filePreview.join(', ')}` : ''}
          </Text>
        ) : null}
        {record.archived && record.snapshotLabel ? (
          <Text color={t.color.muted} dimColor wrap="truncate-end">
            history · {record.snapshotLabel}
          </Text>
        ) : null}
      </Box>
    </Box>
  )
}

function AgentPanelBody({
  history,
  liveAgents,
  scrollRef,
  t,
  variant
}: AgentPanelProps & {
  scrollRef?: MutableRefObject<ScrollBoxRenderable | null>
}) {
  const records = useMemo(() => collectAgentPanelRecords(liveAgents, history), [history, liveAgents])
  const activeCount = records.filter(
    record => record.item.status === 'running' || record.item.status === 'queued'
  ).length
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!activeCount) return
    const timer = setInterval(() => setNow(Date.now()), 500)
    timer.unref?.()

    return () => clearInterval(timer)
  }, [activeCount])

  return (
    <Box
      backgroundColor={t.color.completionBg}
      borderColor={variant === 'overlay' ? t.color.border : undefined}
      borderStyle={variant === 'overlay' ? 'round' : undefined}
      flexDirection="column"
      flexGrow={variant === 'overlay' ? 1 : undefined}
      flexShrink={0}
      height={variant === 'sidebar' ? '100%' : undefined}
      minHeight={0}
      paddingX={variant === 'sidebar' ? 2 : 1}
      paddingY={1}
      width="100%"
    >
      <Box flexDirection="row" flexShrink={0} justifyContent="space-between" marginBottom={1}>
        <Text bold color={t.color.text}>
          <Span color={t.color.accent}>◆ </Span>
          Agents
        </Text>
        <Text color={activeCount ? t.color.accent : t.color.muted}>
          {activeCount ? `${activeCount} live` : `${records.length} done`}
        </Text>
      </Box>
      <scrollbox ref={scrollRef} style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }} viewportCulling>
        <Box flexDirection="column" flexShrink={0}>
          {records.length ? (
            records.map(record => (
              <AgentCard
                key={`${record.archived ? 'past' : 'live'}:${record.item.id}`}
                now={now}
                record={record}
                t={t}
              />
            ))
          ) : (
            <Box alignItems="center" flexDirection="column" flexGrow={1} justifyContent="center" minHeight={5}>
              <Text color={t.color.muted}>No agents yet</Text>
              <Text color={t.color.muted} dimColor>
                Delegated work appears here.
              </Text>
            </Box>
          )}
        </Box>
      </scrollbox>
      <Text color={t.color.muted} dimColor>
        {variant === 'overlay' ? '↑↓ scroll · PgUp/PgDn · F6/Esc close' : 'F6 expand · /agents'}
      </Text>
    </Box>
  )
}

export function AgentPanel(props: Omit<AgentPanelProps, 'variant'>) {
  if (!collectAgentPanelRecords(props.liveAgents, props.history).length) return null

  return <AgentPanelBody {...props} variant="sidebar" />
}

const consumeKey = (event: KeyEvent) => {
  event.preventDefault()
  event.stopPropagation()
}

export function AgentPanelHotkey({
  disabled,
  open,
  onToggle
}: {
  disabled: boolean
  open: boolean
  onToggle: (open: boolean) => void
}) {
  useKeyboard(event => {
    if (disabled || event.name !== 'f6') return
    onToggle(!open)
    consumeKey(event)
  })

  return null
}

export function AgentPanelOverlay({ history, liveAgents, onClose, t }: AgentPanelOverlayProps) {
  const scrollRef = useRef<ScrollBoxRenderable | null>(null)
  const { height, width } = useTerminalDimensions()
  const page = Math.max(4, height - 10)
  const panelHeight = Math.max(1, height - 2)
  const panelWidth = Math.max(1, Math.min(96, width - 2))

  useKeyboard(event => {
    if (event.name === 'escape' || event.name === 'f6' || event.sequence === 'q') {
      onClose()
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
      <Box flexDirection="column" height={panelHeight} maxWidth={96} minWidth={panelWidth} width={panelWidth}>
        <AgentPanelBody history={history} liveAgents={liveAgents} scrollRef={scrollRef} t={t} variant="overlay" />
      </Box>
    </box>
  )
}
