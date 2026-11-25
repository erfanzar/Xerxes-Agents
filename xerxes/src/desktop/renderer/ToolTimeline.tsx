// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Claude Code-style borderless Tool Group Timeline (`ToolTimelineGroup`):
 * - Plain inline muted disclosure (`Edited 1 file · Ran 1 command ›`) with zero
 *   card box, zero checkmark icon, and inline chevron right next to the text.
 * - Quiet running state (`XerxesThinkingMark` + `Working · <phrase> ›`) with a
 *   1–3 item rolling preview strip along a whisper-thin left hairline.
 * - Saved ArtifactCards & Swarm spawn summaries visible outside the fold when settled.
 * - Expanded whisper-thin vertical hairline rail connecting Reasoning, Tool,
 *   Subagent, Checkpoint, and Notice steps with a quiet metadata & bulk-action
 *   toolbar (`+N −M · steps · duration`, `Expand all diffs`, `Collapse`).
 */

import { memo, useMemo, useState, type ReactElement } from 'react'
import type { Block, ToolItem } from './types.js'
import { Icon } from './Icon.js'
import { ToolCallRow, classifyToolItem, toolHasFailed } from './Execution.js'
import { ToolTimelineDiffContext, fileToolOf, fileToolStats, type ToolTimelineDiffCommand } from './FileToolRow.js'
import { ArtifactCard, artifactOfTool, type ArtifactRef } from './ArtifactCard.js'
import { SwarmSpawnSummaryRow, isSwarmToolName, swarmPreviewOf, swarmSpawnSummaryOf } from './SwarmToolCard.js'
import { AgentsCard } from './AgentsCard.js'
import { OutputViewer } from './OutputViewer.js'
import { liveActivityPhrase, thinkingHeadingOf } from './activityPhrase.js'
import { commandGist } from './commandGist.js'
import { XerxesThinkingMark } from './XerxesSprites.js'

export interface ToolTimelineStats {
  readonly stepCount: number
  readonly toolCount: number
  readonly adds: number
  readonly dels: number
  readonly failures: number
  readonly totalDurationSec: number
  readonly durationLabel: string
  readonly headline: string
  readonly hasDiffTools: boolean
}

function toolLabelOf(verb: string | undefined): string {
  const spaced = (typeof verb === 'string' ? verb : '').replace(/[_-]+/g, ' ').trim()
  return spaced ? spaced[0]!.toUpperCase() + spaced.slice(1) : ''
}

function thinkingTailOf(text: string, max = 110): string {
  const clean = (text.split('\n').findLast(line => line.trim()) ?? '').trim()
  return clean.length > max ? `…${clean.slice(-(max - 1))}` : clean
}

function parseDurationSeconds(dur: string | undefined): number {
  if (typeof dur !== 'string') return 0
  const trimmed = dur.trim()
  if (!trimmed) return 0
  const msMatch = /^(\d+(?:\.\d+)?)\s*ms$/i.exec(trimmed)
  if (msMatch) return Number(msMatch[1]) / 1000
  const minSecMatch = /^(\d+)m(?:\s*(\d+(?:\.\d+)?)s)?$/i.exec(trimmed)
  if (minSecMatch) return Number(minSecMatch[1]) * 60 + (minSecMatch[2] ? Number(minSecMatch[2]) : 0)
  const secMatch = /^(\d+(?:\.\d+)?)s$/i.exec(trimmed)
  if (secMatch) return Number(secMatch[1])
  return 0
}

function formatTotalDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0.05) return ''
  if (seconds < 10) return `${seconds.toFixed(1)}s`
  if (seconds < 60) return `${Math.round(seconds)}s`
  const mins = Math.floor(seconds / 60)
  const rem = Math.round(seconds % 60)
  return rem > 0 ? `${mins}m ${rem}s` : `${mins}m`
}

type TimelineBucket = 'edit' | 'command' | 'read' | 'swarm' | 'spawn' | 'search' | 'web' | 'mcp' | 'skill' | 'agent' | 'other'

const BUCKET_PRIORITY: Record<TimelineBucket, number> = {
  edit: 0,
  command: 1,
  read: 2,
  swarm: 3,
  spawn: 4,
  search: 5,
  web: 6,
  mcp: 7,
  skill: 8,
  agent: 9,
  other: 10,
}

function bucketOfTool(item: ToolItem): { readonly bucket: TimelineBucket; readonly distinctKey?: string; readonly weight?: number } {
  const rawName = item.name || item.verb || ''
  if (isSwarmToolName(rawName)) return { bucket: 'swarm' }
  const file = fileToolOf(item)
  if (file) {
    if (file.kind === 'read' || file.kind === 'outline') return { bucket: 'read', distinctKey: file.path }
    return { bucket: 'edit', distinctKey: file.path }
  }
  const verbLower = (item.verb || '').replace(/[_-]+/g, ' ').trim().toLowerCase()
  const nameLower = (item.name || '').trim().toLowerCase()
  if (/^(edit|edit file|file edit|write|write file|append|append file|patch|apply patch|rewrite)$/.test(verbLower)) {
    return { bucket: 'edit', ...(item.path || item.arg ? { distinctKey: item.path || item.arg } : {}) }
  }
  if (/^(read|read file|outline)$/.test(verbLower)) {
    return { bucket: 'read', ...(item.path || item.arg ? { distinctKey: item.path || item.arg } : {}) }
  }
  if (/^(spawn[\s_-]?agents|agent|task[\s_-]?create|handoff)$/.test(verbLower) || /^(spawn_?agents|agent|task_?create|handoff)$/.test(nameLower)) {
    const batch = Number(/^(\d+) agents?\b/.exec(item.arg || '')?.[1] ?? 1)
    return { bucket: 'spawn', weight: Number.isFinite(batch) && batch > 0 ? batch : 1 }
  }
  const semantic = classifyToolItem(item)
  if (semantic.category === 'shell') return { bucket: 'command' }
  if (semantic.category === 'search') return { bucket: 'search' }
  if (semantic.category === 'web') return { bucket: 'web' }
  if (semantic.category === 'mcp') return { bucket: 'mcp' }
  if (semantic.category === 'skill') return { bucket: 'skill' }
  if (semantic.category === 'agent') return { bucket: 'agent' }
  return { bucket: 'other' }
}

function formatBucketPhrase(bucket: TimelineBucket, count: number): string {
  switch (bucket) {
    case 'edit': return `Edited ${count} file${count === 1 ? '' : 's'}`
    case 'command': return `Ran ${count} command${count === 1 ? '' : 's'}`
    case 'read': return `Read ${count} file${count === 1 ? '' : 's'}`
    case 'swarm': return `Immortals · ${count} action${count === 1 ? '' : 's'}`
    case 'spawn': return count === 1 ? 'Started an agent' : `Started ${count} agents`
    case 'search': return `Ran ${count} search${count === 1 ? '' : 'es'}`
    case 'web': return `${count} web lookup${count === 1 ? '' : 's'}`
    case 'mcp': return `${count} MCP call${count === 1 ? '' : 's'}`
    case 'skill': return `Used ${count} skill${count === 1 ? '' : 's'}`
    case 'agent': return `${count} agent step${count === 1 ? '' : 's'}`
    case 'other': return `Used ${count} tool${count === 1 ? '' : 's'}`
  }
}

function buildLiveHeadlinePhrase(blocks: readonly Block[], tools: readonly ToolItem[]): string {
  const workingTool = tools.findLast(item => item.state === 'working')
  if (workingTool && isSwarmToolName(workingTool.name)) {
    const preview = swarmPreviewOf(workingTool)
    return `${preview.verb} ${preview.target}`.trim()
  }
  if (workingTool) {
    const file = fileToolOf(workingTool)
    if (file?.kind === 'patch') {
      const base = file.path.split(/[\\/]/).pop() || file.path
      return `Patching ${base}`
    }
    if (file?.kind === 'outline') {
      const base = file.path.split(/[\\/]/).pop() || file.path
      return `Outlining ${base}`
    }
  }
  return liveActivityPhrase(blocks)
}

function buildSettledHeadline(blocks: readonly Block[], tools: readonly ToolItem[]): string {
  if (tools.length === 0) {
    const spawned = blocks.reduce((total, block) => total + (block.kind === 'agents' ? block.members.length : 0), 0)
    if (spawned > 0) return spawned === 1 ? 'Started 1 agent' : `Started ${spawned} agents`
    if (blocks.some(block => block.kind === 'thinking')) return 'Thought it through'
    return blocks.some(block => block.kind === 'notice' && block.error) ? 'Runtime error' : 'Runtime notice'
  }

  const buckets = new Map<TimelineBucket, Set<string>>()
  let spawnedByCalls = 0
  tools.forEach((item, idx) => {
    const { bucket, distinctKey, weight } = bucketOfTool(item)
    if (bucket === 'spawn') spawnedByCalls += weight ?? 1
    const set = buckets.get(bucket) ?? new Set<string>()
    set.add(distinctKey || `call:${idx}`)
    buckets.set(bucket, set)
  })

  const members = blocks.reduce((total, block) => total + (block.kind === 'agents' ? block.members.length : 0), 0)
  const ranked = [...buckets.entries()]
    .map(([bucket, keys]) => [bucket, bucket === 'spawn' ? Math.max(keys.size, members, spawnedByCalls) : keys.size] as const)
    .sort((a, b) => {
      if (a[0] === 'other' && b[0] !== 'other') return 1
      if (b[0] === 'other' && a[0] !== 'other') return -1
      if (b[1] !== a[1]) return b[1] - a[1]
      return BUCKET_PRIORITY[a[0]] - BUCKET_PRIORITY[b[0]]
    })

  const parts = ranked.slice(0, 3).map(([bucket, count]) => formatBucketPhrase(bucket, count))
  const overflow = ranked.slice(3).reduce((sum, [, count]) => sum + count, 0)
  if (overflow > 0) parts.push(`+${overflow} more`)
  return parts.join(' · ')
}

/** Pure aggregate summary for a group of activity blocks (reused in header & tests). */
export function summarizeToolTimeline(blocks: readonly Block[], running: boolean): ToolTimelineStats {
  const tools = blocks.flatMap(block => block.kind === 'tools' ? block.items : [])
  let stepCount = 0
  for (const block of blocks) {
    if (block.kind === 'tools') stepCount += block.items.length
    else stepCount += 1
  }

  let adds = 0
  let dels = 0
  let totalDurationSec = 0
  let hasDiffTools = false

  for (const item of tools) {
    const failed = toolHasFailed(item)
    const file = fileToolOf(item)
    if (file && (file.kind === 'edit' || file.kind === 'write' || file.kind === 'patch')) {
      hasDiffTools = true
    }
    if (!failed) {
      const stats = fileToolStats(item)
      if (stats) {
        adds += stats.adds
        dels += stats.dels
      }
    }
    totalDurationSec += parseDurationSeconds(item.dur)
  }

  const failures = tools.filter(toolHasFailed).length
    + blocks.filter(block => block.kind === 'notice' && block.error).length
    + blocks.reduce((total, block) => total + (block.kind === 'agents' ? block.members.filter(member => member.status === 'failed').length : 0), 0)

  const headline = running
    ? buildLiveHeadlinePhrase(blocks, tools)
    : buildSettledHeadline(blocks, tools)

  return {
    stepCount,
    toolCount: tools.length,
    adds,
    dels,
    failures,
    totalDurationSec,
    durationLabel: formatTotalDuration(totalDurationSec),
    headline,
    hasDiffTools,
  }
}

interface PreviewEntry {
  readonly key: string
  readonly state: 'working' | 'done' | 'failed'
  readonly category: string
  readonly icon: ReturnType<typeof classifyToolItem>['icon']
  readonly badge: string
  readonly target: string
  readonly metaPill?: string
  readonly metaTone?: 'neutral' | 'ok' | 'failed' | 'working' | 'accent'
  readonly dur?: string
}

function selectLivePreviewEntries(tools: readonly ToolItem[]): PreviewEntry[] {
  const workingTools = tools.filter(item => item.state === 'working')
  if (workingTools.length === 0) return []
  const settledTools = tools.filter(item => item.state !== 'working')

  // Show up to 2 active running tools plus the most recent completed step (1–3 total).
  const chosen: ToolItem[] = []
  const recentSettled = settledTools.at(-1)
  if (recentSettled && workingTools.length < 3) chosen.push(recentSettled)
  chosen.push(...workingTools.slice(-(3 - chosen.length)))

  return chosen.map(item => {
    const failed = toolHasFailed(item)
    const semantic = classifyToolItem(item, toolLabelOf(item.verb))
    const rawTarget = semantic.category === 'shell' && semantic.command
      ? `$ ${commandGist(semantic.command)}`
      : semantic.target
    return {
      key: item.id,
      state: failed ? 'failed' : item.state,
      category: semantic.category,
      icon: semantic.icon,
      badge: semantic.badge,
      target: rawTarget,
      ...(semantic.metaPill ? { metaPill: semantic.metaPill, metaTone: semantic.metaTone } : {}),
      ...(item.dur && item.dur !== '0.0s' && item.state !== 'working' ? { dur: item.dur } : {}),
    }
  })
}

const TimelineBlockStep = memo(function TimelineBlockStep({ block }: { readonly block: Block }): ReactElement | null {
  if (block.kind === 'thinking') {
    const heading = thinkingHeadingOf(block.text)
    const excerpt = heading || thinkingTailOf(block.text)
    const state = block.streaming ? 'working' : 'thinking'
    return <div className="tool-timeline__step" data-kind="thinking" data-state={state}>
      {state === 'working' && <span className="tool-timeline__node" aria-hidden="true" />}
      <div className="tool-timeline__content">
        <details className="thinkrow tool-timeline__think">
          <summary>
            <span className="frow__icon"><Icon name="brain" size={13} /></span>
            <span className="frow__label">Reasoning</span>
            {excerpt && <>
              <span className="frow__sep">·</span>
              <span className="frow__excerpt">{excerpt}{block.streaming ? ' …' : ''}</span>
            </>}
          </summary>
          <div className="thinkrow__body">{block.text}</div>
        </details>
      </div>
    </div>
  }

  if (block.kind === 'tools') {
    return <>
      {block.items.map(item => {
        const failed = toolHasFailed(item)
        const state = failed ? 'failed' : item.state
        return <div key={item.id} className="tool-timeline__step" data-kind="tool" data-state={state}>
          {(state === 'working' || state === 'failed') && <span className="tool-timeline__node" aria-hidden="true" />}
          <div className="tool-timeline__content">
            <ToolCallRow item={item} label={toolLabelOf(item.verb)} />
          </div>
        </div>
      })}
    </>
  }

  if (block.kind === 'agents') {
    const hasWorking = block.members.some(m => m.status === 'working' || m.status === 'running')
    const hasFailed = block.members.some(m => m.status === 'failed')
    const state = hasFailed ? 'failed' : hasWorking ? 'working' : 'done'
    return <div className="tool-timeline__step" data-kind="agents" data-state={state}>
      {(state === 'working' || state === 'failed') && <span className="tool-timeline__node" aria-hidden="true" />}
      <div className="tool-timeline__content">
        <AgentsCard members={block.members} />
      </div>
    </div>
  }

  if (block.kind === 'checkpoint') {
    return <div className="tool-timeline__step" data-kind="checkpoint" data-state="checkpoint">
      <div className="tool-timeline__content">
        <div className="frow frow--sys tool-timeline__checkpoint">
          <span className="frow__icon"><Icon name="clock" size={13} /></span>
          <span className="frow__label">Checkpoint</span>
          <span className="frow__sep">·</span>
          <span className="frow__excerpt">turn {block.turn} end · <span className="add">+{block.adds}</span> <span className="del">−{block.dels}</span> cumulative</span>
        </div>
      </div>
    </div>
  }

  if (block.kind === 'notice') {
    const state = block.error ? 'failed' : 'done'
    const long = block.text.length > 320 || block.text.split('\n').length > 4
    return <div className="tool-timeline__step" data-kind="notice" data-state={state}>
      {state === 'failed' && <span className="tool-timeline__node" aria-hidden="true" />}
      <div className="tool-timeline__content">
        {long ? (
          <details className="activity-notice">
            <summary><span>{block.text.split('\n')[0]!.slice(0, 160)}</span><Icon name="chevron" size={12} /></summary>
            <OutputViewer text={block.text} label="Activity details" />
          </details>
        ) : (
          <div className={`frow frow--sys${block.error ? ' frow--err' : ''}`}>
            {!block.error && <span className="frow__icon"><Icon name="info" size={13} /></span>}
            <span className="frow__excerpt frow__excerpt--wrap" title={block.text.length > 160 ? block.text : undefined}>{block.text}</span>
          </div>
        )}
      </div>
    </div>
  }

  return null
})

/** Deduplicate saved artifacts by ID, keeping only the latest version (or last occurrence on ties). */
export function dedupeArtifacts(tools: readonly ToolItem[]): ArtifactRef[] {
  const byId = new Map<string, ArtifactRef>()
  const order: string[] = []
  for (const item of tools) {
    const artifact = artifactOfTool(item)
    if (!artifact) continue
    const existing = byId.get(artifact.id)
    if (!existing) {
      byId.set(artifact.id, artifact)
      order.push(artifact.id)
    } else if (artifact.version >= existing.version) {
      const merged = existing.embed && artifact.embed === undefined
        ? { ...artifact, embed: true }
        : artifact
      byId.set(artifact.id, merged)
    }
  }
  return order.map(id => byId.get(id)!)
}

export const ToolTimelineGroup = memo(function ToolTimelineGroup({ blocks, active, turnActive, defaultOpen = false }: {
  readonly blocks: readonly Block[]
  readonly active: boolean
  readonly turnActive: boolean
  readonly defaultOpen?: boolean
}): ReactElement {
  const tools = useMemo(() => blocks.flatMap(block => block.kind === 'tools' ? block.items : []), [blocks])
  const liveAgents = useMemo(
    () => blocks.reduce((total, block) => total + (block.kind === 'agents'
      ? block.members.filter(member => member.status === 'working' || member.status === 'running').length
      : 0), 0),
    [blocks],
  )
  const running = active
    || tools.some(item => item.state === 'working')
    || blocks.some(block => block.kind === 'thinking' && block.streaming)
    || (turnActive && liveAgents > 0)

  const [expanded, setExpanded] = useState(defaultOpen)
  const [inspected, setInspected] = useState(defaultOpen)
  const [diffCmd, setDiffCmd] = useState<ToolTimelineDiffCommand>({ mode: 'default', rev: 0 })

  const stats = useMemo(() => summarizeToolTimeline(blocks, running), [blocks, running])
  const livePreview = useMemo(
    () => (running && !expanded ? selectLivePreviewEntries(tools) : []),
    [running, expanded, tools],
  )
  const artifacts = useMemo(() => dedupeArtifacts(tools), [tools])
  const swarmSpawns = useMemo(
    () => tools.flatMap(item => {
      const summary = swarmSpawnSummaryOf(item, toolHasFailed(item))
      return summary ? [summary] : []
    }),
    [tools],
  )

  const showBulkToolbar = stats.toolCount > 4
  const summaryTitle = useMemo(() => {
    const metaParts: string[] = []
    if (stats.adds > 0 || stats.dels > 0) {
      const diffParts: string[] = []
      if (stats.adds > 0) diffParts.push(`+${stats.adds}`)
      if (stats.dels > 0) diffParts.push(`−${stats.dels}`)
      metaParts.push(diffParts.join(' '))
    }
    if (stats.stepCount > 0) metaParts.push(`${stats.stepCount} ${stats.stepCount === 1 ? 'step' : 'steps'}`)
    if (stats.durationLabel) metaParts.push(stats.durationLabel)
    if (stats.failures > 0) metaParts.push(`${stats.failures} failed`)
    return metaParts.length > 0 ? `${stats.headline} · ${metaParts.join(' · ')}` : stats.headline
  }, [stats])

  const toggleAllDiffs = (): void => {
    setDiffCmd(prev => ({
      mode: prev.mode === 'expand' ? 'collapse' : 'expand',
      rev: prev.rev + 1,
    }))
  }

  const errorNoticeOnly = !running
    && blocks.length > 0
    && blocks.every((block): block is Extract<Block, { readonly kind: 'notice' }> => block.kind === 'notice' && block.error)

  if (errorNoticeOnly) {
    return <div className="tool-timeline-wrap tool-timeline tool-timeline--inline-error" data-failed="true">
      {blocks.map(notice => {
        const trimmed = notice.text.trim()
        const lines = trimmed.split('\n')
        const long = trimmed.length > 260 || lines.length > 3
        if (long) {
          const firstLine = (lines.find(line => line.trim()) ?? trimmed).slice(0, 180)
          return <details key={notice.id} className="tool-timeline__inline-error-fold">
            <summary className="tool-timeline__inline-error-summary">
              <span className="tool-timeline__inline-error-line">{firstLine}</span>
              <span className="tool-timeline__chev" aria-hidden="true"><Icon name="chevron" size={12} /></span>
            </summary>
            <div className="tool-timeline__inline-error-body">
              <OutputViewer text={notice.text} label="Error details" />
            </div>
          </details>
        }
        return <p key={notice.id} className="tool-timeline__inline-error" {...(active ? { role: 'alert' } : {})}>{trimmed || notice.text}</p>
      })}
    </div>
  }

  return <div
    className={`tool-timeline-wrap tool-timeline${running ? ' is-running' : ''}${stats.failures > 0 ? ' has-failures' : ''}`}
    data-running={running || undefined}
    data-expanded={expanded || undefined}
    data-failed={stats.failures > 0 || undefined}
  >
    <details
      className={`activity-group${running ? ' is-running' : ''}`}
      open={expanded}
      onToggle={event => {
        const isOpen = event.currentTarget.open
        setExpanded(isOpen)
        if (isOpen) setInspected(true)
      }}
    >
      <summary className="tool-timeline__header" title={summaryTitle}>
        {running ? (
          <>
            <span className="tool-timeline__mark-wrap" aria-hidden="true">
              <XerxesThinkingMark size={13} active={running} className="tool-timeline__mark" />
            </span>
            <span className="tool-timeline__headline tool-timeline__headline--live">
              <span className="tool-timeline__working">Working</span>
              <span className="tool-timeline__sep" aria-hidden="true">·</span>
              <span className="tool-timeline__phrase">{stats.headline}</span>
            </span>
          </>
        ) : (
          <span className="tool-timeline__headline">
            {stats.headline}
            {liveAgents > 0 && (
              <span className="tool-timeline__pending"> · {liveAgents} agent{liveAgents === 1 ? '' : 's'} still running</span>
            )}
          </span>
        )}
        {stats.failures > 0 && (
          <span className="tool-timeline__failed">
            <span className="tool-timeline__sep" aria-hidden="true">·</span>
            <span>{stats.failures} failed</span>
          </span>
        )}
        <span className="tool-timeline__chev" aria-hidden="true"><Icon name="chevron" size={12} /></span>
      </summary>

      <div className="activity-group__body tool-timeline__body">
        {(expanded || inspected) && (
          <ToolTimelineDiffContext.Provider value={diffCmd}>
            {(stats.stepCount > 0 || showBulkToolbar) && (
              <div className="tool-timeline__toolbar">
                <span className="tool-timeline__toolbar-meta">
                  {(stats.adds > 0 || stats.dels > 0) && (
                    <span className="tool-timeline__meta tool-timeline__meta--diff" aria-label={`${stats.adds} additions, ${stats.dels} deletions`}>
                      {stats.adds > 0 && <span className="add">+{stats.adds}</span>}
                      {stats.dels > 0 && <span className="del">−{stats.dels}</span>}
                    </span>
                  )}
                  {(stats.adds > 0 || stats.dels > 0) && (stats.stepCount > 0 || stats.durationLabel) && (
                    <span className="tool-timeline__meta-sep" aria-hidden="true">·</span>
                  )}
                  {stats.stepCount > 0 && (
                    <span className="tool-timeline__meta tool-timeline__meta--steps">
                      {stats.stepCount} {stats.stepCount === 1 ? 'step' : 'steps'}
                    </span>
                  )}
                  {stats.stepCount > 0 && stats.durationLabel && (
                    <span className="tool-timeline__meta-sep" aria-hidden="true">·</span>
                  )}
                  {stats.durationLabel && (
                    <span className="tool-timeline__meta tool-timeline__meta--dur">{stats.durationLabel}</span>
                  )}
                  {stats.failures > 0 && (
                    <>
                      <span className="tool-timeline__meta-sep" aria-hidden="true">·</span>
                      <span className="tool-timeline__meta tool-timeline__meta--failed">{stats.failures} failed</span>
                    </>
                  )}
                </span>
                {showBulkToolbar && (
                  <div className="tool-timeline__toolbar-actions">
                    {stats.hasDiffTools && (
                      <button type="button" className="tool-timeline__action" onClick={toggleAllDiffs}>
                        {diffCmd.mode === 'expand' ? 'Collapse diffs' : 'Expand all diffs'}
                      </button>
                    )}
                    <button type="button" className="tool-timeline__action" onClick={() => setExpanded(false)}>
                      Collapse
                    </button>
                  </div>
                )}
              </div>
            )}
            <div className="tool-timeline__rail">
              {blocks.map(block => <TimelineBlockStep key={block.id} block={block} />)}
            </div>
          </ToolTimelineDiffContext.Provider>
        )}
      </div>
    </details>

    {/* Live rolling preview right below the collapsed header while active */}
    {!expanded && running && livePreview.length > 0 && (
      <div className="tool-timeline__live-strip" aria-label="Live tool progress">
        {livePreview.map(entry => (
          <div key={entry.key} className="tool-timeline__live-row" data-state={entry.state} data-category={entry.category}>
            {(entry.state === 'working' || entry.state === 'failed') && (
              <span className="tool-timeline__live-dot" aria-hidden="true" />
            )}
            <span className="tool-timeline__live-icon" aria-hidden="true"><Icon name={entry.icon} size={12} /></span>
            <span className="tool-timeline__live-badge">{entry.badge}</span>
            {entry.target && <code className="tool-timeline__live-target" title={entry.target}>{entry.target}</code>}
            {entry.metaPill && <span className="tool-timeline__live-meta" data-tone={entry.metaTone ?? 'neutral'}>{entry.metaPill}</span>}
            {entry.dur && <span className="tool-timeline__live-dur">{entry.dur}</span>}
          </div>
        ))}
      </div>
    )}

    {/* Keep saved Artifacts and settled Swarm spawn summaries visible outside the fold */}
    {!expanded && (artifacts.length > 0 || (!running && swarmSpawns.length > 0)) && (
      <div className="tool-timeline__outside">
        {!running && swarmSpawns.map(summary => <SwarmSpawnSummaryRow key={summary.id} summary={summary} />)}
        {artifacts.map(artifact => <ArtifactCard key={artifact.id} artifact={artifact} />)}
      </div>
    )}
  </div>
})
