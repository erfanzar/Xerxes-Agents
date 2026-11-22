// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Semantic cards for Swarm orchestration tools (`swarm`, `post_message`,
 * `read_board`, `swarm_wait`, `swarm_overview`), replacing raw JSON tool rows
 * with scannable Simurgh-sprited Antigravity-grade action cards and one-click
 * navigation to the Swarm Board.
 */

import { Fragment, memo, useState, useSyncExternalStore, type ReactElement } from 'react'
import type { ToolItem } from './types.js'
import { Icon } from './Icon.js'
import { CopyButton } from './CopyButton.js'
import { useDesktopNavigation } from './DesktopPanels.js'
import { parseArgs } from './blocks.js'
import { store } from './store.js'
import type { SwarmMemberSnapshot } from './swarmState.js'
import { SimurghSprite, resolveSimurghRole, type SimurghSpriteRole, type SimurghSpriteState } from './XerxesSprites.js'

const EMPTY_ROSTER: readonly SwarmMemberSnapshot[] = []
const selectSwarmMembers = (): readonly SwarmMemberSnapshot[] =>
  store.getSnapshot().swarm?.members ?? EMPTY_ROSTER

const SWARM_TOOL_NAMES = new Set([
  'swarm',
  'post_message',
  'read_board',
  'swarm_wait',
  'swarm_overview',
  'postmessage',
  'readboard',
  'swarmwait',
  'swarmoverview',
])

function canonicalSwarmToolName(rawName: string | undefined): string {
  const safe = typeof rawName === 'string' ? rawName : ''
  const tail = (safe.split(/[.:/]/).pop() ?? safe)
    .trim()
    .replace(/Tool$/i, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[\s-]+/g, '_')
    .toLowerCase()
  if (tail === 'postmessage') return 'post_message'
  if (tail === 'readboard') return 'read_board'
  if (tail === 'swarmwait') return 'swarm_wait'
  if (tail === 'swarmoverview') return 'swarm_overview'
  return tail
}

/** True when a tool call belongs to the native Swarm tool suite. */
export function isSwarmToolName(name: string | undefined): boolean {
  if (!name) return false
  const normalized = canonicalSwarmToolName(name)
  return SWARM_TOOL_NAMES.has(normalized)
}

function parseJsonRecord(text: string | undefined): Record<string, unknown> | null {
  if (typeof text !== 'string' || !text.trim()) return null
  try {
    const parsed: unknown = JSON.parse(text)
    return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null
  } catch {
    return null
  }
}

function str(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function clipLine(text: string, max = 140): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  if (!flat) return ''
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat
}

function workspaceBadgeOf(mode: string): { readonly label: string; readonly tone: 'share' | 'branch' | 'inherit' } | null {
  if (mode === 'share') return { label: 'share · worktree', tone: 'share' }
  if (mode === 'branch') return { label: 'branch · clone', tone: 'branch' }
  if (mode === 'inherit') return { label: 'inherit', tone: 'inherit' }
  return null
}

function compactModel(raw: string): string {
  const trimmed = raw.trim()
  if (!trimmed) return ''
  const tail = trimmed.split('/').pop() ?? trimmed
  return tail.length > 32 ? `${tail.slice(0, 31)}…` : tail
}

/** Highlight `@mentions`, `#channels`, `#seq` refs, and `[done]` tags inline. */
function renderSwarmSnippet(text: string, max = 180): ReactElement {
  const clipped = clipLine(text, max)
  const tokenRe = /(@[a-zA-Z0-9][a-zA-Z0-9_-]*|\[done\]|#[a-zA-Z0-9_-]+)/g
  const parts: ReactElement[] = []
  let lastIndex = 0
  let key = 0
  for (const match of clipped.matchAll(tokenRe)) {
    const index = match.index ?? 0
    if (index > lastIndex) {
      parts.push(<Fragment key={key++}>{clipped.slice(lastIndex, index)}</Fragment>)
    }
    const token = match[0]!
    if (token.startsWith('@')) {
      parts.push(<span key={key++} className="swarm-tool__mention" data-all={token.toLowerCase() === '@all' || undefined}>{token}</span>)
    } else if (token.toLowerCase() === '[done]') {
      parts.push(<span key={key++} className="swarm-tool__done-tag">[done]</span>)
    } else if (/^#\d+$/.test(token)) {
      parts.push(<span key={key++} className="swarm-tool__seq-ref">{token}</span>)
    } else {
      parts.push(<span key={key++} className="swarm-tool__channel-ref">{token}</span>)
    }
    lastIndex = index + token.length
  }
  if (lastIndex < clipped.length) {
    parts.push(<Fragment key={key++}>{clipped.slice(lastIndex)}</Fragment>)
  }
  return <>{parts}</>
}

export interface ParsedAddedMember {
  readonly name: string
  readonly role?: string
  readonly model?: string
  readonly workspace?: string
  readonly conversationId?: string
}

export function extractAddedMembers(
  args: Record<string, unknown>,
  res: Record<string, unknown> | null,
  rosterMembers: readonly SwarmMemberSnapshot[] = EMPTY_ROSTER,
): {
  readonly members: readonly ParsedAddedMember[]
  readonly displayTarget: string
  readonly primaryName: string
  readonly primaryRole?: string
  readonly model: string
  readonly workspace: string
  readonly brief: string
  readonly fork: string
  readonly briefSeq: number | null
} {
  const rosterByName = new Map(rosterMembers.map(member => [member.name, member] as const))
  const rawAdded = Array.isArray(res?.added) ? res.added : []
  const members: ParsedAddedMember[] = []
  for (const entry of rawAdded) {
    if (!entry || typeof entry !== 'object') continue
    const rec = entry as Record<string, unknown>
    const name = str(rec.name)
    if (!name) continue
    const roster = rosterByName.get(name)
    const role = str(rec.role) || roster?.role || undefined
    const model = str(rec.model_display_name) || str(rec.model) || roster?.model_display_name || roster?.model || undefined
    const workspace = str(rec.workspace) || str(rec.workspace_mode) || roster?.workspace || undefined
    const conversationId = str(rec.conversation_id) || roster?.conversation_id || undefined
    members.push({
      name,
      ...(role ? { role } : {}),
      ...(model ? { model } : {}),
      ...(workspace ? { workspace } : {}),
      ...(conversationId ? { conversationId } : {}),
    })
  }

  // Antigravity ACP returns a singular `{ status: "added", name, model, model_display_name, conversation_id }` object.
  if (members.length === 0 && res && str(res.status).toLowerCase() !== 'error') {
    const singular =
      res.member && typeof res.member === 'object' && !Array.isArray(res.member)
        ? (res.member as Record<string, unknown>)
        : res
    const name = str(singular.name)
    if (name) {
      const roster = rosterByName.get(name)
      const role = str(singular.role) || roster?.role || undefined
      const model =
        str(singular.model_display_name)
        || str(singular.model)
        || roster?.model_display_name
        || roster?.model
        || undefined
      const workspace =
        str(singular.workspace)
        || str(singular.workspace_mode)
        || roster?.workspace
        || undefined
      const conversationId = str(singular.conversation_id) || roster?.conversation_id || undefined
      members.push({
        name,
        ...(role ? { role } : {}),
        ...(model ? { model } : {}),
        ...(workspace ? { workspace } : {}),
        ...(conversationId ? { conversationId } : {}),
      })
    }
  }

  const baseName = str(args.name)
  const tasks = Array.isArray(args.tasks) ? args.tasks.filter((t): t is string => typeof t === 'string' && t.trim().length > 0) : []
  const countArg = num(args.n)
  const expectedGroupCount = tasks.length > 0 ? tasks.length : (countArg && countArg > 1 ? countArg : 0)

  // When agy acp leaves item.output empty, recover member metadata from the live swarm roster.
  if (members.length === 0 && baseName && rosterMembers.length > 0) {
    const matched = expectedGroupCount > 1
      ? rosterMembers.filter(m => m.name === baseName || m.name.startsWith(`${baseName}-`) || m.group === baseName)
      : (() => {
          const exact = rosterByName.get(baseName)
          return exact ? [exact] : []
        })()
    for (const roster of matched) {
      const model = roster.model_display_name || roster.model || undefined
      members.push({
        name: roster.name,
        role: roster.role,
        ...(model ? { model } : {}),
        ...(roster.workspace ? { workspace: roster.workspace } : {}),
        ...(roster.conversation_id ? { conversationId: roster.conversation_id } : {}),
      })
    }
  }

  let displayTarget = baseName || 'worker'
  if (members.length > 1) {
    displayTarget = baseName ? `${baseName}-1..${members.length}` : `${members.length} workers`
  } else if (members.length === 1) {
    displayTarget = members[0]!.name
  } else if (expectedGroupCount > 1 && baseName) {
    displayTarget = `${baseName}-1..${expectedGroupCount}`
  }

  const primaryName = members[0]?.name ?? baseName ?? 'worker'
  const primaryRoster = rosterByName.get(primaryName)
  const primaryRole = members[0]?.role ?? primaryRoster?.role
  const model =
    members[0]?.model
    || str(args.model)
    || primaryRoster?.model_display_name
    || primaryRoster?.model
    || ''
  const workspace = members[0]?.workspace || str(args.workspace) || primaryRoster?.workspace || ''
  const brief = str(args.brief) || tasks[0] || ''
  const fork = str(args.fork)
  const briefSeqs = Array.isArray(res?.brief_seqs) ? res.brief_seqs.filter((s): s is number => typeof s === 'number') : []
  const briefSeq = briefSeqs[0] ?? num(res?.brief_seq) ?? null

  return {
    members,
    displayTarget,
    primaryName,
    ...(primaryRole ? { primaryRole } : {}),
    model,
    workspace,
    brief,
    fork,
    briefSeq,
  }
}

export interface SwarmSpawnSummary {
  readonly id: string
  readonly displayTarget: string
  readonly role: SimurghSpriteRole
  readonly model: string
  readonly workspace: string
  readonly briefPreview: string
  readonly count: number
  readonly state: ToolItem['state']
  readonly failed: boolean
}

/** Extract a concise spawn summary for collapsed tool groups to keep visible outside the fold. */
export function swarmSpawnSummaryOf(
  item: ToolItem,
  failed = item.state === 'failed',
  rosterMembers: readonly SwarmMemberSnapshot[] = selectSwarmMembers(),
): SwarmSpawnSummary | null {
  if (canonicalSwarmToolName(item.name) !== 'swarm') return null
  const args = parseArgs(item.input)
  if (str(args.action).toLowerCase() !== 'add') return null
  const res = parseJsonRecord(item.output)
  const info = extractAddedMembers(args, res, rosterMembers)
  return {
    id: item.id,
    displayTarget: info.displayTarget,
    role: resolveSimurghRole(info.primaryName, info.primaryRole),
    model: compactModel(info.model),
    workspace: info.workspace,
    briefPreview: clipLine(info.brief, 110),
    count: Math.max(1, info.members.length),
    state: item.state,
    failed,
  }
}

/** Compact one-line summary for live rolling preview strips in ToolTimelineGroup. */
export function swarmPreviewOf(
  item: ToolItem,
  rosterMembers: readonly SwarmMemberSnapshot[] = selectSwarmMembers(),
): { readonly verb: string; readonly target: string } {
  const tool = canonicalSwarmToolName(item.name)
  const args = parseArgs(item.input)
  const res = parseJsonRecord(item.output)
  if (tool === 'swarm') {
    const action = str(args.action).toLowerCase()
    if (action === 'add') {
      const info = extractAddedMembers(args, res, rosterMembers)
      return { verb: item.state === 'working' ? 'Spawning' : 'Spawned', target: info.displayTarget }
    }
    if (action === 'model') {
      const name = str(res?.name) || str(args.name) || 'coordinator'
      const model = compactModel(str(res?.model_display_name) || str(res?.model) || str(args.model))
      return { verb: 'Switched', target: model ? `@${name} → ${model}` : `@${name}` }
    }
    if (action === 'kill') {
      const name = str(args.name) || (Array.isArray(res?.killed) && typeof res.killed[0] === 'string' ? res.killed[0] : 'worker')
      return { verb: 'Retired', target: `@${name}` }
    }
    if (action === 'title') {
      const title = str(res?.title) || str(args.title)
      return { verb: 'Named Immortals', target: title ? `"${title}"` : 'Immortals' }
    }
    if (action === 'status') return { verb: 'Immortals', target: 'Checked roster' }
    if (action === 'models') return { verb: 'Immortals', target: 'Listed models' }
    if (action === 'channel') return { verb: 'Immortals', target: str(args.channel) ? `#${str(args.channel)}` : 'Channels' }
    return { verb: 'Immortals', target: action || 'action' }
  }
  if (tool === 'post_message') {
    const text = str(args.text) || str((res?.message as Record<string, unknown> | undefined)?.text)
    return { verb: 'Posted', target: clipLine(text, 64) || 'board message' }
  }
  if (tool === 'read_board') {
    if (args.status === true) return { verb: 'Immortals', target: 'Roster status' }
    const since = num(args.since)
    return { verb: 'Read board', target: since !== null ? `after #${since}` : 'latest messages' }
  }
  if (tool === 'swarm_wait') {
    const wake = num(res?.wake_seq)
    if (wake !== null) return { verb: 'Woken', target: `by #${wake}` }
    const since = num(args.since)
    return { verb: 'Waiting', target: since !== null ? `board after #${since}` : 'on board' }
  }
  if (tool === 'swarm_overview') {
    return { verb: 'Overview', target: clipLine(str(args.summary) || 'Updated live overview', 64) }
  }
  return { verb: 'Immortals', target: item.arg || item.name }
}

/** Compact Swarm spawn strip shown outside a collapsed settled tool group. */
export const SwarmSpawnSummaryRow = memo(function SwarmSpawnSummaryRow({ summary }: { readonly summary: SwarmSpawnSummary }): ReactElement {
  const open = useDesktopNavigation()
  const ws = workspaceBadgeOf(summary.workspace)
  const spriteState: SimurghSpriteState = summary.failed ? 'error' : summary.state === 'working' ? 'working' : 'waiting'
  return <div className="swarm-spawn-summary" data-state={summary.failed ? 'failed' : summary.state}>
    <span className="swarm-spawn-summary__sprite">
      <SimurghSprite role={summary.role} state={spriteState} size={18} animate={summary.state === 'working'} />
    </span>
    <span className="swarm-spawn-summary__main">
      <strong className="swarm-spawn-summary__title">
        {summary.failed ? 'Failed to spawn' : summary.state === 'working' ? 'Spawning' : 'Spawned'} {summary.displayTarget}
      </strong>
      {summary.model && <span className="swarm-tool__pill swarm-tool__pill--model">{summary.model}</span>}
      {ws && <span className="swarm-tool__pill swarm-tool__pill--ws" data-ws={ws.tone}>{ws.label}</span>}
      {summary.briefPreview && <span className="swarm-spawn-summary__brief">{summary.briefPreview}</span>}
    </span>
    <button type="button" className="swarm-tool__open" onClick={() => open('swarm')}>Open Immortals →</button>
  </div>
})

export const SwarmToolCard = memo(function SwarmToolCard({
  item,
  failed: failedProp,
  defaultExpanded = false,
  swarmMembers,
}: {
  readonly item: ToolItem
  readonly failed?: boolean
  readonly defaultExpanded?: boolean
  readonly swarmMembers?: readonly SwarmMemberSnapshot[]
}): ReactElement {
  const open = useDesktopNavigation()
  const liveRoster = useSyncExternalStore(store.subscribe, selectSwarmMembers, selectSwarmMembers)
  const roster = swarmMembers ?? liveRoster
  const [detailsOpen, setDetailsOpen] = useState(defaultExpanded)
  const failed = failedProp ?? item.state === 'failed'
  const tool = canonicalSwarmToolName(item.name)
  const args = parseArgs(item.input)
  const res = parseJsonRecord(item.output)
  const working = item.state === 'working'
  const spriteState: SimurghSpriteState = failed ? 'error' : working ? 'working' : 'waiting'
  const dur = item.dur && item.dur !== '0.0s' ? item.dur : ''
  const errorText = failed ? clipLine((item.error || item.output).replace(/^Tool execution failed:\s*/i, ''), 220) : ''

  let iconNode: ReactElement = <Icon name="tree" size={14} />
  let titleNode: ReactElement = <span className="swarm-tool__title">Immortals action</span>
  const pills: ReactElement[] = []
  let snippetNode: ReactElement | null = null
  let showOpenSwarm = false
  let detailBody: ReactElement | null = null
  let cardVariant = tool

  if (tool === 'swarm') {
    const action = str(args.action).toLowerCase()
    cardVariant = `swarm-${action || 'action'}`

    if (action === 'add') {
      const info = extractAddedMembers(args, res, roster)
      const role = resolveSimurghRole(info.primaryName, info.primaryRole)
      iconNode = <SimurghSprite role={role} state={spriteState} size={20} animate={working} title={info.primaryName} />
      titleNode = <span className="swarm-tool__title">
        {failed ? 'Failed to spawn' : working ? 'Spawning' : 'Spawned'}{' '}
        <strong className="swarm-tool__entity">{info.displayTarget}</strong>
      </span>
      if (info.model) pills.push(<span key="model" className="swarm-tool__pill swarm-tool__pill--model">{compactModel(info.model)}</span>)
      const ws = workspaceBadgeOf(info.workspace)
      if (ws) pills.push(<span key="ws" className="swarm-tool__pill swarm-tool__pill--ws" data-ws={ws.tone}>{ws.label}</span>)
      if (info.fork) pills.push(<span key="fork" className="swarm-tool__pill">fork @{info.fork}</span>)
      if (info.briefSeq !== null) pills.push(<span key="seq" className="swarm-tool__pill swarm-tool__pill--seq">#{info.briefSeq}</span>)
      if (info.brief) snippetNode = <p className="swarm-tool__snippet">{renderSwarmSnippet(info.brief, 180)}</p>
      showOpenSwarm = true
      detailBody = <>
        {info.members.length > 0 && (
          <div className="swarm-tool__roster">
            {info.members.map(member => (
              <div key={member.name} className="swarm-tool__roster-row">
                <SimurghSprite role={resolveSimurghRole(member.name, member.role)} state={spriteState} size={14} animate={false} />
                <span className="swarm-tool__mention">@{member.name}</span>
                {member.model && <span className="swarm-tool__pill swarm-tool__pill--model">{compactModel(member.model)}</span>}
                {member.workspace && <span className="swarm-tool__pill">{member.workspace}</span>}
              </div>
            ))}
          </div>
        )}
        {info.brief && <div className="swarm-tool__fulltext"><strong>Brief</strong><pre>{info.brief}</pre></div>}
      </>
    } else if (action === 'model') {
      const target = str(res?.name) || str(args.name) || 'coordinator'
      const rosterMember = roster.find(member => member.name === target)
      const nextModel = compactModel(
        str(res?.model_display_name)
        || str(res?.model)
        || str(args.model)
        || rosterMember?.model_display_name
        || rosterMember?.model
        || 'default',
      )
      const prevModel = compactModel(str(res?.previous_model))
      const reason = str(res?.reason) || str(args.reason)
      iconNode = <SimurghSprite role={resolveSimurghRole(target)} state={spriteState} size={18} animate={working} title={target} />
      titleNode = <span className="swarm-tool__title">
        Switched <span className="swarm-tool__mention">@{target}</span> → <strong className="swarm-tool__entity">{nextModel}</strong>
      </span>
      if (prevModel && prevModel !== nextModel) pills.push(<span key="prev" className="swarm-tool__pill">from {prevModel}</span>)
      if (res?.resumed === true) pills.push(<span key="resumed" className="swarm-tool__pill swarm-tool__pill--ok">resumed</span>)
      if (reason) snippetNode = <p className="swarm-tool__snippet">{clipLine(reason, 160)}</p>
      showOpenSwarm = true
    } else if (action === 'kill') {
      const killedList = Array.isArray(res?.killed) ? res.killed.filter((k): k is string => typeof k === 'string') : []
      const names = killedList.length > 0 ? killedList : (str(args.name) ? [str(args.name)] : ['worker'])
      const first = names[0]!
      iconNode = <SimurghSprite role={resolveSimurghRole(first)} state="killed" size={18} animate={false} title={first} />
      titleNode = <span className="swarm-tool__title">
        Retired {names.map((name, idx) => <Fragment key={name}>{idx > 0 ? ', ' : ''}<span className="swarm-tool__mention swarm-tool__mention--muted">@{name}</span></Fragment>)}
      </span>
      pills.push(<span key="retired" className="swarm-tool__pill swarm-tool__pill--muted">retired</span>)
    } else if (action === 'title') {
      const title = str(res?.title) || str(args.title) || 'Untitled Immortals'
      iconNode = <SimurghSprite role="coordinator" state={spriteState} size={18} animate={working} />
      titleNode = <span className="swarm-tool__title">Named Immortals &ldquo;<strong className="swarm-tool__entity">{title}</strong>&rdquo;</span>
      showOpenSwarm = true
    } else if (action === 'status') {
      const members = Array.isArray(res?.members) ? res.members : []
      const workingCount = members.filter(m => m && typeof m === 'object' && (m as Record<string, unknown>).state === 'working').length
      const head = num(res?.head)
      iconNode = <Icon name="agents" size={14} />
      titleNode = <span className="swarm-tool__title">
        Checked roster{members.length > 0 ? ` · ${members.length} member${members.length === 1 ? '' : 's'}` : ''}
      </span>
      if (workingCount > 0) pills.push(<span key="wk" className="swarm-tool__pill swarm-tool__pill--working">{workingCount} working</span>)
      if (head !== null) pills.push(<span key="head" className="swarm-tool__pill swarm-tool__pill--seq">#{head}</span>)
      showOpenSwarm = true
      if (members.length > 0) {
        detailBody = <div className="swarm-tool__roster">
          {members.slice(0, 20).map((entry, idx) => {
            if (!entry || typeof entry !== 'object') return null
            const m = entry as Record<string, unknown>
            const mName = str(m.name) || `member-${idx + 1}`
            const mState = str(m.state) || 'waiting'
            const mModel = compactModel(str(m.model_display_name) || str(m.model))
            return <div key={mName} className="swarm-tool__roster-row">
              <SimurghSprite role={resolveSimurghRole(mName, str(m.role) || undefined)} state={mState === 'working' ? 'working' : mState === 'killed' ? 'killed' : 'waiting'} size={14} animate={false} />
              <span className="swarm-tool__mention">@{mName}</span>
              <span className="swarm-tool__pill">{mState}</span>
              {mModel && <span className="swarm-tool__pill swarm-tool__pill--model">{mModel}</span>}
            </div>
          })}
        </div>
      }
    } else if (action === 'models') {
      const models = Array.isArray(res?.models) ? res.models : []
      iconNode = <Icon name="spark" size={14} />
      titleNode = <span className="swarm-tool__title">
        {working ? 'Listing models' : `Listed ${models.length} model${models.length === 1 ? '' : 's'}`}
      </span>
      if (args.all === true) pills.push(<span key="all" className="swarm-tool__pill">all models</span>)
    } else if (action === 'channel') {
      const op = str(args.op) || str(res?.status) || 'list'
      const channelObj = res?.channel && typeof res.channel === 'object' ? (res.channel as Record<string, unknown>) : null
      const channelName = str(args.channel) || str(channelObj?.name) || str(res?.channel)
      const channelList = Array.isArray(res?.channels) ? res.channels : null
      iconNode = <Icon name="chat" size={14} />
      titleNode = <span className="swarm-tool__title">
        {channelName ? <>Channel <span className="swarm-tool__channel-ref">#{channelName}</span></> : channelList ? `Listed ${channelList.length} channel${channelList.length === 1 ? '' : 's'}` : 'Immortals channels'}
      </span>
      if (op && channelName) pills.push(<span key="op" className="swarm-tool__pill">{op}</span>)
      const topic = str(args.topic) || str(channelObj?.topic)
      if (topic) snippetNode = <p className="swarm-tool__snippet">{clipLine(topic, 140)}</p>
      showOpenSwarm = true
    } else {
      iconNode = <Icon name="tree" size={14} />
      titleNode = <span className="swarm-tool__title">Immortals {action || 'action'}</span>
      showOpenSwarm = true
    }
  } else if (tool === 'post_message') {
    const msgObj = res?.message && typeof res.message === 'object' ? (res.message as Record<string, unknown>) : null
    const text = str(args.text) || str(msgObj?.text)
    const seq = num(res?.seq) ?? num(msgObj?.seq)
    const channel = str(args.channel) || str(res?.channel) || str(msgObj?.channel)
    const woken = Array.isArray(res?.woken) ? res.woken.filter((w): w is string => typeof w === 'string') : []
    iconNode = <Icon name="send" size={14} />
    titleNode = <span className="swarm-tool__title">{working ? 'Posting message' : 'Posted message'}</span>
    if (seq !== null) pills.push(<span key="seq" className="swarm-tool__pill swarm-tool__pill--seq">#{seq}</span>)
    if (channel && channel !== 'main') pills.push(<span key="ch" className="swarm-tool__pill swarm-tool__pill--channel">#{channel}</span>)
    if (woken.length > 0) pills.push(<span key="woken" className="swarm-tool__pill swarm-tool__pill--working">woke {woken.map(w => `@${w}`).join(', ')}</span>)
    if (text) snippetNode = <p className="swarm-tool__snippet">{renderSwarmSnippet(text, 200)}</p>
    showOpenSwarm = true
    if (text) detailBody = <div className="swarm-tool__fulltext"><strong>Message</strong><pre>{text}</pre></div>
  } else if (tool === 'read_board') {
    const isRoster = args.status === true || (Array.isArray(res?.members) && !Array.isArray(res?.messages))
    const head = num(res?.head)
    const since = num(args.since)
    const seqs = Array.isArray(args.seqs) ? args.seqs.filter((s): s is number => typeof s === 'number') : []
    iconNode = <Icon name={isRoster ? 'agents' : 'book'} size={14} />
    if (isRoster) {
      const members = Array.isArray(res?.members) ? res.members : []
      titleNode = <span className="swarm-tool__title">
        Roster status{members.length > 0 ? ` · ${members.length} member${members.length === 1 ? '' : 's'}` : ''}
      </span>
      if (head !== null) pills.push(<span key="head" className="swarm-tool__pill swarm-tool__pill--seq">#{head}</span>)
    } else {
      const messages = Array.isArray(res?.messages) ? res.messages : null
      titleNode = <span className="swarm-tool__title">Read board</span>
      if (seqs.length > 0) {
        pills.push(<span key="seqs" className="swarm-tool__pill swarm-tool__pill--seq">{seqs.map(s => `#${s}`).join(', ')}</span>)
      } else if (since !== null && head !== null) {
        pills.push(<span key="range" className="swarm-tool__pill swarm-tool__pill--seq">#{since} → #{head}</span>)
      } else if (since !== null) {
        pills.push(<span key="since" className="swarm-tool__pill swarm-tool__pill--seq">after #{since}</span>)
      } else if (head !== null) {
        pills.push(<span key="head" className="swarm-tool__pill swarm-tool__pill--seq">#{head}</span>)
      }
      if (messages !== null) {
        pills.push(<span key="cnt" className="swarm-tool__pill">{messages.length} message{messages.length === 1 ? '' : 's'}</span>)
        if (messages.length > 0) {
          detailBody = <div className="swarm-tool__messages">
            {messages.slice(0, 15).map((entry, idx) => {
              if (!entry || typeof entry !== 'object') return null
              const m = entry as Record<string, unknown>
              const mSeq = num(m.seq)
              const mAgent = str(m.agent) || 'member'
              const mText = str(m.text)
              return <div key={mSeq ?? idx} className="swarm-tool__msg-row">
                {mSeq !== null && <span className="swarm-tool__seq-ref">#{mSeq}</span>}
                <span className="swarm-tool__mention">@{mAgent}</span>
                <span className="swarm-tool__msg-text">{renderSwarmSnippet(mText, 160)}</span>
              </div>
            })}
          </div>
        }
      }
    }
  } else if (tool === 'swarm_wait') {
    const status = str(res?.status)
    const since = num(args.since) ?? num(res?.since)
    const wakeSeq = num(res?.wake_seq)
    const wakeMsg = str(res?.message)
    const sleeping = working || status === 'parked' || (!status && !failed && wakeSeq === null)
    iconNode = <Icon name={sleeping ? 'hourglass' : 'spark'} size={14} />
    if (status === 'woken' || wakeSeq !== null) {
      titleNode = <span className="swarm-tool__title">
        Woken by <span className="swarm-tool__seq-ref">#{wakeSeq ?? num(res?.head) ?? '?'}</span>
      </span>
      pills.push(<span key="woken" className="swarm-tool__pill swarm-tool__pill--ok">resumed</span>)
      if (wakeMsg) snippetNode = <p className="swarm-tool__snippet">{renderSwarmSnippet(wakeMsg, 160)}</p>
    } else if (status === 'interrupted_by_steer') {
      titleNode = <span className="swarm-tool__title">Interrupted by user steer</span>
      if (wakeMsg) snippetNode = <p className="swarm-tool__snippet">{clipLine(wakeMsg, 160)}</p>
    } else if (status === 'worker_error') {
      const errObj = res?.worker_error && typeof res.worker_error === 'object' ? (res.worker_error as Record<string, unknown>) : null
      const worker = str(errObj?.member_name) || 'worker'
      titleNode = <span className="swarm-tool__title">Woken by error from <span className="swarm-tool__mention">@{worker}</span></span>
      if (wakeMsg) snippetNode = <p className="swarm-tool__snippet">{clipLine(wakeMsg, 160)}</p>
    } else {
      titleNode = <span className="swarm-tool__title">
        Waiting for board activity{since !== null ? <> after <span className="swarm-tool__seq-ref">#{since}</span></> : ''}
      </span>
      pills.push(<span key="sleep" className="swarm-tool__pill swarm-tool__pill--sleep">{working ? 'listening' : 'parked'}</span>)
      if (args.mentions_only === false) pills.push(<span key="any" className="swarm-tool__pill">any message</span>)
    }
  } else if (tool === 'swarm_overview') {
    const overviewObj = res?.overview && typeof res.overview === 'object' ? (res.overview as Record<string, unknown>) : null
    const summary = str(args.summary) || str(overviewObj?.summary)
    const markdown = str(args.markdown) || str(overviewObj?.markdown)
    const fallbackLine = markdown.split('\n').map(l => l.replace(/^#+\s*/, '').trim()).find(Boolean) ?? ''
    iconNode = <Icon name="plan" size={14} />
    titleNode = <span className="swarm-tool__title">{working ? 'Updating live overview' : 'Updated live overview'}</span>
    if (summary || fallbackLine) snippetNode = <p className="swarm-tool__snippet">{clipLine(summary || fallbackLine, 160)}</p>
    showOpenSwarm = true
    if (markdown) detailBody = <div className="swarm-tool__fulltext"><strong>Overview</strong><pre>{markdown}</pre></div>
  }

  return <div className="swarm-tool" data-variant={cardVariant} data-state={failed ? 'failed' : item.state} data-expanded={detailsOpen || undefined}>
    <div className="swarm-tool__head">
      <span className="swarm-tool__icon" aria-hidden="true">{iconNode}</span>
      <div className="swarm-tool__headline">
        {titleNode}
        {pills.length > 0 && <span className="swarm-tool__pills">{pills}</span>}
      </div>
      <div className="swarm-tool__controls">
        {dur && <span className="swarm-tool__dur">{dur}</span>}
        {failed && <span className="swarm-tool__pill swarm-tool__pill--failed">Failed</span>}
        {showOpenSwarm && (
          <button type="button" className="swarm-tool__open" onClick={() => open('swarm')}>Open Immortals →</button>
        )}
        <button
          type="button"
          className="swarm-tool__toggle"
          aria-expanded={detailsOpen}
          aria-label={detailsOpen ? 'Hide Immortals tool details' : 'Show Immortals tool details'}
          onClick={() => setDetailsOpen(value => !value)}
        >
          <Icon name="chevron" size={11} />
        </button>
      </div>
    </div>
    {snippetNode}
    {errorText && <p className="swarm-tool__error">{errorText}</p>}
    {detailsOpen && (
      <div className="swarm-tool__details">
        {detailBody}
        <div className="swarm-tool__actions">
          {item.input && <CopyButton text={item.input} label="Copy input" />}
          {item.output && <CopyButton text={item.output} label="Copy result" />}
        </div>
        <details className="swarm-tool__raw">
          <summary>Raw JSON</summary>
          <code>{item.name} · {item.id}</code>
          <strong>Input</strong>
          <pre>{item.input || '(no arguments)'}</pre>
          <strong>Result</strong>
          <pre>{item.output || '(no result)'}</pre>
        </details>
      </div>
    )}
  </div>
})
