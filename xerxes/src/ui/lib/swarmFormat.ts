// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { CompletionItem } from '../app/interfaces.js'
import type {
  SwarmActivityPhase,
  SwarmBoardMessage,
  SwarmBoardResponse,
  SwarmChannelSnapshot,
  SwarmEventPayload,
  SwarmInspectMessage,
  SwarmInspectToolCall,
  SwarmMemberActivitySnapshot,
  SwarmMemberInspectResponse,
  SwarmMemberRole,
  SwarmMemberSnapshot,
  SwarmMemberState,
  SwarmOverviewSnapshot,
  SwarmStatusSnapshot,
  SwarmWorkerToolActivity,
  SwarmWorkspaceMode
} from '../gatewayTypes.js'

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value && typeof value === 'object' && !Array.isArray(value))

const str = (value: unknown, fallback = ''): string => (typeof value === 'string' ? value : fallback)

const optionalStr = (value: unknown): string | undefined => {
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  return trimmed ? trimmed : undefined
}

const nonNegInt = (value: unknown, fallback = 0): number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.trunc(value) : fallback

const optionalNonNegInt = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.trunc(value) : undefined

const stringArray = (value: unknown): string[] =>
  Array.isArray(value) ? value.map(item => String(item).trim()).filter(Boolean) : []

const normalizeRole = (value: unknown, name: string): SwarmMemberRole => {
  if (value === 'coordinator' || value === 'worker' || value === 'narrator' || value === 'monitor') {
    return value
  }
  if (name === 'coordinator') return 'coordinator'
  if (name === 'narrator' || /^narrator-\d+$/.test(name)) return 'narrator'
  if (name === 'monitor' || /^monitor-\d+$/.test(name)) return 'monitor'
  return 'worker'
}

const normalizeWorkspace = (value: unknown): SwarmWorkspaceMode =>
  value === 'share' || value === 'branch' || value === 'inherit' ? value : 'inherit'

const normalizeMemberState = (value: unknown, killed: boolean): SwarmMemberState => {
  if (killed || value === 'killed') return 'killed'
  if (value === 'working' || value === 'stopped' || value === 'waiting') return value
  return 'waiting'
}

export function normalizeSwarmOverview(raw: unknown): SwarmOverviewSnapshot | null {
  if (!isRecord(raw)) return null
  const markdown = str(raw.markdown)
  const summary = optionalStr(raw.summary)
  const updatedAt = optionalStr(raw.updated_at)
  const author = optionalStr(raw.author)
  if (!markdown.trim() && !summary) return null
  return {
    markdown,
    ...(summary ? { summary } : {}),
    ...(updatedAt ? { updated_at: updatedAt } : {}),
    ...(author ? { author } : {})
  }
}

export function normalizeSwarmChannel(raw: unknown): SwarmChannelSnapshot | null {
  if (!isRecord(raw)) return null
  const name = optionalStr(raw.name)
  if (!name) return null
  return {
    name,
    topic: str(raw.topic),
    created_by: str(raw.created_by, 'coordinator'),
    archived: raw.archived === true,
    subscriber_count: nonNegInt(raw.subscriber_count)
  }
}

export function normalizeSwarmMember(raw: unknown): SwarmMemberSnapshot | null {
  if (!isRecord(raw)) return null
  const name = optionalStr(raw.name)
  if (!name) return null
  const killed = raw.killed === true || raw.state === 'killed'
  const state = normalizeMemberState(raw.state, killed)
  const model = str(raw.model)
  const modelDisplayName = optionalStr(raw.model_display_name) ?? model
  const group = optionalStr(raw.group)
  const addedBy = optionalStr(raw.added_by)
  const conversationId = optionalStr(raw.conversation_id)
  const workspacePath = optionalStr(raw.workspace_path)
  const currentActivity = state === 'working' ? optionalStr(raw.current_activity) : undefined
  const parkedSince = optionalNonNegInt(raw.parked_since)
  const lastPostSeq = optionalNonNegInt(raw.last_post_seq)
  const lastPostTime = optionalStr(raw.last_post_time)
  const error = optionalStr(raw.error)
  return {
    name,
    ...(group ? { group } : {}),
    role: normalizeRole(raw.role, name),
    ...(addedBy ? { added_by: addedBy } : {}),
    ...(conversationId ? { conversation_id: conversationId } : {}),
    model,
    model_display_name: modelDisplayName,
    workspace: normalizeWorkspace(raw.workspace),
    ...(workspacePath ? { workspace_path: workspacePath } : {}),
    channels: stringArray(raw.channels),
    state,
    killed: state === 'killed',
    ...(currentActivity ? { current_activity: currentActivity } : {}),
    ...(parkedSince !== undefined ? { parked_since: parkedSince } : {}),
    ...(typeof raw.mentions_only === 'boolean' ? { mentions_only: raw.mentions_only } : {}),
    last_read_seq: nonNegInt(raw.last_read_seq),
    ...(lastPostSeq !== undefined ? { last_post_seq: lastPostSeq } : {}),
    ...(lastPostTime ? { last_post_time: lastPostTime } : {}),
    ...(error ? { error } : {})
  }
}

export const DEFAULT_OPEN_BOARD_LINK = '[Open the Immortals UI](sidecar://immortals/board/?auto_open=false)'

export function normalizeSwarmStatus(raw: unknown, fallbackSessionId = ''): SwarmStatusSnapshot {
  const record = isRecord(raw) ? raw : {}
  const channels = Array.isArray(record.channels)
    ? record.channels.map(normalizeSwarmChannel).filter((row): row is SwarmChannelSnapshot => row !== null)
    : []
  const members = Array.isArray(record.members)
    ? record.members.map(normalizeSwarmMember).filter((row): row is SwarmMemberSnapshot => row !== null)
    : []
  const head = nonNegInt(record.head)
  const tailFromSeq = optionalNonNegInt(record.tail_from_seq)
  const active =
    typeof record.active === 'boolean'
      ? record.active
      : members.some(member => member.role !== 'coordinator') || head > 0
  return {
    active,
    session_id: optionalStr(record.session_id) ?? fallbackSessionId,
    title: str(record.title),
    board: 'local',
    open_board: optionalStr(record.open_board) ?? DEFAULT_OPEN_BOARD_LINK,
    head,
    ...(tailFromSeq !== undefined ? { tail_from_seq: tailFromSeq } : {}),
    overview: normalizeSwarmOverview(record.overview),
    channels,
    members
  }
}

export function normalizeSwarmBoardMessage(raw: unknown): SwarmBoardMessage | null {
  if (!isRecord(raw)) return null
  const seq = optionalNonNegInt(raw.seq)
  if (seq === undefined || seq <= 0) return null
  return {
    seq,
    agent: optionalStr(raw.agent) ?? 'coordinator',
    channel: optionalStr(raw.channel) ?? 'main',
    text: str(raw.text),
    mentions: stringArray(raw.mentions),
    channel_mentions: stringArray(raw.channel_mentions),
    broadcast: raw.broadcast === true,
    time: str(raw.time)
  }
}

export function normalizeSwarmBoardResponse(raw: unknown): SwarmBoardResponse {
  const record = isRecord(raw) ? raw : {}
  const messages = Array.isArray(record.messages)
    ? record.messages
        .map(normalizeSwarmBoardMessage)
        .filter((msg): msg is SwarmBoardMessage => msg !== null)
        .sort((a, b) => a.seq - b.seq)
    : []
  const maxSeq = messages.reduce((max, msg) => Math.max(max, msg.seq), 0)
  return {
    head: Math.max(nonNegInt(record.head), maxSeq),
    messages
  }
}

export function normalizeSwarmToolActivity(raw: unknown): SwarmWorkerToolActivity | null {
  if (!isRecord(raw)) return null
  const name = optionalStr(raw.name)
  if (!name) return null
  const rawStatus = str(raw.status)
  const status: SwarmWorkerToolActivity['status'] =
    rawStatus === 'running' || rawStatus === 'done' || rawStatus === 'error' ? rawStatus : 'done'
  const summary = optionalStr(raw.summary)
  const finishedAt = optionalStr(raw.finished_at)
  const resultPreview = optionalStr(raw.result_preview)
  return {
    id: optionalStr(raw.id) ?? name,
    name,
    ...(summary ? { summary } : {}),
    status,
    started_at: str(raw.started_at),
    ...(finishedAt ? { finished_at: finishedAt } : {}),
    ...(resultPreview ? { result_preview: resultPreview } : {})
  }
}

export function normalizeSwarmActivitySnapshot(raw: unknown): SwarmMemberActivitySnapshot {
  const record = isRecord(raw) ? raw : {}
  const rawPhase = str(record.phase)
  const phase: SwarmActivityPhase =
    rawPhase === 'thinking' || rawPhase === 'responding' || rawPhase === 'tool' || rawPhase === 'idle'
      ? rawPhase
      : 'idle'
  const summary = optionalStr(record.summary)
  const liveTextPreview = typeof record.live_text_preview === 'string' && record.live_text_preview.trim()
    ? record.live_text_preview
    : undefined
  const updatedAt = optionalStr(record.updated_at)
  const activeTools = Array.isArray(record.active_tools)
    ? record.active_tools
        .map(normalizeSwarmToolActivity)
        .filter((item): item is SwarmWorkerToolActivity => item !== null)
    : []
  const recentTools = Array.isArray(record.recent_tools)
    ? record.recent_tools
        .map(normalizeSwarmToolActivity)
        .filter((item): item is SwarmWorkerToolActivity => item !== null)
    : []
  return {
    phase,
    ...(summary ? { summary } : {}),
    active_tools: activeTools,
    recent_tools: recentTools,
    ...(liveTextPreview ? { live_text_preview: liveTextPreview } : {}),
    ...(updatedAt ? { updated_at: updatedAt } : {})
  }
}

function normalizeSwarmInspectMessage(raw: unknown, fallbackIdx: number): SwarmInspectMessage | null {
  if (!isRecord(raw)) return null
  const rawRole = str(raw.role)
  const role: SwarmInspectMessage['role'] =
    rawRole === 'user' || rawRole === 'assistant' || rawRole === 'tool' ? rawRole : 'assistant'
  const toolName = optionalStr(raw.tool_name)
  const toolCalls: SwarmInspectToolCall[] | undefined = Array.isArray(raw.tool_calls)
    ? raw.tool_calls.flatMap(entry => {
        if (!isRecord(entry)) return []
        const callName = optionalStr(entry.name)
        if (!callName) return []
        return [{
          id: optionalStr(entry.id) ?? callName,
          name: callName,
          args_preview: str(entry.args_preview)
        }]
      })
    : undefined
  return {
    index: nonNegInt(raw.index, fallbackIdx),
    role,
    text: str(raw.text),
    ...(toolName ? { tool_name: toolName } : {}),
    ...(toolCalls && toolCalls.length > 0 ? { tool_calls: toolCalls } : {})
  }
}

export function normalizeSwarmInspectResponse(
  raw: unknown,
  fallbackSessionId = '',
  fallbackMemberName = 'worker'
): SwarmMemberInspectResponse {
  const record = isRecord(raw) ? raw : {}
  const sessionId = optionalStr(record.session_id) ?? fallbackSessionId
  const member =
    normalizeSwarmMember(record.member) ?? {
      name: fallbackMemberName,
      role: normalizeRole(undefined, fallbackMemberName),
      model: '',
      model_display_name: '',
      workspace: 'inherit',
      channels: ['main'],
      state: 'waiting',
      killed: false,
      last_read_seq: 0
    }
  const messages = Array.isArray(record.messages)
    ? record.messages
        .map((item, idx) => normalizeSwarmInspectMessage(item, idx))
        .filter((item): item is SwarmInspectMessage => item !== null)
    : []
  const error = optionalStr(record.error)
  return {
    ...(typeof record.ok === 'boolean' ? { ok: record.ok } : {}),
    session_id: sessionId,
    member,
    conversation_id: optionalStr(record.conversation_id) ?? member.conversation_id ?? `${sessionId}:swarm:${member.name}`,
    activity: normalizeSwarmActivitySnapshot(record.activity),
    messages,
    total_messages: nonNegInt(record.total_messages, messages.length),
    ...(error ? { error } : {})
  }
}

export function normalizeSwarmEventPayload(raw: unknown): SwarmEventPayload {
  const record = isRecord(raw) ? raw : {}
  const sessionId = str(record.session_id)
  const status = isRecord(record.status) ? normalizeSwarmStatus(record.status, sessionId) : undefined
  const message = normalizeSwarmBoardMessage(record.message) ?? undefined
  const memberObj = isRecord(record.member) ? normalizeSwarmMember(record.member) ?? undefined : undefined
  const memberName =
    optionalStr(record.member_name) ??
    (typeof record.member === 'string' ? optionalStr(record.member) : undefined) ??
    memberObj?.name
  const activity = isRecord(record.activity) ? normalizeSwarmActivitySnapshot(record.activity) : undefined
  const head = Math.max(nonNegInt(record.head), status?.head ?? 0, message?.seq ?? 0)
  return {
    session_id: sessionId || status?.session_id || '',
    kind: str(record.kind, message ? 'message_posted' : 'status'),
    head,
    ...(status ? { status } : {}),
    ...(message ? { message } : {}),
    ...(memberName ? { member_name: memberName } : {}),
    ...(memberObj ? { member: memberObj } : {}),
    ...(activity ? { activity } : {})
  }
}

// ── Counts & formatting ──────────────────────────────────────────────

export interface SwarmCounts {
  total: number
  working: number
  waiting: number
  stopped: number
  killed: number
  living: number
}

export function summarizeSwarmCounts(status: SwarmStatusSnapshot | null | undefined): SwarmCounts {
  const members = status?.members ?? []
  let working = 0
  let waiting = 0
  let stopped = 0
  let killed = 0
  for (const member of members) {
    if (member.killed || member.state === 'killed') killed += 1
    else if (member.state === 'working') working += 1
    else if (member.state === 'stopped') stopped += 1
    else waiting += 1
  }
  return {
    total: members.length,
    working,
    waiting,
    stopped,
    killed,
    living: working + waiting + stopped
  }
}

/** Matches `[Open the Immortals UI](sidecar://immortals/board...)`, legacy `sidecar://swarm/swarm_board...`, and raw URIs. */
const MD_SWARM_SIDECAR_RE = /\[([^\]]+)\]\(\s*sidecar:\/\/(?:immortals\/board|swarm\/swarm_board)[^)\s]*\s*\)/gi
const RAW_SWARM_SIDECAR_RE = /\bsidecar:\/\/(?:immortals\/board|swarm\/swarm_board)[^\s)]*/gi

export const SWARM_BOARD_TUI_HINT = 'Immortals Board active — run /immortals to open'

/**
 * Rewrite desktop/webview `sidecar://immortals/board` (and legacy `sidecar://swarm/swarm_board`)
 * Markdown links into a clean terminal-native hint so TUI transcripts never dump raw sidecar URIs.
 */
export function formatSwarmMarkdownHint(content: string): { content: string; hasSwarmBoardLink: boolean } {
  const lower = content.toLowerCase()
  if (!lower.includes('sidecar://immortals/board') && !lower.includes('sidecar://swarm/swarm_board')) {
    return { content, hasSwarmBoardLink: false }
  }
  let next = content.replace(MD_SWARM_SIDECAR_RE, '').replace(RAW_SWARM_SIDECAR_RE, '')
  // Collapse surplus trailing blank lines left behind when `open_board` was its own final paragraph.
  next = next.replace(/\n{3,}/g, '\n\n').trimEnd()
  return { content: next, hasSwarmBoardLink: true }
}

export type SwarmMessageTokenKind = 'done' | 'stall' | 'broadcast' | 'channel_mention' | 'mention' | 'seq_ref' | 'text'

export interface SwarmMessageToken {
  kind: SwarmMessageTokenKind
  value: string
}

const MESSAGE_TOKEN_RE =
  /(\[done\]|\[stall\]|\[quota\]|@#[a-z0-9][a-z0-9_-]*|@all\b|@[a-z0-9][a-z0-9_-]*|#[1-9]\d*\b)/gi

/** Split a board message body into styled semantic spans without dropping any whitespace or text. */
export function tokenizeSwarmMessageText(text: string): SwarmMessageToken[] {
  if (!text) return []
  const tokens: SwarmMessageToken[] = []
  let lastIndex = 0
  for (const match of text.matchAll(MESSAGE_TOKEN_RE)) {
    const start = match.index ?? 0
    const raw = match[0]
    if (start > lastIndex) {
      tokens.push({ kind: 'text', value: text.slice(lastIndex, start) })
    }
    const lower = raw.toLowerCase()
    let kind: SwarmMessageTokenKind = 'text'
    if (lower === '[done]') kind = 'done'
    else if (lower === '[stall]' || lower === '[quota]') kind = 'stall'
    else if (lower === '@all') kind = 'broadcast'
    else if (lower.startsWith('@#')) kind = 'channel_mention'
    else if (lower.startsWith('@')) kind = 'mention'
    else if (lower.startsWith('#')) kind = 'seq_ref'
    tokens.push({ kind, value: raw })
    lastIndex = start + raw.length
  }
  if (lastIndex < text.length) {
    tokens.push({ kind: 'text', value: text.slice(lastIndex) })
  }
  return tokens
}

/** Compact an ISO or `YYYY-MM-DD HH:MM:SS` timestamp to `HH:MM:SS` when unambiguous. */
export function formatSwarmTime(raw: string): string {
  const trimmed = raw.trim()
  if (!trimmed) return ''
  const match = trimmed.match(/(?:T|\s)(\d{2}:\d{2}:\d{2})/)
  return match?.[1] ?? trimmed
}

/** All selectable channel filters in canonical order (`*`, `main`, then custom channels). */
export function collectSwarmChannelTabs(
  status: SwarmStatusSnapshot | null | undefined,
  messages: readonly SwarmBoardMessage[] = []
): SwarmChannelSnapshot[] {
  const byName = new Map<string, SwarmChannelSnapshot>()
  byName.set('main', {
    name: 'main',
    topic: 'Shared Immortals board',
    created_by: 'coordinator',
    archived: false,
    subscriber_count: status?.members.filter(m => !m.killed).length ?? 0
  })
  for (const ch of status?.channels ?? []) {
    byName.set(ch.name, ch)
  }
  for (const msg of messages) {
    if (msg.channel && !byName.has(msg.channel)) {
      byName.set(msg.channel, {
        name: msg.channel,
        topic: '',
        created_by: msg.agent,
        archived: false,
        subscriber_count: 0
      })
    }
  }
  const main = byName.get('main')!
  const rest = [...byName.values()]
    .filter(ch => ch.name !== 'main')
    .sort((a, b) => a.name.localeCompare(b.name))
  return [main, ...rest]
}

export function formatSwarmStatusPage(
  status: SwarmStatusSnapshot,
  messages: readonly SwarmBoardMessage[] = []
): string {
  const counts = summarizeSwarmCounts(status)
  const title = status.title.trim() || 'Untitled Immortals'
  const lines: string[] = [
    `Immortals · ${title}`,
    `State: ${status.active ? 'active' : 'inactive'} · ${counts.working} working · ${counts.waiting} waiting${counts.stopped ? ` · ${counts.stopped} stopped` : ''}${counts.killed ? ` · ${counts.killed} killed` : ''} · head #${status.head}`
  ]

  if (status.overview?.markdown.trim()) {
    lines.push('', 'OVERVIEW')
    if (status.overview.summary) lines.push(`Summary: ${status.overview.summary}`)
    lines.push(status.overview.markdown.trim())
  }

  lines.push('', 'ROSTER')
  if (!status.members.length) {
    lines.push('  (no Immortals members)')
  } else {
    for (const m of status.members) {
      const lineage = m.added_by ? ` · by @${m.added_by}` : ''
      const post = m.last_post_seq !== undefined ? ` · post #${m.last_post_seq}` : ''
      const model = m.model_display_name || m.model || 'default'
      lines.push(
        `  @${m.name} [${m.state}] (${m.role}${lineage}) · ${model} · ws:${m.workspace} · read #${m.last_read_seq}${post}`
      )
      if (m.error) lines.push(`    error: ${m.error}`)
    }
  }

  const channels = collectSwarmChannelTabs(status, messages)
  if (channels.length > 0) {
    lines.push('', 'CHANNELS')
    for (const ch of channels) {
      lines.push(
        `  #${ch.name}${ch.archived ? ' [archived]' : ''} · ${ch.subscriber_count} subscriber${ch.subscriber_count === 1 ? '' : 's'}${ch.topic ? ` — ${ch.topic}` : ''}`
      )
    }
  }

  if (messages.length > 0) {
    lines.push('', 'RECENT BOARD')
    for (const msg of messages.slice(-10)) {
      const time = formatSwarmTime(msg.time)
      const oneLine = msg.text.replace(/\s+/g, ' ').trim()
      const preview = oneLine.length > 120 ? `${oneLine.slice(0, 119)}…` : oneLine
      lines.push(`  #${msg.seq}${time ? ` ${time}` : ''} @${msg.agent} [#${msg.channel}]: ${preview}`)
    }
  }

  lines.push('', 'Tip: run /immortals or /immortals board to open the interactive Immortals Board.')
  return lines.join('\n')
}

export function formatSwarmChannelsPage(status: SwarmStatusSnapshot): string {
  const channels = collectSwarmChannelTabs(status)
  const lines = [`Immortals Channels · ${status.title.trim() || 'Immortals'}`]
  for (const ch of channels) {
    lines.push(
      `  #${ch.name}${ch.archived ? ' [archived]' : ''} · ${ch.subscriber_count} subscriber${ch.subscriber_count === 1 ? '' : 's'} · created by @${ch.created_by}${ch.topic ? ` — ${ch.topic}` : ''}`
    )
  }
  return lines.join('\n')
}

const SWARM_SUBCOMMANDS: readonly { arg: string; meta: string }[] = [
  { arg: 'board', meta: 'open the interactive Immortals Board overlay' },
  { arg: 'status', meta: 'inspect live Immortals roster, overview and recent messages' },
  { arg: 'channels', meta: 'list #main and custom Immortals channels' },
  { arg: 'stop', meta: 'stop all living Immortals workers' },
  { arg: 'kill all', meta: 'kill all living Immortals workers' }
]

/** Local `/immortals <subcommand>` (and legacy `/swarm <subcommand>`) completions. */
export function swarmSubcommandCompletions(
  input: string,
  status: SwarmStatusSnapshot | null | undefined
): CompletionItem[] {
  const match = input.match(/^\/(immortals|swarm)\s+((?:kill\s+\S*|\S*))$/i)
  if (!match) return []
  const cmd = (match[1] ?? 'immortals').toLowerCase()
  const lowerArg = (match[2] ?? '').replace(/\s+/g, ' ').toLowerCase()

  const livingKillable = (status?.members ?? []).filter(
    m => m.role !== 'coordinator' && !m.killed && m.state !== 'killed'
  )

  const candidates: CompletionItem[] = [
    ...SWARM_SUBCOMMANDS.map(sub => ({
      display: `${cmd} ${sub.arg}`,
      group: 'activity',
      meta: sub.meta,
      text: `/${cmd} ${sub.arg}`
    })),
    ...livingKillable.map(m => ({
      display: `${cmd} kill ${m.name}`,
      group: 'activity',
      meta: `kill ${m.role} @${m.name} (${m.state})`,
      text: `/${cmd} kill ${m.name}`
    }))
  ]

  if (!lowerArg) return candidates
  return candidates.filter(item => item.display.toLowerCase().startsWith(`${cmd} ${lowerArg}`))
}
