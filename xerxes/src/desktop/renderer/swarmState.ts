// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Pure wire-contract parsers, roster/channel selectors, and message-token
 * helpers for Xerxes Native Swarm in the Desktop and VS Code renderers.
 *
 * The renderer never imports `src/swarm/` directly; all state arrives via
 * Daemon v35 RPCs (`swarm.status`, `swarm.board`, `swarm.post`, `swarm.kill`,
 * `swarm.model`, `swarm.activate`) and live `'swarm_event'` pushes.
 */

export type SwarmMemberRole = 'coordinator' | 'worker' | 'narrator' | 'monitor'
export type SwarmWorkspaceMode = 'inherit' | 'share' | 'branch'
export type SwarmMemberState = 'waiting' | 'working' | 'stopped' | 'killed'

export interface SwarmOverviewSnapshot {
  readonly markdown: string
  readonly summary?: string
  readonly updated_at?: string
  readonly author?: string
}

export interface SwarmChannelSnapshot {
  readonly name: string
  readonly topic: string
  readonly created_by: string
  readonly archived: boolean
  readonly subscriber_count: number
}

export interface SwarmMemberSnapshot {
  readonly name: string
  readonly group?: string
  readonly role: SwarmMemberRole
  readonly added_by?: string
  readonly model: string
  readonly model_display_name: string
  readonly workspace: SwarmWorkspaceMode
  readonly workspace_path?: string
  readonly channels: readonly string[]
  readonly state: SwarmMemberState
  readonly killed: boolean
  readonly parked_since?: number
  readonly mentions_only?: boolean
  readonly last_read_seq: number
  readonly last_post_seq?: number
  readonly last_post_time?: string
  readonly current_activity?: string
  readonly conversation_id?: string
  readonly error?: string
}

export interface SwarmWorkerToolActivity {
  readonly id: string
  readonly name: string
  readonly summary?: string
  readonly status: 'running' | 'done' | 'error'
  readonly started_at: string
  readonly finished_at?: string
  readonly result_preview?: string
}

export interface SwarmMemberActivitySnapshot {
  readonly phase: 'idle' | 'thinking' | 'responding' | 'tool'
  readonly summary?: string
  readonly active_tools: readonly SwarmWorkerToolActivity[]
  readonly recent_tools: readonly SwarmWorkerToolActivity[]
  readonly live_text_preview?: string
  readonly updated_at?: string
}

export interface SwarmInspectMessage {
  readonly index: number
  readonly role: 'user' | 'assistant' | 'tool'
  readonly text: string
  readonly tool_name?: string
  readonly tool_calls?: ReadonlyArray<{
    readonly id: string
    readonly name: string
    readonly args_preview: string
  }>
}

export interface SwarmMemberInspectResult {
  readonly session_id: string
  readonly member: SwarmMemberSnapshot
  readonly conversation_id: string
  readonly activity: SwarmMemberActivitySnapshot
  readonly messages: readonly SwarmInspectMessage[]
  readonly total_messages: number
}

export interface SwarmStatusSnapshot {
  readonly active: boolean
  readonly session_id: string
  readonly title: string
  readonly board: 'local'
  readonly open_board: string
  readonly head: number
  readonly overview: SwarmOverviewSnapshot | null
  readonly channels: readonly SwarmChannelSnapshot[]
  readonly members: readonly SwarmMemberSnapshot[]
}

export interface SwarmBoardMessage {
  readonly seq: number
  readonly agent: string
  readonly channel: string
  readonly text: string
  readonly mentions: readonly string[]
  readonly channel_mentions: readonly string[]
  readonly broadcast: boolean
  readonly time: string
}

export interface SwarmBoardSnapshot {
  readonly head: number
  readonly messages: readonly SwarmBoardMessage[]
}

export interface SwarmEventPayload {
  readonly session_id: string
  readonly kind: string
  readonly head: number
  readonly status?: SwarmStatusSnapshot
  readonly message?: SwarmBoardMessage
  readonly member?: SwarmMemberSnapshot
  readonly member_name?: string
  readonly member_state?: SwarmMemberState
  readonly current_activity?: string
  readonly activity?: SwarmMemberActivitySnapshot
}

export interface SwarmMemberCounts {
  readonly total: number
  readonly working: number
  readonly waiting: number
  readonly stopped: number
  readonly killed: number
  /** Living non-coordinator members eligible for Stop All / Kill. */
  readonly livingWorkers: number
}

export type SwarmTextToken =
  | { readonly kind: 'text'; readonly raw: string }
  | { readonly kind: 'code'; readonly raw: string; readonly code: string }
  | { readonly kind: 'bold'; readonly raw: string; readonly text: string }
  | { readonly kind: 'italic'; readonly raw: string; readonly text: string }
  | { readonly kind: 'strike'; readonly raw: string; readonly text: string }
  | { readonly kind: 'done'; readonly raw: string }
  | { readonly kind: 'stall'; readonly raw: string }
  | { readonly kind: 'broadcast'; readonly raw: string }
  | { readonly kind: 'channel_mention'; readonly raw: string; readonly channel: string }
  | { readonly kind: 'mention'; readonly raw: string; readonly name: string }
  | { readonly kind: 'seq_ref'; readonly raw: string; readonly seq: number }

export type SwarmMessageBlock =
  | { readonly kind: 'heading'; readonly level: 1 | 2 | 3 | 4 | 5 | 6; readonly text: string }
  | { readonly kind: 'paragraph'; readonly text: string }
  | { readonly kind: 'list'; readonly items: readonly string[] }
  | { readonly kind: 'ordered_list'; readonly start: number; readonly items: readonly string[] }
  | { readonly kind: 'code'; readonly language: string; readonly code: string }
  | { readonly kind: 'quote'; readonly text: string }
  | { readonly kind: 'hr' }
  | { readonly kind: 'table'; readonly headers: readonly string[]; readonly rows: readonly (readonly string[])[] }

export type SwarmMemberFilter = 'all' | 'active' | 'waiting' | 'retired'

export const SWARM_MESSAGE_FOLD_LINES = 6
export const SWARM_MESSAGE_FOLD_CHARS = 420
export const SWARM_MESSAGE_PREVIEW_LINES = 5
export const SWARM_MESSAGE_PREVIEW_CHARS = 380

export interface SwarmMessagePreview {
  readonly foldable: boolean
  readonly preview: string
  readonly totalLines: number
  readonly displayLines: number
  readonly totalChars: number
}

export type SwarmSlashAction =
  | { readonly kind: 'open_board' }
  | { readonly kind: 'kill'; readonly target: string }
  | { readonly kind: 'daemon_slash'; readonly command: string }
  | { readonly kind: 'activate_task'; readonly task: string }

const str = (value: unknown): string => (typeof value === 'string' ? value : '')
const num = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function stringList(value: unknown): readonly string[] {
  if (!Array.isArray(value)) return []
  return value.filter((entry): entry is string => typeof entry === 'string' && entry.length > 0)
}

function memberRoleOf(value: unknown, name: string): SwarmMemberRole {
  if (value === 'coordinator' || value === 'worker' || value === 'narrator' || value === 'monitor') {
    return value
  }
  if (name === 'coordinator') return 'coordinator'
  if (name === 'narrator' || /^narrator-\d+$/.test(name)) return 'narrator'
  if (name === 'monitor' || /^monitor-\d+$/.test(name)) return 'monitor'
  return 'worker'
}

function workspaceModeOf(value: unknown): SwarmWorkspaceMode {
  return value === 'share' || value === 'branch' ? value : 'inherit'
}

function memberStateOf(value: unknown, killed: boolean): SwarmMemberState {
  if (killed || value === 'killed') return 'killed'
  if (value === 'working' || value === 'waiting' || value === 'stopped') return value
  return 'waiting'
}

/** True for `sidecar://immortals/...` and legacy `sidecar://swarm/...` links. */
export function isSwarmSidecarHref(href: string): boolean {
  const clean = href.trim()
  return /^sidecar:\/\/(?:immortals|swarm)(?:\/|$|[?#])/i.test(clean)
}

export function parseSwarmOverview(value: unknown): SwarmOverviewSnapshot | null {
  if (!isPlainObject(value)) return null
  const markdown = str(value.markdown)
  const summary = str(value.summary)
  const updatedAt = str(value.updated_at)
  const author = str(value.author)
  if (!markdown && !summary) return null
  return {
    markdown,
    ...(summary ? { summary } : {}),
    ...(updatedAt ? { updated_at: updatedAt } : {}),
    ...(author ? { author } : {}),
  }
}

export function parseSwarmChannel(value: unknown): SwarmChannelSnapshot | null {
  if (!isPlainObject(value)) return null
  const name = str(value.name).trim()
  if (!name) return null
  return {
    name,
    topic: str(value.topic),
    created_by: str(value.created_by) || 'coordinator',
    archived: value.archived === true,
    subscriber_count: Math.max(0, Math.trunc(num(value.subscriber_count) ?? 0)),
  }
}

export function parseSwarmMember(value: unknown): SwarmMemberSnapshot | null {
  if (!isPlainObject(value)) return null
  const name = str(value.name).trim()
  if (!name) return null
  const killed = value.killed === true || value.state === 'killed'
  const state = memberStateOf(value.state, killed)
  const role = memberRoleOf(value.role, name)
  const model = str(value.model)
  const modelDisplayName = str(value.model_display_name) || model
  const group = str(value.group)
  const addedBy = str(value.added_by)
  const workspacePath = str(value.workspace_path)
  const parkedSince = num(value.parked_since)
  const lastPostSeq = num(value.last_post_seq)
  const lastPostTime = str(value.last_post_time)
  const currentActivity = str(value.current_activity)
  const conversationId = str(value.conversation_id)
  const error = str(value.error)
  const channels = stringList(value.channels)
  return {
    name,
    ...(group ? { group } : {}),
    role,
    ...(addedBy ? { added_by: addedBy } : {}),
    model,
    model_display_name: modelDisplayName,
    workspace: workspaceModeOf(value.workspace),
    ...(workspacePath ? { workspace_path: workspacePath } : {}),
    channels: channels.length > 0 ? channels : ['main'],
    state,
    killed: state === 'killed',
    ...(parkedSince !== null ? { parked_since: Math.max(0, Math.trunc(parkedSince)) } : {}),
    ...(typeof value.mentions_only === 'boolean' ? { mentions_only: value.mentions_only } : {}),
    last_read_seq: Math.max(0, Math.trunc(num(value.last_read_seq) ?? 0)),
    ...(lastPostSeq !== null ? { last_post_seq: Math.max(0, Math.trunc(lastPostSeq)) } : {}),
    ...(lastPostTime ? { last_post_time: lastPostTime } : {}),
    ...(currentActivity ? { current_activity: currentActivity } : {}),
    ...(conversationId ? { conversation_id: conversationId } : {}),
    ...(error ? { error } : {}),
  }
}

/** Validate a `swarm.status` response or embedded `swarm_event.status` object. */
export function parseSwarmStatus(value: unknown): SwarmStatusSnapshot | null {
  if (!isPlainObject(value)) return null
  if (value.ok === false || value.kind === 'error') return null
  // Require at least one canonical swarm status field so generic `{ ok: true }` mocks stay null.
  if (typeof value.active !== 'boolean' && !Array.isArray(value.members) && typeof value.head !== 'number') {
    return null
  }
  const members = Array.isArray(value.members)
    ? value.members.map(parseSwarmMember).filter((row): row is SwarmMemberSnapshot => row !== null)
    : []
  const channels = Array.isArray(value.channels)
    ? value.channels.map(parseSwarmChannel).filter((row): row is SwarmChannelSnapshot => row !== null)
    : []
  const head = Math.max(0, Math.trunc(num(value.head) ?? 0))
  const active = typeof value.active === 'boolean' ? value.active : members.some(member => member.role !== 'coordinator')
  return {
    active,
    session_id: str(value.session_id),
    title: str(value.title),
    board: 'local',
    open_board: str(value.open_board) || '[Open the Immortals UI](sidecar://immortals/board/?auto_open=false)',
    head,
    overview: parseSwarmOverview(value.overview),
    channels,
    members,
  }
}

export function parseSwarmMessage(value: unknown): SwarmBoardMessage | null {
  if (!isPlainObject(value)) return null
  const seq = num(value.seq)
  const agent = str(value.agent).trim()
  if (seq === null || seq < 1 || !agent) return null
  return {
    seq: Math.trunc(seq),
    agent,
    channel: str(value.channel).trim() || 'main',
    text: str(value.text),
    mentions: stringList(value.mentions),
    channel_mentions: stringList(value.channel_mentions),
    broadcast: value.broadcast === true,
    time: str(value.time),
  }
}

export function parseSwarmBoard(value: unknown): SwarmBoardSnapshot | null {
  if (!isPlainObject(value) || value.ok === false || value.kind === 'error') return null
  if (!Array.isArray(value.messages) && typeof value.head !== 'number') return null
  const messages = Array.isArray(value.messages)
    ? value.messages.map(parseSwarmMessage).filter((row): row is SwarmBoardMessage => row !== null)
    : []
  const maxSeq = messages.reduce((max, row) => Math.max(max, row.seq), 0)
  const head = Math.max(maxSeq, Math.trunc(num(value.head) ?? 0))
  return {
    head,
    messages: mergeSwarmMessages([], messages),
  }
}

function parseWorkerToolActivity(value: unknown): SwarmWorkerToolActivity | null {
  if (!isPlainObject(value)) return null
  const id = str(value.id).trim()
  const name = str(value.name).trim()
  if (!id || !name) return null
  const rawStatus = value.status
  const status: SwarmWorkerToolActivity['status'] =
    rawStatus === 'running' || rawStatus === 'error' ? rawStatus : 'done'
  const summary = str(value.summary)
  const finishedAt = str(value.finished_at)
  const resultPreview = str(value.result_preview)
  return {
    id,
    name,
    ...(summary ? { summary } : {}),
    status,
    started_at: str(value.started_at),
    ...(finishedAt ? { finished_at: finishedAt } : {}),
    ...(resultPreview ? { result_preview: resultPreview } : {}),
  }
}

export function parseSwarmMemberActivity(value: unknown): SwarmMemberActivitySnapshot | null {
  if (!isPlainObject(value)) return null
  const rawPhase = value.phase
  const phase: SwarmMemberActivitySnapshot['phase'] =
    rawPhase === 'thinking' || rawPhase === 'responding' || rawPhase === 'tool'
      ? rawPhase
      : 'idle'
  const summary = str(value.summary)
  const liveTextPreview = str(value.live_text_preview)
  const updatedAt = str(value.updated_at)
  const activeTools = Array.isArray(value.active_tools)
    ? value.active_tools.map(parseWorkerToolActivity).filter((item): item is SwarmWorkerToolActivity => item !== null)
    : []
  const recentTools = Array.isArray(value.recent_tools)
    ? value.recent_tools.map(parseWorkerToolActivity).filter((item): item is SwarmWorkerToolActivity => item !== null)
    : []
  return {
    phase,
    ...(summary ? { summary } : {}),
    active_tools: activeTools,
    recent_tools: recentTools,
    ...(liveTextPreview ? { live_text_preview: liveTextPreview } : {}),
    ...(updatedAt ? { updated_at: updatedAt } : {}),
  }
}

export function parseSwarmMemberInspect(value: unknown): SwarmMemberInspectResult | null {
  if (!isPlainObject(value) || value.ok === false || value.kind === 'error') return null
  const member = parseSwarmMember(value.member)
  if (!member) return null
  const activity = parseSwarmMemberActivity(value.activity) ?? {
    phase: 'idle',
    active_tools: [],
    recent_tools: [],
  }
  const messages: SwarmInspectMessage[] = Array.isArray(value.messages)
    ? value.messages.flatMap((raw, fallbackIndex) => {
        if (!isPlainObject(raw)) return []
        const role = raw.role
        if (role !== 'user' && role !== 'assistant' && role !== 'tool') return []
        const index = Math.max(0, Math.trunc(num(raw.index) ?? fallbackIndex))
        const text = str(raw.text)
        const toolName = str(raw.tool_name)
        const toolCalls = Array.isArray(raw.tool_calls)
          ? raw.tool_calls.flatMap(call => {
              if (!isPlainObject(call)) return []
              const id = str(call.id)
              const name = str(call.name)
              if (!name) return []
              return [{ id: id || name, name, args_preview: str(call.args_preview) }]
            })
          : undefined
        return [{
          index,
          role,
          text,
          ...(toolName ? { tool_name: toolName } : {}),
          ...(toolCalls && toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        }]
      })
    : []
  const totalMessages = Math.max(
    messages.length,
    Math.trunc(num(value.total_messages) ?? messages.length),
  )
  return {
    session_id: str(value.session_id),
    member,
    conversation_id: str(value.conversation_id) || `${str(value.session_id)}:swarm:${member.name}`,
    activity,
    messages,
    total_messages: totalMessages,
  }
}

/** Maximum board messages retained in renderer memory (preserves recent tail). */
export const SWARM_MESSAGE_MEMORY_CAP = 500

export function parseSwarmEvent(value: unknown): SwarmEventPayload | null {
  if (!isPlainObject(value)) return null
  const status = parseSwarmStatus(value.status)
  const message = parseSwarmMessage(value.message)
  const member = parseSwarmMember(value.member)
  const activity = parseSwarmMemberActivity(value.activity)
  const memberName = str(value.member_name).trim() || str(value.name).trim() || member?.name || ''
  const rawMemberState = value.member_state
  const memberState: SwarmMemberState | undefined =
    rawMemberState === 'working' || rawMemberState === 'waiting' || rawMemberState === 'stopped' || rawMemberState === 'killed'
      ? rawMemberState
      : undefined
  const currentActivity = str(value.current_activity) || activity?.summary || ''
  const sessionId = str(value.session_id) || status?.session_id || ''
  const head = Math.max(
    0,
    Math.trunc(num(value.head) ?? status?.head ?? message?.seq ?? 0),
  )
  return {
    session_id: sessionId,
    kind: str(value.kind) || 'updated',
    head,
    ...(status ? { status } : {}),
    ...(message ? { message } : {}),
    ...(member ? { member } : {}),
    ...(memberName ? { member_name: memberName } : {}),
    ...(memberState ? { member_state: memberState } : {}),
    ...(currentActivity ? { current_activity: currentActivity } : {}),
    ...(activity ? { activity } : {}),
  }
}

/**
 * Apply a lightweight `swarm_event` delta (`member_activity` or `member_updated`
 * without a full 28 KB `status` snapshot) while preserving `===` object identity
 * for every untouched member in the roster.
 */
export function applySwarmStatusDelta(
  current: SwarmStatusSnapshot,
  event: SwarmEventPayload,
): SwarmStatusSnapshot {
  const nextHead = Math.max(current.head, event.head)
  const targetName = event.member?.name ?? event.member_name
  let members = current.members
  let membersChanged = false

  if (event.member) {
    const idx = current.members.findIndex(m => m.name === event.member!.name)
    if (idx >= 0) {
      const prev = current.members[idx]!
      const next = event.member
      if (
        prev.state !== next.state
        || prev.killed !== next.killed
        || prev.current_activity !== next.current_activity
        || prev.last_read_seq !== next.last_read_seq
        || prev.last_post_seq !== next.last_post_seq
        || prev.model !== next.model
        || prev.error !== next.error
      ) {
        const copy = current.members.slice()
        copy[idx] = next
        members = copy
        membersChanged = true
      }
    } else {
      members = [...current.members, event.member]
      membersChanged = true
    }
  } else if (targetName && (event.current_activity !== undefined || event.member_state !== undefined)) {
    const idx = current.members.findIndex(m => m.name === targetName)
    if (idx >= 0) {
      const prev = current.members[idx]!
      const nextActivity = event.current_activity ?? prev.current_activity
      const nextState = event.member_state ?? prev.state
      if (prev.current_activity !== nextActivity || prev.state !== nextState) {
        const copy = current.members.slice()
        copy[idx] = {
          ...prev,
          state: nextState,
          killed: nextState === 'killed',
          ...(nextActivity ? { current_activity: nextActivity } : {}),
        }
        members = copy
        membersChanged = true
      }
    }
  }

  if (!membersChanged && nextHead === current.head) return current
  return {
    ...current,
    head: nextHead,
    members,
  }
}

/** Merge board messages by monotonic `seq`, keeping chronological ascending order and bounded memory. */
export function mergeSwarmMessages(
  existing: readonly SwarmBoardMessage[],
  incoming: readonly SwarmBoardMessage[],
  cap = SWARM_MESSAGE_MEMORY_CAP,
): readonly SwarmBoardMessage[] {
  if (incoming.length === 0) {
    return existing.length > cap ? existing.slice(-cap) : existing
  }
  const bySeq = new Map<number, SwarmBoardMessage>()
  for (const msg of existing) bySeq.set(msg.seq, msg)
  for (const msg of incoming) bySeq.set(msg.seq, msg)
  const sorted = [...bySeq.values()].sort((a, b) => a.seq - b.seq)
  return sorted.length > cap ? sorted.slice(-cap) : sorted
}

/** Always includes `#main` first, then custom channels in stable order. */
export function swarmChannelsOf(
  status: SwarmStatusSnapshot | null | undefined,
  messages: readonly SwarmBoardMessage[] = [],
): readonly SwarmChannelSnapshot[] {
  const byName = new Map<string, SwarmChannelSnapshot>()
  const mainSubCount = status
    ? status.members.filter(member => !member.killed).length
    : 0
  byName.set('main', {
    name: 'main',
    topic: 'Shared Immortals board',
    created_by: 'coordinator',
    archived: false,
    subscriber_count: mainSubCount,
  })
  for (const channel of status?.channels ?? []) {
    byName.set(channel.name, channel)
  }
  for (const message of messages) {
    if (!byName.has(message.channel)) {
      byName.set(message.channel, {
        name: message.channel,
        topic: '',
        created_by: 'coordinator',
        archived: false,
        subscriber_count: 0,
      })
    }
  }
  const main = byName.get('main')!
  const rest = [...byName.values()]
    .filter(channel => channel.name !== 'main')
    .sort((a, b) => Number(a.archived) - Number(b.archived) || a.name.localeCompare(b.name))
  return [main, ...rest]
}

export function swarmMemberCounts(members: readonly SwarmMemberSnapshot[]): SwarmMemberCounts {
  let working = 0
  let waiting = 0
  let stopped = 0
  let killed = 0
  let livingWorkers = 0
  for (const member of members) {
    if (member.killed || member.state === 'killed') {
      killed += 1
      continue
    }
    if (member.role !== 'coordinator') livingWorkers += 1
    if (member.state === 'working') working += 1
    else if (member.state === 'stopped') stopped += 1
    else waiting += 1
  }
  return {
    total: members.length,
    working,
    waiting,
    stopped,
    killed,
    livingWorkers,
  }
}

function statePriority(member: SwarmMemberSnapshot): number {
  if (member.killed || member.state === 'killed') return 3
  if (member.state === 'working') return 0
  if (member.state === 'waiting') return 1
  return 2
}

function rolePriority(role: SwarmMemberRole): number {
  if (role === 'coordinator') return 0
  if (role === 'worker') return 1
  if (role === 'narrator') return 2
  return 3
}

/**
 * Clean hierarchical roster order:
 * 1. Coordinator first
 * 2. Living top-level members (`working` -> `waiting` -> `stopped`), with any
 *    worker-spawned helpers (`added_by` naming another worker) nested directly
 *    underneath their parent so lineage is immediately scannable
 * 3. Killed members at the bottom (`DESIGN.md` roster rule)
 */
export function sortSwarmMembers(members: readonly SwarmMemberSnapshot[]): readonly SwarmMemberSnapshot[] {
  const baseCompare = (a: SwarmMemberSnapshot, b: SwarmMemberSnapshot): number => {
    if (a.role === 'coordinator' && b.role !== 'coordinator') return -1
    if (b.role === 'coordinator' && a.role !== 'coordinator') return 1
    const killDiff = Number(a.killed || a.state === 'killed') - Number(b.killed || b.state === 'killed')
    if (killDiff !== 0) return killDiff
    const stateDiff = statePriority(a) - statePriority(b)
    if (stateDiff !== 0) return stateDiff
    const roleDiff = rolePriority(a.role) - rolePriority(b.role)
    if (roleDiff !== 0) return roleDiff
    return a.name.localeCompare(b.name)
  }

  const byName = new Map(members.map(member => [member.name, member] as const))
  const childrenOf = new Map<string, SwarmMemberSnapshot[]>()
  const roots: SwarmMemberSnapshot[] = []

  for (const member of members) {
    const parent = member.added_by
    const hasWorkerParent =
      Boolean(parent) &&
      parent !== 'coordinator' &&
      parent !== member.name &&
      byName.has(parent!) &&
      Boolean(member.killed || member.state === 'killed') ===
        Boolean(byName.get(parent!)!.killed || byName.get(parent!)!.state === 'killed')
    if (hasWorkerParent) {
      const list = childrenOf.get(parent!) ?? []
      list.push(member)
      childrenOf.set(parent!, list)
    } else {
      roots.push(member)
    }
  }

  roots.sort(baseCompare)
  for (const list of childrenOf.values()) list.sort(baseCompare)

  const out: SwarmMemberSnapshot[] = []
  const visited = new Set<string>()
  const visit = (member: SwarmMemberSnapshot): void => {
    if (visited.has(member.name)) return
    visited.add(member.name)
    out.push(member)
    for (const child of childrenOf.get(member.name) ?? []) visit(child)
  }
  for (const root of roots) visit(root)
  for (const member of [...members].sort(baseCompare)) visit(member)
  return out
}

const TOKEN_BOUNDARY_BEFORE = /[\s([{"',;:<>]/
const UNDERSCORE_ITALIC_BEFORE = /[\s([{"']/
const UNDERSCORE_ITALIC_AFTER = /[\s)\]}"'.,;:!?]/
const SWARM_BASE_TOKEN_PARTS = [
  '(`[^`\\n]+`)',
  '(\\[(?:done|stall)\\])',
  '(@#[a-zA-Z0-9][a-zA-Z0-9_-]{0,31}\\b)',
  '(@all\\b)',
  '(@[a-zA-Z0-9][a-zA-Z0-9_-]{0,31}\\b)',
  '(#[1-9]\\d*\\b)',
] as const

const tokenPatternCache = new Map<string, RegExp>()

function swarmTokenPattern(allowBold: boolean, allowItalic: boolean, allowStrike: boolean): RegExp {
  const key = `${allowBold ? 1 : 0}${allowItalic ? 1 : 0}${allowStrike ? 1 : 0}`
  const cached = tokenPatternCache.get(key)
  if (cached) return cached
  const parts: string[] = [SWARM_BASE_TOKEN_PARTS[0]]
  if (allowBold) parts.push('(\\*\\*[^\\s*\\n](?:[^\\n]*?[^\\s*\\n])?\\*\\*)')
  if (allowStrike) parts.push('(~~[^\\s~\\n](?:[^\\n]*?[^\\s~\\n])?~~)')
  if (allowItalic) {
    parts.push('(\\*[^\\s*\\n](?:[^*\\n]*?[^\\s*\\n])?\\*)')
    parts.push('(_[^\\s_\\n](?:[^_\\n]*?[^\\s_\\n])?_)')
  }
  parts.push(...SWARM_BASE_TOKEN_PARTS.slice(1))
  const compiled = new RegExp(parts.join('|'), 'gi')
  tokenPatternCache.set(key, compiled)
  return compiled
}

/**
 * Tokenize a board message into plain text, inline `` `code` `` spans,
 * `**bold**`, `*italic*` / `_italic_`, `~~strikethrough~~`, `[done]` / `[stall]`
 * status badges, `@all` / `@#channel` / `@worker` chips, and clickable `#<seq>` refs.
 */
export function tokenizeSwarmMessage(
  input: string,
  options: {
    readonly allowBold?: boolean
    readonly allowItalic?: boolean
    readonly allowStrike?: boolean
  } = {},
): readonly SwarmTextToken[] {
  if (!input) return []
  const allowBold = options.allowBold !== false
  const allowItalic = options.allowItalic !== false
  const allowStrike = options.allowStrike !== false
  const pattern = swarmTokenPattern(allowBold, allowItalic, allowStrike)
  const tokens: SwarmTextToken[] = []
  let cursor = 0
  const pushText = (chunk: string): void => {
    if (!chunk) return
    const prev = tokens.at(-1)
    if (prev?.kind === 'text') {
      tokens[tokens.length - 1] = { kind: 'text', raw: prev.raw + chunk }
    } else {
      tokens.push({ kind: 'text', raw: chunk })
    }
  }

  for (const match of input.matchAll(pattern)) {
    const start = match.index ?? 0
    const raw = match[0]
    const prevChar = start > 0 ? input[start - 1]! : ''
    const afterChar = start + raw.length < input.length ? input[start + raw.length]! : ''

    if (raw.startsWith('`') && raw.endsWith('`') && raw.length >= 2) {
      if (start > cursor) pushText(input.slice(cursor, start))
      tokens.push({ kind: 'code', raw, code: raw.slice(1, -1) })
      cursor = start + raw.length
      continue
    }

    if (allowBold && raw.startsWith('**') && raw.endsWith('**') && raw.length >= 5) {
      if (start > 0 && /[a-zA-Z0-9*]/.test(prevChar)) continue
      if (afterChar === '*') continue
      if (start > cursor) pushText(input.slice(cursor, start))
      tokens.push({ kind: 'bold', raw, text: raw.slice(2, -2) })
      cursor = start + raw.length
      continue
    }

    if (allowStrike && raw.startsWith('~~') && raw.endsWith('~~') && raw.length >= 5) {
      if (start > 0 && /[a-zA-Z0-9~]/.test(prevChar)) continue
      if (start > cursor) pushText(input.slice(cursor, start))
      tokens.push({ kind: 'strike', raw, text: raw.slice(2, -2) })
      cursor = start + raw.length
      continue
    }

    if (allowItalic && raw.startsWith('*') && raw.endsWith('*') && !raw.startsWith('**') && raw.length >= 3) {
      if ((start > 0 && /[a-zA-Z0-9*]/.test(prevChar)) || /[a-zA-Z0-9*]/.test(afterChar)) continue
      if (start > cursor) pushText(input.slice(cursor, start))
      tokens.push({ kind: 'italic', raw, text: raw.slice(1, -1) })
      cursor = start + raw.length
      continue
    }

    if (allowItalic && raw.startsWith('_') && raw.endsWith('_') && !raw.startsWith('__') && raw.length >= 3) {
      const okBefore = start === 0 || UNDERSCORE_ITALIC_BEFORE.test(prevChar)
      const okAfter = !afterChar || UNDERSCORE_ITALIC_AFTER.test(afterChar)
      if (!okBefore || !okAfter) continue
      if (start > cursor) pushText(input.slice(cursor, start))
      tokens.push({ kind: 'italic', raw, text: raw.slice(1, -1) })
      cursor = start + raw.length
      continue
    }

    const isTag = raw.startsWith('[')
    const validBoundary = isTag || start === 0 || TOKEN_BOUNDARY_BEFORE.test(prevChar)
    if (!validBoundary) continue

    if (start > cursor) pushText(input.slice(cursor, start))
    const lower = raw.toLowerCase()
    if (lower === '[done]') {
      tokens.push({ kind: 'done', raw })
    } else if (lower === '[stall]') {
      tokens.push({ kind: 'stall', raw })
    } else if (raw.startsWith('@#')) {
      tokens.push({ kind: 'channel_mention', raw, channel: raw.slice(2) })
    } else if (lower === '@all') {
      tokens.push({ kind: 'broadcast', raw })
    } else if (raw.startsWith('@')) {
      tokens.push({ kind: 'mention', raw, name: raw.slice(1) })
    } else if (raw.startsWith('#')) {
      const seq = Number.parseInt(raw.slice(1), 10)
      if (Number.isSafeInteger(seq) && seq >= 1) {
        tokens.push({ kind: 'seq_ref', raw, seq })
      } else {
        pushText(raw)
      }
    }
    cursor = start + raw.length
  }

  if (cursor < input.length) pushText(input.slice(cursor))
  return tokens
}

const HEADING_LINE_PATTERN = /^\s*(#{1,6})\s+(.+?)\s*#*\s*$/
const FENCE_OPEN_PATTERN = /^\s*(`{3,}|~{3,})([^\s`~]*)\s*$/
const HR_LINE_PATTERN = /^\s*(?:---+|\*\*\*+|___+)\s*$/
const QUOTE_LINE_PATTERN = /^\s*>\s?(.*)$/
const BULLET_LINE_PATTERN = /^\s*[-*•]\s+(.*)$/
const ORDERED_LIST_PATTERN = /^\s*(\d+)[.)]\s+(.*)$/
const TABLE_SEP_PATTERN = /^\s*\|?\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)+\s*\|?\s*$/

function splitTableRow(line: string): string[] {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim())
}

/**
 * Split a board message into structured Markdown blocks (headings, paragraphs,
 * bullet & ordered lists, fenced code blocks, blockquotes, rules, and tables)
 * so worker reports never leak raw `###` / `---` / fenced syntax.
 */
export function parseSwarmMessageBlocks(input: string): readonly SwarmMessageBlock[] {
  if (!input) return []
  const lines = input.split(/\r?\n/)
  const blocks: SwarmMessageBlock[] = []
  let paragraphLines: string[] = []
  let listItems: string[] = []
  let orderedItems: string[] = []
  let orderedStart = 1
  let quoteLines: string[] = []
  let fence: { char: string; length: number; language: string; lines: string[] } | null = null

  const flushParagraph = (): void => {
    if (paragraphLines.length === 0) return
    blocks.push({ kind: 'paragraph', text: paragraphLines.join('\n') })
    paragraphLines = []
  }
  const flushList = (): void => {
    if (listItems.length === 0) return
    blocks.push({ kind: 'list', items: listItems })
    listItems = []
  }
  const flushOrdered = (): void => {
    if (orderedItems.length === 0) return
    blocks.push({ kind: 'ordered_list', start: orderedStart, items: orderedItems })
    orderedItems = []
    orderedStart = 1
  }
  const flushQuote = (): void => {
    if (quoteLines.length === 0) return
    blocks.push({ kind: 'quote', text: quoteLines.join('\n') })
    quoteLines = []
  }
  const flushAll = (): void => {
    flushParagraph()
    flushList()
    flushOrdered()
    flushQuote()
  }

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i] ?? ''

    if (fence) {
      const closeRe = new RegExp(`^\\s*${fence.char}{${fence.length},}\\s*$`)
      if (closeRe.test(line)) {
        blocks.push({ kind: 'code', language: fence.language, code: fence.lines.join('\n') })
        fence = null
      } else {
        fence.lines.push(line)
      }
      continue
    }

    const fenceOpen = FENCE_OPEN_PATTERN.exec(line)
    if (fenceOpen) {
      flushAll()
      const marker = fenceOpen[1]!
      fence = { char: marker[0]!, length: marker.length, language: (fenceOpen[2] ?? '').trim(), lines: [] }
      continue
    }

    if (!line.trim()) {
      flushAll()
      continue
    }

    const headingMatch = HEADING_LINE_PATTERN.exec(line)
    if (headingMatch) {
      flushAll()
      const level = Math.min(6, Math.max(1, headingMatch[1]!.length)) as 1 | 2 | 3 | 4 | 5 | 6
      blocks.push({ kind: 'heading', level, text: headingMatch[2]!.trim() })
      continue
    }

    if (HR_LINE_PATTERN.test(line)) {
      flushAll()
      blocks.push({ kind: 'hr' })
      continue
    }

    if (line.includes('|') && i + 1 < lines.length && TABLE_SEP_PATTERN.test(lines[i + 1] ?? '')) {
      flushAll()
      const headers = splitTableRow(line)
      const rows: string[][] = []
      let j = i + 2
      for (; j < lines.length; j += 1) {
        const rowLine = lines[j] ?? ''
        if (!rowLine.trim() || !rowLine.includes('|')) break
        rows.push(splitTableRow(rowLine))
      }
      blocks.push({ kind: 'table', headers, rows })
      i = j - 1
      continue
    }

    const quoteMatch = QUOTE_LINE_PATTERN.exec(line)
    if (quoteMatch) {
      flushParagraph()
      flushList()
      flushOrdered()
      quoteLines.push(quoteMatch[1] ?? '')
      continue
    }

    const bulletMatch = BULLET_LINE_PATTERN.exec(line)
    if (bulletMatch) {
      flushParagraph()
      flushOrdered()
      flushQuote()
      listItems.push(bulletMatch[1] ?? '')
      continue
    }

    const orderedMatch = ORDERED_LIST_PATTERN.exec(line)
    if (orderedMatch) {
      flushParagraph()
      flushList()
      flushQuote()
      if (orderedItems.length === 0) orderedStart = Number.parseInt(orderedMatch[1]!, 10) || 1
      orderedItems.push(orderedMatch[2] ?? '')
      continue
    }

    flushList()
    flushOrdered()
    flushQuote()
    paragraphLines.push(line)
  }

  if (fence) {
    blocks.push({ kind: 'code', language: fence.language, code: fence.lines.join('\n') })
  }
  flushAll()
  return blocks
}

/**
 * Format ISO-8601 (`2026-11-22T01:57:02.980Z`), date-time (`2026-11-22 01:57:02`),
 * or clock (`01:57:02`) timestamps into a crisp `HH:MM` inline label.
 */
export function formatSwarmShortTime(raw: string): string {
  const trimmed = raw.trim()
  if (!trimmed) return ''
  const match = /(?:^|[\sT])(\d{2}:\d{2})(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?$/i.exec(trimmed)
  if (match?.[1]) return match[1]
  const isoClock = /T(\d{2}:\d{2})/i.exec(trimmed)
  return isoClock?.[1] ?? trimmed
}

function sanitizeTruncatedPreview(snippet: string): string {
  const lines = snippet.split('\n')
  const lastIndex = lines.length - 1
  let last = lines[lastIndex] ?? ''
  const ticks = (last.match(/`/g) ?? []).length
  if (ticks % 2 === 1) {
    const cut = last.lastIndexOf('`')
    if (cut >= 0) last = last.slice(0, cut).trimEnd()
  }
  const bolds = (last.match(/\*\*/g) ?? []).length
  if (bolds % 2 === 1) {
    const cut = last.lastIndexOf('**')
    if (cut >= 0) last = last.slice(0, cut).trimEnd()
  }
  lines[lastIndex] = last
  return lines.join('\n').trimEnd()
}

/**
 * Compute whether a board message should auto-fold (`> 6` lines OR `> 420` chars)
 * and return a calm ~5-line / ~380-char preview snippet.
 */
export function previewSwarmMessage(text: string): SwarmMessagePreview {
  const normalized = text.replace(/\r\n/g, '\n')
  const lines = normalized.split('\n')
  const totalLines = lines.length
  const totalChars = normalized.length
  const foldable = totalLines > SWARM_MESSAGE_FOLD_LINES || totalChars > SWARM_MESSAGE_FOLD_CHARS
  const displayLines =
    totalLines > SWARM_MESSAGE_FOLD_LINES
      ? totalLines
      : Math.max(totalLines, Math.ceil(totalChars / 75))

  if (!foldable) {
    return {
      foldable: false,
      preview: text,
      totalLines,
      displayLines,
      totalChars,
    }
  }

  let preview = lines.slice(0, SWARM_MESSAGE_PREVIEW_LINES).join('\n')
  let clippedChars = false
  if (preview.length > SWARM_MESSAGE_PREVIEW_CHARS) {
    const windowText = preview.slice(0, SWARM_MESSAGE_PREVIEW_CHARS)
    const lastSpace = Math.max(windowText.lastIndexOf(' '), windowText.lastIndexOf('\n'))
    const cut = lastSpace >= Math.floor(SWARM_MESSAGE_PREVIEW_CHARS * 0.68) ? lastSpace : SWARM_MESSAGE_PREVIEW_CHARS
    preview = `${sanitizeTruncatedPreview(preview.slice(0, cut))}…`
    clippedChars = true
  } else {
    preview = preview.trimEnd()
  }

  if (!clippedChars && totalLines > SWARM_MESSAGE_PREVIEW_LINES) {
    preview = `${sanitizeTruncatedPreview(preview)}…`
  }

  return {
    foldable: true,
    preview,
    totalLines,
    displayLines,
    totalChars,
  }
}

/** Filter a sorted Immortals roster by lifecycle bucket and free-text query. */
export function filterSwarmMembers(
  members: readonly SwarmMemberSnapshot[],
  filter: SwarmMemberFilter,
  query: string,
): readonly SwarmMemberSnapshot[] {
  const needle = query.trim().toLowerCase()
  return members.filter(member => {
    const isRetired = member.killed || member.state === 'killed' || member.state === 'stopped'
    if (filter === 'active' && (isRetired || member.state !== 'working')) return false
    if (filter === 'waiting' && (isRetired || member.state !== 'waiting')) return false
    if (filter === 'retired' && !isRetired) return false
    if (!needle) return true
    return (
      member.name.toLowerCase().includes(needle) ||
      member.role.toLowerCase().includes(needle) ||
      (member.group?.toLowerCase().includes(needle) ?? false) ||
      (member.added_by?.toLowerCase().includes(needle) ?? false) ||
      member.model.toLowerCase().includes(needle) ||
      member.model_display_name.toLowerCase().includes(needle) ||
      (member.current_activity?.toLowerCase().includes(needle) ?? false) ||
      member.channels.some(channel => channel.toLowerCase().includes(needle))
    )
  })
}

/** Parse `/immortals ...` (and legacy `/swarm ...`) input from the Desktop or VS Code composer. */
export function parseSwarmSlashCommand(rawInput: string): SwarmSlashAction | null {
  const trimmed = rawInput.trim()
  const match = /^\/(?:immortals|swarm)(?:\s+([\s\S]*))?$/i.exec(trimmed)
  if (!match) return null
  const rest = (match[1] ?? '').trim()
  if (!rest || /^board$/i.test(rest) || /^ui$/i.test(rest)) {
    return { kind: 'open_board' }
  }
  if (/^stop$/i.test(rest)) {
    return { kind: 'kill', target: 'all' }
  }
  const killMatch = /^kill\s+([a-zA-Z0-9_-]+)$/i.exec(rest)
  if (killMatch?.[1]) {
    return { kind: 'kill', target: killMatch[1] }
  }
  if (/^(?:status|channels)$/i.test(rest)) {
    return { kind: 'daemon_slash', command: `/immortals ${rest.toLowerCase()}` }
  }
  return { kind: 'activate_task', task: rest }
}

export interface SwarmMentionCandidate {
  readonly name: string
  readonly role: SwarmMemberRole | 'broadcast'
  readonly detail: string
}

/** Match a trailing `@partial` mention token being typed in the board composer. */
export function swarmMentionCandidates(
  draft: string,
  members: readonly SwarmMemberSnapshot[],
): readonly SwarmMentionCandidate[] {
  const match = /(?:^|\s)@([a-zA-Z0-9_-]*)$/.exec(draft)
  if (!match) return []
  const query = (match[1] ?? '').toLowerCase()
  const living = members.filter(member => !member.killed && member.state !== 'killed')
  const candidates: SwarmMentionCandidate[] = [
    { name: 'all', role: 'broadcast', detail: 'Wake every member' },
    ...living.map(member => ({
      name: member.name,
      role: member.role,
      detail: member.current_activity || member.state,
    })),
  ]
  return candidates.filter(candidate => candidate.name.toLowerCase().startsWith(query))
}

/** Replace the trailing `@partial` token in `draft` with `@<name> `. */
export function applySwarmMentionCandidate(draft: string, name: string): string {
  return draft.replace(/(^|\s)@[a-zA-Z0-9_-]*$/, `$1@${name} `)
}
