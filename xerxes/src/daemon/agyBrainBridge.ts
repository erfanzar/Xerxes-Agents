// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Read-only disk projection and live watcher bridging Antigravity ACP (`agy acp`)
 * brain directories into Xerxes's canonical Immortals Swarm, Artifacts, and
 * Subagent wire models.
 *
 * Strictly read-only: Xerxes never writes, renames, or deletes anything in an
 * Antigravity brain directory. Artifacts are shown, not edited; change them in
 * Antigravity itself.
 */

import { existsSync, readdirSync, readFileSync, statSync, watch, type FSWatcher } from 'node:fs'
import { open, readdir, readFile, realpath, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { basename, dirname, extname, isAbsolute, join, resolve } from 'node:path'

import { splitCommandString } from '../acp/externalAgents.js'
import { ValidationError } from '../core/errors.js'
import { sniffImageMediaType } from '../core/multimodal.js'
import type { JsonRpcPayload } from '../protocol/jsonRpc.js'
import {
  MAX_ARTIFACT_CONTENT_BYTES,
  MAX_ARTIFACT_TITLE_CHARS,
  summaryOf,
  type Artifact,
  type ArtifactKind,
  type ArtifactSummary,
} from '../session/artifacts.js'
import {
  COORDINATOR_NAME,
  MAIN_CHANNEL,
  SWARM_OPEN_BOARD_MARKDOWN,
  type SwarmBoardMessage,
  type SwarmBoardMessageKind,
  type SwarmChannelSummary,
  type SwarmInspectMessage,
  type SwarmLifecycleEvent,
  type SwarmMemberActivitySnapshot,
  type SwarmMemberInspectResult,
  type SwarmMemberRole,
  type SwarmMemberSnapshot,
  type SwarmMemberState,
  type SwarmOverviewSnapshot,
  type SwarmStatusSnapshot,
  type SwarmWorkerToolActivity,
  type SwarmWorkspaceMode,
} from '../swarm/types.js'
import { sessionHistoryPage } from './historyPage.js'
import type { DaemonSession } from './runtime.js'

const EXTERNAL_SESSION_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
const AGY_ARTIFACT_FILENAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._ -]{0,200}\.(?:md|markdown|html|htm|png|jpe?g|gif|webp|svg)$/i
const AGY_IMAGE_ARTIFACT_EXT_RE = /\.(?:png|jpe?g|gif|webp|svg)$/i
const MAX_AGY_IMAGE_ARTIFACT_BYTES = 4 * 1024 * 1024
const AGY_SVG_ROOT_RE = /<svg[\s>]/i
const AGY_TASK_LOG_FILENAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.log$/
const AGY_ARTIFACT_PREFIX = 'agy:'
export const AGY_TASK_ID_PREFIX = 'agy-task:'
const MAX_BG_TRANSCRIPT_BYTES = 2 * 1024 * 1024
const MAX_BG_LOG_INSPECT_CHARS = 200_000
const CHANNEL_MENTION_REGEX = /(?:^|(?<=[\s([{"',;:]))@#([a-z0-9][a-z0-9_-]{0,31})(?=$|[^a-z0-9_-])/gi
const MEMBER_MENTION_REGEX = /(?:^|(?<=[\s([{"',;:]))@([a-z0-9][a-z0-9_-]{0,23})(?=$|[^a-z0-9_-])/gi
const MONITOR_ROLE_PATTERN = /^monitor(?:-\d+)?$/i
const NARRATOR_ROLE_PATTERN = /^narrator(?:-\d+)?$/i
const SWARM_WORKER_ROLE_PATTERN = /^swarm\s+worker\b/i
const GROUP_SUFFIX_PATTERN = /^([a-z0-9][a-z0-9_-]*)-(\d+)$/i

const DEFAULT_INSPECT_LIMIT = 100
const MAX_INSPECT_LIMIT = 200
const MAX_INSPECT_TRANSCRIPT_BYTES = 1024 * 1024 // 1 MiB tail ceiling for worker inspection
const MAX_ACTIVITY_TRANSCRIPT_BYTES = 64 * 1024 // 64 KiB tail ceiling for live activity polling
const MAX_BOARD_FILE_BYTES = 8 * 1024 * 1024
const MAX_RECENT_TOOLS = 8
const MAX_RESULT_PREVIEW_CHARS = 240
const MAX_LIVE_TEXT_CHARS = 240
const MAX_INSPECT_MESSAGE_CHARS = 12_000
const MAX_ARGS_PREVIEW_CHARS = 220
const DEFAULT_WATCH_DEBOUNCE_MS = 120
const DEFAULT_WORKING_POLL_INTERVAL_MS = 1500
const MAX_SUBAGENT_HISTORY_TEXT_CHARS = 32_000

export interface AgyBrainLocation {
  readonly brainRoot: string
  readonly externalSessionId: string
  readonly logsDir: string
  readonly messagesDir: string
  readonly sessionDir: string
  readonly subagentsDir: string
  readonly swarmDir: string
  readonly tasksDir: string
}

export interface AgyBackgroundTaskSnapshot {
  readonly id: string
  readonly rawTaskId: string
  readonly kind: 'shell' | 'schedule' | 'task'
  readonly title: string
  readonly label: string
  readonly detail: string
  readonly state: 'running' | 'scheduled' | 'completed' | 'failed' | 'cancelled'
  readonly startedAt: number | null
  readonly endedAt: number | null
  readonly exitCode: number | null
  readonly nextRunAt: string | null
  readonly durationSeconds?: number
  readonly cronExpression?: string
  readonly isDaemon?: boolean
  readonly logFileName?: string
  readonly outputPreview?: string
}

interface CachedBackgroundTasksEntry {
  readonly mtimeMs: number
  readonly size: number
  readonly taskFilesSig: string
  readonly tasks: readonly AgyBackgroundTaskSnapshot[]
}

export interface ResolveAgyBrainOptions {
  readonly bindingCommand?: string
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly externalSessionId: string
  readonly homeDir?: string
}

export interface AgyBrainBridgeOptions {
  readonly debounceMs?: number
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly homeDir?: string
  readonly pollIntervalMs?: number
}

export interface AgyWatchSessionParams {
  readonly bindingCommand?: string
  readonly externalSessionId: string
  readonly modelDisplayMap?: ReadonlyMap<string, string>
  readonly onBackgroundChange?: () => void
  readonly onSwarmEvent: (event: SwarmLifecycleEvent, workspaceDir: string) => void
  readonly sessionId: string
  readonly workspaceDir: string
}

export interface AgyReadBoardOptions {
  readonly channel?: string
  readonly limit?: number
  readonly seqs?: readonly number[]
  readonly since?: number
}

export interface AgyInspectMemberOptions {
  readonly limit?: number
  readonly modelDisplayMap?: ReadonlyMap<string, string>
}

export interface AgySubagentInspectResult {
  readonly agent: JsonRpcPayload
  readonly history?: JsonRpcPayload
  readonly history_session_id?: string
}

interface ParsedTranscriptStep {
  readonly cacheReadTokens: number
  readonly content: string
  readonly createdAt: string
  readonly error?: string
  readonly inputTokens: number
  readonly model?: string
  readonly outputTokens: number
  readonly source: string
  readonly status: string
  readonly stepIndex: number
  readonly thinking: string
  readonly toolCalls: ReadonlyArray<{
    readonly args: unknown
    readonly name: string
  }>
  readonly type: string
}

interface TranscriptAnalysis {
  readonly activity: SwarmMemberActivitySnapshot
  readonly apiCalls: number
  readonly currentActivity?: string
  readonly firstCreatedAt?: string
  readonly inputTokens: number
  readonly lastAssistantText?: string
  readonly lastCreatedAt?: string
  readonly lastError?: string
  readonly lastModel?: string
  readonly lastUserPrompt?: string
  readonly messages: readonly SwarmInspectMessage[]
  readonly outputTokens: number
  readonly toolCount: number
  readonly totalMessages: number
}

interface CachedTranscriptEntry {
  readonly analysis: TranscriptAnalysis
  readonly maxBytes: number
  readonly mtimeMs: number
  readonly size: number
}

interface RawAgyMember {
  readonly added_by?: string
  readonly applied_model?: string
  readonly channels?: readonly string[]
  readonly conversation_id?: string
  readonly desired_model?: string
  readonly desired_state?: string
  readonly error?: string
  readonly group?: string
  readonly killed?: boolean
  readonly last_read_seq?: number
  readonly model?: string
  readonly model_display_name?: string
  readonly phase?: string
  readonly role?: string
  readonly wait_since?: number
  readonly wake_mode?: string
  readonly workspace_mode?: string
  readonly workspace_path?: string
}

interface ParsedSwarmDiskState {
  readonly activitiesByMember: ReadonlyMap<string, SwarmMemberActivitySnapshot>
  readonly exists: boolean
  readonly messages: readonly SwarmBoardMessage[]
  readonly narratorConvId?: string
  readonly status: SwarmStatusSnapshot
}

interface WatchedSessionState {
  debounceTimer: ReturnType<typeof setTimeout> | undefined
  lastActivities: ReadonlyMap<string, string>
  lastBackgroundSig: string
  lastBoardSeq: number
  lastSnapshot: SwarmStatusSnapshot | undefined
  readonly location: AgyBrainLocation
  logsWatchDir: string | undefined
  logsWatcher: FSWatcher | undefined
  modelDisplayMap: ReadonlyMap<string, string> | undefined
  narratorWatchDir: string | undefined
  narratorWatcher: FSWatcher | undefined
  onBackgroundChange: (() => void) | undefined
  readonly onSwarmEvent: (event: SwarmLifecycleEvent, workspaceDir: string) => void
  pollTimer: ReturnType<typeof setInterval> | undefined
  refreshing: boolean
  refreshQueued: boolean
  readonly sessionId: string
  swarmWatchDir: string | undefined
  swarmWatcher: FSWatcher | undefined
  tasksWatchDir: string | undefined
  tasksWatcher: FSWatcher | undefined
  readonly workspaceDir: string
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : undefined
}

function readFiniteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function expandHomePath(raw: string, homeDir: string): string {
  const trimmed = raw.trim()
  if (!trimmed) return trimmed
  if (trimmed === '~') return homeDir
  if (trimmed.startsWith('~/') || trimmed.startsWith('~\\')) {
    return join(homeDir, trimmed.slice(2))
  }
  return trimmed
}

/** Extract `--gemini-dir` and `--app-data-dir` overrides from a binding command string. */
export function parseAgyDirFlags(command: string | undefined): {
  readonly appDataDir?: string
  readonly geminiDir?: string
} {
  if (!command || !command.trim()) return {}
  const tokens = splitCommandString(command)
  let geminiDir: string | undefined
  let appDataDir: string | undefined
  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i] ?? ''
    if (token === '--gemini-dir' && i + 1 < tokens.length) {
      geminiDir = tokens[i + 1]?.trim() || undefined
      i += 1
    } else if (token.startsWith('--gemini-dir=')) {
      geminiDir = token.slice('--gemini-dir='.length).trim() || undefined
    } else if (token === '--app-data-dir' && i + 1 < tokens.length) {
      appDataDir = tokens[i + 1]?.trim() || undefined
      i += 1
    } else if (token.startsWith('--app-data-dir=')) {
      appDataDir = token.slice('--app-data-dir='.length).trim() || undefined
    }
  }
  return {
    ...(geminiDir ? { geminiDir } : {}),
    ...(appDataDir ? { appDataDir } : {}),
  }
}

/**
 * Resolve the Antigravity brain root and session paths for a given external session ID.
 * Honors `--gemini-dir` / `--app-data-dir` flags if present in `bindingCommand`,
 * defaulting to `~/.gemini/antigravity-cli/brain/<extId>` with fallback to
 * `~/.gemini/antigravity/brain/<extId>`.
 */
export function resolveAgyBrainLocation(options: ResolveAgyBrainOptions): AgyBrainLocation | undefined {
  const externalSessionId = options.externalSessionId.trim()
  if (!externalSessionId || !EXTERNAL_SESSION_ID_PATTERN.test(externalSessionId)) {
    return undefined
  }
  if (externalSessionId === '.' || externalSessionId === '..' || externalSessionId.includes('..')) {
    return undefined
  }

  const homeDir = resolve(options.homeDir ?? options.env?.HOME ?? homedir())
  const flags = parseAgyDirFlags(options.bindingCommand)

  const candidates: string[] = []
  const pushUnique = (dir: string): void => {
    const normalized = resolve(dir)
    if (!candidates.includes(normalized)) {
      candidates.push(normalized)
    }
  }

  if (flags.geminiDir || flags.appDataDir) {
    const geminiRaw = expandHomePath(flags.geminiDir ?? '.gemini', homeDir)
    const geminiBase = isAbsolute(geminiRaw) ? geminiRaw : join(homeDir, geminiRaw)
    const appRaw = expandHomePath(flags.appDataDir ?? 'antigravity-cli', homeDir)
    const appBase = isAbsolute(appRaw) ? appRaw : join(geminiBase, appRaw)
    const explicitRoot = resolve(join(appBase, 'brain'))
    const sessionDir = join(explicitRoot, externalSessionId)
    return {
      brainRoot: explicitRoot,
      externalSessionId,
      sessionDir,
      swarmDir: join(sessionDir, '.system_generated', 'swarm'),
      subagentsDir: join(sessionDir, '.system_generated', 'subagents'),
      logsDir: join(sessionDir, '.system_generated', 'logs'),
      messagesDir: join(sessionDir, '.system_generated', 'messages'),
      tasksDir: join(sessionDir, '.system_generated', 'tasks'),
    }
  }

  pushUnique(join(homeDir, '.gemini', 'antigravity-cli', 'brain'))
  pushUnique(join(homeDir, '.gemini', 'antigravity', 'brain'))

  // Prefer a candidate that already contains `.system_generated/swarm`, then any existing session dir.
  let chosenRoot: string | undefined
  for (const root of candidates) {
    if (existsSync(join(root, externalSessionId, '.system_generated', 'swarm'))) {
      chosenRoot = root
      break
    }
  }
  if (!chosenRoot) {
    for (const root of candidates) {
      if (existsSync(join(root, externalSessionId))) {
        chosenRoot = root
        break
      }
    }
  }
  const brainRoot = chosenRoot ?? candidates[0]!
  const sessionDir = join(brainRoot, externalSessionId)
  return {
    brainRoot,
    externalSessionId,
    sessionDir,
    swarmDir: join(sessionDir, '.system_generated', 'swarm'),
    subagentsDir: join(sessionDir, '.system_generated', 'subagents'),
    logsDir: join(sessionDir, '.system_generated', 'logs'),
    messagesDir: join(sessionDir, '.system_generated', 'messages'),
    tasksDir: join(sessionDir, '.system_generated', 'tasks'),
  }
}

/** Parse and validate an `agy:<filename>` artifact ID into a safe basename (never allows traversal). */
export function parseAgyArtifactFilename(id: string): string | undefined {
  const trimmed = id.trim()
  if (!trimmed.startsWith(AGY_ARTIFACT_PREFIX)) return undefined
  const filename = trimmed.slice(AGY_ARTIFACT_PREFIX.length)
  if (!isValidAgyArtifactFilename(filename)) return undefined
  return filename
}

/** True if a basename is a valid top-level Antigravity artifact file (`.md`/`.html`, no path separators). */
export function isValidAgyArtifactFilename(filename: string): boolean {
  if (!filename || filename.includes('/') || filename.includes('\\') || filename.includes('\0')) {
    return false
  }
  if (filename === '.' || filename === '..' || filename.startsWith('.') || filename.includes('..')) {
    return false
  }
  return AGY_ARTIFACT_FILENAME_PATTERN.test(filename)
}

/** Resolve an `agy:<filename>` ID if `rawPath` points to a top-level artifact file inside `location.sessionDir`. */
export function artifactIdForAgyPath(location: AgyBrainLocation, rawPath: string): string | undefined {
  const trimmed = rawPath.trim()
  if (!trimmed) return undefined
  const resolvedPath = resolve(trimmed)
  const expectedDir = resolve(location.sessionDir)
  if (dirname(resolvedPath) !== expectedDir) return undefined
  const file = basename(resolvedPath)
  if (!isValidAgyArtifactFilename(file)) return undefined
  return `${AGY_ARTIFACT_PREFIX}${file}`
}

/** Safely extract `@mentions`, `@#channel` mentions, and `@all` broadcast from external AGY board text. */
export function parseExternalBoardMentions(text: string): {
  readonly broadcast: boolean
  readonly channel_mentions: string[]
  readonly mentions: string[]
} {
  const channelMentions: string[] = []
  const seenChannels = new Set<string>()
  for (const match of text.matchAll(CHANNEL_MENTION_REGEX)) {
    const ch = match[1]?.toLowerCase()
    if (ch && !seenChannels.has(ch)) {
      seenChannels.add(ch)
      channelMentions.push(ch)
    }
  }

  let broadcast = false
  const mentions: string[] = []
  const seenMentions = new Set<string>()
  for (const match of text.matchAll(MEMBER_MENTION_REGEX)) {
    const token = match[1]?.toLowerCase()
    if (!token) continue
    if (token === 'all') {
      broadcast = true
      continue
    }
    if (!seenMentions.has(token)) {
      seenMentions.add(token)
      mentions.push(token)
    }
  }

  return { mentions, channel_mentions: channelMentions, broadcast }
}

function inferAgyMemberRole(name: string, rawRole?: string): SwarmMemberRole {
  if (name === COORDINATOR_NAME || rawRole === 'coordinator') return 'coordinator'
  if (NARRATOR_ROLE_PATTERN.test(name) || rawRole === 'narrator') return 'narrator'
  if (MONITOR_ROLE_PATTERN.test(name) || rawRole === 'monitor') return 'monitor'
  return 'worker'
}

function mapAgyMemberState(raw: RawAgyMember): {
  readonly killed: boolean
  readonly mentions_only?: boolean
  readonly parked_since?: number
  readonly state: SwarmMemberState
} {
  const phase = raw.phase?.trim().toUpperCase() ?? ''
  const desiredState = raw.desired_state?.trim().toUpperCase() ?? ''
  const wakeMode = raw.wake_mode?.trim().toLowerCase() ?? ''

  if (raw.killed === true || phase === 'KILLED') {
    return { state: 'killed', killed: true }
  }
  if (phase === 'STOPPED' || desiredState === 'STOPPED') {
    return { state: 'stopped', killed: false }
  }
  if (wakeMode === 'working') {
    return { state: 'working', killed: false }
  }
  if (wakeMode === 'mentions') {
    const waitSince = readFiniteNumber(raw.wait_since)
    return {
      state: 'waiting',
      killed: false,
      mentions_only: true,
      ...(waitSince !== undefined ? { parked_since: Math.max(0, Math.trunc(waitSince)) } : {}),
    }
  }
  if (wakeMode === 'all') {
    const waitSince = readFiniteNumber(raw.wait_since)
    return {
      state: 'waiting',
      killed: false,
      mentions_only: false,
      ...(waitSince !== undefined ? { parked_since: Math.max(0, Math.trunc(waitSince)) } : {}),
    }
  }
  return { state: phase === 'RUNNING' && !wakeMode ? 'working' : 'waiting', killed: false }
}

function normalizeWorkspaceMode(raw?: string): SwarmWorkspaceMode {
  return raw === 'share' || raw === 'branch' ? raw : 'inherit'
}

function normalizeBoardKind(raw: unknown): SwarmBoardMessageKind | undefined {
  if (raw === 'brief' || raw === 'message' || raw === 'feedback' || raw === 'system') {
    return raw
  }
  return undefined
}

async function readJsonFileSafe(filePath: string): Promise<Record<string, unknown> | undefined> {
  try {
    const raw = await readFile(filePath, 'utf8')
    const parsed: unknown = JSON.parse(raw)
    return isRecord(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

function readJsonFileSafeSync(filePath: string): Record<string, unknown> | undefined {
  try {
    const raw = readFileSync(filePath, 'utf8')
    const parsed: unknown = JSON.parse(raw)
    return isRecord(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

async function readTailBoundedUtf8(filePath: string, maxBytes: number): Promise<string | undefined> {
  try {
    const stats = await stat(filePath)
    if (!stats.isFile() || stats.size === 0) return ''
    if (stats.size <= maxBytes) {
      return await readFile(filePath, 'utf8')
    }
    const startOffset = stats.size - maxBytes
    const handle = await open(filePath, 'r')
    try {
      const buf = Buffer.alloc(maxBytes)
      const { bytesRead } = await handle.read(buf, 0, maxBytes, startOffset)
      const raw = buf.subarray(0, bytesRead).toString('utf8')
      // Drop the first partial NDJSON line when slicing mid-file.
      const firstNewline = raw.indexOf('\n')
      return firstNewline >= 0 ? raw.slice(firstNewline + 1) : ''
    } finally {
      await handle.close()
    }
  } catch {
    return undefined
  }
}

function readTailBoundedUtf8Sync(filePath: string, maxBytes: number): string | undefined {
  try {
    const stats = statSync(filePath)
    if (!stats.isFile() || stats.size === 0) return ''
    if (stats.size <= maxBytes) {
      return readFileSync(filePath, 'utf8')
    }
    // For sync fast-path on large files, slice from Buffer via readFileSync only if reasonably bounded,
    // otherwise fall back to returning undefined so async inspect handles huge files cleanly.
    if (stats.size > 4 * 1024 * 1024) return undefined
    const buf = readFileSync(filePath)
    const raw = buf.subarray(buf.length - maxBytes).toString('utf8')
    const firstNewline = raw.indexOf('\n')
    return firstNewline >= 0 ? raw.slice(firstNewline + 1) : ''
  } catch {
    return undefined
  }
}

function parseBoardMessagesFromText(raw: string): SwarmBoardMessage[] {
  const bySeq = new Map<number, SwarmBoardMessage>()
  for (const line of raw.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed) continue
    let parsed: unknown
    try {
      parsed = JSON.parse(trimmed)
    } catch {
      continue
    }
    if (!isRecord(parsed)) continue
    const seq = readFiniteNumber(parsed.seq)
    const agent = readString(parsed.agent)
    const text = typeof parsed.text === 'string' ? parsed.text : undefined
    if (seq === undefined || seq < 1 || !agent || text === undefined) continue
    const seqInt = Math.trunc(seq)
    const channel = readString(parsed.channel) ?? MAIN_CHANNEL
    const time = readString(parsed.time) ?? ''
    const kind = normalizeBoardKind(parsed.kind)
    const parsedMentions = parseExternalBoardMentions(text)
    bySeq.set(seqInt, {
      seq: seqInt,
      agent,
      channel,
      text,
      mentions: parsedMentions.mentions,
      channel_mentions: parsedMentions.channel_mentions,
      broadcast: parsedMentions.broadcast,
      time,
      ...(kind ? { kind } : {}),
    })
  }
  return [...bySeq.values()].sort((left, right) => left.seq - right.seq)
}

function renderOverviewValue(value: unknown, depth = 0): string {
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (Array.isArray(value)) {
    const indent = '  '.repeat(depth)
    const lines: string[] = []
    for (const item of value) {
      if (typeof item === 'string' && item.trim()) {
        lines.push(`${indent}- ${item.trim()}`)
      } else if (isRecord(item)) {
        const title = readString(item.title) ?? readString(item.name) ?? readString(item.id) ?? readString(item.label)
        const status = readString(item.status) ?? readString(item.state) ?? readString(item.phase)
        const detail =
          readString(item.summary) ??
          readString(item.line) ??
          readString(item.detail) ??
          readString(item.description) ??
          readString(item.text) ??
          readString(item.goal)
        const owner = readString(item.owner) ?? readString(item.agent)
        const headParts: string[] = []
        if (title) headParts.push(`**${title}**`)
        if (status) headParts.push(`[${status}]`)
        if (owner) headParts.push(`(@${owner})`)
        if (detail) headParts.push(headParts.length > 0 ? `— ${detail}` : detail)
        if (headParts.length > 0) {
          lines.push(`${indent}- ${headParts.join(' ')}`)
        } else {
          const kv = Object.entries(item)
            .filter(([, v]) => typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean')
            .map(([k, v]) => `${k}: ${String(v)}`)
            .join(', ')
          if (kv) lines.push(`${indent}- ${kv}`)
        }
      }
    }
    return lines.join('\n')
  }
  if (isRecord(value)) {
    const title = readString(value.title)
    const line = readString(value.line) ?? readString(value.summary) ?? readString(value.text)
    if (title && line) return `**${title}** — ${line}`
    if (title) return `**${title}**`
    if (line) return line
    const entries = Object.entries(value)
      .map(([k, v]) => {
        const rendered = renderOverviewValue(v, depth + 1)
        return rendered ? `- **${k}**: ${rendered}` : ''
      })
      .filter(Boolean)
    return entries.join('\n')
  }
  return ''
}

function synthesizeSwarmOverview(
  view: Record<string, unknown> | undefined,
  plan: Record<string, unknown> | undefined,
  narratorName?: string,
): SwarmOverviewSnapshot | null {
  const sections: string[] = []
  let summary: string | undefined
  let updatedAt: string | undefined
  let author: string | undefined

  if (view) {
    author = narratorName ?? 'narrator'
    updatedAt = readString(view.updated_at)
    const headline = readString(view.headline)
    const phase = readString(view.phase)
    const goal = renderOverviewValue(view.goal)
    const now = renderOverviewValue(view.now)
    const runningSummary = renderOverviewValue(view.running_summary)
    const attention = renderOverviewValue(view.attention)
    const workstreams = renderOverviewValue(view.workstreams)
    const threads = renderOverviewValue(view.threads)
    const milestones = renderOverviewValue(view.milestones)
    const risks = renderOverviewValue(view.risks)
    const feedbackAck = renderOverviewValue(view.feedback_ack)

    summary = headline ?? (typeof view.running_summary === 'string' ? view.running_summary.trim() : undefined)
    if (headline) {
      sections.push(`## ${headline}${phase ? ` (${phase})` : ''}`)
    } else if (phase) {
      sections.push(`**Phase:** ${phase}`)
    }
    if (goal) sections.push(`### Goal\n${goal}`)
    if (now) sections.push(`### Now\n${now}`)
    if (runningSummary) sections.push(`### Summary\n${runningSummary}`)
    if (attention) sections.push(`### Attention\n${attention}`)
    if (workstreams) sections.push(`### Workstreams\n${workstreams}`)
    if (threads) sections.push(`### Threads\n${threads}`)
    if (milestones) sections.push(`### Milestones\n${milestones}`)
    if (risks) sections.push(`### Risks\n${risks}`)
    if (feedbackAck) sections.push(`### Feedback\n${feedbackAck}`)
  }

  if (plan && sections.length === 0) {
    author = author ?? COORDINATOR_NAME
    const goalObj = isRecord(plan.goal) ? plan.goal : undefined
    const goalTitle = readString(goalObj?.title)
    const goalLine = readString(goalObj?.line) ?? (typeof plan.goal === 'string' ? plan.goal.trim() : undefined)
    summary = summary ?? goalTitle ?? goalLine
    if (goalTitle) sections.push(`## ${goalTitle}`)
    if (goalLine) sections.push(goalLine)
    const tasks = isRecord(plan.tasks) ? plan.tasks : undefined
    if (tasks) {
      const taskLines = Object.entries(tasks)
        .filter(([, v]) => typeof v === 'string' && v.trim())
        .map(([name, desc]) => `- **@${name}**: ${String(desc).trim()}`)
      if (taskLines.length > 0) {
        sections.push(`### Tasks\n${taskLines.join('\n')}`)
      }
    }
    const threads = renderOverviewValue(plan.threads)
    if (threads) {
      sections.push(`### Threads\n${threads}`)
    }
  }

  const markdown = sections.join('\n\n').trim()
  if (!markdown && !summary) return null
  return {
    markdown: markdown || summary || '',
    ...(summary ? { summary: summary.slice(0, 280) } : {}),
    ...(updatedAt ? { updated_at: updatedAt } : {}),
    ...(author ? { author } : {}),
  }
}

function unwrapDoubleEncodedString(value: unknown): unknown {
  if (typeof value !== 'string') return value
  const trimmed = value.trim()
  if (trimmed.length >= 2 && trimmed.startsWith('"') && trimmed.endsWith('"')) {
    try {
      const parsed: unknown = JSON.parse(trimmed)
      if (typeof parsed === 'string') return parsed
    } catch {
      return value
    }
  }
  return value
}

function formatArgsPreview(rawArgs: unknown): string {
  if (rawArgs === undefined || rawArgs === null) return ''
  if (typeof rawArgs === 'string') {
    const cleaned = String(unwrapDoubleEncodedString(rawArgs)).replace(/\s+/g, ' ').trim()
    return cleaned.length > MAX_ARGS_PREVIEW_CHARS
      ? `${cleaned.slice(0, MAX_ARGS_PREVIEW_CHARS - 1)}…`
      : cleaned
  }
  if (isRecord(rawArgs)) {
    const normalized: Record<string, unknown> = {}
    for (const [key, val] of Object.entries(rawArgs)) {
      normalized[key] = unwrapDoubleEncodedString(val)
    }
    const preferred =
      readString(normalized.toolSummary) ??
      readString(normalized.CommandLine) ??
      readString(normalized.Path) ??
      readString(normalized.path) ??
      readString(normalized.query) ??
      readString(normalized.action)
    if (preferred) {
      const oneLine = preferred.replace(/\s+/g, ' ').trim()
      return oneLine.length > MAX_ARGS_PREVIEW_CHARS
        ? `${oneLine.slice(0, MAX_ARGS_PREVIEW_CHARS - 1)}…`
        : oneLine
    }
    try {
      const serialized = JSON.stringify(normalized)
      return serialized.length > MAX_ARGS_PREVIEW_CHARS
        ? `${serialized.slice(0, MAX_ARGS_PREVIEW_CHARS - 1)}…`
        : serialized
    } catch {
      return ''
    }
  }
  return String(rawArgs).slice(0, MAX_ARGS_PREVIEW_CHARS)
}

const LEADING_AGY_TOOL_TIMESTAMPS_RE = /^(?:\s*(?:Created|Completed)\s+At:\s*[^\r\n]*(?:\r?\n|$))+/i

function stripAgyToolTimestamps(content: string): string {
  return content.replace(LEADING_AGY_TOOL_TIMESTAMPS_RE, '').trim()
}

function stripSystemMessageEnvelope(content: string): string {
  const trimmed = content.trim()
  if (!trimmed) return ''
  let inner: string | undefined
  const match = /<SYSTEM_MESSAGE>\s*([\s\S]*?)\s*<\/SYSTEM_MESSAGE>/i.exec(trimmed)
  if (match) {
    inner = (match[1] ?? '').trim()
  } else if (/^<SYSTEM_MESSAGE>/i.test(trimmed)) {
    inner = trimmed.replace(/^<SYSTEM_MESSAGE>\s*/i, '').replace(/\s*<\/SYSTEM_MESSAGE>\s*$/i, '').trim()
  }
  if (inner === undefined) return trimmed
  const contentIdx = inner.indexOf('content=')
  if (contentIdx >= 0 && /^\[Message\]/i.test(inner)) {
    return inner.slice(contentIdx + 'content='.length).trim()
  }
  return inner
}

function clipText(value: string, maxChars: number): string {
  if (value.length <= maxChars) return value
  return `${value.slice(0, Math.max(1, maxChars - 1))}…`
}

function firstHeadingOrLine(text: string): string | undefined {
  for (const rawLine of text.split(/\r?\n/)) {
    const cleaned = rawLine.trim().replace(/^#+\s*/, '').replace(/^\*\*|\*\*$/g, '').trim()
    if (cleaned) return clipText(cleaned, 100)
  }
  return undefined
}

function parseTranscriptSteps(raw: string): ParsedTranscriptStep[] {
  const steps: ParsedTranscriptStep[] = []
  let fallbackIdx = 0
  for (const line of raw.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed) continue
    let parsed: unknown
    try {
      parsed = JSON.parse(trimmed)
    } catch {
      continue
    }
    if (!isRecord(parsed)) continue
    const stepIndex = readFiniteNumber(parsed.step_index) ?? fallbackIdx
    fallbackIdx = stepIndex + 1
    const rawToolCalls = Array.isArray(parsed.tool_calls) ? parsed.tool_calls : []
    const toolCalls: Array<{ readonly args: unknown; readonly name: string }> = []
    for (const item of rawToolCalls) {
      if (!isRecord(item)) continue
      const name = readString(item.name)
      if (!name) continue
      toolCalls.push({ name, args: item.args })
    }
    const errStr = readString(parsed.error)
    const modelStr = readString(parsed.model)
    steps.push({
      stepIndex,
      source: readString(parsed.source) ?? '',
      type: readString(parsed.type) ?? '',
      status: (readString(parsed.status) ?? 'DONE').toUpperCase(),
      createdAt: readString(parsed.created_at) ?? '',
      ...(modelStr ? { model: modelStr } : {}),
      inputTokens: Math.max(0, Math.trunc(readFiniteNumber(parsed.input_tokens) ?? 0)),
      cacheReadTokens: Math.max(0, Math.trunc(readFiniteNumber(parsed.cache_read_tokens) ?? 0)),
      outputTokens: Math.max(0, Math.trunc(readFiniteNumber(parsed.output_tokens) ?? 0)),
      thinking: typeof parsed.thinking === 'string' ? parsed.thinking : '',
      content: typeof parsed.content === 'string' ? parsed.content : '',
      toolCalls,
      ...(errStr ? { error: errStr } : {}),
    })
  }
  return steps
}

function analyzeTranscriptSteps(
  steps: readonly ParsedTranscriptStep[],
  memberState: SwarmMemberState,
): TranscriptAnalysis {
  const messages: SwarmInspectMessage[] = []
  const recentTools: SwarmWorkerToolActivity[] = []
  const pendingTools: Array<{
    readonly id: string
    readonly name: string
    readonly startedAt: string
    readonly summary?: string
  }> = []

  let inputTokens = 0
  let outputTokens = 0
  let apiCalls = 0
  let toolCount = 0
  let lastModel: string | undefined
  let lastAssistantText: string | undefined
  let lastUserPrompt: string | undefined
  let lastNonSystemError: string | undefined
  let lastCreatedAt: string | undefined
  let firstCreatedAt: string | undefined
  let lastStep: ParsedTranscriptStep | undefined

  for (const step of steps) {
    if (step.createdAt) {
      if (!firstCreatedAt) firstCreatedAt = step.createdAt
      lastCreatedAt = step.createdAt
    }
    if (step.model) lastModel = step.model
    inputTokens += step.inputTokens + step.cacheReadTokens
    outputTokens += step.outputTokens

    if (step.type === 'USER_INPUT' || step.type === 'SYSTEM_MESSAGE') {
      const cleaned = stripSystemMessageEnvelope(step.content)
      if (!cleaned) continue
      if (step.type === 'USER_INPUT' || !lastUserPrompt) {
        if (!cleaned.startsWith('[swarm] new board message #')) {
          lastUserPrompt = cleaned
        }
      }
      // Avoid flooding inspector with bare "[swarm] new board message #N" wakeup pings when verbose.
      if (step.type === 'SYSTEM_MESSAGE' && /^\[swarm\] new board message #\d+$/i.test(cleaned)) {
        continue
      }
      messages.push({
        index: messages.length,
        role: 'user',
        text: clipText(cleaned, MAX_INSPECT_MESSAGE_CHARS),
      })
      continue
    }

    if (step.type === 'PLANNER_RESPONSE') {
      apiCalls += 1
      lastStep = step
      lastNonSystemError = step.status === 'ERROR' && step.error ? clipText(step.error, 240) : undefined
      // Clear any orphaned pending tools from an earlier planner response before recording new calls.
      pendingTools.length = 0
      const formattedCalls = step.toolCalls.map((call, idx) => {
        const id = `step-${step.stepIndex}-tc-${idx + 1}`
        const argsPreview = formatArgsPreview(call.args)
        toolCount += 1
        pendingTools.push({
          id,
          name: call.name,
          ...(argsPreview ? { summary: argsPreview } : {}),
          startedAt: step.createdAt,
        })
        return {
          id,
          name: call.name,
          args_preview: argsPreview,
        }
      })
      const displayText = step.content.trim()
        || (step.thinking.trim() ? step.thinking.trim() : '')
      if (step.content.trim()) {
        lastAssistantText = step.content.trim()
      }
      if (displayText || formattedCalls.length > 0) {
        messages.push({
          index: messages.length,
          role: 'assistant',
          text: clipText(displayText, MAX_INSPECT_MESSAGE_CHARS),
          ...(formattedCalls.length > 0 ? { tool_calls: formattedCalls } : {}),
        })
      }
      continue
    }

    // GENERIC or other tool result step
    lastStep = step
    const matched = pendingTools.shift()
    const isErr = step.status === 'ERROR' || Boolean(step.error)
    const rawResult = stripAgyToolTimestamps(step.error || step.content || '')
    const resultPreview = rawResult ? clipText(rawResult.replace(/\s+/g, ' ').trim(), MAX_RESULT_PREVIEW_CHARS) : undefined
    if (isErr && rawResult) {
      lastNonSystemError = clipText(rawResult.replace(/\s+/g, ' ').trim(), 240)
    } else if (!isErr) {
      lastNonSystemError = undefined
    }
    const toolName = matched?.name ?? 'tool'
    const toolId = matched?.id ?? `step-${step.stepIndex}`
    recentTools.push({
      id: toolId,
      name: toolName,
      ...(matched?.summary ? { summary: matched.summary } : {}),
      status: isErr ? 'error' : 'done',
      started_at: matched?.startedAt || step.createdAt,
      ...(step.createdAt ? { finished_at: step.createdAt } : {}),
      ...(resultPreview ? { result_preview: resultPreview } : {}),
    })
    if (recentTools.length > MAX_RECENT_TOOLS) {
      recentTools.splice(0, recentTools.length - MAX_RECENT_TOOLS)
    }
    messages.push({
      index: messages.length,
      role: 'tool',
      tool_name: toolName,
      text: clipText(rawResult.trim() || (isErr ? 'Tool failed' : '(empty output)'), MAX_INSPECT_MESSAGE_CHARS),
    })
  }

  const activeTools: SwarmWorkerToolActivity[] = memberState === 'working'
    ? pendingTools.map(item => ({
        id: item.id,
        name: item.name,
        ...(item.summary ? { summary: item.summary } : {}),
        status: 'running',
        started_at: item.startedAt,
      }))
    : []

  let phase: SwarmMemberActivitySnapshot['phase'] = 'idle'
  let summary: string | undefined
  let liveTextPreview: string | undefined

  if (memberState === 'working') {
    const runningTool = activeTools.at(-1)
    if (runningTool) {
      phase = 'tool'
      summary = runningTool.summary
        ? `${runningTool.name}: ${clipText(runningTool.summary, 90)}`
        : `Running ${runningTool.name}…`
    } else if (lastStep?.type === 'PLANNER_RESPONSE') {
      if (lastStep.content.trim()) {
        phase = 'responding'
        summary = 'Responding…'
        liveTextPreview = lastStep.content.trim().slice(-MAX_LIVE_TEXT_CHARS)
      } else if (lastStep.thinking.trim()) {
        phase = 'thinking'
        summary = firstHeadingOrLine(lastStep.thinking) ?? 'Thinking…'
      } else {
        phase = 'thinking'
        summary = 'Working…'
      }
    } else {
      const lastRecent = recentTools.at(-1)
      phase = 'thinking'
      summary = lastRecent
        ? `After ${lastRecent.name}…`
        : 'Working…'
    }
  }

  const activity: SwarmMemberActivitySnapshot = {
    phase,
    ...(summary ? { summary } : {}),
    active_tools: activeTools,
    recent_tools: recentTools,
    ...(liveTextPreview ? { live_text_preview: liveTextPreview } : {}),
    ...(lastCreatedAt ? { updated_at: lastCreatedAt } : {}),
  }

  return {
    activity,
    ...(summary ? { currentActivity: summary } : {}),
    messages,
    totalMessages: messages.length,
    inputTokens,
    outputTokens,
    apiCalls,
    toolCount,
    ...(lastModel ? { lastModel } : {}),
    ...(lastAssistantText ? { lastAssistantText } : {}),
    ...(lastUserPrompt ? { lastUserPrompt } : {}),
    ...(lastNonSystemError ? { lastError: lastNonSystemError } : {}),
    ...(firstCreatedAt ? { firstCreatedAt } : {}),
    ...(lastCreatedAt ? { lastCreatedAt } : {}),
  }
}

function activitySignatureOf(activity: SwarmMemberActivitySnapshot): string {
  const active = activity.active_tools.map(item => `${item.id}:${item.status}`).join(',')
  const lastRecent = activity.recent_tools.at(-1)
  const recent = lastRecent ? `${activity.recent_tools.length}:${lastRecent.id}:${lastRecent.status}` : '0'
  return `${activity.phase}|${activity.summary ?? ''}|${activity.live_text_preview ?? ''}|${active}|${recent}`
}

function memberSignatureOf(member: SwarmMemberSnapshot): string {
  return JSON.stringify({
    state: member.state,
    killed: member.killed,
    model: member.model,
    workspace: member.workspace,
    last_read_seq: member.last_read_seq,
    last_post_seq: member.last_post_seq,
    parked_since: member.parked_since,
    mentions_only: member.mentions_only,
    error: member.error,
    channels: member.channels,
  })
}

function humanizeArtifactTitle(filename: string, content: string, kind: ArtifactKind): string {
  if (kind === 'markdown') {
    for (const rawLine of content.split(/\r?\n/).slice(0, 20)) {
      const trimmed = rawLine.trim()
      if (!trimmed) continue
      const heading = /^#\s+(.+)$/.exec(trimmed)
      if (heading && heading[1]?.trim()) {
        return heading[1].trim().slice(0, MAX_ARTIFACT_TITLE_CHARS)
      }
    }
  } else if (kind === 'html') {
    const match = /<title[^>]*>([^<]+)<\/title>/i.exec(content.slice(0, 4096))
    if (match && match[1]?.trim()) {
      return match[1].trim().slice(0, MAX_ARTIFACT_TITLE_CHARS)
    }
  }
  const stem = filename.replace(/\.[^.]+$/, '')
  const words = stem
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
  if (!words) return filename
  return words.replace(/\b[a-z]/g, char => char.toUpperCase()).slice(0, MAX_ARTIFACT_TITLE_CHARS)
}

function artifactKindFromFilename(filename: string): ArtifactKind {
  const ext = extname(filename).toLowerCase()
  if (ext === '.html' || ext === '.htm') return 'html'
  if (AGY_IMAGE_ARTIFACT_EXT_RE.test(ext)) return 'image'
  return 'markdown'
}

function normalizeIsoTimestamp(value: unknown, fallback: string): string {
  if (typeof value === 'string' && value.trim()) {
    const ms = Date.parse(value)
    if (Number.isFinite(ms)) return new Date(ms).toISOString()
  }
  return fallback
}

function mapSubagentState(rawState: string | undefined): 'running' | 'completed' | 'failed' | 'cancelled' {
  const normalized = (rawState ?? '').trim().toUpperCase()
  if (normalized.includes('ALIVE') || normalized.includes('RUNNING') || normalized.includes('ACTIVE')) {
    return 'running'
  }
  if (normalized.includes('FAIL') || normalized.includes('ERROR')) {
    return 'failed'
  }
  if (normalized.includes('CANCEL') || normalized.includes('KILL') || normalized.includes('ABORT')) {
    return 'cancelled'
  }
  return 'completed'
}

function buildSubagentDaemonMessages(steps: readonly ParsedTranscriptStep[]): {
  readonly messages: DaemonSession['messages']
  readonly thinkingContent: string[]
  readonly toolExecutions: Array<Record<string, unknown>>
} {
  const messages: DaemonSession['messages'] = []
  const thinkingContent: string[] = []
  const toolExecutions: Array<Record<string, unknown>> = []
  const pendingCalls: Array<{ readonly args: Record<string, unknown>; readonly id: string; readonly name: string }> = []

  for (const step of steps) {
    if (step.type === 'USER_INPUT' || step.type === 'SYSTEM_MESSAGE') {
      const cleaned = stripSystemMessageEnvelope(step.content)
      if (!cleaned) continue
      messages.push({
        role: 'user',
        content: clipText(cleaned, MAX_SUBAGENT_HISTORY_TEXT_CHARS),
      })
      continue
    }
    if (step.type === 'PLANNER_RESPONSE') {
      pendingCalls.length = 0
      const toolCalls = step.toolCalls.map((call, idx) => {
        const id = `agy-step-${step.stepIndex}-tc-${idx + 1}`
        const argsObj: Record<string, unknown> = isRecord(call.args) ? call.args : {}
        pendingCalls.push({ id, name: call.name, args: argsObj })
        return {
          id,
          type: 'function',
          function: {
            name: call.name,
            arguments: formatArgsPreview(call.args),
          },
        }
      })
      const thinking = step.thinking.trim()
      if (thinking) {
        thinkingContent.push(clipText(thinking, MAX_SUBAGENT_HISTORY_TEXT_CHARS))
      }
      messages.push({
        role: 'assistant',
        content: clipText(step.content, MAX_SUBAGENT_HISTORY_TEXT_CHARS),
        ...(thinking ? { thinking: clipText(thinking, MAX_SUBAGENT_HISTORY_TEXT_CHARS) } : {}),
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      })
      continue
    }
    const matched = pendingCalls.shift()
    const toolCallId = matched?.id ?? `agy-step-${step.stepIndex}`
    const toolName = matched?.name ?? 'tool'
    const resultText = clipText(stripAgyToolTimestamps(step.error || step.content || ''), MAX_SUBAGENT_HISTORY_TEXT_CHARS)
    const isErr = step.status === 'ERROR' || Boolean(step.error)
    messages.push({
      role: 'tool',
      tool_call_id: toolCallId,
      name: toolName,
      content: resultText,
    })
    toolExecutions.push({
      toolCallId,
      name: toolName,
      inputs: matched?.args ?? {},
      result: resultText,
      permitted: true,
      durationMs: 0,
      ...(isErr ? { error: resultText.slice(0, 240) || 'Tool execution failed' } : {}),
    })
  }

  return { messages, thinkingContent, toolExecutions }
}

function emptySwarmStatus(sessionId: string): SwarmStatusSnapshot {
  return {
    active: false,
    session_id: sessionId,
    title: '',
    board: 'local',
    open_board: SWARM_OPEN_BOARD_MARKDOWN,
    head: 0,
    tail_from_seq: 1,
    overview: null,
    channels: [
      {
        name: MAIN_CHANNEL,
        topic: '',
        created_by: COORDINATOR_NAME,
        archived: false,
        subscriber_count: 0,
      },
    ],
    members: [],
  }
}

/**
 * Antigravity brain filesystem bridge for Xerxes DaemonServer.
 *
 * Projects `.system_generated/swarm/`, worker `transcript.jsonl`, top-level
 * brain artifacts, and non-swarm `.system_generated/subagents/` into canonical
 * Xerxes snapshots and live debounced `SwarmLifecycleEvent` pushes.
 */
export class AgyBrainBridge {
  private readonly options: AgyBrainBridgeOptions
  private readonly transcriptCache = new Map<string, CachedTranscriptEntry>()
  private readonly backgroundTasksCache = new Map<string, CachedBackgroundTasksEntry>()
  private readonly watches = new Map<string, WatchedSessionState>()
  private readonly pinnedArtifactsBySession = new Map<string, Set<string>>()
  private readonly artifactVersions = new Map<string, number>()

  constructor(options: AgyBrainBridgeOptions = {}) {
    this.options = options
  }

  resolveLocation(input: {
    readonly bindingCommand?: string
    readonly externalSessionId: string
  }): AgyBrainLocation | undefined {
    return resolveAgyBrainLocation({
      externalSessionId: input.externalSessionId,
      ...(input.bindingCommand ? { bindingCommand: input.bindingCommand } : {}),
      ...(this.options.homeDir ? { homeDir: this.options.homeDir } : {}),
      ...(this.options.env ? { env: this.options.env } : {}),
    })
  }

  hasSwarmData(location: AgyBrainLocation): boolean {
    return (
      existsSync(join(location.swarmDir, 'swarm.json')) ||
      existsSync(join(location.swarmDir, 'members.json')) ||
      existsSync(join(location.swarmDir, 'board.jsonl'))
    )
  }

  /**
   * Resolve read-only preview roots for an Antigravity session: its own brain
   * session directory plus any swarm member brain directories strictly verified
   * to be direct child directories of `location.brainRoot` listed in `members.json`.
   */
  async listAllowedPreviewRoots(location: AgyBrainLocation): Promise<string[]> {
    const roots: string[] = []
    const seen = new Set<string>()
    const resolvedBrainRoot = resolve(location.brainRoot)
    let realBrainRoot: string
    try {
      realBrainRoot = await realpath(resolvedBrainRoot)
    } catch {
      return roots
    }

    const pushIfDirectBrainChild = async (candidateDir: string): Promise<void> => {
      const resolvedCandidate = resolve(candidateDir)
      if (dirname(resolvedCandidate) !== resolvedBrainRoot) return
      try {
        const realCandidate = await realpath(resolvedCandidate)
        if (dirname(realCandidate) !== realBrainRoot) return
        if (!(await stat(realCandidate)).isDirectory()) return
        if (!seen.has(realCandidate)) {
          seen.add(realCandidate)
          roots.push(realCandidate)
        }
      } catch {
        // Missing or unreadable brain directory cannot authorize preview reads.
      }
    }

    await pushIfDirectBrainChild(location.sessionDir)

    const membersDoc = await readJsonFileSafe(join(location.swarmDir, 'members.json'))
    const rawMembers = membersDoc && isRecord(membersDoc.members) ? membersDoc.members : undefined
    if (rawMembers) {
      for (const value of Object.values(rawMembers)) {
        if (!isRecord(value)) continue
        const convId = readString(value.conversation_id)
        if (
          !convId ||
          !EXTERNAL_SESSION_ID_PATTERN.test(convId) ||
          convId === '.' ||
          convId === '..' ||
          convId.includes('..')
        ) {
          continue
        }
        await pushIfDirectBrainChild(join(location.brainRoot, convId))
      }
    }

    return roots
  }

  /** Number of sessions currently watched (diagnostic / test assertion helper). */
  watchedSessionCount(): number {
    return this.watches.size
  }

  /** Number of currently armed timers across all watched sessions (diagnostic / test assertion helper). */
  activeTimerCount(): number {
    let count = 0
    for (const state of this.watches.values()) {
      if (state.debounceTimer !== undefined) count += 1
      if (state.pollTimer !== undefined) count += 1
    }
    return count
  }

  async readSwarmStatus(
    sessionId: string,
    location: AgyBrainLocation,
    modelDisplayMap?: ReadonlyMap<string, string>,
  ): Promise<SwarmStatusSnapshot> {
    const parsed = await this.readSwarmDiskState(sessionId, location, modelDisplayMap)
    return parsed.status
  }

  async readSwarmBoard(
    location: AgyBrainLocation,
    options: AgyReadBoardOptions = {},
  ): Promise<{ readonly head: number; readonly messages: SwarmBoardMessage[] }> {
    const boardPath = join(location.swarmDir, 'board.jsonl')
    const rawBoard = await readTailBoundedUtf8(boardPath, MAX_BOARD_FILE_BYTES)
    const allMessages = rawBoard ? parseBoardMessagesFromText(rawBoard) : []
    const head = allMessages.at(-1)?.seq ?? 0

    let filtered = allMessages
    if (options.seqs && options.seqs.length > 0) {
      const wanted = new Set(options.seqs)
      filtered = allMessages.filter(msg => wanted.has(msg.seq))
    } else {
      if (options.channel && options.channel !== '*') {
        filtered = filtered.filter(msg => msg.channel === options.channel)
      }
      if (options.since !== undefined) {
        filtered = filtered.filter(msg => msg.seq > options.since!)
      }
      const limit = options.limit !== undefined && options.limit > 0 ? options.limit : undefined
      if (limit !== undefined) {
        filtered = options.since !== undefined
          ? filtered.slice(0, limit)
          : filtered.slice(-limit)
      }
    }

    return { head, messages: filtered }
  }

  async inspectSwarmMember(
    sessionId: string,
    location: AgyBrainLocation,
    memberName: string,
    options: AgyInspectMemberOptions = {},
  ): Promise<SwarmMemberInspectResult | undefined> {
    const cleanName = memberName.trim()
    if (!cleanName) {
      throw new ValidationError('name', 'member name must not be empty')
    }
    const parsed = await this.readSwarmDiskState(sessionId, location, options.modelDisplayMap)
    const member = parsed.status.members.find(item => item.name === cleanName)
    if (!member) return undefined

    const requestedLimit = options.limit
    const limit = typeof requestedLimit === 'number' && Number.isFinite(requestedLimit)
      ? Math.min(MAX_INSPECT_LIMIT, Math.max(1, Math.trunc(requestedLimit)))
      : DEFAULT_INSPECT_LIMIT

    const conversationId = member.conversation_id || location.externalSessionId
    const analysis = await this.readTranscriptAnalysis(
      location.brainRoot,
      conversationId,
      member.state,
      MAX_INSPECT_TRANSCRIPT_BYTES,
    )

    const allMessages = analysis?.messages ?? []
    const startOffset = Math.max(0, allMessages.length - limit)
    const sliced = allMessages.slice(startOffset).map((msg, idx) => ({
      ...msg,
      index: startOffset + idx,
    }))

    const activity: SwarmMemberActivitySnapshot = analysis?.activity ?? {
      phase: member.state === 'working' ? 'thinking' : 'idle',
      ...(member.state === 'working' ? { summary: 'Working…' } : {}),
      active_tools: [],
      recent_tools: [],
    }

    const enrichedMember: SwarmMemberSnapshot = {
      ...member,
      ...(analysis?.currentActivity ? { current_activity: analysis.currentActivity } : {}),
      ...(!member.error && analysis?.lastError ? { error: analysis.lastError } : {}),
    }

    return {
      session_id: sessionId,
      member: enrichedMember,
      conversation_id: conversationId,
      activity,
      messages: sliced,
      total_messages: allMessages.length,
    }
  }

  /** Attach (or update) a live debounced watcher for an Antigravity session's swarm directory. */
  watchSession(params: AgyWatchSessionParams): void {
    const location = this.resolveLocation({
      externalSessionId: params.externalSessionId,
      ...(params.bindingCommand ? { bindingCommand: params.bindingCommand } : {}),
    })
    if (!location) return

    const existing = this.watches.get(params.sessionId)
    if (
      existing &&
      existing.location.sessionDir === location.sessionDir &&
      existing.workspaceDir === params.workspaceDir
    ) {
      existing.modelDisplayMap = params.modelDisplayMap
      if (params.onBackgroundChange) existing.onBackgroundChange = params.onBackgroundChange
      this.ensureDirectoryWatchers(existing)
      return
    }
    if (existing) {
      this.unwatchSession(params.sessionId)
    }

    const state: WatchedSessionState = {
      sessionId: params.sessionId,
      workspaceDir: params.workspaceDir,
      location,
      modelDisplayMap: params.modelDisplayMap,
      onBackgroundChange: params.onBackgroundChange,
      onSwarmEvent: params.onSwarmEvent,
      lastSnapshot: undefined,
      lastBackgroundSig: '',
      lastBoardSeq: 0,
      lastActivities: new Map<string, string>(),
      swarmWatcher: undefined,
      swarmWatchDir: undefined,
      narratorWatcher: undefined,
      narratorWatchDir: undefined,
      tasksWatcher: undefined,
      tasksWatchDir: undefined,
      logsWatcher: undefined,
      logsWatchDir: undefined,
      debounceTimer: undefined,
      pollTimer: undefined,
      refreshing: false,
      refreshQueued: false,
    }
    this.watches.set(params.sessionId, state)
    this.ensureDirectoryWatchers(state)
    void this.refreshWatchedSession(state)
  }

  /** Trigger an immediate debounced refresh for a watched session (e.g. after a `swarm` tool call). */
  refreshSession(sessionId: string): void {
    const state = this.watches.get(sessionId)
    if (!state) return
    this.ensureDirectoryWatchers(state)
    this.scheduleWatchedRefresh(state)
  }

  /** Stop watching a session and release all file watchers and timers immediately. */
  unwatchSession(sessionId: string): void {
    const state = this.watches.get(sessionId)
    if (!state) return
    this.watches.delete(sessionId)
    if (state.debounceTimer !== undefined) {
      clearTimeout(state.debounceTimer)
      state.debounceTimer = undefined
    }
    if (state.pollTimer !== undefined) {
      clearInterval(state.pollTimer)
      state.pollTimer = undefined
    }
    try {
      state.swarmWatcher?.close()
    } catch {
      // Ignore already-closed watcher.
    }
    state.swarmWatcher = undefined
    state.swarmWatchDir = undefined
    try {
      state.narratorWatcher?.close()
    } catch {
      // Ignore already-closed watcher.
    }
    state.narratorWatcher = undefined
    state.narratorWatchDir = undefined
    try {
      state.tasksWatcher?.close()
    } catch {
      // Ignore already-closed watcher.
    }
    state.tasksWatcher = undefined
    state.tasksWatchDir = undefined
    try {
      state.logsWatcher?.close()
    } catch {
      // Ignore already-closed watcher.
    }
    state.logsWatcher = undefined
    state.logsWatchDir = undefined
  }

  /** Dispose all active watchers, timers, and caches on daemon shutdown. */
  dispose(): void {
    for (const sessionId of [...this.watches.keys()]) {
      this.unwatchSession(sessionId)
    }
    this.transcriptCache.clear()
    this.backgroundTasksCache.clear()
  }

  // ── Artifacts ─────────────────────────────────────────────────────────────

  setSessionPinnedArtifactIds(sessionId: string, pinnedIds: Iterable<string>): void {
    const set = new Set<string>()
    for (const id of pinnedIds) {
      if (parseAgyArtifactFilename(id)) set.add(id)
    }
    if (set.size > 0) {
      this.pinnedArtifactsBySession.set(sessionId, set)
    } else {
      this.pinnedArtifactsBySession.delete(sessionId)
    }
  }

  getSessionPinnedArtifactIds(sessionId: string): string[] {
    const set = this.pinnedArtifactsBySession.get(sessionId)
    return set ? [...set] : []
  }

  async listArtifacts(
    sessionId: string,
    location: AgyBrainLocation,
    persistedPinnedIds?: readonly string[],
  ): Promise<readonly ArtifactSummary[]> {
    if (persistedPinnedIds && !this.pinnedArtifactsBySession.has(sessionId)) {
      this.setSessionPinnedArtifactIds(sessionId, persistedPinnedIds)
    }
    let entries: string[]
    try {
      entries = await readdir(location.sessionDir)
    } catch {
      return []
    }
    const summaries: ArtifactSummary[] = []
    for (const filename of entries) {
      if (!isValidAgyArtifactFilename(filename)) continue
      const artifact = await this.readArtifactByFilename(sessionId, location, filename)
      if (artifact) {
        summaries.push(summaryOf(artifact))
      }
    }
    return summaries.sort(compareArtifactSummaries)
  }

  async readArtifact(
    sessionId: string,
    location: AgyBrainLocation,
    id: string,
    persistedPinnedIds?: readonly string[],
  ): Promise<Artifact | undefined> {
    if (persistedPinnedIds && !this.pinnedArtifactsBySession.has(sessionId)) {
      this.setSessionPinnedArtifactIds(sessionId, persistedPinnedIds)
    }
    const filename = parseAgyArtifactFilename(id)
    if (!filename) return undefined
    return this.readArtifactByFilename(sessionId, location, filename)
  }

  async pinArtifact(
    sessionId: string,
    location: AgyBrainLocation,
    id: string,
    pinned: boolean,
    persistedPinnedIds?: readonly string[],
  ): Promise<ArtifactSummary | undefined> {
    if (persistedPinnedIds && !this.pinnedArtifactsBySession.has(sessionId)) {
      this.setSessionPinnedArtifactIds(sessionId, persistedPinnedIds)
    }
    const filename = parseAgyArtifactFilename(id)
    if (!filename) return undefined
    const existing = await this.readArtifactByFilename(sessionId, location, filename)
    if (!existing) return undefined
    let set = this.pinnedArtifactsBySession.get(sessionId)
    if (!set) {
      set = new Set<string>()
      this.pinnedArtifactsBySession.set(sessionId, set)
    }
    if (pinned) set.add(existing.id)
    else set.delete(existing.id)
    return { ...summaryOf(existing), pinned }
  }

  // ── Background Tasks, Schedules & Live Logs ──────────────────────────────

  /**
   * Synchronously project background commands, schedules/timers, and workflows
   * from an Antigravity session's `.system_generated/logs/transcript.jsonl`,
   * `.system_generated/messages/`, and `.system_generated/tasks/*.log`.
   */
  listBackgroundTasksSync(
    location: AgyBrainLocation,
    cwd = '',
  ): readonly AgyBackgroundTaskSnapshot[] {
    const transcriptPath = join(location.logsDir, 'transcript.jsonl')
    let mtimeMs = 0
    let size = 0
    try {
      const st = statSync(transcriptPath)
      mtimeMs = st.mtimeMs
      size = st.size
    } catch {
      // Transcript may not exist yet; still check tasksDir for live logs.
    }

    let taskFileNames: string[] = []
    const taskFileStats = new Map<string, { readonly mtimeMs: number; readonly size: number }>()
    try {
      taskFileNames = readdirSync(location.tasksDir)
        .filter(name => isValidAgyTaskLogFilename(name))
        .sort()
      for (const name of taskFileNames) {
        try {
          const st = statSync(join(location.tasksDir, name))
          taskFileStats.set(name, { mtimeMs: st.mtimeMs, size: st.size })
        } catch {
          // Ignore disappeared log file.
        }
      }
    } catch {
      taskFileNames = []
    }

    const taskFilesSig = taskFileNames
      .map(name => `${name}:${taskFileStats.get(name)?.size ?? 0}:${Math.trunc(taskFileStats.get(name)?.mtimeMs ?? 0)}`)
      .join(',')
    const cacheKey = location.sessionDir
    const cached = this.backgroundTasksCache.get(cacheKey)
    if (cached && cached.mtimeMs === mtimeMs && cached.size === size && cached.taskFilesSig === taskFilesSig) {
      return cached.tasks
    }

    const rawTranscript = mtimeMs > 0 ? readTailBoundedUtf8Sync(transcriptPath, MAX_BG_TRANSCRIPT_BYTES) : undefined
    const steps = rawTranscript ? parseTranscriptSteps(rawTranscript) : []
    const tasks = projectAgyBackgroundTasksFromSteps(steps, location, taskFileStats, cwd)

    if (this.backgroundTasksCache.size >= 64 && !this.backgroundTasksCache.has(cacheKey)) {
      const oldest = this.backgroundTasksCache.keys().next().value
      if (oldest !== undefined) this.backgroundTasksCache.delete(oldest)
    }
    this.backgroundTasksCache.set(cacheKey, { mtimeMs, size, taskFilesSig, tasks })
    return tasks
  }

  /**
   * Inspect one Antigravity background task (e.g. `agy-task:<convId>/task-2558`)
   * with the retained tail of its `.system_generated/tasks/task-<N>.log` output.
   */
  inspectBackgroundTask(
    location: AgyBrainLocation,
    taskId: string,
    maxOutputChars = 24_000,
    cwd = '',
    fallbackTask?: AgyBackgroundTaskSnapshot,
  ): JsonRpcPayload | undefined {
    const cleanId = taskId.trim()
    const rawTarget = cleanId.startsWith(AGY_TASK_ID_PREFIX)
      ? cleanId.slice(AGY_TASK_ID_PREFIX.length)
      : cleanId
    if (!rawTarget) return undefined

    const diskTasks = this.listBackgroundTasksSync(location, cwd)
    const matched =
      diskTasks.find(t => t.id === cleanId || t.rawTaskId === rawTarget) ??
      fallbackTask

    const inferredLog = matched?.logFileName ?? inferAgyTaskLogFilename(rawTarget)
    let output = ''
    let truncated = false
    if (inferredLog && isValidAgyTaskLogFilename(inferredLog)) {
      const logPath = join(location.tasksDir, inferredLog)
      const cap = Math.max(1024, Math.min(maxOutputChars, MAX_BG_LOG_INSPECT_CHARS))
      const raw = readTailBoundedUtf8Sync(logPath, cap * 2)
      if (raw !== undefined) {
        if (raw.length > cap) {
          output = raw.slice(raw.length - cap)
          truncated = true
        } else {
          output = raw
        }
      }
    }
    if (!output && matched?.outputPreview) {
      output = matched.outputPreview
    }
    if (!matched && !output) return undefined

    const title = matched?.title || rawTarget
    const label = matched?.label || title.slice(0, 120)
    const running = matched ? matched.state === 'running' || matched.state === 'scheduled' : false
    return {
      id: matched?.id ?? `${AGY_TASK_ID_PREFIX}${rawTarget}`,
      kind: matched?.isDaemon ? 'daemon' : matched?.kind === 'schedule' ? 'schedule' : 'background',
      label,
      command: title,
      cwd: cwd || matched?.detail || '',
      running,
      startedAt: matched?.startedAt ?? Date.now(),
      ...(matched?.endedAt ? { endedAt: matched.endedAt } : {}),
      exitCode: matched?.exitCode ?? null,
      outputChars: output.length,
      output,
      outputTruncated: truncated,
      canWrite: false,
      canInterrupt: running,
      canKill: running,
      source: 'antigravity',
      rawTaskId: matched?.rawTaskId ?? rawTarget,
    }
  }

  // ── Non-Swarm Subagents ───────────────────────────────────────────────────

  /**
   * Synchronously project non-swarm Antigravity subagents from `.system_generated/subagents/*.json`
   * (excluding `Swarm Worker (...)` entries which belong in the Immortals Board).
   */
  listSubagentSnapshotsSync(location: AgyBrainLocation): JsonRpcPayload[] {
    let entries: string[]
    try {
      entries = readdirSync(location.subagentsDir)
    } catch {
      return []
    }
    if (entries.length === 0) return []

    const swarmConvIds = this.readSwarmConversationIdsSync(location)
    const rows: JsonRpcPayload[] = []

    for (const filename of entries.sort()) {
      if (!filename.endsWith('.json')) continue
      const fullPath = join(location.subagentsDir, filename)
      const record = readJsonFileSafeSync(fullPath)
      if (!record) continue
      const convId = readString(record.conversationId) ?? readString(record.conversation_id)
      if (!convId || !EXTERNAL_SESSION_ID_PATTERN.test(convId)) continue
      const descriptor = isRecord(record.subagentDescriptor) ? record.subagentDescriptor : undefined
      const role = readString(descriptor?.role) ?? ''
      const typeName = readString(descriptor?.typeName) ?? 'subagent'

      // Exclude swarm workers — they live in the Immortals Board, not Subagents.
      if (SWARM_WORKER_ROLE_PATTERN.test(role) || swarmConvIds.has(convId)) {
        continue
      }

      const status = mapSubagentState(readString(record.state))
      const analysis = this.readTranscriptAnalysisSync(
        location.brainRoot,
        convId,
        status === 'running' ? 'working' : 'waiting',
        MAX_ACTIVITY_TRANSCRIPT_BYTES,
      )
      let fileMtimeIso: string | undefined
      try {
        fileMtimeIso = statSync(fullPath).mtime.toISOString()
      } catch {
        fileMtimeIso = undefined
      }

      const title = role || typeName || convId.slice(0, 8)
      const summary = analysis?.currentActivity
        ?? (analysis?.lastAssistantText ? clipText(analysis.lastAssistantText.replace(/\s+/g, ' '), 180) : undefined)
      const row: JsonRpcPayload = {
        id: convId,
        status,
        name: title,
        title,
        agent_id: typeName,
        history_session_id: convId,
        ...(analysis?.lastModel ? { model: analysis.lastModel } : {}),
        ...(summary ? { summary } : {}),
        ...(analysis?.lastError ? { error: analysis.lastError } : {}),
        ...(analysis?.firstCreatedAt || fileMtimeIso
          ? { created_at: analysis?.firstCreatedAt ?? fileMtimeIso }
          : {}),
        ...(analysis?.lastCreatedAt || fileMtimeIso
          ? { updated_at: analysis?.lastCreatedAt ?? fileMtimeIso }
          : {}),
        ...(analysis ? { api_calls: analysis.apiCalls } : {}),
        ...(analysis ? { tool_count: analysis.toolCount } : {}),
        ...(analysis ? { input_tokens: analysis.inputTokens } : {}),
        ...(analysis ? { output_tokens: analysis.outputTokens } : {}),
      }
      rows.push(row)
    }

    return rows
  }

  /** Inspect a non-swarm Antigravity subagent and optionally synthesize a paginated `sessionHistoryPage`. */
  async inspectSubagent(
    location: AgyBrainLocation,
    taskId: string,
    options: { readonly before?: unknown; readonly limit?: number } = {},
  ): Promise<AgySubagentInspectResult | undefined> {
    const cleanId = taskId.trim()
    if (!cleanId || !EXTERNAL_SESSION_ID_PATTERN.test(cleanId)) return undefined
    const filePath = join(location.subagentsDir, `${cleanId}.json`)
    const record = await readJsonFileSafe(filePath)
    if (!record) return undefined
    const convId = readString(record.conversationId) ?? readString(record.conversation_id) ?? cleanId
    const descriptor = isRecord(record.subagentDescriptor) ? record.subagentDescriptor : undefined
    const role = readString(descriptor?.role) ?? ''
    if (SWARM_WORKER_ROLE_PATTERN.test(role)) return undefined

    const typeName = readString(descriptor?.typeName) ?? 'subagent'
    const status = mapSubagentState(readString(record.state))
    const transcriptPath = join(location.brainRoot, convId, '.system_generated', 'logs', 'transcript.jsonl')
    const raw = await readTailBoundedUtf8(transcriptPath, MAX_INSPECT_TRANSCRIPT_BYTES)
    const steps = raw ? parseTranscriptSteps(raw) : []
    const analysis = analyzeTranscriptSteps(steps, status === 'running' ? 'working' : 'waiting')

    const title = role || typeName || convId.slice(0, 8)
    const prompt = (analysis.lastUserPrompt ?? '').slice(0, 16_000)
    const output = (analysis.lastAssistantText ?? '').slice(0, 16_000)
    const summary = analysis.currentActivity
      ?? (output ? clipText(output.replace(/\s+/g, ' '), 180) : undefined)

    const agent: JsonRpcPayload = {
      id: convId,
      status,
      name: title,
      title,
      agent_id: typeName,
      history_session_id: convId,
      ...(analysis.lastModel ? { model: analysis.lastModel } : {}),
      ...(summary ? { summary } : {}),
      ...(analysis.lastError ? { error: analysis.lastError } : {}),
      ...(analysis.firstCreatedAt ? { created_at: analysis.firstCreatedAt } : {}),
      ...(analysis.lastCreatedAt ? { updated_at: analysis.lastCreatedAt } : {}),
      api_calls: analysis.apiCalls,
      tool_count: analysis.toolCount,
      input_tokens: analysis.inputTokens,
      output_tokens: analysis.outputTokens,
      prompt,
      output,
      retained: true,
    }

    if (options.limit === undefined) {
      return { agent }
    }

    const synthesized = buildSubagentDaemonMessages(steps)
    const page = sessionHistoryPage(
      {
        id: convId,
        messages: synthesized.messages,
        toolExecutions: synthesized.toolExecutions,
        thinkingContent: synthesized.thinkingContent,
      },
      options.limit,
      options.before,
    )
    return {
      agent,
      history_session_id: convId,
      history: page as unknown as JsonRpcPayload,
    }
  }

  // ── Internal helpers ──────────────────────────────────────────────────────

  private async readArtifactByFilename(
    sessionId: string,
    location: AgyBrainLocation,
    filename: string,
  ): Promise<Artifact | undefined> {
    const filePath = resolve(location.sessionDir, filename)
    if (dirname(filePath) !== resolve(location.sessionDir)) return undefined
    let realSessionDir: string
    let realFilePath: string
    try {
      realSessionDir = await realpath(location.sessionDir)
      realFilePath = await realpath(filePath)
    } catch {
      return undefined
    }
    if (dirname(realFilePath) !== realSessionDir) return undefined
    let stats: Awaited<ReturnType<typeof stat>>
    try {
      stats = await stat(realFilePath)
    } catch {
      return undefined
    }
    const kind = artifactKindFromFilename(filename)
    let content: string
    if (kind === 'image') {
      // Symlinks named .png pointing to non-image targets are refused by target extension too.
      if (!AGY_IMAGE_ARTIFACT_EXT_RE.test(realFilePath)) return undefined
      if (!stats.isFile() || stats.size === 0 || stats.size > MAX_AGY_IMAGE_ARTIFACT_BYTES) return undefined
      let bytes: Buffer
      try {
        bytes = await readFile(realFilePath)
      } catch {
        return undefined
      }
      if (bytes.length === 0 || bytes.length > MAX_AGY_IMAGE_ARTIFACT_BYTES) return undefined
      const ext = extname(realFilePath).toLowerCase()
      if (ext === '.svg') {
        let svgText: string
        try {
          svgText = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
        } catch {
          return undefined
        }
        if (!AGY_SVG_ROOT_RE.test(svgText)) return undefined
        content = `data:image/svg+xml;base64,${bytes.toString('base64')}`
      } else {
        const sniffed = sniffImageMediaType(bytes)
        if (!sniffed) return undefined
        content = `data:${sniffed};base64,${bytes.toString('base64')}`
      }
    } else {
      if (!stats.isFile() || stats.size > MAX_ARTIFACT_CONTENT_BYTES) return undefined
      try {
        content = await readFile(realFilePath, 'utf8')
      } catch {
        return undefined
      }
    }

    const meta = await readJsonFileSafe(`${filePath}.metadata.json`)
    const id = `${AGY_ARTIFACT_PREFIX}${filename}`
    const metaTitle = readString(meta?.title)
    const title = metaTitle
      ? metaTitle.slice(0, MAX_ARTIFACT_TITLE_CHARS)
      : humanizeArtifactTitle(filename, content, kind)
    const createdIso = (stats.birthtimeMs > 0 ? stats.birthtime : stats.mtime).toISOString()
    const updatedIso = normalizeIsoTimestamp(meta?.updatedAt, stats.mtime.toISOString())
    const pinned = this.pinnedArtifactsBySession.get(sessionId)?.has(id) === true
    const version = this.artifactVersions.get(`${sessionId}:${id}`) ?? 1

    return {
      id,
      title,
      kind,
      content,
      created_at: createdIso,
      updated_at: updatedIso,
      pinned,
      version,
    }
  }

  private readSwarmConversationIdsSync(location: AgyBrainLocation): Set<string> {
    const ids = new Set<string>()
    const membersDoc = readJsonFileSafeSync(join(location.swarmDir, 'members.json'))
    const rawMembers = membersDoc && isRecord(membersDoc.members) ? membersDoc.members : undefined
    if (!rawMembers) return ids
    for (const value of Object.values(rawMembers)) {
      if (!isRecord(value)) continue
      const convId = readString(value.conversation_id)
      if (convId) ids.add(convId)
    }
    return ids
  }

  private async readSwarmDiskState(
    sessionId: string,
    location: AgyBrainLocation,
    modelDisplayMap?: ReadonlyMap<string, string>,
  ): Promise<ParsedSwarmDiskState> {
    const [swarmDoc, membersDoc, planDoc, rawBoard] = await Promise.all([
      readJsonFileSafe(join(location.swarmDir, 'swarm.json')),
      readJsonFileSafe(join(location.swarmDir, 'members.json')),
      readJsonFileSafe(join(location.swarmDir, 'plan.json')),
      readTailBoundedUtf8(join(location.swarmDir, 'board.jsonl'), MAX_BOARD_FILE_BYTES),
    ])

    const exists = Boolean(swarmDoc || membersDoc || (rawBoard !== undefined && rawBoard.length > 0))
    if (!exists) {
      return {
        exists: false,
        status: emptySwarmStatus(sessionId),
        messages: [],
        activitiesByMember: new Map(),
      }
    }

    const messages = rawBoard ? parseBoardMessagesFromText(rawBoard) : []
    const lastPostByAgent = new Map<string, { readonly seq: number; readonly time: string }>()
    const briefAuthorByTarget = new Map<string, string>()
    const channelNames = new Set<string>([MAIN_CHANNEL])

    for (const msg of messages) {
      channelNames.add(msg.channel)
      lastPostByAgent.set(msg.agent, { seq: msg.seq, time: msg.time })
      if (msg.kind === 'brief' && msg.mentions.length > 0) {
        const leadingMatch = /^@([a-z0-9][a-z0-9_-]{0,23})\b/i.exec(msg.text.trim())
        const primaryTarget = leadingMatch?.[1]?.toLowerCase() ?? msg.mentions[0]
        if (primaryTarget && !briefAuthorByTarget.has(primaryTarget)) {
          briefAuthorByTarget.set(primaryTarget, msg.agent)
        }
      }
    }

    const rawMembersRecord = membersDoc && isRecord(membersDoc.members) ? membersDoc.members : {}
    const memberNames = Object.keys(rawMembersRecord)
    const groupCounts = new Map<string, number>()
    for (const name of memberNames) {
      const m = GROUP_SUFFIX_PATTERN.exec(name)
      if (m && m[1]) {
        const prefix = m[1].toLowerCase()
        groupCounts.set(prefix, (groupCounts.get(prefix) ?? 0) + 1)
      }
    }

    const orderedNames = [...memberNames].sort((left, right) => {
      if (left === COORDINATOR_NAME) return -1
      if (right === COORDINATOR_NAME) return 1
      return left.localeCompare(right)
    })

    let narratorConvId: string | undefined
    let narratorName: string | undefined
    const activitiesByMember = new Map<string, SwarmMemberActivitySnapshot>()
    const memberSnapshots: SwarmMemberSnapshot[] = []
    let maxObservedSeq = messages.at(-1)?.seq ?? 0

    for (const name of orderedNames) {
      const rawValue = rawMembersRecord[name]
      if (!isRecord(rawValue)) continue
      const raw = rawValue as unknown as RawAgyMember
      const role = inferAgyMemberRole(name, readString(raw.role))
      const mappedState = mapAgyMemberState(raw)
      const convId = readString(raw.conversation_id)
        ?? (role === 'coordinator' ? location.externalSessionId : undefined)
      if (role === 'narrator' && convId && !narratorConvId) {
        narratorConvId = convId
        narratorName = name
      }

      const model = readString(raw.applied_model)
        ?? readString(raw.model)
        ?? readString(raw.desired_model)
        ?? ''
      const modelDisplayName =
        (model ? modelDisplayMap?.get(model) : undefined) ??
        readString(raw.model_display_name) ??
        model

      const memberChannels = Array.isArray(raw.channels)
        ? raw.channels.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
        : [MAIN_CHANNEL]
      for (const ch of memberChannels) channelNames.add(ch)

      const lastReadSeq = Math.max(0, Math.trunc(readFiniteNumber(raw.last_read_seq) ?? 0))
      const lastPost = lastPostByAgent.get(name)
      maxObservedSeq = Math.max(
        maxObservedSeq,
        lastReadSeq,
        lastPost?.seq ?? 0,
        mappedState.parked_since ?? 0,
      )

      let currentActivity: string | undefined
      let transcriptError: string | undefined
      if (convId && mappedState.state === 'working') {
        const analysis = await this.readTranscriptAnalysis(
          location.brainRoot,
          convId,
          mappedState.state,
          MAX_ACTIVITY_TRANSCRIPT_BYTES,
        )
        if (analysis) {
          activitiesByMember.set(name, analysis.activity)
          currentActivity = analysis.currentActivity
          transcriptError = analysis.lastError
        }
      }

      const explicitGroup = readString(raw.group)
      const groupMatch = GROUP_SUFFIX_PATTERN.exec(name)
      const inferredGroup =
        explicitGroup ??
        (groupMatch?.[1] && (groupCounts.get(groupMatch[1].toLowerCase()) ?? 0) >= 2
          ? groupMatch[1].toLowerCase()
          : undefined)
      const addedBy =
        readString(raw.added_by) ??
        (role !== 'coordinator' ? briefAuthorByTarget.get(name.toLowerCase()) : undefined)
      const workspacePath = readString(raw.workspace_path)
      const memberError = readString(raw.error) ?? transcriptError

      memberSnapshots.push({
        name,
        ...(inferredGroup ? { group: inferredGroup } : {}),
        role,
        ...(addedBy ? { added_by: addedBy } : {}),
        model,
        model_display_name: modelDisplayName,
        workspace: normalizeWorkspaceMode(readString(raw.workspace_mode)),
        ...(workspacePath ? { workspace_path: workspacePath } : {}),
        channels: memberChannels.length > 0 ? memberChannels : [MAIN_CHANNEL],
        state: mappedState.state,
        killed: mappedState.killed,
        ...(convId ? { conversation_id: convId } : {}),
        ...(currentActivity ? { current_activity: currentActivity } : {}),
        ...(mappedState.parked_since !== undefined ? { parked_since: mappedState.parked_since } : {}),
        ...(mappedState.mentions_only !== undefined ? { mentions_only: mappedState.mentions_only } : {}),
        last_read_seq: lastReadSeq,
        ...(lastPost ? { last_post_seq: lastPost.seq, last_post_time: lastPost.time } : {}),
        ...(memberError ? { error: memberError } : {}),
      })
    }

    // Load narrator view.json (fallback to coordinator's own swarm_view/view.json if present).
    let viewDoc: Record<string, unknown> | undefined
    if (narratorConvId) {
      viewDoc = await readJsonFileSafe(join(location.brainRoot, narratorConvId, 'swarm_view', 'view.json'))
    }
    if (!viewDoc) {
      viewDoc = await readJsonFileSafe(join(location.sessionDir, 'swarm_view', 'view.json'))
    }

    const overview = synthesizeSwarmOverview(viewDoc, planDoc, narratorName)
    const channels: SwarmChannelSummary[] = [...channelNames]
      .sort((a, b) => (a === MAIN_CHANNEL ? -1 : b === MAIN_CHANNEL ? 1 : a.localeCompare(b)))
      .map(channelName => ({
        name: channelName,
        topic: '',
        created_by: COORDINATOR_NAME,
        archived: false,
        subscriber_count: memberSnapshots.filter(
          member => !member.killed && member.channels.includes(channelName),
        ).length,
      }))

    const title = readString(membersDoc?.title) ?? ''
    const active = memberSnapshots.length > 0 || messages.length > 0

    return {
      exists: true,
      messages,
      activitiesByMember,
      ...(narratorConvId ? { narratorConvId } : {}),
      status: {
        active,
        session_id: sessionId,
        title,
        board: 'local',
        open_board: SWARM_OPEN_BOARD_MARKDOWN,
        head: maxObservedSeq,
        tail_from_seq: 1,
        overview,
        channels,
        members: memberSnapshots,
      },
    }
  }

  private async readTranscriptAnalysis(
    brainRoot: string,
    conversationId: string,
    memberState: SwarmMemberState,
    maxBytes: number,
  ): Promise<TranscriptAnalysis | undefined> {
    if (!EXTERNAL_SESSION_ID_PATTERN.test(conversationId)) return undefined
    const transcriptPath = join(brainRoot, conversationId, '.system_generated', 'logs', 'transcript.jsonl')
    let stats: Awaited<ReturnType<typeof stat>>
    try {
      stats = await stat(transcriptPath)
    } catch {
      return undefined
    }
    if (!stats.isFile()) return undefined

    const cacheKey = `${transcriptPath}|${memberState}`
    const cached = this.transcriptCache.get(cacheKey)
    if (
      cached &&
      cached.mtimeMs === stats.mtimeMs &&
      cached.size === stats.size &&
      cached.maxBytes >= maxBytes
    ) {
      return cached.analysis
    }

    const raw = await readTailBoundedUtf8(transcriptPath, maxBytes)
    if (raw === undefined) return undefined
    const steps = parseTranscriptSteps(raw)
    const analysis = analyzeTranscriptSteps(steps, memberState)
    this.setTranscriptCache(cacheKey, {
      mtimeMs: stats.mtimeMs,
      size: stats.size,
      maxBytes,
      analysis,
    })
    return analysis
  }

  private readTranscriptAnalysisSync(
    brainRoot: string,
    conversationId: string,
    memberState: SwarmMemberState,
    maxBytes: number,
  ): TranscriptAnalysis | undefined {
    if (!EXTERNAL_SESSION_ID_PATTERN.test(conversationId)) return undefined
    const transcriptPath = join(brainRoot, conversationId, '.system_generated', 'logs', 'transcript.jsonl')
    let stats: ReturnType<typeof statSync>
    try {
      stats = statSync(transcriptPath)
    } catch {
      return undefined
    }
    if (!stats.isFile()) return undefined

    const cacheKey = `${transcriptPath}|${memberState}`
    const cached = this.transcriptCache.get(cacheKey)
    if (
      cached &&
      cached.mtimeMs === stats.mtimeMs &&
      cached.size === stats.size &&
      cached.maxBytes >= maxBytes
    ) {
      return cached.analysis
    }

    const raw = readTailBoundedUtf8Sync(transcriptPath, maxBytes)
    if (raw === undefined) return undefined
    const steps = parseTranscriptSteps(raw)
    const analysis = analyzeTranscriptSteps(steps, memberState)
    this.setTranscriptCache(cacheKey, {
      mtimeMs: stats.mtimeMs,
      size: stats.size,
      maxBytes,
      analysis,
    })
    return analysis
  }

  private setTranscriptCache(key: string, entry: CachedTranscriptEntry): void {
    if (this.transcriptCache.size >= 128 && !this.transcriptCache.has(key)) {
      const oldest = this.transcriptCache.keys().next().value
      if (oldest !== undefined) this.transcriptCache.delete(oldest)
    }
    this.transcriptCache.set(key, entry)
  }

  private ensureDirectoryWatchers(state: WatchedSessionState, narratorConvId?: string): void {
    if (this.watches.get(state.sessionId) !== state) return

    // Pick the deepest existing directory along swarmDir -> .system_generated -> sessionDir.
    const desiredSwarmDir = existsSync(state.location.swarmDir)
      ? state.location.swarmDir
      : existsSync(join(state.location.sessionDir, '.system_generated'))
        ? join(state.location.sessionDir, '.system_generated')
        : existsSync(state.location.sessionDir)
          ? state.location.sessionDir
          : undefined

    if (desiredSwarmDir && desiredSwarmDir !== state.swarmWatchDir) {
      try {
        state.swarmWatcher?.close()
      } catch {
        // Ignore close error.
      }
      state.swarmWatcher = undefined
      state.swarmWatchDir = undefined
      try {
        const watcher = watch(desiredSwarmDir, () => {
          this.scheduleWatchedRefresh(state)
        })
        watcher.on('error', () => {
          // Directory may have been removed or recreated; re-evaluate lazily.
          if (state.swarmWatcher === watcher) {
            try {
              watcher.close()
            } catch {
              // Ignore.
            }
            state.swarmWatcher = undefined
            state.swarmWatchDir = undefined
          }
        })
        state.swarmWatcher = watcher
        state.swarmWatchDir = desiredSwarmDir
      } catch {
        // Fallback polling still keeps active swarms updated if fs.watch is unavailable.
      }
    }

    if (narratorConvId) {
      const narratorRoot = join(state.location.brainRoot, narratorConvId)
      const narratorViewDir = join(narratorRoot, 'swarm_view')
      const desiredNarratorDir = existsSync(narratorViewDir)
        ? narratorViewDir
        : existsSync(narratorRoot)
          ? narratorRoot
          : undefined
      if (desiredNarratorDir && desiredNarratorDir !== state.narratorWatchDir) {
        try {
          state.narratorWatcher?.close()
        } catch {
          // Ignore.
        }
        state.narratorWatcher = undefined
        state.narratorWatchDir = undefined
        try {
          const watcher = watch(desiredNarratorDir, () => {
            this.scheduleWatchedRefresh(state)
          })
          watcher.on('error', () => {
            if (state.narratorWatcher === watcher) {
              try {
                watcher.close()
              } catch {
                // Ignore.
              }
              state.narratorWatcher = undefined
              state.narratorWatchDir = undefined
            }
          })
          state.narratorWatcher = watcher
          state.narratorWatchDir = desiredNarratorDir
        } catch {
          // Ignore unavailable fs.watch on narrator view dir.
        }
      }
    }
    // Also watch `.system_generated/tasks` and `.system_generated/logs` so
    // background shell commands, schedules, and task completions push live updates.
    const desiredTasksDir = existsSync(state.location.tasksDir)
      ? state.location.tasksDir
      : undefined
    if (desiredTasksDir && desiredTasksDir !== state.tasksWatchDir) {
      try {
        state.tasksWatcher?.close()
      } catch {
        // Ignore.
      }
      state.tasksWatcher = undefined
      state.tasksWatchDir = undefined
      try {
        const watcher = watch(desiredTasksDir, () => {
          this.scheduleWatchedRefresh(state)
        })
        watcher.on('error', () => {
          if (state.tasksWatcher === watcher) {
            try {
              watcher.close()
            } catch {
              // Ignore.
            }
            state.tasksWatcher = undefined
            state.tasksWatchDir = undefined
          }
        })
        state.tasksWatcher = watcher
        state.tasksWatchDir = desiredTasksDir
      } catch {
        // Ignore unavailable fs.watch on tasks dir.
      }
    }

    const desiredLogsDir = existsSync(state.location.logsDir)
      ? state.location.logsDir
      : undefined
    if (desiredLogsDir && desiredLogsDir !== state.logsWatchDir) {
      try {
        state.logsWatcher?.close()
      } catch {
        // Ignore.
      }
      state.logsWatcher = undefined
      state.logsWatchDir = undefined
      try {
        const watcher = watch(desiredLogsDir, () => {
          this.scheduleWatchedRefresh(state)
        })
        watcher.on('error', () => {
          if (state.logsWatcher === watcher) {
            try {
              watcher.close()
            } catch {
              // Ignore.
            }
            state.logsWatcher = undefined
            state.logsWatchDir = undefined
          }
        })
        state.logsWatcher = watcher
        state.logsWatchDir = desiredLogsDir
      } catch {
        // Ignore unavailable fs.watch on logs dir.
      }
    }
  }

  private scheduleWatchedRefresh(state: WatchedSessionState): void {
    if (this.watches.get(state.sessionId) !== state) return
    if (state.debounceTimer !== undefined) {
      clearTimeout(state.debounceTimer)
    }
    const delay = this.options.debounceMs ?? DEFAULT_WATCH_DEBOUNCE_MS
    state.debounceTimer = setTimeout(() => {
      state.debounceTimer = undefined
      void this.refreshWatchedSession(state)
    }, delay)
    state.debounceTimer.unref?.()
  }

  private syncWorkingPollTimer(state: WatchedSessionState, hasWorkingMembers: boolean): void {
    if (this.watches.get(state.sessionId) !== state) return
    if (hasWorkingMembers) {
      if (state.pollTimer === undefined) {
        const intervalMs = this.options.pollIntervalMs ?? DEFAULT_WORKING_POLL_INTERVAL_MS
        state.pollTimer = setInterval(() => {
          void this.refreshWatchedSession(state)
        }, intervalMs)
        state.pollTimer.unref?.()
      }
    } else if (state.pollTimer !== undefined) {
      clearInterval(state.pollTimer)
      state.pollTimer = undefined
    }
  }

  private async refreshWatchedSession(
    state: WatchedSessionState,
  ): Promise<void> {
    if (this.watches.get(state.sessionId) !== state) return
    if (state.refreshing) {
      state.refreshQueued = true
      return
    }
    state.refreshing = true
    try {
      const parsed = await this.readSwarmDiskState(
        state.sessionId,
        state.location,
        state.modelDisplayMap,
      )
      if (this.watches.get(state.sessionId) !== state) return

      this.ensureDirectoryWatchers(state, parsed.narratorConvId)

      const nextStatus = parsed.status
      const bgTasks = this.listBackgroundTasksSync(state.location)
      const bgSig = bgTasks.map(task => `${task.id}:${task.state}:${task.endedAt ?? ''}:${task.exitCode ?? ''}`).join('|')
      if (bgSig !== state.lastBackgroundSig) {
        const hadPrev = state.lastBackgroundSig !== '' || bgTasks.length > 0
        state.lastBackgroundSig = bgSig
        if (hadPrev) {
          state.onBackgroundChange?.()
        }
      }
      const hasLiveBg = bgTasks.some(task => task.state === 'running' || task.state === 'scheduled')
      const hasWorking = (nextStatus.active && nextStatus.members.some(m => m.state === 'working')) || hasLiveBg
      this.syncWorkingPollTimer(state, hasWorking)

      const nextActivitySigs = new Map<string, string>()
      for (const [name, activity] of parsed.activitiesByMember.entries()) {
        nextActivitySigs.set(name, activitySignatureOf(activity))
      }

      const prevStatus = state.lastSnapshot
      const prevBoardSeq = state.lastBoardSeq
      const prevActivitySigs = state.lastActivities
      const nextBoardSeq = parsed.messages.at(-1)?.seq ?? prevBoardSeq

      state.lastSnapshot = nextStatus
      state.lastBoardSeq = Math.max(prevBoardSeq, nextBoardSeq)
      state.lastActivities = nextActivitySigs

      const emit = (event: SwarmLifecycleEvent): void => {
        if (this.watches.get(state.sessionId) !== state) return
        try {
          state.onSwarmEvent(event, state.workspaceDir)
        } catch {
          // Observer errors never break watcher loop.
        }
      }

      if (!prevStatus) {
        if (nextStatus.active) {
          emit({
            session_id: state.sessionId,
            kind: 'activated',
            head: nextStatus.head,
            status: nextStatus,
          })
        }
        return
      }

      if (!prevStatus.active && nextStatus.active) {
        emit({
          session_id: state.sessionId,
          kind: 'activated',
          head: nextStatus.head,
          status: nextStatus,
        })
      }

      if (nextStatus.active && nextStatus.title && nextStatus.title !== prevStatus.title) {
        emit({
          session_id: state.sessionId,
          kind: 'titled',
          head: nextStatus.head,
          status: nextStatus,
        })
      }

      const prevMembersByName = new Map(prevStatus.members.map(m => [m.name, m]))
      for (const member of nextStatus.members) {
        const prevMember = prevMembersByName.get(member.name)
        if (!prevMember) {
          emit({
            session_id: state.sessionId,
            kind: 'member_added',
            head: nextStatus.head,
            status: nextStatus,
            member,
          })
          continue
        }
        if (!prevMember.killed && member.killed) {
          emit({
            session_id: state.sessionId,
            kind: 'member_killed',
            head: nextStatus.head,
            status: nextStatus,
            member,
          })
          continue
        }
        if (memberSignatureOf(prevMember) !== memberSignatureOf(member)) {
          emit({
            session_id: state.sessionId,
            kind: 'member_updated',
            head: nextStatus.head,
            status: nextStatus,
            member,
          })
          continue
        }
        const nextActSig = nextActivitySigs.get(member.name)
        const prevActSig = prevActivitySigs.get(member.name)
        if (nextActSig && nextActSig !== prevActSig) {
          const activity = parsed.activitiesByMember.get(member.name)
          if (activity) {
            emit({
              session_id: state.sessionId,
              kind: 'member_activity',
              head: nextStatus.head,
              status: nextStatus,
              member,
              activity,
            })
          }
        }
      }

      for (const message of parsed.messages) {
        if (message.seq > prevBoardSeq) {
          emit({
            session_id: state.sessionId,
            kind: 'message_posted',
            head: nextStatus.head,
            status: nextStatus,
            message,
          })
        }
      }

      if (
        nextStatus.overview !== null &&
        JSON.stringify(prevStatus.overview) !== JSON.stringify(nextStatus.overview)
      ) {
        emit({
          session_id: state.sessionId,
          kind: 'overview_updated',
          head: nextStatus.head,
          status: nextStatus,
        })
      }
    } finally {
      state.refreshing = false
      if (state.refreshQueued && this.watches.get(state.sessionId) === state) {
        state.refreshQueued = false
        void this.refreshWatchedSession(state)
      }
    }
  }
}

function compareArtifactSummaries(left: ArtifactSummary, right: ArtifactSummary): number {
  if (left.pinned !== right.pinned) return left.pinned ? -1 : 1
  if (left.updated_at !== right.updated_at) return left.updated_at < right.updated_at ? 1 : -1
  return left.id < right.id ? -1 : left.id > right.id ? 1 : 0
}

/** Validate a safe task log basename (e.g. `task-2558.log`) inside `.system_generated/tasks/`. */
export function isValidAgyTaskLogFilename(filename: string): boolean {
  if (!filename || filename.includes('/') || filename.includes('\\') || filename.includes('\0') || filename.includes('..')) {
    return false
  }
  return AGY_TASK_LOG_FILENAME_PATTERN.test(filename)
}

/** Infer `task-<N>.log` from a raw AGY task id like `<convId>/task-2558` or `task-2558`. */
export function inferAgyTaskLogFilename(rawTaskId: string): string | undefined {
  const match = /(?:^|\/)(task-[A-Za-z0-9._-]+)$/i.exec(rawTaskId.trim())
  if (!match) return undefined
  const candidate = `${match[1]!}.log`
  return isValidAgyTaskLogFilename(candidate) ? candidate : undefined
}

/** Extract an Antigravity background task id from tool output text. */
export function extractAgyTaskIdFromText(text: string): string | undefined {
  const match =
    /background task with task id:\s*([^\s"'<>]+)/i.exec(text) ??
    /TaskId["']?\s*[:=]\s*["']?([^\s"'<>]+)/i.exec(text) ??
    /task[_\s-]*id["']?\s*[:=]\s*["']?([A-Za-z0-9._-]+\/task-[A-Za-z0-9._-]+|task-[A-Za-z0-9._-]+)/i.exec(text)
  const raw = match?.[1]?.replace(/[.,;:]+$/, '').trim()
  return raw || undefined
}

/** Extract the `.system_generated/tasks/<file>.log` basename from tool output text. */
export function extractAgyLogFileNameFromText(text: string): string | undefined {
  const match = /\/\.system_generated\/tasks\/([A-Za-z0-9._-]+\.log)\b/i.exec(text)
  const candidate = match?.[1]?.trim()
  return candidate && isValidAgyTaskLogFilename(candidate) ? candidate : undefined
}

/** Parse a command exit code from Antigravity background task completion text. */
export function parseAgyExitCodeFromText(text: string): number | null {
  const match = /Command exited with code\s+(-?\d+)/i.exec(text) ?? /exit[_\s]+code\s*[:=]\s*(-?\d+)/i.exec(text)
  if (!match) return null
  const code = Number(match[1])
  return Number.isFinite(code) ? code : null
}

function parseIsoEpochMs(raw: string | undefined): number | null {
  if (!raw) return null
  const parsed = Date.parse(raw)
  return Number.isFinite(parsed) ? parsed : null
}

function projectAgyBackgroundTasksFromSteps(
  steps: readonly ParsedTranscriptStep[],
  location: AgyBrainLocation,
  taskFileStats: ReadonlyMap<string, { readonly mtimeMs: number; readonly size: number }>,
  cwd: string,
): AgyBackgroundTaskSnapshot[] {
  interface MutableTask {
    id: string
    rawTaskId: string
    kind: 'shell' | 'schedule' | 'task'
    title: string
    label: string
    detail: string
    state: 'running' | 'scheduled' | 'completed' | 'failed' | 'cancelled'
    startedAt: number | null
    endedAt: number | null
    exitCode: number | null
    nextRunAt: string | null
    durationSeconds?: number
    cronExpression?: string
    isDaemon?: boolean
    logFileName?: string
    outputPreview?: string
  }

  const byRawId = new Map<string, MutableTask>()
  const pendingToolCalls: Array<{ readonly name: string; readonly args: Record<string, unknown>; readonly createdAt: number | null }> = []

  const normalizeName = (name: string): string =>
    (name.split(/[.:/]/).pop() ?? name).trim().toLowerCase()

  for (const step of steps) {
    const stepMs = parseIsoEpochMs(step.createdAt)

    if (step.type === 'PLANNER_RESPONSE' && step.toolCalls.length > 0) {
      for (const tc of step.toolCalls) {
        const norm = normalizeName(tc.name)
        const args = isRecord(tc.args) ? tc.args : {}
        if (norm === 'manage_task') {
          const action = readString(args.Action)?.toLowerCase() ?? readString(args.action)?.toLowerCase() ?? ''
          const targetId = readString(args.TaskId) ?? readString(args.task_id) ?? ''
          if (action === 'kill' && targetId) {
            const existing = byRawId.get(targetId)
            if (existing) {
              existing.state = 'cancelled'
              existing.endedAt = stepMs ?? Date.now()
            }
          }
          continue
        }
        if (norm === 'run_command' || norm === 'schedule' || norm === 'run_workflow') {
          pendingToolCalls.push({ name: norm, args, createdAt: stepMs })
        }
      }
      continue
    }

    if (step.type === 'GENERIC' && step.content) {
      const taskId = extractAgyTaskIdFromText(step.content)
      const logFileName = extractAgyLogFileNameFromText(step.content) ?? (taskId ? inferAgyTaskLogFilename(taskId) : undefined)
      const hasBgMarker = /Tool is running as a background task|Scheduled (?:timer|cron|task)/i.test(step.content)
      if (!taskId && !hasBgMarker) {
        // Consume matching synchronous tool call if present
        if (pendingToolCalls.length > 0 && /Command exited with code/i.test(step.content)) {
          pendingToolCalls.shift()
        }
        continue
      }

      // Match the most relevant preceding tool call
      const callIdx = pendingToolCalls.findIndex(item =>
        item.name === 'schedule'
          ? /schedule|timer|cron/i.test(step.content) || Boolean(item.args.DurationSeconds ?? item.args.CronExpression)
          : item.name === 'run_command' || item.name === 'run_workflow',
      )
      const matchedCall = callIdx >= 0 ? pendingToolCalls.splice(callIdx, 1)[0] : undefined
      const rawTaskId = taskId ?? `step-${step.stepIndex}`
      const startedAt = matchedCall?.createdAt ?? stepMs ?? Date.now()

      if (matchedCall?.name === 'schedule' || /\b(?:DurationSeconds|CronExpression|timer|cron)\b/i.test(step.content)) {
        const args = matchedCall?.args ?? {}
        const durSec = readFiniteNumber(args.DurationSeconds ?? args.duration_seconds)
        const cronExpr = readString(args.CronExpression ?? args.cron_expression)
        const prompt = readString(args.Prompt ?? args.prompt) ??
          /Task Description:\s*(.+)/i.exec(step.content)?.[1]?.trim() ??
          'Scheduled task'
        const isDaemon = args.IsDaemon === true || args.is_daemon === true
        const nextRunAt = durSec !== undefined && durSec > 0
          ? new Date(startedAt + durSec * 1000).toISOString()
          : null
        const label = cronExpr
          ? `Cron · ${cronExpr}`
          : durSec !== undefined
            ? `Timer · in ${durSec}s`
            : 'Schedule'
        const statusUpper = step.status.toUpperCase()
        const state: MutableTask['state'] =
          statusUpper === 'ERROR' ? 'failed' : statusUpper === 'CANCELLED' ? 'cancelled' : 'scheduled'
        byRawId.set(rawTaskId, {
          id: `${AGY_TASK_ID_PREFIX}${rawTaskId}`,
          rawTaskId,
          kind: 'schedule',
          title: prompt.slice(0, 2000),
          label,
          detail: cronExpr
            ? `Antigravity cron · ${cronExpr}${isDaemon ? ' · daemon' : ''}`
            : durSec !== undefined
              ? `Antigravity timer · ${durSec}s`
              : 'Antigravity schedule',
          state,
          startedAt,
          endedAt: null,
          exitCode: null,
          nextRunAt,
          ...(durSec !== undefined ? { durationSeconds: durSec } : {}),
          ...(cronExpr ? { cronExpression: cronExpr } : {}),
          ...(isDaemon ? { isDaemon: true } : {}),
          ...(logFileName ? { logFileName } : {}),
        })
      } else {
        const args = matchedCall?.args ?? {}
        const cmd = readString(args.CommandLine ?? args.command ?? args.cmd) ??
          /Task Description:\s*(.+)/i.exec(step.content)?.[1]?.trim() ??
          rawTaskId
        const summary = readString(args.toolSummary ?? args.Description ?? args.description) ?? cmd.slice(0, 120)
        const isDaemon = args.IsDaemon === true || args.is_daemon === true
        const kind: 'shell' | 'task' = matchedCall?.name === 'run_workflow' ? 'task' : 'shell'
        const statusUpper = step.status.toUpperCase()
        const state: MutableTask['state'] =
          statusUpper === 'ERROR' ? 'failed' : statusUpper === 'CANCELLED' ? 'cancelled' : 'running'
        const logOutMatch = /Log output:\s*\n([\s\S]*)$/i.exec(step.content)
        byRawId.set(rawTaskId, {
          id: `${AGY_TASK_ID_PREFIX}${rawTaskId}`,
          rawTaskId,
          kind,
          title: cmd.slice(0, 2000),
          label: summary.slice(0, 200),
          detail: cwd || location.sessionDir,
          state,
          startedAt,
          endedAt: null,
          exitCode: null,
          nextRunAt: null,
          ...(isDaemon ? { isDaemon: true } : {}),
          ...(logFileName ? { logFileName } : {}),
          ...(logOutMatch?.[1]?.trim() ? { outputPreview: logOutMatch[1].trim().slice(0, 8000) } : {}),
        })
      }
      continue
    }

    if (step.type === 'SYSTEM_MESSAGE' && step.content) {
      // Example: [Message] timestamp=2026-11-23T16:45:52Z sender=<convId>/task-2558 priority=... content=...
      const senderMatch = /\bsender=([^\s\]]+)/.exec(step.content)
      const sender = senderMatch?.[1]?.trim()
      if (!sender) continue
      const existing = byRawId.get(sender)
      if (!existing) continue
      const tsMatch = /\btimestamp=([^\s\]]+)/.exec(step.content)
      const msgMs = parseIsoEpochMs(tsMatch?.[1]) ?? stepMs ?? Date.now()
      const exitCode = parseAgyExitCodeFromText(step.content)
      if (existing.kind === 'schedule' && existing.cronExpression) {
        // Recurring cron keeps running until cancelled or max iterations reached.
        if (/stopped|completed|cancelled|finished/i.test(step.content)) {
          existing.state = /cancelled/i.test(step.content) ? 'cancelled' : 'completed'
          existing.endedAt = msgMs
        }
      } else {
        existing.endedAt = msgMs
        existing.exitCode = exitCode
        if (/cancelled|killed/i.test(step.content)) {
          existing.state = 'cancelled'
        } else if (exitCode !== null && exitCode !== 0) {
          existing.state = 'failed'
        } else if (/\bfailed\b|\berror\b/i.test(step.content) && exitCode === null && existing.kind !== 'schedule') {
          existing.state = 'failed'
        } else {
          existing.state = 'completed'
        }
      }
    }
  }

  // Also include any `.system_generated/tasks/task-*.log` files not yet captured in the transcript tail.
  for (const [fileName, fileStat] of taskFileStats.entries()) {
    const stem = fileName.replace(/\.log$/i, '')
    const fullRawId = `${location.externalSessionId}/${stem}`
    if (byRawId.has(fullRawId) || byRawId.has(stem)) continue
    const recentlyModified = Date.now() - fileStat.mtimeMs < 15_000
    byRawId.set(fullRawId, {
      id: `${AGY_TASK_ID_PREFIX}${fullRawId}`,
      rawTaskId: fullRawId,
      kind: 'shell',
      title: stem,
      label: `Background task · ${stem}`,
      detail: cwd || location.tasksDir,
      state: recentlyModified ? 'running' : 'completed',
      startedAt: Math.trunc(fileStat.mtimeMs),
      endedAt: recentlyModified ? null : Math.trunc(fileStat.mtimeMs),
      exitCode: null,
      nextRunAt: null,
      logFileName: fileName,
    })
  }

  return [...byRawId.values()]
}
