// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Folds the daemon's ordered event stream into displayable transcript blocks.
 *
 * Pure-ish by design: `push` consumes one wire event, `snapshot` returns the
 * current fold, `finalize` flushes streaming buffers so an ended turn leaves
 * no trailing scratch state. Events group into contiguous RUNS in stream
 * order — think, tools, think, tools, text each open their own block, so the
 * transcript reads like the turn actually unfolded instead of appending
 * everything to one thinking card and one tools card.
 */

import type { AgentMember, Block, ToolContentBlock, ToolDiffEntry, ToolItem } from './types.js'
import { reflectionNotice, type ReflectionNotice } from './learning.js'
import { harnessOriginOf } from './types.js'
import { transcriptContent, isLegacyHarnessPrompt } from './transcriptContent.js'

/** One contiguous run of same-kind scratch events inside an active turn. */
interface ToolRun {
  readonly kind: 'tools'
  readonly order: string[]
}
interface TextRun {
  readonly kind: 'text'
  text: string
}
interface ThinkingRun {
  readonly kind: 'thinking'
  text: string
}
interface NoticeRun { readonly kind: 'notice'; readonly text: string; readonly error: boolean }

/**
 * Daemon notices are written for the TUI, where `/runs inspect <id>` is a
 * command you can actually type. In the desktop it is a raw UUID on its own
 * line that does nothing — and a failed run already appears in the Activity
 * rail, where its output is one click away. Two dead lines per failure adds
 * up fast: a task with a handful of failed commands spends more of the
 * transcript on unfollowable instructions than on the failures themselves.
 *
 * Only the trailing slash-command line is dropped; the message keeps every
 * word that describes what happened.
 */
export function noticeText(message: string): string {
  return message
    .split('\n')
    .filter(line => !/^\s*\/runs\s+inspect\s+\S+\s*$/.test(line))
    .join('\n')
    .trim() || message.trim()
}
interface ReflectionRun { readonly kind: 'reflection'; readonly reflection: ReflectionNotice }
type Run = ToolRun | TextRun | ThinkingRun | NoticeRun | ReflectionRun

/** Coarse duration label from a wire millisecond count. */
function dur(ms: unknown): string {
  if (typeof ms !== 'number' || !Number.isFinite(ms)) return ''
  return ms >= 10_000 ? `${Math.round(ms / 1000)}s` : `${(ms / 1000).toFixed(1)}s`
}

/** Human tool verb: `ext.fs.read` → `read`, `FileEditTool` → `edit`. */
function verbOf(name: unknown): string {
  const raw = typeof name === 'string' ? name : ''
  const tail = (raw.split(/[.:]/).pop() ?? raw).replace(/tool$/i, '')
  // CamelCase names arrive as one word on the wire; split them so the trail
  // reads `edit`, not `fileedittool`.
  const split = tail.replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase().trim()
  return split || raw.toLowerCase()
}

/** Keys whose value summarizes a call best, in preference order. */
const ARG_KEYS = [
  'file_path', 'path', 'Path',
  'cmd', 'command', 'CommandLine', 'toolSummary',
  'pattern', 'Pattern', 'query', 'Query',
  'url', 'skill', 'name',
]

/** Maximum inline display characters retained per tool input/output string (64 KB). */
export const MAX_TOOL_DETAIL_CHARS = 65_536

/** Bound giant tool input/output payloads so multi-MB reads cannot bloat React state & DOM. */
export function capToolDetail(value: string, limit = MAX_TOOL_DETAIL_CHARS): string {
  if (value.length <= limit) return value
  const kb = Math.max(1, Math.round(value.length / 1024))
  return `${value.slice(0, limit)}\n… (truncated, ${kb} KB total)`
}

function detailOf(value: unknown): string {
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return ''
    if (value.length > MAX_TOOL_DETAIL_CHARS) return capToolDetail(value)
    try {
      return capToolDetail(JSON.stringify(JSON.parse(trimmed) as unknown, null, 2))
    } catch {
      return capToolDetail(value)
    }
  }
  if (value === undefined || value === null) return ''
  try {
    return capToolDetail(JSON.stringify(value, null, 2))
  } catch {
    return capToolDetail(String(value))
  }
}

/** First-argument-ish summary of a tool call's arguments (or structured diff/locations fallback). */
function argOf(args: unknown, structuredDiff?: StructuredDiffPayload | null, locations?: unknown): string {
  const clip = (value: string): string => (value.length > 72 ? `${value.slice(0, 69)}…` : value)
  const parsed = parseArgs(args)
  const cronExpr = typeof parsed.CronExpression === 'string'
    ? parsed.CronExpression.trim()
    : typeof parsed.cron_expression === 'string'
      ? parsed.cron_expression.trim()
      : ''
  const durSec = typeof parsed.DurationSeconds === 'number' && Number.isFinite(parsed.DurationSeconds)
    ? parsed.DurationSeconds
    : typeof parsed.duration_seconds === 'number' && Number.isFinite(parsed.duration_seconds)
      ? parsed.duration_seconds
      : null
  const schedPrompt = typeof parsed.Prompt === 'string'
    ? parsed.Prompt.trim()
    : typeof parsed.prompt === 'string'
      ? parsed.prompt.trim()
      : ''
  if (cronExpr || durSec !== null) {
    const timing = cronExpr || `in ${durSec}s`
    return clip(schedPrompt ? `${timing} · ${schedPrompt}` : timing)
  }
  for (const key of ARG_KEYS) {
    const value = parsed[key]
    if (typeof value === 'string' && value.trim()) return clip(value)
  }
  if (structuredDiff?.path) return clip(structuredDiff.path)
  const locPath = firstLocationPath(locations)
  if (locPath) return clip(locPath)
  const first = Object.values(parsed)[0]
  if (typeof first === 'string' && first.trim()) return clip(first)
  if (typeof args === 'string') {
    const trimmed = args.trim()
    // Never leak raw or truncated JSON objects/arrays as a human-readable argument.
    if (trimmed && !trimmed.startsWith('{') && !trimmed.startsWith('[')) return clip(trimmed)
  }
  return ''
}

/** Extract the text of a turn_begin user_input (string or Part[]). */
export function userTextOf(payload: Readonly<Record<string, unknown>>): string {
  if (typeof payload.user_input !== 'string') return ''
  return payload.user_input
}

const SWARM_WAKE_PATTERN = /^(?:Immortals|Swarm) (?:board #\d+|worker error)\b/
const SWARM_BOARD_WAKE_DETAIL = /^(?:Immortals|Swarm) board #(\d+)\s*\((@[^)]+)\)\s*$/
const SWARM_ERROR_WAKE_DETAIL = /^(?:Immortals|Swarm) worker error\s*\((@[^)]+)\)\s*$/
const LEGACY_BOARD_WAKE_PATTERN = /^\[board\] the latest message is #\d+/
const LEGACY_BOARD_WAKE_DETAIL = /^\[board\] the latest message is #\d+[^\n]*New board message #(\d+) from (@[^\s]+)/

/** True when text is an internal Immortals/Swarm wake prompt rather than human chat. */
export function isInternalSwarmWakeText(text: string): boolean {
  const trimmed = text.trim()
  return SWARM_WAKE_PATTERN.test(trimmed) || LEGACY_BOARD_WAKE_PATTERN.test(trimmed)
}

/** Format an internal coordinator wake into a quiet, compact notice line. */
export function formatInternalWakeNotice(text: string): string {
  const trimmed = text.trim()
  const boardMatch = SWARM_BOARD_WAKE_DETAIL.exec(trimmed)
  if (boardMatch) return `Immortals · #${boardMatch[1]} from ${boardMatch[2]}`
  const errMatch = SWARM_ERROR_WAKE_DETAIL.exec(trimmed)
  if (errMatch) return `Immortals · worker error from ${errMatch[1]}`
  const legacyMatch = LEGACY_BOARD_WAKE_DETAIL.exec(trimmed)
  if (legacyMatch) return `Immortals · #${legacyMatch[1]} from ${legacyMatch[2]}`
  return trimmed
}

/** Normalize tool names across Xerxes CamelCase and ACP snake_case (`edit_file` → `editfile`). */
export function normalizeToolName(name: unknown): string {
  if (typeof name !== 'string') return ''
  return (name.split(/[.:/]/).pop() ?? name).trim().toLowerCase().replace(/[\s_-]+/g, '')
}

/** Structured file diff metadata attached by ACP `tool_call`, `tool_result`, or `approval_request`. */
export interface StructuredDiffPayload {
  readonly path: string
  readonly oldText?: string
  readonly newText?: string
}

/** Parse a wire structured diff (`{ path, old_text?, new_text? }` or camelCase equivalents). */
export function parseStructuredDiff(value: unknown): StructuredDiffPayload | null {
  if (!isRecord(value)) return null
  const path = typeof value.path === 'string' && value.path.trim() ? value.path.trim() : ''
  const oldText = typeof value.old_text === 'string'
    ? value.old_text
    : typeof value.oldText === 'string'
      ? value.oldText
      : undefined
  const newText = typeof value.new_text === 'string'
    ? value.new_text
    : typeof value.newText === 'string'
      ? value.newText
      : undefined
  if (!path && oldText === undefined && newText === undefined) return null
  return {
    path,
    ...(oldText !== undefined ? { oldText } : {}),
    ...(newText !== undefined ? { newText } : {}),
  }
}

/** Parse all structured diffs from `diffs[]` (falling back to a single `diff`). */
export function parseStructuredDiffs(diffsValue: unknown, singleDiffValue?: unknown): readonly StructuredDiffPayload[] {
  const out: StructuredDiffPayload[] = []
  if (Array.isArray(diffsValue)) {
    for (const entry of diffsValue) {
      const parsed = parseStructuredDiff(entry)
      if (parsed) out.push(parsed)
    }
  }
  if (out.length === 0 && singleDiffValue !== undefined) {
    const single = parseStructuredDiff(singleDiffValue)
    if (single) out.push(single)
  }
  return out
}

/** Parse rich ACP/MCP tool content blocks (`text`, `image`, `audio`, `resource`, `resource_link`, `terminal`). */
export function parseToolContentBlocks(value: unknown): readonly ToolContentBlock[] {
  if (!Array.isArray(value)) return []
  const out: ToolContentBlock[] = []
  for (const raw of value) {
    if (!isRecord(raw)) continue
    const type = typeof raw.type === 'string' ? raw.type.trim().toLowerCase() : ''
    if (type === 'text' && typeof raw.text === 'string') {
      out.push({ type: 'text', text: raw.text })
      continue
    }
    if (type === 'image') {
      const data = typeof raw.data === 'string' ? raw.data : ''
      const mimeType =
        (typeof raw.mime_type === 'string' && raw.mime_type.trim())
        || (typeof raw.mimeType === 'string' && raw.mimeType.trim())
        || ''
      const uri = typeof raw.uri === 'string' && raw.uri.trim() ? raw.uri.trim() : undefined
      if (data && mimeType) {
        out.push({ type: 'image', data, mime_type: mimeType, ...(uri ? { uri } : {}) })
      }
      continue
    }
    if (type === 'audio') {
      const data = typeof raw.data === 'string' ? raw.data : ''
      const mimeType =
        (typeof raw.mime_type === 'string' && raw.mime_type.trim())
        || (typeof raw.mimeType === 'string' && raw.mimeType.trim())
        || ''
      if (data && mimeType) {
        out.push({ type: 'audio', data, mime_type: mimeType })
      }
      continue
    }
    if (type === 'resource') {
      const nested = isRecord(raw.resource) ? raw.resource : raw
      const uri = typeof nested.uri === 'string' && nested.uri.trim() ? nested.uri.trim() : ''
      if (!uri) continue
      const mimeType =
        (typeof nested.mime_type === 'string' && nested.mime_type.trim())
        || (typeof nested.mimeType === 'string' && nested.mimeType.trim())
        || undefined
      const text = typeof nested.text === 'string' ? nested.text : undefined
      const blob = typeof nested.blob === 'string' ? nested.blob : undefined
      out.push({
        type: 'resource',
        uri,
        ...(mimeType ? { mime_type: mimeType } : {}),
        ...(text !== undefined ? { text } : {}),
        ...(blob !== undefined ? { blob } : {}),
      })
      continue
    }
    if (type === 'resource_link') {
      const uri = typeof raw.uri === 'string' && raw.uri.trim() ? raw.uri.trim() : ''
      if (!uri) continue
      const name = typeof raw.name === 'string' && raw.name.trim() ? raw.name.trim() : undefined
      const title = typeof raw.title === 'string' && raw.title.trim() ? raw.title.trim() : undefined
      const description = typeof raw.description === 'string' && raw.description.trim() ? raw.description.trim() : undefined
      const mimeType =
        (typeof raw.mime_type === 'string' && raw.mime_type.trim())
        || (typeof raw.mimeType === 'string' && raw.mimeType.trim())
        || undefined
      out.push({
        type: 'resource_link',
        uri,
        ...(name ? { name } : {}),
        ...(title ? { title } : {}),
        ...(description ? { description } : {}),
        ...(mimeType ? { mime_type: mimeType } : {}),
      })
      continue
    }
    if (type === 'terminal') {
      const terminalId =
        (typeof raw.terminal_id === 'string' && raw.terminal_id.trim())
        || (typeof raw.terminalId === 'string' && raw.terminalId.trim())
        || ''
      if (terminalId) out.push({ type: 'terminal', terminal_id: terminalId })
    }
  }
  return out
}

function diffCountsOf(diff: StructuredDiffPayload): { readonly adds: number; readonly dels: number } {
  const oldText = diff.oldText
  const newText = diff.newText
  if (oldText && newText) {
    const span = trimmedChangedLines(oldText, newText, 0)
    return { adds: span.after.length, dels: span.before.length }
  }
  return {
    adds: newText ? splitDiffLines(newText).length : 0,
    dels: oldText ? splitDiffLines(oldText).length : 0,
  }
}

function toToolDiffEntries(diffs: readonly StructuredDiffPayload[]): readonly ToolDiffEntry[] {
  return diffs.map(entry => {
    const counts = diffCountsOf(entry)
    return {
      adds: counts.adds,
      dels: counts.dels,
      ...(entry.path ? { path: entry.path } : {}),
      ...(entry.oldText !== undefined ? { old_text: entry.oldText } : {}),
      ...(entry.newText !== undefined ? { new_text: entry.newText } : {}),
    }
  })
}

/** Extract the first file path from an ACP `locations` array (`[{ path, line? }]`). */
export function firstLocationPath(locations: unknown): string {
  if (!Array.isArray(locations)) return ''
  for (const entry of locations) {
    if (isRecord(entry) && typeof entry.path === 'string' && entry.path.trim()) {
      return entry.path.trim()
    }
  }
  return ''
}

function splitDiffLines(text: string): string[] {
  return text ? text.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n') : []
}

/**
 * Strip identical leading and trailing lines between two full-file snapshots,
 * retaining up to `contextLines` of surrounding context on each side.
 */
export function trimmedChangedLines(
  before: string,
  after: string,
  contextLines = 0,
): { readonly before: readonly string[]; readonly after: readonly string[] } {
  const a = splitDiffLines(before)
  const b = splitDiffLines(after)
  let prefix = 0
  const maxPrefix = Math.min(a.length, b.length)
  while (prefix < maxPrefix && a[prefix] === b[prefix]) prefix += 1
  if (prefix === a.length && prefix === b.length) return { before: [], after: [] }
  let suffix = 0
  const maxSuffix = Math.min(a.length - prefix, b.length - prefix)
  while (suffix < maxSuffix && a[a.length - 1 - suffix] === b[b.length - 1 - suffix]) suffix += 1
  const ctx = Math.max(0, Math.floor(contextLines))
  const start = Math.max(0, prefix - ctx)
  const endA = Math.min(a.length, a.length - suffix + ctx)
  const endB = Math.min(b.length, b.length - suffix + ctx)
  return {
    before: a.slice(start, endA),
    after: b.slice(start, endB),
  }
}

/** Edit-family tool names whose arguments carry real diff material. */
const EDIT_TOOLS = new Set(['fileedittool', 'editfile', 'writefile', 'appendfile'])

export function isEditTool(name: unknown): boolean {
  return EDIT_TOOLS.has(normalizeToolName(name))
}

/**
 * Best-effort single-pass extractor for truncated JSON object previews
 * (e.g. `{"CommandLine":"ENABLE_DISTRIBUTED_INIT=0 ...` cut off at 200 chars).
 * Recovers all complete top-level primitive fields plus any partially-written
 * trailing string field without ever throwing.
 */
function parseTruncatedJsonObject(raw: string): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  const len = raw.length
  let i = 0
  while (i < len && /\s/.test(raw[i]!)) i += 1
  if (raw[i] !== '{') return out
  i += 1

  const skipWs = (): void => {
    while (i < len && /\s/.test(raw[i]!)) i += 1
  }

  const readString = (allowUnterminated: boolean): string | null => {
    if (raw[i] !== '"') return null
    i += 1
    let acc = ''
    while (i < len) {
      const ch = raw[i]!
      if (ch === '"') {
        i += 1
        return acc
      }
      if (ch === '\\') {
        if (i + 1 >= len) {
          i = len
          return allowUnterminated ? acc : null
        }
        const esc = raw[i + 1]!
        if (esc === 'u' && i + 5 < len) {
          const hex = raw.slice(i + 2, i + 6)
          if (/^[0-9a-fA-F]{4}$/.test(hex)) {
            acc += String.fromCharCode(parseInt(hex, 16))
            i += 6
            continue
          }
        }
        if (esc === '"') acc += '"'
        else if (esc === '\\') acc += '\\'
        else if (esc === '/') acc += '/'
        else if (esc === 'n') acc += '\n'
        else if (esc === 'r') acc += '\r'
        else if (esc === 't') acc += '\t'
        else if (esc === 'b') acc += '\b'
        else if (esc === 'f') acc += '\f'
        else acc += esc
        i += 2
        continue
      }
      acc += ch
      i += 1
    }
    return allowUnterminated ? acc.replace(/…$/, '').trimEnd() : null
  }

  const skipNested = (): boolean => {
    const open = raw[i]
    if (open !== '{' && open !== '[') return false
    let depth = 0
    let inStr = false
    while (i < len) {
      const ch = raw[i]!
      if (inStr) {
        if (ch === '\\') i += 2
        else {
          if (ch === '"') inStr = false
          i += 1
        }
        continue
      }
      if (ch === '"') { inStr = true; i += 1; continue }
      if (ch === '{' || ch === '[') depth += 1
      else if (ch === '}' || ch === ']') {
        depth -= 1
        if (depth === 0) { i += 1; return true }
      }
      i += 1
    }
    return false
  }

  while (i < len) {
    skipWs()
    if (i >= len || raw[i] === '}') break
    if (raw[i] === ',') { i += 1; continue }
    const key = readString(false)
    if (key === null) break
    skipWs()
    if (raw[i] !== ':') break
    i += 1
    skipWs()
    if (i >= len) break
    const ch = raw[i]!
    if (ch === '"') {
      const val = readString(true)
      if (val !== null && val !== '') out[key] = val
    } else if (raw.startsWith('true', i)) {
      out[key] = true
      i += 4
    } else if (raw.startsWith('false', i)) {
      out[key] = false
      i += 5
    } else if (raw.startsWith('null', i)) {
      out[key] = null
      i += 4
    } else if (ch === '-' || (ch >= '0' && ch <= '9')) {
      const start = i
      while (i < len && /[0-9eE.+\-]/.test(raw[i]!)) i += 1
      const num = Number(raw.slice(start, i))
      if (Number.isFinite(num)) out[key] = num
    } else if (ch === '{' || ch === '[') {
      if (!skipNested()) break
    } else {
      break
    }
  }
  return out
}

/**
 * Parse tool-call arguments (JSON string or object) without throwing.
 * Recovers fields from truncated JSON previews (e.g. older remote daemons that
 * clip `inflight.tools[i].arguments` at 200 chars) so tool rows and status
 * phrases never leak raw JSON or fall back to empty placeholders.
 */
export function parseArgs(args: unknown): Record<string, unknown> {
  if (typeof args === 'string' && args.trim()) {
    const trimmed = args.trim()
    try {
      const parsed = JSON.parse(trimmed) as unknown
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>
      }
    } catch {
      if (trimmed.startsWith('{')) {
        return parseTruncatedJsonObject(trimmed)
      }
      return {}
    }
  }
  if (args && typeof args === 'object' && !Array.isArray(args)) return args as Record<string, unknown>
  return {}
}

function firstStringField(record: Readonly<Record<string, unknown>>, keys: readonly string[]): string | undefined {
  for (const key of keys) {
    const value = record[key]
    if (typeof value === 'string') return value
  }
  return undefined
}

/**
 * File + line delta an edit call will produce, or null for other tools.
 * Supports both Xerxes snake_case (`file_path`, `old_string`, `new_string`, `content`),
 * Antigravity ACP PascalCase (`Path`, `Target`, `Replacement`, `Content`), and
 * structured ACP `diff` / `locations` payloads.
 */
export function editStatsOf(
  name: unknown,
  args: unknown,
  structuredDiff?: unknown,
  locations?: unknown,
): { path: string; adds: number; dels: number; isNew: boolean } | null {
  const parsedDiff = parseStructuredDiff(structuredDiff)
  if (!isEditTool(name) && !parsedDiff) return null
  const parsed = parseArgs(args)
  const path =
    (typeof parsed.file_path === 'string' && parsed.file_path.trim())
    || (typeof parsed.path === 'string' && parsed.path.trim())
    || (typeof parsed.Path === 'string' && parsed.Path.trim())
    || parsedDiff?.path
    || firstLocationPath(locations)
    || ''
  if (!path) return null
  const argOld = firstStringField(parsed, ['old_string', 'Target', 'old_text', 'oldText'])
  const argNew = firstStringField(parsed, ['new_string', 'Replacement', 'new_text', 'newText', 'content', 'Content'])
  const hasArgText = argOld !== undefined || argNew !== undefined
  const norm = normalizeToolName(name)
  const mode = typeof parsed.edit_mode === 'string' ? parsed.edit_mode : 'search_replace'
  const oldText = hasArgText ? argOld : parsedDiff?.oldText
  const newText = hasArgText ? argNew : parsedDiff?.newText
  const isNew = norm === 'writefile' && mode !== 'search_replace' ? true : !oldText
  if (!hasArgText && parsedDiff && !isNew && typeof oldText === 'string' && typeof newText === 'string') {
    const span = trimmedChangedLines(oldText, newText, 0)
    return {
      path,
      adds: span.after.length,
      dels: span.before.length,
      isNew: false,
    }
  }
  const count = (value: unknown): number =>
    typeof value === 'string' && value.length ? splitDiffLines(value).length : 0
  return {
    path,
    adds: count(newText),
    dels: isNew ? 0 : count(oldText),
    isNew,
  }
}

/** The daemon's "Run finished" notice for a subagent run. */
export function isAgentRunNotice(payload: Readonly<Record<string, unknown>>): boolean {
  return payload.title === 'Run finished' && typeof payload.body === 'string' && /^agent (succeeded|failed|cancelled|interrupted):/.test(payload.body)
}

const CONTROL_STATE_NOTICE_PATTERNS: readonly RegExp[] = [
  /^Model set to .+$/i,
  /^Agents:\s*`?(?:off|auto|eager)`?\.?$/i,
  /^Thinking(?:\s+effort\s+set\s+to)?:\s*`?[^`\n]+`?\.?$/i,
  /^Permission mode:\s*`?(?:accept-all|auto|manual|plan)`?\.?$/i,
  /^YOLO mode (?:ON|OFF)\.?$/i,
  /^Switched to provider profile\s+`?[^`\n]+`?\.?$/i,
  /^provider\s+`?[^`\n]+`?\s+(?:saved and activated|deleted)\.?$/i,
]

/**
 * Routine composer/picker state-change confirmations (`Model set to …`,
 * `Agents: \`eager\`.`, `Thinking: \`high\`.`, `Permission mode: \`auto\`.`).
 * The composer trigger and picker already reflect these changes inline, so
 * repeating them as `(i)` rows clutters the conversation and welcome screen.
 * Warnings, errors, and multi-line command output are never suppressed.
 */
export function isControlStateNotice(payload: Readonly<Record<string, unknown>>): boolean {
  const severity = String(payload.severity ?? payload.level ?? 'info').toLowerCase()
  if (severity.includes('error') || severity.includes('fatal') || severity.includes('warn')) return false
  const raw =
    (typeof payload.body === 'string' && payload.body) ||
    (typeof payload.message === 'string' && payload.message) ||
    ''
  const trimmed = raw.trim()
  if (!trimmed || trimmed.includes('\n')) return false
  return CONTROL_STATE_NOTICE_PATTERNS.some(pattern => pattern.test(trimmed))
}

export class BlockBuilder {
  private blocks: Block[] = []
  private seq = 1
  /** Contiguous same-kind event runs, in stream order. */
  private runs: Run[] = []
  private runIds = new WeakMap<Run, number>()
  /** Structural-sharing cache for settled (non-trailing) live runs in the current turn. */
  private settledRunBlocks = new WeakMap<Run, Block>()
  private cachedSnapshot: readonly Block[] | null = null
  private cachedSnapshotActive: boolean | null = null
  /** Every tool call/result of the turn by id, regardless of which run holds the row. */
  private tools = new Map<string, ToolItem>()
  /**
   * The turn's spawned-subagents card. Deliberately NOT a run: it updates in
   * place as members start/finish, mid-turn and (via the committed copy)
   * after the turn ends while background children settle. Cleared on
   * finalize so the next turn's spawn opens a fresh card.
   */
  private agentsCard: { readonly id: number; readonly members: readonly AgentMember[] } | null = null
  /** The run that was trailing when the card opened — normally the spawn call's tools run. */
  private agentsAnchor: Run | null = null

  private invalidateSnapshot(): void {
    this.cachedSnapshot = null
    this.cachedSnapshotActive = null
  }

  private nextId(): number {
    return this.seq++
  }

  /**
   * Create-or-update the turn's spawned-agents card (dsh in-chat batch).
   * Members merge by key in the STORE; the builder just renders the map.
   * Mid-turn the card trails the live runs; at finalize it commits after
   * them — chronologically right behind the spawn row either way.
   */
  pushAgents(members: readonly AgentMember[]): void {
    if (members.length === 0 && !this.agentsCard) return
    this.invalidateSnapshot()
    if (this.agentsCard) {
      this.agentsCard = { id: this.agentsCard.id, members }
      const committed = this.blocks.findIndex(block => block.id === this.agentsCard!.id)
      if (committed >= 0) {
        const next = this.blocks.slice()
        next[committed] = { kind: 'agents', id: this.agentsCard.id, members }
        this.blocks = next
      }
      return
    }
    this.agentsCard = { id: this.nextId(), members }
    this.agentsAnchor = this.runs.at(-1) ?? null
  }

  private agentsBlock(): Block | null {
    const card = this.agentsCard
    if (!card || this.blocks.some(block => block.id === card.id)) return null
    return { kind: 'agents', id: card.id, members: card.members }
  }

  /**
   * The run a kind appends to: the trailing run when it matches, otherwise a
   * freshly opened one — this is what makes think→tools→think produce two
   * thinking blocks instead of appending to the first.
   */
  private openRun(kind: 'tools' | 'text' | 'thinking'): Run {
    const last = this.runs[this.runs.length - 1]
    if (last && last.kind === kind) return last
    const run: Run =
      kind === 'tools' ? { kind: 'tools', order: [] } : kind === 'thinking' ? { kind: 'thinking', text: '' } : { kind: 'text', text: '' }
    this.runIds.set(run, this.nextId())
    this.runs.push(run)
    return run
  }

  /** The run whose trail holds this tool id, scanning newest first. */
  private runHolding(id: string): ToolRun | undefined {
    for (let i = this.runs.length - 1; i >= 0; i--) {
      const run = this.runs[i]
      if (run && run.kind === 'tools' && run.order.includes(id)) return run
    }
    return undefined
  }

  /** Consume one wire event. Unknown types are ignored, not errors. */
  push(type: string, payload: Readonly<Record<string, unknown>>): void {
    switch (type) {
      case 'turn_begin': {
        const raw =
          typeof payload.user_input === 'string'
            ? payload.user_input
            : typeof payload.text === 'string'
              ? payload.text
              : ''
        const text = raw.trim()
        const origin = payload.origin
        if (text && (origin === 'agent' || origin === 'monitor' || isInternalSwarmWakeText(text))) {
          this.pushNotice(formatInternalWakeNotice(text))
        }
        break
      }
      case 'text_part': {
        if (typeof payload.text === 'string' && payload.text) {
          const run = this.openRun('text')
          if (run.kind === 'text') {
            run.text += payload.text
            this.settledRunBlocks.delete(run)
            this.invalidateSnapshot()
          }
        }
        break
      }
      case 'think_part': {
        const delta = typeof payload.think === 'string' ? payload.think : typeof payload.text === 'string' ? payload.text : ''
        if (delta) {
          const run = this.openRun('thinking')
          if (run.kind === 'thinking') {
            run.text += delta
            this.settledRunBlocks.delete(run)
            this.invalidateSnapshot()
          }
        }
        break
      }
      case 'tool_call': {
        const id = typeof payload.id === 'string' ? payload.id : typeof payload.tool_call_id === 'string' ? payload.tool_call_id : `t${this.seq++}`
        const run = this.openRun('tools')
        if (run.kind === 'tools' && !run.order.includes(id)) run.order.push(id)
        const parsedDiffs = parseStructuredDiffs(payload.diffs, payload.diff)
        const structuredDiff = parsedDiffs[0] ?? null
        const diffEntries = parsedDiffs.length > 0 ? toToolDiffEntries(parsedDiffs) : undefined
        const contentBlocks = parseToolContentBlocks(payload.content_blocks ?? payload.contentBlocks)
        const title = typeof payload.title === 'string' && payload.title.trim() ? payload.title.trim() : undefined
        const kind = typeof payload.kind === 'string' && payload.kind.trim() ? payload.kind.trim() : undefined
        const stats = editStatsOf(payload.name, payload.arguments ?? payload.inputs, structuredDiff, payload.locations)
        const totalAdds = diffEntries && diffEntries.length > 1
          ? diffEntries.reduce((sum, entry) => sum + (entry.adds ?? 0), 0)
          : stats?.adds
        const totalDels = diffEntries && diffEntries.length > 1
          ? diffEntries.reduce((sum, entry) => sum + (entry.dels ?? 0), 0)
          : stats?.dels
        const toolDiff = totalAdds !== undefined && totalDels !== undefined
          ? {
              adds: totalAdds,
              dels: totalDels,
              ...(structuredDiff ? {
                ...(structuredDiff.path ? { path: structuredDiff.path } : {}),
                ...(structuredDiff.oldText !== undefined ? { old_text: structuredDiff.oldText } : {}),
                ...(structuredDiff.newText !== undefined ? { new_text: structuredDiff.newText } : {}),
              } : {}),
            }
          : undefined
        const rawArgs = payload.arguments ?? payload.inputs
        const initialOutput = detailOf(payload.output ?? payload.result)
        this.tools.set(id, {
          id,
          verb: verbOf(payload.name),
          arg: argOf(rawArgs, structuredDiff, payload.locations) || title || '',
          dur: '',
          state: 'working',
          name: typeof payload.name === 'string' ? payload.name : '',
          ...(title ? { title } : {}),
          ...(kind ? { kind } : {}),
          input: detailOf(rawArgs),
          output: initialOutput,
          ...(stats ? { path: stats.path } : {}),
          ...(toolDiff ? { diff: toolDiff } : {}),
          ...(diffEntries ? { diffs: diffEntries } : {}),
          ...(contentBlocks.length > 0 ? { contentBlocks } : {}),
        })
        this.settledRunBlocks.delete(run)
        this.invalidateSnapshot()
        break
      }
      case 'tool_update': {
        const id = typeof payload.tool_call_id === 'string' && payload.tool_call_id
          ? payload.tool_call_id
          : typeof payload.id === 'string'
            ? payload.id
            : ''
        if (!id) break
        const existing = this.tools.get(id)
        const parsedDiffs = parseStructuredDiffs(payload.diffs, payload.diff)
        const structuredDiff = parsedDiffs[0] ?? null
        const diffEntries = parsedDiffs.length > 0 ? toToolDiffEntries(parsedDiffs) : existing?.diffs
        const incomingBlocks = parseToolContentBlocks(payload.content_blocks ?? payload.contentBlocks)
        const contentBlocks = incomingBlocks.length > 0 ? incomingBlocks : existing?.contentBlocks
        const toolName = existing?.name || (typeof payload.name === 'string' ? payload.name : '')
        const title = typeof payload.title === 'string' && payload.title.trim() ? payload.title.trim() : existing?.title
        const kind = typeof payload.kind === 'string' && payload.kind.trim() ? payload.kind.trim() : existing?.kind
        const rawInputs = payload.inputs !== undefined ? payload.inputs : payload.arguments
        const nextInput = rawInputs !== undefined ? detailOf(rawInputs) : ''
        const input = nextInput || existing?.input || ''
        const rawOutput = payload.output !== undefined ? payload.output : payload.result
        const nextOutput = rawOutput !== undefined ? detailOf(rawOutput) : ''
        const output = nextOutput || existing?.output || ''
        const stats = (structuredDiff || !existing?.diff)
          ? editStatsOf(toolName, rawInputs ?? (input ? parseArgs(input) : undefined), structuredDiff, payload.locations)
          : null
        const totalAdds = diffEntries && diffEntries.length > 1
          ? diffEntries.reduce((sum, entry) => sum + (entry.adds ?? 0), 0)
          : stats?.adds
        const totalDels = diffEntries && diffEntries.length > 1
          ? diffEntries.reduce((sum, entry) => sum + (entry.dels ?? 0), 0)
          : stats?.dels
        const mergedDiff = structuredDiff
          ? {
              adds: totalAdds ?? existing?.diff?.adds ?? 0,
              dels: totalDels ?? existing?.diff?.dels ?? 0,
              ...(structuredDiff.path ? { path: structuredDiff.path } : {}),
              ...(structuredDiff.oldText !== undefined ? { old_text: structuredDiff.oldText } : {}),
              ...(structuredDiff.newText !== undefined ? { new_text: structuredDiff.newText } : {}),
            }
          : existing?.diff
            ? existing.diff
            : totalAdds !== undefined && totalDels !== undefined
              ? { adds: totalAdds, dels: totalDels }
              : undefined
        const resolvedPath = structuredDiff?.path || firstLocationPath(payload.locations) || stats?.path || existing?.path || ''
        const nextArg = argOf(rawInputs, structuredDiff, payload.locations) || title || existing?.arg || ''
        const rawStatus = typeof payload.status === 'string' ? payload.status.trim().toLowerCase() : ''
        const state: ToolItem['state'] =
          rawStatus === 'failed' || rawStatus === 'error'
            ? 'failed'
            : rawStatus === 'completed' || rawStatus === 'done'
              ? 'done'
              : existing?.state === 'done' || existing?.state === 'failed'
                ? existing.state
                : 'working'
        const item: ToolItem = {
          id,
          verb: existing?.verb ?? verbOf(toolName),
          arg: nextArg,
          dur: existing?.dur ?? '',
          state,
          name: toolName,
          ...(title ? { title } : {}),
          ...(kind ? { kind } : {}),
          input,
          output,
          ...(existing?.error ? { error: existing.error } : {}),
          ...(resolvedPath ? { path: resolvedPath } : {}),
          ...(mergedDiff ? { diff: mergedDiff } : {}),
          ...(diffEntries && diffEntries.length > 0 ? { diffs: diffEntries } : {}),
          ...(contentBlocks && contentBlocks.length > 0 ? { contentBlocks } : {}),
        }
        if (existing) {
          this.tools.set(id, item)
          const owner = this.runHolding(id)
          if (owner) this.settledRunBlocks.delete(owner)
        } else {
          const run = this.openRun('tools')
          if (run.kind === 'tools' && !run.order.includes(id)) run.order.push(id)
          this.tools.set(id, item)
          this.settledRunBlocks.delete(run)
        }
        if (this.blocks.some(block => block.kind === 'tools' && block.items.some(tool => tool.id === id))) {
          this.blocks = this.blocks.map(block => {
            if (block.kind !== 'tools' || !block.items.some(tool => tool.id === id)) return block
            const items = block.items.map(tool => tool.id === id ? item : tool)
            return { ...block, items, running: items.some(tool => tool.state === 'working') }
          })
        }
        this.invalidateSnapshot()
        break
      }
      case 'tool_result': {
        const id = typeof payload.tool_call_id === 'string' ? payload.tool_call_id : typeof payload.id === 'string' ? payload.id : ''
        const existing = this.tools.get(id)
        const parsedDiffs = parseStructuredDiffs(payload.diffs, payload.diff)
        const structuredDiff = parsedDiffs[0] ?? null
        const diffEntries = parsedDiffs.length > 0 ? toToolDiffEntries(parsedDiffs) : existing?.diffs
        const incomingBlocks = parseToolContentBlocks(payload.content_blocks ?? payload.contentBlocks)
        const contentBlocks = incomingBlocks.length > 0 ? incomingBlocks : existing?.contentBlocks
        const toolName = existing?.name || (typeof payload.name === 'string' ? payload.name : '')
        const title = typeof payload.title === 'string' && payload.title.trim() ? payload.title.trim() : existing?.title
        const kind = typeof payload.kind === 'string' && payload.kind.trim() ? payload.kind.trim() : existing?.kind
        const stats = (structuredDiff || !existing?.diff)
          ? editStatsOf(toolName, payload.arguments ?? (existing?.input ? parseArgs(existing.input) : undefined), structuredDiff, payload.locations)
          : null
        const totalAdds = diffEntries && diffEntries.length > 1
          ? diffEntries.reduce((sum, entry) => sum + (entry.adds ?? 0), 0)
          : stats?.adds
        const totalDels = diffEntries && diffEntries.length > 1
          ? diffEntries.reduce((sum, entry) => sum + (entry.dels ?? 0), 0)
          : stats?.dels
        const mergedDiff = structuredDiff
          ? {
              adds: totalAdds ?? existing?.diff?.adds ?? 0,
              dels: totalDels ?? existing?.diff?.dels ?? 0,
              ...(structuredDiff.path ? { path: structuredDiff.path } : {}),
              ...(structuredDiff.oldText !== undefined ? { old_text: structuredDiff.oldText } : {}),
              ...(structuredDiff.newText !== undefined ? { new_text: structuredDiff.newText } : {}),
            }
          : existing?.diff
            ? existing.diff
            : totalAdds !== undefined && totalDels !== undefined
              ? { adds: totalAdds, dels: totalDels }
              : undefined
        const resolvedPath = existing?.path || stats?.path || structuredDiff?.path || firstLocationPath(payload.locations) || ''
        const rawResultOutput = detailOf(payload.return_value ?? payload.result ?? payload.output)
        const output = rawResultOutput || existing?.output || ''
        // Saved native tool results predate an explicit error field. Permission
        // to execute is independent of whether execution succeeded.
        const error = toolFailureText(payload, output)
        const item: ToolItem = {
          id,
          verb: existing?.verb ?? verbOf(payload.name),
          arg: existing?.arg || argOf(payload.arguments, structuredDiff, payload.locations) || title || '',
          dur: dur(payload.duration_ms),
          state: error ? 'failed' : 'done',
          name: toolName,
          ...(title ? { title } : {}),
          ...(kind ? { kind } : {}),
          input: existing?.input || detailOf(payload.arguments),
          output,
          ...(error ? { error } : {}),
          ...(resolvedPath ? { path: resolvedPath } : {}),
          ...(mergedDiff ? { diff: mergedDiff } : {}),
          ...(diffEntries && diffEntries.length > 0 ? { diffs: diffEntries } : {}),
          ...(contentBlocks && contentBlocks.length > 0 ? { contentBlocks } : {}),
        }
        if (existing) {
          this.tools.set(id, item)
          const owner = this.runHolding(id)
          if (owner) this.settledRunBlocks.delete(owner)
        } else {
          // A result with no matching call (replay quirks): the row lands on
          // the trailing tools run, opening one if the stream just switched.
          const run = this.openRun('tools')
          if (run.kind === 'tools' && !run.order.includes(id)) run.order.push(id)
          this.tools.set(id, item)
          this.settledRunBlocks.delete(run)
        }
        // Steering can place a user bubble after a still-running tool group.
        // Its eventual result updates that original group, not a duplicate.
        // Avoid O(N) array cloning when no committed block holds this tool id.
        if (this.blocks.some(block => block.kind === 'tools' && block.items.some(tool => tool.id === id))) {
          this.blocks = this.blocks.map(block => {
            if (block.kind !== 'tools' || !block.items.some(tool => tool.id === id)) return block
            const items = block.items.map(tool => tool.id === id ? item : tool)
            return { ...block, items, running: items.some(tool => tool.state === 'working') }
          })
        }
        this.invalidateSnapshot()
        break
      }
      case 'notification': {
        if (payload.category === 'reflection' && payload.type === 'proposal') {
          const reflection = reflectionNotice(payload.reflection)
          if (reflection) {
            if (this.blocks.some(block => block.kind === 'reflection' && block.reflection.id === reflection.id) || this.runs.some(run => run.kind === 'reflection' && run.reflection.id === reflection.id)) break
            const run: ReflectionRun = { kind: 'reflection', reflection }, id = this.nextId()
            if (this.runs.length) { this.runIds.set(run, id); this.runs.push(run) }
            else this.blocks = [...this.blocks, { ...run, id }]
            this.invalidateSnapshot()
            break
          }
        }
        // Slash-command resume emits history notifications rather than live
        // tool events. Keep their semantics without pretending the bounded
        // replay payload contains the original output.
        if (payload.category === 'history') {
          const replay = isRecord(payload.payload) ? payload.payload : {}
          const body = typeof payload.body === 'string' ? payload.body : ''
          if (payload.type === 'replay_tool') {
            const id = `replay-${this.nextId()}`
            this.push('tool_call', { id, name: replay.name, arguments: replay.context })
            this.push('tool_result', {
              tool_call_id: id, name: replay.name, duration_ms: replay.duration_ms,
              result: 'History preview only. Original output is not included in this replay.',
              ...(replay.ok === false ? { error: typeof replay.preview === 'string' && replay.preview ? replay.preview : 'Tool execution failed.' } : {}),
            })
            break
          }
          if (payload.type === 'replay_user') {
            this.pushUser(body.replace(/^✨\s?/, ''))
            break
          }
          if (payload.type === 'replay_assistant') {
            this.finalize()
            if (typeof replay.thinking === 'string' && replay.thinking.trim()) this.push('think_part', { think: replay.thinking })
            this.push('text_part', { text: body })
            this.finalize()
            break
          }
          if (payload.type === 'resumed') { this.finalize(); break }
        }
        const message =
          (typeof payload.body === 'string' && payload.body) ||
          (typeof payload.message === 'string' && payload.message) ||
          ''
        if (!message) break
        // A subagent's run closing ("agent succeeded: … · attempt 2", once per
        // revision) is already on its Subagents card; as rows it repeated.
        if (isAgentRunNotice(payload)) break
        // Composer/picker state echoes ("Model set to …", "Agents: `eager`.")
        // are already reflected directly in the picker trigger and controls.
        if (isControlStateNotice(payload)) break
        const severity = String(payload.severity ?? payload.level ?? 'info').toLowerCase()
        const notice: NoticeRun = { kind: 'notice', error: severity.includes('error') || severity.includes('fatal'), text: noticeText(message) }
        const id = this.nextId()
        if (this.runs.length) { this.runIds.set(notice, id); this.runs.push(notice) }
        else this.blocks = [...this.blocks, { ...notice, id }]
        this.invalidateSnapshot()
        break
      }
      default:
        break
    }
  }

  /** A quiet system/wake notice row landing at the fold's end. */
  pushNotice(text: string, error = false): number {
    this.commitRuns(false)
    const clean = noticeText(text)
    const last = this.blocks.at(-1)
    if (last && last.kind === 'notice' && last.text === clean && last.error === error) {
      return last.id
    }
    const id = this.nextId()
    this.blocks = [...this.blocks, { kind: 'notice', id, error, text: clean }]
    this.invalidateSnapshot()
    return id
  }

  /** A user line lands at the fold's end (submit ordering, replays). */
  pushUser(text: string, contextSummary = false, images: readonly string[] = []): number {
    // Internal Immortals/Swarm coordinator wakes must render as quiet notices,
    // never as right-aligned human chat bubbles.
    if (!contextSummary && images.length === 0 && isInternalSwarmWakeText(text)) {
      return this.pushNotice(formatInternalWakeNotice(text))
    }
    this.commitRuns(false)
    const id = this.nextId()
    this.blocks = [...this.blocks, { kind: 'user', id, text, ...(contextSummary ? { contextSummary: true } : {}), ...(images.length ? { images } : {}) }]
    this.invalidateSnapshot()
    return id
  }

  /** Remove one specific block by its minted id (e.g. a promoted queued-steer bubble). */
  removeBlockById(blockId: number): boolean {
    const index = this.blocks.findIndex(block => block.id === blockId)
    if (index < 0) return false
    this.blocks = [...this.blocks.slice(0, index), ...this.blocks.slice(index + 1)]
    this.invalidateSnapshot()
    return true
  }

  /**
   * Undo the most recent pushUser matching `text` — a submit the daemon
   * rejected must not linger as a delivered-looking ghost bubble. When
   * `withinTrailingUsers` is true, scans across contiguous trailing user
   * bubbles (e.g. multiple queued follow-ups before a promoted steer).
   */
  rollbackUser(text: string, withinTrailingUsers = false): void {
    for (let i = this.blocks.length - 1; i >= 0; i--) {
      const block = this.blocks[i]!
      if (block.kind === 'user') {
        if (block.text === text) {
          this.blocks = [...this.blocks.slice(0, i), ...this.blocks.slice(i + 1)]
          this.invalidateSnapshot()
          return
        }
        if (!withinTrailingUsers) return
      } else if (withinTrailingUsers) {
        return
      }
    }
  }

  /** A turn boundary with cumulative edit totals — the checkpoint marker. */
  pushCheckpoint(turn: number, totals: { adds: number; dels: number }): void {
    // Only turns that changed something earn a marker; a conversational
    // exchange should not sprinkle dashed rows through the transcript.
    if (totals.adds === 0 && totals.dels === 0) return
    this.blocks = [...this.blocks, { kind: 'checkpoint', id: this.nextId(), turn, adds: totals.adds, dels: totals.dels }]
    this.invalidateSnapshot()
  }

  /**
   * Current fold. While a turn is active, trailing runs render as
   * streaming-tailed blocks in stream order; `finalize` commits them as
   * finished blocks. Preserves `===` identity for completed blocks and
   * settled earlier runs so `React.memo` bails out cleanly during streaming.
   */
  snapshot(turnActive: boolean): readonly Block[] {
    if (this.cachedSnapshot !== null && this.cachedSnapshotActive === turnActive) {
      return this.cachedSnapshot
    }
    if (!turnActive) {
      this.cachedSnapshot = this.blocks
      this.cachedSnapshotActive = false
      return this.blocks
    }
    // Only the TRAILING run can still receive deltas — an earlier run closed
    // the moment a different kind opened after it. Marking every run
    // streaming kept a blinking caret on text the model had already finished
    // (the "caret sticks around after the agent is done talking" bug).
    const tail = this.runs.length - 1
    const live = this.runs.map((run, index) => {
      if (index < tail) {
        let settled = this.settledRunBlocks.get(run)
        if (!settled) {
          settled = this.runBlock(run, index, false)
          this.settledRunBlocks.set(run, settled)
        }
        return settled
      }
      return this.runBlock(run, index, true)
    })
    // The card sits right behind the run that spawned it, not behind whatever
    // streamed last — otherwise it trails the final answer.
    const card = this.agentsBlock()
    if (card) {
      const at = this.agentsAnchor ? this.runs.indexOf(this.agentsAnchor) : -1
      live.splice(at >= 0 ? at + 1 : live.length, 0, card)
    }
    const result = live.length === 0 ? this.blocks : [...this.blocks, ...live]
    this.cachedSnapshot = result
    this.cachedSnapshotActive = true
    return result
  }

  /** Render one run as a display block. `streaming` tails the live one. */
  private runBlock(run: Run, index: number, streaming: boolean): Block {
    // Allocate identity at first activity and retain it through finalization.
    const id = this.runIds.get(run)!
    if (run.kind === 'notice' || run.kind === 'reflection') return { ...run, id }
    if (run.kind === 'tools') {
      const items = run.order.map(itemId => this.tools.get(itemId)).filter((t): t is ToolItem => t !== undefined)
      // `running` follows the items, not the caret: a still-working tool is
      // live even when a later text/thinking run took over the stream tail.
      return { kind: 'tools', id, items, running: items.some(t => t.state === 'working') }
    }
    if (run.kind === 'thinking') return { kind: 'thinking', id, text: run.text, streaming }
    return { kind: 'agent', id, text: run.text, streaming }
  }

  /** Commit every run into the fold, in stream order; call on turn_end. */
  finalize(): void {
    this.commitRuns(true)
  }

  private commitRuns(ended: boolean): void {
    let mutated = false
    // Close any tool still marked working: the turn ended, so nothing can
    // land a result anymore. Closing before rendering keeps the trail honest.
    if (ended) {
      for (const [id, tool] of this.tools) {
        if (tool.state === 'working') {
          this.tools.set(id, { ...tool, state: 'failed', error: 'Turn ended before a tool result was received' })
          mutated = true
        }
      }
    }
    if (this.tools.size > 0 && this.blocks.some(block => block.kind === 'tools' && block.items.some(item => this.tools.has(item.id)))) {
      this.blocks = this.blocks.map(block => {
        if (block.kind !== 'tools') return block
        let blockChanged = false
        const items = block.items.map(item => {
          const updated = this.tools.get(item.id)
          if (updated && updated !== item) { blockChanged = true; return updated }
          return item
        })
        if (!blockChanged) return block
        mutated = true
        return { ...block, items, running: items.some(item => item.state === 'working') }
      })
    }
    if (this.runs.length > 0) {
      const appended: Block[] = []
      for (const [index, run] of this.runs.entries()) {
        const block = !ended && this.settledRunBlocks.get(run) ? this.settledRunBlocks.get(run)! : this.runBlock(run, index, false)
        if (block.kind === 'tools') appended.push(block)
        else if (block.kind === 'thinking') appended.push(block.streaming ? { ...block, streaming: false } : block)
        else if (block.kind === 'agent') appended.push(block.streaming ? { ...block, streaming: false } : block)
        else if (block.kind === 'notice' || block.kind === 'reflection') appended.push(block)
        if (run === this.agentsAnchor) {
          const card = this.agentsBlock()
          if (card) appended.push(card)
        }
      }
      this.blocks = [...this.blocks, ...appended]
      this.runs = []
      mutated = true
    }
    if (ended && this.tools.size > 0) {
      this.tools.clear()
      mutated = true
    }
    // Commit the agents card behind the drained runs, keeping the reference:
    // terminal statuses still land on it while background children settle.
    // Idempotent — all() finalizes again and must not double-commit.
    const card = ended ? this.agentsBlock() : null
    if (card) {
      this.blocks = [...this.blocks, card]
      mutated = true
    }
    if (mutated) this.invalidateSnapshot()
  }

  /**
   * Close the current agents card (the store calls this at the next
   * turn_begin): the committed card stays and keeps receiving status
   * updates until then; a later spawn opens a fresh card.
   */
  closeAgentsCard(): void {
    this.agentsCard = null
    this.agentsAnchor = null
  }

  /** Replace the whole fold (session open/resume hydration). */
  reset(blocks: Block[] = []): void {
    this.blocks = blocks
    // Seeded blocks (legacy replay) carry ids from another builder; live runs
    // must mint past them or React sees two groups with one key.
    for (const block of blocks) if (block.id >= this.seq) this.seq = block.id + 1
    this.runs = []
    this.tools.clear()
    this.agentsCard = null
    this.agentsAnchor = null
    this.invalidateSnapshot()
  }

  /** Older pages must not finalize or replace the active streaming buffers. */
  prepend(blocks: Block[]): void {
    this.blocks = [...blocks, ...this.blocks]
    this.invalidateSnapshot()
  }

  /** Finalizes, then returns the fold — a turn end, never a read mid-turn. */
  all(): readonly Block[] {
    return this.finalize(), this.blocks
  }

  /**
   * Read-only probe over committed blocks and live runs. Asking "is there a
   * user message yet?" through `all()` finalized the running turn: its working
   * tools turned failed, their results opened duplicate rows and the reply
   * split into two bubbles.
   */
  some(predicate: (block: Block) => boolean): boolean {
    return this.snapshot(true).some(predicate)
  }
}

/** Optional session-record siblings the replay fold can lean on. */
export interface StoredHydration {
  /** Ordered tool executions — the session record's `tool_executions`. */
  readonly executions?: unknown
  /** Per-assistant-turn reasoning — the session record's `thinking_content`. */
  readonly thinking?: unknown
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

export function toolFailureText(payload: Readonly<Record<string, unknown>>, output = detailOf(payload.return_value ?? payload.result ?? payload.output)): string {
  return typeof payload.error === 'string' && payload.error ? payload.error
    : payload.permitted === false ? output || 'Tool permission denied'
    : output.startsWith('Tool execution failed:') ? output : ''
}

/**
 * Replay one stored execution as the same call+result event pair the live
 * stream emits, so a resumed transcript shows the identical verb + compact
 * arg + duration row instead of silently dropping the tool activity.
 */
function pushStoredExecution(builder: BlockBuilder, exec: unknown, index: number, content?: unknown): void {
  if (!isRecord(exec)) return // no fabricated rows for missing records
  const id = typeof exec.toolCallId === 'string' && exec.toolCallId ? exec.toolCallId : `stored-${index}`
  builder.push('tool_call', {
    id,
    name: exec.name,
    arguments: exec.inputs ?? exec.arguments,
    ...(exec.diff !== undefined ? { diff: exec.diff } : {}),
    ...(exec.locations !== undefined ? { locations: exec.locations } : {}),
  })
  builder.push('tool_result', {
    tool_call_id: id,
    name: exec.name,
    result: content ?? exec.result,
    ...(exec.diff !== undefined ? { diff: exec.diff } : {}),
    ...(exec.locations !== undefined ? { locations: exec.locations } : {}),
    ...(exec.permitted === false ? { error: typeof exec.result === 'string' ? exec.result : 'Tool permission denied' } : {}),
    ...(typeof exec.durationMs === 'number' ? { duration_ms: exec.durationMs } : {}),
    ...(typeof exec.error === 'string' && exec.error ? { error: exec.error } : {}),
  })
}

/**
 * Convert stored `session.transcript` messages into seed blocks that read
 * like the live fold: ordered think → tools → text runs. Tool activity is
 * reconstructed from the session's `tool_executions` (the stored twin of
 * the streamed calls) and per-turn reasoning from `thinking_content` —
 * without them the replay would misfile tool results as thinking text.
 */
export function blocksFromStoredMessages(messages: unknown, hydration: StoredHydration = {}): Block[] {
  if (!Array.isArray(messages)) return []
  const builder = new BlockBuilder()
  const executions = Array.isArray(hydration.executions) ? hydration.executions : []
  const thinking = Array.isArray(hydration.thinking) ? hydration.thinking : []
  const executionsById = new Map(executions.filter(isRecord).filter(exec => typeof exec.toolCallId === 'string').map(exec => [exec.toolCallId as string, exec]))
  const calls = new Map<string, Record<string, unknown>>()
  let executionIndex = 0
  let assistantIndex = 0
  for (const message of messages) {
    if (!message || typeof message !== 'object') continue
    const record = message as Record<string, unknown>
    const role = typeof record.role === 'string' ? record.role : ''

    if (role === 'tool') {
      // Retained executions can be truncated independently of messages. Match
      // identities first; positional association is only for legacy rows with no ID.
      const id = typeof record.tool_call_id === 'string' ? record.tool_call_id : ''
      const execution = id ? executionsById.get(id) : executions[executionIndex]
      if (id && calls.has(id)) {
        const call = calls.get(id)!
        const stored = isRecord(execution) ? execution : null
        builder.push('tool_result', { tool_call_id: id, name: call.name, result: record.content ?? stored?.result,
          duration_ms: stored?.durationMs,
          ...(stored?.permitted === false ? { error: typeof stored.result === 'string' ? stored.result : 'Tool permission denied' } : {}),
          ...(typeof stored?.error === 'string' && stored.error ? { error: stored.error } : {}),
        })
      } else pushStoredExecution(builder, execution, executionIndex, record.content)
      executionIndex += 1
      continue
    }

    // Reasoning precedes its answer, as it streamed.
    const turnThinking = role === 'assistant' ? thinking[assistantIndex] : undefined
    if (role === 'assistant') assistantIndex += 1
    if (typeof turnThinking === 'string' && turnThinking.trim()) {
      builder.push('think_part', { think: turnThinking })
    }

    const emitCalls = (): void => {
      if (role !== 'assistant' || !Array.isArray(record.tool_calls)) return
      for (const rawCall of record.tool_calls) {
        if (!isRecord(rawCall) || typeof rawCall.id !== 'string' || !isRecord(rawCall.function)) continue
        const call = rawCall.function
        calls.set(rawCall.id, call)
        builder.push('tool_call', { id: rawCall.id, name: call.name, arguments: call.arguments })
      }
    }
    const content = transcriptContent(record)
    if (typeof content === 'string') {
      if (!content.trim()) { emitCalls(); continue }
      // Internal coordinator/agent wakes become quiet notices, never user chat bubbles.
      if (role === 'user' && (record.origin === 'agent' || isInternalSwarmWakeText(content))) {
        builder.pushNotice(formatInternalWakeNotice(content))
        emitCalls()
        continue
      }
      // The harness wrote it (goal round, monitor, reminder): not shown.
      if (role === 'user' && (harnessOriginOf(record.origin) || isLegacyHarnessPrompt(content))) { emitCalls(); continue }
      if (role === 'user') builder.pushUser(content, record.xerxes_compaction_summary === true)
      else if (role === 'assistant') builder.push('text_part', { text: content })
      else if (role !== 'system') builder.push('think_part', { think: content })
      emitCalls()
      continue
    }
    if (!Array.isArray(content)) { emitCalls(); continue }
    // Parts fold sequentially — a [think, text] assistant message replays as
    // a thinking block then an agent block, matching the live run grammar.
    const userParts: string[] = []
    const userImages: string[] = []
    for (const part of content) {
      if (!part || typeof part !== 'object') continue
      const p = part as Record<string, unknown>
      // The image a person attached: shown in their bubble, as sent. Only
      // inline data and web URLs render; anything else is not displayable.
      if (role === 'user' && p.type === 'image_url' && isRecord(p.image_url) && typeof p.image_url.url === 'string'
        && /^(data:image\/|https?:\/\/)/.test(p.image_url.url)) { userImages.push(p.image_url.url); continue }
      if (p.type === 'tool_use') {
        builder.push('tool_call', {
          id: typeof p.id === 'string' && p.id ? p.id : `stored-use-${executionIndex++}`,
          name: p.name,
          arguments: p.input ?? p.arguments,
        })
        continue
      }
      if (p.type === 'tool_result') {
        builder.push('tool_result', {
          tool_call_id: typeof p.tool_use_id === 'string' ? p.tool_use_id : typeof p.id === 'string' ? p.id : '',
          result: p.content ?? p.result,
          ...(p.is_error === true ? { error: detailOf(p.content) || 'Tool execution failed' } : {}),
          ...(typeof p.error === 'string' && p.error ? { error: p.error } : {}),
        })
        continue
      }
      const think = typeof p.think === 'string' ? p.think : typeof p.thinking === 'string' ? p.thinking : ''
      const text = typeof p.text === 'string' ? p.text : ''
      if (think) builder.push('think_part', { think })
      if (text) {
        if (role === 'user') userParts.push(text)
        else builder.push('text_part', { text })
      }
    }
    if (userParts.length || userImages.length) {
      const joined = userParts.join('\n')
      if (record.origin === 'agent' || isInternalSwarmWakeText(joined)) {
        if (joined.trim()) builder.pushNotice(formatInternalWakeNotice(joined))
      } else if (!harnessOriginOf(record.origin)) {
        builder.pushUser(joined, record.xerxes_compaction_summary === true, userImages)
      }
    }
    emitCalls()
  }
  return [...builder.all()]
}
