// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readdir, readFile, realpath, stat } from 'node:fs/promises'
import { basename, dirname, join, resolve } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'

export const EXPORT_SCHEMA = 'xerxes.session.export.v1'
export const DEFAULT_EXPORT_FORMAT = 'json'
export const LOVELY_PIRATE_FORMAT = 'lovely-pirate'
export const EXPORT_FORMATS = ['json', 'jsonl', 'md', LOVELY_PIRATE_FORMAT] as const

export type SessionExportFormat = (typeof EXPORT_FORMATS)[number]
export type SessionRecordJson = Record<string, unknown>
export type SessionExportPathResolver = (path: string) => string | Promise<string>

/** Raised when a requested persisted session cannot be found or selected unambiguously. */
export class SessionExportError extends Error {
  constructor(message: string) {
    super(message)
    this.name = new.target.name
  }
}

/** One on-disk daemon session plus its filesystem provenance. */
export interface SavedSession {
  readonly mtimeMs: number
  readonly path: string
  readonly projectDirectory?: string
  readonly record: SessionRecordJson
}

export interface SavedSessionSummary {
  readonly agent_id: string
  readonly id: string
  readonly key: string
  readonly messages: number
  readonly path: string
  readonly project_dir: string
  readonly session_id: string
  readonly title: string
  readonly turn_count: number
  readonly updated_at: string
}

export interface SessionExport {
  readonly archive_included: boolean
  readonly archive_messages: readonly unknown[]
  readonly archive_path: string
  readonly exported_at: string
  readonly live_messages: readonly unknown[]
  readonly messages: readonly unknown[]
  readonly metadata: Readonly<Record<string, unknown>>
  readonly record_path: string
  readonly runtime: {
    readonly agent_id: string
    readonly interaction_mode: string
    readonly model: string
    readonly model_provider: string
    readonly plan_mode: boolean
    readonly workspace: string
  }
  readonly schema: string
  readonly session: SavedSessionSummary
  readonly thinking_content: unknown
  readonly tool_executions: unknown
  readonly usage: {
    readonly total_input_tokens: number
    readonly total_output_tokens: number
  }
}

export interface ListSavedSessionsOptions {
  readonly pathResolver?: SessionExportPathResolver
  readonly projectDir?: string
  readonly storeDir?: string
}

export interface BuildSessionExportOptions {
  readonly includeArchive?: boolean
  readonly now?: () => Date
}

const MARKDOWN_CODE_FENCE = String.fromCharCode(96).repeat(3)

/** Return the default legacy-compatible daemon transcript directory. */
export function defaultSessionStoreDir(
  options: { readonly homeDirectory?: string; readonly environment?: NodeJS.ProcessEnv } = {},
): string {
  return join(options.homeDirectory ?? xerxesHome(options.environment), 'sessions')
}

/** Discover non-empty saved session JSON records, ordered newest first. */
export async function listSavedSessions(options: ListSavedSessionsOptions = {}): Promise<SavedSession[]> {
  const root = await resolvedPath(options.storeDir ?? defaultSessionStoreDir(), options.pathResolver)
  const projectDirectory = options.projectDir === undefined
    ? undefined
    : await resolvedPath(options.projectDir, options.pathResolver)
  const entries = await directoryEntries(root)

  const sessions: SavedSession[] = []
  for (const entry of entries) {
    if (!entry.isFile() || entry.name.startsWith('.') || !entry.name.endsWith('.json')) {
      continue
    }
    const saved = await readSavedSession(join(root, entry.name), options.pathResolver)
    if (!saved || (projectDirectory !== undefined && saved.projectDirectory !== projectDirectory)) {
      continue
    }
    sessions.push(saved)
  }
  return sessions.sort(compareSavedSessions)
}

/** Produce compact metadata for lists and selection diagnostics. */
export function savedSessionSummary(saved: SavedSession): SavedSessionSummary {
  const record = saved.record
  const sessionId = firstText(record.session_id) || basename(saved.path, '.json')
  return {
    id: sessionId,
    session_id: sessionId,
    key: firstText(record.key),
    title: recordTitle(record),
    agent_id: firstText(record.agent_id),
    project_dir: saved.projectDirectory ?? recordProjectDirectoryText(record),
    updated_at: firstText(record.updated_at),
    turn_count: recordTurnCount(record),
    messages: recordMessageCount(record),
    path: saved.path,
  }
}

/** Select an exact or unique prefix match, or the latest session when query is empty. */
export async function selectSavedSession(
  query = '',
  options: ListSavedSessionsOptions = {},
): Promise<SavedSession> {
  const sessions = await listSavedSessions(options)
  if (!sessions.length) {
    const scope = options.projectDir === undefined
      ? ''
      : ' for project ' + await resolvedPath(options.projectDir, options.pathResolver)
    throw new SessionExportError('No saved Xerxes sessions found' + scope + '.')
  }
  const needle = query.trim()
  if (!needle) {
    return sessions[0] as SavedSession
  }

  const lower = needle.toLowerCase()
  const exact: SavedSession[] = []
  const prefix: SavedSession[] = []
  for (const saved of sessions) {
    const summary = savedSessionSummary(saved)
    const values = [summary.id, summary.session_id, summary.key, summary.title]
    if (values.some(value => needle === value || lower === value.toLowerCase())) {
      exact.push(saved)
    } else if (values.some(value => value.startsWith(needle) || value.toLowerCase().startsWith(lower))) {
      prefix.push(saved)
    }
  }
  const matches = exact.length ? exact : prefix
  if (!matches.length) {
    const scope = options.projectDir === undefined
      ? ''
      : ' in project ' + await resolvedPath(options.projectDir, options.pathResolver)
    throw new SessionExportError('No saved Xerxes session matched ' + JSON.stringify(needle) + scope + '.')
  }
  if (matches.length > 1) {
    const ids = matches.slice(0, 8).map(item => savedSessionSummary(item).id).join(', ')
    throw new SessionExportError('Session query ' + JSON.stringify(needle) + ' matched multiple sessions: ' + ids)
  }
  return matches[0] as SavedSession
}

/** Return the adjacent archive path used for a legacy session record. */
export function archivePathFor(recordPath: string): string {
  const name = basename(recordPath, '.json')
  return join(dirname(recordPath), name + '.archive.jsonl')
}

/** Read valid JSON-object archive lines, ignoring malformed or blank records. */
export async function readArchiveMessages(recordPath: string): Promise<SessionRecordJson[]> {
  const archivePath = archivePathFor(recordPath)
  let content: string
  try {
    content = await readFile(archivePath, 'utf8')
  } catch {
    return []
  }
  const messages: SessionRecordJson[] = []
  for (const line of content.split(/\r?\n/)) {
    if (!line.trim()) {
      continue
    }
    try {
      const value: unknown = JSON.parse(line)
      if (isRecord(value)) {
        messages.push(value)
      }
    } catch {
      // A partial archive line does not invalidate the rest of the trace.
    }
  }
  return messages
}

/** Build a complete serialisable trace that joins archived and live messages. */
export async function buildSessionExport(
  saved: SavedSession,
  options: BuildSessionExportOptions = {},
): Promise<SessionExport> {
  const includeArchive = options.includeArchive ?? true
  const archivePath = archivePathFor(saved.path)
  const archiveMessages = includeArchive ? await readArchiveMessages(saved.path) : []
  const liveMessages = arrayValue(saved.record.messages)
  const metadata = recordValue(saved.record.metadata)
  const now = options.now ?? (() => new Date())
  const exportedAt = isoTimestamp(now)
  const archiveExists = await pathExists(archivePath)
  return {
    schema: EXPORT_SCHEMA,
    exported_at: exportedAt,
    session: savedSessionSummary(saved),
    record_path: saved.path,
    archive_path: archiveExists ? archivePath : '',
    archive_included: includeArchive,
    messages: [...archiveMessages, ...liveMessages],
    live_messages: liveMessages,
    archive_messages: archiveMessages,
    metadata,
    thinking_content: saved.record.thinking_content ?? [],
    tool_executions: saved.record.tool_executions ?? [],
    usage: {
      total_input_tokens: integerValue(saved.record.total_input_tokens),
      total_output_tokens: integerValue(saved.record.total_output_tokens),
    },
    runtime: {
      interaction_mode: firstText(saved.record.interaction_mode, saved.record.mode),
      plan_mode: saved.record.plan_mode === true,
      agent_id: firstText(saved.record.agent_id),
      workspace: firstText(saved.record.workspace),
      model: firstText(saved.record.model, metadata.model),
      model_provider: firstText(saved.record.model_provider, saved.record.provider, metadata.provider),
    },
  }
}

/** Render a trace as JSON, JSONL, Markdown, or Lovely Pirate external-event JSONL. */
export function formatSessionExport(exportRecord: SessionExport, outputFormat: string): string {
  const format = outputFormat.trim().toLowerCase()
  switch (format) {
    case 'json':
      return JSON.stringify(exportRecord, null, 2) + '\n'
    case 'jsonl':
      return formatJsonl(exportRecord)
    case LOVELY_PIRATE_FORMAT:
    case 'lp-jsonl':
      return formatLovelyPirateJsonl(exportRecord)
    case 'md':
      return formatMarkdown(exportRecord)
    default:
      throw new SessionExportError('unsupported export format: ' + outputFormat)
  }
}

function compareSavedSessions(left: SavedSession, right: SavedSession): number {
  const leftUpdatedAt = firstText(left.record.updated_at)
  const rightUpdatedAt = firstText(right.record.updated_at)
  if (leftUpdatedAt !== rightUpdatedAt) {
    return leftUpdatedAt > rightUpdatedAt ? -1 : 1
  }
  if (left.mtimeMs !== right.mtimeMs) {
    return right.mtimeMs - left.mtimeMs
  }
  return left.path.localeCompare(right.path)
}

async function readSavedSession(
  path: string,
  pathResolver: SessionExportPathResolver | undefined,
): Promise<SavedSession | undefined> {
  try {
    const [text, details] = await Promise.all([readFile(path, 'utf8'), stat(path)])
    const parsed: unknown = JSON.parse(text)
    if (!isRecord(parsed) || !recordHasHistory(parsed)) {
      return undefined
    }
    const projectRaw = recordProjectDirectoryText(parsed)
    const projectDirectory = projectRaw ? await resolvedPath(projectRaw, pathResolver) : undefined
    return {
      path,
      record: parsed,
      mtimeMs: details.mtimeMs,
      ...(projectDirectory === undefined ? {} : { projectDirectory }),
    }
  } catch {
    return undefined
  }
}

function recordHasHistory(record: SessionRecordJson): boolean {
  return recordMessageCount(record) > 0 || recordTurnCount(record) > 0
}

function recordMessageCount(record: SessionRecordJson): number {
  return Array.isArray(record.messages) ? record.messages.length : 0
}

function recordProjectDirectoryText(record: SessionRecordJson): string {
  return firstText(record.cwd, record.project_dir)
}

function recordTitle(record: SessionRecordJson): string {
  const title = firstText(recordValue(record.metadata).title)
  return title || titleFromMessages(record.messages)
}

function recordTurnCount(record: SessionRecordJson): number {
  return integerValue(record.turn_count)
}

function titleFromMessages(messages: unknown): string {
  if (!Array.isArray(messages)) {
    return ''
  }
  for (const message of messages) {
    if (!isRecord(message) || firstText(message.role) !== 'user') {
      continue
    }
    const text = messageText(message.content).replace(/\s+/g, ' ').trim()
    if (text) {
      return text.length > 80 ? text.slice(0, 77) + '...' : text
    }
  }
  return ''
}

function formatJsonl(exportRecord: SessionExport): string {
  const lines = [JSON.stringify({
    type: 'session',
    schema: exportRecord.schema,
    exported_at: exportRecord.exported_at,
    session: exportRecord.session,
    usage: exportRecord.usage,
    runtime: exportRecord.runtime,
  })]
  const archiveCount = exportRecord.archive_messages.length
  for (let index = 0; index < exportRecord.messages.length; index += 1) {
    lines.push(JSON.stringify({
      type: 'message',
      index,
      source: index < archiveCount ? 'archive' : 'live',
      message: exportRecord.messages[index],
    }))
  }
  for (const [index, toolExecution] of arrayValue(exportRecord.tool_executions).entries()) {
    lines.push(JSON.stringify({ type: 'tool_execution', index, tool_execution: toolExecution }))
  }
  return lines.join('\n') + '\n'
}

function formatLovelyPirateJsonl(exportRecord: SessionExport): string {
  const lines = [JSON.stringify(lovelyPirateMetaEvent(exportRecord))]
  const archiveCount = exportRecord.archive_messages.length
  for (let index = 0; index < exportRecord.messages.length; index += 1) {
    const event = lovelyPirateMessageEvent(
      exportRecord.messages[index],
      index,
      index < archiveCount ? 'archive' : 'live',
    )
    if (event) {
      lines.push(JSON.stringify(event))
    }
  }
  return lines.join('\n') + '\n'
}

function lovelyPirateMetaEvent(exportRecord: SessionExport): Record<string, unknown> {
  const metadata = exportRecord.metadata
  const inputTokens = integerValue(exportRecord.usage.total_input_tokens)
  const outputTokens = integerValue(exportRecord.usage.total_output_tokens)
  const payload: Record<string, unknown> = {
    id: firstText(exportRecord.session.id, exportRecord.session.session_id),
    session_id: firstText(exportRecord.session.session_id, exportRecord.session.id),
    source: 'xerxes',
    schema: exportRecord.schema || EXPORT_SCHEMA,
    cwd: firstText(exportRecord.session.project_dir),
    title: firstText(exportRecord.session.title),
    model_provider: firstText(exportRecord.runtime.model_provider, metadata.model_provider, metadata.provider),
    model: firstText(exportRecord.runtime.model, metadata.model),
    cli_version: firstText(metadata.cli_version, metadata.version),
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
    message_count: exportRecord.messages.length,
    record_path: exportRecord.record_path,
    archive_path: exportRecord.archive_path,
    updated_at: exportRecord.session.updated_at,
    exported_at: exportRecord.exported_at,
  }
  const tools = metadata.tools ?? metadata.available_tools
  if (hasContent(tools)) {
    payload.tools = tools
  }
  return {
    timestamp: firstText(metadata.started_at, metadata.created_at, exportRecord.exported_at),
    type: 'external_session_meta',
    payload,
  }
}

function lovelyPirateMessageEvent(
  message: unknown,
  index: number,
  source: 'archive' | 'live',
): Record<string, unknown> | undefined {
  if (!isRecord(message)) {
    return undefined
  }
  const role = firstText(message.role)
  if (!role) {
    return undefined
  }
  const event: Record<string, unknown> = {
    type: 'external_message',
    role,
    content: messageText(message.content),
    index,
    source,
  }
  const timestamp = firstText(message.timestamp, message.created_at, message.updated_at)
  if (timestamp) {
    event.timestamp = timestamp
  }
  const reasoning = firstText(message.reasoning_content, message.reasoning)
  if (reasoning) {
    event.reasoning_content = reasoning
  }
  if (hasContent(message.tool_calls)) {
    event.tool_calls = message.tool_calls
  }
  const toolCallId = firstText(message.tool_call_id)
  if (toolCallId) {
    event.tool_call_id = toolCallId
  }
  const name = firstText(message.name, message.tool_name)
  if (name) {
    event.name = name
  }
  return event
}

function formatMarkdown(exportRecord: SessionExport): string {
  const lines = [
    '# Xerxes Session Export: ' + exportRecord.session.id,
    '',
    '- Project: ' + markdownCode(exportRecord.session.project_dir),
    '- Title: ' + exportRecord.session.title,
    '- Updated: ' + exportRecord.session.updated_at,
    '- Exported: ' + exportRecord.exported_at,
    '- Messages: ' + exportRecord.messages.length,
    '- Record: ' + markdownCode(exportRecord.record_path),
  ]
  if (exportRecord.archive_path) {
    lines.push('- Archive: ' + markdownCode(exportRecord.archive_path))
  }
  lines.push('', '## Messages', '')
  const archiveCount = exportRecord.archive_messages.length
  for (let index = 0; index < exportRecord.messages.length; index += 1) {
    const message = exportRecord.messages[index]
    if (!isRecord(message)) {
      continue
    }
    const role = firstText(message.role) || 'message'
    const source = index < archiveCount ? 'archive' : 'live'
    lines.push('### ' + (index + 1) + '. ' + role + ' (' + source + ')', '', messageText(message.content).trimEnd(), '')
    if (hasContent(message.tool_calls)) {
      lines.push(MARKDOWN_CODE_FENCE + 'json', JSON.stringify(message.tool_calls, null, 2), MARKDOWN_CODE_FENCE, '')
    }
  }
  return lines.join('\n').trimEnd() + '\n'
}

function markdownCode(value: string): string {
  return String.fromCharCode(96) + value + String.fromCharCode(96)
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? [...value] : []
}

function firstText(...values: unknown[]): string {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
    if (value !== null && value !== undefined && !Array.isArray(value) && !isRecord(value)) {
      const text = String(value).trim()
      if (text) {
        return text
      }
    }
  }
  return ''
}

function hasContent(value: unknown): boolean {
  if (Array.isArray(value)) {
    return value.length > 0
  }
  if (isRecord(value)) {
    return Object.keys(value).length > 0
  }
  return Boolean(value)
}

function integerValue(value: unknown): number {
  const number = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(number) ? Math.trunc(number) : 0
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isoTimestamp(now: () => Date): string {
  const value = now()
  if (!(value instanceof Date) || Number.isNaN(value.valueOf())) {
    throw new RangeError('now must return a valid Date')
  }
  return value.toISOString()
}

function messageText(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }
  if (isRecord(content)) {
    return scalarText(content.text) || scalarText(content.content)
  }
  if (Array.isArray(content)) {
    const parts: string[] = []
    for (const item of content) {
      if (typeof item === 'string') {
        parts.push(item)
      } else if (isRecord(item)) {
        const text = scalarText(item.text) || scalarText(item.content)
        if (text) {
          parts.push(text)
        }
      }
    }
    return parts.join('\n')
  }
  return scalarText(content)
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path)
    return true
  } catch {
    return false
  }
}

async function directoryEntries(path: string) {
  try {
    return await readdir(path, { encoding: 'utf8', withFileTypes: true })
  } catch {
    return []
  }
}

function recordValue(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {}
}

async function resolvedPath(path: string, pathResolver: SessionExportPathResolver | undefined): Promise<string> {
  if (pathResolver) {
    return await pathResolver(path)
  }
  try {
    return await realpath(path)
  } catch {
    return resolve(path)
  }
}

function scalarText(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  if (value === null || value === undefined || isRecord(value) || Array.isArray(value)) {
    return ''
  }
  return String(value)
}
