// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

const MAX_INLINE_AGENTS = 8
const MAX_FALLBACK_CHARS = 96
const MAX_PATH_PREVIEW_CHARS = 72

export interface ToolStartDisplay {
  context: string
}

export function summarizeToolStartDisplay(name: string, context: string, verboseArgs?: string): ToolStartDisplay {
  const toolName = normalToolName(name)

  if (toolName === 'spawnagents') {
    return summarizeSpawnAgents(context, verboseArgs)
  }

  const fileOperation = summarizeFileOperation(toolName, context, verboseArgs)
  if (fileOperation) {
    return fileOperation
  }

  const raw = verboseArgs || context
  const parsed = parseObject(raw)

  if (parsed) {
    return { context: summarizeStructuredArgs(toolName, parsed, context) }
  }

  // OpenTUI keeps tool calls as Grok-style one-line rows. A non-JSON
  // context can already be a useful daemon-provided label; raw argument
  // blobs are never carried into the persistent transcript.
  return { context: compact(context || raw) }
}

function summarizeStructuredArgs(toolName: string, parsed: Record<string, unknown>, fallbackContext: string): string {
  const command = commandPreview(parsed)
  if (command) {
    return command
  }

  const path = firstString(parsed, ['file_path', 'filePath', 'path', 'target_path', 'targetPath', 'directory', 'cwd'])
  const query = firstString(parsed, ['query', 'pattern', 'search', 'needle', 'url'])

  if (path && query) {
    return `${compact(query)} in ${compactPath(path)}`
  }
  if (path) {
    return compactPath(path)
  }
  if (query) {
    return compact(query)
  }

  // Many tools are addressed by an identifier rather than a path or a query.
  // Without this they fell through to the raw-blob fallback below and the
  // transcript rendered rows like `Task Output Tool {"task_id":"r9-timeout"}`
  // — the JSON syntax being pure noise around the one word that mattered.
  const identifier = firstString(parsed, IDENTIFIER_KEYS)
  if (identifier) {
    return compact(identifier)
  }

  // Patch/content/prompt payloads can be enormous or secret-bearing. Keep a
  // semantic hint for those calls without serializing their values.
  if (firstString(parsed, ['patch', 'diff'])) {
    return 'apply patch'
  }
  if (firstString(parsed, ['content', 'body', 'text', 'prompt'])) {
    return toolName.includes('write') ? 'write content' : ''
  }

  // Last resort for a shape nothing above recognized: render the scalar
  // entries as `key=value`, which stays readable at a glance. Echoing the
  // serialized object here is what produced the JSON-in-the-transcript rows,
  // so `fallbackContext` is only used when it is not itself a blob.
  const entries = describeEntries(parsed)
  if (entries) {
    return entries
  }

  return parseObject(fallbackContext) ? '' : compact(fallbackContext)
}

/** Keys that name *what* a call addresses when there is no path or query. */
const IDENTIFIER_KEYS = [
  'task_id',
  'taskId',
  'agent_id',
  'agentId',
  'session_id',
  'sessionId',
  'terminal_id',
  'skill',
  'tool',
  'name',
  'id',
  'key'
]

const MAX_DESCRIBED_ENTRIES = 3

/** `key=value` for the scalar fields of an otherwise unrecognized payload. */
function describeEntries(parsed: Record<string, unknown>): string {
  const parts: string[] = []

  for (const [key, value] of Object.entries(parsed)) {
    if (value === null || typeof value === 'object') {
      continue
    }

    parts.push(`${key}=${String(value)}`)

    if (parts.length === MAX_DESCRIBED_ENTRIES) {
      break
    }
  }

  return compact(parts.join(' · '))
}

function commandPreview(parsed: Record<string, unknown>): string {
  const commandValue = parsed.command ?? parsed.cmd
  const argsValue = parsed.args ?? parsed.arguments
  const command = Array.isArray(commandValue)
    ? commandValue.map(String).filter(Boolean)
    : typeof commandValue === 'string'
      ? [commandValue]
      : []
  const args = Array.isArray(argsValue)
    ? argsValue.map(String).filter(Boolean)
    : typeof argsValue === 'string' && argsValue.trim()
      ? [argsValue.trim()]
      : []

  return compact([...command, ...args].join(' '))
}

/**
 * Parse the roster back out of a SpawnAgents summary line ("8 agents: a, b, +2
 * more"). The transcript row is a rendered string by the time ToolStep sees
 * it; this recovers the displayable names and the overflow count for the live
 * status roster beneath the row.
 */
export function spawnRosterFromSummary(args: string): { extra: number; names: string[] } | null {
  const match = args.match(/^(\d+) agents?: (.+)$/u)
  if (!match) return null
  const extraMatch = match[2]!.match(/, \+(\d+) more$/u)
  const extra = extraMatch ? Number(extraMatch[1]) : 0
  const body = extraMatch ? match[2]!.slice(0, -extraMatch[0].length) : match[2]!
  const names = body.split(',').map(name => name.trim()).filter(Boolean)
  return names.length || extra ? { extra, names } : null
}

/**
 * Recover the roster from a whole Spawn Agents transcript line, quoted-context
 * and all: `Spawn Agents("3 agents: a, b, +1 more") ✓`. The generic trail
 * parser splits legacy ': ' inside the quoted context, so spawn rows need
 * their own extraction. Matches in-flight lines (no ✓/✗) too.
 */
export function spawnRosterFromLine(line: string): { extra: number; names: string[] } | null {
  const body = line.replace(/ [✓✗]$/u, '')
  const match = body.match(/^Spawn Agents\("(\d+ agents?: .*)"\)/u)
  return match ? spawnRosterFromSummary(match[1]!) : null
}

function summarizeSpawnAgents(context: string, verboseArgs?: string): ToolStartDisplay {
  const raw = verboseArgs || context
  const parsed = parseObject(raw)
  const agents = Array.isArray(parsed?.agents) ? parsed.agents : []
  const names = agents.map(agentName).filter(Boolean)
  const count = agents.length || names.length
  if (count > 0) {
    const shown = names.slice(0, MAX_INLINE_AGENTS)
    const suffix = names.length > MAX_INLINE_AGENTS ? `, +${names.length - MAX_INLINE_AGENTS} more` : ''
    const roster = shown.length ? `: ${shown.join(', ')}${suffix}` : ''
    const wait = typeof parsed?.wait === 'boolean' ? ` · wait=${parsed.wait}` : ''

    return { context: `${count} agent${count === 1 ? '' : 's'}${roster}${wait}` }
  }

  const fallbackNames = Array.from(raw.matchAll(/["']name["']\s*:\s*["']([^"']+)["']/g), m => m[1]).filter(Boolean)
  if (fallbackNames.length) {
    const shown = fallbackNames.slice(0, MAX_INLINE_AGENTS)
    const suffix = fallbackNames.length > MAX_INLINE_AGENTS ? `, +${fallbackNames.length - MAX_INLINE_AGENTS} more` : ''

    return { context: `${fallbackNames.length} agents: ${shown.join(', ')}${suffix}` }
  }

  return { context: compact(raw) }
}

function summarizeFileOperation(toolName: string, context: string, verboseArgs?: string): ToolStartDisplay | null {
  const raw = verboseArgs || context
  const parsed = parseObject(raw)
  if (!parsed) {
    return null
  }

  if (isWriteTool(toolName, parsed)) {
    const filePath = firstString(parsed, ['file_path', 'path', 'target_path', 'target'])
    const content = firstString(parsed, ['content', 'body', 'text', 'new_string'])
    const overwrite = typeof parsed.overwrite === 'boolean' ? ` · overwrite=${parsed.overwrite}` : ''
    const size = content ? ` · ${fmtChars(content.length)}` : ''
    const prefix = filePath ? compactPath(filePath) : compact(context || 'file')

    return { context: `write ${prefix}${size}${overwrite}` }
  }

  if (isMoveTool(toolName, parsed)) {
    const source = firstString(parsed, ['source', 'source_path', 'src', 'old_path', 'from', 'path'])
    const destination = firstString(parsed, ['destination', 'destination_path', 'dest', 'new_path', 'to'])
    if (source || destination) {
      return { context: `${compactPath(source || 'source')} -> ${compactPath(destination || 'destination')}` }
    }
  }

  return null
}

function isWriteTool(toolName: string, parsed: Record<string, unknown>): boolean {
  if (toolName === 'writefile' || toolName === 'writefiletool' || toolName === 'writefilefn') {
    return true
  }

  return (
    (toolName === 'filesystemtools' || toolName === 'filesystemtool') &&
    firstString(parsed, ['operation', 'op']).toLowerCase() === 'write'
  )
}

function isMoveTool(toolName: string, parsed: Record<string, unknown>): boolean {
  if (toolName === 'movefile') {
    return true
  }

  return (
    (toolName === 'movefiletool' ||
      toolName === 'filesystemtools' ||
      toolName === 'filesystemtool' ||
      toolName === 'movefilefn' ||
      toolName === 'movefile') &&
    (toolName.includes('move') || firstString(parsed, ['operation', 'op']).toLowerCase() === 'move')
  )
}

function firstString(record: Record<string, unknown>, keys: string[]): string {
  for (const key of keys) {
    const value = record[key]
    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }

  return ''
}

function compactPath(path: string): string {
  if (path.length <= MAX_PATH_PREVIEW_CHARS) {
    return path
  }

  const parts = path.split('/')
  const file = parts.at(-1) || path
  if (file.length + 2 >= MAX_PATH_PREVIEW_CHARS) {
    return `…${file.slice(-(MAX_PATH_PREVIEW_CHARS - 1))}`
  }

  const parent = parts.at(-2)
  const suffix = parent ? `${parent}/${file}` : file

  return suffix.length + 2 > MAX_PATH_PREVIEW_CHARS ? `…${suffix.slice(-(MAX_PATH_PREVIEW_CHARS - 1))}` : `…/${suffix}`
}

function fmtChars(count: number): string {
  if (count < 1000) {
    return `${count} chars`
  }

  return `${(count / 1000).toFixed(count < 10_000 ? 1 : 0)}k chars`
}

const normalToolName = (name: string) => name.replace(/[^a-z0-9]+/gi, '').toLowerCase()

function parseObject(raw: string): Record<string, unknown> | null {
  try {
    const value = JSON.parse(raw)

    return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
  } catch {
    return null
  }
}

function agentName(value: unknown): string {
  if (!value || typeof value !== 'object') {
    return ''
  }
  const record = value as Record<string, unknown>
  const name = record.name ?? record.agent_name ?? record.id

  return typeof name === 'string' ? name.trim() : ''
}

function compact(raw: string): string {
  const oneLine = raw.replace(/\s+/g, ' ').trim()

  return oneLine.length > MAX_FALLBACK_CHARS ? `${oneLine.slice(0, MAX_FALLBACK_CHARS - 1)}…` : oneLine
}
