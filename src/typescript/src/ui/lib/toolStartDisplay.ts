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

  // Patch/content/prompt payloads can be enormous or secret-bearing. Keep a
  // semantic hint for those calls without serializing their values.
  if (firstString(parsed, ['patch', 'diff'])) {
    return 'apply patch'
  }
  if (firstString(parsed, ['content', 'body', 'text', 'prompt'])) {
    return toolName.includes('write') ? 'write content' : ''
  }

  return compact(fallbackContext)
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
