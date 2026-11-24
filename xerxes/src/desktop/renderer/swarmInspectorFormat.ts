// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Presentation formatters for the Immortals worker inspector transcript.
 *
 * Cleans up Antigravity/daemon transcript artifacts (unclosed `<SYSTEM_MESSAGE>`
 * tags, duplicated `Created At:` / `Completed At:` tool headers, raw escaped
 * `read_board` JSON walls) on the client so both current and older remote
 * daemons render cleanly.
 */

export interface CleanedInspectorPrompt {
  readonly text: string
  readonly isSystemWake: boolean
}

export interface FormattedInspectorToolResult {
  readonly summary: string
  readonly body: string
  readonly collapsible: boolean
  readonly isJson: boolean
}

const LEADING_AGY_TIMESTAMPS_RE = /^(?:\s*(?:Created|Completed)\s+At:\s*[^\r\n]*(?:\r?\n|$))+/i

/** Strip repeated `Created At:` / `Completed At:` boilerplate lines prepended by Antigravity tool steps. */
export function stripAgyToolTimestampBoilerplate(raw: string): string {
  return raw.replace(LEADING_AGY_TIMESTAMPS_RE, '').trim()
}

/**
 * Strip `<SYSTEM_MESSAGE>` / `<USER_REQUEST>` wrappers (including truncated or
 * unclosed tags) and classify board/swarm wake pings so the inspector renders
 * them as quiet system notices instead of raw XML blocks.
 */
export function stripInspectorSystemEnvelope(raw: string): CleanedInspectorPrompt {
  const trimmed = raw.trim()
  if (!trimmed) return { text: '', isSystemWake: false }

  let text = trimmed
  let hadSystemTag = false

  const closedSys = /^<SYSTEM_MESSAGE>\s*([\s\S]*?)\s*<\/SYSTEM_MESSAGE>$/i.exec(text)
  if (closedSys) {
    hadSystemTag = true
    text = (closedSys[1] ?? '').trim()
  } else if (/^<SYSTEM_MESSAGE>/i.test(text)) {
    hadSystemTag = true
    text = text.replace(/^<SYSTEM_MESSAGE>\s*/i, '').replace(/\s*<\/SYSTEM_MESSAGE>\s*$/i, '').trim()
  }

  if (hadSystemTag) {
    const contentIdx = text.indexOf('content=')
    if (contentIdx >= 0 && /^\[Message\]/i.test(text)) {
      text = text.slice(contentIdx + 'content='.length).trim()
    }
  }

  const userReq = /^<USER_REQUEST>\s*([\s\S]*?)\s*<\/USER_REQUEST>$/i.exec(text)
  if (userReq) {
    text = (userReq[1] ?? '').trim()
  } else if (/^<USER_REQUEST>/i.test(text)) {
    text = text.replace(/^<USER_REQUEST>\s*/i, '').replace(/\s*<\/USER_REQUEST>\s*$/i, '').trim()
  }

  const isSystemWake =
    hadSystemTag ||
    /^\[(?:board|swarm)\]\s/i.test(text) ||
    /^The following is a <SYSTEM_MESSAGE>/i.test(trimmed)

  return { text, isSystemWake }
}

function tryParseJson(value: string): unknown {
  const trimmed = value.trim()
  if (!trimmed || (trimmed[0] !== '{' && trimmed[0] !== '[')) return undefined
  try {
    return JSON.parse(trimmed) as unknown
  } catch {
    return undefined
  }
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function clipInline(text: string, maxChars: number): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  if (flat.length <= maxChars) return flat
  return `${flat.slice(0, Math.max(1, maxChars - 1))}…`
}

/**
 * Format a worker tool-result entry for display in the Immortals inspector.
 * Strips duplicate timestamps, formats `read_board` JSON into readable message
 * lines, and folds long/structured payloads behind a concise summary.
 */
export function formatInspectorToolResult(
  toolName: string | undefined,
  rawText: string,
): FormattedInspectorToolResult {
  const cleaned = stripAgyToolTimestampBoilerplate(rawText)
  if (!cleaned) {
    return { summary: '(empty output)', body: '', collapsible: false, isJson: false }
  }

  const normalizedTool = (toolName ?? '').trim().toLowerCase()
  const parsed = tryParseJson(cleaned)
  const obj = asRecord(parsed)

  if (obj && (normalizedTool === 'read_board' || normalizedTool === 'swarm_read' || Array.isArray(obj.messages))) {
    const messages = Array.isArray(obj.messages) ? obj.messages : []
    const head = typeof obj.head === 'number' ? obj.head : undefined
    if (messages.length === 0) {
      const summary = head !== undefined ? `No new board messages · head #${head}` : 'No new board messages'
      return { summary, body: '', collapsible: false, isJson: false }
    }
    const lines: string[] = []
    const first = asRecord(messages[0])
    for (const rawItem of messages) {
      const item = asRecord(rawItem)
      if (!item) continue
      const seq = typeof item.seq === 'number' || typeof item.seq === 'string' ? `#${item.seq}` : ''
      const agent = typeof item.agent === 'string' && item.agent ? `@${item.agent}` : ''
      const text = typeof item.text === 'string' ? item.text.trim() : ''
      const header = [seq, agent].filter(Boolean).join(' ')
      lines.push(header ? `${header}: ${text}` : text)
    }
    const seqPart =
      messages.length === 1 && first && (typeof first.seq === 'number' || typeof first.seq === 'string')
        ? ` (#${first.seq}${typeof first.agent === 'string' && first.agent ? ` from @${first.agent}` : ''})`
        : ''
    const headPart = head !== undefined ? ` · head #${head}` : ''
    const summary = `Read ${messages.length} board message${messages.length === 1 ? '' : 's'}${seqPart}${headPart}`
    return {
      summary,
      body: lines.join('\n\n'),
      collapsible: true,
      isJson: false,
    }
  }

  if (obj && (normalizedTool === 'post_board' || normalizedTool === 'swarm_post')) {
    const seq = obj.seq ?? obj.message_seq
    if (typeof seq === 'number' || typeof seq === 'string') {
      return {
        summary: `Posted #${seq}`,
        body: JSON.stringify(obj, null, 2),
        collapsible: false,
        isJson: true,
      }
    }
  }

  if (parsed !== undefined) {
    const pretty = JSON.stringify(parsed, null, 2)
    const keys = obj ? Object.keys(obj) : []
    const summary = obj
      ? typeof obj.message === 'string' && obj.message.trim()
        ? clipInline(obj.message, 120)
        : typeof obj.status === 'string' && obj.status.trim()
          ? `status: ${obj.status}`
          : keys.length > 0
            ? `{ ${keys.slice(0, 4).join(', ')}${keys.length > 4 ? ', …' : ''} }`
            : '{}'
      : Array.isArray(parsed)
        ? `${parsed.length} item${parsed.length === 1 ? '' : 's'}`
        : clipInline(cleaned, 120)
    return {
      summary,
      body: pretty,
      collapsible: pretty.length > 140 || pretty.includes('\n'),
      isJson: true,
    }
  }

  const lineCount = cleaned.split(/\r?\n/).length
  const collapsible = cleaned.length > 260 || lineCount > 4
  return {
    summary: collapsible ? clipInline(cleaned, 120) : cleaned,
    body: cleaned,
    collapsible,
    isJson: false,
  }
}

/** Pretty-print JSON tool-call arguments when expanded in the inspector. */
export function formatInspectorToolArgs(rawArgs: string): string {
  const trimmed = rawArgs.trim()
  if (!trimmed) return ''
  const parsed = tryParseJson(trimmed)
  if (parsed !== undefined) {
    return JSON.stringify(parsed, null, 2)
  }
  return trimmed
}
