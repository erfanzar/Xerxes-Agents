// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Pure state types, `/btw` + `/side` slash-command parser, and `'btw_event'`
 * stream reducer for Xerxes Ephemeral Side Questions.
 */

export interface SideQuestionToolEvent {
  readonly name: string
  readonly target?: string
  readonly status: 'running' | 'done' | 'error'
}

export interface SideQuestionTurn {
  readonly requestId: string
  readonly question: string
  readonly answer: string
  readonly model: string
  readonly status: 'streaming' | 'done' | 'error'
  readonly error?: string
  readonly tools?: readonly SideQuestionToolEvent[]
}

export type SideQuestionSlashAction =
  | { readonly kind: 'open' }
  | { readonly kind: 'ask'; readonly question: string; readonly model?: string }

const MAX_SIDE_QUESTION_TOOLS = 24

/** Parse `/btw` or `/side` composer slash commands (optionally supporting `-m`/`--model <id>`). */
export function parseSideQuestionSlashCommand(input: string): SideQuestionSlashAction | null {
  const trimmed = input.trim()
  const match = /^\/(?:btw|side)(?:\s+([\s\S]*))?$/i.exec(trimmed)
  if (!match) return null
  let rest = (match[1] ?? '').trim()
  if (!rest) return { kind: 'open' }
  let model = ''
  const modelMatch = /^(?:--model|-m)\s+(\S+)(?:\s+([\s\S]*))?$/i.exec(rest)
  if (modelMatch) {
    model = (modelMatch[1] ?? '').trim()
    rest = (modelMatch[2] ?? '').trim()
  }
  if (!rest) return { kind: 'open' }
  return model ? { kind: 'ask', question: rest, model } : { kind: 'ask', question: rest }
}

/** Parse a tool event from a daemon `'btw_event'` payload (`kind: 'tool'`). */
export function parseSideQuestionToolPayload(
  payload: Readonly<Record<string, unknown>>,
): SideQuestionToolEvent | null {
  const nested =
    payload.tool && typeof payload.tool === 'object' && !Array.isArray(payload.tool)
      ? (payload.tool as Readonly<Record<string, unknown>>)
      : null
  const rawName =
    (typeof nested?.name === 'string' && nested.name.trim())
    || (typeof payload.tool_name === 'string' && payload.tool_name.trim())
    || (typeof payload.name === 'string' && payload.name.trim())
    || ''
  if (!rawName) return null
  const rawTarget =
    (typeof nested?.target === 'string' && nested.target.trim())
    || (typeof payload.target === 'string' && payload.target.trim())
    || ''
  const rawStatus =
    (typeof nested?.status === 'string' && nested.status)
    || (typeof payload.status === 'string' && payload.status)
    || 'running'
  const status: SideQuestionToolEvent['status'] =
    rawStatus === 'done' || rawStatus === 'error' ? rawStatus : 'running'
  return {
    name: rawName,
    ...(rawTarget ? { target: rawTarget } : {}),
    status,
  }
}

/** Merge a tool status update into an existing turn's read-only tool pill list. */
export function mergeSideQuestionTool(
  existing: readonly SideQuestionToolEvent[] | undefined,
  incoming: SideQuestionToolEvent,
): readonly SideQuestionToolEvent[] {
  const list = existing ? [...existing] : []
  const incomingTarget = incoming.target ?? ''
  for (let idx = list.length - 1; idx >= 0; idx -= 1) {
    const item = list[idx]!
    if (item.name === incoming.name && (item.target ?? '') === incomingTarget && item.status === 'running') {
      list[idx] = incoming
      return list
    }
  }
  list.push(incoming)
  return list.length > MAX_SIDE_QUESTION_TOOLS ? list.slice(list.length - MAX_SIDE_QUESTION_TOOLS) : list
}

/** Mark any in-flight tool pills on a turn as settled when the answer finishes or errors. */
export function settleSideQuestionTools(
  tools: readonly SideQuestionToolEvent[] | undefined,
  terminal: 'done' | 'error' = 'done',
): readonly SideQuestionToolEvent[] | undefined {
  if (!tools || tools.length === 0) return tools
  let changed = false
  const next = tools.map(tool => {
    if (tool.status !== 'running') return tool
    changed = true
    return { ...tool, status: terminal }
  })
  return changed ? next : tools
}

/** Format a read-only tool event into a crisp pill verb + short target label. */
export function formatSideQuestionToolLabel(tool: SideQuestionToolEvent): {
  readonly verb: string
  readonly target?: string
} {
  const tail = (tool.name.split(/[.:/]/).pop() ?? tool.name).trim()
  const lower = tail.toLowerCase().replace(/tool$/i, '').replace(/[_-]+/g, '')
  let verb: string
  if (lower === 'readfile' || lower === 'read') verb = 'Read'
  else if (lower === 'grep' || lower === 'ripgrep' || lower === 'search' || lower === 'searchfiles') verb = 'Grep'
  else if (lower === 'glob' || lower === 'findfiles' || lower === 'find') verb = 'Glob'
  else if (lower === 'fileoutline' || lower === 'outline') verb = 'Outline'
  else if (lower === 'listdir' || lower === 'ls' || lower === 'list') verb = 'List'
  else if (lower === 'lsp') verb = 'LSP'
  else if (lower === 'webfetch') verb = 'Fetch'
  else if (lower === 'websearch') verb = 'Search'
  else {
    const clean = tail.replace(/Tool$/i, '').replace(/[_-]+/g, ' ').trim()
    verb = clean ? clean[0]!.toUpperCase() + clean.slice(1) : 'Tool'
  }

  const rawTarget = tool.target?.trim() ?? ''
  if (!rawTarget) return { verb }
  // Shorten file paths to their basename while preserving patterns/queries intact.
  const looksLikePath = (rawTarget.includes('/') || rawTarget.includes('\\')) && !/\s/.test(rawTarget)
  const displayTarget = looksLikePath
    ? (rawTarget.replaceAll('\\', '/').split('/').filter(Boolean).at(-1) ?? rawTarget)
    : rawTarget
  const clipped = displayTarget.length > 36 ? `${displayTarget.slice(0, 35)}…` : displayTarget
  return { verb, target: clipped }
}

/** Fold a daemon `'btw_event'` payload into the current ephemeral turn list. */
export function applyBtwEventPayload(
  turns: readonly SideQuestionTurn[],
  payload: Readonly<Record<string, unknown>>,
): {
  readonly turns: readonly SideQuestionTurn[]
  readonly streaming: boolean
  readonly error: string | null
} | null {
  const requestId = typeof payload.request_id === 'string' ? payload.request_id.trim() : ''
  const kind = typeof payload.kind === 'string' ? payload.kind : ''
  if (
    !requestId
    || (kind !== 'begin' && kind !== 'delta' && kind !== 'tool' && kind !== 'end' && kind !== 'error')
  ) {
    return null
  }
  const question = typeof payload.question === 'string' ? payload.question : ''
  const delta = typeof payload.delta === 'string' ? payload.delta : ''
  const answer = typeof payload.answer === 'string' ? payload.answer : ''
  const model = typeof payload.model === 'string' ? payload.model : ''
  const errorText = typeof payload.error === 'string' ? payload.error : ''
  const toolEvent = kind === 'tool' ? parseSideQuestionToolPayload(payload) : null
  if (kind === 'tool' && !toolEvent) return null

  const existingIndex = turns.findIndex(item => item.requestId === requestId)
  const next: SideQuestionTurn[] = [...turns]

  if (existingIndex === -1) {
    if (kind === 'begin') {
      next.push({
        requestId,
        question: question || 'Side question',
        answer: '',
        model,
        status: 'streaming',
      })
    } else if (kind === 'delta') {
      next.push({
        requestId,
        question: question || 'Side question',
        answer: delta,
        model,
        status: 'streaming',
      })
    } else if (kind === 'tool' && toolEvent) {
      next.push({
        requestId,
        question: question || 'Side question',
        answer: '',
        model,
        status: 'streaming',
        tools: [toolEvent],
      })
    } else if (kind === 'end') {
      next.push({
        requestId,
        question: question || 'Side question',
        answer,
        model,
        status: 'done',
      })
    } else {
      const err = errorText || 'Side question failed'
      next.push({
        requestId,
        question: question || 'Side question',
        answer: '',
        model,
        status: 'error',
        error: err,
      })
    }
  } else {
    const current = next[existingIndex]!
    if (kind === 'begin') {
      next[existingIndex] = {
        ...current,
        ...(question ? { question } : {}),
        ...(model ? { model } : {}),
        status: 'streaming',
      }
    } else if (kind === 'delta') {
      next[existingIndex] = {
        ...current,
        answer: current.answer + delta,
        ...(model ? { model } : {}),
        status: 'streaming',
      }
    } else if (kind === 'tool' && toolEvent) {
      next[existingIndex] = {
        ...current,
        ...(question ? { question } : {}),
        ...(model ? { model } : {}),
        tools: mergeSideQuestionTool(current.tools, toolEvent),
        status: 'streaming',
      }
    } else if (kind === 'end') {
      const settledTools = settleSideQuestionTools(current.tools, 'done')
      next[existingIndex] = {
        ...current,
        ...(question ? { question } : {}),
        answer: answer || current.answer,
        ...(model ? { model } : {}),
        ...(settledTools ? { tools: settledTools } : {}),
        status: 'done',
      }
    } else {
      const err = errorText || 'Side question failed'
      const settledTools = settleSideQuestionTools(current.tools, 'error')
      next[existingIndex] = {
        ...current,
        ...(question ? { question } : {}),
        ...(model ? { model } : {}),
        ...(settledTools ? { tools: settledTools } : {}),
        status: 'error',
        error: err,
      }
    }
  }

  const anyStreaming = next.some(item => item.status === 'streaming')
  const latestError = kind === 'error' ? (errorText || 'Side question failed') : null
  return {
    turns: next,
    streaming: anyStreaming,
    error: latestError,
  }
}
