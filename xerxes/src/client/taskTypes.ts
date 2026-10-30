// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { str } from './parse.js'

export type TaskStatus = 'idle' | 'working' | 'needs_you' | 'failed' | 'unknown'

/** What the task is observably doing right now. `none` means no evidence either way. */
export type TaskActivity =
  | { readonly kind: 'none' }
  | { readonly kind: 'thinking' }
  | { readonly kind: 'tool'; readonly name: string }
  | { readonly kind: 'agents'; readonly waiting: number }

export const NO_ACTIVITY: TaskActivity = Object.freeze({ kind: 'none' as const })
const THINKING: TaskActivity = Object.freeze({ kind: 'thinking' as const })

export function taskStatusOf(value: unknown): TaskStatus {
  if (value === 'idle' || value === 'failed' || value === 'needs_you' || value === 'unknown') return value
  if (value === 'working' || value === 'running' || value === 'busy' || value === 'streaming') return 'working'
  if (value === 'waiting' || value === 'waiting_permission' || value === 'waiting_question') return 'needs_you'
  return 'unknown'
}

export type ToolFamily = 'command' | 'read' | 'edit' | 'search' | 'web' | 'wait' | 'agents' | 'plan' | 'other'

const FAMILIES: ReadonlyArray<readonly [ToolFamily, RegExp]> = [
  ['command', /^(exec_?command|check_?command|bash|shell|run_?command|pty_\w+|kill_?command)$/],
  ['edit', /^(file_?edit|edit|write_?file|append_?file|notebook_?edit|multi_?edit|apply_?patch)$/],
  ['read', /^(read_?file|read|list_?dir|ls|view)$/],
  ['search', /^(grep|glob|search|find|lsp|tool_?search|search_?history)$/],
  ['web', /^(web_?scraper|web_?fetch|web_?search|google_?search|duck_?duck_?go_?search|api_?client|url_?analyzer|rss_?reader|browser\w*)$/],
  ['wait', /^(await_?agents)$/],
  ['agents', /^(agent|spawn_?agents|workflow|task_?create|handoff|send_?message|check_?agent_?messages|peek_?agent|task_?(get|list|output|stop|update)|reset_?agent)$/],
  ['plan', /^(todo_?write|plan|enter_?plan_?mode|exit_?plan_?mode)$/],
]

/** `FileEditTool` → `file_edit`, `mcp.server:TodoWrite` → `todo_write`: the comparable base of a tool name. */
export function toolBaseName(name: string): string {
  return (name.split(/[.:]/).pop() ?? name)
    .replace(/Tool$/, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[\s-]+/g, '_')
    .toLowerCase()
}

/** `FileEditTool` / `exec_command` / `mcp.server:GrepTool` → a family. */
export function toolFamily(name: string): ToolFamily {
  const tail = toolBaseName(name)
  for (const [family, pattern] of FAMILIES) if (pattern.test(tail)) return family
  return 'other'
}

/** One entry of the task's todo list (the latest TodoWriteTool / todo_write call). */
export interface TodoItem {
  readonly content: string
  readonly status: 'pending' | 'in_progress' | 'completed'
}

/**
 * One subagent of a task (`agents.list` / `agents.inspect` row, camel-cased).
 * `status` is the daemon's own string (running, completed, failed, cancelled, interrupted, queued …).
 */
export interface AgentRow {
  readonly id: string
  readonly name: string
  readonly title: string
  readonly status: string
  readonly model?: string
  readonly startedAt?: string
  readonly updatedAt?: string
  readonly toolCalls?: number
  readonly summary?: string
  readonly error?: string
}

/** One step of a subagent (`agents.inspect`). Text came from the run: plain text only. */
export interface AgentStep {
  readonly kind: 'text' | 'thinking' | 'tool' | 'notice'
  readonly text?: string
  readonly name?: string
  readonly args?: string
  readonly output?: string
  readonly state?: 'working' | 'done' | 'error'
}

export type ToolOrbState = 'working' | 'searching' | 'shaping' | 'connecting' | 'solving'

/** Which orb animation matches one running tool call. */
export function toolOrbState(name: string): ToolOrbState {
  switch (toolFamily(name)) {
    case 'read': case 'search': case 'web': return 'searching'
    case 'edit': return 'shaping'
    case 'wait': case 'agents': return 'connecting'
    case 'plan': return 'solving'
    default: return 'working'
  }
}

/** Present-tense phrase for a running tool, without its arguments (they may be sensitive). */
export function toolPhrase(name: string): string {
  switch (toolFamily(name)) {
    case 'command': return 'Running a command'
    case 'edit': return 'Editing files'
    case 'read': return 'Reading files'
    case 'search': return 'Searching the code'
    case 'web': return 'Searching the web'
    case 'wait': return 'Waiting on agents'
    case 'agents': return 'Working with agents'
    case 'plan': return 'Updating the plan'
    default: {
      const tail = (name.split(/[.:]/).pop() ?? name).replace(/Tool$/, '').replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_-]+/g, ' ').trim().toLowerCase()
      return tail ? `Using ${tail}` : 'Working'
    }
  }
}

/** Activity evidence after one event; unrelated events leave it alone. */
export function activityAfter(type: string, payload: Readonly<Record<string, unknown>>, current: TaskActivity, activeAgents = 0): TaskActivity {
  switch (type) {
    case 'think_part': return current.kind === 'thinking' ? current : THINKING
    case 'tool_call': {
      const name = str(payload.name) ?? 'tool'
      return toolFamily(name) === 'wait' ? { kind: 'agents', waiting: activeAgents } : { kind: 'tool', name }
    }
    case 'tool_result': case 'text_part': case 'turn_begin': case 'turn_end': case 'turn_error': case 'error':
      return current.kind === 'none' ? current : NO_ACTIVITY
    default: return current
  }
}
