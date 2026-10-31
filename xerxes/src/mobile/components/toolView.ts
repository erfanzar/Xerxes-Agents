// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Pure presentation helpers for tool rows, agent rows and question options.
 * JSX-free so the runtime tsconfig (no --jsx) can type-check it and tests can
 * call it directly.
 */

import { toolBaseName, toolFamily, type ToolFamily } from '../../client/taskTypes.js'

export type ToolRunState = 'working' | 'done' | 'error'

export interface ToolDiff {
  readonly path: string
  readonly before: string
  readonly after: string
}

/** One tool call as a row shows it, from a transcript block or an agent step. */
export interface ToolView {
  readonly name: string
  readonly summary: string
  readonly state: ToolRunState
  readonly args?: string | undefined
  readonly output?: string | undefined
  readonly diff?: ToolDiff | undefined
  readonly durationMs?: number | undefined
}

/** Output longer than this many lines starts collapsed behind "Show all". */
export const OUTPUT_PREVIEW_LINES = 20

const ICONS: Readonly<Record<ToolFamily, string>> = {
  command: '$', read: '≡', edit: '±', search: '⌕', web: '↗', wait: '…', agents: '⧉', plan: '☐', other: '•',
}

const VERBS: Readonly<Partial<Record<ToolFamily, readonly [running: string, finished: string]>>> = {
  command: ['Running', 'Ran'],
  read: ['Reading', 'Read'],
  edit: ['Editing', 'Edited'],
  search: ['Searching', 'Searched'],
  web: ['Fetching', 'Fetched'],
  wait: ['Waiting on agents', 'Waited on agents'],
  plan: ['Updating the plan', 'Updated the plan'],
}

/** `mcp.server:GrepTool` → `Grep`, `lesson_record` → `Lesson record`. */
export function humanToolName(name: string): string {
  const tail = (name.split(/[.:]/).pop() ?? name)
    .replace(/Tool$/, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .trim()
  if (tail === '') return 'Tool'
  return tail.charAt(0).toUpperCase() + tail.slice(1).toLowerCase()
}

export function toolIcon(name: string): string {
  return ICONS[toolFamily(name)]
}

/** Command-family tools that do not run a command themselves (they poll, stop or type into one). */
const BASE_VERBS: Readonly<Record<string, readonly [running: string, finished: string]>> = {
  check_command: ['Checking a command', 'Checked a command'],
  kill_command: ['Stopping a command', 'Stopped a command'],
  pty_open: ['Opening a terminal', 'Opened a terminal'],
  pty_write: ['Typing in a terminal', 'Typed in a terminal'],
  pty_close: ['Closing a terminal', 'Closed a terminal'],
  pty_list: ['Listing terminals', 'Listed terminals'],
}

/** "Ran", "Editing", or the tool's own name for tools without a verb. */
export function toolVerb(name: string, state: ToolRunState): string {
  const verbs = BASE_VERBS[toolBaseName(name)] ?? VERBS[toolFamily(name)]
  if (verbs === undefined) return humanToolName(name)
  return state === 'working' ? verbs[0] : verbs[1]
}

function firstLine(text: string): string {
  return (text.split('\n').find(line => line.trim() !== '') ?? '').trim()
}

/** The arguments as an object when they are the pretty-printed JSON the Hub sends for non-shell calls. */
export function argsObject(args: string | undefined): Readonly<Record<string, unknown>> | undefined {
  if (args === undefined) return undefined
  const text = args.trim()
  if (!text.startsWith('{')) return undefined
  try {
    const parsed = JSON.parse(text) as unknown
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed) ? parsed as Record<string, unknown> : undefined
  } catch { return undefined }
}

/** `{ "proc_id": "p3", "wait_ms": 60000 }` → `proc_id=p3 wait_ms=60000` (scalar values only, clipped). */
function compactArgs(record: Readonly<Record<string, unknown>>): string {
  const parts: string[] = []
  for (const [key, value] of Object.entries(record)) {
    if (typeof value !== 'string' && typeof value !== 'number' && typeof value !== 'boolean') continue
    const text = String(value).replace(/\s+/g, ' ').trim()
    if (text === '') continue
    parts.push(`${key}=${text.length > 40 ? `${text.slice(0, 39)}…` : text}`)
    if (parts.length === 3) break
  }
  const line = parts.join(' ')
  return line.length > 120 ? `${line.slice(0, 119)}…` : line
}

/**
 * What the call acted on: the command, the edited path, else the Hub's one-line
 * summary, else the scalar arguments. Never the opening brace of a JSON object.
 */
export function toolTarget(tool: ToolView): string {
  if (tool.diff !== undefined && tool.diff.path !== '') return tool.diff.path
  const record = argsObject(tool.args)
  if (toolFamily(tool.name) === 'command' && record === undefined && tool.args !== undefined && tool.args.trim() !== '') return firstLine(tool.args)
  if (tool.summary !== '') return tool.summary
  return record === undefined ? '' : compactArgs(record)
}

/** "350ms", "4.2s", "38s", "2m 05s". Undefined or negative durations render nothing. */
export function formatDuration(ms: number | undefined): string {
  if (ms === undefined || !Number.isFinite(ms) || ms < 0) return ''
  if (ms < 1000) return `${Math.round(ms)}ms`
  const seconds = ms / 1000
  if (seconds < 10) return `${seconds.toFixed(1)}s`
  if (seconds < 60) return `${Math.round(seconds)}s`
  const whole = Math.round(seconds)
  return `${Math.floor(whole / 60)}m ${String(whole % 60).padStart(2, '0')}s`
}

export function outputPreview(text: string, expanded: boolean, limit = OUTPUT_PREVIEW_LINES): { readonly text: string; readonly lines: number; readonly clipped: boolean } {
  const lines = text.split('\n')
  if (expanded || lines.length <= limit) return { text, lines: lines.length, clipped: false }
  return { text: lines.slice(0, limit).join('\n'), lines: lines.length, clipped: true }
}

const RECOMMENDED = /\s*\(\s*recommended\s*\)\s*/i

/** Tasks mark their suggestion "(Recommended)"; the chip shows a tag instead of the marker. */
export function optionLabel(option: string): { readonly label: string; readonly recommended: boolean } {
  if (!RECOMMENDED.test(option)) return { label: option, recommended: false }
  const label = option.replace(RECOMMENDED, ' ').replace(/\s+/g, ' ').trim()
  return { label: label === '' ? option : label, recommended: true }
}

export type AgentTone = 'busy' | 'done' | 'problem' | 'calm'

/** The daemon's status string (running | completed | failed | cancelled | interrupted | queued …) as a tone and a word. */
export function agentStatus(status: string): { readonly tone: AgentTone; readonly label: string; readonly active: boolean } {
  const value = status.trim().toLowerCase()
  if (value === 'running' || value === 'working' || value === 'streaming' || value === 'starting') return { tone: 'busy', label: 'Running', active: true }
  if (value === 'queued' || value === 'pending' || value === 'waiting') return { tone: 'busy', label: 'Queued', active: true }
  if (value === 'completed' || value === 'done' || value === 'finished' || value === 'succeeded') return { tone: 'done', label: 'Done', active: false }
  if (value === 'failed' || value === 'error' || value === 'errored') return { tone: 'problem', label: 'Failed', active: false }
  if (value === 'cancelled' || value === 'canceled' || value === 'stopped') return { tone: 'calm', label: 'Stopped', active: false }
  if (value === 'interrupted') return { tone: 'problem', label: 'Interrupted', active: false }
  return { tone: 'calm', label: value === '' ? 'Unknown' : value.charAt(0).toUpperCase() + value.slice(1), active: false }
}

/** One-line summary of an agent step's arguments, for the row target. */
export function argsSummary(args: string | undefined): string {
  if (args === undefined) return ''
  const record = argsObject(args)
  if (record !== undefined) return compactArgs(record)
  const line = firstLine(args)
  return line.length > 120 ? `${line.slice(0, 119)}…` : line
}
