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

import type { AgentMember, Block, ToolItem } from './types.js'

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
type Run = ToolRun | TextRun | ThinkingRun

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
const ARG_KEYS = ['file_path', 'path', 'cmd', 'command', 'pattern', 'query', 'url', 'skill', 'name']

function detailOf(value: unknown): string {
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return ''
    try {
      return JSON.stringify(JSON.parse(trimmed) as unknown, null, 2)
    } catch {
      return value
    }
  }
  if (value === undefined || value === null) return ''
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

/** First-argument-ish summary of a tool call's arguments. */
function argOf(args: unknown): string {
  const clip = (value: string): string => (value.length > 72 ? `${value.slice(0, 69)}…` : value)
  const parsed = parseArgs(args)
  for (const key of ARG_KEYS) {
    const value = parsed[key]
    if (typeof value === 'string' && value.trim()) return clip(value)
  }
  const first = Object.values(parsed)[0]
  if (typeof first === 'string' && first.trim()) return clip(first)
  if (typeof args === 'string' && args.trim()) return clip(args)
  return ''
}

/** Extract the text of a turn_begin user_input (string or Part[]). */
export function userTextOf(payload: Readonly<Record<string, unknown>>): string {
  if (typeof payload.user_input !== 'string') return ''
  return payload.user_input
}

/** Edit-family tool names whose arguments carry real diff material. */
const EDIT_TOOLS = new Set(['fileedittool', 'writefile', 'appendfile'])

export function isEditTool(name: unknown): boolean {
  return typeof name === 'string' && EDIT_TOOLS.has(name.split(/[.:]/).pop()?.toLowerCase() ?? '')
}

/**
 * Parse tool-call arguments (JSON string or object) without throwing.
 * Malformed arguments still render as a trail row; they just carry no diff.
 */
export function parseArgs(args: unknown): Record<string, unknown> {
  if (typeof args === 'string' && args.trim()) {
    try {
      const parsed = JSON.parse(args) as unknown
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>
      }
    } catch {
      return {}
    }
  }
  if (args && typeof args === 'object' && !Array.isArray(args)) return args as Record<string, unknown>
  return {}
}

/**
 * File + line delta an edit call will produce, or null for other tools.
 * `WriteFile`/`AppendFile` have no old text: everything they land is an add.
 */
export function editStatsOf(
  name: unknown,
  args: unknown,
): { path: string; adds: number; dels: number; isNew: boolean } | null {
  if (!isEditTool(name)) return null
  const parsed = parseArgs(args)
  const path = typeof parsed.file_path === 'string' ? parsed.file_path : typeof parsed.path === 'string' ? parsed.path : ''
  if (!path) return null
  const mode = typeof parsed.edit_mode === 'string' ? parsed.edit_mode : 'search_replace'
  const isNew = String(name).toLowerCase().endsWith('writefile') && mode !== 'search_replace' ? true : !parsed.old_string
  const count = (value: unknown): number =>
    typeof value === 'string' && value.length ? value.replace(/\n$/, '').split('\n').length : 0
  return {
    path,
    adds: count(parsed.new_string ?? parsed.content),
    dels: isNew ? 0 : count(parsed.old_string),
    isNew,
  }
}

export class BlockBuilder {
  private blocks: Block[] = []
  private seq = 1
  /** Contiguous same-kind event runs, in stream order. */
  private runs: Run[] = []
  /** Every tool call/result of the turn by id, regardless of which run holds the row. */
  private tools = new Map<string, ToolItem>()
  /**
   * The turn's spawned-subagents card. Deliberately NOT a run: it updates in
   * place as members start/finish, mid-turn and (via the committed copy)
   * after the turn ends while background children settle. Cleared on
   * finalize so the next turn's spawn opens a fresh card.
   */
  private agentsCard: { readonly id: number; readonly members: readonly AgentMember[] } | null = null

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
    if (this.agentsCard) {
      this.agentsCard = { id: this.agentsCard.id, members }
      const committed = this.blocks.findIndex(block => block.id === this.agentsCard!.id)
      if (committed >= 0) this.blocks[committed] = { kind: 'agents', id: this.agentsCard.id, members }
      return
    }
    this.agentsCard = { id: this.nextId(), members }
  }

  /**
   * The run a kind appends to: the trailing run when it matches, otherwise a
   * freshly opened one — this is what makes think→tools→think produce two
   * thinking blocks instead of appending to the first.
   */
  private openRun(kind: Run['kind']): Run {
    const last = this.runs[this.runs.length - 1]
    if (last && last.kind === kind) return last
    const run: Run =
      kind === 'tools' ? { kind: 'tools', order: [] } : kind === 'thinking' ? { kind: 'thinking', text: '' } : { kind: 'text', text: '' }
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
      case 'text_part': {
        if (typeof payload.text === 'string' && payload.text) {
          const run = this.openRun('text')
          if (run.kind === 'text') run.text += payload.text
        }
        break
      }
      case 'think_part': {
        const delta = typeof payload.think === 'string' ? payload.think : typeof payload.text === 'string' ? payload.text : ''
        if (delta) {
          const run = this.openRun('thinking')
          if (run.kind === 'thinking') run.text += delta
        }
        break
      }
      case 'tool_call': {
        const id = typeof payload.id === 'string' ? payload.id : typeof payload.tool_call_id === 'string' ? payload.tool_call_id : `t${this.seq++}`
        const run = this.openRun('tools')
        if (run.kind === 'tools' && !run.order.includes(id)) run.order.push(id)
        const stats = editStatsOf(payload.name, payload.arguments)
        this.tools.set(id, {
          id,
          verb: verbOf(payload.name),
          arg: argOf(payload.arguments),
          dur: '',
          state: 'working',
          name: typeof payload.name === 'string' ? payload.name : '',
          input: detailOf(payload.arguments),
          output: '',
          ...(stats ? { path: stats.path, diff: { adds: stats.adds, dels: stats.dels } } : {}),
        })
        break
      }
      case 'tool_result': {
        const id = typeof payload.tool_call_id === 'string' ? payload.tool_call_id : typeof payload.id === 'string' ? payload.id : ''
        const existing = this.tools.get(id)
        const stats = editStatsOf(payload.name, existing ? undefined : payload.arguments)
        const item: ToolItem = {
          id,
          verb: existing?.verb ?? verbOf(payload.name),
          arg: existing?.arg ?? '',
          dur: dur(payload.duration_ms),
          state: typeof payload.error === 'string' && payload.error ? 'failed' : 'done',
          name: existing?.name ?? (typeof payload.name === 'string' ? payload.name : ''),
          input: existing?.input ?? detailOf(payload.arguments),
          output: detailOf(payload.return_value ?? payload.result ?? payload.output),
          ...(typeof payload.error === 'string' && payload.error ? { error: payload.error } : {}),
          ...(existing?.path ? { path: existing.path } : stats ? { path: stats.path } : {}),
          ...(existing?.diff ? { diff: existing.diff } : stats ? { diff: { adds: stats.adds, dels: stats.dels } } : {}),
        }
        if (existing) this.tools.set(id, item)
        else {
          // A result with no matching call (replay quirks): the row lands on
          // the trailing tools run, opening one if the stream just switched.
          const run = this.openRun('tools')
          if (run.kind === 'tools' && !run.order.includes(id)) run.order.push(id)
          this.tools.set(id, item)
        }
        break
      }
      case 'notification': {
        const message =
          (typeof payload.body === 'string' && payload.body) ||
          (typeof payload.message === 'string' && payload.message) ||
          ''
        if (!message) break
        this.finalize()
        const severity = String(payload.severity ?? payload.level ?? 'info').toLowerCase()
        this.blocks.push({ kind: 'notice', id: this.nextId(), error: severity.includes('error') || severity.includes('fatal'), text: message })
        break
      }
      default:
        break
    }
  }

  /** A user line lands at the fold's end (submit ordering, replays). */
  pushUser(text: string): void {
    this.finalize()
    this.blocks.push({ kind: 'user', id: this.nextId(), text })
  }

  /**
   * Undo the most recent pushUser matching `text` — a submit the daemon
   * rejected must not linger as a delivered-looking ghost bubble.
   */
  rollbackUser(text: string): void {
    for (let i = this.blocks.length - 1; i >= 0; i--) {
      const block = this.blocks[i]!
      if (block.kind === 'user') {
        if (block.text === text) this.blocks.splice(i, 1)
        return
      }
    }
  }

  /** A turn boundary with cumulative edit totals — the checkpoint marker. */
  pushCheckpoint(turn: number, totals: { adds: number; dels: number }): void {
    // Only turns that changed something earn a marker; a conversational
    // exchange should not sprinkle dashed rows through the transcript.
    if (totals.adds === 0 && totals.dels === 0) return
    this.blocks.push({ kind: 'checkpoint', id: this.nextId(), turn, adds: totals.adds, dels: totals.dels })
  }

  /**
   * Current fold. While a turn is active, trailing runs render as
   * streaming-tailed blocks in stream order; `finalize` commits them as
   * finished blocks.
   */
  snapshot(turnActive: boolean): readonly Block[] {
    if (!turnActive) return this.blocks
    // Only the TRAILING run can still receive deltas — an earlier run closed
    // the moment a different kind opened after it. Marking every run
    // streaming kept a blinking caret on text the model had already finished
    // (the "caret sticks around after the agent is done talking" bug).
    const tail = this.runs.length - 1
    const live = this.runs.map((run, index) => this.runBlock(run, index, index === tail))
    // The agents card trails the live activity — it updates in place.
    if (this.agentsCard) {
      return [...this.blocks, ...live, { kind: 'agents', id: this.agentsCard.id, members: this.agentsCard.members }]
    }
    return [...this.blocks, ...live]
  }

  /** Render one run as a display block. `streaming` tails the live one. */
  private runBlock(run: Run, index: number, streaming: boolean): Block {
    // Scratch ids live above any committed id and descend per run, so React
    // keys stay stable while runs only ever append during a turn.
    const id = Number.MAX_SAFE_INTEGER - index
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
    // Close any tool still marked working: the turn ended, so nothing can
    // land a result anymore. Closing before rendering keeps the trail honest.
    for (const [id, tool] of this.tools) {
      if (tool.state === 'working') this.tools.set(id, { ...tool, state: 'done', dur: tool.dur || '' })
    }
    for (const [index, run] of this.runs.entries()) {
      const block = this.runBlock(run, index, false)
      if (block.kind === 'tools') this.blocks.push({ ...block, id: this.nextId(), running: false })
      else if (block.kind === 'thinking') this.blocks.push({ ...block, id: this.nextId(), streaming: false })
      else if (block.kind === 'agent') this.blocks.push({ ...block, id: this.nextId(), streaming: false })
    }
    this.runs = []
    this.tools.clear()
    // Commit the agents card behind the drained runs, keeping the reference:
    // terminal statuses still land on it while background children settle.
    // Idempotent — all() finalizes again and must not double-commit.
    if (this.agentsCard && !this.blocks.some(block => block.id === this.agentsCard!.id)) {
      this.blocks.push({ kind: 'agents', id: this.agentsCard.id, members: this.agentsCard.members })
    }
  }

  /**
   * Close the current agents card (the store calls this at the next
   * turn_begin): the committed card stays and keeps receiving status
   * updates until then; a later spawn opens a fresh card.
   */
  closeAgentsCard(): void {
    this.agentsCard = null
  }

  /** Replace the whole fold (session open/resume hydration). */
  reset(blocks: Block[] = []): void {
    this.blocks = blocks
    this.runs = []
    this.tools.clear()
    this.agentsCard = null
  }

  all(): readonly Block[] {
    return this.finalize(), this.blocks
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

/**
 * Replay one stored execution as the same call+result event pair the live
 * stream emits, so a resumed transcript shows the identical verb + compact
 * arg + duration row instead of silently dropping the tool activity.
 */
function pushStoredExecution(builder: BlockBuilder, exec: unknown, index: number): void {
  if (!isRecord(exec)) return // no fabricated rows for missing records
  const id = typeof exec.toolCallId === 'string' && exec.toolCallId ? exec.toolCallId : `stored-${index}`
  builder.push('tool_call', { id, name: exec.name, arguments: exec.inputs ?? exec.arguments })
  builder.push('tool_result', {
    tool_call_id: id,
    name: exec.name,
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
  let executionIndex = 0
  let assistantIndex = 0
  for (const message of messages) {
    if (!message || typeof message !== 'object') continue
    const record = message as Record<string, unknown>
    const role = typeof record.role === 'string' ? record.role : ''

    if (role === 'tool') {
      // A stored result: its call lives in tool_executions at the same order.
      pushStoredExecution(builder, executions[executionIndex], executionIndex)
      executionIndex += 1
      continue
    }

    // Reasoning precedes its answer, as it streamed.
    const turnThinking = role === 'assistant' ? thinking[assistantIndex] : undefined
    if (role === 'assistant') assistantIndex += 1
    if (typeof turnThinking === 'string' && turnThinking.trim()) {
      builder.push('think_part', { think: turnThinking })
    }

    const content = record.content
    if (typeof content === 'string') {
      if (!content.trim()) continue
      if (role === 'user') builder.pushUser(content)
      else if (role === 'assistant') builder.push('text_part', { text: content })
      else if (role !== 'system') builder.push('think_part', { think: content })
      continue
    }
    if (!Array.isArray(content)) continue
    // Parts fold sequentially — a [think, text] assistant message replays as
    // a thinking block then an agent block, matching the live run grammar.
    const userParts: string[] = []
    for (const part of content) {
      if (!part || typeof part !== 'object') continue
      const p = part as Record<string, unknown>
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
    if (userParts.length) builder.pushUser(userParts.join('\n'))
  }
  return [...builder.all()]
}
