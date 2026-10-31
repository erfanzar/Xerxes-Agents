// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

import { plural } from '../../client/format.js'
import type { Block } from '../../client/sessionFold.js'
import { toolBaseName, toolFamily } from '../../client/taskTypes.js'
import { Markdown } from './Markdown.js'
import { ToolRow } from './ToolRow.js'
import type { ToolView } from './toolView.js'

type ToolBlock = Extract<Block, { kind: 'tool' }>

/** Runs of this many consecutive tool calls fold into one collapsed group; each step stays its own row inside. */
export const TOOL_GROUP_MIN = 2

export function toolViewOf(block: ToolBlock): ToolView {
  return {
    name: block.name,
    summary: block.summary,
    state: block.state,
    args: block.args,
    output: block.output,
    diff: block.diff,
    durationMs: block.durationMs,
  }
}

function BlockView({ block }: { block: Block }): ReactElement {
  switch (block.kind) {
    case 'user':
      return (
        <>
          <p className="block-who">{block.steer ? 'Steer' : 'You'}{block.author === undefined ? '' : ` · ${block.author}`}</p>
          <p className="block-text">{block.text}</p>
          {block.images > 0 ? <p className="block-meta">{plural(block.images, 'image')} attached</p> : null}
        </>
      )
    case 'assistant':
      return (
        <>
          <p className="block-who">Xerxes{block.streaming ? <span className="block-meta"> · writing…</span> : null}</p>
          <Markdown text={block.text} className="block-md" />
          {block.streaming ? <span className="stream-caret" aria-hidden="true" /> : null}
        </>
      )
    case 'thinking':
      return (
        <details className="fold">
          <summary>{block.streaming ? 'Thinking…' : 'Thought it through'}</summary>
          <p className="block-text block-dim">{block.text}</p>
        </details>
      )
    case 'tool':
      return <ToolRow tool={toolViewOf(block)} />
    case 'notice':
      return <p className={`block-notice${block.error ? ' is-error' : ''}`}>{block.text}</p>
  }
}

/** Stable across a live tool being replaced by its authoritative history action. */
export function blockKey(block: Block): string {
  return block.kind === 'tool' ? `tool:${block.callId}` : block.id
}

function isTool(block: Block): block is ToolBlock {
  return block.kind === 'tool'
}

type GroupKind = 'run' | 'check' | 'read' | 'edit' | 'search' | 'web' | 'agents' | 'plan' | 'other'

const GROUP_ORDER: readonly GroupKind[] = ['run', 'check', 'edit', 'read', 'search', 'web', 'agents', 'plan', 'other']

/** Polling, stopping or typing into a command is not running one: count it apart. */
const COMMAND_HELPERS = new Set(['check_command', 'kill_command', 'pty_open', 'pty_write', 'pty_close', 'pty_list'])

function groupKind(name: string): GroupKind {
  const family = toolFamily(name)
  if (family === 'command') return COMMAND_HELPERS.has(toolBaseName(name)) ? 'check' : 'run'
  if (family === 'wait') return 'agents'
  return family
}

function groupPart(kind: GroupKind, count: number): string {
  switch (kind) {
    case 'run': return `ran ${plural(count, 'command')}`
    case 'check': return plural(count, 'command check')
    case 'edit': return `edited ${plural(count, 'file')}`
    case 'read': return `read ${plural(count, 'file')}`
    case 'search': return plural(count, 'search', 'searches')
    case 'web': return plural(count, 'web fetch', 'web fetches')
    case 'agents': return plural(count, 'agent step')
    case 'plan': return plural(count, 'plan update')
    case 'other': return plural(count, 'other tool')
  }
}

/** "Ran 3 commands, 2 command checks, read 1 file". */
export function groupLabel(tools: readonly ToolBlock[]): string {
  const counts = new Map<GroupKind, number>()
  for (const tool of tools) {
    const kind = groupKind(tool.name)
    counts.set(kind, (counts.get(kind) ?? 0) + 1)
  }
  const text = GROUP_ORDER.filter(kind => counts.has(kind)).map(kind => groupPart(kind, counts.get(kind)!)).join(', ')
  return text.charAt(0).toUpperCase() + text.slice(1)
}

/**
 * The transcript. Consecutive tool calls fold into one collapsed group
 * ("Ran 3 commands, read 2 files") between the replies; opening it lists
 * every call as its own expandable row (verb, target, state, duration, and
 * inside it the full command, diff and output).
 */
export function BlockList({ blocks, prefix }: { blocks: readonly Block[]; prefix: string }): ReactElement | null {
  if (blocks.length === 0) return null
  const rows: (Block | ToolBlock[])[] = []
  let run: ToolBlock[] = []
  const flush = (): void => {
    if (run.length >= TOOL_GROUP_MIN) rows.push(run)
    else rows.push(...run)
    run = []
  }
  for (const block of blocks) {
    if (isTool(block)) { run.push(block); continue }
    flush()
    rows.push(block)
  }
  flush()
  return (
    <ol className="blocks" aria-label="Conversation">
      {rows.map(row => {
        if ('kind' in row) return (
          <li key={`${prefix}:${blockKey(row)}`} data-block-id={blockKey(row)} className={`block block-${row.kind}`}>
            <BlockView block={row} />
          </li>
        )
        const failed = row.filter(tool => tool.state === 'error').length
        const running = row.some(tool => tool.state === 'working')
        return (
          <li key={`${prefix}:group:${row[0]!.callId}`} data-block-id={`group:${row[0]!.callId}`} className="block block-tools">
            <details className="fold tool-group">
              <summary>{groupLabel(row)}{running ? <span className="tool-state tool-working"> · running</span> : null}{failed > 0 ? <span className="tool-state tool-error"> · {failed} failed</span> : null}</summary>
              <ol className="tool-rows" aria-label="Tool steps">
                {row.map(tool => <li key={tool.callId} data-block-id={blockKey(tool)}><ToolRow tool={toolViewOf(tool)} /></li>)}
              </ol>
            </details>
          </li>
        )
      })}
    </ol>
  )
}
