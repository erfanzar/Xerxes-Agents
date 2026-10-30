// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

import { plural } from '../../client/format.js'
import type { Block } from '../../client/sessionFold.js'
import { toolFamily } from '../../client/taskTypes.js'
import { Markdown } from './Markdown.js'
import { ToolRow } from './ToolRow.js'
import type { ToolView } from './toolView.js'

type ToolBlock = Extract<Block, { kind: 'tool' }>

/** Runs of this many consecutive reads/searches fold into one group; each step stays its own row. */
export const QUIET_GROUP_MIN = 3

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

function quiet(block: Block): block is ToolBlock {
  if (block.kind !== 'tool') return false
  const family = toolFamily(block.name)
  return family === 'read' || family === 'search'
}

function groupLabel(tools: readonly ToolBlock[]): string {
  const reads = tools.filter(tool => toolFamily(tool.name) === 'read').length
  const searches = tools.length - reads
  const running = tools.some(tool => tool.state === 'working')
  const parts: string[] = []
  if (reads > 0) parts.push(plural(reads, 'file'))
  if (searches > 0) parts.push(plural(searches, 'search', 'searches'))
  return `${running ? 'Exploring' : 'Explored'} ${parts.join(', ')}`
}

/**
 * The transcript: every tool call is its own expandable row (verb, target,
 * state, duration). Only runs of reads and searches fold together, and each
 * step inside that group still expands on its own.
 */
export function BlockList({ blocks, prefix }: { blocks: readonly Block[]; prefix: string }): ReactElement | null {
  if (blocks.length === 0) return null
  const rows: (Block | ToolBlock[])[] = []
  let run: ToolBlock[] = []
  const flush = (): void => {
    if (run.length >= QUIET_GROUP_MIN) rows.push(run)
    else rows.push(...run)
    run = []
  }
  for (const block of blocks) {
    if (quiet(block)) { run.push(block); continue }
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
