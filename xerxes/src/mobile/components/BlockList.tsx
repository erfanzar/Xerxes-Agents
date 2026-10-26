// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

import { plural } from '../../client/format.js'
import type { Block } from '../../client/sessionFold.js'
import { toolPhrase } from '../../client/taskTypes.js'

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
          <p className="block-who">Xerxes</p>
          <p className="block-text">{block.text}</p>
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
      return (
        <details className="fold">
          <summary>
            {toolPhrase(block.name)}
            <span className={`tool-state tool-${block.state}`}>{block.state === 'working' ? ' · running' : block.state === 'error' ? ' · failed' : ' · done'}</span>
          </summary>
          {block.summary === '' ? null : <code className="block-code">{block.summary}</code>}
          {block.output === undefined ? null : <pre className="block-code">{block.output}</pre>}
        </details>
      )
    case 'notice':
      return <p className={`block-notice${block.error ? ' is-error' : ''}`}>{block.text}</p>
  }
}

type ToolBlock = Extract<Block, { kind: 'tool' }>
const COMMAND_TOOLS = new Set(['exec_command', 'exec_shell', 'executeshell', 'bash', 'shell', 'run_command'])

/** Stable across a live tool being replaced by its authoritative history action. */
export function blockKey(block: Block): string {
  return block.kind === 'tool' ? `tool:${block.callId}` : block.id
}

export function BlockList({ blocks, prefix }: { blocks: readonly Block[]; prefix: string }): ReactElement | null {
  if (blocks.length === 0) return null
  const rows: (Block | ToolBlock[])[] = []
  for (const block of blocks) {
    const last = rows.at(-1)
    if (block.kind === 'tool') {
      if (Array.isArray(last)) last.push(block)
      else rows.push([block])
    } else rows.push(block)
  }
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
        const commands = row.every(tool => COMMAND_TOOLS.has(tool.name.toLowerCase()))
        const label = `${running ? commands ? 'Running' : 'Using' : commands ? 'Ran' : 'Used'} ${plural(row.length, commands ? 'command' : 'tool')}`
        return (
          <li key={`${prefix}:group:${row[0]!.callId}`} data-block-id={`group:${row[0]!.callId}`} className="block block-tools">
            <details className="fold tool-group">
              <summary>{label}{running ? <span className="tool-state tool-working"> · running</span> : null}{failed > 0 ? <span className="tool-state tool-error"> · {failed} failed</span> : null}</summary>
              <ol className="tool-rows" aria-label="Tool steps">
                {row.map(tool => <li key={tool.callId} data-block-id={blockKey(tool)}><BlockView block={tool} /></li>)}
              </ol>
            </details>
          </li>
        )
      })}
    </ol>
  )
}
