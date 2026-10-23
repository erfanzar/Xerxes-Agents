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

export function BlockList({ blocks, prefix }: { blocks: readonly Block[]; prefix: string }): ReactElement | null {
  if (blocks.length === 0) return null
  return (
    <ol className="blocks">
      {blocks.map((block, index) => (
        <li key={`${prefix}${index}:${block.id}`} className={`block block-${block.kind}`}>
          <BlockView block={block} />
        </li>
      ))}
    </ol>
  )
}
