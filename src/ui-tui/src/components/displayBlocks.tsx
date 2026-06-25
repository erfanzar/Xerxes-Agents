// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Render tool_result display blocks (normalized by lib/displayBlocks.ts).

import { Box, stringWidth, Text, useStdout } from '@xerxes/ink'

import type { DisplayBlock } from '../gatewayTypes.js'
import { diffLineKind } from '../lib/markdown.js'
import { todoItems } from '../lib/displayBlocks.js'
import { DEFAULT_THEME, type Theme } from '../theme.js'

const TODO_GLYPH: Record<string, string> = {
  completed: '✓',
  done: '✓',
  in_progress: '◐',
  pending: '○',
  cancelled: '✗'
}

const DIFF_PREVIEW_LINES = 18
const DIFF_PADDING_LEFT = 1
const ELLIPSIS = '…'

const clipCells = (text: string, width: number): string => {
  if (width <= 0 || stringWidth(text) <= width) {
    return text
  }

  const target = Math.max(1, width - stringWidth(ELLIPSIS))
  let out = ''
  let used = 0

  for (const ch of [...text]) {
    const w = stringWidth(ch)
    if (used + w > target) {
      break
    }
    out += ch
    used += w
  }

  return out + ELLIPSIS
}

const diffStats = (lines: string[]) => {
  let adds = 0
  let dels = 0

  for (const line of lines) {
    if (line.startsWith('+') && !line.startsWith('+++')) {
      adds++
    } else if (line.startsWith('-') && !line.startsWith('---')) {
      dels++
    }
  }

  return { adds, dels }
}

function DiffBlock({ diff, theme }: { diff: string; theme: Theme }) {
  const term = useStdout().stdout?.columns ?? 100
  const lines = diff.split('\n').filter((line, i, all) => line !== '' || i < all.length - 1)
  const shown = lines.slice(0, DIFF_PREVIEW_LINES)
  const hidden = Math.max(0, lines.length - shown.length)
  const { adds, dels } = diffStats(lines)
  const width = Math.max(12, term - DIFF_PADDING_LEFT - 8)

  return (
    <Box flexDirection="column" paddingLeft={DIFF_PADDING_LEFT}>
      <Text color={theme.color.muted} wrap="truncate-end">
        diff <Text color={theme.color.diffAddedWord}>+{adds}</Text>{' '}
        <Text color={theme.color.diffRemovedWord}>-{dels}</Text>
        {hidden ? (
          <Text color={theme.color.muted}>
            {' '}
            · showing {shown.length}/{lines.length} lines
          </Text>
        ) : null}
      </Text>

      {shown.map((line, i) => {
        const kind = diffLineKind(line)
        const color =
          kind === 'add'
            ? theme.color.diffAddedWord
            : kind === 'del'
              ? theme.color.diffRemovedWord
              : kind === 'meta'
                ? theme.color.muted
                : theme.color.text
        return (
          <Text color={color} dimColor={!kind && line.startsWith(' ')} key={i} wrap="truncate-end">
            <Text color={theme.color.muted}>│ </Text>
            {clipCells(line, width)}
          </Text>
        )
      })}

      {hidden > 0 ? (
        <Text color={theme.color.muted} wrap="truncate-end">
          │ {ELLIPSIS} {hidden} more line{hidden === 1 ? '' : 's'}
        </Text>
      ) : null}
    </Box>
  )
}

function OneBlock({ block, theme }: { block: DisplayBlock; theme: Theme }) {
  switch (block.type) {
    case 'brief':
      return <Text color={theme.color.muted}>{block.body}</Text>
    case 'generic':
      return <Text color={theme.color.text}>{block.content}</Text>
    case 'background_task':
      return (
        <Text color={theme.color.toolName}>
          ⧗ {block.title}
          {block.status ? ` — ${block.status}` : ''}
        </Text>
      )
    case 'diff':
      return <DiffBlock diff={block.diff} theme={theme} />
    case 'todo':
      return (
        <Box flexDirection="column">
          {todoItems(block).map((it, i) => {
            const done = it.status === 'completed' || it.status === 'done'
            return (
              <Text key={i} color={done ? theme.color.muted : theme.color.text}>
                {' '}
                {TODO_GLYPH[it.status] ?? '○'} {it.content}
              </Text>
            )
          })}
        </Box>
      )
  }
}

export interface DisplayBlocksProps {
  blocks: DisplayBlock[]
  theme?: Theme
}

export function DisplayBlocks({ blocks, theme = DEFAULT_THEME }: DisplayBlocksProps) {
  return (
    <Box flexDirection="column">
      {blocks.map((block, i) => (
        <OneBlock key={i} block={block} theme={theme} />
      ))}
    </Box>
  )
}
