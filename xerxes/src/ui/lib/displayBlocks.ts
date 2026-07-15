// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Pure normalization of tool_result.display_blocks (wire_events.py DisplayBlock
// union) into a typed, render-ready form. Unknown/malformed blocks are dropped.

import type { DisplayBlock } from '../gatewayTypes.js'

export interface TodoItem {
  status: string
  content: string
}

const asStr = (v: unknown, d = ''): string => (typeof v === 'string' ? v : d)

function normalizeOne(raw: unknown): DisplayBlock | null {
  if (!raw || typeof raw !== 'object') {
    return null
  }
  const b = raw as Record<string, unknown>
  switch (b.type) {
    case 'brief':
      return { type: 'brief', body: asStr(b.body) }
    case 'diff':
      return { type: 'diff', diff: asStr(b.diff), language: asStr(b.language) }
    case 'background_task':
      return { type: 'background_task', title: asStr(b.title), status: asStr(b.status) }
    case 'generic':
      return { type: 'generic', content: asStr(b.content) }
    case 'todo':
      return { type: 'todo', items: Array.isArray(b.items) ? (b.items as Record<string, unknown>[]) : [] }
    default:
      return null
  }
}

/** Coerce a raw display_blocks payload into a typed, validated list. */
export function normalizeDisplayBlocks(raw: unknown): DisplayBlock[] {
  if (!Array.isArray(raw)) {
    return []
  }
  return raw.map(normalizeOne).filter((b): b is DisplayBlock => b !== null)
}

/** Extract todo items into a uniform {status, content} shape. */
export function todoItems(block: Extract<DisplayBlock, { type: 'todo' }>): TodoItem[] {
  return block.items.map(it => ({ status: asStr(it.status, 'pending'), content: asStr(it.content) }))
}

/** A short one-line summary for a tool result row (used when no blocks render). */
export function summarizeResult(returnValue: string, durationMs: number): string {
  const firstLine = returnValue.split('\n', 1)[0]?.trim() ?? ''
  if (firstLine) {
    return firstLine.length > 120 ? firstLine.slice(0, 117) + '…' : firstLine
  }
  return durationMs ? `done (${Math.round(durationMs)}ms)` : 'done'
}
