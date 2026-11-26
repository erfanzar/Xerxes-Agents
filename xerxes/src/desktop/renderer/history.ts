// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { blocksFromStoredMessages } from './blocks.js'
import type { Block } from './types.js'

export interface HistoryPage {
  actions: Array<{ id: string; messages: unknown[]; executions: unknown[]; thinking: unknown[] }>
  before: string | null
  has_more: boolean
}
export function readHistoryPage(value: unknown): HistoryPage | null {
  if (value === undefined) return null
  if (!value || typeof value !== 'object') throw new Error('Invalid history page')
  const page = value as HistoryPage
  if (!Array.isArray(page.actions) || page.actions.length > 100 || typeof page.has_more !== 'boolean' || (page.before !== null && typeof page.before !== 'string') || page.has_more !== Boolean(page.before)) throw new Error('Invalid history cursor')
  const ids = new Set<string>()
  for (const action of page.actions) {
    if (!action || typeof action.id !== 'string' || ids.has(action.id) || !Array.isArray(action.messages) || !Array.isArray(action.executions) || !Array.isArray(action.thinking)) throw new Error('Invalid history action')
    ids.add(action.id)
  }
  return page
}
const MSG_ACTION_RE = /^(\d+):message$/
const TOOL_ACTION_RE = /^(\d+):tool(?::|$)/

export function historyBlocks(page: HistoryPage, ids = new Map<string, number>()): Block[] {
  const perAction = page.actions.map(action => ({
    actionId: action.id,
    blocks: blocksFromStoredMessages(action.messages, { executions: action.executions, thinking: action.thinking }).map((block, index) => {
      const key = `${action.id}:${index}`
      // Negative IDs are reserved for history; live and committed runs use positive IDs.
      const id = ids.get(key) ?? -(ids.size + 1)
      ids.set(key, id)
      return { ...block, id }
    }),
  }))

  // Older remote daemons may emit `<idx>:message` (thinking + final reply) followed
  // by `<idx>:tool:*` for a flat assistant turn. When no subsequent assistant reply
  // exists in the same turn, place tools after thinking and BEFORE the final answer.
  const out: Block[] = []
  let i = 0
  while (i < perAction.length) {
    const current = perAction[i]!
    const msgMatch = MSG_ACTION_RE.exec(current.actionId)
    if (msgMatch && current.blocks.some(b => b.kind === 'agent')) {
      const msgIndex = msgMatch[1]!
      let toolEnd = i + 1
      while (toolEnd < perAction.length) {
        const tm = TOOL_ACTION_RE.exec(perAction[toolEnd]!.actionId)
        if (!tm || tm[1] !== msgIndex) break
        toolEnd += 1
      }
      if (toolEnd > i + 1) {
        let hasLaterAgentInTurn = false
        for (let k = toolEnd; k < perAction.length; k += 1) {
          const blocks = perAction[k]!.blocks
          if (blocks.some(b => b.kind === 'user')) break
          if (blocks.some(b => b.kind === 'agent')) {
            hasLaterAgentInTurn = true
            break
          }
        }
        if (!hasLaterAgentInTurn) {
          const thinkingBlocks = current.blocks.filter(b => b.kind === 'thinking')
          const nonThinkingBlocks = current.blocks.filter(b => b.kind !== 'thinking')
          out.push(...thinkingBlocks)
          for (let t = i + 1; t < toolEnd; t += 1) out.push(...perAction[t]!.blocks)
          out.push(...nonThinkingBlocks)
          i = toolEnd
          continue
        }
      }
    }
    out.push(...current.blocks)
    i += 1
  }
  return out
}
