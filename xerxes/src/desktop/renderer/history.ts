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
export function historyBlocks(page: HistoryPage, ids = new Map<string, number>()): Block[] {
  return page.actions.flatMap(action => blocksFromStoredMessages(action.messages, { executions: action.executions, thinking: action.thinking }).map((block, index) => {
    const key = `${action.id}:${index}`
    // Negative IDs are reserved for history; live and committed runs use positive IDs.
    const id = ids.get(key) ?? -(ids.size + 1)
    ids.set(key, id)
    return { ...block, id }
  }))
}
