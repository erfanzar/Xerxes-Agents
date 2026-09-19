// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Block } from './types.js'

/** A pending approval stays visible rather than inside a collapsed section. */
export function isGroupedActivity(block: Block, approvalToolId?: string): boolean {
  return block.kind === 'thinking' || (block.kind === 'tools' && !block.items.some(item => item.id === approvalToolId))
}

/** Tool call identities survive the live-to-committed block ID change. */
export function activityGroupKey(blocks: readonly Block[]): string {
  const firstTool = blocks.find(block => block.kind === 'tools')
  return firstTool?.kind === 'tools' && firstTool.items[0]
    ? `tool:${firstTool.items[0].id}`
    : `block:${blocks[0]?.id}`
}

/** Scope call IDs to their user turn: providers may reuse them in later turns. */
export function keyedActivityGroups(blocks: readonly Block[], approvalToolId?: string): { key: string; blocks: Block[] }[] {
  let turn = 'history'
  return groupActivity(blocks, approvalToolId).map(group => {
    if (group[0]?.kind === 'user') turn = `user:${group[0].id}`
    return { key: `${turn}:${activityGroupKey(group)}`, blocks: group }
  })
}

/** Keep prose, decisions, and approval requests outside activity disclosures. */
export function groupActivity(blocks: readonly Block[], approvalToolId?: string): Block[][] {
  const groups: Block[][] = []
  let activity: Block[] | null = null
  for (const block of blocks) {
    const canGroup = isGroupedActivity(block, approvalToolId)
    if (canGroup) {
      if (!activity) { activity = []; groups.push(activity) }
      activity.push(block)
    } else { activity = null; groups.push([block]) }
  }
  return groups
}
