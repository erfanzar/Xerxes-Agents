// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Block } from './types.js'

/** Pure conversation stays in the feed; operational events share a disclosure. */
export function isGroupedActivity(block: Block, approvalToolId?: string): boolean {
  return block.kind !== 'user' && block.kind !== 'agent'
}

/**
 * Only work earns a disclosure. A group of nothing but checkpoints or info
 * notices folded into a collapsed "Work activity" box trailing the final
 * answer — a header with nothing to say. Such groups render their rows
 * plainly. Errors stay folded: the failure card already states them.
 */
export function isDisclosedActivity(blocks: readonly Block[], approvalToolId?: string): boolean {
  return blocks.length > 0 && isGroupedActivity(blocks[0]!, approvalToolId)
    && blocks.some(block => block.kind === 'tools' || block.kind === 'thinking' || block.kind === 'agents' || block.kind === 'notice' && block.error)
}

/** The first activity owns the group even before a tool has been called. */
export function activityGroupKey(blocks: readonly Block[]): string {
  return `block:${blocks[0]?.id}`
}

/** Scope call IDs to their user turn: providers may reuse them in later turns. */
export function keyedActivityGroups(blocks: readonly Block[], approvalToolId?: string): { key: string; blocks: Block[] }[] {
  let turn = 'history'
  return groupActivity(blocks, approvalToolId).map(group => {
    if (group[0]?.kind === 'user') turn = `user:${group[0].id}`
    return { key: `${turn}:${activityGroupKey(group)}`, blocks: group }
  })
}

/**
 * Only conversation prose separates groups; approval controls render separately.
 * A turn-end checkpoint arrives after the final answer — it joins that turn's
 * last activity group rather than trailing the answer on its own.
 */
export function groupActivity(blocks: readonly Block[], approvalToolId?: string): Block[][] {
  const groups: Block[][] = []
  let activity: Block[] | null = null
  let turnActivity: Block[] | null = null
  for (const block of blocks) {
    if (block.kind === 'user') turnActivity = null
    if (block.kind === 'checkpoint' && !activity && turnActivity) { turnActivity.push(block); continue }
    const canGroup = isGroupedActivity(block, approvalToolId)
    if (canGroup) {
      if (!activity) { activity = []; groups.push(activity) }
      activity.push(block)
      turnActivity = activity
    } else { activity = null; groups.push([block]) }
  }
  return groups
}
