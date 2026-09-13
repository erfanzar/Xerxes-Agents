// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Block } from './types.js'

/** Keep prose, decisions, and approval requests outside activity disclosures. */
export function groupActivity(blocks: readonly Block[], approvalToolId?: string): Block[][] {
  const groups: Block[][] = []
  let activity: Block[] | null = null
  for (const block of blocks) {
    const canGroup = block.kind === 'thinking' || (block.kind === 'tools' && !block.items.some(item => item.id === approvalToolId))
    if (canGroup) {
      if (!activity) { activity = []; groups.push(activity) }
      activity.push(block)
    } else { activity = null; groups.push([block]) }
  }
  return groups
}
