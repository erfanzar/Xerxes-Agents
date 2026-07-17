// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { SnapshotManager, type SnapshotRecord } from './snapshots.js'

/** Textual workspace diff relative to a shadow-git snapshot. */
export interface SnapshotDiff {
  readonly added: number
  readonly diffText: string
  readonly fileCount: number
  readonly removed: number
  readonly snapshot: SnapshotRecord
}

/** Compare the current workspace with a snapshot without restoring it. */
export async function diffAgainstSnapshot(manager: SnapshotManager, ref: string): Promise<SnapshotDiff> {
  const snapshot = manager.get(ref)
  if (!snapshot) throw new Error(`snapshot not found: ${ref}`)
  const diffText = await manager.runGit(['diff', snapshot.commitSha, '--', '.'])
  const summary = summarizeDiff(diffText)
  return { snapshot, diffText, ...summary }
}

export function summarizeDiff(diffText: string): Omit<SnapshotDiff, 'diffText' | 'snapshot'> {
  let fileCount = 0
  let added = 0
  let removed = 0
  for (const line of diffText.split(/\r?\n/)) {
    if (line.startsWith('diff --git ')) fileCount += 1
    else if (line.startsWith('+') && !line.startsWith('+++')) added += 1
    else if (line.startsWith('-') && !line.startsWith('---')) removed += 1
  }
  return { fileCount, added, removed }
}
