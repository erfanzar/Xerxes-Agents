// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { fileAtRow, fileIndexFollowingRow, indexDiffFiles } from './diffFiles.js'
import type { DiffLine } from './gitDiff.js'

const L = (kind: DiffLine['kind'], text: string): DiffLine => ({ kind, text })

const DIFF: DiffLine[] = [
  L('file', 'diff --git a/src/one.ts b/src/one.ts'),
  L('meta', 'index 111..222 100644'),
  L('hunk', '@@ -1,3 +1,4 @@'),
  L('context', ' keep'),
  L('add', '+added'),
  L('add', '+added again'),
  L('del', '-gone'),
  L('file', 'diff --git a/src/two.ts b/src/two.ts'),
  L('hunk', '@@ -9,2 +9,2 @@'),
  L('del', '-old')
]

describe('indexDiffFiles', () => {
  it('finds each file, its row, and its counts', () => {
    expect(indexDiffFiles(DIFF)).toEqual([
      { deletions: 1, insertions: 2, line: 0, name: 'src/one.ts' },
      { deletions: 1, insertions: 0, line: 7, name: 'src/two.ts' }
    ])
  })

  it('does not open three entries for one file', () => {
    // `diff --git`, `---` and `+++` all classify as 'file' rows.
    const lines = [
      L('file', 'diff --git a/src/x.ts b/src/x.ts'),
      L('file', '--- a/src/x.ts'),
      L('file', '+++ b/src/x.ts'),
      L('add', '+one')
    ]

    expect(indexDiffFiles(lines).map(f => f.name)).toEqual(['src/x.ts'])
  })

  it('names a rename by where the file ended up', () => {
    const lines = [L('file', 'diff --git a/src/old.ts b/src/new.ts')]

    expect(indexDiffFiles(lines)[0]!.name).toBe('src/new.ts')
  })

  it('is empty for a clean tree', () => {
    expect(indexDiffFiles([])).toEqual([])
  })
})

describe('fileAtRow', () => {
  it('reports which file a row belongs to', () => {
    const files = indexDiffFiles(DIFF)

    expect(fileAtRow(files, 0)).toBe(0)
    expect(fileAtRow(files, 6)).toBe(0)
    expect(fileAtRow(files, 7)).toBe(1)
    expect(fileAtRow(files, 99)).toBe(1)
  })
})

describe('fileIndexFollowingRow', () => {
  const files = indexDiffFiles(DIFF)

  it('follows the top visible row into the file whose section contains it', () => {
    expect(fileIndexFollowingRow(files, 0, 0)).toBe(0)
    expect(fileIndexFollowingRow(files, 6, 0)).toBe(0)
    expect(fileIndexFollowingRow(files, 7, 0)).toBe(1)
    expect(fileIndexFollowingRow(files, 99, 0)).toBe(1)
  })

  it('returns the current index unchanged when the file did not move', () => {
    // The identity result is the contract that lets a scroll poll skip the
    // state update — and its re-render — until a boundary is crossed.
    expect(fileIndexFollowingRow(files, 3, 0)).toBe(0)
    expect(fileIndexFollowingRow(files, 99, 1)).toBe(1)
  })

  it('is stable for an empty index', () => {
    expect(fileIndexFollowingRow([], 4, 0)).toBe(0)
  })
})
