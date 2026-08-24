// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { branchLabel, diffLabel, EMPTY_PULSE, parseNumstat, parseStatusPorcelainV2 } from './repoPulse.js'

const STATUS = `# branch.oid a3f9c2100000000000000000000000000000abcd
# branch.head fix/scheduler-lease
# branch.ab +2 -1
1 .M N... 100644 100644 100644 aaa bbb packages/runtime/scheduler.ts
1 M. N... 100644 100644 100644 ccc ddd packages/runtime/lease.ts
2 R. N... 100644 100644 100644 eee fff R100 new.ts\told.ts
? notes.md
`

describe('parseStatusPorcelainV2', () => {
  it('reads the branch, how far out of step it is, and how much it owes you', () => {
    expect(parseStatusPorcelainV2(STATUS)).toEqual({
      ahead: 2,
      behind: 1,
      branch: 'fix/scheduler-lease',
      // Ordinary changes, renames and untracked files all count: they are all
      // things the tree owes you.
      dirty: 4
    })
  })

  it('reports no branch on a detached HEAD rather than the word itself', () => {
    expect(parseStatusPorcelainV2('# branch.head (detached)\n').branch).toBeNull()
  })

  it('reports no upstream as zero ahead and zero behind', () => {
    const parsed = parseStatusPorcelainV2('# branch.head main\n')

    expect(parsed).toEqual({ ahead: 0, behind: 0, branch: 'main', dirty: 0 })
  })
})

describe('parseNumstat', () => {
  it('sums line counts and counts the files that carry them', () => {
    expect(parseNumstat('21\t6\tscheduler.ts\n371\t0\tscheduler.test.ts\n')).toEqual({
      additions: 392,
      changedFiles: 2,
      deletions: 6
    })
  })

  it('counts a binary file without inventing line counts for it', () => {
    expect(parseNumstat('-\t-\tlogo.png\n')).toEqual({ additions: 0, changedFiles: 1, deletions: 0 })
  })
})

describe('branchLabel', () => {
  it('names the branch and how far out of step it is', () => {
    expect(branchLabel({ ahead: 2, behind: 0, branch: 'main' })).toBe('⎇ main ↑2')
    expect(branchLabel({ ahead: 0, behind: 3, branch: 'main' })).toBe('⎇ main ↓3')
    expect(branchLabel({ ahead: 2, behind: 3, branch: 'main' })).toBe('⎇ main ↑2 ↓3')
  })

  it('says nothing at all outside a repository', () => {
    expect(branchLabel({ ahead: 0, behind: 0, branch: null })).toBe('')
  })
})

describe('diffLabel', () => {
  it('states the totals and the file count', () => {
    expect(diffLabel({ ...EMPTY_PULSE, additions: 418, changedFiles: 4, deletions: 96 })).toBe('+418 −96 · 4 files')
  })

  it('omits a zero half rather than printing it', () => {
    expect(diffLabel({ ...EMPTY_PULSE, additions: 371, changedFiles: 1 })).toBe('+371 · 1 file')
  })

  it('says nothing on a clean tree, so the chip that reads it is not shown', () => {
    expect(diffLabel(EMPTY_PULSE)).toBe('')
  })
})
