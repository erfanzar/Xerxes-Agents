// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// What the working tree currently owes you, parsed from git's own porcelain.
//
// This exists because of one line in the home screen's design notes: "chips
// carry their own consequence — counts, ages, file totals — so the choice is
// informed before the keypress. A chip with nothing true to say is not
// shown." A chip that says `/diff review working tree` and nothing else is
// a menu item; the same chip saying `+418 −96 · 4 files` is a decision.
//
// Parsing is separated from running git so the shapes below can be tested
// against real porcelain without a repository.

export interface RepoPulse {
  /** Commits ahead of upstream; 0 when there is no upstream. */
  ahead: number
  behind: number
  branch: null | string
  /** Files changed in the working tree, tracked and untracked. */
  dirty: number
  additions: number
  deletions: number
  /** Files with staged or unstaged content changes (excludes untracked). */
  changedFiles: number
}

export const EMPTY_PULSE: RepoPulse = {
  ahead: 0,
  behind: 0,
  branch: null,
  dirty: 0,
  additions: 0,
  deletions: 0,
  changedFiles: 0
}

/**
 * Parse `git status --porcelain=v2 --branch --untracked-files=normal`.
 *
 * v2 rather than v1 because it carries `# branch.ab +n -m` — the ahead/behind
 * pair the statusbar shows — so one call answers both "which branch" and
 * "how far out of step", instead of two.
 */
export function parseStatusPorcelainV2(stdout: string): Pick<RepoPulse, 'ahead' | 'behind' | 'branch' | 'dirty'> {
  let ahead = 0
  let behind = 0
  let branch: null | string = null
  let dirty = 0

  for (const raw of stdout.split('\n')) {
    const line = raw.trimEnd()

    if (!line) {
      continue
    }

    if (line.startsWith('# branch.head ')) {
      const head = line.slice('# branch.head '.length).trim()
      branch = head && head !== '(detached)' ? head : null
      continue
    }

    if (line.startsWith('# branch.ab ')) {
      const match = /^# branch\.ab \+(\d+) -(\d+)$/.exec(line)

      if (match) {
        ahead = Number(match[1])
        behind = Number(match[2])
      }

      continue
    }

    // 1 = ordinary change, 2 = rename/copy, u = unmerged, ? = untracked.
    // All four are files the tree owes you; only `#` headers are not.
    if (/^[12u?] /.test(line)) {
      dirty += 1
    }
  }

  return { ahead, behind, branch, dirty }
}

/** Sum `git diff --numstat`. Binary files report `-` and contribute no lines. */
export function parseNumstat(stdout: string): Pick<RepoPulse, 'additions' | 'changedFiles' | 'deletions'> {
  let additions = 0
  let deletions = 0
  let changedFiles = 0

  for (const raw of stdout.split('\n')) {
    const line = raw.trim()

    if (!line) {
      continue
    }

    const [add, del] = line.split('\t')
    changedFiles += 1

    if (add && add !== '-') {
      additions += Number(add) || 0
    }

    if (del && del !== '-') {
      deletions += Number(del) || 0
    }
  }

  return { additions, changedFiles, deletions }
}

/**
 * `⎇ main ↑2` — branch plus how far out of step, or '' with no branch.
 *
 * Behind is shown as well as ahead because "you have work to push" and "you
 * are about to conflict" are different problems and only one of them is
 * fixed by pushing.
 */
export function branchLabel({ ahead, behind, branch }: Pick<RepoPulse, 'ahead' | 'behind' | 'branch'>): string {
  if (!branch) {
    return ''
  }

  return `⎇ ${branch}${ahead ? ` ↑${ahead}` : ''}${behind ? ` ↓${behind}` : ''}`
}

/**
 * `+418 −96 · 4 files`, or '' when the tree is clean.
 *
 * A U+2212 minus rather than a hyphen: it aligns with the digits either side
 * of it on the mono grid, which is the whole reason the diff totals read as a
 * pair rather than as a word.
 */
export function diffLabel({ additions, changedFiles, deletions }: RepoPulse): string {
  if (!changedFiles) {
    return ''
  }

  const totals = [additions ? `+${additions}` : '', deletions ? `−${deletions}` : ''].filter(Boolean).join(' ')
  const files = `${changedFiles} file${changedFiles === 1 ? '' : 's'}`

  return totals ? `${totals} · ${files}` : files
}
