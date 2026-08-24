// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// The diff viewer rendered one long scroll. With forty-five changed files
// that means no sense of where you are in the change set and no way to reach
// file forty except by paging there.
//
// The parsed diff already knows where every file starts — this turns that
// into an index the panel can show and jump to.

import type { DiffLine } from './gitDiff.js'

export interface DiffFileEntry {
  deletions: number
  insertions: number
  /** Row offset of this file's header within the rendered diff. */
  line: number
  name: string
}

/** `diff --git a/src/x.ts b/src/x.ts` / `--- a/src/x.ts` → `src/x.ts`. */
const fileName = (text: string): string => {
  const both = /^diff --git a\/(.+?) b\/(.+)$/.exec(text)

  if (both) {
    // A rename shows both sides; the destination is the one you'd go looking
    // for, since that is where the file lives now.
    return both[2]!.trim()
  }

  const single = /^(?:\+\+\+|---)\s+[ab]\/(.+)$/.exec(text)

  if (single) {
    return single[1]!.trim()
  }

  return text.replace(/^diff --git\s+/, '').trim()
}

/**
 * One entry per file in the parsed diff, in render order, each carrying the
 * row it starts at so selecting it can scroll straight there.
 */
export function indexDiffFiles(lines: readonly DiffLine[]): DiffFileEntry[] {
  const files: DiffFileEntry[] = []

  lines.forEach((line, index) => {
    if (line.kind === 'file') {
      const name = fileName(line.text)

      // `diff --git` and the `---`/`+++` pair all classify as 'file'; only
      // the first of a run opens a new entry, or every file would appear
      // three times.
      const current = files.at(-1)

      if (current && current.name === name) {
        return
      }

      files.push({ deletions: 0, insertions: 0, line: index, name })

      return
    }

    const current = files.at(-1)

    if (!current) {
      return
    }

    if (line.kind === 'add') {
      current.insertions++
    } else if (line.kind === 'del') {
      current.deletions++
    }
  })

  return files
}

/** Index of the file containing row `row`, for "where am I" highlighting. */
export function fileAtRow(files: readonly DiffFileEntry[], row: number): number {
  let found = 0

  for (let i = 0; i < files.length; i++) {
    if (files[i]!.line <= row) {
      found = i
    } else {
      break
    }
  }

  return found
}

/**
 * Selection index that follows the top visible row: the file whose section
 * contains `row`. Returns `current` unchanged when that file did not move,
 * so a scroll-position poll can call this every tick and skip the state
 * update — and its re-render — entirely until the viewport actually crosses
 * into another file's section.
 */
export function fileIndexFollowingRow(files: readonly DiffFileEntry[], row: number, current: number): number {
  const next = fileAtRow(files, row)

  return next === current ? current : next
}
