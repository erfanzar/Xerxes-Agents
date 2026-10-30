// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Line diff for the phone's edit rows. Pure and bounded: the common prefix and
 * suffix are trimmed first, and a middle too large for an LCS table is shown as
 * "all removed, then all added" rather than freezing the phone.
 */

export type DiffOp = { readonly kind: 'same' | 'add' | 'del'; readonly text: string }
export type DiffRow = DiffOp | { readonly kind: 'gap'; readonly count: number }

/** Upper bound on LCS cells (rows × columns) computed on the phone. */
export const DIFF_CELL_LIMIT = 250_000

function lines(text: string): string[] {
  return text === '' ? [] : text.replace(/\r\n?/g, '\n').split('\n')
}

export function diffLines(before: string, after: string): DiffOp[] {
  const a = lines(before)
  const b = lines(after)
  let start = 0
  while (start < a.length && start < b.length && a[start] === b[start]) start += 1
  let endA = a.length
  let endB = b.length
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) { endA -= 1; endB -= 1 }

  const ops: DiffOp[] = a.slice(0, start).map(text => ({ kind: 'same', text }))
  const midA = a.slice(start, endA)
  const midB = b.slice(start, endB)
  const n = midA.length
  const m = midB.length
  if (n * m > DIFF_CELL_LIMIT) {
    for (const text of midA) ops.push({ kind: 'del', text })
    for (const text of midB) ops.push({ kind: 'add', text })
  } else {
    // table[i * (m + 1) + j] = LCS length of midA[i:] and midB[j:], so the walk goes forward.
    const width = m + 1
    const table = new Uint32Array((n + 1) * width)
    for (let i = n - 1; i >= 0; i -= 1) {
      for (let j = m - 1; j >= 0; j -= 1) {
        table[i * width + j] = midA[i] === midB[j]
          ? table[(i + 1) * width + j + 1]! + 1
          : Math.max(table[(i + 1) * width + j]!, table[i * width + j + 1]!)
      }
    }
    let i = 0
    let j = 0
    while (i < n && j < m) {
      if (midA[i] === midB[j]) { ops.push({ kind: 'same', text: midA[i]! }); i += 1; j += 1 }
      else if (table[(i + 1) * width + j]! >= table[i * width + j + 1]!) { ops.push({ kind: 'del', text: midA[i]! }); i += 1 }
      else { ops.push({ kind: 'add', text: midB[j]! }); j += 1 }
    }
    for (; i < n; i += 1) ops.push({ kind: 'del', text: midA[i]! })
    for (; j < m; j += 1) ops.push({ kind: 'add', text: midB[j]! })
  }
  for (const text of a.slice(endA)) ops.push({ kind: 'same', text })
  return ops
}

/** Unified rows: changes with `context` unchanged lines around them; longer unchanged runs become a gap. */
export function diffRows(ops: readonly DiffOp[], context = 3): DiffRow[] {
  const changed = ops.map(op => op.kind !== 'same')
  if (!changed.includes(true)) return []
  const distance = new Array<number>(ops.length).fill(Infinity)
  let last = -Infinity
  for (let i = 0; i < ops.length; i += 1) { if (changed[i]) last = i; distance[i] = i - last }
  last = Infinity
  for (let i = ops.length - 1; i >= 0; i -= 1) { if (changed[i]) last = i; distance[i] = Math.min(distance[i]!, last - i) }
  const rows: DiffRow[] = []
  let gap = 0
  for (let i = 0; i < ops.length; i += 1) {
    if (distance[i]! <= context) {
      if (gap > 0) { rows.push({ kind: 'gap', count: gap }); gap = 0 }
      rows.push(ops[i]!)
    } else gap += 1
  }
  if (gap > 0) rows.push({ kind: 'gap', count: gap })
  return rows
}

export function diffCounts(ops: readonly DiffOp[]): { readonly added: number; readonly removed: number } {
  let added = 0
  let removed = 0
  for (const op of ops) { if (op.kind === 'add') added += 1; else if (op.kind === 'del') removed += 1 }
  return { added, removed }
}
