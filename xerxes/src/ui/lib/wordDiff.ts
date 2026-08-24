// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Word-level intra-line highlighting for the F7 diff viewer.
//
// Mockup 07's decision note: "File index answers 'what changed'; word
// highlights answer 'what exactly changed in this line'". A hunk rewrite
// arrives as a run of del rows immediately followed by a run of add rows —
// the same lines rewritten. Pairing those runs line by line and trimming
// the common prefix/suffix off each pair yields the exact character range
// that changed on both sides. Two linear scans per paired row, no edit
// matrix: cheap enough to recompute per refresh, deterministic enough to
// unit-test as pure math.

import type { DiffLine } from './gitDiff.js'

/**
 * Half-open character range `[start, end)` within one row's code text —
 * the content after its +/- marker, matching what the renderer draws.
 */
export interface WordRange {
  readonly end: number
  readonly start: number
}

/** How strongly the word span tints its background over the add-row color. */
export const WORD_TINT_ADD = 0.3
/** Del spans sit slightly lighter than adds, per mockup 07 (.26 vs .3). */
export const WORD_TINT_DEL = 0.26

const HEX_COLOR = /^#([0-9a-f]{6})$/i
const FUNC_COLOR = /^rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$/i

const parseRgbColor = (color: string): [number, number, number] | null => {
  const hex = HEX_COLOR.exec(color.trim())

  if (hex) {
    const packed = Number.parseInt(hex[1]!, 16)
    return [(packed >> 16) & 0xff, (packed >> 8) & 0xff, packed & 0xff]
  }

  const fn = FUNC_COLOR.exec(color.trim())

  if (fn) {
    return [Number(fn[1]), Number(fn[2]), Number(fn[3])]
  }

  return null
}

/**
 * Blend two CSS colors (`#rrggbb` or `rgb(r, g, b)`) — `ratio` 0 returns
 * `base`, 1 returns `overlay`, values outside clamp. An unparseable operand
 * returns `base` unchanged: theme roles may legitimately be formats this
 * cannot mix (e.g. an ansi256() override), and degrading to the plain row
 * tint beats emitting an invalid escape sequence.
 */
export function blendColors(base: string, overlay: string, ratio: number): string {
  const from = parseRgbColor(base)
  const to = parseRgbColor(overlay)

  if (!from || !to || !Number.isFinite(ratio)) {
    return base
  }

  const t = Math.min(1, Math.max(0, ratio))
  const channel = (i: 0 | 1 | 2): string =>
    Math.round(from[i]! + (to[i]! - from[i]!) * t)
      .toString(16)
      .padStart(2, '0')

  return '#' + channel(0) + channel(1) + channel(2)
}

/**
 * Trim the shared prefix and suffix off one del/add pair. Returns null when
 * the code texts are identical — nothing to highlight.
 */
const changedRange = (
  removed: string,
  added: string
): { added: WordRange; removed: WordRange } | null => {
  const maxPrefix = Math.min(removed.length, added.length)

  let prefix = 0

  while (prefix < maxPrefix && removed[prefix] === added[prefix]) {
    prefix += 1
  }

  // The suffix scan must stop where the prefix claimed characters, or an
  // overlap could produce inverted ranges on short pairs.
  const maxSuffix = maxPrefix - prefix

  let suffix = 0

  while (
    suffix < maxSuffix &&
    removed[removed.length - 1 - suffix] === added[added.length - 1 - suffix]
  ) {
    suffix += 1
  }

  const range = (text: string): WordRange => ({
    end: text.length - suffix,
    start: prefix
  })

  const removedRange = range(removed)
  const addedRange = range(added)
  const empty = (r: WordRange): boolean => r.end <= r.start

  if (empty(removedRange) && empty(addedRange)) {
    return null
  }

  return { added: addedRange, removed: removedRange }
}

/**
 * Map from row index in `lines` to the changed character range of that
 * row's code text. Only del/add rows inside a paired rewrite run appear;
 * context rows, headers, and surplus lines past the shorter of the two
 * runs never do.
 */
export function intraLineWordRanges(lines: readonly DiffLine[]): ReadonlyMap<number, WordRange> {
  const ranges = new Map<number, WordRange>()

  // Indices of the del-run currently being collected and the add-run after
  // it. A pairing window covers exactly one del-run plus the add-run that
  // follows it; any other row kind closes the window.
  let delRun: number[] = []
  let addRun: number[] = []

  const flushPairing = (): void => {
    const pairs = Math.min(delRun.length, addRun.length)

    for (let i = 0; i < pairs; i += 1) {
      const delIndex = delRun[i]!
      const addIndex = addRun[i]!
      const change = changedRange(lines[delIndex]!.text.slice(1), lines[addIndex]!.text.slice(1))

      if (!change) {
        continue
      }

      ranges.set(delIndex, change.removed)
      ranges.set(addIndex, change.added)
    }

    delRun = []
    addRun = []
  }

  lines.forEach((line, index) => {
    if (line.kind === 'del') {
      if (addRun.length > 0) {
        flushPairing()
      }

      delRun.push(index)
    } else if (line.kind === 'add') {
      // Pure insertions have no old side to diff against.
      if (delRun.length > 0) {
        addRun.push(index)
      }
    } else {
      flushPairing()
    }
  })

  flushPairing()

  return ranges
}
