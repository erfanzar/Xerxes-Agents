// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Pure incremental-markdown chunker. Re-parsing the whole assistant buffer on
// every stream delta is O(total) per delta — wasteful on long replies. This
// splits the text at the last STABLE block boundary (a blank line that is not
// inside a fenced code block) into a frozen prefix + an in-flight suffix, so
// only the suffix re-parses each delta. Logic ported from Xerxes' streaming
// chunker (MIT); math-fence handling dropped (our parser has no TeX).

/**
 * Is a code fence still open at byte offset `end`? Splitting the prefix while a
 * fence is open would orphan the opening ``` and let the suffix render as
 * broken markdown, so the boundary search must avoid these positions.
 */
export function fenceOpenAt(s: string, end: number): boolean {
  let codeOpen = false
  let i = 0
  while (i < end) {
    const nl = s.indexOf('\n', i)
    const lineEnd = nl < 0 || nl > end ? end : nl
    const line = s.slice(i, lineEnd).trim()
    if (/^(?:`{3,}|~{3,})/.test(line)) {
      codeOpen = !codeOpen
    }
    if (nl < 0 || nl >= end) {
      break
    }
    i = nl + 1
  }
  return codeOpen
}

/**
 * Index just after the last `\n\n` before EOF that sits OUTSIDE a fenced code
 * block (i.e. the start of the next block), or -1 if no safe boundary exists.
 */
export function findStableBoundary(text: string): number {
  let idx = text.length
  while (idx > 0) {
    const boundary = text.lastIndexOf('\n\n', idx - 1)
    if (boundary < 0) {
      return -1
    }
    const splitAt = boundary + 2
    if (!fenceOpenAt(text, splitAt)) {
      return splitAt
    }
    idx = boundary
  }
  return -1
}

export interface StreamSplit {
  /** Frozen, fully-formed blocks — safe to memoize on this exact string. */
  stablePrefix: string
  /** The in-flight tail that still re-parses every delta. */
  unstableSuffix: string
}

/**
 * Split `text` into a monotonically-growing stable prefix and an unstable
 * suffix. `prevPrefix` is the prefix from the previous delta; the boundary only
 * ever advances (never retreats), keeping the prefix's memo key stable across
 * deltas. Pure — the React component just stashes `prevPrefix` in a ref.
 */
export function splitStreaming(text: string, prevPrefix: string): StreamSplit {
  // Defensive: if the text no longer starts with our recorded prefix (a new
  // turn reused the component), reset.
  const base = text.startsWith(prevPrefix) ? prevPrefix : ''
  const boundary = findStableBoundary(text)
  const stablePrefix = boundary > base.length ? text.slice(0, boundary) : base
  return { stablePrefix, unstableSuffix: text.slice(stablePrefix.length) }
}

// ── Render-ready chunk state ────────────────────────────────────────────

export interface StreamingChunks {
  /**
   * Frozen stable blocks, oldest first, each stripped of its trailing blank
   * line. A chunk's string never changes once appended, so a memoized
   * markdown component keyed on it never re-parses.
   *
   * The trailing `\n\n` is stripped because the renderer interleaves its own
   * one-row spacer between chunk elements (reproducing the blank line a
   * single <markdown> document inserts between top-level blocks) — and
   * OpenTUI preserves a trailing blank line after a heading block, which
   * would otherwise double that spacing.
   */
  chunks: readonly string[]
  /** Raw stable prefix backing `chunks`; the `splitStreaming` memo key. */
  prefix: string
}

export const STREAMING_CHUNKS_EMPTY: StreamingChunks = { chunks: [], prefix: '' }

export interface StreamingRender {
  /** Memoizable stable chunks. */
  chunks: readonly string[]
  /** Unstable tail that still re-parses every delta ('' when nothing is in flight). */
  tail: string
  /** State to feed back in on the next delta. */
  state: StreamingChunks
}

/**
 * Advance the chunk split for one stream delta. Chunks only ever grow, and
 * existing chunk strings are never mutated; a text that no longer starts
 * with the previous prefix (a new turn reusing the component) resets the
 * state instead of corrupting the chunk list.
 */
export function splitStreamingRender(text: string, prev: StreamingChunks): StreamingRender {
  const reset = !text.startsWith(prev.prefix)
  const base = reset ? STREAMING_CHUNKS_EMPTY : prev
  const { stablePrefix, unstableSuffix } = splitStreaming(text, base.prefix)

  if (stablePrefix === base.prefix) {
    return { chunks: base.chunks, tail: unstableSuffix, state: base }
  }

  const fresh = stablePrefix.slice(base.prefix.length)
  const block = fresh.endsWith('\n\n') ? fresh.slice(0, -2) : fresh
  const chunks = [...base.chunks, block]

  return { chunks, tail: unstableSuffix, state: { chunks, prefix: stablePrefix } }
}
