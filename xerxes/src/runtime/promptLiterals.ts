// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Blank out the spans of a prompt where a word is being *shown* rather than
 * *said*, so keyword triggers cannot fire on quoted or fenced text.
 *
 * Every keyword surface in the runtime matches `\bword\b` against the raw
 * prompt, which cannot tell an instruction from a mention. Asking to "fix the
 * parser in ultrathink.ts", pasting a log line containing `think harder`, or
 * quoting documentation into a code fence all escalate the turn's thinking
 * budget — silently, and at real cost in tokens and latency.
 *
 * Masking preserves length and line structure so a caller can keep matching
 * with its existing patterns and still report honest offsets into the original
 * text; only the characters inside literal spans become spaces.
 */

/** Fence markers that open a block span running to the matching closing fence. */
const FENCE = /^[ \t]*(`{3,}|~{3,})/

/**
 * Replace literal spans with spaces, preserving every index and newline.
 *
 * Handled: fenced code blocks (``` and ~~~), inline code spans, and single- or
 * double-quoted runs. An unterminated span masks to the end of its line
 * (quotes, inline code) or the end of the text (fences) — the conservative
 * direction, because a keyword after an unclosed quote is more likely part of
 * the pasted content than an instruction.
 */
export function maskPromptLiterals(text: string): string {
  const out = [...text]
  const lines = splitWithOffsets(text)
  let fence: string | undefined
  for (const { start, value } of lines) {
    const marker = FENCE.exec(value)?.[1]
    if (fence !== undefined) {
      blank(out, start, start + value.length)
      // A closing fence must be at least as long as the one that opened it.
      if (marker && marker[0] === fence[0] && marker.length >= fence.length) fence = undefined
      continue
    }
    if (marker) {
      fence = marker
      blank(out, start, start + value.length)
      continue
    }
    maskInlineSpans(out, start, value)
  }
  return out.join('')
}

/** True when `pattern` matches somewhere the text is actually speaking. */
export function matchesOutsideLiterals(text: string, pattern: RegExp): boolean {
  return pattern.test(maskPromptLiterals(text))
}

function maskInlineSpans(out: string[], base: number, line: string): void {
  let index = 0
  while (index < line.length) {
    const character = line[index]
    if (character === '\\') {
      index += 2
      continue
    }
    if (character !== '`' && character !== '"' && character !== "'") {
      index += 1
      continue
    }
    // An apostrophe inside a word ("don't", "user's") opens nothing; requiring a
    // non-word character before the quote keeps ordinary prose unmasked.
    if (character === "'" && index > 0 && /\w/.test(line[index - 1] ?? '')) {
      index += 1
      continue
    }
    const close = findClose(line, index, character)
    const end = close === -1 ? line.length : close
    blank(out, base + index + 1, base + end)
    index = close === -1 ? line.length : close + 1
  }
}

function findClose(line: string, open: number, delimiter: string): number {
  for (let index = open + 1; index < line.length; index += 1) {
    if (line[index] === '\\') {
      index += 1
      continue
    }
    if (line[index] === delimiter) return index
  }
  return -1
}

function blank(out: string[], start: number, end: number): void {
  for (let index = start; index < end && index < out.length; index += 1) {
    if (out[index] !== '\n') out[index] = ' '
  }
}

function splitWithOffsets(text: string): { readonly start: number; readonly value: string }[] {
  const lines: { start: number; value: string }[] = []
  let start = 0
  for (const value of text.split('\n')) {
    lines.push({ start, value })
    start += value.length + 1
  }
  return lines
}
