// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Pure markdown → block tree parser. This is the LOGIC half of the renderer:
// no React, no terminal — just `string → Block[]` and `string → Inline[]`, so
// it is fully unit-testable. The OpenTUI view turns these into native nodes.
// Supports the subset that matters for assistant
// output: headings, lists, blockquotes, fenced code (with diff coloring),
// horizontal rules, tables, and inline code/bold/italic/links.

export type Block =
  | { type: 'heading'; level: number; text: string }
  | { type: 'paragraph'; text: string }
  | { type: 'code'; lang: string; lines: string[] }
  | { type: 'list'; ordered: boolean; items: string[] }
  | { type: 'quote'; lines: string[] }
  | { type: 'hr' }
  | { type: 'table'; header: string[]; rows: string[][] }

export type Inline =
  | { type: 'text'; text: string }
  | { type: 'code'; text: string }
  | { type: 'bold'; text: string }
  | { type: 'italic'; text: string }
  | { type: 'link'; text: string; url: string }

const FENCE_RE = /^(```+|~~~+)\s*([\w+-]*)\s*$/
const HEADING_RE = /^(#{1,6})\s+(.*)$/
const HR_RE = /^(?:\s*[-*_]){3,}\s*$/
const UL_RE = /^\s*[-*+]\s+(.*)$/
const OL_RE = /^\s*\d+[.)]\s+(.*)$/
const QUOTE_RE = /^\s*>\s?(.*)$/

function isTableSeparator(line: string): boolean {
  return /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$/.test(line)
}

function splitTableRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map(c => c.trim())
}

/** Parse markdown text into a flat list of blocks. */
export function parseMarkdown(input: string): Block[] {
  const lines = input.replace(/\r\n?/g, '\n').split('\n')
  const blocks: Block[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]!

    // Fenced code block.
    const fence = FENCE_RE.exec(line)
    if (fence) {
      const close = fence[1]!
      const lang = fence[2] ?? ''
      const body: string[] = []
      i += 1
      while (i < lines.length && !lines[i]!.startsWith(close[0]!.repeat(3))) {
        body.push(lines[i]!)
        i += 1
      }
      i += 1 // consume the closing fence (or EOF)
      blocks.push({ type: 'code', lang, lines: body })
      continue
    }

    // Blank line.
    if (line.trim() === '') {
      i += 1
      continue
    }

    // Horizontal rule.
    if (HR_RE.test(line)) {
      blocks.push({ type: 'hr' })
      i += 1
      continue
    }

    // Heading.
    const heading = HEADING_RE.exec(line)
    if (heading) {
      blocks.push({ type: 'heading', level: heading[1]!.length, text: heading[2]!.trim() })
      i += 1
      continue
    }

    // Table: a header row followed by a separator row.
    if (line.includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1]!)) {
      const header = splitTableRow(line)
      const rows: string[][] = []
      i += 2
      while (i < lines.length && lines[i]!.includes('|') && lines[i]!.trim() !== '') {
        rows.push(splitTableRow(lines[i]!))
        i += 1
      }
      blocks.push({ type: 'table', header, rows })
      continue
    }

    // Blockquote (consecutive `>` lines).
    if (QUOTE_RE.test(line)) {
      const quote: string[] = []
      while (i < lines.length && QUOTE_RE.test(lines[i]!)) {
        quote.push(QUOTE_RE.exec(lines[i]!)![1]!)
        i += 1
      }
      blocks.push({ type: 'quote', lines: quote })
      continue
    }

    // List (a run of consecutive list items of the same ordering).
    if (UL_RE.test(line) || OL_RE.test(line)) {
      const ordered = OL_RE.test(line) && !UL_RE.test(line)
      const re = ordered ? OL_RE : UL_RE
      const items: string[] = []
      while (i < lines.length && re.test(lines[i]!)) {
        items.push(re.exec(lines[i]!)![1]!)
        i += 1
      }
      blocks.push({ type: 'list', ordered, items })
      continue
    }

    // Paragraph (consecutive non-blank, non-structural lines).
    const para: string[] = []
    while (
      i < lines.length &&
      lines[i]!.trim() !== '' &&
      !FENCE_RE.test(lines[i]!) &&
      !HEADING_RE.test(lines[i]!) &&
      !HR_RE.test(lines[i]!) &&
      !QUOTE_RE.test(lines[i]!) &&
      !UL_RE.test(lines[i]!) &&
      !OL_RE.test(lines[i]!)
    ) {
      para.push(lines[i]!)
      i += 1
    }
    blocks.push({ type: 'paragraph', text: para.join('\n') })
  }

  return blocks
}

const INLINE_RE = /(`[^`]+`)|(\*\*[^*]+\*\*|__[^_]+__)|(\*[^*]+\*|_[^_]+_)|(\[[^\]]+\]\([^)]+\))/

/** Parse inline markdown (code, bold, italic, links) into spans. */
export function parseInline(input: string): Inline[] {
  const out: Inline[] = []
  let rest = input

  while (rest.length > 0) {
    const m = INLINE_RE.exec(rest)
    if (!m || m.index === undefined) {
      out.push({ type: 'text', text: rest })
      break
    }
    if (m.index > 0) {
      out.push({ type: 'text', text: rest.slice(0, m.index) })
    }
    const tok = m[0]
    if (m[1]) {
      out.push({ type: 'code', text: tok.slice(1, -1) })
    } else if (m[2]) {
      out.push({ type: 'bold', text: tok.slice(2, -2) })
    } else if (m[3]) {
      out.push({ type: 'italic', text: tok.slice(1, -1) })
    } else if (m[4]) {
      const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(tok)!
      out.push({ type: 'link', text: link[1]!, url: link[2]! })
    }
    rest = rest.slice(m.index + tok.length)
  }

  // Merge adjacent plain-text spans for tidiness.
  return out.filter(s => !(s.type === 'text' && s.text === ''))
}

/** Classify a code line for diff coloring: 'add' | 'del' | 'meta' | null. */
export function diffLineKind(line: string): 'add' | 'del' | 'meta' | null {
  if (/^\+(?!\+\+)/.test(line)) {
    return 'add'
  }
  if (/^-(?!--)/.test(line)) {
    return 'del'
  }
  if (/^(@@|diff |index |\+\+\+|---)/.test(line)) {
    return 'meta'
  }
  return null
}
