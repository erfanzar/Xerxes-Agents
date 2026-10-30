// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Small markdown renderer for chat replies on the phone.
 *
 * It builds React elements only (no HTML parsing, no dangerouslySetInnerHTML),
 * so model-written text can never inject markup. Links open only for http(s)
 * and mailto targets; anything else renders as plain text.
 */

import type { ReactElement } from 'react'
import { Fragment, memo } from 'react'

// Emphasis needs a boundary on both sides, so globs and paths (`src/**/*.ts`, `a*b*c`) stay literal.
const INLINE_RE = /(`[^`\n]+`)|((?<![\w*/\\])\*\*(?=[^\s*])[^*\n]*[^\s*]\*\*(?![\w*]))|((?<![\w*/\\])\*(?=[^\s*])(?:[^*\n]*[^\s*])?\*(?![\w*/]))|(\[[^\]\n]+\]\([^)\s]*\))/g
// The info string may hold anything after the language (`ts title="a b"`); a backtick fence's info has no backtick.
const FENCE_OPEN = /^\s*(```+|~~~+)\s*([^\s`]*)(.*)$/
const FENCE_CLOSE = /^\s*(```+|~~~+)\s*$/

/** A fence opener, or undefined. Backtick fences may not carry a backtick in their info string. */
function fenceOpen(line: string): { readonly marker: string; readonly language: string } | undefined {
  const match = FENCE_OPEN.exec(line)
  if (!match) return undefined
  const marker = match[1]!
  if (marker.startsWith('`') && `${match[2] ?? ''}${match[3] ?? ''}`.includes('`')) return undefined
  return { marker, language: match[2] ?? '' }
}

/** Closes only with the same marker character, at least as long as the opener. */
function closesFence(line: string, marker: string): boolean {
  const match = FENCE_CLOSE.exec(line)
  return match !== null && match[1]![0] === marker[0] && match[1]!.length >= marker.length
}
const HEADING = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/
const HR = /^\s*(-{3,}|\*{3,}|_{3,})\s*$/
const LIST = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/
const QUOTE = /^\s*>\s?(.*)$/
const TABLE_DIVIDER = /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/
const CHECK = /^\[([ xX])\]\s*(.*)$/

function safeHref(href: string): string | undefined {
  return /^(https?:\/\/|mailto:)/i.test(href) ? href : undefined
}

/** Split one run of text into code / bold / italic / link / plain spans. */
export function inlinePieces(text: string): ReactElement[] {
  const pieces: ReactElement[] = []
  let last = 0
  let key = 0
  for (const match of text.matchAll(INLINE_RE)) {
    const start = match.index ?? 0
    if (start > last) pieces.push(<Fragment key={key++}>{text.slice(last, start)}</Fragment>)
    const [raw, code, bold, italic, link] = match
    if (code) pieces.push(<code key={key++}>{code.slice(1, -1)}</code>)
    else if (bold) pieces.push(<strong key={key++}>{bold.slice(2, -2)}</strong>)
    else if (italic) pieces.push(<em key={key++}>{italic.slice(1, -1)}</em>)
    else if (link) {
      const parsed = /^\[([^\]]+)\]\(([^)\s]*)\)$/.exec(link)
      const label = parsed?.[1] ?? raw
      const href = safeHref(parsed?.[2] ?? '')
      pieces.push(href === undefined
        ? <Fragment key={key++}>{label}</Fragment>
        : <a key={key++} href={href} target="_blank" rel="noopener noreferrer">{label}</a>)
    }
    last = start + raw.length
  }
  if (last < text.length) pieces.push(<Fragment key={key++}>{text.slice(last)}</Fragment>)
  return pieces
}

interface ListLine {
  readonly indent: number
  readonly ordered: boolean
  readonly start: number
  text: string
}

interface ListNode {
  readonly line: ListLine
  readonly children: ListNode[]
}

function nest(lines: readonly ListLine[]): ListNode[] {
  const roots: ListNode[] = []
  const stack: ListNode[] = []
  for (const line of lines) {
    const node: ListNode = { line, children: [] }
    while (stack.length > 0 && stack.at(-1)!.line.indent >= line.indent) stack.pop()
    const parent = stack.at(-1)
    if (parent) parent.children.push(node)
    else roots.push(node)
    stack.push(node)
  }
  return roots
}

function ListItemText({ text }: { text: string }): ReactElement {
  const check = CHECK.exec(text)
  if (!check) return <>{inlinePieces(text)}</>
  const done = check[1]!.toLowerCase() === 'x'
  return <><span className={`md-box${done ? ' is-done' : ''}`} aria-label={done ? 'done' : 'not done'}>{done ? '✓' : ''}</span>{inlinePieces(check[2] ?? '')}</>
}

function ListView({ nodes }: { nodes: readonly ListNode[] }): ReactElement {
  const first = nodes[0]
  const items = nodes.map((node, index) => (
    <li key={index}>
      <ListItemText text={node.line.text} />
      {node.children.length > 0 ? <ListView nodes={node.children} /> : null}
    </li>
  ))
  if (first?.line.ordered === true) return first.line.start === 1 ? <ol>{items}</ol> : <ol start={first.line.start}>{items}</ol>
  return <ul>{items}</ul>
}

function splitRow(line: string): string[] {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim())
}

function isTableStart(line: string, next: string | undefined): boolean {
  return line.trimStart().startsWith('|') && next !== undefined && TABLE_DIVIDER.test(next)
}

function startsBlock(line: string, next: string | undefined): boolean {
  return fenceOpen(line) !== undefined || HEADING.test(line) || HR.test(line) || LIST.test(line) || QUOTE.test(line) || isTableStart(line, next)
}

/** Fold markdown into display blocks: headings, lists, code, tables, quotes, paragraphs. */
export function markdownBlocks(markdown: string): ReactElement[] {
  const lines = markdown.replace(/\r\n?/g, '\n').split('\n')
  const out: ReactElement[] = []
  let key = 0
  let i = 0
  while (i < lines.length) {
    const line = lines[i]!
    if (line.trim() === '') { i += 1; continue }

    const fence = fenceOpen(line)
    if (fence) {
      const body: string[] = []
      let j = i + 1
      while (j < lines.length && !closesFence(lines[j]!, fence.marker)) body.push(lines[j++]!)
      const language = fence.language
      out.push(
        <pre key={key++} className="md-code" data-lang={language === '' ? undefined : language}>
          <code>{body.join('\n')}</code>
        </pre>,
      )
      i = j + 1
      continue
    }

    const heading = HEADING.exec(line)
    if (heading) {
      const text = inlinePieces(heading[2] ?? '')
      out.push(heading[1]!.length <= 2 ? <h3 key={key++}>{text}</h3> : <h4 key={key++}>{text}</h4>)
      i += 1
      continue
    }

    if (HR.test(line)) { out.push(<hr key={key++} />); i += 1; continue }

    if (isTableStart(line, lines[i + 1])) {
      const head = splitRow(line)
      const rows: string[][] = []
      let j = i + 2
      while (j < lines.length && lines[j]!.trimStart().startsWith('|')) rows.push(splitRow(lines[j++]!))
      out.push(
        <div key={key++} className="md-table" role="region" aria-label="Table" tabIndex={0}>
          <table>
            <thead><tr>{head.map((cell, c) => <th key={c}>{inlinePieces(cell)}</th>)}</tr></thead>
            <tbody>{rows.map((row, r) => <tr key={r}>{row.map((cell, c) => <td key={c}>{inlinePieces(cell)}</td>)}</tr>)}</tbody>
          </table>
        </div>,
      )
      i = j
      continue
    }

    if (QUOTE.test(line)) {
      const quoted: string[] = []
      let j = i
      for (; j < lines.length; j++) {
        const more = QUOTE.exec(lines[j]!)
        if (!more) break
        quoted.push(more[1] ?? '')
      }
      out.push(<blockquote key={key++}>{markdownBlocks(quoted.join('\n'))}</blockquote>)
      i = j
      continue
    }

    if (LIST.test(line)) {
      const items: ListLine[] = []
      let j = i
      while (j < lines.length) {
        const current = lines[j]!
        const item = LIST.exec(current)
        if (item && !HR.test(current)) {
          const marker = item[2]!
          const ordered = /\d/.test(marker)
          items.push({ indent: item[1]!.replace(/\t/g, '    ').length, ordered, start: ordered ? Number.parseInt(marker, 10) : 1, text: item[3] ?? '' })
          j += 1
          continue
        }
        const previous = items.at(-1)
        if (previous && /^\s+\S/.test(current) && fenceOpen(current) === undefined) {
          previous.text = `${previous.text} ${current.trim()}`
          j += 1
          continue
        }
        // A blank line between items keeps one list.
        if (current.trim() === '' && j + 1 < lines.length && LIST.test(lines[j + 1]!) && !HR.test(lines[j + 1]!)) { j += 1; continue }
        break
      }
      out.push(<ListView key={key++} nodes={nest(items)} />)
      i = j
      continue
    }

    const paragraph: string[] = [line.trim()]
    let j = i + 1
    while (j < lines.length && lines[j]!.trim() !== '' && !startsBlock(lines[j]!, lines[j + 1])) paragraph.push(lines[j++]!.trim())
    out.push(
      <p key={key++}>
        {paragraph.map((text, index) => <Fragment key={index}>{index > 0 ? <br /> : null}{inlinePieces(text)}</Fragment>)}
      </p>,
    )
    i = j
  }
  return out
}

/** A chat reply rendered as markdown inside a `.md` container. */
export const Markdown = memo(function Markdown({ text, className }: { text: string; className?: string }): ReactElement {
  return <div className={className === undefined ? 'md' : `md ${className}`}>{markdownBlocks(text)}</div>
})
