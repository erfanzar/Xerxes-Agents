// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Minimal markdown renderer for agent responses, plan bodies, and questions.
 *
 * The renderer is sandboxed-renderer friendly: it produces React elements
 * only — no HTML parsing, no dangerouslySetInnerHTML — so agent-authored
 * markdown can never inject markup into the desktop shell. Links render as
 * anchors that never navigate (the shell has no external opener; a click
 * would point the whole window at a URL), and fenced code renders as a
 * bordered pre block in the shell's flat mono style.
 */

import type { ReactElement, ReactNode } from 'react'
import { Fragment } from 'react'

const INLINE_RE = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\s][^*]*\*)|(\[[^\]]+\]\([^)\s]*\))/g

/** Never-navigating link: the shell cannot open external targets, so a
 * click is a no-op; the URL stays visible as the title tooltip. */
function InlineLink({ label, href }: { label: string; href: string }): ReactElement {
  return (
    <a
      className="md__link"
      href={href}
      title={href}
      onClick={event => {
        event.preventDefault()
        event.stopPropagation()
      }}
    >
      {label}
    </a>
  )
}

/** Split one line into code / bold / italic / link / plain spans. */
function inlinePieces(line: string): ReactElement[] {
  const pieces: ReactElement[] = []
  let last = 0
  let key = 0
  for (const match of line.matchAll(INLINE_RE)) {
    const start = match.index ?? 0
    if (start > last) pieces.push(<Fragment key={key++}>{line.slice(last, start)}</Fragment>)
    const [raw, code, bold, italic, link] = match
    if (code) pieces.push(<code key={key++}>{code.slice(1, -1)}</code>)
    else if (bold) pieces.push(<strong key={key++}>{bold.slice(2, -2)}</strong>)
    else if (italic) pieces.push(<em key={key++}>{italic.slice(1, -1)}</em>)
    else if (link) {
      const parsed = /^\[([^\]]+)\]\(([^)\s]*)\)$/.exec(link)
      if (parsed) pieces.push(<InlineLink key={key++} label={parsed[1] ?? ''} href={parsed[2] ?? ''} />)
      else pieces.push(<Fragment key={key++}>{raw}</Fragment>)
    }
    last = start + raw.length
  }
  if (last < line.length) pieces.push(<Fragment key={key++}>{line.slice(last)}</Fragment>)
  return pieces
}

const FENCE_RE = /^\s*(```+|~~~+)\s*(\S*)?\s*$/
const TABLE_DIVIDER_RE = /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/
const HR_RE = /^\s*(-{3,}|\*{3,}|_{3,})\s*$/

interface TableRow {
  readonly cells: readonly string[]
  readonly header: boolean
}

function splitRow(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '')
  return trimmed.split('|').map(cell => cell.trim())
}

/** Fold markdown text into display blocks (headings / lists / code / tables / paragraphs). */
export function markdownBlocks(markdown: string, prefix = 'md'): ReactElement[] {
  const lines = markdown.split('\n')
  const out: ReactElement[] = []
  let paragraph: string[] = []
  let list: Array<{ text: string; checked: boolean | null }> = []
  let key = 0

  const flushParagraph = (): void => {
    if (paragraph.length) {
      out.push(<p key={`${prefix}-p${key++}`}>{inlinePieces(paragraph.join(' '))}</p>)
      paragraph = []
    }
  }
  const flushList = (): void => {
    if (list.length) {
      out.push(
        <ul key={`${prefix}-l${key++}`}>
          {list.map((item, index) => (
            <li key={index} className={item.checked === null ? undefined : 'md__check'}>
              {item.checked === null ? null : <span className={`md__box${item.checked ? ' is-done' : ''}`}>{item.checked ? '✓' : ''}</span>}
              {inlinePieces(item.text)}
            </li>
          ))}
        </ul>,
      )
      list = []
    }
  }
  const flushAll = (): void => {
    flushParagraph()
    flushList()
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? ''
    const fence = FENCE_RE.exec(line)
    if (fence) {
      flushAll()
      // Unclosed fence at EOF (streaming cut) still renders as code.
      const body: string[] = []
      let j = i + 1
      for (; j < lines.length; j++) {
        if (FENCE_RE.test(lines[j] ?? '')) break
        body.push(lines[j] ?? '')
      }
      out.push(
        <pre key={`${prefix}-c${key++}`} className="md__code" data-lang={fence[2] || undefined}>
          <code>{body.join('\n')}</code>
        </pre>,
      )
      i = j
      continue
    }

    // Table: a pipe row followed by a divider row. Cells carry inline spans.
    if (line.trimStart().startsWith('|') && i + 1 < lines.length && TABLE_DIVIDER_RE.test(lines[i + 1] ?? '')) {
      flushAll()
      const rows: TableRow[] = [{ cells: splitRow(line), header: true }]
      let j = i + 2
      for (; j < lines.length; j++) {
        const row = lines[j] ?? ''
        if (!row.trimStart().startsWith('|')) break
        rows.push({ cells: splitRow(row), header: false })
      }
      const head = rows[0]?.cells ?? []
      out.push(
        <table key={`${prefix}-t${key++}`} className="md__table">
          <thead>
            <tr>{head.map((cell, ci) => <th key={ci}>{inlinePieces(cell)}</th>)}</tr>
          </thead>
          <tbody>
            {rows.slice(1).map((row, ri) => (
              <tr key={ri}>{row.cells.map((cell, ci) => <td key={ci}>{inlinePieces(cell)}</td>)}</tr>
            ))}
          </tbody>
        </table>,
      )
      i = j - 1
      continue
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line)
    const check = /^\s*[-*]\s+\[([ xX])]\s*(.*)$/.exec(line)
    const bullet = /^\s*[-*]\s+(.*)$/.exec(line)
    const ordered = /^\s*(\d+)[.)]\s+(.*)$/.exec(line)
    const quote = /^\s*>\s?(.*)$/.exec(line)
    if (heading) {
      flushAll()
      const level = heading[1]!.length
      const Tag = (level <= 1 ? 'h1' : level === 2 ? 'h2' : 'h3') as 'h1' | 'h2' | 'h3'
      out.push(<Tag key={`${prefix}-h${key++}`}>{inlinePieces(heading[2]!)}</Tag>)
    } else if (HR_RE.test(line)) {
      flushAll()
      out.push(<hr key={`${prefix}-r${key++}`} />)
    } else if (quote) {
      flushAll()
      const quoted: string[] = [quote[1] ?? '']
      let j = i + 1
      for (; j < lines.length; j++) {
        const more = /^\s*>\s?(.*)$/.exec(lines[j] ?? '')
        if (!more) break
        quoted.push(more[1] ?? '')
      }
      out.push(
        <blockquote key={`${prefix}-q${key++}`}>
          <p>{inlinePieces(quoted.join(' '))}</p>
        </blockquote>,
      )
      i = j - 1
      continue
    } else if (check) {
      flushParagraph()
      list.push({ text: check[2]!, checked: check[1]!.toLowerCase() === 'x' })
    } else if (bullet) {
      flushParagraph()
      list.push({ text: bullet[1]!, checked: null })
    } else if (ordered) {
      flushParagraph()
      list.push({ text: ordered[2]!, checked: null })
    } else if (!line.trim()) {
      flushAll()
    } else {
      flushList()
      paragraph.push(line.trim())
    }
  }
  flushAll()
  return out
}

/** Render markdown inside a `.md` container. */
export function Markdown({ text, className, tail }: { text: string; className?: string; tail?: ReactNode }): ReactElement {
  const cls = className ? `md ${className}` : 'md'
  if (!tail) return <div className={cls}>{markdownBlocks(text)}</div>

  // Glue the streaming caret to the end of the text: trailing PROSE lines
  // fold into one paragraph that carries the tail inline. Structural endings
  // (fences, tables, lists, headings, quotes) can't — the tail follows them.
  const lines = text.split('\n')
  let last = lines.length - 1
  while (last >= 0 && !lines[last]!.trim()) last -= 1
  const STRUCTURAL = /^\s*(```|~~~|#{1,4}\s|[-*]\s|\d+[.)]\s|\||>\s?|(-{3,}|\*{3,}|_{3,})\s*$)/
  const plain = (line: string): boolean => line.trim() !== '' && !STRUCTURAL.test(line)
  if (last < 0 || !plain(lines[last]!)) {
    return <div className={cls}>{markdownBlocks(text)}{tail}</div>
  }
  let start = last
  while (start > 0 && plain(lines[start - 1]!)) start -= 1
  const head = lines.slice(0, start).join('\n')
  const trailing = lines.slice(start).map(line => line.trim()).join(' ')
  return (
    <div className={cls}>
      {markdownBlocks(head)}
      <p>{inlinePieces(trailing)}{tail}</p>
    </div>
  )
}
