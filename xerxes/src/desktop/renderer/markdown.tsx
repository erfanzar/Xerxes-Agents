// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Minimal markdown renderer for agent responses, plan bodies, and questions.
 *
 * The renderer is sandboxed-renderer friendly: it produces React elements
 * only — no HTML parsing, no dangerouslySetInnerHTML — so agent-authored
 * markdown can never inject markup into the desktop shell. Links never
 * navigate the window: http, https and mailto open in the system browser
 * through the host, file paths open in the Files panel, anything else is
 * inert text (chatLinks.tsx). Images load as data URLs through the host's
 * validated read (ImageView.tsx). Fenced code renders as a bordered pre
 * block in the shell's flat mono style.
 */

import type { ReactElement, ReactNode } from 'react'
import { Fragment, memo, useContext, useEffect, useRef, useState } from 'react'

import { ChatLink, SWARM_BOARD_SENTENCE_RE, autolinked, isSwarmBoardCommand } from './chatLinks.js'
import { CodeRunContext, commandOf, isRunnableBlock } from './codeRun.js'
import { CopyButton } from './CopyButton.js'
import { Icon } from './Icon.js'
import { ChatImage } from './ImageView.js'
import { MathView, MermaidBlock, highlightCode } from './richMarkdown.js'
import { ArtifactWidgetEmbed, WidgetFrame } from './WidgetFrame.js'

/** Run beside Copy: types the command into an idle shell of the Terminal tab, or a new one. */
function RunButton({ code }: { code: string }): ReactElement | null {
  const run = useContext(CodeRunContext)
  const [state, setState] = useState<'idle' | 'running' | 'sent' | 'failed'>('idle')
  const [message, setMessage] = useState('')
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])
  if (!run) return null
  return (
    <button
      type="button"
      className="copybtn runbtn"
      disabled={state === 'running'}
      aria-label={state === 'sent' ? 'Sent to the terminal' : 'Run in terminal'}
      title={state === 'failed' ? message || 'Could not run it' : 'Run in an idle Terminal shell (a new one if none is idle)'}
      onClick={event => {
        event.stopPropagation()
        event.preventDefault()
        setState('running')
        clearTimeout(timer.current)
        run(commandOf(code)).then(
          () => setState('sent'),
          failure => { setMessage(failure instanceof Error ? failure.message : String(failure)); setState('failed') },
        ).finally(() => { timer.current = setTimeout(() => setState('idle'), 1800) })
      }}
    >
      {state === 'running' ? 'Running…' : state === 'sent' ? 'Sent' : state === 'failed' ? 'Failed' : 'Run'}
    </button>
  )
}

const CodeBlock = memo(function CodeBlock({ code, language }: { code: string; language: string }): ReactElement {
  return (
    <figure className="md__codewrap">
      <figcaption className="md__codebar">
        <span className="md__lang">{language || 'text'}</span>
        <span className="md__codeactions">
          {isRunnableBlock(language, code) && <RunButton code={code} />}
          <CopyButton text={code} label="Copy code" />
        </span>
      </figcaption>
      <pre className="md__code" data-lang={language || undefined}>
        <code>{highlightCode(code, language)}</code>
      </pre>
    </figure>
  )
})

/** Extensions of files a task reads or writes; anything else in backticks stays plain code. */
const FILE_EXTENSIONS = new Set([
  'csv', 'tsv', 'xlsx', 'xls', 'docx', 'doc', 'pptx', 'pdf', 'md', 'txt', 'json', 'jsonl', 'yaml', 'yml', 'toml', 'xml', 'html', 'htm',
  'png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'log', 'sql', 'ipynb', 'py', 'ts', 'tsx', 'js', 'mjs', 'jsx', 'css', 'sh', 'rb', 'go', 'rs',
  'java', 'c', 'h', 'cpp', 'zip',
])

/** A relative or absolute file path with a known extension: `out/summary.csv`, `notes.md`. */
export function looksLikeFilePath(text: string): boolean {
  if (!text || text.length > 260 || /\s|:\/\//.test(text)) return false
  if (!/^(?:\.{1,2}\/|\/|~\/)?[\w.@+-]+(?:\/[\w.@+-]+)*$/.test(text)) return false
  const dot = text.lastIndexOf('.')
  return dot > 0 && dot > text.lastIndexOf('/') + 1 && FILE_EXTENSIONS.has(text.slice(dot + 1).toLowerCase())
}

// swarm-board hint | code | image `![alt](src "title")` | bold | italic | link `[label](href "title")` | math `$x^2$` / `\(x\)`
// Inline math needs a non-space right after the opening `$` and right before
// the closing one, and no word character after it, so prices ("$5 and $10")
// and shell variables ("$HOME/$USER", "$PATH:$HOME") stay text.
const INLINE_RE = new RegExp(
  `(${SWARM_BOARD_SENTENCE_RE.source})|(\`[^\`]+\`)|(!\\[[^\\]]*\\]\\([^)\\s]+(?:\\s+"[^"]*")?\\))|(\\*\\*[^*]+\\*\\*)|(\\*[^*\\s][^*]*\\*)|(\\[[^\\]]+\\]\\([^)\\s]*(?:\\s+"[^"]*")?\\))|(\\$[^\\s$](?:[^$\\n]*[^\\s$])?\\$(?!\\w)|\\\\\\([^\\n]+?\\\\\\))`,
  'gi',
)
const IMAGE_RE = /^!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)$/
const LINK_RE = /^\[([^\]]+)\]\(([^)\s]*)(?:\s+"[^"]*")?\)$/

/**
 * Split one line into swarm hint / code / image / bold / italic / link / plain spans.
 * Links open through the host (chatLinks.tsx); bare URLs and `/swarm board` hints
 * in plain text become links too, never inside fenced code.
 */
function inlinePieces(line: string, bareUrls = true): ReactElement[] {
  const pieces: ReactElement[] = []
  let last = 0
  let key = 0
  const nextKey = (): number => key++
  const plain = (text: string): void => { pieces.push(<Fragment key={key++}>{bareUrls ? autolinked(text, nextKey) : text}</Fragment>) }
  for (const match of line.matchAll(INLINE_RE)) {
    const start = match.index ?? 0
    if (start > last) plain(line.slice(last, start))
    const [raw, swarmHint, code, image, bold, italic, link, math] = match
    if (swarmHint) {
      pieces.push(<ChatLink key={key++} label="Open Immortals →" href="sidecar://immortals/board" />)
    }
    else if (math) {
      const tex = math.startsWith('$') ? math.slice(1, -1) : math.slice(2, -2)
      pieces.push(<MathView key={key++} tex={tex} display={false} />)
    }
    else if (code) {
      const text = code.slice(1, -1)
      // `/swarm board` opens the Immortals Board; `out/summary.csv` opens that file.
      pieces.push(isSwarmBoardCommand(text)
        ? <ChatLink key={key++} label="Open Immortals →" href="sidecar://immortals/board" className="md__link--code" />
        : looksLikeFilePath(text)
          ? <ChatLink key={key++} label={<code>{text}</code>} href={text} className="md__link--code" />
          : <code key={key++}>{text}</code>)
    }
    else if (image) {
      const parsed = IMAGE_RE.exec(image)
      if (parsed) pieces.push(<ChatImage key={key++} alt={parsed[1] ?? ''} src={parsed[2] ?? ''} />)
      else plain(raw)
    }
    // Emphasis holds code and markdown links too (**`out/summary.csv`** is
    // bold code that opens the file); a bare URL in it stays text, as before.
    else if (bold) pieces.push(<strong key={key++}>{inlinePieces(bold.slice(2, -2), false)}</strong>)
    else if (italic) pieces.push(<em key={key++}>{inlinePieces(italic.slice(1, -1), false)}</em>)
    else if (link) {
      const parsed = LINK_RE.exec(link)
      if (parsed) pieces.push(<ChatLink key={key++} label={parsed[1] ?? ''} href={parsed[2] ?? ''} />)
      else plain(raw)
    }
    last = start + raw.length
  }
  if (last < line.length) plain(line.slice(last))
  return pieces
}

const FENCE_RE = /^\s*(```+|~~~+)\s*(\S*)(?:\s+(.*?))?\s*$/
const FENCE_CLOSE_RE = /^\s*(```+|~~~+)\s*$/
const TABLE_DIVIDER_RE = /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/
const HR_RE = /^\s*(-{3,}|\*{3,}|_{3,})\s*$/
const XERXES_EMBED_RE = /^<xerxes-embed\s+([^>]+?)(?:\s*\/>|>\s*<\/xerxes-embed>)$/i
const AGENT_EMBED_RE = /^<agent-embed\s+([^>]+?)(?:\s*\/>|>\s*<\/agent-embed>)$/i
const ARTIFACT_ID_RE = /^[A-Za-z0-9_-]{2,64}$/
const HTML_TAG_HINT_RE = /<\s*(?:!doctype|html|head|body|div|canvas|svg|script|style|section|main|header|article|p|span|button|input)\b/i

function extractAttr(attrs: string, name: string): string | undefined {
  const re = new RegExp(`(?:^|\\s)${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`, 'i')
  const match = re.exec(attrs)
  const value = match?.[1] ?? match?.[2]
  return value !== undefined ? value.trim() : undefined
}

/** Parse a standalone `<xerxes-embed>` or `<agent-embed src="artifact://...">` line. */
export function parseEmbedLine(line: string): { readonly artifactId: string; readonly title?: string } | null {
  const trimmed = line.trim()
  const xerxes = XERXES_EMBED_RE.exec(trimmed)
  if (xerxes) {
    const attrs = xerxes[1] ?? ''
    const artifactId = extractAttr(attrs, 'artifact')
    if (!artifactId || !ARTIFACT_ID_RE.test(artifactId)) return null
    const title = extractAttr(attrs, 'title')
    return title ? { artifactId, title } : { artifactId }
  }
  const agent = AGENT_EMBED_RE.exec(trimmed)
  if (agent) {
    const attrs = agent[1] ?? ''
    const src = extractAttr(attrs, 'src')
    if (!src || !/^artifact:\/\//i.test(src)) return null
    const artifactId = src.replace(/^artifact:\/\//i, '').trim()
    if (!ARTIFACT_ID_RE.test(artifactId)) return null
    const title = extractAttr(attrs, 'title')
    return title ? { artifactId, title } : { artifactId }
  }
  return null
}

function parseFenceTitle(info: string): string | undefined {
  const trimmed = info.trim()
  if (!trimmed) return undefined
  const quoted = extractAttr(` ${trimmed}`, 'title')
  if (quoted) return quoted
  return trimmed
}

function resolveInlineWidgetHtml(code: string, lang: string): string {
  if (lang === '3d' && !HTML_TAG_HINT_RE.test(code)) {
    return [
      '<div class="x-canvas-wrap">',
      '  <canvas id="scene" style="width:100%;height:320px;display:block"></canvas>',
      '</div>',
      '<script>',
      '(function () {',
      "  const canvas = document.getElementById('scene');",
      code,
      '})();',
      '</script>',
    ].join('\n')
  }
  return code
}

interface TableRow {
  readonly cells: readonly string[]
  readonly header: boolean
}

function splitRow(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '')
  return trimmed.split('|').map(cell => cell.trim())
}

export const MARKDOWN_CACHE_LIMIT = 256
const MARKDOWN_CACHE_MAX_CHARS = 65_536
const markdownBlockCache = new Map<string, ReactElement[]>()

/** Clear the bounded Markdown parse LRU cache (primarily for deterministic tests). */
export function clearMarkdownCache(): void {
  markdownBlockCache.clear()
}

/** Current number of finalized entries retained in the Markdown parse LRU cache. */
export function markdownCacheSize(): number {
  return markdownBlockCache.size
}

/** Fold markdown text into display blocks (headings / lists / code / tables / paragraphs). */
export function markdownBlocks(markdown: string, prefix = 'md', cacheable = true): ReactElement[] {
  const canCache = cacheable && prefix === 'md' && markdown.length > 0 && markdown.length <= MARKDOWN_CACHE_MAX_CHARS
  if (canCache) {
    const cached = markdownBlockCache.get(markdown)
    if (cached) {
      // Refresh LRU recency.
      markdownBlockCache.delete(markdown)
      markdownBlockCache.set(markdown, cached)
      return cached
    }
  }
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
              {item.checked === null ? null : <span className={`md__box${item.checked ? ' is-done' : ''}`}>{item.checked ? <Icon name="check" size={11} /> : null}</span>}
              <span className="md__item">{inlinePieces(item.text)}</span>
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
    const embed = parseEmbedLine(line)
    if (embed) {
      flushAll()
      out.push(
        <ArtifactWidgetEmbed
          key={`${prefix}-e${key++}`}
          artifactId={embed.artifactId}
          {...(embed.title ? { title: embed.title } : {})}
        />,
      )
      continue
    }

    const fence = FENCE_RE.exec(line)
    if (fence) {
      flushAll()
      // Unclosed fence at EOF (streaming cut) still renders as code.
      const openMarker = fence[1] ?? '```'
      const body: string[] = []
      let j = i + 1
      for (; j < lines.length; j++) {
        const closeMatch = FENCE_CLOSE_RE.exec(lines[j] ?? '')
        if (closeMatch && closeMatch[1]![0] === openMarker[0] && closeMatch[1]!.length >= openMarker.length) break
        body.push(lines[j] ?? '')
      }
      // The fence language was captured into data-lang and then rendered by
      // nothing, and a code block in a coding tool had no way to be copied.
      const code = body.join('\n')
      const language = fence[2] || ''
      const fenceInfo = fence[3] || ''
      const lang = language.toLowerCase()
      const closed = j < lines.length
      if (lang === 'mermaid') out.push(<MermaidBlock key={`${prefix}-c${key++}`} source={code} />)
      else if (lang === 'math' || lang === 'latex' || lang === 'tex') out.push(<MathView key={`${prefix}-c${key++}`} tex={code} display />)
      else if (closed && (lang === 'widget' || lang === 'html-widget' || lang === '3d')) {
        const widgetTitle = parseFenceTitle(fenceInfo)
        out.push(
          <WidgetFrame
            key={`${prefix}-w${key++}`}
            html={resolveInlineWidgetHtml(code, lang)}
            mode="inline"
            {...(widgetTitle ? { title: widgetTitle } : {})}
          />,
        )
      }
      else out.push(<CodeBlock key={`${prefix}-c${key++}`} code={code} language={language} />)
      i = j
      continue
    }

    // Display math: `$$` on its own line (or `$$ … $$` on one line), and `\[ … \]`.
    const mathOpen = /^\s*(\$\$|\\\[)(.*)$/.exec(line)
    if (mathOpen) {
      const close = mathOpen[1] === '$$' ? '$$' : '\\]'
      const rest = mathOpen[2] ?? ''
      const closeAt = rest.lastIndexOf(close)
      if (closeAt >= 0 && rest.slice(closeAt + close.length).trim() === '') {
        flushAll()
        out.push(<MathView key={`${prefix}-m${key++}`} tex={rest.slice(0, closeAt).trim()} display />)
        continue
      }
      let j = i + 1
      const body: string[] = rest.trim() ? [rest] : []
      for (; j < lines.length; j++) {
        const next = lines[j] ?? ''
        const at = next.indexOf(close)
        if (at >= 0) { if (next.slice(0, at).trim()) body.push(next.slice(0, at)); break }
        body.push(next)
      }
      // Without a closing marker this is prose that happens to start with $$.
      if (j < lines.length) {
        flushAll()
        out.push(<MathView key={`${prefix}-m${key++}`} tex={body.join('\n').trim()} display />)
        i = j
        continue
      }
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
        <div key={`${prefix}-t${key++}`} className="md__table-scroll" role="region" aria-label="Table contents" tabIndex={0}><table className="md__table">
          <thead>
            <tr>{head.map((cell, ci) => <th key={ci}>{inlinePieces(cell)}</th>)}</tr>
          </thead>
          <tbody>
            {rows.slice(1).map((row, ri) => (
              <tr key={ri}>{row.cells.map((cell, ci) => <td key={ci}>{inlinePieces(cell)}</td>)}</tr>
            ))}
          </tbody>
        </table></div>,
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
  if (canCache) {
    if (markdownBlockCache.size >= MARKDOWN_CACHE_LIMIT) {
      const oldest = markdownBlockCache.keys().next().value
      if (oldest !== undefined) markdownBlockCache.delete(oldest)
    }
    markdownBlockCache.set(markdown, out)
  }
  return out
}

/** Render markdown inside a `.md` container. */
export const Markdown = memo(function Markdown({ text, className, tail }: { text: string; className?: string; tail?: ReactNode }): ReactElement {
  const cls = className ? `md ${className}` : 'md'
  if (!tail) return <div className={cls}>{markdownBlocks(text, 'md', true)}</div>

  // Glue the streaming caret to the end of the text: trailing PROSE lines
  // fold into one paragraph that carries the tail inline. Structural endings
  // (fences, tables, lists, headings, quotes) can't — the tail follows them.
  const lines = text.split('\n')
  let last = lines.length - 1
  while (last >= 0 && !lines[last]!.trim()) last -= 1
  const STRUCTURAL = /^\s*(```|~~~|#{1,4}\s|[-*]\s|\d+[.)]\s|\||>\s?|(-{3,}|\*{3,}|_{3,})\s*$|<xerxes-embed\b|<agent-embed\b)/i
  const plain = (line: string): boolean => line.trim() !== '' && !STRUCTURAL.test(line)
  if (last < 0 || !plain(lines[last]!)) {
    return <div className={cls}>{markdownBlocks(text, 'md', false)}{tail}</div>
  }
  let start = last
  while (start > 0 && plain(lines[start - 1]!)) start -= 1
  const head = lines.slice(0, start).join('\n')
  const trailing = lines.slice(start).map(line => line.trim()).join(' ')
  return (
    <div className={cls}>
      {markdownBlocks(head, 'md', true)}
      <p>{inlinePieces(trailing)}{tail}</p>
    </div>
  )
})
