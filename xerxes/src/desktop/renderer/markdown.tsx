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

// swarm-board hint | code | image | bold+italic | bold | italic | strikethrough | link | math
// Inline math needs a non-space right after the opening `$` and right before
// the closing one, and no word character after it, so prices ("$5 and $10")
// and shell variables ("$HOME/$USER", "$PATH:$HOME") stay text.
// Underscore emphasis requires non-word boundaries so snake_case identifiers stay literal.
const INLINE_RE = new RegExp(
  `(${SWARM_BOARD_SENTENCE_RE.source})`
  + `|(\`[^\`]+\`)`
  + `|(!\\[[^\\]]*\\]\\([^)\\s]+(?:\\s+"[^"]*")?\\))`
  + `|(\\*\\*\\*[^*\\n]+\\*\\*\\*)`
  + `|(\\*\\*[^*]+\\*\\*|(?<![\\w])__[^_\\s](?:[^_\\n]*[^_\\s])?__(?![\\w]))`
  + `|(\\*[^*\\s][^*]*\\*|(?<![\\w])_[^_\\s](?:[^_\\n]*[^_\\s])?_(?![\\w]))`
  + `|(~~[^~\\s](?:[^~\\n]*[^~\\s])?~~)`
  + `|(\\[[^\\]]+\\]\\([^)\\s]*(?:\\s+"[^"]*")?\\))`
  + `|(\\$[^\\s$](?:[^$\\n]*[^\\s$])?\\$(?!\\w)|\\\\\\([^\\n]+?\\\\\\))`,
  'gi',
)
const IMAGE_RE = /^!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)$/
const LINK_RE = /^\[([^\]]+)\]\(([^)\s]*)(?:\s+"[^"]*")?\)$/
const LINK_LABEL_FORMAT_RE = /[`*_~$\\]/

/**
 * Split one line into swarm hint / code / image / bold / italic / strikethrough / link / plain spans.
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
    const [raw, swarmHint, code, image, boldItalic, bold, italic, strike, link, math] = match
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
    else if (boldItalic) pieces.push(<strong key={key++}><em>{inlinePieces(boldItalic.slice(3, -3), false)}</em></strong>)
    else if (bold) pieces.push(<strong key={key++}>{inlinePieces(bold.slice(2, -2), false)}</strong>)
    else if (italic) pieces.push(<em key={key++}>{inlinePieces(italic.slice(1, -1), false)}</em>)
    else if (strike) pieces.push(<del key={key++}>{inlinePieces(strike.slice(2, -2), false)}</del>)
    else if (link) {
      const parsed = LINK_RE.exec(link)
      if (parsed) {
        const rawLabel = parsed[1] ?? ''
        const label: ReactNode = LINK_LABEL_FORMAT_RE.test(rawLabel) ? inlinePieces(rawLabel, false) : rawLabel
        pieces.push(<ChatLink key={key++} label={label} href={parsed[2] ?? ''} />)
      }
      else plain(raw)
    }
    last = start + raw.length
  }
  if (last < line.length) plain(line.slice(last))
  return pieces
}

type GfmAlertKind = 'note' | 'tip' | 'important' | 'warning' | 'caution'
const GFM_ALERT_RE = /^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\](?:\s+(.*))?$/i
const GFM_ALERTS: Readonly<Record<GfmAlertKind, { readonly label: string; readonly icon: 'info' | 'spark' | 'warning' | 'error' }>> = {
  note: { label: 'Note', icon: 'info' },
  tip: { label: 'Tip', icon: 'spark' },
  important: { label: 'Important', icon: 'info' },
  warning: { label: 'Warning', icon: 'warning' },
  caution: { label: 'Caution', icon: 'error' },
}
const MAX_QUOTE_DEPTH = 6

function parseTableAlignments(divider: string): Array<'left' | 'center' | 'right' | undefined> {
  return splitRow(divider).map(cell => {
    const trimmed = cell.trim()
    if (/^:-+:$/.test(trimmed)) return 'center'
    if (/^-+:$/.test(trimmed)) return 'right'
    if (/^:-+$/.test(trimmed)) return 'left'
    return undefined
  })
}

function isTableHeaderPair(line: string, next: string | undefined): boolean {
  if (!next || !line.includes('|') || !next.includes('|')) return false
  return TABLE_DIVIDER_RE.test(next)
}

interface DesktopListLine {
  readonly indent: number
  readonly ordered: boolean
  readonly start: number
  readonly checked: boolean | null
  text: string
}

interface DesktopListNode {
  readonly line: DesktopListLine
  readonly children: DesktopListNode[]
}

function nestDesktopList(lines: readonly DesktopListLine[]): DesktopListNode[] {
  const roots: DesktopListNode[] = []
  const stack: DesktopListNode[] = []
  for (const line of lines) {
    const node: DesktopListNode = { line, children: [] }
    while (stack.length > 0 && stack.at(-1)!.line.indent >= line.indent) stack.pop()
    const parent = stack.at(-1)
    if (parent) parent.children.push(node)
    else roots.push(node)
    stack.push(node)
  }
  return roots
}

function renderDesktopListNodes(nodes: readonly DesktopListNode[], keyPrefix: string): ReactElement {
  const first = nodes[0]
  const items = nodes.map((node, index) => (
    <li key={index} className={node.line.checked === null ? undefined : 'md__check'}>
      {node.line.checked === null ? null : (
        <span className={`md__box${node.line.checked ? ' is-done' : ''}`}>
          {node.line.checked ? <Icon name="check" size={11} /> : null}
        </span>
      )}
      <span className="md__item">
        {inlinePieces(node.line.text)}
        {node.children.length > 0 ? renderDesktopListNodes(node.children, `${keyPrefix}-${index}`) : null}
      </span>
    </li>
  ))
  if (first?.line.ordered) {
    return (
      <ol key={keyPrefix} start={first.line.start !== 1 ? first.line.start : undefined}>
        {items}
      </ol>
    )
  }
  return <ul key={keyPrefix}>{items}</ul>
}

const FENCE_RE = /^\s*(```+|~~~+)\s*(\S*)(?:\s+(.*?))?\s*$/
const FENCE_CLOSE_RE = /^\s*(```+|~~~+)\s*$/
const TABLE_DIVIDER_RE = /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/
const HR_RE = /^\s*(-{3,}|\*{3,}|_{3,})\s*$/
const XERXES_EMBED_RE = /^<xerxes-embed\s+([^>]+?)(?:\s*\/>|>\s*<\/xerxes-embed>)$/i
const AGENT_EMBED_RE = /^<agent-embed\s+([^>]+?)(?:\s*\/>|>\s*<\/agent-embed>)$/i
const ARTIFACT_ID_RE = /^[A-Za-z0-9_-]{2,64}$/
const AGY_HTML_ARTIFACT_FILE_RE = /^[A-Za-z0-9][A-Za-z0-9._ -]{0,200}\.(?:html|htm)$/i
const AGY_IMAGE_ARTIFACT_FILE_RE = /^[A-Za-z0-9][A-Za-z0-9._ -]{0,200}\.(?:png|jpe?g|gif|webp|svg)$/i
const AGY_BRAIN_FILE_PATH_RE = /\/brain\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\/([^/]+)$/i
const HTML_TAG_HINT_RE = /<\s*(?:!doctype|html|head|body|div|canvas|svg|script|style|section|main|header|article|p|span|button|input)\b/i

function extractAttr(attrs: string, name: string): string | undefined {
  const re = new RegExp(`(?:^|\\s)${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`, 'i')
  const match = re.exec(attrs)
  const value = match?.[1] ?? match?.[2]
  return value !== undefined ? value.trim() : undefined
}

function isValidEmbedArtifactId(artifactId: string): boolean {
  if (ARTIFACT_ID_RE.test(artifactId)) return true
  if (artifactId.startsWith('agy:')) {
    const filename = artifactId.slice('agy:'.length)
    return !filename.includes('..') && AGY_HTML_ARTIFACT_FILE_RE.test(filename)
  }
  return false
}

/**
 * Resolve an Antigravity `file:///.../brain/<session>/<file>.html` embed URI to
 * its read-only `agy:<file>.html` artifact id. Containment inside the active
 * session's brain directory is enforced again on the daemon by `AgyBrainBridge`.
 */
function resolveAgyFileEmbedArtifactId(src: string): string | null {
  if (!/^file:\/\//i.test(src)) return null
  const withoutQuery = src.replace(/[?#].*$/, '')
  const normalized = withoutQuery
    .replace(/^file:\/\/(?:localhost)?/i, '')
    .replaceAll('\\', '/')
  const match = AGY_BRAIN_FILE_PATH_RE.exec(normalized)
  if (!match?.[1]) return null
  let filename: string
  try {
    filename = decodeURIComponent(match[1])
  } catch {
    return null
  }
  if (
    filename.includes('/')
    || filename.includes('\\')
    || filename.includes('..')
    || !AGY_HTML_ARTIFACT_FILE_RE.test(filename)
  ) {
    return null
  }
  return `agy:${filename}`
}

/** Parse a standalone `<agent-embed src="file:///.../brain/<session>/<file>.(png|jpg|...)" />` image embed. */
function parseAgyImageEmbedLine(line: string): { readonly src: string; readonly alt: string } | null {
  const agent = AGENT_EMBED_RE.exec(line.trim())
  if (!agent) return null
  const attrs = agent[1] ?? ''
  const src = extractAttr(attrs, 'src')
  if (!src || !/^file:\/\//i.test(src)) return null
  const withoutQuery = src.replace(/[?#].*$/, '')
  const normalized = withoutQuery.replace(/^file:\/\/(?:localhost)?/i, '').replaceAll('\\', '/')
  const match = AGY_BRAIN_FILE_PATH_RE.exec(normalized)
  if (!match?.[1]) return null
  let filename: string
  try {
    filename = decodeURIComponent(match[1])
  } catch {
    return null
  }
  if (
    filename.includes('/')
    || filename.includes('\\')
    || filename.includes('..')
    || !AGY_IMAGE_ARTIFACT_FILE_RE.test(filename)
  ) {
    return null
  }
  const title = extractAttr(attrs, 'title')
  return { src, alt: title || filename }
}

/** Parse a standalone `<xerxes-embed>` or `<agent-embed src="artifact://...|file:///.../brain/.../*.html">` line. */
export function parseEmbedLine(line: string): { readonly artifactId: string; readonly title?: string } | null {
  const trimmed = line.trim()
  const xerxes = XERXES_EMBED_RE.exec(trimmed)
  if (xerxes) {
    const attrs = xerxes[1] ?? ''
    const artifactId = extractAttr(attrs, 'artifact')
    if (!artifactId || !isValidEmbedArtifactId(artifactId)) return null
    const title = extractAttr(attrs, 'title')
    return title ? { artifactId, title } : { artifactId }
  }
  const agent = AGENT_EMBED_RE.exec(trimmed)
  if (agent) {
    const attrs = agent[1] ?? ''
    const src = extractAttr(attrs, 'src')
    if (!src) return null
    const title = extractAttr(attrs, 'title')
    if (/^artifact:\/\//i.test(src)) {
      const artifactId = src.replace(/^artifact:\/\//i, '').trim()
      if (!isValidEmbedArtifactId(artifactId)) return null
      return title ? { artifactId, title } : { artifactId }
    }
    const agyArtifactId = resolveAgyFileEmbedArtifactId(src)
    if (agyArtifactId) {
      return title ? { artifactId: agyArtifactId, title } : { artifactId: agyArtifactId }
    }
    return null
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

/** Split a GFM table row without breaking on `|` inside backtick code spans or escaped `\|`. */
export function splitTableRow(line: string): string[] {
  let trimmed = line.trim()
  if (trimmed.startsWith('|')) trimmed = trimmed.slice(1)
  if (trimmed.endsWith('|')) {
    let backslashes = 0
    for (let k = trimmed.length - 2; k >= 0 && trimmed[k] === '\\'; k -= 1) backslashes += 1
    if (backslashes % 2 === 0) trimmed = trimmed.slice(0, -1)
  }
  const cells: string[] = []
  let current = ''
  let inCodeTicks = 0
  for (let i = 0; i < trimmed.length; i += 1) {
    const ch = trimmed[i]!
    if (ch === '`') {
      let ticks = 1
      while (i + ticks < trimmed.length && trimmed[i + ticks] === '`') ticks += 1
      current += trimmed.slice(i, i + ticks)
      if (inCodeTicks === 0) inCodeTicks = ticks
      else if (inCodeTicks === ticks) inCodeTicks = 0
      i += ticks - 1
      continue
    }
    if (ch === '\\' && trimmed[i + 1] === '|') {
      current += '|'
      i += 1
      continue
    }
    if (ch === '|' && inCodeTicks === 0) {
      cells.push(current.trim())
      current = ''
      continue
    }
    current += ch
  }
  cells.push(current.trim())
  return cells
}

/** True when `lines` ends inside an unclosed fenced code block or display math block (`$$` / `\[`). */
export function hasOpenStreamingBlock(lines: readonly string[]): boolean {
  let openFence: { char: string; length: number } | null = null
  let openMathClose: '$$' | '\\]' | null = null
  for (const line of lines) {
    if (openFence) {
      const closeMatch = FENCE_CLOSE_RE.exec(line)
      if (closeMatch && closeMatch[1]![0] === openFence.char && closeMatch[1]!.length >= openFence.length) {
        openFence = null
      }
      continue
    }
    if (openMathClose) {
      if (line.includes(openMathClose)) openMathClose = null
      continue
    }
    const fence = FENCE_RE.exec(line)
    if (fence) {
      openFence = { char: fence[1]![0]!, length: fence[1]!.length }
      continue
    }
    const mathOpen = /^\s*(\$\$|\\\[)(.*)$/.exec(line)
    if (mathOpen) {
      const close = mathOpen[1] === '$$' ? '$$' : '\\]'
      const rest = mathOpen[2] ?? ''
      const closeAt = rest.lastIndexOf(close)
      if (closeAt >= 0 && rest.slice(closeAt + close.length).trim() === '') continue
      if (mathOpen[1] === '\\[' || rest.trim() === '' || /^\s/.test(rest)) {
        openMathClose = close
      }
    }
  }
  return openFence !== null || openMathClose !== null
}

function splitRow(line: string): string[] {
  return splitTableRow(line)
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

/** Fold markdown text into display blocks (headings / lists / code / tables / quotes / alerts / paragraphs). */
export function markdownBlocks(markdown: string, prefix = 'md', cacheable = true, streaming = false, depth = 0): ReactElement[] {
  const canCache = cacheable && !streaming && depth === 0 && prefix === 'md' && markdown.length > 0 && markdown.length <= MARKDOWN_CACHE_MAX_CHARS
  if (canCache) {
    const cached = markdownBlockCache.get(markdown)
    if (cached) {
      // Refresh LRU recency.
      markdownBlockCache.delete(markdown)
      markdownBlockCache.set(markdown, cached)
      return cached
    }
  }
  const lines = markdown.replace(/\r\n?/g, '\n').split('\n')
  const out: ReactElement[] = []
  let paragraph: Array<{ readonly text: string; readonly hardBreak: boolean }> = []
  let list: DesktopListLine[] = []
  let key = 0

  const flushParagraph = (): void => {
    if (paragraph.length) {
      const nodes: ReactNode[] = []
      paragraph.forEach((entry, idx) => {
        if (idx > 0) {
          nodes.push(paragraph[idx - 1]!.hardBreak ? <br key={`br-${idx}`} /> : ' ')
        }
        nodes.push(<Fragment key={`seg-${idx}`}>{inlinePieces(entry.text)}</Fragment>)
      })
      out.push(<p key={`${prefix}-p${key++}`}>{nodes}</p>)
      paragraph = []
    }
  }
  const flushList = (): void => {
    if (list.length) {
      const groups: DesktopListNode[][] = []
      for (const root of nestDesktopList(list)) {
        const lastGroup = groups.at(-1)
        if (lastGroup && lastGroup[0]!.line.ordered === root.line.ordered) lastGroup.push(root)
        else groups.push([root])
      }
      for (const group of groups) {
        out.push(renderDesktopListNodes(group, `${prefix}-l${key++}`))
      }
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
    const imageEmbed = parseAgyImageEmbedLine(line)
    if (imageEmbed) {
      flushAll()
      out.push(<p key={`${prefix}-ie${key++}`}><ChatImage src={imageEmbed.src} alt={imageEmbed.alt} /></p>)
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
      // Without a closing marker, render as display math while streaming (so an
      // in-progress `$$` block never collapses into a single-line paragraph);
      // in finalized markdown, an unclosed `$$` stays prose.
      if (j < lines.length || (streaming && (mathOpen[1] === '\\[' || rest.trim() === '' || /^\s/.test(rest)))) {
        flushAll()
        out.push(<MathView key={`${prefix}-m${key++}`} tex={body.join('\n').trim()} display />)
        i = j
        continue
      }
    }

    // Table: a GFM pipe row followed by a divider row (with optional column alignment).
    if (isTableHeaderPair(line, lines[i + 1])) {
      flushAll()
      const alignments = parseTableAlignments(lines[i + 1] ?? '')
      const rows: TableRow[] = [{ cells: splitRow(line), header: true }]
      let j = i + 2
      for (; j < lines.length; j++) {
        const row = lines[j] ?? ''
        if (!row.trim() || !row.includes('|')) break
        rows.push({ cells: splitRow(row), header: false })
      }
      const head = rows[0]?.cells ?? []
      const alignStyle = (col: number): { readonly textAlign: 'left' | 'center' | 'right' } | undefined => {
        const align = alignments[col]
        return align ? { textAlign: align } : undefined
      }
      out.push(
        <div key={`${prefix}-t${key++}`} className="md__table-scroll" role="region" aria-label="Table contents" tabIndex={0}><table className="md__table">
          <thead>
            <tr>{head.map((cell, ci) => <th key={ci} style={alignStyle(ci)}>{inlinePieces(cell)}</th>)}</tr>
          </thead>
          <tbody>
            {rows.slice(1).map((row, ri) => (
              <tr key={ri}>{row.cells.map((cell, ci) => <td key={ci} style={alignStyle(ci)}>{inlinePieces(cell)}</td>)}</tr>
            ))}
          </tbody>
        </table></div>,
      )
      i = j - 1
      continue
    }

    const heading = /^\s*(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line)
    const check = /^(\s*)[-*+]\s+\[([ xX])]\s*(.*)$/.exec(line)
    const bullet = /^(\s*)[-*+]\s+(.*)$/.exec(line)
    const ordered = /^(\s*)(\d+)[.)]\s+(.*)$/.exec(line)
    const quote = /^\s*>\s?(.*)$/.exec(line)
    if (heading) {
      flushAll()
      const level = heading[1]!.length
      const Tag = (level <= 1 ? 'h1' : level === 2 ? 'h2' : 'h3') as 'h1' | 'h2' | 'h3'
      out.push(<Tag key={`${prefix}-h${key++}`}>{inlinePieces(heading[2]!.trim())}</Tag>)
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
      const firstNonEmptyIdx = quoted.findIndex(entry => entry.trim() !== '')
      const alertMatch = firstNonEmptyIdx >= 0 ? GFM_ALERT_RE.exec(quoted[firstNonEmptyIdx]!.trim()) : null
      if (alertMatch) {
        const kind = alertMatch[1]!.toLowerCase() as GfmAlertKind
        const meta = GFM_ALERTS[kind]
        const trailing = (alertMatch[2] ?? '').trim()
        const restLines = [
          ...(trailing ? [trailing] : []),
          ...quoted.slice(firstNonEmptyIdx + 1),
        ]
        const bodyMd = restLines.join('\n').trim()
        const alertKey = `${prefix}-a${key++}`
        out.push(
          <aside key={alertKey} className={`md__alert md__alert--${kind}`} role="note" data-alert={kind}>
            <p className="md__alert-title">
              <Icon name={meta.icon} size={14} />
              <span>{meta.label}</span>
            </p>
            {bodyMd ? (
              <div className="md__alert-body">
                {depth < MAX_QUOTE_DEPTH
                  ? markdownBlocks(bodyMd, `${alertKey}-b`, false, streaming, depth + 1)
                  : <p>{inlinePieces(bodyMd)}</p>}
              </div>
            ) : null}
          </aside>,
        )
      } else {
        const quoteKey = `${prefix}-q${key++}`
        const quoteBody = quoted.join('\n')
        out.push(
          <blockquote key={quoteKey}>
            {depth < MAX_QUOTE_DEPTH
              ? markdownBlocks(quoteBody, `${quoteKey}-b`, false, streaming, depth + 1)
              : <p>{inlinePieces(quoted.join(' '))}</p>}
          </blockquote>,
        )
      }
      i = j - 1
      continue
    } else if (check) {
      flushParagraph()
      list.push({
        indent: check[1]!.replace(/\t/g, '    ').length,
        ordered: false,
        start: 1,
        text: check[3]!,
        checked: check[2]!.toLowerCase() === 'x',
      })
    } else if (bullet) {
      flushParagraph()
      list.push({
        indent: bullet[1]!.replace(/\t/g, '    ').length,
        ordered: false,
        start: 1,
        text: bullet[2]!,
        checked: null,
      })
    } else if (ordered) {
      flushParagraph()
      list.push({
        indent: ordered[1]!.replace(/\t/g, '    ').length,
        ordered: true,
        start: Number.parseInt(ordered[2]!, 10) || 1,
        text: ordered[3]!,
        checked: null,
      })
    } else if (!line.trim()) {
      flushAll()
    } else if (list.length > 0 && /^\s+\S/.test(line)) {
      list[list.length - 1]!.text = `${list[list.length - 1]!.text} ${line.trim()}`
    } else {
      flushList()
      const hardBreak = /  +$/.test(line) || /\\$/.test(line.trimEnd())
      const cleaned = hardBreak ? line.trimEnd().replace(/\\$/, '').trim() : line.trim()
      paragraph.push({ text: cleaned, hardBreak })
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
  // (fences, tables, lists, headings, quotes, or an unclosed code/math block)
  // can't — the tail follows them without collapsing code or math into `<p>`.
  const lines = text.split('\n')
  if (hasOpenStreamingBlock(lines)) {
    return <div className={cls}>{markdownBlocks(text, 'md', false, true)}{tail}</div>
  }
  let last = lines.length - 1
  while (last >= 0 && !lines[last]!.trim()) last -= 1
  const STRUCTURAL = /^\s*(```|~~~|\$\$|\\\[|#{1,6}\s|[-*]\s|\d+[.)]\s|\||>\s?|(-{3,}|\*{3,}|_{3,})\s*$|<xerxes-embed\b|<agent-embed\b)/i
  const plain = (line: string): boolean => line.trim() !== '' && !STRUCTURAL.test(line)
  if (last < 0 || !plain(lines[last]!)) {
    return <div className={cls}>{markdownBlocks(text, 'md', false, true)}{tail}</div>
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
