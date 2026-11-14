// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The rich parts of agent markdown: syntax-highlighted code, math and
 * Mermaid diagrams. Like markdown.tsx, nothing here ever hands a library's
 * HTML to the DOM:
 * - code: lowlight's token tree becomes React spans;
 * - math: KaTeX's MathML (trust off, so \href and friends stay inert text)
 *   is tokenized here and rebuilt through an element and attribute allowlist;
 * - Mermaid: loaded only when a diagram appears, rendered with
 *   securityLevel 'strict' and SVG labels, and shown as an <img> of a data
 *   URL, where no script or event handler in the SVG can ever run.
 */

import katex from 'katex'
import type { ReactElement, ReactNode } from 'react'
import { createElement, useEffect, useState } from 'react'

import { classOf, HIGHLIGHT_LIMIT, lowlight, type HastNode } from './codeHighlight.js'
import { CopyButton } from './CopyButton.js'

// ── Code ──────────────────────────────────────────────────────────────────

function hastToReact(nodes: readonly HastNode[], prefix: string): ReactNode[] {
  return nodes.map((node, index) => {
    if (node.type === 'text') return node.value ?? ''
    if (node.type !== 'element') return null
    const key = `${prefix}${index}`
    return createElement('span', { key, className: classOf(node.properties?.className) }, ...hastToReact(node.children ?? [], `${key}-`))
  })
}

/** Highlighted code as React nodes, or the plain text for an unknown language. */
export function highlightCode(code: string, language: string): ReactNode {
  const lang = language.toLowerCase()
  if (!lang || code.length > HIGHLIGHT_LIMIT || !lowlight.registered(lang)) return code
  try {
    return hastToReact(lowlight.highlight(lang, code).children, 'h')
  } catch {
    return code
  }
}

// ── Math ──────────────────────────────────────────────────────────────────

const MATH_TAGS = new Set([
  'math', 'semantics', 'mrow', 'mi', 'mn', 'mo', 'ms', 'mtext', 'mspace', 'mstyle', 'mpadded', 'mphantom', 'menclose',
  'msup', 'msub', 'msubsup', 'mfrac', 'msqrt', 'mroot', 'mover', 'munder', 'munderover', 'mtable', 'mtr', 'mtd', 'mlabeledtr',
])
const MATH_ATTRS = new Set([
  'mathvariant', 'stretchy', 'fence', 'separator', 'lspace', 'rspace', 'linethickness', 'displaystyle', 'scriptlevel',
  'accent', 'accentunder', 'columnalign', 'rowalign', 'rowspacing', 'columnspacing', 'columnlines', 'rowlines', 'width',
  'height', 'depth', 'voffset', 'notation', 'minsize', 'maxsize', 'symmetric', 'largeop', 'movablelimits', 'mathcolor',
  'display', 'xmlns',
])

interface MathNode {
  readonly tag: string
  readonly attrs: Record<string, string>
  readonly children: Array<MathNode | string>
}

const ENTITY: Record<string, string> = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'" }
function decodeEntities(text: string): string {
  return text.replace(/&(#x[0-9a-f]+|#\d+|amp|lt|gt|quot|apos);/gi, (raw, name: string) => {
    if (name[0] === '#') {
      const code = name[1] === 'x' || name[1] === 'X' ? Number.parseInt(name.slice(2), 16) : Number.parseInt(name.slice(1), 10)
      return Number.isFinite(code) && code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : ''
    }
    return ENTITY[name.toLowerCase()] ?? raw
  })
}

/** Tokenize KaTeX's well-formed MathML. Anything unexpected is dropped, never passed on. */
export function parseMathMarkup(markup: string): MathNode {
  const root: MathNode = { tag: '#root', attrs: {}, children: [] }
  const stack: MathNode[] = [root]
  const pattern = /<(\/?)([a-zA-Z][\w-]*)((?:\s+[\w:-]+="[^"]*")*)\s*(\/?)>|([^<]+)/g
  for (const match of markup.matchAll(pattern)) {
    const top = stack[stack.length - 1]!
    if (match[5] !== undefined) { top.children.push(decodeEntities(match[5])); continue }
    const tag = match[2]!.toLowerCase()
    if (match[1]) {
      if (stack.length > 1 && top.tag === tag) stack.pop()
      continue
    }
    const attrs: Record<string, string> = {}
    for (const attr of (match[3] ?? '').matchAll(/([\w:-]+)="([^"]*)"/g)) attrs[attr[1]!.toLowerCase()] = decodeEntities(attr[2]!)
    const node: MathNode = { tag, attrs, children: [] }
    top.children.push(node)
    if (!match[4]) stack.push(node)
  }
  return root
}

function mathToReact(node: MathNode | string, key: string): ReactNode {
  if (typeof node === 'string') return node
  // The TeX source annotation is for copy tools, not for display.
  if (node.tag === 'annotation' || node.tag === 'annotation-xml') return null
  const children = node.children.map((child, index) => mathToReact(child, `${key}-${index}`))
  if (!MATH_TAGS.has(node.tag)) return children
  const props: Record<string, string> = { key }
  for (const [name, value] of Object.entries(node.attrs)) if (MATH_ATTRS.has(name)) props[name] = value
  return createElement(node.tag, props, ...children)
}

/** TeX rendered as MathML React elements; malformed TeX shows as its source. */
export function MathView({ tex, display }: { tex: string; display: boolean }): ReactElement {
  let rendered: ReactNode = null
  try {
    const markup = katex.renderToString(tex, { output: 'mathml', displayMode: display, throwOnError: false, trust: false, strict: 'ignore', maxExpand: 500, maxSize: 50 })
    rendered = mathToReact(parseMathMarkup(markup), 'm')
  } catch {
    rendered = null
  }
  if (!display) return <span className="md__math" title={tex}>{rendered ?? <code>{tex}</code>}</span>
  return (
    <figure className="md__mathblock">
      <div className="md__mathbody">{rendered ?? <code>{tex}</code>}</div>
      <CopyButton text={tex} label="Copy TeX" />
    </figure>
  )
}

// ── Mermaid ───────────────────────────────────────────────────────────────

type MermaidApi = (typeof import('mermaid'))['default']
let mermaidLoad: Promise<MermaidApi> | undefined
let mermaidSequence = 0
/** A streaming diagram changes with every token; render only once it settles. */
const MERMAID_SETTLE_MS = 300
const MERMAID_SOURCE_LIMIT = 50_000

async function renderMermaid(source: string): Promise<string> {
  if (source.length > MERMAID_SOURCE_LIMIT) throw new Error('This diagram is too large to draw here.')
  const mermaid = await (mermaidLoad ??= import('mermaid').then(module => module.default))
  const light = typeof document !== 'undefined' && document.documentElement.dataset.theme === 'light'
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    htmlLabels: false,
    suppressErrorRendering: true,
    theme: light ? 'neutral' : 'dark',
    fontFamily: 'ui-sans-serif, -apple-system, BlinkMacSystemFont, sans-serif',
  })
  mermaidSequence += 1
  const { svg } = await mermaid.render(`xerxes-mermaid-${mermaidSequence}`, source)
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

type MermaidState = { readonly kind: 'pending' } | { readonly kind: 'ready'; readonly url: string } | { readonly kind: 'error'; readonly message: string }

function currentTheme(): string {
  return typeof document === 'undefined' ? '' : document.documentElement.dataset.theme ?? ''
}

/** The page's data-theme, updated when Settings switches it. */
function useDocumentTheme(): string {
  const [theme, setTheme] = useState(currentTheme)
  useEffect(() => {
    if (typeof document === 'undefined' || typeof MutationObserver === 'undefined') return
    const observer = new MutationObserver(() => setTheme(currentTheme()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => observer.disconnect()
  }, [])
  return theme
}

/** A Mermaid fence: the drawn diagram, its source behind a toggle, and Copy. */
export function MermaidBlock({ source }: { source: string }): ReactElement {
  const [state, setState] = useState<MermaidState>({ kind: 'pending' })
  const [showSource, setShowSource] = useState(false)
  // The diagram's colours come from the theme at draw time, so redraw on a switch.
  const theme = useDocumentTheme()
  useEffect(() => {
    let live = true
    const timer = setTimeout(() => {
      renderMermaid(source).then(
        url => { if (live) setState({ kind: 'ready', url }) },
        (error: unknown) => { if (live) setState({ kind: 'error', message: error instanceof Error ? error.message.split('\n')[0] ?? 'Could not draw it' : 'Could not draw it' }) },
      )
    }, MERMAID_SETTLE_MS)
    return () => { live = false; clearTimeout(timer) }
  }, [source, theme])
  const sourceView = <pre className="md__code" data-lang="mermaid"><code>{source}</code></pre>
  return (
    <figure className="md__codewrap md__mermaid" data-state={state.kind}>
      <figcaption className="md__codebar">
        <span className="md__lang">Diagram</span>
        <span className="md__codeactions">
          <button type="button" className="copybtn" aria-pressed={showSource} onClick={() => setShowSource(value => !value)}>{showSource ? 'Diagram' : 'Source'}</button>
          <CopyButton text={source} label="Copy diagram source" />
        </span>
      </figcaption>
      {showSource || state.kind !== 'ready'
        ? <>
            {state.kind === 'error' && !showSource && <p className="md__mermaid-note" role="status">Could not draw this diagram: {state.message}</p>}
            {state.kind === 'pending' && !showSource && <p className="md__mermaid-note" role="status">Drawing the diagram…</p>}
            {sourceView}
          </>
        : <div className="md__mermaid-view"><img src={state.url} alt="Diagram" draggable={false} /></div>}
    </figure>
  )
}
