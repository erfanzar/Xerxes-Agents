// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Sandboxed Generative UI & Interactive 3D Widget renderer.
 *
 * Every agent-authored HTML/3D widget executes inside an isolated
 * `<iframe sandbox="allow-scripts">` loaded from a revoked-on-unmount `blob:`
 * URL with a strict offline CSP (`default-src 'none'`). Semantic dark/light
 * tokens, utility classes, live theme syncing, inline auto-height reporting,
 * and `window.Xerxes3D` are injected cleanly into the document head before any
 * widget script runs.
 */

import {
  createContext,
  memo,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ReactElement,
} from 'react'
import { createPortal } from 'react-dom'

import { CopyButton } from './CopyButton.js'
import { Icon } from './Icon.js'
import { desktopCall, desktopError, record, text } from './desktopRpc.js'
import { isPerfPaused, readDocumentTheme, subscribeDocumentTheme, subscribePerfPaused } from './pageVisibility.js'
import { highlightCode } from './richMarkdown.js'
import { store } from './store.js'
import { XERXES_3D_RUNTIME_SCRIPT } from './xerxes3dRuntime.js'

/** Active daemon session key for inline `<ArtifactWidgetEmbed>` fetches. */
export const WidgetSessionContext = createContext<string>('')

/** Host callback that opens a saved artifact inside the right-hand Artifacts panel. */
export const WidgetOpenArtifactContext = createContext<((artifactId: string) => void) | null>(null)

export interface WidgetFrameProps {
  readonly html: string
  readonly title?: string
  readonly mode?: 'inline' | 'panel'
  readonly artifactId?: string
  readonly onOpenPanel?: () => void
  readonly className?: string
}

const INNER_WIDGET_CSP =
  "default-src 'none'; script-src 'unsafe-inline' data:; style-src 'unsafe-inline'; img-src data: blob:; font-src data:;"

const INLINE_MIN_HEIGHT = 120
const INLINE_MAX_HEIGHT = 520
const INLINE_DEFAULT_HEIGHT = 340
const INLINE_3D_DEFAULT_HEIGHT = 360

const THREE_D_HINT_RE = /\bXerxes3D\b|webgl|three(?:\.js)?|perspective\s*\(|rotate3d\s*\(/i
const INLINE_SCRIPT_RE = /<script(\b[^>]*)>([\s\S]*?)<\/script>/gi

function utf8ToBase64(source: string): string {
  if (typeof TextEncoder === 'function' && typeof btoa === 'function') {
    const bytes = new TextEncoder().encode(source)
    let binary = ''
    for (let offset = 0; offset < bytes.length; offset += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000))
    }
    return btoa(binary)
  }
  return Buffer.from(source, 'utf8').toString('base64')
}

/**
 * Rewrite inline `<script>` elements into `data:text/javascript;base64,...`
 * external script tags so sandboxed `blob:` iframes execute cleanly under
 * Chromium's CSP3 inheritance without permitting `'unsafe-inline'` in the host.
 */
function externalizeInlineScripts(html: string): string {
  return html.replace(INLINE_SCRIPT_RE, (full, attrs: string, body: string) => {
    if (/\bsrc\s*=/i.test(attrs)) return full
    const typeMatch = /\btype\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i.exec(attrs)
    const mime = (typeMatch?.[1] ?? typeMatch?.[2] ?? typeMatch?.[3] ?? '').trim().toLowerCase()
    if (mime && mime !== 'text/javascript' && mime !== 'application/javascript' && mime !== 'module') {
      return full
    }
    if (!body.trim()) return full
    const dataUrl = `data:text/javascript;base64,${utf8ToBase64(body)}`
    return `<script${attrs} src="${dataUrl}"></script>`
  })
}

function looksLike3dWidget(html: string): boolean {
  return THREE_D_HINT_RE.test(html)
}

function readHostTheme(): 'dark' | 'light' {
  if (typeof document === 'undefined') return 'dark'
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark'
}

function escapeJsString(value: string): string {
  return value.replaceAll('\\', '\\\\').replaceAll("'", "\\'").replaceAll('<', '\\u003c').replaceAll('>', '\\u003e')
}

function widgetTokensCss(mode: 'inline' | 'panel'): string {
  const modeRule = mode === 'panel'
    ? 'html, body { height: 100%; min-height: 100%; background: var(--background); }'
    : 'html, body { height: auto; min-height: 0; overflow-x: hidden; background: transparent; }'

  return [
    ':root {',
    '  color-scheme: dark;',
    '  --background: #0f0f13;',
    '  --content: #131318;',
    '  --card: #18181f;',
    '  --sidebar: #111116;',
    '  --card-border: rgba(255, 255, 255, 0.09);',
    '  --border: rgba(255, 255, 255, 0.12);',
    '  --foreground: #ececf1;',
    '  --muted-foreground: #9ca3af;',
    '  --placeholder: #6b7280;',
    '  --code-foreground: #e5e7eb;',
    '  --primary: #a855f7;',
    '  --primary-foreground: #ffffff;',
    '  --secondary: #23232e;',
    '  --secondary-foreground: #e5e7eb;',
    '  --accent: #fbbf24;',
    '  --link: #7aa2f7;',
    '  --success: #34d399;',
    '  --warning: #fbbf24;',
    '  --error: #f87171;',
    '  --chart-1: #a855f7;',
    '  --chart-2: #38bdf8;',
    '  --chart-3: #fbbf24;',
    '  --chart-4: #34d399;',
    '  --chart-5: #f472b6;',
    '  --chart-6: #818cf8;',
    '}',
    'html.light {',
    '  color-scheme: light;',
    '  --background: #fafaf9;',
    '  --content: #ffffff;',
    '  --card: #f4f4f5;',
    '  --sidebar: #f3f3f4;',
    '  --card-border: rgba(17, 24, 39, 0.10);',
    '  --border: rgba(17, 24, 39, 0.14);',
    '  --foreground: #18181b;',
    '  --muted-foreground: #52525b;',
    '  --placeholder: #71717a;',
    '  --code-foreground: #27272a;',
    '  --primary: #7e22ce;',
    '  --primary-foreground: #ffffff;',
    '  --secondary: #e4e4e7;',
    '  --secondary-foreground: #18181b;',
    '  --accent: #d97706;',
    '  --link: #2563eb;',
    '  --success: #059669;',
    '  --warning: #d97706;',
    '  --error: #dc2626;',
    '  --chart-1: #7e22ce;',
    '  --chart-2: #0284c7;',
    '  --chart-3: #d97706;',
    '  --chart-4: #059669;',
    '  --chart-5: #db2777;',
    '  --chart-6: #4f46e5;',
    '}',
    '*, *::before, *::after { box-sizing: border-box; }',
    'html, body {',
    '  margin: 0;',
    '  padding: 0;',
    "  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;",
    '  font-size: 13.5px;',
    '  line-height: 1.5;',
    '  color: var(--foreground);',
    '  -webkit-font-smoothing: antialiased;',
    '}',
    modeRule,
    'canvas { display: block; max-width: 100%; }',
    '.x-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }',
    '.x-col { display: flex; flex-direction: column; gap: 8px; }',
    '.x-between { justify-content: space-between; }',
    '.x-grid { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }',
    '.x-card { background: var(--card); border: 1px solid var(--card-border); border-radius: 10px; padding: 12px; }',
    '.x-btn {',
    '  display: inline-flex; align-items: center; justify-content: center; gap: 6px;',
    '  padding: 5px 10px; border-radius: 7px; border: 1px solid var(--border);',
    '  background: var(--secondary); color: var(--secondary-foreground);',
    '  font: 500 12px/1.2 inherit; cursor: pointer;',
    '  transition: background 120ms ease, border-color 120ms ease, opacity 120ms ease;',
    '}',
    '.x-btn:hover { border-color: var(--primary); }',
    '.x-btn.is-active, .x-btn--primary { background: var(--primary); border-color: var(--primary); color: var(--primary-foreground); }',
    '.x-pill {',
    '  display: inline-flex; align-items: center; gap: 5px; padding: 2px 8px;',
    '  border-radius: 999px; border: 1px solid var(--card-border); background: var(--card);',
    '  color: var(--muted-foreground); font-size: 11px; font-weight: 500;',
    '}',
    '.x-range { accent-color: var(--primary); cursor: pointer; }',
    ".x-mono { font-family: 'SF Mono', ui-monospace, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }",
    '.x-muted { color: var(--muted-foreground); font-size: 12px; }',
    '.x-canvas-wrap { position: relative; width: 100%; overflow: hidden; border-radius: 8px; }',
  ].join('\n')
}

function widgetBridgeScript(mode: 'inline' | 'panel', embedId: string): string {
  const safeId = escapeJsString(embedId)
  const reportHeight = mode === 'inline'
    ? [
        '  var lastSent = 0;',
        '  function measureHeight() {',
        '    var body = document.body;',
        '    if (!body) return 0;',
        '    var rectH = body.getBoundingClientRect ? body.getBoundingClientRect().height : 0;',
        '    var maxBottom = rectH;',
        '    var kids = body.children;',
        '    for (var i = 0; i < kids.length; i++) {',
        '      var r = kids[i].getBoundingClientRect ? kids[i].getBoundingClientRect() : null;',
        '      if (r && r.bottom > maxBottom) maxBottom = r.bottom;',
        '    }',
        '    return Math.ceil(Math.max(maxBottom, body.scrollHeight || 0));',
        '  }',
        '  function postHeight() {',
        '    var h = measureHeight();',
        '    if (!h || Math.abs(h - lastSent) < 2) return;',
        '    lastSent = h;',
        "    try { window.parent.postMessage({ type: 'xerxes-widget-height', embedId: '" + safeId + "', height: h }, '*'); } catch (_) {}",
        '  }',
        "  window.addEventListener('DOMContentLoaded', postHeight);",
        "  window.addEventListener('load', postHeight);",
        "  window.addEventListener('resize', postHeight);",
        '  setTimeout(postHeight, 40);',
        '  setTimeout(postHeight, 180);',
        "  if (typeof ResizeObserver === 'function') {",
        '    var ro = new ResizeObserver(postHeight);',
        "    window.addEventListener('DOMContentLoaded', function () { if (document.body) ro.observe(document.body); });",
        '    if (document.body) ro.observe(document.body);',
        '  }',
        "  if (typeof MutationObserver === 'function') {",
        '    var mo = new MutationObserver(postHeight);',
        "    window.addEventListener('DOMContentLoaded', function () { if (document.body) mo.observe(document.body, { childList: true, subtree: true, attributes: true }); });",
        '    if (document.body) mo.observe(document.body, { childList: true, subtree: true, attributes: true });',
        '  }',
      ].join('\n')
    : ''

  return [
    '(function () {',
    "  'use strict';",
    "  window.addEventListener('message', function (event) {",
    '    var data = event && event.data;',
    "    if (!data || typeof data !== 'object' || data.type !== 'xerxes-widget-theme') return;",
    "    var isLight = data.theme === 'light';",
    "    document.documentElement.classList.toggle('light', isLight);",
    "    document.documentElement.setAttribute('data-theme', isLight ? 'light' : 'dark');",
    "    window.dispatchEvent(new Event('themechange'));",
    '  });',
    reportHeight,
    '})();',
  ].join('\n')
}

function applyHtmlThemeAttributes(htmlOpenTag: string, theme: 'dark' | 'light'): string {
  let tag = htmlOpenTag
  if (/\bdata-theme\s*=/i.test(tag)) {
    tag = tag.replace(/\bdata-theme\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)/i, `data-theme="${theme}"`)
  } else {
    tag = tag.replace(/>$/, ` data-theme="${theme}">`)
  }
  if (theme === 'light') {
    if (/\bclass\s*=\s*"([^"]*)"/i.test(tag)) {
      tag = tag.replace(/\bclass\s*=\s*"([^"]*)"/i, (_, existing: string) =>
        /\blight\b/.test(existing) ? `class="${existing}"` : `class="${existing.trim()} light"`,
      )
    } else if (/\bclass\s*=\s*'([^']*)'/i.test(tag)) {
      tag = tag.replace(/\bclass\s*=\s*'([^']*)'/i, (_, existing: string) =>
        /\blight\b/.test(existing) ? `class='${existing}'` : `class='${existing.trim()} light'`,
      )
    } else {
      tag = tag.replace(/>$/, ' class="light">')
    }
  }
  return tag
}

/**
 * Wrap a standalone HTML document OR bare HTML/script fragment in a secure,
 * offline-capable Xerxes widget shell containing:
 *  1. Strict inner CSP (`default-src 'none'; script-src 'unsafe-inline'; ...`)
 *  2. Semantic dark/light CSS tokens & utility classes
 *  3. Theme-sync + inline auto-height bridge
 *  4. Zero-dependency `window.Xerxes3D` 3D engine
 */
export function wrapWidgetHtml(
  rawHtml: string,
  options?: {
    readonly mode?: 'inline' | 'panel'
    readonly theme?: 'dark' | 'light'
    readonly embedId?: string
  },
): string {
  const mode = options?.mode ?? 'inline'
  const theme = options?.theme ?? 'dark'
  const embedId = options?.embedId ?? 'widget'
  const trimmed = rawHtml.trim()

  const headInject = [
    `<meta charset="utf-8" />`,
    `<meta name="viewport" content="width=device-width, initial-scale=1" />`,
    `<meta http-equiv="Content-Security-Policy" content="${INNER_WIDGET_CSP}" />`,
    `<style data-xerxes-widget-tokens="1">\n${widgetTokensCss(mode)}\n</style>`,
    `<script data-xerxes-widget-bridge="1">\n${widgetBridgeScript(mode, embedId)}\n</script>`,
    `<script data-xerxes-3d="1">\n${XERXES_3D_RUNTIME_SCRIPT}\n</script>`,
  ].join('\n')

  const isFullDoc = /^\s*(?:<!doctype\s+html\b|<html\b)/i.test(trimmed)
  if (!isFullDoc) {
    const lightAttr = theme === 'light' ? ' class="light"' : ''
    return externalizeInlineScripts(
      `<!doctype html>\n<html lang="en" data-theme="${theme}"${lightAttr}>\n<head>\n${headInject}\n</head>\n<body>\n${rawHtml}\n</body>\n</html>`,
    )
  }

  let doc = rawHtml
  const htmlMatch = /<html\b[^>]*>/i.exec(doc)
  if (htmlMatch) {
    const updatedTag = applyHtmlThemeAttributes(htmlMatch[0], theme)
    doc = doc.slice(0, htmlMatch.index) + updatedTag + doc.slice(htmlMatch.index + htmlMatch[0].length)
  }

  const headMatch = /<head\b[^>]*>/i.exec(doc)
  if (headMatch) {
    const insertAt = headMatch.index + headMatch[0].length
    return externalizeInlineScripts(`${doc.slice(0, insertAt)}\n${headInject}\n${doc.slice(insertAt)}`)
  }

  const afterHtml = /<html\b[^>]*>/i.exec(doc)
  if (afterHtml) {
    const insertAt = afterHtml.index + afterHtml[0].length
    return externalizeInlineScripts(`${doc.slice(0, insertAt)}\n<head>\n${headInject}\n</head>\n${doc.slice(insertAt)}`)
  }

  const lightAttr = theme === 'light' ? ' class="light"' : ''
  return externalizeInlineScripts(
    `<!doctype html>\n<html lang="en" data-theme="${theme}"${lightAttr}>\n<head>\n${headInject}\n</head>\n<body>\n${doc}\n</body>\n</html>`,
  )
}

export const WidgetFrame = memo(function WidgetFrame(props: WidgetFrameProps): ReactElement {
  const { html, title, mode = 'inline', artifactId, onOpenPanel, className } = props
  const openArtifactFromContext = useContext(WidgetOpenArtifactContext)
  const rawUid = useId().replace(/[^a-zA-Z0-9_-]/g, '')
  const embedId = `xw-${rawUid}`
  const is3d = useMemo(() => looksLike3dWidget(html), [html])

  const [theme, setTheme] = useState<'dark' | 'light'>(() => readDocumentTheme())
  const initialThemeRef = useRef<'dark' | 'light'>(theme)
  const [reloadNonce, setReloadNonce] = useState(0)
  const [showSource, setShowSource] = useState(false)
  const [fullscreen, setFullscreen] = useState(false)
  const [inViewport, setInViewport] = useState(true)
  const [inlineHeight, setInlineHeight] = useState<number>(() =>
    is3d ? INLINE_3D_DEFAULT_HEIGHT : INLINE_DEFAULT_HEIGHT,
  )

  const rootRef = useRef<HTMLElement>(null)
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const modalIframeRef = useRef<HTMLIFrameElement>(null)
  const [blobUrl, setBlobUrl] = useState<string>('')
  const [modalBlobUrl, setModalBlobUrl] = useState<string>('')

  const openPanelHandler = useMemo(() => {
    if (onOpenPanel) return onOpenPanel
    if (artifactId && openArtifactFromContext) return () => openArtifactFromContext(artifactId)
    return null
  }, [onOpenPanel, artifactId, openArtifactFromContext])

  // Build and revoke blob URL for main iframe without reloading on theme switches.
  // Also release the blob URL immediately while inspecting HTML source.
  useEffect(() => {
    if (showSource) {
      setBlobUrl('')
      return
    }
    if (typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function' || typeof Blob === 'undefined') {
      return
    }
    const wrapped = wrapWidgetHtml(html, { mode, theme: initialThemeRef.current, embedId })
    const blob = new Blob([wrapped], { type: 'text/html;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    setBlobUrl(url)
    return () => {
      URL.revokeObjectURL(url)
    }
  }, [html, mode, embedId, reloadNonce, showSource])

  // Build and revoke blob URL for fullscreen modal view (always panel mode).
  useEffect(() => {
    if (!fullscreen) {
      setModalBlobUrl('')
      return
    }
    if (typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function' || typeof Blob === 'undefined') {
      return
    }
    const wrapped = wrapWidgetHtml(html, { mode: 'panel', theme: readHostTheme(), embedId: `${embedId}-fs` })
    const blob = new Blob([wrapped], { type: 'text/html;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    setModalBlobUrl(url)
    return () => {
      URL.revokeObjectURL(url)
    }
  }, [fullscreen, html, embedId, reloadNonce])

  const pushTheme = useCallback((frame: HTMLIFrameElement | null, nextTheme: 'dark' | 'light') => {
    try {
      frame?.contentWindow?.postMessage({ type: 'xerxes-widget-theme', theme: nextTheme }, '*')
    } catch {
      // Sandboxed frame not yet ready; onLoad will sync immediately.
    }
  }, [])

  const pushPause = useCallback((frame: HTMLIFrameElement | null, paused: boolean) => {
    try {
      frame?.contentWindow?.postMessage({ type: 'xerxes-widget-pause', paused }, '*')
    } catch {
      // Sandboxed frame not yet ready.
    }
  }, [])

  // Live theme sync from host <html data-theme="..."> via shared observer.
  useEffect(() => {
    return subscribeDocumentTheme(() => {
      const next = readDocumentTheme()
      initialThemeRef.current = next
      setTheme(next)
      pushTheme(iframeRef.current, next)
      pushTheme(modalIframeRef.current, next)
    })
  }, [pushTheme])

  // Pause 3D/animated widget loops when offscreen in a long transcript.
  useEffect(() => {
    if (mode !== 'inline' || typeof IntersectionObserver === 'undefined') return
    const node = rootRef.current
    if (!node) return
    const observer = new IntersectionObserver(
      entries => {
        const entry = entries[0]
        if (entry) setInViewport(entry.isIntersecting)
      },
      { rootMargin: '120px' },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [mode])

  // Forward host pause (hidden/occluded/unfocused) + offscreen state into live iframes.
  useEffect(() => {
    const sync = (): void => {
      const hostPaused = isPerfPaused()
      pushPause(iframeRef.current, hostPaused || (!inViewport && mode === 'inline'))
      pushPause(modalIframeRef.current, hostPaused)
    }
    sync()
    return subscribePerfPaused(sync)
  }, [inViewport, mode, pushPause, blobUrl, modalBlobUrl])

  // Receive inline auto-height messages strictly matching this frame & embedId.
  useEffect(() => {
    if (mode !== 'inline' || typeof window === 'undefined') return
    const onMessage = (event: MessageEvent): void => {
      if (iframeRef.current && event.source !== iframeRef.current.contentWindow) return
      const payload = event.data as { type?: unknown; embedId?: unknown; height?: unknown } | null
      if (!payload || typeof payload !== 'object') return
      if (payload.type !== 'xerxes-widget-height' || payload.embedId !== embedId) return
      if (typeof payload.height !== 'number' || !Number.isFinite(payload.height)) return
      setInlineHeight(Math.max(INLINE_MIN_HEIGHT, Math.min(INLINE_MAX_HEIGHT, Math.ceil(payload.height))))
    }
    window.addEventListener('message', onMessage)
    return () => window.removeEventListener('message', onMessage)
  }, [mode, embedId])

  // Escape closes fullscreen modal cleanly.
  useEffect(() => {
    if (!fullscreen || typeof window === 'undefined') return
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        setFullscreen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [fullscreen])

  const rootClasses = [
    'widget-frame',
    mode === 'panel' ? 'widget-frame--panel' : 'widget-frame--inline',
    showSource ? 'is-source' : '',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ')

  const frameTitle = title || (is3d ? 'Interactive 3D scene' : 'Interactive widget')

  return (
    <section
      ref={rootRef}
      className={rootClasses}
      data-mode={mode}
      data-3d={is3d ? 'true' : undefined}
      data-artifact-id={artifactId || undefined}
      aria-label={frameTitle}
    >
      {mode === 'inline' && (
        <header className="widget-frame__bar">
          <span className="widget-frame__badge" data-3d={is3d ? 'true' : undefined}>
            {is3d ? '3D' : 'Interactive'}
          </span>
          {title && <span className="widget-frame__title" title={title}>{title}</span>}
          <span className="widget-frame__spacer" />
          <div className="widget-frame__actions">
            <button
              type="button"
              className="widget-frame__btn"
              title="Reload widget"
              aria-label="Reload widget"
              onClick={() => setReloadNonce(n => n + 1)}
            >
              <Icon name="retry" size={12} />
              <span className="widget-frame__btn-label">Reload</span>
            </button>
            <button
              type="button"
              className={`widget-frame__btn${showSource ? ' is-active' : ''}`}
              aria-pressed={showSource}
              title={showSource ? 'Show interactive preview' : 'Inspect HTML source'}
              onClick={() => setShowSource(v => !v)}
            >
              <Icon name={showSource ? 'eye' : 'code'} size={12} />
              <span className="widget-frame__btn-label">{showSource ? 'Preview' : 'Source'}</span>
            </button>
            {openPanelHandler && (
              <button
                type="button"
                className="widget-frame__btn"
                title="Open in Artifacts panel"
                onClick={openPanelHandler}
              >
                <Icon name="half" size={12} />
                <span className="widget-frame__btn-label">Open in panel</span>
              </button>
            )}
            <button
              type="button"
              className="widget-frame__btn widget-frame__btn--icon"
              title="Fullscreen view"
              aria-label="Fullscreen view"
              onClick={() => setFullscreen(true)}
            >
              <Icon name="expand" size={12} />
            </button>
          </div>
        </header>
      )}

      <div className="widget-frame__stage">
        {mode === 'panel' && (
          <button
            type="button"
            className="widget-frame__panel-expand"
            title="Fullscreen view"
            aria-label="Fullscreen view"
            onClick={() => setFullscreen(true)}
          >
            <Icon name="expand" size={12} />
          </button>
        )}
        {showSource ? (
          <div className="widget-frame__source">
            <div className="widget-frame__source-bar">
              <span>html</span>
              <CopyButton text={html} label="Copy widget source" />
            </div>
            <pre className="widget-frame__code">
              <code>{highlightCode(html, 'html')}</code>
            </pre>
          </div>
        ) : (
          <iframe
            ref={iframeRef}
            className="widget-frame__iframe"
            title={frameTitle}
            sandbox="allow-scripts"
            src={blobUrl || undefined}
            style={mode === 'inline' ? { height: `${inlineHeight}px` } : undefined}
            onLoad={() => {
              pushTheme(iframeRef.current, theme)
              pushPause(iframeRef.current, isPerfPaused() || (!inViewport && mode === 'inline'))
            }}
          />
        )}
      </div>

      {fullscreen && typeof document !== 'undefined' && createPortal(
        <div
          className="widget-frame__modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label={`${frameTitle} — fullscreen`}
          onClick={event => {
            if (event.target === event.currentTarget) setFullscreen(false)
          }}
        >
          <div className="widget-frame__modal">
            <header className="widget-frame__bar widget-frame__modal-bar">
              <span className="widget-frame__badge" data-3d={is3d ? 'true' : undefined}>
                {is3d ? '3D' : 'Interactive'}
              </span>
              <span className="widget-frame__title" title={frameTitle}>{frameTitle}</span>
              <span className="widget-frame__spacer" />
              <div className="widget-frame__actions">
                <button
                  type="button"
                  className="widget-frame__btn"
                  title="Reload widget"
                  onClick={() => setReloadNonce(n => n + 1)}
                >
                  <Icon name="retry" size={12} />
                  <span>Reload</span>
                </button>
                <button
                  type="button"
                  className="widget-frame__btn widget-frame__btn--icon"
                  title="Close fullscreen (Esc)"
                  aria-label="Close fullscreen"
                  onClick={() => setFullscreen(false)}
                >
                  <Icon name="close" size={13} />
                </button>
              </div>
            </header>
            <iframe
              ref={modalIframeRef}
              className="widget-frame__iframe widget-frame__iframe--modal"
              title={`${frameTitle} (fullscreen)`}
              sandbox="allow-scripts"
              src={modalBlobUrl || undefined}
              onLoad={() => {
                pushTheme(modalIframeRef.current, theme)
                pushPause(modalIframeRef.current, isPerfPaused())
              }}
            />
          </div>
        </div>,
        document.body,
      )}
    </section>
  )
})

interface LoadedArtifactWidget {
  readonly title: string
  readonly kind: string
  readonly content: string
  readonly version: number
}

export function ArtifactWidgetEmbed(props: {
  readonly artifactId: string
  readonly title?: string
}): ReactElement {
  const { artifactId, title } = props
  const ctxSessionKey = useContext(WidgetSessionContext)
  const openArtifact = useContext(WidgetOpenArtifactContext)
  const sessionKey = ctxSessionKey || store.getSnapshot().sessionKey

  const [loaded, setLoaded] = useState<LoadedArtifactWidget | null>(null)
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [reloadNonce, setReloadNonce] = useState<number>(0)

  // Auto-refresh inline embed if ArtifactTool (or an ACP file write for `agy:` embeds) updates it mid-session.
  useEffect(() => {
    if (typeof window === 'undefined' || !window.xerxes?.onEvent) return
    const agyFilename = artifactId.startsWith('agy:') ? artifactId.slice('agy:'.length) : ''
    return window.xerxes.onEvent(({ type, payload }) => {
      if (type !== 'tool_result' || payload.permitted === false) return
      const name = text(payload.name)
      if (name === 'ArtifactTool') {
        const resultText = text(payload.result)
        if (resultText.includes(`id ${artifactId},`)) {
          setReloadNonce(n => n + 1)
        }
        return
      }
      if (agyFilename && (name === 'write_file' || name === 'edit_file' || name === 'FileWriteTool' || name === 'FileEditTool')) {
        const resultText = text(payload.result)
        const argsText = text(payload.arguments)
        if (resultText.includes(agyFilename) || argsText.includes(agyFilename)) {
          setReloadNonce(n => n + 1)
        }
      }
    })
  }, [artifactId])

  useEffect(() => {
    if (typeof window === 'undefined' || !window.xerxes?.call) {
      setLoading(false)
      return
    }
    let live = true
    setLoading(true)
    setError('')
    void desktopCall(window.xerxes, sessionKey, 'artifacts.read', { id: artifactId })
      .then(result => {
        if (!live) return
        const art = record(result.artifact)
        const content = text(art.content)
        const artTitle = text(art.title) || title || artifactId
        const kind = text(art.kind) || 'html'
        const version = typeof art.version === 'number' ? art.version : 1
        if (!content) throw new Error('Artifact content is empty')
        setLoaded({ title: artTitle, kind, content, version })
        setLoading(false)
      })
      .catch(reason => {
        if (!live) return
        setError(desktopError(reason))
        setLoading(false)
      })
    return () => {
      live = false
    }
  }, [artifactId, sessionKey, title, reloadNonce])

  const openPanel = useMemo(
    () => (openArtifact ? () => openArtifact(artifactId) : undefined),
    [openArtifact, artifactId],
  )

  if (loading && !loaded) {
    return (
      <div className="widget-frame widget-frame--inline widget-frame--loading" data-artifact-id={artifactId} role="status" aria-live="polite">
        <header className="widget-frame__bar">
          <span className="widget-frame__badge">Interactive</span>
          <span className="widget-frame__title">{title || `Artifact ${artifactId}`}</span>
          <span className="widget-frame__spacer" />
          {openPanel && (
            <button type="button" className="widget-frame__btn" onClick={openPanel}>
              <Icon name="half" size={12} />
              <span>Open in panel</span>
            </button>
          )}
        </header>
        <div className="widget-frame__shimmer">
          <span className="widget-frame__shimmer-line" />
          <span className="widget-frame__shimmer-line is-short" />
        </div>
      </div>
    )
  }

  if (error || !loaded) {
    return (
      <div className="widget-frame widget-frame--inline widget-frame--error" data-artifact-id={artifactId} role="alert">
        <header className="widget-frame__bar">
          <span className="widget-frame__badge">Artifact</span>
          <span className="widget-frame__title">{title || artifactId}</span>
          <span className="widget-frame__spacer" />
          <div className="widget-frame__actions">
            <button type="button" className="widget-frame__btn" onClick={() => setReloadNonce(n => n + 1)}>
              <Icon name="retry" size={12} />
              <span>Retry</span>
            </button>
            {openPanel && (
              <button type="button" className="widget-frame__btn" onClick={openPanel}>
                <Icon name="half" size={12} />
                <span>Open in panel</span>
              </button>
            )}
          </div>
        </header>
        <p className="widget-frame__error-text">{error || 'Unable to load interactive artifact.'}</p>
      </div>
    )
  }

  if (loaded.kind !== 'html') {
    return (
      <div className="widget-frame widget-frame--inline widget-frame--fallback" data-artifact-id={artifactId}>
        <header className="widget-frame__bar">
          <span className="widget-frame__badge">{loaded.kind}</span>
          <span className="widget-frame__title">{title || loaded.title}</span>
          <span className="widget-frame__spacer" />
          {openPanel && (
            <button type="button" className="widget-frame__btn" onClick={openPanel}>
              <Icon name="half" size={12} />
              <span>Open in panel</span>
            </button>
          )}
        </header>
        <p className="widget-frame__error-text">Open in the Artifacts panel to read this {loaded.kind} document.</p>
      </div>
    )
  }

  return (
    <WidgetFrame
      html={loaded.content}
      title={title || loaded.title}
      mode="inline"
      artifactId={artifactId}
      {...(openPanel ? { onOpenPanel: openPanel } : {})}
    />
  )
}
