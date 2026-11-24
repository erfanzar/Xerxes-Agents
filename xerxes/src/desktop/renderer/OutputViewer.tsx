// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useRef, useState, type ReactElement } from 'react'
import { createPortal } from 'react-dom'
import { copyToClipboard } from './CopyButton.js'
import { Icon } from './Icon.js'

/**
 * Plain text from terminal output: drop color/cursor sequences (CSI), shell
 * integration and title messages (OSC — the `]1337;…` / `]133;…` noise), and
 * resolve carriage returns the way a terminal would, so a progress bar shows
 * its final state instead of every frame. Output with no escapes is returned
 * unchanged.
 */
export function plainTerminalText(raw: string): string {
  if (!/[\x1b\r\x07\x08]/.test(raw)) return raw
  const stripped = raw
    .replace(/\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g, '')   // OSC … BEL / ST
    .replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '')                 // CSI
    .replace(/\x1b[PX^_][^\x1b]*\x1b\\/g, '')              // DCS / SOS / PM / APC
    .replace(/\x1b[@-Z\\-_]/g, '')                           // other two-byte escapes
    .replace(/\x1b/g, '')
  const lines = stripped.replace(/\r\n/g, '\n').split('\n').map(line => {
    let text = line
    while (text.includes('\x08')) text = text.replace(/[^\x08]\x08/, '').replace(/^\x08+/, '')
    const segments = text.split('\r')
    return segments.reduce((screen, segment) => segment + screen.slice(segment.length), '')
  })
  return lines.join('\n').replace(/[\x00-\x08\x0b-\x1f\x7f]/g, '')
}

/** Text stays text. Only valid JSON string/envelope encoding is decoded; terminal escapes are dropped. */
export function readableOutput(raw: string): string {
  try {
    const value: unknown = JSON.parse(raw)
    if (typeof value === 'string') return plainTerminalText(value)
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      const object = value as Record<string, unknown>
      for (const key of ['stdout', 'output']) if (typeof object[key] === 'string') return plainTerminalText(object[key] as string)
    }
  } catch { /* Plain logs and partial streaming JSON remain verbatim. */ }
  return plainTerminalText(raw)
}

/** Lines a compact (in-conversation) preview shows before "Show all". */
const PREVIEW_LINES = 14

/** Shared log surface. Polling updates never move the user's reading position. */
export function OutputViewer({ text, label = 'Output', empty = 'No output recorded yet.', compact = false }: {
  text: string; label?: string; empty?: string
  /** Inside a tool step (Claude Code's): the text alone, with copy and expand as hover icons. */
  compact?: boolean
}): ReactElement {
  const [wrap, setWrap] = useState(true)
  const [expanded, setExpanded] = useState(false)
  const [full, setFull] = useState(false)
  const [copied, setCopied] = useState('')
  const dialog = useRef<HTMLDialogElement>(null)
  const expandButton = useRef<HTMLButtonElement>(null)
  const copyTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      clearTimeout(copyTimer.current)
    }
  }, [])
  useEffect(() => {
    if (!expanded) return
    dialog.current?.showModal()
    return () => { queueMicrotask(() => expandButton.current?.focus()) }
  }, [expanded])
  const copy = (): void => {
    if (!text) return
    clearTimeout(copyTimer.current)
    void copyToClipboard(text).then(
      ok => {
        if (!mounted.current) return
        setCopied(ok ? 'Copied' : 'Copy failed')
        copyTimer.current = setTimeout(() => {
          if (mounted.current) setCopied('')
        }, 1800)
      },
      () => {
        if (!mounted.current) return
        setCopied('Copy failed')
        copyTimer.current = setTimeout(() => {
          if (mounted.current) setCopied('')
        }, 1800)
      },
    )
  }
  const controls = <>
    <button aria-pressed={wrap} onClick={() => setWrap(value => !value)}>Wrap lines</button>
    <button disabled={!text} onClick={copy}>{copied || 'Copy output'}</button>
  </>
  const displayText = compact ? text.replace(/(?:\r?\n[ \t]*)+$/, '') : text
  const body = <pre className="output-viewer__text" data-wrap={wrap} aria-label={label === 'Output' ? 'Command output' : label} tabIndex={0}>{displayText || empty}</pre>
  const fullscreenBody = compact
    ? <pre className="output-viewer__text" data-wrap={wrap} aria-label={label === 'Output' ? 'Command output' : label} tabIndex={0}>{text || empty}</pre>
    : body
  const fullscreen = expanded && createPortal(<dialog ref={dialog} className="output-dialog" aria-label={`${label} expanded`} onCancel={() => setExpanded(false)} onClose={() => setExpanded(false)}>
    <header><h2>{label}</h2>{controls}<button autoFocus onClick={() => setExpanded(false)} aria-label="Close expanded output"><Icon name="close" size={16}/></button></header>
    {fullscreenBody}
  </dialog>, document.body)
  // A preview is capped (about 14 lines) and does not scroll, so the wheel keeps moving the page.
  const lineCount = displayText ? displayText.split('\n').length : 0
  if (compact) return <section className="output-viewer output-viewer--compact" aria-label={`${label} viewer`} data-full={full || undefined}>
    {body}
    {lineCount > PREVIEW_LINES && <button className="output-viewer__more" aria-expanded={full} onClick={() => setFull(value => !value)}>{full ? 'Show less' : `Show all ${lineCount.toLocaleString()} lines`}</button>}
    <div className="output-viewer__actions">
      <button disabled={!text} onClick={copy} aria-label={copied || 'Copy output'} title={copied || 'Copy output'}><Icon name={copied === 'Copied' ? 'check' : 'copy'} size={13}/></button>
      <button ref={expandButton} aria-label={`Expand ${label.toLowerCase()}`} title={`Expand ${label.toLowerCase()}`} onClick={() => setExpanded(true)}><Icon name="expand" size={13}/></button>
    </div>
    {fullscreen}
  </section>
  return <section className="output-viewer" aria-label={`${label} viewer`}>
    <header><strong>{label}</strong><span className="output-viewer__count">{text ? `${text.split('\n').length.toLocaleString()} lines` : ''}</span><div className="output-viewer__actions">{controls}<button ref={expandButton} aria-label={`Expand ${label.toLowerCase()}`} onClick={() => setExpanded(true)}><Icon name="expand" size={14}/></button></div></header>
    {body}
    {fullscreen}
  </section>
}
