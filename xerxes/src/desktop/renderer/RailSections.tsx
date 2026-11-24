// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The right panel's Subagents and Artifacts sections, laid out as
 * a "Search …" field, grouped rows (icon, title, muted kind, age on the
 * right), and a detail view under a "Sections › X › title" breadcrumb.
 * Artifacts are titled documents the model saves with ArtifactTool; the
 * daemon serves them through `artifacts.list` / `artifacts.read` /
 * `artifacts.pin`.
 */

import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react'
import { T3Icon, type T3IconName } from './T3Icons.js'
import { XerxesThinkingMark } from './XerxesSprites.js'
import { Markdown } from './markdown.js'
import { pageIsInactive, subscribePerfPaused } from './pageVisibility.js'
import { copyToClipboard } from './CopyButton.js'
import { highlightCode } from './richMarkdown.js'
import { desktopCall, desktopError, record, text } from './desktopRpc.js'
import { createListPoll, type ListPoll } from './listPoll.js'
import { agentKindLabel, agentState } from './AgentRoster.js'
import { AgentInspector } from './AgentInspector.js'
import { artifactFromToolResultPayload } from './ArtifactCard.js'
import { ChatImage } from './ImageView.js'
import { store } from './store.js'
import type { SwarmStatusSnapshot } from './swarmState.js'
import { WidgetFrame } from './WidgetFrame.js'
import type { SessionRow } from './types.js'

export interface ArtifactRow {
  readonly id: string
  readonly title: string
  readonly kind: string
  readonly language?: string
  readonly size: number
  readonly createdAt: string
  readonly updatedAt: string
  readonly pinned: boolean
  readonly version: number
}

/** `artifacts.list` rows as the panel reads them; malformed rows are dropped, never guessed. */
export function artifactRowsOf(value: unknown): readonly ArtifactRow[] {
  if (!Array.isArray(value)) return []
  return value.flatMap(raw => {
    const row = record(raw)
    const id = text(row.id), title = text(row.title)
    if (!id || !title) return []
    const language = text(row.language)
    return [{
      id, title,
      kind: text(row.kind) || 'markdown',
      ...(language ? { language } : {}),
      size: typeof row.size === 'number' ? row.size : 0,
      createdAt: text(row.created_at),
      updatedAt: text(row.updated_at),
      pinned: row.pinned === true,
      version: typeof row.version === 'number' ? row.version : 1,
    }]
  })
}

/** "17h", "3m", "2d": a compact age. Empty for an unreadable date. */
export function compactAge(iso: string, now = Date.now()): string {
  const at = Date.parse(iso)
  if (!Number.isFinite(at)) return ''
  const s = Math.max(0, Math.round((now - at) / 1000))
  if (s < 60) return 'now'
  if (s < 3600) return `${Math.floor(s / 60)}m`
  if (s < 86400) return `${Math.floor(s / 3600)}h`
  return `${Math.floor(s / 86400)}d`
}

/** "Nov 10, 2026" for an ISO timestamp; empty when it does not parse. */
const dateLabel = (iso: string): string => {
  const at = new Date(iso)
  return Number.isFinite(at.getTime()) ? at.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : ''
}

/** A Markdown artifact's leading "# Title" is its page title; the rest is the body. */
export function documentTitle(content: string, fallback: string): { title: string; body: string } {
  const match = /^\s*#[ \t]+([^\n]+?)[ \t]*#*[ \t]*(?:\n|$)/.exec(content)
  return match ? { title: match[1]!.trim(), body: content.slice(match[0].length) } : { title: fallback, body: content }
}

/** The h1–h3 headings of a Markdown body, in order, skipping fenced code. */
export function documentOutline(markdown: string): Array<{ level: number; text: string }> {
  const out: Array<{ level: number; text: string }> = []
  let fenced = false
  for (const line of markdown.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) { fenced = !fenced; continue }
    if (fenced) continue
    const match = /^(#{1,3})[ \t]+(.+?)[ \t]*#*[ \t]*$/.exec(line)
    if (match) out.push({ level: match[1]!.length, text: match[2]!.replace(/[*_`]/g, '') })
  }
  return out
}

const sizeLabel = (bytes: number): string => bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : bytes >= 1024 ? `${Math.round(bytes / 1024)} KB` : `${bytes} B`

function kindBadgeLabel(row: ArtifactRow): string {
  if (row.kind === 'html') return 'Interactive'
  if (row.kind === 'image') return 'Image'
  if (row.kind === 'markdown') return 'Doc'
  if (row.kind === 'code') return row.language || 'Code'
  return 'Text'
}

function kindFullLabel(row: ArtifactRow): string {
  if (row.kind === 'html') return 'Interactive'
  if (row.kind === 'image') return 'Image'
  if (row.kind === 'markdown') return 'Document'
  if (row.kind === 'code') return row.language ? `Code · ${row.language}` : 'Code'
  return 'Text'
}

function artifactT3Icon(kind: string): T3IconName {
  if (kind === 'html') return 'sparkles'
  if (kind === 'image') return 'image'
  if (kind === 'code') return 'codeEditor'
  if (kind === 'markdown') return 'fileText'
  return 'layers'
}

/** Picks a crisp T3 Lucide glyph matching a subagent's role (research, verifier, planner, or default bot). */
export function subagentRoleIcon(row: SessionRow): T3IconName {
  const haystack = `${row.title} ${row.agentDetails?.baseAgent ?? ''}`.toLowerCase()
  if (/(research|explor|search|investigat|study|read|audit)/.test(haystack)) return 'compass'
  if (/(verif|test|check|review|lint|qa|guard|security)/.test(haystack)) return 'shieldCheck'
  if (/(plan|architect|design|strateg)/.test(haystack)) return 'sparkles'
  if (/(monitor|watch|stall)/.test(haystack)) return 'activity'
  if (/(swarm|coord|orchestr|immortal)/.test(haystack)) return 'network'
  return 'bot'
}

function editedAgeLabel(iso: string): string {
  const age = compactAge(iso)
  return !age || age === 'now' ? 'edited just now' : `edited ${age} ago`
}

function readingStatsLabel(markdown: string): string {
  const trimmed = markdown.trim()
  if (!trimmed) return ''
  const words = trimmed.split(/\s+/).length
  const minutes = Math.max(1, Math.round(words / 220))
  return `${words.toLocaleString('en-US')} ${words === 1 ? 'word' : 'words'} · ${minutes} min read`
}

function SectionSearch({ placeholder, value, onChange }: { placeholder: string; value: string; onChange: (value: string) => void }): ReactElement {
  return <label className="rsec__search"><T3Icon name="search" size={13} /><input aria-label={placeholder} placeholder={placeholder} value={value} onChange={event => onChange(event.target.value)} spellCheck={false} /></label>
}

export function ArtifactsSection({
  sessionKey,
  sessionId,
  online,
  onTitle,
  onBack,
  target,
  initialDocument,
  initialSource,
  initialRows,
}: {
  sessionKey: string
  sessionId?: string
  online: boolean
  onTitle?: (title: string) => void
  onBack?: () => void
  target?: { readonly id: string; readonly count: number }
  initialDocument?: { readonly row: ArtifactRow; readonly content: string }
  initialSource?: boolean
  initialRows?: readonly ArtifactRow[]
}): ReactElement {
  const [rows, setRows] = useState<readonly ArtifactRow[]>(initialRows ?? [])
  const [error, setError] = useState('')
  const [listError, setListError] = useState('')
  const [loaded, setLoaded] = useState(Boolean(initialRows))
  const [needle, setNeedle] = useState('')
  const [pickedId, setPickedId] = useState(initialDocument?.row.id ?? '')
  const [document, setDocument] = useState<{ row: ArtifactRow; content: string } | null>(initialDocument ?? null)
  const artifactId = target?.id ?? pickedId
  const open = document?.row.id === artifactId ? document : null
  const [announced, setAnnounced] = useState<{ id: string; version: number } | null>(null)
  const listedVersion = rows.find(row => row.id === artifactId)?.version
  const announcedVersion = announced?.id === artifactId ? announced.version : undefined
  const [source, setSource] = useState(initialSource ?? false)
  const [outlineOpen, setOutlineOpen] = useState(false)
  const [widgetKey, setWidgetKey] = useState(0)
  const docBody = useRef<HTMLDivElement>(null)
  const [reload, setReload] = useState(0)
  // Edit mode: a draft of the content, saved through artifacts.update.
  const [draft, setDraft] = useState<string | null>(null)
  const [pendingDiscardAction, setPendingDiscardAction] = useState<'cancel' | 'back' | null>(null)
  const [saving, setSaving] = useState(false)
  const [copied, setCopied] = useState<'idle' | 'copied' | 'failed'>('idle')
  const copiedTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  useEffect(() => () => clearTimeout(copiedTimer.current), [])
  useEffect(() => { setOutlineOpen(false); setWidgetKey(0); setPendingDiscardAction(null) }, [artifactId])
  const copy = (value: string): void => {
    void copyToClipboard(value).then(ok => {
      setCopied(ok ? 'copied' : 'failed')
      clearTimeout(copiedTimer.current)
      copiedTimer.current = setTimeout(() => setCopied('idle'), 1400)
    })
  }
  const save = (): void => {
    if (!open || draft === null || saving) return
    setSaving(true)
    void desktopCall(window.xerxes, sessionKey, 'artifacts.update', { id: open.row.id, content: draft })
      .then(result => {
        const art = record(result.artifact)
        const row = artifactRowsOf([art])[0]
        if (row?.id === open.row.id) setDocument(current => !current || current.row.version <= row.version ? { row, content: text(art.content) } : current)
        setDraft(null); setPendingDiscardAction(null); setError(''); setReload(value => value + 1)
      })
      .catch(reason => setError(desktopError(reason)))
      .finally(() => setSaving(false))
  }
  const unsupported = useRef(false)
  // A slow list call (queued behind a busy turn) is never discarded by the next poll tick.
  const listPoll = useRef<ListPoll | null>(null)
  listPoll.current ??= createListPoll(
    key => desktopCall(window.xerxes, key, 'artifacts.list'),
    result => { setRows(artifactRowsOf(result.artifacts)); setListError(''); setLoaded(true) },
    reason => {
      const message = desktopError(reason)
      // An older runtime has no artifact methods: say what to do, not the raw RPC error.
      const outdated = /unknown method/i.test(message)
      if (outdated) unsupported.current = true
      setListError(outdated ? 'outdated' : message)
      setLoaded(true)
    },
    { pauseWhenInactive: true },
  )
  useEffect(() => {
    if (!online || unsupported.current) return
    listPoll.current?.request(sessionKey, true)
  }, [sessionKey, online, reload])
  // Artifacts arrive mid-turn; a cheap list poll keeps the section current
  // while the window is active, pausing completely when hidden/occluded/unfocused.
  useEffect(() => {
    const timer = setInterval(() => {
      if (!unsupported.current && !pageIsInactive()) setReload(value => value + 1)
    }, 10_000)
    const unsubPause = subscribePerfPaused(() => {
      if (!unsupported.current && !pageIsInactive()) listPoll.current?.resume()
    })
    return () => {
      clearInterval(timer)
      unsubPause()
    }
  }, [])
  useEffect(() => { onTitle?.(open?.row.title ?? '') }, [open?.row.title])
  useEffect(() => {
    if (!online) return
    const pendingInputs = new Map<string, string>()
    return window.xerxes.onEvent(({ type, payload }) => {
      if (sessionId && typeof payload.session_id === 'string' && payload.session_id !== sessionId) return
      if (type === 'tool_call') {
        const callId = text(payload.tool_call_id) || text(payload.id)
        const rawArgs =
          typeof payload.arguments === 'string'
            ? payload.arguments
            : payload.arguments && typeof payload.arguments === 'object'
              ? JSON.stringify(payload.arguments)
              : ''
        if (callId && rawArgs) {
          if (pendingInputs.size >= 64) {
            const oldest = pendingInputs.keys().next().value
            if (oldest !== undefined) pendingInputs.delete(oldest)
          }
          pendingInputs.set(callId, rawArgs)
        }
        return
      }
      if (type !== 'tool_result') return
      const callId = text(payload.tool_call_id) || text(payload.id)
      const fallbackInput = callId ? (pendingInputs.get(callId) ?? '') : ''
      if (callId) pendingInputs.delete(callId)
      const saved = artifactFromToolResultPayload(payload, fallbackInput)
      if (!saved) return
      if (saved.id.startsWith('agy:')) setReload(value => value + 1)
      if (artifactId && saved.id === artifactId) {
        setAnnounced(previous => {
          if (saved.id.startsWith('agy:')) {
            return { id: saved.id, version: (previous?.id === artifactId ? previous.version : 0) + 1 }
          }
          return previous?.id === artifactId && previous.version >= saved.version ? previous : saved
        })
      }
    })
  }, [artifactId, online, sessionId])
  // Both notification and list revisions refresh the same detail without
  // replacing the editor draft or changing its preview/source mode.
  useEffect(() => {
    if (!online || !artifactId) return
    let live = true
    setError('')
    void desktopCall(window.xerxes, sessionKey, 'artifacts.read', { id: artifactId })
      .then(result => {
        if (!live) return
        const art = record(result.artifact)
        const row = artifactRowsOf([art])[0]
        if (row?.id !== artifactId) throw new Error('The requested artifact was not returned')
        setDocument(current => current?.row.id === row.id && current.row.version > row.version ? current : { row, content: text(art.content) })
      })
      .catch(reason => { if (live) setError(desktopError(reason)) })
    return () => { live = false }
  }, [artifactId, target?.count, sessionKey, online, listedVersion, announcedVersion])
  const isDirtyDraft = draft !== null && open !== null && draft !== open.content
  const performBack = (): void => {
    setPickedId(''); setDocument(null); setDraft(null); setPendingDiscardAction(null); setSource(false); setOutlineOpen(false); setWidgetKey(0); setError('')
    onBack?.()
  }
  const requestDiscard = (action: 'cancel' | 'back'): void => {
    if (isDirtyDraft) {
      setPendingDiscardAction(action)
      return
    }
    setPendingDiscardAction(null)
    if (action === 'back') performBack()
    else setDraft(null)
  }
  const confirmDiscard = (): void => {
    const action = pendingDiscardAction
    setPendingDiscardAction(null)
    if (action === 'back') performBack()
    else setDraft(null)
  }
  const back = (): void => { requestDiscard('back') }
  const pin = (row: ArtifactRow): void => {
    void desktopCall(window.xerxes, sessionKey, 'artifacts.pin', { id: row.id, pinned: !row.pinned })
      .then(() => { setReload(value => value + 1); setDocument(current => current?.row.id === row.id ? { ...current, row: { ...current.row, pinned: !row.pinned } } : current) })
      .catch(reason => setError(desktopError(reason)))
  }
  if (open) {
    const { row } = open
    const isHtml = row.kind === 'html'
    const isImage = row.kind === 'image'
    const isMarkdown = row.kind === 'markdown'
    const hasPreviewToggle = isMarkdown || isHtml
    const isMarkdownDoc = isMarkdown && !source && draft === null
    const isHtmlPreview = isHtml && !source && draft === null
    // A Markdown artifact reads as a document: its own "# Title" becomes the page
    // title (shown once), and its headings make the outline.
    const doc = isMarkdown ? documentTitle(open.content, row.title) : null
    const outline = doc && isMarkdownDoc ? documentOutline(doc.body) : []
    const readStats = isMarkdownDoc ? readingStatsLabel(doc?.body ?? open.content) : ''
    const sourceLanguage = isMarkdown ? 'markdown' : isHtml ? 'html' : (row.language || '')
    const jump = (index: number): void => {
      docBody.current?.querySelectorAll('.artdoc__body :is(h1,h2,h3)')[index]?.scrollIntoView({ block: 'start' })
      setOutlineOpen(false)
    }
    return <div className="rsec rsec--detail" data-kind={row.kind}>
      <div className="rsec__bar">
        <button type="button" className="rsec__icon rsec__back" aria-label="Back to artifacts" title="Back to artifacts" onClick={back}>
          <T3Icon name="arrowLeft" size={14} />
        </button>
        {outline.length > 0 && (
          <button type="button" className="rsec__icon artdoc__outline-toggle" aria-expanded={outlineOpen} aria-label="Outline" title="Document outline" onClick={() => setOutlineOpen(value => !value)}>
            <T3Icon name="squareMenu" size={14} />
          </button>
        )}
        {!isMarkdownDoc && (
          <>
            <span className="rsec__kind-pill" data-kind={row.kind}>{kindBadgeLabel(row)}</span>
            <strong className="rsec__title" title={row.title}>{row.title}</strong>
            {row.version > 1 && <span className="rsec__ver-pill">v{row.version}</span>}
          </>
        )}
        <span className="rsec__spacer" />
        <span className="rsec__actions">
          {draft !== null ? <>
            <button type="button" className="rsec__text" onClick={() => requestDiscard('cancel')} disabled={saving}>Cancel</button>
            <button type="button" className="rsec__text rsec__text--primary" onClick={save} disabled={saving || draft === open.content}>{saving ? 'Saving…' : 'Save'}</button>
          </> : <>
            {hasPreviewToggle && (
              <span className="rsec__seg" role="group" aria-label="View mode">
                <button
                  type="button"
                  className="rsec__seg-btn"
                  aria-pressed={!source}
                  title={isHtml ? 'Show live interactive preview' : 'Show rendered document'}
                  onClick={() => setSource(false)}
                >
                  Preview
                </button>
                <button
                  type="button"
                  className="rsec__seg-btn"
                  aria-pressed={source}
                  title={isHtml ? 'Show HTML source' : 'Show Markdown source'}
                  onClick={() => setSource(true)}
                >
                  Source
                </button>
              </span>
            )}
            {isHtmlPreview && (
              <button
                type="button"
                className="rsec__icon"
                aria-label="Reload preview"
                title="Reload preview"
                onClick={() => setWidgetKey(value => value + 1)}
              >
                <T3Icon name="refresh" size={14} />
              </button>
            )}
            <button type="button" className="rsec__icon" aria-pressed={row.pinned} aria-label={row.pinned ? 'Unpin artifact' : 'Pin artifact'} title={row.pinned ? 'Unpin' : 'Pin'} onClick={() => pin(row)}><T3Icon name={row.pinned ? 'pinOff' : 'pin'} size={14} /></button>
            {/* Antigravity artifacts live in agy's own brain directory, which Xerxes never writes. */}
            {!isImage && !row.id.startsWith('agy:') && <button type="button" className="rsec__icon" aria-label="Edit artifact" title="Edit" disabled={!online} onClick={() => setDraft(open.content)}><T3Icon name="edit" size={14} /></button>}
            {!isImage && <button type="button" className="rsec__icon" data-copied={copied} aria-label={copied === 'copied' ? 'Copied' : 'Copy artifact'} title={copied === 'copied' ? 'Copied' : copied === 'failed' ? 'Copy failed — select and press ⌘C' : 'Copy'} onClick={() => copy(open.content)}><T3Icon name={copied === 'copied' ? 'check' : copied === 'failed' ? 'alertCircle' : 'copy'} size={14} /></button>}
          </>}
        </span>
      </div>
      {!isMarkdownDoc && (
        <p className="rsec__meta">
          <span>{kindFullLabel(row)}</span>
          <span className="rsec__meta-dot" aria-hidden="true">·</span>
          <span>{sizeLabel(row.size)}</span>
          <span className="rsec__meta-dot" aria-hidden="true">·</span>
          <span>{editedAgeLabel(row.updatedAt)}</span>
          {row.version > 1 && <>
            <span className="rsec__meta-dot" aria-hidden="true">·</span>
            <span className="rsec__meta-ver">v{row.version}</span>
          </>}
          {draft !== null && <>
            <span className="rsec__meta-dot" aria-hidden="true">·</span>
            <span className="rsec__meta-editing">Editing</span>
          </>}
        </p>
      )}
      {pendingDiscardAction && (
        <div className="rsec__confirm" role="alertdialog" aria-label="Unsaved artifact changes">
          <span>Discard unsaved changes to this artifact?</span>
          <span className="rsec__confirm-actions">
            <button type="button" className="rsec__text" onClick={() => setPendingDiscardAction(null)}>Keep editing</button>
            <button type="button" className="rsec__text rsec__text--primary" onClick={confirmDiscard}>Discard</button>
          </span>
        </div>
      )}
      {error && <p className="studio-error" role="alert">{error}</p>}
      {outlineOpen && outline.length > 0 && <nav className="artdoc__outline" aria-label="Outline">
        {outline.map((heading, index) => <button type="button" key={index} data-level={heading.level} title={heading.text} onClick={() => jump(index)}>{heading.text}</button>)}
      </nav>}
      <div className="rsec__body" ref={docBody}>
        {draft !== null
          ? <textarea className="rsec__editor" aria-label={`Edit ${row.title}`} value={draft} spellCheck={false} autoFocus
              onChange={event => { setDraft(event.target.value); if (pendingDiscardAction) setPendingDiscardAction(null) }}
              onKeyDown={event => {
                if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') { event.preventDefault(); save() }
                if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); requestDiscard('cancel') }
              }} />
          : isImage
            ? <div className="rsec__image-stage">
                <ChatImage src={open.content} alt={row.title} />
              </div>
          : isHtmlPreview
            ? <div className="rsec__widget-stage">
                <WidgetFrame
                  key={`${row.id}:${row.version}:${widgetKey}`}
                  html={open.content}
                  title={row.title}
                  mode="panel"
                  artifactId={row.id}
                  className="rsec__widget-frame"
                />
              </div>
            : source || !isMarkdown
              ? <pre className="rsec__source"><code>{sourceLanguage ? highlightCode(open.content, sourceLanguage) : open.content}</code></pre>
              : <article className="artdoc">
                  <header className="artdoc__head">
                    <h1 className="artdoc__title">{doc?.title ?? row.title}</h1>
                    <div className="artdoc__chips rsec__meta">
                      <span className="artdoc__chip artdoc__chip--kind" data-kind={row.kind}>Document</span>
                      <span className="artdoc__chip">{sizeLabel(row.size)}</span>
                      <span className="artdoc__chip">
                        <T3Icon name="timer" size={11} />
                        {dateLabel(row.updatedAt) ? `${dateLabel(row.updatedAt)} · ${editedAgeLabel(row.updatedAt)}` : editedAgeLabel(row.updatedAt)}
                      </span>
                      {row.version > 1 && <span className="artdoc__chip artdoc__chip--ver">v{row.version}</span>}
                      {readStats && <span className="artdoc__chip">{readStats}</span>}
                    </div>
                  </header>
                  <Markdown text={doc?.body ?? open.content} className="rsec__doc artdoc__body" />
                </article>}
      </div>
    </div>
  }
  if (artifactId) return <div className="rsec rsec--detail">
    <div className="rsec__bar">
      <button type="button" className="rsec__icon rsec__back" aria-label="Back to artifacts" title="Back to artifacts" onClick={back}>
        <T3Icon name="arrowLeft" size={14} />
      </button>
    </div>
    {error ? <p className="studio-error" role="alert">{error}</p> : <p className="rsec__empty" role="status">{online ? 'Opening document…' : 'Waiting for workspace connection…'}</p>}
  </div>
  const shown = rows.filter(row => !needle || row.title.toLowerCase().includes(needle.toLowerCase()))
  return <div className="rsec">
    <SectionSearch placeholder="Search artifacts" value={needle} onChange={setNeedle} />
    {listError === 'outdated'
      ? <p className="rsec__empty" role="status">This workspace runtime is older than the app and can't list artifacts yet. They appear here after the runtime restarts on the new build.</p>
      : (listError || error) && <p className="studio-error" role="alert">{listError || error}</p>}
    {loaded && !listError && rows.length === 0 && <p className="rsec__empty">No artifacts yet. Reports, plans and documents the agent saves with the artifact tool appear here.</p>}
    {rows.length > 0 && shown.length === 0 && needle.trim() && (
      <div className="rsec__empty rsec__empty--filter" role="status">
        <span>No matching artifacts</span>
        <button type="button" className="rsec__text" onClick={() => setNeedle('')}>Clear filter</button>
      </div>
    )}
    {shown.length > 0 && <div className="rsec__group"><span>Artifacts</span><span className="rsec__count">{shown.length}</span></div>}
    {shown.map(row => <div className="rsec__row rsec__row--artifact" key={row.id}>
      <button type="button" className="rsec__open" onClick={() => { setError(''); setPickedId(row.id) }} title={row.title}>
        <span className="rsec__row-icon" data-kind={row.kind} aria-hidden="true"><T3Icon name={artifactT3Icon(row.kind)} size={14} /></span>
        <span className="rsec__name">{row.title}</span>
        <span className="rsec__kind-pill" data-kind={row.kind}>{kindBadgeLabel(row)}</span>
        {row.version > 1 && <span className="rsec__ver-pill">v{row.version}</span>}
        <span className="rsec__age">{compactAge(row.updatedAt)}</span>
      </button>
      <button type="button" className="rsec__icon rsec__pin" aria-pressed={row.pinned} aria-label={row.pinned ? 'Unpin artifact' : 'Pin artifact'} onClick={() => pin(row)}><T3Icon name={row.pinned ? 'pinOff' : 'pin'} size={13} /></button>
    </div>)}
  </div>
}

export function SubagentsSection({
  rows,
  onInspect,
  sessionKey,
  online,
  onTitle,
  swarm,
  onOpenSwarm,
}: {
  rows: readonly SessionRow[]
  onInspect?: (id: string) => void
  sessionKey?: string
  online?: boolean
  onTitle?: (title: string) => void
  swarm?: SwarmStatusSnapshot | null
  onOpenSwarm?: () => void
}): ReactElement {
  const [needle, setNeedle] = useState('')
  const [hideFinished, setHideFinished] = useState(false)
  // A subagent opens inside the panel (Sections › Subagents › title), never over the conversation.
  const [openId, setOpenId] = useState('')
  const openRow = rows.find(row => row.id === openId)
  useEffect(() => { onTitle?.(openRow?.title ?? '') }, [openRow?.title])
  // Every hook runs before the detail view's early return (React's hook order).
  const filtered = useMemo(() => rows.filter(row => !needle || row.title.toLowerCase().includes(needle.toLowerCase())), [rows, needle])
  if (openRow && sessionKey !== undefined) {
    return <div className="rsec rsec--detail rsec--agent">
      <div className="rsec__bar">
        <button className="rsec__icon" aria-label="Back to subagents" onClick={() => setOpenId('')}><T3Icon name="arrowLeft" size={14} /></button>
        <span className="rsec__agent-ico" data-tone={agentState(openRow.status).tone} aria-hidden="true"><T3Icon name={subagentRoleIcon(openRow)} size={14} /></span>
        <strong className="rsec__title" title={openRow.title}>{openRow.title}</strong>
        {onInspect && <button className="rsec__icon" aria-label="Open in the conversation" title="Open in the conversation" onClick={() => onInspect(openRow.id)}><T3Icon name="maximize" size={13} /></button>}
      </div>
      <div className="rsec__body rsec__agent"><AgentInspector row={openRow} rows={rows} sessionKey={sessionKey} online={online ?? false} /></div>
    </div>
  }
  const running = filtered.filter(row => agentState(row.status).priority < 2)
  const finished = filtered.filter(row => agentState(row.status).priority >= 2)
  const line = (row: SessionRow): ReactElement => {
    const state = agentState(row.status)
    const kind = agentKindLabel(row.agentDetails)
    const roleIcon = subagentRoleIcon(row)
    return <button key={row.id} className="rsec__open rsec__row rsec__row--subagent" data-tone={state.tone} onClick={() => sessionKey !== undefined ? setOpenId(row.id) : onInspect?.(row.id)} title={row.title}>
      <span className="rsec__agent-ico" data-tone={state.tone} aria-hidden="true">
        {state.tone === 'working'
          ? <XerxesThinkingMark size={14} active />
          : state.tone === 'failed'
            ? <T3Icon name="alertCircle" size={14} />
            : <T3Icon name={roleIcon} size={14} />}
      </span>
      <span className="rsec__name">{row.title}</span>
      {kind && <span className="rsec__kind">• {kind}</span>}
      {state.tone !== 'done'
        ? <span className="rsec__state" data-tone={state.tone}>{state.label.toLowerCase()}</span>
        : <span className="rsec__done-ico" aria-hidden="true"><T3Icon name="check" size={12} /></span>}
      <span className="rsec__age">{row.age}</span>
    </button>
  }
  return <div className="rsec">
    <SectionSearch placeholder="Search subagents" value={needle} onChange={setNeedle} />
    {rows.length === 0 && (
      swarm?.active ? (() => {
        const livingWorkers = swarm.members.filter(
          member => member.role !== 'coordinator' && !member.killed && member.state !== 'killed',
        )
        const runningWorkers = livingWorkers.filter(member => member.state === 'working').length
        const summary = runningWorkers > 0
          ? `${runningWorkers} Immortals ${runningWorkers === 1 ? 'worker' : 'workers'} running`
          : livingWorkers.length > 0
            ? `${livingWorkers.length} Immortals ${livingWorkers.length === 1 ? 'worker' : 'workers'} active`
            : 'Immortals swarm active'
        return <div className="rsec__empty rsec__empty--swarm" role="status">
          <span>{summary} — </span>
          <button
            type="button"
            className="linkish rsec__swarm-jump"
            onClick={() => {
              if (onOpenSwarm) onOpenSwarm()
              else store.openSwarmBoard()
            }}
          >
            Open Immortals →
          </button>
        </div>
      })() : <p className="rsec__empty">No subagents in this task yet.</p>
    )}
    {rows.length > 0 && filtered.length === 0 && needle.trim() && (
      <div className="rsec__empty rsec__empty--filter" role="status">
        <span>No matching subagents</span>
        <button type="button" className="rsec__text" onClick={() => setNeedle('')}>Clear filter</button>
      </div>
    )}
    {running.length > 0 && <><div className="rsec__group"><span>Running</span><span className="rsec__count">{running.length}</span></div>{running.map(line)}</>}
    {finished.length > 0 && <>
      {/* the hide/show control sits on the Finished header, above the rows it folds. */}
      <div className="rsec__group"><span>Finished</span><span className="rsec__count">{finished.length}</span>
        <button className="rsec__toggle" aria-expanded={!hideFinished} onClick={() => setHideFinished(value => !value)}>{hideFinished ? 'Show finished' : 'Hide finished'}<T3Icon name={hideFinished ? 'chevronRight' : 'chevronDown'} size={12} /></button>
      </div>
      {!hideFinished && finished.map(line)}
    </>}
  </div>
}
