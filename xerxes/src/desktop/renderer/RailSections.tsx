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
import { Icon } from './Icon.js'
import { Markdown } from './markdown.js'
import { copyToClipboard } from './CopyButton.js'
import { highlightCode } from './richMarkdown.js'
import { desktopCall, desktopError, record, text } from './desktopRpc.js'
import { agentKindLabel, agentState } from './AgentRoster.js'
import { AgentInspector } from './AgentInspector.js'
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

const sizeLabel = (bytes: number): string => bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : bytes >= 1024 ? `${Math.round(bytes / 1024)} KB` : `${bytes} B`
const kindLabel = (row: ArtifactRow): string => row.kind === 'markdown' ? 'md' : row.kind === 'code' ? (row.language || 'code') : row.kind

function SectionSearch({ placeholder, value, onChange }: { placeholder: string; value: string; onChange: (value: string) => void }): ReactElement {
  return <label className="rsec__search"><Icon name="search" size={13} /><input aria-label={placeholder} placeholder={placeholder} value={value} onChange={event => onChange(event.target.value)} spellCheck={false} /></label>
}

export function ArtifactsSection({ sessionKey, online, onTitle }: { sessionKey: string; online: boolean; onTitle?: (title: string) => void }): ReactElement {
  const [rows, setRows] = useState<readonly ArtifactRow[]>([])
  const [error, setError] = useState('')
  const [loaded, setLoaded] = useState(false)
  const [needle, setNeedle] = useState('')
  const [open, setOpen] = useState<{ row: ArtifactRow; content: string } | null>(null)
  const [source, setSource] = useState(false)
  const [reload, setReload] = useState(0)
  // Edit mode: a draft of the content, saved through artifacts.update.
  const [draft, setDraft] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [copied, setCopied] = useState<'idle' | 'copied' | 'failed'>('idle')
  const copiedTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  useEffect(() => () => clearTimeout(copiedTimer.current), [])
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
        setOpen({ row: artifactRowsOf([art])[0] ?? open.row, content: text(art.content) })
        setDraft(null); setError(''); setReload(value => value + 1)
      })
      .catch(reason => setError(desktopError(reason)))
      .finally(() => setSaving(false))
  }
  const unsupported = useRef(false)
  useEffect(() => {
    if (!online || unsupported.current) return
    let live = true
    void desktopCall(window.xerxes, sessionKey, 'artifacts.list')
      .then(result => { if (live) { setRows(artifactRowsOf(result.artifacts)); setError(''); setLoaded(true) } })
      .catch(reason => {
        if (!live) return
        const message = desktopError(reason)
        // An older runtime has no artifact methods: say what to do, not the raw RPC error.
        const outdated = /unknown method/i.test(message)
        if (outdated) unsupported.current = true
        setError(outdated ? 'outdated' : message)
        setLoaded(true)
      })
    // Artifacts arrive mid-turn; a cheap list poll keeps the section current
    // (not against a runtime that doesn't have the method).
    const timer = setInterval(() => { if (!unsupported.current) setReload(value => value + 1) }, 5000)
    return () => { live = false; clearInterval(timer) }
  }, [sessionKey, online, reload])
  useEffect(() => { onTitle?.(open?.row.title ?? '') }, [open?.row.title])
  const show = (row: ArtifactRow): void => {
    setError('')
    void desktopCall(window.xerxes, sessionKey, 'artifacts.read', { id: row.id })
      .then(result => { const art = record(result.artifact); setOpen({ row: artifactRowsOf([art])[0] ?? row, content: text(art.content) }); setSource(false); setDraft(null) })
      .catch(reason => setError(desktopError(reason)))
  }
  const pin = (row: ArtifactRow): void => {
    void desktopCall(window.xerxes, sessionKey, 'artifacts.pin', { id: row.id, pinned: !row.pinned })
      .then(() => { setReload(value => value + 1); if (open?.row.id === row.id) setOpen({ ...open, row: { ...open.row, pinned: !row.pinned } }) })
      .catch(reason => setError(desktopError(reason)))
  }
  if (open) {
    const { row } = open
    return <div className="rsec rsec--detail">
      <div className="rsec__bar">
        <button className="rsec__icon" aria-label="Back to artifacts" title="Back to artifacts" onClick={() => { setOpen(null); setDraft(null) }}><Icon name="sidebar" size={14} /></button>
        <Icon name="file" size={13} /><strong className="rsec__title" title={row.title}>{row.title}</strong>
        <span className="rsec__spacer" />
        {draft !== null ? <>
          <button className="rsec__text" onClick={() => setDraft(null)} disabled={saving}>Cancel</button>
          <button className="rsec__text rsec__text--primary" onClick={save} disabled={saving || draft === open.content}>{saving ? 'Saving…' : 'Save'}</button>
        </> : <>
          <button className="rsec__icon" aria-pressed={row.pinned} aria-label={row.pinned ? 'Unpin artifact' : 'Pin artifact'} title={row.pinned ? 'Unpin' : 'Pin'} onClick={() => pin(row)}><Icon name="pin" size={14} /></button>
          {row.kind === 'markdown' && <button className="rsec__text" aria-pressed={source} title={source ? 'Show the rendered document' : 'Show the Markdown source'} onClick={() => setSource(value => !value)}>{source ? 'Preview' : 'Source'}</button>}
          <button className="rsec__icon" aria-label="Edit artifact" title="Edit" disabled={!online} onClick={() => setDraft(open.content)}><Icon name="note" size={14} /></button>
          <button className="rsec__icon" data-copied={copied} aria-label={copied === 'copied' ? 'Copied' : 'Copy artifact'} title={copied === 'copied' ? 'Copied' : copied === 'failed' ? 'Copy failed — select and press ⌘C' : 'Copy'} onClick={() => copy(open.content)}><Icon name={copied === 'copied' ? 'check' : copied === 'failed' ? 'warning' : 'copy'} size={14} /></button>
        </>}
      </div>
      <p className="rsec__meta">{row.kind === 'markdown' ? 'Document' : row.kind === 'code' ? 'Code' : row.kind === 'html' ? 'HTML' : 'Text'} · {sizeLabel(row.size)} · edited {compactAge(row.updatedAt) || 'just now'} ago{row.version > 1 ? ` · v${row.version}` : ''}</p>
      {error && <p className="studio-error" role="alert">{error}</p>}
      <div className="rsec__body">
        {draft !== null
          ? <textarea className="rsec__editor" aria-label={`Edit ${row.title}`} value={draft} spellCheck={false} autoFocus
              onChange={event => setDraft(event.target.value)}
              onKeyDown={event => {
                if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') { event.preventDefault(); save() }
                if (event.key === 'Escape') { event.preventDefault(); setDraft(null) }
              }} />
          : source || row.kind !== 'markdown'
            ? <pre className="rsec__source"><code>{row.kind === 'markdown' ? highlightCode(open.content, 'markdown') : row.language ? highlightCode(open.content, row.language) : open.content}</code></pre>
            : <Markdown text={open.content} className="rsec__doc" />}
      </div>
    </div>
  }
  const shown = rows.filter(row => !needle || row.title.toLowerCase().includes(needle.toLowerCase()))
  return <div className="rsec">
    <SectionSearch placeholder="Search artifacts" value={needle} onChange={setNeedle} />
    {error === 'outdated'
      ? <p className="rsec__empty" role="status">This workspace runtime is older than the app and can't list artifacts yet. They appear here after the runtime restarts on the new build.</p>
      : error && <p className="studio-error" role="alert">{error}</p>}
    {loaded && !error && rows.length === 0 && <p className="rsec__empty">No artifacts yet. Reports, plans and documents the agent saves with the artifact tool appear here.</p>}
    {shown.length > 0 && <div className="rsec__group"><span>Artifacts</span><span className="rsec__count">{shown.length}</span></div>}
    {shown.map(row => <div className="rsec__row" key={row.id}>
      <button className="rsec__open" onClick={() => show(row)} title={row.title}>
        <Icon name="file" size={13} /><span className="rsec__name">{row.title}</span><span className="rsec__kind">• {kindLabel(row)}</span>
        <span className="rsec__age">{compactAge(row.updatedAt)}</span>
      </button>
      <button className="rsec__icon rsec__pin" aria-pressed={row.pinned} aria-label={row.pinned ? 'Unpin artifact' : 'Pin artifact'} onClick={() => pin(row)}><Icon name="pin" size={13} /></button>
    </div>)}
  </div>
}

export function SubagentsSection({ rows, onInspect, sessionKey, online, onTitle }: { rows: readonly SessionRow[]; onInspect?: (id: string) => void; sessionKey?: string; online?: boolean; onTitle?: (title: string) => void }): ReactElement {
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
        <button className="rsec__icon" aria-label="Back to subagents" onClick={() => setOpenId('')}><span className="settings-back__icon"><Icon name="arrow" size={13} /></span></button>
        <strong className="rsec__title" title={openRow.title}>{openRow.title}</strong>
        {onInspect && <button className="rsec__icon" aria-label="Open in the conversation" title="Open in the conversation" onClick={() => onInspect(openRow.id)}><Icon name="expand" size={13} /></button>}
      </div>
      <div className="rsec__body rsec__agent"><AgentInspector row={openRow} rows={rows} sessionKey={sessionKey} online={online ?? false} /></div>
    </div>
  }
  const running = filtered.filter(row => agentState(row.status).priority < 2)
  const finished = filtered.filter(row => agentState(row.status).priority >= 2)
  const line = (row: SessionRow): ReactElement => {
    const state = agentState(row.status)
    const kind = agentKindLabel(row.agentDetails)
    return <button key={row.id} className="rsec__open rsec__row" data-tone={state.tone} onClick={() => sessionKey !== undefined ? setOpenId(row.id) : onInspect?.(row.id)} title={row.title}>
      <Icon name="agents" size={13} /><span className="rsec__name">{row.title}</span>
      {kind && <span className="rsec__kind">• {kind}</span>}
      {state.tone !== 'done' && <span className="rsec__state">{state.label.toLowerCase()}</span>}
      <span className="rsec__age">{row.age}</span>
    </button>
  }
  return <div className="rsec">
    <SectionSearch placeholder="Search subagents" value={needle} onChange={setNeedle} />
    {rows.length === 0 && <p className="rsec__empty">No subagents in this task yet.</p>}
    {running.length > 0 && <><div className="rsec__group"><span>Running</span><span className="rsec__count">{running.length}</span></div>{running.map(line)}</>}
    {finished.length > 0 && <>
      {/* the hide/show control sits on the Finished header, above the rows it folds. */}
      <div className="rsec__group"><span>Finished</span><span className="rsec__count">{finished.length}</span>
        <button className="rsec__toggle" aria-expanded={!hideFinished} onClick={() => setHideFinished(value => !value)}>{hideFinished ? 'Show finished' : 'Hide finished'}<Icon name="caretDown" size={11} /></button>
      </div>
      {!hideFinished && finished.map(line)}
    </>}
  </div>
}
