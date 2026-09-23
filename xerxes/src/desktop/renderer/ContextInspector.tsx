// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useRef, useState, type ReactElement } from 'react'
import { desktopCall, desktopError, record, records } from './desktopRpc.js'
import type { Snapshot } from './store.js'
import { Icon } from './Icon.js'

/** `goal_policy` → `Goal policy`; paths and prose titles pass through untouched. */
export function contextTitle(title: string): string {
  if (!/^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/.test(title)) return title
  const words = title.replace(/_/g, ' ')
  return words[0]!.toUpperCase() + words.slice(1)
}

/**
 * Prompt text arrives hard-wrapped at ~80 columns. In a 300px rail those
 * breaks land mid-line and every sentence wraps twice. Join wrapped lines
 * back into paragraphs, but leave lists, headings, tables, indented blocks
 * and fenced code exactly as written — their line breaks carry meaning.
 */
export function reflowContextText(text: string): string {
  const structural = /^\s*(?:[-*•+]\s|\d+[.)]\s|#|>|\||```|\s{2,}\S)/
  return text.replace(/\r\n/g, '\n').split(/\n{2,}/).map(paragraph => {
    if (paragraph.includes('```')) return paragraph
    const lines = paragraph.split('\n')
    let out = lines[0] ?? ''
    for (const line of lines.slice(1)) {
      out += structural.test(line) || structural.test(out.split('\n').at(-1) ?? '') ? `\n${line}` : ` ${line.trim()}`
    }
    return out
  }).join('\n\n')
}

export const contextSections = ['instructions', 'memory', 'conversation', 'tools', 'compaction'] as const
type Section = typeof contextSections[number]
type Control = { scope: 'global' | 'project'; path: string; pinned: boolean; excluded: boolean }
type Entry = { index: number; title: string; text: string; truncated: boolean; estimated_tokens: number; control?: Control }
type Page = { generation: string; controls_revision?: number; note: string; section: Section; offset: number; next_offset: number | null; entries: Entry[]; sections: { id: string; count: number; available: boolean; estimated_tokens: number; provenance: string }[] }
export function parseContextPage(value: unknown, section: Section, offset: number): Page {
  const data = record(value)
  if (data.section !== section || data.offset !== offset || typeof data.generation !== 'string' || typeof data.note !== 'string' || (data.next_offset !== null && (!Number.isSafeInteger(data.next_offset) || Number(data.next_offset) <= offset))) throw new Error('Context changed or returned an invalid page. Refresh the inspector.')
  const sections = records(data.sections)
  if (sections.length !== contextSections.length || new Set(sections.map(row => row.id)).size !== contextSections.length || sections.some(row => !contextSections.includes(row.id as Section) || typeof row.available !== 'boolean' || typeof row.count !== 'number' || typeof row.estimated_tokens !== 'number' || typeof row.provenance !== 'string')) throw new Error('Invalid context sections')
  const entries = records(data.entries)
  if (entries.length > 20) throw new Error('Invalid context page size')
  for (const entry of entries) {
    if (!Number.isSafeInteger(entry.index) || typeof entry.title !== 'string' || typeof entry.text !== 'string' || typeof entry.truncated !== 'boolean' || typeof entry.estimated_tokens !== 'number') throw new Error('Invalid context entry')
    if (entry.control !== undefined) {
      const control = record(entry.control)
      if (section !== 'memory' || !['global','project'].includes(String(control.scope)) || typeof control.path !== 'string' || typeof control.pinned !== 'boolean' || typeof control.excluded !== 'boolean' || !Number.isSafeInteger(data.controls_revision) || Number(data.controls_revision) < 0) throw new Error('Invalid context controls')
    }
  }
  return data as unknown as Page
}

export function ContextInspector({ snap }: { snap: Snapshot }): ReactElement {
  const [section, setSection] = useState<Section>('instructions')
  const [offset, setOffset] = useState(0)
  const [refresh, setRefresh] = useState(0)
  const [page, setPage] = useState<Page | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const generation = useRef<string | undefined>(undefined)
  const mutation = useRef(false)
  const epoch = useRef(0)
  useEffect(() => { generation.current = undefined; setOffset(0); setNotice('') }, [snap.sessionKey])
  useEffect(() => {
    const version = ++epoch.current
    setPage(null); setError(''); setLoading(true)
    if (snap.connection !== 'online') { setError('Reconnect to inspect context.'); setLoading(false); return }
    void desktopCall(window.xerxes, snap.sessionKey, 'context.inspect', { section, offset, ...(generation.current ? { generation: generation.current } : {}) })
      .then(value => { if (epoch.current === version) { const next = parseContextPage(value, section, offset); generation.current = next.generation; setPage(next) } })
      .catch(failure => { if (epoch.current === version) setError(desktopError(failure)) })
      .finally(() => { if (epoch.current === version) setLoading(false) })
    return () => { epoch.current++ }
  }, [snap.sessionKey, snap.connection, section, offset, refresh])
  const change = async (control: Control, action: string): Promise<void> => {
    if (!page || mutation.current || loading || error || snap.turnActive) return
    mutation.current = true; setSaving(true); setNotice(''); const version = epoch.current
    try {
      await desktopCall(window.xerxes, snap.sessionKey, 'context.control', { action, scope: control.scope, path: control.path, revision: page.controls_revision, generation: page.generation })
      if (epoch.current === version) { setNotice('Saved. Applies on the next turn.'); generation.current = undefined; setOffset(0); setRefresh(value => value + 1) }
    } catch (failure) { if (epoch.current === version) setError(desktopError(failure)) }
    finally { mutation.current = false; setSaving(false) }
  }
  const summary = page?.sections.find(row => row.id === section)
  const paged = page ? !(offset === 0 && page.next_offset === null) : false
  return <section className="context-inspector" aria-label="Context inspector">
    <p className="context-inspector__lead">What the agent sees. Estimates are local; no provider request is made.</p>
    <div className="context-inspector__bar">
      <select aria-label="Context section" value={section} disabled={saving} onChange={event => { setSection(event.target.value as Section); setOffset(0) }}>{contextSections.map(id => <option key={id} value={id}>{id[0]!.toUpperCase()+id.slice(1)}</option>)}</select>
      <button className="context-inspector__refresh" title="Refresh context" aria-label="Refresh context" disabled={loading || saving} onClick={() => { generation.current = undefined; setOffset(0); setRefresh(value => value + 1) }}><Icon name="retry" size={13} /></button>
    </div>
    {summary && <p className="context-inspector__meta">{summary.available ? `${summary.count.toLocaleString()} ${summary.count === 1 ? 'entry' : 'entries'} · ${section === 'compaction' ? 'not in model context' : `~${summary.estimated_tokens.toLocaleString()} tokens`} · ${summary.provenance}` : 'Not assembled yet'}</p>}
    {loading && <p className="context-inspector__meta" role="status">Loading context…</p>}{error && <p className="studio-error" role="alert">{error}</p>}{notice && <p className="context-inspector__meta" role="status">{notice}</p>}
    {page && <>
      {page.entries.length === 0 && <p className="context-inspector__meta">No entries in this section.</p>}
      <div className="context-sources">
      {page.entries.map(entry => {
        const title = contextTitle(entry.title)
        return <details className="context-source" key={`${page.generation}:${section}:${entry.index}`}>
          <summary>
            <span className="context-source__title" title={entry.title}>{title}</span>
            {entry.control?.pinned ? <span className="context-source__tag">Pinned</span> : entry.control?.excluded ? <span className="context-source__tag">Excluded</span> : null}
            <span className="context-source__tokens">~{entry.estimated_tokens.toLocaleString()}</span>
          </summary>
          <div className="context-source__body" tabIndex={0}>{reflowContextText(entry.text)}</div>
          {entry.truncated && <p className="context-inspector__meta">Excerpt truncated at 8,000 characters.</p>}
          {entry.control && <div className="lsp-actions">
            <button disabled={saving || loading || !!error || snap.turnActive} onClick={() => void change(entry.control!, entry.control!.pinned ? 'unpin' : 'pin')}>{entry.control.pinned ? 'Unpin' : 'Pin source'}</button>
            <button disabled={saving || loading || !!error || snap.turnActive} onClick={() => void change(entry.control!, entry.control!.excluded ? 'include' : 'exclude')}>{entry.control.excluded ? 'Include source' : 'Exclude source'}</button>
          </div>}
        </details>
      })}
      </div>
      {section === 'memory' && snap.turnActive && <p className="context-inspector__meta">Wait for this turn to finish before changing memory sources.</p>}
      {paged && <div className="context-pager">
        <button aria-label="Previous page" disabled={loading || saving || !!error || offset === 0} onClick={() => setOffset(Math.max(0, offset - 20))}><Icon name="chevron" size={12} /></button>
        <span>{page.entries.length ? `${offset + 1}–${offset + page.entries.length}` : '0'}{summary ? ` of ${summary.count.toLocaleString()}` : ''}</span>
        <button aria-label="Next page" disabled={loading || saving || !!error || page.next_offset === null} onClick={() => setOffset(page.next_offset!)}><Icon name="chevron" size={12} /></button>
      </div>}
      <p className="context-inspector__note">{page.note}</p>
    </>}
  </section>
}

export function ContextInspectorDisclosure({ snap }: { snap: Snapshot }): ReactElement {
  const [open, setOpen] = useState(false)
  return <details className="activity-context" onToggle={event => setOpen(event.currentTarget.open)}><summary>Inspect context & memory</summary>{open && <ContextInspector key={snap.sessionKey} snap={snap}/>}</details>
}
