// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Cross-runtime Chat & History Importer Modal.
 *
 * Lets users browse and import existing conversations (with full transcript
 * history) from Antigravity (`agy`), Claude Code, OpenAI Codex, or any
 * connected ACP runtime into a fresh Xerxes session pinned to that runtime.
 */

import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactElement } from 'react'

import { RuntimeBrandIcon } from './BrandMark.js'
import { PickerLayer } from './Overlays.js'
import { T3Icon } from './T3Icons.js'
import { useDialogFocus } from './dialogFocus.js'
import {
  ageOf,
  store,
  type ExternalImportableRuntimeSummary,
  type ExternalImportableSessionRow,
} from './store.js'
import { compactModelTriggerName } from './T3ModelPicker.js'

export const OPEN_IMPORT_CHAT_EVENT = 'xerxes:open-import-chat'

/** Open the cross-runtime Import Chat dialog from any surface (optionally pre-filtered to one runtime). */
export function openImportChatModal(runtime?: string): void {
  if (typeof window === 'undefined') return
  window.dispatchEvent(new CustomEvent(OPEN_IMPORT_CHAT_EVENT, { detail: { runtime: runtime ?? 'all' } }))
}

function folderLabelOf(cwd: string): string {
  const clean = cwd.replace(/\/+$/, '')
  if (!clean) return ''
  return clean.split(/[\\/]/).pop() || clean
}

export function ImportChatModal({
  open,
  initialRuntime = 'all',
  currentCwd,
  onClose,
  onImported,
  initialSessions,
  initialRuntimes,
}: {
  readonly open: boolean
  readonly initialRuntime?: string
  readonly currentCwd?: string
  readonly onClose: () => void
  readonly onImported?: (sessionId: string) => void
  /** Test seam for deterministic SSR verification. */
  readonly initialSessions?: readonly ExternalImportableSessionRow[]
  readonly initialRuntimes?: readonly ExternalImportableRuntimeSummary[]
}): ReactElement | null {
  const [runtime, setRuntime] = useState<string>(initialRuntime || 'all')
  const [allProjects, setAllProjects] = useState<boolean>(true)
  const [query, setQuery] = useState<string>('')
  const [sessions, setSessions] = useState<readonly ExternalImportableSessionRow[]>(initialSessions ?? [])
  const [runtimes, setRuntimes] = useState<readonly ExternalImportableRuntimeSummary[]>(initialRuntimes ?? [])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [importingId, setImportingId] = useState<string | null>(null)
  const [manualOpen, setManualOpen] = useState<boolean>(false)
  const [manualRuntime, setManualRuntime] = useState<string>('antigravity')
  const [manualId, setManualId] = useState<string>('')
  const dialogRef = useRef<HTMLDivElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)

  useDialogFocus(dialogRef, open)

  useEffect(() => {
    if (!open) return
    setRuntime(initialRuntime || 'all')
    if (initialRuntime && initialRuntime !== 'all') {
      setManualRuntime(initialRuntime)
    }
    setError(null)
  }, [open, initialRuntime])

  const load = useCallback(async (opts?: { readonly allProjects?: boolean; readonly runtime?: string }) => {
    setLoading(true)
    setError(null)
    const res = await store.listImportableExternalSessions({
      runtime: opts?.runtime ?? 'all',
      allProjects: opts?.allProjects ?? allProjects,
      limit: 120,
    })
    setLoading(false)
    if (!res.ok) {
      setError(res.error || 'Could not load external chats.')
      return
    }
    setSessions(res.sessions)
    setRuntimes(res.runtimes)
  }, [allProjects])

  useEffect(() => {
    if (!open || initialSessions !== undefined) return
    void load({ allProjects, runtime: 'all' })
  }, [open, allProjects, initialSessions, load])

  useEffect(() => {
    if (open) searchRef.current?.focus()
  }, [open])

  const visibleSessions = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return sessions.filter(item => {
      if (runtime !== 'all' && item.runtimeId !== runtime) return false
      if (!needle) return true
      return (
        item.title.toLowerCase().includes(needle)
        || item.id.toLowerCase().includes(needle)
        || item.cwd.toLowerCase().includes(needle)
        || item.runtimeName.toLowerCase().includes(needle)
        || (item.preview?.toLowerCase().includes(needle) ?? false)
        || (item.model?.toLowerCase().includes(needle) ?? false)
      )
    })
  }, [sessions, runtime, query])

  const runtimeTabs = useMemo(() => {
    const counts = new Map<string, number>()
    for (const s of sessions) {
      counts.set(s.runtimeId, (counts.get(s.runtimeId) ?? 0) + 1)
    }
    const base = runtimes.length > 0
      ? runtimes
      : [
          { id: 'antigravity', name: 'Antigravity', provider: 'antigravity' as const, available: true, count: counts.get('antigravity') ?? 0 },
          { id: 'claude-code', name: 'Claude Code', provider: 'claude-code' as const, available: true, count: counts.get('claude-code') ?? 0 },
          { id: 'codex', name: 'Codex', provider: 'codex' as const, available: true, count: counts.get('codex') ?? 0 },
        ]
    return base.map(r => ({ ...r, count: counts.get(r.id) ?? r.count }))
  }, [runtimes, sessions])

  const handleImport = async (item: ExternalImportableSessionRow): Promise<void> => {
    if (importingId) return
    setImportingId(`${item.runtimeId}:${item.id}`)
    setError(null)
    try {
      const res = await store.importExternalSession({
        runtimeId: item.runtimeId,
        externalSessionId: item.id,
        ...(item.cwd ? { cwd: item.cwd } : currentCwd ? { cwd: currentCwd } : {}),
        title: item.title,
        ...(item.source ? { source: item.source } : {}),
        connect: true,
      })
      if (!res.ok) {
        setError(res.error || 'Could not import conversation.')
        return
      }
      onClose()
      if (res.sessionId) onImported?.(res.sessionId)
    } finally {
      setImportingId(null)
    }
  }

  const handleManualSubmit = async (event: FormEvent): Promise<void> => {
    event.preventDefault()
    const cleanId = manualId.trim()
    if (!cleanId || importingId) return
    setImportingId(`${manualRuntime}:${cleanId}`)
    setError(null)
    try {
      const res = await store.importExternalSession({
        runtimeId: manualRuntime,
        externalSessionId: cleanId,
        ...(currentCwd ? { cwd: currentCwd } : {}),
        connect: true,
      })
      if (!res.ok) {
        setError(res.error || 'Could not import that session ID.')
        return
      }
      setManualId('')
      onClose()
      if (res.sessionId) onImported?.(res.sessionId)
    } finally {
      setImportingId(null)
    }
  }

  if (!open) return null

  return (
    <PickerLayer>
      <div
        className="backdrop import-chat-backdrop"
        onClick={() => { if (!importingId) onClose() }}
        onKeyDown={event => {
          if (event.key === 'Escape') {
            event.preventDefault()
            event.stopPropagation()
            if (!importingId) onClose()
          }
        }}
      >
        <div
          ref={dialogRef}
          className="import-chat-modal"
          role="dialog"
          aria-modal="true"
          aria-label="Import chat from runtime"
          onClick={event => event.stopPropagation()}
        >
          <div className="import-chat-modal__head">
            <div className="import-chat-modal__titles">
              <h2>Import chat from runtime</h2>
              <p>Import existing conversations and full history from Antigravity, Claude Code, Codex, or ACP runtimes.</p>
            </div>
            <div className="import-chat-modal__head-actions">
              <button
                type="button"
                className="t3-icon-btn"
                title="Refresh list"
                aria-label="Refresh importable chats"
                disabled={loading || Boolean(importingId)}
                onClick={() => void load({ allProjects, runtime: 'all' })}
              >
                <T3Icon name="refresh" size={14} />
              </button>
              <button
                type="button"
                className="t3-icon-btn"
                title="Close"
                aria-label="Close import dialog"
                disabled={Boolean(importingId)}
                onClick={onClose}
              >
                <T3Icon name="close" size={14} />
              </button>
            </div>
          </div>

          <div className="import-chat-modal__tabs" role="tablist" aria-label="Filter by runtime">
            <button
              type="button"
              role="tab"
              aria-selected={runtime === 'all'}
              className={`import-chat-modal__tab${runtime === 'all' ? ' is-active' : ''}`}
              onClick={() => setRuntime('all')}
            >
              <span>All</span>
              <span className="import-chat-modal__count">{sessions.length}</span>
            </button>
            {runtimeTabs.map(tab => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={runtime === tab.id}
                className={`import-chat-modal__tab${runtime === tab.id ? ' is-active' : ''}`}
                onClick={() => {
                  setRuntime(tab.id)
                  setManualRuntime(tab.id)
                }}
              >
                <RuntimeBrandIcon provider={tab.provider} size={13} />
                <span>{tab.name}</span>
                <span className="import-chat-modal__count">{tab.count}</span>
              </button>
            ))}
          </div>

          <div className="import-chat-modal__controls">
            <label className="import-chat-modal__search">
              <T3Icon name="search" size={13} />
              <input
                ref={searchRef}
                type="text"
                value={query}
                onChange={event => setQuery(event.currentTarget.value)}
                placeholder="Search by title, session ID, model, or folder…"
                aria-label="Search importable chats"
                spellCheck={false}
              />
              {query && (
                <button
                  type="button"
                  className="import-chat-modal__clear"
                  aria-label="Clear search"
                  onClick={() => setQuery('')}
                >
                  <T3Icon name="close" size={11} />
                </button>
              )}
            </label>
            <button
              type="button"
              className={`import-chat-modal__scope-btn${allProjects ? ' is-on' : ''}`}
              aria-pressed={allProjects}
              title={allProjects ? 'Showing chats from all folders on this machine' : 'Showing chats matching the active folder'}
              onClick={() => setAllProjects(prev => !prev)}
            >
              {allProjects ? 'All folders' : 'This folder'}
            </button>
          </div>

          {error && <p className="import-chat-modal__err" role="alert">{error}</p>}

          <div className="import-chat-modal__list" role="list" aria-label="Importable conversations">
            {loading && sessions.length === 0 ? (
              <div className="import-chat-modal__empty">Scanning external runtime sessions…</div>
            ) : visibleSessions.length === 0 ? (
              <div className="import-chat-modal__empty">
                {query
                  ? 'No matching chats found.'
                  : 'No chats found for this filter. Try switching to All folders or importing directly by session ID below.'}
              </div>
            ) : (
              visibleSessions.map(item => {
                const key = `${item.runtimeId}:${item.id}`
                const busy = importingId === key
                const folder = folderLabelOf(item.cwd)
                const age = item.updatedAt ? ageOf(item.updatedAt) : ''
                const modelShort = item.model ? compactModelTriggerName(item.model) : ''
                return (
                  <div key={key} className="import-chat-modal__row" role="listitem">
                    <div className="import-chat-modal__row-main">
                      <div className="import-chat-modal__row-top">
                        <RuntimeBrandIcon provider={item.provider} size={14} />
                        <span className="import-chat-modal__row-title" title={item.title}>{item.title}</span>
                        {item.alreadyImported && (
                          <span className="import-chat-modal__badge">Imported</span>
                        )}
                      </div>
                      {item.preview && (
                        <div className="import-chat-modal__row-preview">{item.preview}</div>
                      )}
                      <div className="import-chat-modal__row-meta">
                        <span>{item.runtimeName}</span>
                        {modelShort && <span>· {modelShort}</span>}
                        {item.turnCount && item.turnCount > 0 && (
                          <span>· {item.turnCount} {item.turnCount === 1 ? 'turn' : 'turns'}</span>
                        )}
                        {folder && <span title={item.cwd}>· {folder}</span>}
                        {age && <span>· {age}</span>}
                        <span className="import-chat-modal__row-id" title={item.id}>· {item.id.slice(0, 8)}</span>
                      </div>
                    </div>
                    <button
                      type="button"
                      className="btn import-chat-modal__btn"
                      disabled={Boolean(importingId)}
                      onClick={() => void handleImport(item)}
                    >
                      {busy ? 'Importing…' : item.alreadyImported ? 'Re-import' : 'Import'}
                    </button>
                  </div>
                )
              })
            )}
          </div>

          <div className="import-chat-modal__foot">
            {!manualOpen ? (
              <button
                type="button"
                className="import-chat-modal__manual-toggle"
                onClick={() => setManualOpen(true)}
              >
                <T3Icon name="plus" size={12} />
                <span>Import by session ID…</span>
              </button>
            ) : (
              <form className="import-chat-modal__manual" onSubmit={event => void handleManualSubmit(event)}>
                <select
                  aria-label="External runtime for manual import"
                  value={manualRuntime}
                  onChange={event => setManualRuntime(event.currentTarget.value)}
                >
                  {runtimeTabs.map(tab => (
                    <option key={tab.id} value={tab.id}>{tab.name}</option>
                  ))}
                </select>
                <input
                  type="text"
                  value={manualId}
                  onChange={event => setManualId(event.currentTarget.value)}
                  placeholder="Paste external session UUID…"
                  aria-label="External session ID"
                  spellCheck={false}
                />
                <button
                  type="submit"
                  className="btn btn--solid"
                  disabled={!manualId.trim() || Boolean(importingId)}
                >
                  {importingId ? 'Importing…' : 'Import'}
                </button>
              </form>
            )}
          </div>
        </div>
      </div>
    </PickerLayer>
  )
}
