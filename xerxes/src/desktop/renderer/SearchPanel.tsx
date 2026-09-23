// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Session search (⌘K → "Search sessions & messages…"): a debounce-fed
 * `session.search` box over the daemon's persisted transcripts. Each hit is
 * one matched message — role, excerpt, session title, age — and opening one
 * rides the same resume flow as the sidebar (⏎ or click). Mounted only while
 * open, exactly like the command palette.
 */

import { useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type ReactElement } from 'react'

import { useDialogFocus } from './dialogFocus.js'
import { store, type Snapshot } from './store.js'

function ageLabel(updatedAt: string): string {
  if (!updatedAt) return ''
  const epoch = Date.parse(updatedAt)
  if (Number.isNaN(epoch)) return ''
  const minutes = Math.floor((Date.now() - epoch) / 60_000)
  if (minutes < 1) return 'now'
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  return `${Math.floor(hours / 24)}d`
}

export function SessionSearch({ snap }: { snap: Snapshot }): ReactElement | null {
  if (!snap.searchOpen) return null
  const [needle, setNeedle] = useState('')
  const [cursor, setCursor] = useState(0)
  const ref = useRef<HTMLInputElement>(null)
  useDialogFocus(ref)

  // Debounced live search: 250ms after the last keystroke, ≥2 chars. The
  // store drops stale responses by sequence, so laggy answers never win.
  useEffect(() => {
    const timer = setTimeout(() => { store.runSessionSearch(needle) }, 250)
    return () => clearTimeout(timer)
  }, [needle])
  useEffect(() => {
    setCursor(0)
  }, [needle])

  const hits = snap.searchResults
  // openSession stopped being a mid-turn no-op: it now navigates, or hands
  // the target to another window so the running turn keeps its connection —
  // which is exactly what the sidebar rows already do. The gate here was
  // resting on a comment the store had made false.
  const open = (sessionId: string): void => {
    store.closeSessionSearch()
    void store.openSession(sessionId)
  }
  const onKey = (event: ReactKeyboardEvent): void => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setCursor(value => Math.min(value + 1, Math.max(hits.length - 1, 0)))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setCursor(value => Math.max(value - 1, 0))
    } else if (event.key === 'Enter') {
      event.preventDefault()
      const hit = hits[cursor]
      if (hit) open(hit.sessionId)
    } else if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      store.closeSessionSearch()
    }
  }

  const statsLine = useMemo(() => {
    if (snap.searchSearching) return 'searching…'
    if (snap.searchError) return snap.searchError
    const stats = snap.searchStats
    const count = hits.length
    if (!stats) return count ? `${count} match${count === 1 ? '' : 'es'}` : ''
    const base = `index: ${stats.sessions} session${stats.sessions === 1 ? '' : 's'} · ${stats.searchableMessages} messages`
    return count ? `${count} match${count === 1 ? '' : 'es'} — ${base}` : base
  }, [hits.length, snap.searchError, snap.searchSearching, snap.searchStats])

  return (
    <>
      <div className="backdrop backdrop--clear" onClick={() => store.closeSessionSearch()} />
      <div className="palette" role="dialog" aria-label="Search sessions and messages">
        <input
          ref={ref}
          className="palette__in"
          placeholder="Search across every saved session…"
          spellCheck={false}
          value={needle}
          onChange={e => setNeedle(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Escape') {
              // GlobalKeys must not turn this Escape into turn-cancel.
              e.preventDefault()
              e.stopPropagation()
              store.closeSessionSearch()
            } else {
              onKey(e)
            }
          }}
        />
        <div className="palette__list">
          {hits.length === 0 && needle.trim().length >= 2 && !snap.searchSearching && (
            <div className="search-status" role="status">{snap.searchError ? `Search failed: ${snap.searchError}` : 'No messages match.'}</div>
          )}
          {needle.trim().length < 2 && (
            <div className="search-status">Type at least two characters to search every saved conversation.</div>
          )}
          {hits.map((hit, index) => (
            <button
              key={`${hit.sessionId}:${hit.messageIndex}`}
              className={`srow${index === cursor ? ' is-sel' : ''}`}
              title={snap.turnActive ? `open “${hit.title || hit.sessionId}” in its own window — this task keeps running` : `open “${hit.title || hit.sessionId}”`}
              onMouseEnter={() => setCursor(index)}
              onClick={() => open(hit.sessionId)}
            >
              <span className="srow__head">
                <span className="srow__title">{hit.title || `#${hit.sessionId.slice(0, 6)}`}</span>
                <span className="srow__meta">
                  <code>{hit.role || 'message'}</code>
                  {ageLabel(hit.updatedAt) ? ` · ${ageLabel(hit.updatedAt)}` : ''}
                </span>
              </span>
              <span className="srow__excerpt">{hit.excerpt}</span>
            </button>
          ))}
        </div>
        <div className="palette__foot">
          <span><kbd>↑↓</kbd> select</span>
          <span><kbd>⏎</kbd> open</span>
          <span><kbd>esc</kbd> close</span>
          <span className="palette__hint">{statsLine}</span>
        </div>
      </div>
    </>
  )
}
