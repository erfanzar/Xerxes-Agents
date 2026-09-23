// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Find in the visible surface (⌘F). The transcript is the primary surface
 * of this app and had no search of its own — only the cross-session FTS,
 * which answers a different question.
 *
 * This drives Chromium's own find, so it highlights and scrolls exactly the
 * way the rest of the platform does and needs no text index. It searches
 * what is mounted: collapsed activity groups and other tabs are not in the
 * DOM, which the empty result says out loud rather than pretending.
 */

import { useEffect, useRef, useState, type ReactElement } from 'react'
import { Icon } from './Icon.js'

type Finder = (text: string, caseSensitive: boolean, backwards: boolean, wrap: boolean) => boolean

export function FindBar(): ReactElement | null {
  const [open, setOpen] = useState(false)
  const [needle, setNeedle] = useState('')
  const [missed, setMissed] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const show = (): void => { setOpen(true); setMissed(false); requestAnimationFrame(() => input.current?.select()) }
    window.addEventListener('xerxes:find', show)
    return () => window.removeEventListener('xerxes:find', show)
  }, [])

  useEffect(() => {
    if (!open) return
    // Capture, so Escape closes the find bar before the global ladder reads
    // it as "stop the running turn".
    const onKey = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      event.stopImmediatePropagation()
      close()
    }
    document.addEventListener('keydown', onKey, true)
    return () => document.removeEventListener('keydown', onKey, true)
  }, [open])

  const close = (): void => {
    setOpen(false)
    window.getSelection()?.removeAllRanges()
  }

  const step = (backwards: boolean): void => {
    const text = needle.trim()
    if (!text) return
    const find = (window as unknown as { find?: Finder }).find
    // Collapse first: Chromium resumes from the current selection, so a
    // stale range from the previous term skips matches above it.
    const hit = find ? find(text, false, backwards, true) : false
    setMissed(!hit)
  }

  if (!open) return null
  return (
    <div className="findbar" role="search" aria-label="Find in page">
      <input
        ref={input}
        className="findbar__in"
        value={needle}
        spellCheck={false}
        placeholder="Find in this view…"
        aria-label="Find in this view"
        onChange={event => { setNeedle(event.target.value); setMissed(false) }}
        onKeyDown={event => {
          if (event.key !== 'Enter') return
          event.preventDefault()
          step(event.shiftKey)
        }}
      />
      <button aria-label="Previous match" title="Previous match (⇧⏎)" onClick={() => step(true)}><Icon name="arrowUp" size={13} /></button>
      <button aria-label="Next match" title="Next match (⏎)" onClick={() => step(false)}><Icon name="arrowDown" size={13} /></button>
      {missed && <span className="findbar__miss" role="status">Not in view</span>}
      <button aria-label="Close find" title="Close (esc)" onClick={close}><Icon name="close" size={13} /></button>
    </div>
  )
}
