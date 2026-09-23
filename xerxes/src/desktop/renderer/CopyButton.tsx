// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Copy-to-clipboard for transcript content.
 *
 * Nothing in the transcript was copyable except by drag-selecting it —
 * in a tool whose whole output is code and command results. The renderer
 * loads from file:// with a restrictive CSP and has no clipboard
 * capability exposed through the preload, so this uses the same
 * `execCommand` path the session-id copy in App.tsx already relies on.
 */

import { useEffect, useRef, useState, type ReactElement } from 'react'

/**
 * `navigator.clipboard` needs a secure context, document focus and a
 * transient activation; on this file:// renderer it rejects often enough
 * that a button relying on it alone reports "Copy failed" for content that
 * copies fine. Try it, then fall back to the synchronous path.
 */
export function copyToClipboard(text: string): boolean {
  void navigator.clipboard?.writeText(text).catch(() => {})
  const area = document.createElement('textarea')
  area.value = text
  area.setAttribute('readonly', '')
  area.style.position = 'fixed'
  area.style.opacity = '0'
  document.body.appendChild(area)
  area.select()
  let copied = false
  try {
    copied = document.execCommand('copy')
  } catch {
    copied = false
  }
  area.remove()
  return copied
}

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }): ReactElement {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle')
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])
  return (
    <button
      type="button"
      className="copybtn"
      // The transcript rows are click targets of their own (expand,
      // inspect); copying must not also toggle them.
      onClick={event => {
        event.stopPropagation()
        event.preventDefault()
        setState(copyToClipboard(text) ? 'copied' : 'failed')
        clearTimeout(timer.current)
        timer.current = setTimeout(() => setState('idle'), 1400)
      }}
      aria-label={state === 'copied' ? 'Copied' : label}
      title={label}
    >
      {state === 'copied' ? 'Copied' : state === 'failed' ? 'Press ⌘C' : label}
    </button>
  )
}
