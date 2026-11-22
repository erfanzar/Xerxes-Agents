// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * T3 Code-style resizable bottom Terminal Drawer (`data-thread-terminal-drawer`).
 * Docks beneath the chat canvas and composer, shares Xerxes's durable PTY shells
 * via `TerminalPanel`, persists its height in localStorage, and lets the user
 * pop it out to the right panel or close it with ⌘J / Ctrl+`.
 */

import { useState, type ReactElement } from 'react'
import type { Snapshot } from './store.js'
import { TerminalPanel } from './TerminalPanel.js'
import { T3Icon } from './T3Icons.js'

export const TERMINAL_DRAWER_KEY = 'xerxes.desktop.terminalDrawer.v1'
export const TERMINAL_DRAWER_DEFAULT_HEIGHT = 280
export const TERMINAL_DRAWER_MIN_HEIGHT = 180
export const TERMINAL_DRAWER_MAX_HEIGHT = 620

export function clampTerminalDrawerHeight(height: unknown): number {
  if (typeof height !== 'number' || !Number.isFinite(height)) return TERMINAL_DRAWER_DEFAULT_HEIGHT
  return Math.max(TERMINAL_DRAWER_MIN_HEIGHT, Math.min(TERMINAL_DRAWER_MAX_HEIGHT, Math.round(height)))
}

export function loadTerminalDrawerHeight(): number {
  if (typeof window === 'undefined') return TERMINAL_DRAWER_DEFAULT_HEIGHT
  try {
    const raw = window.localStorage.getItem(TERMINAL_DRAWER_KEY)
    if (!raw) return TERMINAL_DRAWER_DEFAULT_HEIGHT
    const parsed = JSON.parse(raw) as { height?: unknown }
    return clampTerminalDrawerHeight(parsed?.height)
  } catch {
    return TERMINAL_DRAWER_DEFAULT_HEIGHT
  }
}

export function saveTerminalDrawerHeight(height: number): void {
  try {
    window.localStorage?.setItem(TERMINAL_DRAWER_KEY, JSON.stringify({ height: clampTerminalDrawerHeight(height) }))
  } catch {
    /* Drawer remains resizable when storage is unavailable. */
  }
}

export function TerminalDrawer({
  snap,
  height,
  onResize,
  onMoveToRightPanel,
  onClose,
}: {
  readonly snap: Snapshot
  readonly height: number
  readonly onResize: (nextHeight: number) => void
  readonly onMoveToRightPanel: () => void
  readonly onClose: () => void
}): ReactElement {
  const [drag, setDrag] = useState<{ readonly y: number; readonly height: number } | null>(null)
  const clamped = clampTerminalDrawerHeight(height)
  const folder = snap.cwd.replaceAll('\\', '/').split('/').filter(Boolean).at(-1) || 'workspace'

  const updateHeight = (next: number): void => {
    const bounded = clampTerminalDrawerHeight(next)
    onResize(bounded)
    saveTerminalDrawerHeight(bounded)
  }

  return (
    <section
      className="t3-term-drawer"
      data-thread-terminal-drawer="true"
      aria-label="Terminal drawer"
      style={{ height: `${clamped}px` }}
    >
      <div
        className="t3-term-drawer__resize"
        role="separator"
        aria-label="Resize terminal drawer"
        aria-orientation="horizontal"
        aria-valuenow={clamped}
        aria-valuemin={TERMINAL_DRAWER_MIN_HEIGHT}
        aria-valuemax={TERMINAL_DRAWER_MAX_HEIGHT}
        tabIndex={0}
        onPointerDown={event => {
          event.currentTarget.focus()
          event.currentTarget.setPointerCapture(event.pointerId)
          setDrag({ y: event.clientY, height: clamped })
          event.preventDefault()
        }}
        onPointerMove={event => {
          if (drag) updateHeight(drag.height + (drag.y - event.clientY))
        }}
        onPointerUp={() => setDrag(null)}
        onLostPointerCapture={() => setDrag(null)}
        onKeyDown={event => {
          if (event.key === 'ArrowUp') { event.preventDefault(); updateHeight(clamped + 24) }
          else if (event.key === 'ArrowDown') { event.preventDefault(); updateHeight(clamped - 24) }
          else if (event.key === 'Home') { event.preventDefault(); updateHeight(TERMINAL_DRAWER_MIN_HEIGHT) }
          else if (event.key === 'End') { event.preventDefault(); updateHeight(TERMINAL_DRAWER_MAX_HEIGHT) }
        }}
      />
      <header className="t3-term-drawer__head">
        <div className="t3-term-drawer__title">
          <T3Icon name="terminal" size={13} />
          <span>Terminal</span>
          <span className="t3-term-drawer__meta">{folder}{snap.branch ? ` · ${snap.branch}` : ''}</span>
        </div>
        <div className="t3-term-drawer__actions">
          <button
            type="button"
            className="t3-term-drawer__btn"
            title="Move terminal to right panel"
            aria-label="Move terminal to right panel"
            onClick={onMoveToRightPanel}
          >
            <T3Icon name="panelRight" size={13} />
          </button>
          <button
            type="button"
            className="t3-term-drawer__btn"
            title="Close terminal drawer (⌘J)"
            aria-label="Close terminal drawer"
            onClick={onClose}
          >
            <T3Icon name="close" size={13} />
          </button>
        </div>
      </header>
      <div className="t3-term-drawer__body">
        <TerminalPanel snap={snap} />
      </div>
    </section>
  )
}
