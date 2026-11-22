// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useSyncExternalStore } from 'react'

/**
 * Whether this page is out of sight: hidden, minimized, or covered by another
 * workspace view in the same window (the main process marks the root
 * `data-occluded`). A window keeps one page per workspace it has opened, so
 * polling that ignores this runs once per open workspace, all the time.
 * Pollers skip their runtime call while this is true and keep their timer.
 */
export function pageIsBackground(doc: Document | undefined = typeof document !== 'undefined' ? document : undefined): boolean {
  if (!doc) return false
  return doc.visibilityState === 'hidden'
    || doc.hidden === true
    || Boolean(doc.documentElement?.hasAttribute?.('data-occluded'))
}

/**
 * Whether the window is currently unfocused (blurred or marked
 * `data-window-unfocused` by `main.tsx`).
 */
export function pageIsUnfocused(doc: Document | undefined = typeof document !== 'undefined' ? document : undefined): boolean {
  if (!doc) return false
  if (doc.documentElement?.hasAttribute?.('data-window-unfocused')) return true
  if (typeof doc.hasFocus === 'function' && !doc.hasFocus()) return true
  return false
}

/**
 * True when the page is hidden, occluded by another workspace view, or
 * unfocused — background pollers and non-essential timers should pause or
 * aggressively back off.
 */
export function pageIsInactive(doc: Document | undefined = typeof document !== 'undefined' ? document : undefined): boolean {
  return pageIsBackground(doc) || pageIsUnfocused(doc)
}

/** Current animation/perf pause flag (true when hidden, occluded, or unfocused). */
export function isPerfPaused(doc: Document | undefined = typeof document !== 'undefined' ? document : undefined): boolean {
  if (!doc) return false
  if (doc.documentElement?.getAttribute?.('data-perf-paused') === 'true') return true
  return pageIsInactive(doc)
}

const perfPauseListeners = new Set<() => void>()
let lastKnownPerfPaused: boolean | null = null

function syncSvgSmilAnimations(doc: Document, paused: boolean): void {
  if (typeof doc.querySelectorAll !== 'function') return
  const svgs = doc.querySelectorAll('svg')
  for (const node of svgs) {
    const svg = node as SVGSVGElement & {
      pauseAnimations?: () => void
      unpauseAnimations?: () => void
    }
    try {
      if (paused) svg.pauseAnimations?.()
      else svg.unpauseAnimations?.()
    } catch {
      // Headless DOM stubs may expose svg nodes without SMIL timelines.
    }
  }
}

/**
 * Synchronize `data-perf-paused="true"` on `<html>` and pause/unpause all SVG
 * SMIL animation timelines when the window is hidden, occluded, or unfocused.
 */
export function syncPerfPaused(doc: Document | undefined = typeof document !== 'undefined' ? document : undefined): boolean {
  if (!doc || !doc.documentElement) return false
  const paused = pageIsInactive(doc)
  const root = doc.documentElement
  if (paused) {
    if (root.getAttribute?.('data-perf-paused') !== 'true') {
      root.setAttribute?.('data-perf-paused', 'true')
    }
  } else {
    if (root.hasAttribute?.('data-perf-paused')) {
      root.removeAttribute?.('data-perf-paused')
    }
  }
  syncSvgSmilAnimations(doc, paused)
  if (lastKnownPerfPaused !== paused) {
    lastKnownPerfPaused = paused
    for (const listener of perfPauseListeners) listener()
  }
  return paused
}

/** Subscribe to changes in `isPerfPaused()` (visibility, occlusion, focus). */
export function subscribePerfPaused(listener: () => void): () => void {
  perfPauseListeners.add(listener)
  return () => {
    perfPauseListeners.delete(listener)
  }
}

const getServerPerfPaused = (): boolean => false
const getClientPerfPaused = (): boolean => isPerfPaused()

/**
 * React hook returning whether animations & background work are paused.
 * Returns `false` during SSR so static snapshots remain identical.
 */
export function usePerfPaused(): boolean {
  return useSyncExternalStore(subscribePerfPaused, getClientPerfPaused, getServerPerfPaused)
}

/**
 * Wire `visibilitychange`, `focus`, `blur`, and root `data-occluded` /
 * `data-window-unfocused` attribute mutations to `syncPerfPaused()`.
 */
export function installPerfPauseListeners(
  win: Window | undefined = typeof window !== 'undefined' ? window : undefined,
  doc: Document | undefined = typeof document !== 'undefined' ? document : undefined,
): () => void {
  if (!doc) return () => {}
  const sync = (): void => {
    syncPerfPaused(doc)
  }
  const onFocus = (): void => {
    doc.documentElement?.removeAttribute?.('data-window-unfocused')
    sync()
  }
  const onBlur = (): void => {
    doc.documentElement?.setAttribute?.('data-window-unfocused', '')
    sync()
  }
  doc.addEventListener?.('visibilitychange', sync)
  win?.addEventListener?.('focus', onFocus)
  win?.addEventListener?.('blur', onBlur)
  let observer: MutationObserver | null = null
  if (typeof MutationObserver !== 'undefined' && doc.documentElement) {
    observer = new MutationObserver(sync)
    observer.observe(doc.documentElement, {
      attributes: true,
      attributeFilter: ['data-occluded', 'data-window-unfocused'],
    })
  }
  sync()
  return () => {
    doc.removeEventListener?.('visibilitychange', sync)
    win?.removeEventListener?.('focus', onFocus)
    win?.removeEventListener?.('blur', onBlur)
    observer?.disconnect()
  }
}

// ── Shared document `data-theme` observer ──────────────────────────────────
// Avoids attaching a separate MutationObserver per WidgetFrame or MermaidBlock.

const themeListeners = new Set<() => void>()
let themeObserver: MutationObserver | null = null

export function readDocumentTheme(doc: Document | undefined = typeof document !== 'undefined' ? document : undefined): 'dark' | 'light' {
  if (!doc) return 'dark'
  return doc.documentElement?.getAttribute?.('data-theme') === 'light' ? 'light' : 'dark'
}

export function subscribeDocumentTheme(listener: () => void): () => void {
  themeListeners.add(listener)
  if (!themeObserver && typeof document !== 'undefined' && typeof MutationObserver !== 'undefined' && document.documentElement) {
    themeObserver = new MutationObserver(() => {
      for (const cb of themeListeners) cb()
    })
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
  }
  return () => {
    themeListeners.delete(listener)
    if (themeListeners.size === 0 && themeObserver) {
      themeObserver.disconnect()
      themeObserver = null
    }
  }
}
