// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'
import type { RailPanel } from './railTabs.js'

export interface PanelLayout {
  sidebarWidth: number
  /** Legacy single side-panel width; now only the Activity fallback for old saves. */
  inspectorWidth: number
  sidebarHidden: boolean
  /** Each side panel keeps its own width, as Claude Code does: resizing Files never moves Activity. */
  panelWidths: Readonly<Partial<Record<RailPanel, number>>>
}
const defaults: PanelLayout = { sidebarWidth: 272, inspectorWidth: 340, sidebarHidden: false, panelWidths: {} }

/**
 * Default and minimum width per side panel. Files, Changes, Terminal and Browser
 * defaults are Claude Code's measured panel widths (1800pt window, full-height
 * divider in .xerxes/review/claude-app 02–05); Activity is the width Erfan set.
 * The minimums keep each panel's own controls usable.
 */
export const PANEL_SIZES: Readonly<Record<RailPanel, { readonly width: number; readonly min: number }>> = {
  activity: { width: 366, min: 300 },
  subagents: { width: 366, min: 300 },
  swarm: { width: 520, min: 360 },
  btw: { width: 420, min: 320 },
  background: { width: 366, min: 300 },
  outputs: { width: 366, min: 300 },
  usage: { width: 366, min: 300 },
  docs: { width: 640, min: 400 },
  files: { width: 730, min: 420 },
  review: { width: 918, min: 480 },
  terminal: { width: 666, min: 360 },
  browser: { width: 538, min: 420 },
}
const PANEL_MAX = 1400

/** The saved width of one panel, or its default; an old single width still counts for Activity. */
export function panelWidthOf(layout: PanelLayout, panel: RailPanel): number {
  const saved = layout.panelWidths[panel]
  if (saved !== undefined) return saved
  if (panel === 'activity' && layout.inspectorWidth !== defaults.inspectorWidth) return layout.inspectorWidth
  return PANEL_SIZES[panel].width
}
/**
 * Vibes uses Claude Code's measured sidebar width (270) while the person has not
 * resized it; side panels take their own widths from PANEL_SIZES.
 */
export function vibesPanelLayout(layout: PanelLayout): PanelLayout {
  return {
    ...layout,
    sidebarWidth: layout.sidebarWidth === defaults.sidebarWidth ? 270 : layout.sidebarWidth,
  }
}
const key = 'xerxes.desktop.layout.v1'
export function parsePanelLayout(value: unknown): PanelLayout {
  const record = value && typeof value === 'object' ? value as Record<string, unknown> : {}
  const width = (name: string, fallback: number, min: number, max: number): number => typeof record[name] === 'number' && Number.isFinite(record[name]) ? Math.max(min, Math.min(max, record[name])) : fallback
  const savedPanels = record.panelWidths && typeof record.panelWidths === 'object' ? record.panelWidths as Record<string, unknown> : {}
  const panelWidths: Partial<Record<RailPanel, number>> = {}
  for (const [panel, size] of Object.entries(PANEL_SIZES) as [RailPanel, { width: number; min: number }][]) {
    const value = savedPanels[panel]
    if (typeof value === 'number' && Number.isFinite(value)) panelWidths[panel] = Math.max(size.min, Math.min(PANEL_MAX, value))
  }
  return { sidebarWidth: width('sidebarWidth', 272, 180, 360), inspectorWidth: width('inspectorWidth', 340, 260, PANEL_MAX), sidebarHidden: record.sidebarHidden === true, panelWidths }
}
export function usePanelLayout(): { layout: PanelLayout; setLayout: (patch: Partial<PanelLayout>) => void } {
  const [layout, update] = useState<PanelLayout>(() => {
    if (typeof window === 'undefined') return defaults
    try { return parsePanelLayout(JSON.parse(window.localStorage.getItem(key) || '{}')) } catch { return defaults }
  })
  const setLayout = (patch: Partial<PanelLayout>): void => update(current => {
    const next = parsePanelLayout({ ...current, ...patch })
    try { window.localStorage.setItem(key, JSON.stringify(next)) } catch { /* Layout remains usable when storage is unavailable. */ }
    return next
  })
  return { layout, setLayout }
}

/** Pointer capture keeps resizing local to the divider; arrows provide the same operation. */
export function PanelDivider({ label, value, min, max, reverse = false, onChange }: {
  label: string; value: number; min: number; max: number; reverse?: boolean; onChange: (value: number) => void
}): ReactElement {
  const [drag, setDrag] = useState<{ x: number; value: number } | null>(null)
  const change = (next: number): void => onChange(Math.max(min, Math.min(max, next)))
  return <div className="panel-divider" role="separator" aria-label={label} aria-orientation="vertical" aria-valuenow={value} aria-valuemin={min} aria-valuemax={max} tabIndex={0}
    onPointerDown={event => { event.currentTarget.focus(); event.currentTarget.setPointerCapture(event.pointerId); setDrag({ x: event.clientX, value }); event.preventDefault() }}
    onPointerMove={event => { if (drag) change(drag.value + (event.clientX - drag.x) * (reverse ? -1 : 1)) }}
    onPointerUp={() => setDrag(null)} onLostPointerCapture={() => setDrag(null)}
    onKeyDown={event => { if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); change(value + (event.key === 'ArrowRight' ? 16 : -16) * (reverse ? -1 : 1)) } else if (event.key === 'Home') { event.preventDefault(); change(min) } else if (event.key === 'End') { event.preventDefault(); change(max) } }} />
}
