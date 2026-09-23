// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'

export interface PanelLayout {
  sidebarWidth: number
  inspectorWidth: number
  sidebarHidden: boolean
}
const defaults: PanelLayout = { sidebarWidth: 240, inspectorWidth: 340, sidebarHidden: false }
const key = 'xerxes.desktop.layout.v1'
export function parsePanelLayout(value: unknown): PanelLayout {
  const record = value && typeof value === 'object' ? value as Record<string, unknown> : {}
  const width = (name: string, fallback: number, min: number, max: number): number => typeof record[name] === 'number' && Number.isFinite(record[name]) ? Math.max(min, Math.min(max, record[name])) : fallback
  return { sidebarWidth: width('sidebarWidth', 240, 180, 360), inspectorWidth: width('inspectorWidth', 340, 260, 1400), sidebarHidden: record.sidebarHidden === true }
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
