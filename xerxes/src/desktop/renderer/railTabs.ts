// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The right panel's open tabs. A tab is a section (Files, Changes, …) or one
 * item inside a section (an artifact), so several things stay open side by
 * side. Pure, so the open/close/focus rules are testable without React.
 */

/** Views that open in the task-context rail beside the conversation. */
export type RailPanel = 'subagents' | 'swarm' | 'btw' | 'docs' | 'files' | 'outputs' | 'review' | 'terminal' | 'background' | 'browser' | 'activity' | 'usage'

export interface RailTab {
  readonly panel: RailPanel
  /** One item inside the section, e.g. an artifact id. */
  readonly item?: string
  /** Shown instead of the section name once known (an artifact's title). */
  readonly title?: string
}

/** How much the right panel's header shows at its current width. */
export type RailFit = 'wide' | 'compact' | 'narrow'
/** Below 460px tab titles shorten and item headers wrap; below 360px inactive tabs are icons only. */
export function railFitOf(width: number): RailFit {
  return width < 360 ? 'narrow' : width < 460 ? 'compact' : 'wide'
}

/** Most tabs kept open; opening one more drops the oldest that is not active. */
export const MAX_RAIL_TABS = 10

export const tabKey = (tab: Pick<RailTab, 'panel' | 'item'>): string => tab.item ? `${tab.panel}:${tab.item}` : tab.panel
export const panelOfKey = (key: string): RailPanel => key.split(':')[0] as RailPanel

/** Open a tab (or keep the one already open); the new tab goes last. */
export function openTab(tabs: readonly RailTab[], tab: RailTab, active?: string): readonly RailTab[] {
  const key = tabKey(tab)
  if (tabs.some(open => tabKey(open) === key)) return tabs
  const next = [...tabs, tab]
  if (next.length <= MAX_RAIL_TABS) return next
  const drop = next.findIndex(open => tabKey(open) !== active && tabKey(open) !== key)
  return drop < 0 ? next : next.filter((_, index) => index !== drop)
}

/** Close a tab; the active one hands focus to its right neighbour, else its left. Null when none are left. */
export function closeTab(tabs: readonly RailTab[], key: string, active: string): { readonly tabs: readonly RailTab[]; readonly active: string | null } {
  const index = tabs.findIndex(tab => tabKey(tab) === key)
  if (index < 0) return { tabs, active }
  const next = tabs.filter((_, at) => at !== index)
  if (next.length === 0) return { tabs: next, active: null }
  if (key !== active) return { tabs: next, active }
  return { tabs: next, active: tabKey(next[Math.min(index, next.length - 1)]!) }
}

/** Give a tab its item's title; an empty title keeps the old one. */
export function retitleTab(tabs: readonly RailTab[], key: string, title: string): readonly RailTab[] {
  const clean = title.trim()
  if (!clean) return tabs
  return tabs.map(tab => tabKey(tab) === key && tab.title !== clean ? { ...tab, title: clean } : tab)
}
