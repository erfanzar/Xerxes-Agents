// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Workspace grouping for the sidebar. A workspace is the folder a chat ran
 * in; the sidebar reads like a project tree — folder name, then its chats —
 * instead of one flat list across every project on the machine.
 */
import type { ContextSession, WorkspaceContext } from '../main/contextNavigation.js'

export interface GroupableRow {
  readonly id: string
  readonly cwd?: string
}

export interface WorkspaceGroup<T> {
  readonly name: string
  /** The group's own project path, when every row agrees on one. */
  readonly cwd: string
  readonly rows: T[]
}

/** Folder basename for display; empty cwd collapses to a fallback bucket. */
export function workspaceName(cwd: string | undefined): string {
  const clean = (cwd ?? '').trim().replace(/\/+$/, '')
  if (!clean) return ''
  const parts = clean.split('/')
  return parts[parts.length - 1] || clean
}

/**
 * Group rows under their folder names, preserving the list's own recency
 * order — first group seen stays first, and rows keep their sort inside.
 * When `currentCwd` is given, the workspace the user is in is hoisted to
 * the top: New task always lands there, so it leads the list.
 */
export function groupByWorkspace<T extends GroupableRow>(
  rows: readonly T[],
  currentCwd = '',
  remembered: readonly string[] = [],
): Array<WorkspaceGroup<T>> {
  const order: string[] = []
  const groups = new Map<string, { cwd: string; rows: T[] }>()
  for (const row of rows) {
    const name = row.cwd ?? ''
    const bucket = groups.get(name)
    if (bucket) bucket.rows.push(row)
    else {
      order.push(name)
      groups.set(name, { cwd: row.cwd ?? '', rows: [row] })
    }
  }
  for (const cwd of remembered) {
    if (!groups.has(cwd)) { order.push(cwd); groups.set(cwd, { cwd, rows: [] }) }
  }
  const listed = order.map(name => {
    const group = groups.get(name)!
    return { name: workspaceName(group.cwd) || 'Other', cwd: group.cwd, rows: group.rows }
  })
  const current = currentCwd
  if (!current) return listed
  const home = listed.find(group => group.cwd === current)
  if (!home) return listed
  return [home, ...listed.filter(group => group !== home)]
}

/** The label of this computer's own sections, as the main process names it. */
export const THIS_MAC = 'This Mac'

/** The host a view's own tasks run on: an `ssh:<target>:` storage scope names its SSH target. */
export function hostOfScope(scope: string | undefined): string {
  if (!scope?.startsWith('ssh:')) return THIS_MAC
  return scope.slice(4).replace(/:+$/, '') || 'SSH'
}

interface SectionBase {
  /** Host and folder: two sections with one key are the same project. */
  readonly key: string
  readonly host: string
  readonly folder: string
  /** `This Mac · Xerxes-Agents`, `v5p-128-dev · prismcore`. */
  readonly label: string
  readonly cwd: string
}
/** A sidebar section: this view's own folder, or a folder of another open view or saved host. */
export type SidebarSection<T> =
  | SectionBase & { readonly kind: 'here'; readonly rows: T[] }
  | SectionBase & { readonly kind: 'context'; readonly context: WorkspaceContext; readonly rows: ContextSession[] }

const sectionKey = (host: string, cwd: string): string => `${host}\u0000${cwd.replace(/\/+$/, '') || '/'}`
const byName = (left: string, right: string): number =>
  left.localeCompare(right, 'en', { sensitivity: 'base' }) || (left < right ? -1 : left > right ? 1 : 0)

/**
 * Every section the sidebar lists, one per host and folder, in one fixed
 * order: this computer first, then each host, folders by name inside it.
 * The order never depends on which view is open, so opening a task in
 * another folder (another view) shows the same list in the same order;
 * before, a view drew the other contexts first and its own folders last,
 * and a local view seen from an SSH view was one "This Mac" heap of every
 * folder's tasks. A task listed by this view is not repeated under a context.
 */
export function sidebarSections<T extends GroupableRow>(
  groups: readonly WorkspaceGroup<T>[],
  contexts: readonly WorkspaceContext[],
  host: string,
): Array<SidebarSection<T>> {
  const sections: Array<SidebarSection<T>> = []
  const shown = new Set<string>()
  const keys = new Set<string>()
  for (const group of groups) {
    const key = sectionKey(host, group.cwd)
    keys.add(key)
    for (const row of group.rows) shown.add(row.id)
    sections.push({ kind: 'here', key, host, folder: group.name, label: `${host} · ${group.name}`, cwd: group.cwd, rows: group.rows })
  }
  for (const context of contexts) {
    const rows = context.sessions.filter(row => !shown.has(row.id))
    if (context.remote) {
      const contextHost = context.alias || context.label
      const cwd = context.workspacePath ?? context.workspace
      const key = sectionKey(contextHost, cwd)
      if (rows.length === 0 && keys.has(key)) continue
      keys.add(key)
      sections.push({ kind: 'context', key, host: contextHost, folder: workspaceName(cwd) || 'SSH', label: context.label, cwd, context, rows })
      continue
    }
    // A local view's daemon lists every folder on this computer: one section per folder.
    for (const group of groupByWorkspace(rows, '', context.workspace ? [context.workspace] : [])) {
      const key = sectionKey(THIS_MAC, group.cwd)
      if (group.rows.length === 0 && keys.has(key)) continue
      keys.add(key)
      sections.push({ kind: 'context', key, host: THIS_MAC, folder: group.name, label: `${THIS_MAC} · ${group.name}`, cwd: group.cwd, context, rows: group.rows })
    }
  }
  const rank = (section: SidebarSection<T>): number => section.host === THIS_MAC ? 0 : 1
  return sections.sort((a, b) => rank(a) - rank(b) || byName(a.host, b.host) || byName(a.folder, b.folder) || byName(a.cwd, b.cwd))
}

/** How many tasks a sidebar section lists before "Show N more". */
export const SIDEBAR_SECTION_ROWS = 5

/**
 * The rows a section shows: the first `limit`, unless it is expanded. The
 * open task is never hidden by the limit, so selecting an older task does
 * not make it vanish from its own section.
 */
export function limitRows<T extends GroupableRow>(rows: readonly T[], limit: number, all: boolean, keepId?: string | null): { shown: T[]; hidden: number } {
  if (all || rows.length <= limit) return { shown: [...rows], hidden: 0 }
  const shown = rows.slice(0, limit)
  const kept = keepId ? rows.slice(limit).find(row => row.id === keepId) : undefined
  if (kept) shown.push(kept)
  return { shown, hidden: rows.length - shown.length }
}

/** Sections folded down to their name, remembered across launches and windows. */
export const COLLAPSED_SECTIONS_KEY = 'xerxes.desktop.sidebar.collapsed.v1'
interface KeyStore { getItem(key: string): string | null; setItem(key: string, value: string): void }
const defaultStore = (): KeyStore | undefined => {
  try { return (globalThis as { localStorage?: KeyStore }).localStorage } catch { return undefined }
}

export function loadCollapsedSections(storage: KeyStore | undefined = defaultStore()): Set<string> {
  try {
    const value: unknown = JSON.parse(storage?.getItem(COLLAPSED_SECTIONS_KEY) || '[]')
    if (!Array.isArray(value)) return new Set()
    return new Set(value.filter((key): key is string => typeof key === 'string' && key.length <= 4096).slice(0, 500))
  } catch { return new Set() }
}

export function saveCollapsedSections(keys: ReadonlySet<string>, storage: KeyStore | undefined = defaultStore()): void {
  try { storage?.setItem(COLLAPSED_SECTIONS_KEY, JSON.stringify([...keys].slice(-500))) } catch { /* The fold stays for this window only. */ }
}

/**
 * Sidebar order: newest conversation first, by latest-message time. A row
 * with no known time (a chat that has not spoken yet) leads; ties keep their
 * incoming order, so the list never shuffles between equal rows.
 */
export function sidebarOrder<T extends GroupableRow & { readonly activeAt?: number }>(
  rows: readonly T[],
  activity: Readonly<Record<string, number>> = {},
): T[] {
  const time = (row: T): number => activity[row.id] ?? row.activeAt ?? Number.POSITIVE_INFINITY
  return rows.map((row, index) => ({ row, index, at: time(row) }))
    .sort((a, b) => (b.at - a.at) || (a.index - b.index))
    .map(entry => entry.row)
}
