// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Workspace grouping for the sidebar. A workspace is the folder a chat ran
 * in; the sidebar reads like a project tree — folder name, then its chats —
 * instead of one flat list across every project on the machine.
 */

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
