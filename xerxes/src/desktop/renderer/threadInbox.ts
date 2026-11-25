// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Sidebar thread organization state: Inbox vs Folders view,
 * per-project scope filtering, Pinned / Active / Snoozed / Settled partitions,
 * deterministic project monograms, and undoable sidebar actions.
 */

import { useCallback, useEffect, useState, useSyncExternalStore } from 'react'

export const THREAD_INBOX_KEY = 'xerxes.desktop.threadInbox.v2'
const LEGACY_THREAD_INBOX_KEY = 'xerxes.desktop.threadInbox.v1'

export type SidebarViewMode = 'inbox' | 'folders'

export const PROJECT_MONOGRAM_COLORS = [
  'blue',
  'indigo',
  'violet',
  'purple',
  'pink',
  'rose',
  'orange',
  'amber',
  'emerald',
  'teal',
  'cyan',
  'sky',
] as const

export type ProjectMonogramColor = typeof PROJECT_MONOGRAM_COLORS[number]

export interface ThreadInboxState {
  readonly viewMode: SidebarViewMode
  readonly projectScope: readonly string[]
  /** Workspace directories explicitly removed from the sidebar by the user (re-added automatically if opened again). */
  readonly hiddenWorkspaces: readonly string[]
  readonly pinnedIds: readonly string[]
  readonly settledIds: readonly string[]
  /** Thread ID → wake timestamp (epoch ms). Automatically expires once `now >= wakeAt`. */
  readonly snoozed: Readonly<Record<string, number>>
  readonly snoozedOpen: boolean
  readonly settledOpen: boolean
}

export interface InboxUndoEntry {
  readonly threadId: string
  readonly title: string
  readonly verb: 'Settled' | 'Restored' | 'Snoozed' | 'Woke' | 'Pinned' | 'Unpinned' | 'Removed'
  readonly previous: ThreadInboxState
}

const DEFAULT_INBOX_STATE: ThreadInboxState = {
  viewMode: 'folders',
  projectScope: [],
  hiddenWorkspaces: [],
  pinnedIds: [],
  settledIds: [],
  snoozed: {},
  snoozedOpen: false,
  settledOpen: false,
}

interface StorageLike {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
}

function defaultStorage(): StorageLike | undefined {
  try {
    return (globalThis as { localStorage?: StorageLike }).localStorage
  } catch {
    return undefined
  }
}

function uniqueStrings(value: unknown, max = 500): string[] {
  if (!Array.isArray(value)) return []
  const seen = new Set<string>()
  const out: string[] = []
  for (const item of value) {
    if (typeof item !== 'string' || !item || item.length > 512 || seen.has(item)) continue
    seen.add(item)
    out.push(item)
    if (out.length >= max) break
  }
  return out
}

export function normalizeWorkspacePath(cwd: string): string {
  const trimmed = cwd.trim().replace(/\/+$/, '')
  return trimmed || (cwd.trim().startsWith('/') ? '/' : '')
}

export function parseThreadInboxState(value: unknown): ThreadInboxState {
  const record = value && typeof value === 'object' ? (value as Record<string, unknown>) : {}
  const viewMode: SidebarViewMode = record.viewMode === 'inbox' ? 'inbox' : 'folders'
  const projectScope = uniqueStrings(record.projectScope, 200)
  const hiddenWorkspaces = uniqueStrings(
    Array.isArray(record.hiddenWorkspaces)
      ? record.hiddenWorkspaces.map(item => (typeof item === 'string' ? normalizeWorkspacePath(item) : ''))
      : [],
    200,
  )
  const pinnedIds = uniqueStrings(record.pinnedIds, 200)
  const settledIds = uniqueStrings(record.settledIds, 500)
  const snoozed: Record<string, number> = {}
  if (record.snoozed && typeof record.snoozed === 'object' && !Array.isArray(record.snoozed)) {
    let count = 0
    for (const [id, wakeAt] of Object.entries(record.snoozed as Record<string, unknown>)) {
      if (!id || id.length > 256 || typeof wakeAt !== 'number' || !Number.isFinite(wakeAt) || wakeAt <= 0) continue
      snoozed[id] = wakeAt
      count += 1
      if (count >= 500) break
    }
  }
  return {
    viewMode,
    projectScope,
    hiddenWorkspaces,
    pinnedIds,
    settledIds,
    snoozed,
    snoozedOpen: record.snoozedOpen === true,
    settledOpen: record.settledOpen === true,
  }
}

export function loadThreadInboxState(storage: StorageLike | undefined = defaultStorage()): ThreadInboxState {
  try {
    const raw = storage?.getItem(THREAD_INBOX_KEY)
    if (raw) return parseThreadInboxState(JSON.parse(raw))
    // Migrate v1 user pins/snoozed/settled while upgrading untouched defaults to Machine + Project folders.
    const legacy = storage?.getItem(LEGACY_THREAD_INBOX_KEY)
    if (legacy) {
      const parsed = JSON.parse(legacy) as Record<string, unknown>
      return parseThreadInboxState({ ...parsed, viewMode: 'folders' })
    }
    return DEFAULT_INBOX_STATE
  } catch {
    return DEFAULT_INBOX_STATE
  }
}

export function saveThreadInboxState(state: ThreadInboxState, storage: StorageLike | undefined = defaultStorage()): void {
  try {
    storage?.setItem(THREAD_INBOX_KEY, JSON.stringify(state))
  } catch {
    /* In-memory state stays active when storage is unavailable. */
  }
}

// ── Shared reactive store so Sidebar, SessionMenu, Header, and Settings stay in sync ──

let cachedState: ThreadInboxState | null = null
const listeners = new Set<() => void>()

function getState(): ThreadInboxState {
  if (!cachedState) cachedState = loadThreadInboxState()
  return cachedState
}

function emit(): void {
  for (const listener of listeners) listener()
}

export function setThreadInboxState(updater: (current: ThreadInboxState) => ThreadInboxState): ThreadInboxState {
  const next = parseThreadInboxState(updater(getState()))
  cachedState = next
  saveThreadInboxState(next)
  emit()
  return next
}

export function setSidebarViewMode(viewMode: SidebarViewMode): void {
  setThreadInboxState(current => ({ ...current, viewMode }))
}

export function subscribeThreadInbox(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

export function resetThreadInboxCacheForTests(): void {
  cachedState = null
}

// ── Pure inbox mutations ──

export function isSnoozedActive(snoozed: Readonly<Record<string, number>>, id: string, now: number = Date.now()): boolean {
  const wakeAt = snoozed[id]
  return typeof wakeAt === 'number' && wakeAt > now
}

export function togglePinInState(state: ThreadInboxState, id: string): ThreadInboxState {
  const already = state.pinnedIds.includes(id)
  if (already) {
    return { ...state, pinnedIds: state.pinnedIds.filter(item => item !== id) }
  }
  const nextSnoozed = { ...state.snoozed }
  delete nextSnoozed[id]
  return {
    ...state,
    pinnedIds: [id, ...state.pinnedIds],
    settledIds: state.settledIds.filter(item => item !== id),
    snoozed: nextSnoozed,
  }
}

export function setSettledInState(state: ThreadInboxState, id: string, settled: boolean): ThreadInboxState {
  if (settled) {
    const nextSnoozed = { ...state.snoozed }
    delete nextSnoozed[id]
    return {
      ...state,
      pinnedIds: state.pinnedIds.filter(item => item !== id),
      settledIds: [id, ...state.settledIds.filter(item => item !== id)],
      snoozed: nextSnoozed,
    }
  }
  return {
    ...state,
    settledIds: state.settledIds.filter(item => item !== id),
  }
}

export function hideWorkspaceInState(state: ThreadInboxState, cwd: string): ThreadInboxState {
  const clean = normalizeWorkspacePath(cwd)
  if (!clean) return state
  return {
    ...state,
    projectScope: state.projectScope.filter(item => normalizeWorkspacePath(item) !== clean),
    hiddenWorkspaces: state.hiddenWorkspaces.includes(clean)
      ? state.hiddenWorkspaces
      : [...state.hiddenWorkspaces, clean],
  }
}

export function unhideWorkspaceInState(state: ThreadInboxState, cwd: string): ThreadInboxState {
  const clean = normalizeWorkspacePath(cwd)
  if (!clean || !state.hiddenWorkspaces.includes(clean)) return state
  return {
    ...state,
    hiddenWorkspaces: state.hiddenWorkspaces.filter(item => item !== clean),
  }
}

export function setSnoozedInState(state: ThreadInboxState, id: string, wakeAt: number | null): ThreadInboxState {
  const nextSnoozed = { ...state.snoozed }
  if (wakeAt === null) {
    delete nextSnoozed[id]
    return { ...state, snoozed: nextSnoozed }
  }
  nextSnoozed[id] = wakeAt
  return {
    ...state,
    pinnedIds: state.pinnedIds.filter(item => item !== id),
    settledIds: state.settledIds.filter(item => item !== id),
    snoozed: nextSnoozed,
  }
}

export interface SnoozePreset {
  readonly id: string
  readonly label: string
  readonly detail: string
  readonly wakeAt: number
}

export function snoozePresets(now: number = Date.now()): readonly SnoozePreset[] {
  const oneHour = now + 60 * 60_000
  const threeHours = now + 3 * 60 * 60_000
  const tomorrow = new Date(now)
  tomorrow.setDate(tomorrow.getDate() + 1)
  tomorrow.setHours(9, 0, 0, 0)
  const nextWeek = new Date(now)
  nextWeek.setDate(nextWeek.getDate() + 7)
  nextWeek.setHours(9, 0, 0, 0)
  return [
    { id: '1h', label: '1 hour', detail: formatClockTime(oneHour), wakeAt: oneHour },
    { id: '3h', label: '3 hours', detail: formatClockTime(threeHours), wakeAt: threeHours },
    { id: 'tomorrow', label: 'Tomorrow', detail: '9:00 AM', wakeAt: tomorrow.getTime() },
    { id: 'week', label: 'Next week', detail: '7 days', wakeAt: nextWeek.getTime() },
  ]
}

function formatClockTime(ms: number): string {
  try {
    return new Date(ms).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  } catch {
    return ''
  }
}

export function formatSnoozeRemaining(wakeAt: number, now: number = Date.now()): string {
  const diff = wakeAt - now
  if (diff <= 0) return 'now'
  const minutes = Math.ceil(diff / 60_000)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h`
  const days = Math.round(hours / 24)
  return `${days}d`
}

export interface InboxPartition<T> {
  readonly pinned: readonly T[]
  readonly active: readonly T[]
  readonly snoozed: readonly T[]
  readonly settled: readonly T[]
}

/**
 * Partition thread rows into 4 Inbox buckets:
 * Pinned → Active → Snoozed → Settled.
 * Active live turns automatically surface out of Settled so running work is never hidden.
 */
export function partitionInboxThreads<T extends { readonly id: string; readonly status?: string }>(
  rows: readonly T[],
  state: ThreadInboxState,
  now: number = Date.now(),
): InboxPartition<T> {
  const pinOrder = new Map<string, number>()
  state.pinnedIds.forEach((id, index) => pinOrder.set(id, index))
  const settledSet = new Set(state.settledIds)

  const pinned: T[] = []
  const active: T[] = []
  const snoozed: T[] = []
  const settled: T[] = []

  for (const row of rows) {
    const running = row.status === 'working' || row.status === 'running'
    if (pinOrder.has(row.id)) {
      pinned.push(row)
    } else if (!running && isSnoozedActive(state.snoozed, row.id, now)) {
      snoozed.push(row)
    } else if (!running && settledSet.has(row.id)) {
      settled.push(row)
    } else {
      active.push(row)
    }
  }

  pinned.sort((a, b) => (pinOrder.get(a.id) ?? 0) - (pinOrder.get(b.id) ?? 0))
  return { pinned, active, snoozed, settled }
}

/**
 * Deterministic 1–2 letter project monogram & color token
 * (e.g. `Xerxes-Agents` → `XA`, `workspace` → `WO`).
 */
export function monogramForProject(
  name: string,
  seed?: string,
): { readonly text: string; readonly color: ProjectMonogramColor } {
  const clean = name.trim() || 'W'
  const words = clean
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .split(/[^A-Za-z0-9]+/)
    .filter(Boolean)

  let text: string
  if (words.length >= 2) {
    text = `${words[0]![0]!}${words[1]![0]!}`.toUpperCase()
  } else {
    const single = words[0] ?? clean
    const digit = single.match(/[0-9]/)
    text = digit && single.length >= 2 && single[0] !== digit[0]
      ? `${single[0]!}${digit[0]!}`.toUpperCase()
      : single.slice(0, 2).toUpperCase()
  }

  const hashSource = (seed || clean).toLowerCase()
  let hash = 2166136261
  for (let i = 0; i < hashSource.length; i += 1) {
    hash ^= hashSource.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  const color = PROJECT_MONOGRAM_COLORS[Math.abs(hash) % PROJECT_MONOGRAM_COLORS.length]!
  return { text, color }
}

/**
 * React hook returning reactive ThreadInboxState + undoable actions.
 */
export function useThreadInbox(): {
  readonly state: ThreadInboxState
  readonly undo: InboxUndoEntry | null
  readonly dismissUndo: () => void
  readonly applyUndo: () => void
  readonly setViewMode: (mode: SidebarViewMode) => void
  readonly toggleProjectScope: (cwd: string) => void
  readonly clearProjectScope: () => void
  readonly togglePin: (id: string, title: string) => void
  readonly settle: (id: string, title: string, settled: boolean) => void
  readonly snooze: (id: string, title: string, wakeAt: number | null) => void
  readonly toggleShelf: (shelf: 'snoozed' | 'settled') => void
  readonly hideWorkspace: (cwd: string, label: string) => void
  readonly unhideWorkspace: (cwd: string) => void
} {
  const state = useSyncExternalStore(subscribeThreadInbox, getState, () => DEFAULT_INBOX_STATE)
  const [undo, setUndo] = useState<InboxUndoEntry | null>(null)

  useEffect(() => {
    if (typeof window === 'undefined') return
    const onStorage = (event: StorageEvent): void => {
      if (event.key === null || event.key === THREAD_INBOX_KEY) {
        cachedState = loadThreadInboxState()
        emit()
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])

  // Auto-dismiss the undo toast after 7 seconds.
  useEffect(() => {
    if (!undo) return
    const timer = setTimeout(() => setUndo(null), 7000)
    return () => clearTimeout(timer)
  }, [undo])

  const setViewMode = useCallback((viewMode: SidebarViewMode) => {
    setThreadInboxState(current => ({ ...current, viewMode }))
  }, [])

  const toggleProjectScope = useCallback((cwd: string) => {
    setThreadInboxState(current => {
      const exists = current.projectScope.includes(cwd)
      return {
        ...current,
        projectScope: exists
          ? current.projectScope.filter(item => item !== cwd)
          : [...current.projectScope, cwd],
      }
    })
  }, [])

  const clearProjectScope = useCallback(() => {
    setThreadInboxState(current => ({ ...current, projectScope: [] }))
  }, [])

  const togglePin = useCallback((id: string, title: string) => {
    const previous = getState()
    const wasPinned = previous.pinnedIds.includes(id)
    setThreadInboxState(current => togglePinInState(current, id))
    setUndo({ threadId: id, title: title || 'Thread', verb: wasPinned ? 'Unpinned' : 'Pinned', previous })
  }, [])

  const settle = useCallback((id: string, title: string, settled: boolean) => {
    const previous = getState()
    setThreadInboxState(current => setSettledInState(current, id, settled))
    setUndo({ threadId: id, title: title || 'Thread', verb: settled ? 'Settled' : 'Restored', previous })
  }, [])

  const snooze = useCallback((id: string, title: string, wakeAt: number | null) => {
    const previous = getState()
    setThreadInboxState(current => setSnoozedInState(current, id, wakeAt))
    setUndo({ threadId: id, title: title || 'Thread', verb: wakeAt === null ? 'Woke' : 'Snoozed', previous })
  }, [])

  const toggleShelf = useCallback((shelf: 'snoozed' | 'settled') => {
    setThreadInboxState(current =>
      shelf === 'snoozed'
        ? { ...current, snoozedOpen: !current.snoozedOpen }
        : { ...current, settledOpen: !current.settledOpen },
    )
  }, [])

  const hideWorkspace = useCallback((cwd: string, label: string) => {
    const previous = getState()
    setThreadInboxState(current => hideWorkspaceInState(current, cwd))
    setUndo({ threadId: cwd, title: label || cwd, verb: 'Removed', previous })
  }, [])

  const unhideWorkspace = useCallback((cwd: string) => {
    setThreadInboxState(current => unhideWorkspaceInState(current, cwd))
  }, [])

  const applyUndo = useCallback(() => {
    setUndo(current => {
      if (!current) return null
      setThreadInboxState(() => current.previous)
      return null
    })
  }, [])

  const dismissUndo = useCallback(() => setUndo(null), [])

  return {
    state,
    undo,
    dismissUndo,
    applyUndo,
    setViewMode,
    toggleProjectScope,
    clearProjectScope,
    togglePin,
    settle,
    snooze,
    toggleShelf,
    hideWorkspace,
    unhideWorkspace,
  }
}
