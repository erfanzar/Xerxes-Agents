// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Pure Electron host performance & resource hygiene primitives:
 * - Safe Chromium switches (disable unused background networking/sync/update services)
 * - Startup window deduplication & active-first surface selection (avoids N-renderer launch storms)
 * - Live background WebContentsView LRU eviction & idle hibernation policy
 * - Bounded Map helper for long-lived main-process state
 * - Debounced idle HTTP/code cache trimmer (never clears cookies/localStorage/IndexedDB)
 */

import { cleanFolder } from './contextNavigation.js'
import type { SavedWindow } from './windowState.js'

/** Safe Chromium switches that eliminate background services without affecting app rendering or BrowserTab HTTP(S). */
export const CHROMIUM_PERFORMANCE_SWITCHES: ReadonlyArray<readonly [string] | readonly [string, string]> = [
  ['disable-background-networking'],
  ['disable-component-update'],
  ['disable-domain-reliability'],
  ['disable-sync'],
  ['no-first-run'],
]

export interface CommandLineLike {
  appendSwitch(switchName: string, value?: string): void
}

/** Register low-overhead Chromium switches before app.whenReady(). */
export function applyChromiumPerformanceSwitches(commandLine: CommandLineLike): void {
  for (const entry of CHROMIUM_PERFORMANCE_SWITCHES) {
    if (entry.length === 2) commandLine.appendSwitch(entry[0], entry[1])
    else commandLine.appendSwitch(entry[0])
  }
}

/** Canonical identity of a workspace surface for deduplicating accumulated saved views. */
export function surfaceIdentityKey(state: Pick<SavedWindow, 'workspace' | 'sessionId' | 'remote'>): string {
  const session = state.sessionId ?? ''
  if (state.remote) {
    return `ssh:${state.remote.target}:${cleanFolder(state.remote.workspacePath)}:${session}`
  }
  return `local:${state.workspace ? cleanFolder(state.workspace) : ''}:${session}`
}

export interface StartupWindowPlan {
  readonly group: string
  /** The single primary surface materialized immediately as the OS window's base page. */
  readonly primary: SavedWindow
  /** Deduplicated inactive surfaces kept as lightweight metadata (zero renderer/daemon cost at launch). */
  readonly deferred: readonly SavedWindow[]
}

/**
 * Group, deduplicate, and select one active startup surface per OS window group.
 * Restoring 10 accumulated views eagerly at launch spawns 10 Chromium renderer processes
 * (~900 MB) and triggers 10 simultaneous daemon session hydrations (~1.7 GB). Starting
 * only the active surface per OS window keeps launch instant while every session stays
 * reachable from the sidebar and saved hosts list.
 */
export function planStartupSurfaces(saved: readonly SavedWindow[]): StartupWindowPlan[] {
  if (saved.length === 0) return []
  const byGroup = new Map<string, SavedWindow[]>()
  for (const row of saved) {
    const group = row.windowGroup ?? 'legacy-workspaces'
    let list = byGroup.get(group)
    if (!list) {
      list = []
      byGroup.set(group, list)
    }
    list.push(row)
  }

  const plans: StartupWindowPlan[] = []
  for (const [group, rows] of byGroup) {
    // Deduplicate identical (endpoint, folder, session) entries, preserving active=true and latest bounds.
    const deduped: SavedWindow[] = []
    const indexByKey = new Map<string, number>()
    for (const row of rows) {
      const key = surfaceIdentityKey(row)
      const existingIndex = indexByKey.get(key)
      if (existingIndex === undefined) {
        indexByKey.set(key, deduped.length)
        deduped.push(row)
      } else {
        const existing = deduped[existingIndex]!
        deduped[existingIndex] = {
          ...row,
          active: existing.active === true || row.active === true,
        }
      }
    }

    // Prefer the last surface marked active; otherwise fall back to the most recent row.
    let primaryIndex = -1
    for (let i = deduped.length - 1; i >= 0; i -= 1) {
      if (deduped[i]!.active === true) {
        primaryIndex = i
        break
      }
    }
    if (primaryIndex === -1) primaryIndex = deduped.length - 1

    const primary: SavedWindow = { ...deduped[primaryIndex]!, active: true }
    const deferred: SavedWindow[] = []
    for (let i = 0; i < deduped.length; i += 1) {
      if (i !== primaryIndex) deferred.push({ ...deduped[i]!, active: false })
    }
    plans.push({ group, primary, deferred })
  }
  return plans
}

/** Keep at most this many idle warm background WebContentsViews per OS window. */
export const MAX_WARM_BACKGROUND_VIEWS = 2
/** Retire an idle, occluded background WebContentsView after 10 minutes unseen. */
export const IDLE_VIEW_HIBERNATE_MS = 10 * 60 * 1000

export interface BackgroundViewCandidate {
  readonly id: number
  /** Base window page cannot be independently closed without closing the OS window. */
  readonly isBasePage: boolean
  /** Currently shown surface in this window. */
  readonly isActive: boolean
  /** True while a turn is actively running in this view (never evict mid-turn). */
  readonly hasActiveTurn: boolean
  /** True while an SSH connect or voice dictation request is in flight. */
  readonly isBusy: boolean
  /** Epoch ms when this surface was last shown/activated. */
  readonly lastActiveAt: number
}

export interface ViewEvictionOptions {
  readonly maxWarmBackgroundViews?: number
  readonly idleHibernateMs?: number
}

/**
 * Decide which idle background WebContentsViews on a host window should be retired to reclaim
 * Chromium renderer RAM and daemon connections. Never returns the base page, the active view,
 * or any view with a running turn or in-flight host operation.
 */
export function pickEvictableBackgroundViews(
  candidates: readonly BackgroundViewCandidate[],
  now: number,
  options: ViewEvictionOptions = {},
): number[] {
  const maxWarm = Math.max(0, options.maxWarmBackgroundViews ?? MAX_WARM_BACKGROUND_VIEWS)
  const idleMs = Math.max(0, options.idleHibernateMs ?? IDLE_VIEW_HIBERNATE_MS)

  const backgroundViews = candidates.filter(c => !c.isBasePage && !c.isActive)
  const evictable = backgroundViews
    .filter(c => !c.hasActiveTurn && !c.isBusy)
    .sort((a, b) => a.lastActiveAt - b.lastActiveAt)

  const toEvict = new Set<number>()

  // 1. Retire views that have been idle & hidden past the hibernation threshold.
  if (idleMs > 0) {
    for (const candidate of evictable) {
      if (now - candidate.lastActiveAt >= idleMs) toEvict.add(candidate.id)
    }
  }

  // 2. Enforce the per-window warm background view cap (oldest LRU first).
  const remainingWarmCount = backgroundViews.length - toEvict.size
  if (remainingWarmCount > maxWarm) {
    let excess = remainingWarmCount - maxWarm
    for (const candidate of evictable) {
      if (excess <= 0) break
      if (toEvict.has(candidate.id)) continue
      toEvict.add(candidate.id)
      excess -= 1
    }
  }

  return [...toEvict]
}

/** Insert into a Map while keeping its size capped by evicting oldest keys first. */
export function setBoundedMapEntry<K, V>(map: Map<K, V>, key: K, value: V, maxEntries: number): void {
  const limit = Math.max(1, maxEntries)
  if (map.has(key)) map.delete(key)
  while (map.size >= limit) {
    const oldest = map.keys().next().value
    if (oldest === undefined) break
    map.delete(oldest)
  }
  map.set(key, value)
}

/**
 * Pick the HTTP(S) URL to preserve when suspending an inactive BrowserTab WebContentsView.
 * Falls back to `retryUrl` when the tab is suspended mid-load before `did-navigate` commits.
 */
export function resolveSuspendedBrowserUrl(currentUrl: string, retryUrl: string): string {
  if (/^https?:\/\//i.test(currentUrl)) return currentUrl
  if (/^https?:\/\//i.test(retryUrl)) return retryUrl
  return ''
}

export interface IdleCacheTrimmerOptions {
  /** Safe cache-only trim action (HTTP cache + code cache only; never cookies/localStorage/IndexedDB). */
  readonly trim: () => Promise<void> | void
  /** How long the app must remain hidden/minimized/locked before trimming caches. Default: 5 minutes. */
  readonly idleDelayMs?: number
  /** Minimum cooldown between consecutive cache trims. Default: 15 minutes. */
  readonly minIntervalMs?: number
  readonly now?: () => number
  readonly setTimer?: (fn: () => void, ms: number) => ReturnType<typeof setTimeout>
  readonly clearTimer?: (handle: ReturnType<typeof setTimeout>) => void
}

export interface IdleCacheTrimmer {
  markBackground(): void
  markForeground(): void
  /** Immediate trim (e.g. on system suspend/lock), still respecting minIntervalMs cooldown. */
  triggerNow(): void
  dispose(): void
}

export const DEFAULT_CACHE_TRIM_IDLE_MS = 5 * 60 * 1000
export const DEFAULT_CACHE_TRIM_COOLDOWN_MS = 15 * 60 * 1000

/**
 * Schedules periodic safe HTTP/code cache trimming after extended background idle or system lock,
 * with a cooldown so rapid window switches never thrash Chromium's caches.
 */
export function createIdleCacheTrimmer(options: IdleCacheTrimmerOptions): IdleCacheTrimmer {
  const idleDelayMs = options.idleDelayMs ?? DEFAULT_CACHE_TRIM_IDLE_MS
  const minIntervalMs = options.minIntervalMs ?? DEFAULT_CACHE_TRIM_COOLDOWN_MS
  const now = options.now ?? Date.now
  const schedule = options.setTimer ?? ((fn, ms) => {
    const timer = setTimeout(fn, ms)
    timer.unref?.()
    return timer
  })
  const cancel = options.clearTimer ?? (handle => clearTimeout(handle))

  let timer: ReturnType<typeof setTimeout> | undefined
  let lastTrimAt = Number.NEGATIVE_INFINITY
  let disposed = false
  let inFlight = false

  const cancelPending = (): void => {
    if (timer !== undefined) {
      cancel(timer)
      timer = undefined
    }
  }

  const runTrim = (): void => {
    cancelPending()
    if (disposed || inFlight) return
    const current = now()
    if (current - lastTrimAt < minIntervalMs) return
    lastTrimAt = current
    inFlight = true
    try {
      const result = options.trim()
      if (result && typeof (result as PromiseLike<void>).then === 'function') {
        void Promise.resolve(result)
          .catch(() => { /* Cache trimming is best-effort and must never surface fatal errors. */ })
          .finally(() => { inFlight = false })
      } else {
        inFlight = false
      }
    } catch {
      inFlight = false
    }
  }

  return {
    markBackground() {
      if (disposed || timer !== undefined) return
      const elapsed = now() - lastTrimAt
      const wait = Math.max(idleDelayMs, minIntervalMs - elapsed)
      timer = schedule(runTrim, wait)
    },
    markForeground() {
      cancelPending()
    },
    triggerNow() {
      runTrim()
    },
    dispose() {
      disposed = true
      cancelPending()
    },
  }
}
