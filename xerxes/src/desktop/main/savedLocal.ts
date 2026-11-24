// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  cleanFolder,
  contextLabel,
  contextSessions,
  inFolder,
  latestSessionIn,
  sessionsFor,
  type ContextSession,
  type WorkspaceContext,
} from './contextNavigation.js'

const DEFAULT_REFRESH_MS = 15_000
const DEFAULT_FIRST_LOAD_TIMEOUT_MS = 1_500
const FIRST_LOCAL_CONTEXT_ID = -1_000_000

export interface SavedLocalPorts {
  readonly workspaces: () => readonly string[]
  readonly fetchSessions: () => Promise<{ readonly saved: unknown; readonly live: unknown }>
  readonly now?: () => number
  readonly refreshMs?: number
  readonly firstLoadTimeoutMs?: number
}

/**
 * Read-only local ("This Mac") sidebar contexts for remote SSH views when no
 * local workspace view is currently materialized. Mirrors SavedHostContexts so
 * local projects display their real session counts/rows and open their newest
 * task (or a chosen task) on click.
 */
export class SavedLocalContexts {
  private readonly idsByFolder = new Map<string, number>()
  private readonly foldersById = new Map<number, string>()
  private nextId = FIRST_LOCAL_CONTEXT_ID
  private cachedSessions: ContextSession[] = []
  private lastAttempt = -Infinity
  private loadedOnce = false
  private pending: Promise<void> | undefined
  private disposed = false

  constructor(private readonly ports: SavedLocalPorts) {}

  has(id: number): boolean {
    return this.foldersById.has(id)
  }

  /** Trigger a cached single-flight refresh; optionally wait briefly on first load so initial paint has sessions. */
  async refresh(options?: { readonly awaitInitial?: boolean }): Promise<void> {
    if (this.disposed) return
    const now = (this.ports.now ?? Date.now)()
    const ttl = this.ports.refreshMs ?? DEFAULT_REFRESH_MS
    if (!this.pending && now - this.lastAttempt >= ttl) {
      this.lastAttempt = now
      this.pending = (async () => {
        const { saved, live } = await this.ports.fetchSessions()
        if (this.disposed) return
        this.cachedSessions = contextSessions(saved, live)
        this.loadedOnce = true
      })()
        .catch(() => {
          // Never block an SSH view when the local daemon is unavailable.
        })
        .finally(() => {
          this.pending = undefined
        })
    }
    if (options?.awaitInitial && !this.loadedOnce && this.pending) {
      let timer: ReturnType<typeof setTimeout> | undefined
      const timeoutMs = Math.max(0, this.ports.firstLoadTimeoutMs ?? DEFAULT_FIRST_LOAD_TIMEOUT_MS)
      await Promise.race([
        this.pending,
        new Promise<void>(resolve => {
          timer = setTimeout(resolve, timeoutMs)
          timer.unref?.()
        }),
      ])
      if (timer !== undefined) clearTimeout(timer)
    }
  }

  /**
   * Synchronous snapshot of local "This Mac" folder contexts.
   * Folders already covered by an open local view are skipped (or stripped of
   * duplicate sessions) so the sidebar never shows duplicate sections.
   */
  contexts(options?: {
    readonly coveredFolders?: ReadonlySet<string>
    readonly includeSessions?: boolean
  }): WorkspaceContext[] {
    const includeSessions = options?.includeSessions !== false
    const covered = options?.coveredFolders
    const savedFolders = this.ports.workspaces().map(cleanFolder)
    const allFolders: string[] = [...new Set(savedFolders)]
    if (includeSessions) {
      for (const row of this.cachedSessions) {
        const cwd = cleanFolder(row.cwd)
        if (!allFolders.some(folder => inFolder(cwd, folder))) {
          allFolders.push(cwd)
        }
      }
    }
    const result: WorkspaceContext[] = []
    for (const folder of allFolders) {
      if (covered?.has(folder)) continue
      const id = this.idFor(folder)
      const folderSessions = includeSessions
        ? sessionsFor(this.cachedSessions, folder, allFolders)
        : []
      result.push({
        id,
        label: contextLabel('This Mac', folder),
        workspace: folder,
        remote: false,
        saved: true,
        sessions: folderSessions,
      })
    }
    return result
  }

  /** Resolve a click on a saved local folder or one of its sessions. */
  selection(id: number, sessionId?: unknown): { readonly directory: string; readonly sessionId?: string } {
    const folder = this.foldersById.get(id)
    if (!folder) throw new Error('Saved local workspace is unavailable')
    if (sessionId === undefined) {
      const latest = latestSessionIn(this.cachedSessions, folder)
      return latest === undefined ? { directory: folder } : { directory: folder, sessionId: latest }
    }
    if (typeof sessionId !== 'string' || !/^[a-zA-Z0-9_-]{1,256}$/.test(sessionId)) {
      throw new Error('Invalid local session id')
    }
    const row = this.cachedSessions.find(item => item.id === sessionId)
    return { directory: row ? cleanFolder(row.cwd) : folder, sessionId }
  }

  dispose(): void {
    this.disposed = true
  }

  private idFor(folder: string): number {
    let id = this.idsByFolder.get(folder)
    if (id === undefined) {
      id = this.nextId--
      this.idsByFolder.set(folder, id)
      this.foldersById.set(id, folder)
    }
    return id
  }
}
