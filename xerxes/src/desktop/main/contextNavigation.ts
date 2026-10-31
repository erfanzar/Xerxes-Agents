// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { RemoteTarget } from './remote.js'

export interface ContextSession { id: string; cwd: string; title: string; status: string }
export interface WorkspaceContext {
  id: number
  label: string
  workspace: string
  remote: boolean
  sessions: ContextSession[]
  /** Saved hosts use stable negative ids; open views use Electron's positive ids. */
  saved?: boolean
  alias?: string
  workspacePath?: string
  availability?: 'online' | 'offline' | 'no-runtime'
  reason?: string
}
/** Storage scope (remembered selections): one per endpoint. Navigation uses `contextKey`. */
export const contextScope = (remote: RemoteTarget | null | undefined): string => remote ? `ssh:${remote.target}` : 'local'

/** `/a/b/` → `/a/b`; the root stays `/`. */
export function cleanFolder(path: string): string {
  const trimmed = path.replace(/\/+$/, '')
  return trimmed || '/'
}

/** Whether `cwd` is the project folder or inside it. */
export function inFolder(cwd: string, folder: string): boolean {
  const root = cleanFolder(folder)
  const path = cleanFolder(cwd)
  return root === '/' || path === root || path.startsWith(`${root}/`)
}

/**
 * Sidebar identity of a context: the endpoint AND its project folder. Keying
 * by host alone made two folders on one SSH host (EasyDeL and prismcore on
 * v5p-128-dev) one context, so whichever was listed first hid the other.
 */
export const contextKey = (remote: RemoteTarget | null | undefined): string =>
  remote ? `ssh:${remote.target}:${cleanFolder(remote.workspacePath)}` : 'local'

/** `v5p-128-dev · easydel-src`: which host, and which project on it. */
export function contextLabel(alias: string, folder: string): string {
  const name = cleanFolder(folder).split('/').filter(Boolean).at(-1)
  return name ? `${alias} · ${name}` : `${alias} · SSH`
}

/** The project folder a session belongs to: the deepest of `folders` containing its cwd, else the first one. */
export function ownerFolder(cwd: string, folders: readonly string[]): string | undefined {
  let best: string | undefined
  for (const folder of folders.map(cleanFolder)) {
    if (inFolder(cwd, folder) && (best === undefined || folder.length > best.length)) best = folder
  }
  return best ?? (folders[0] === undefined ? undefined : cleanFolder(folders[0]))
}

/**
 * The host's sessions that belong to `folder`. A host's daemon lists every
 * project on it, so with two saved folders each context showed both
 * projects' tasks. With a single folder nothing is filtered: tasks in other
 * folders of that host stay reachable from it.
 */
export function sessionsFor(rows: readonly ContextSession[], folder: string, folders: readonly string[]): ContextSession[] {
  // Host order first, so a task outside every saved folder lands in one
  // context (the host's first folder), not in all of them.
  const distinct = [...new Set([...folders, folder].map(cleanFolder))]
  if (distinct.length <= 1) return [...rows]
  const own = cleanFolder(folder)
  return rows.filter(row => ownerFolder(row.cwd, distinct) === own)
}

/** The newest task of a folder (rows arrive newest first), to open instead of a blank chat. */
export function latestSessionIn(rows: readonly ContextSession[], folder: string): string | undefined {
  return rows.find(row => inFolder(row.cwd, folder))?.id
}
export function sameRemote(left: RemoteTarget | null | undefined, right: RemoteTarget): boolean {
  return left?.target === right.target && left.workspacePath === right.workspacePath
}
/** Keep remote/local session identities separate; live state wins within one endpoint. */
export function contextSessions(saved: unknown, live: unknown): ContextSession[] {
  const rows = new Map<string, ContextSession>()
  for (const value of [...(Array.isArray(saved) ? saved.slice(0, 200) : []), ...(Array.isArray(live) ? live.slice(0, 200) : [])]) {
    if (!value || typeof value !== 'object') continue
    const row = value as Record<string, unknown>
    const id = row.id ?? row.session_id
    const kind = row.kind ?? row.session_kind ?? 'main'
    if (kind !== 'main' || typeof id !== 'string' || !/^[a-zA-Z0-9_-]{1,256}$/.test(id) ||
        typeof row.cwd !== 'string' || !row.cwd.startsWith('/') || row.cwd.length > 4096 || /[\x00-\x1f\x7f]/.test(row.cwd)) continue
    const previous = rows.get(id)
    const title = typeof row.title === 'string' ? row.title.replace(/[\x00-\x1f\x7f]/g, ' ').slice(0, 512).trim() : ''
    rows.set(id, { id, cwd: row.cwd,
      title: title || previous?.title || `Session ${id.slice(0, 8)}`,
      status: row.active_turn_id || row.status === 'working' || row.status === 'acting' ? 'working' : row.status === 'failed' ? 'failed' : 'idle',
    })
  }
  return [...rows.values()]
}
