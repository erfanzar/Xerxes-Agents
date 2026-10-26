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
export const contextScope = (remote: RemoteTarget | null | undefined): string => remote ? `ssh:${remote.target}` : 'local'
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
