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
}
export const contextScope = (remote: RemoteTarget | null | undefined): string => remote ? `ssh:${remote.target}` : 'local'
export function sameRemote(left: RemoteTarget | null | undefined, right: RemoteTarget): boolean {
  return left?.target === right.target && left.workspacePath === right.workspacePath
}
/** Keep remote/local session identities separate; live state wins within one endpoint. */
export function contextSessions(saved: unknown, live: unknown): ContextSession[] {
  const rows = new Map<string, ContextSession>()
  for (const value of [...(Array.isArray(saved) ? saved : []), ...(Array.isArray(live) ? live : [])]) {
    if (!value || typeof value !== 'object') continue
    const row = value as Record<string, unknown>
    const id = row.id ?? row.session_id
    const kind = row.kind ?? row.session_kind ?? 'main'
    if (kind !== 'main' || typeof id !== 'string' || !id || typeof row.cwd !== 'string') continue
    const previous = rows.get(id)
    rows.set(id, { id, cwd: row.cwd,
      title: typeof row.title === 'string' && row.title.trim() ? row.title : previous?.title ?? `Session ${id.slice(0, 8)}`,
      status: row.active_turn_id || row.status === 'working' || row.status === 'acting' ? 'working' : row.status === 'failed' ? 'failed' : 'idle',
    })
  }
  return [...rows.values()]
}
