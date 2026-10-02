// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

const SESSION_ID = /^[a-zA-Z0-9_-]{1,256}$/

/**
 * The session one workspace surface (a window's page or a view) shows, as the
 * daemon last confirmed it. Every page load of the surface resumes this id.
 * It used to be handed out once: a reload the renderer started itself (the
 * error screen's "Reload window", View > Reload) then found no id and fell
 * back to the workspace-wide last-selected session, which another view or
 * window on the same folder overwrites, so the reloaded view showed that
 * other conversation, or a blank one, in place of its own.
 */
export class SurfaceSession {
  private session: string | null
  private resolved: string | null = null

  constructor(sessionId: string | null) { this.session = sessionId }

  /** What a (re)loading page of this surface resumes. */
  get current(): string | null { return this.session }
  /** The folder the daemon resolved the session to, once it has answered. */
  get cwd(): string | null { return this.resolved }

  /** The surface was rebound (another workspace or host); `sessionId` may be null for a new chat. */
  bind(sessionId: string | null): void {
    this.session = sessionId
    this.resolved = null
  }

  /** Follow an RPC result; true when the surface now shows a different session. */
  observe(method: string, result: Record<string, unknown>): boolean {
    if (method !== 'initialize' && method !== 'session.open') return false
    const session = result.session as Record<string, unknown> | undefined
    if (!session || typeof session.id !== 'string' || !SESSION_ID.test(session.id)) return false
    const cwd = typeof result.cwd === 'string' && result.cwd ? result.cwd : typeof session.cwd === 'string' && session.cwd ? session.cwd : null
    if (cwd) this.resolved = cwd
    if (session.id === this.session) return false
    this.session = session.id
    return true
  }
}
