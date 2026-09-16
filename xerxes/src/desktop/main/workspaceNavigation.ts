// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Navigation activates an existing view; it never reloads or rebinds its session. */
export interface WorkspaceView {
  readonly workspace: string | null
  readonly sessionId: string | null
  readonly remote: unknown
  isDestroyed(): boolean
  isMinimized(): boolean
  restore(): void
  show(): void
  focus(): void
}

export function activateWorkspaceView(views: readonly WorkspaceView[], workspace: string, sessionId?: string): boolean {
  const view = views.find(candidate => !candidate.remote && candidate.workspace === workspace &&
    (!sessionId || candidate.sessionId === sessionId) && !candidate.isDestroyed())
  if (!view) return false
  if (view.isMinimized()) view.restore()
  view.show()
  view.focus()
  return true
}
