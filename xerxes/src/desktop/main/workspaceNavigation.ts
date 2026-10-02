// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Navigation activates an existing view; it never reloads or rebinds its session. */
export interface WorkspaceView {
  readonly workspace: string | null
  /**
   * The folder the daemon resolved the view to (its git root, symlinks
   * resolved). Sidebar rows carry this form, not the folder the person
   * picked, so matching `workspace` alone opened another view each time a
   * subfolder or symlinked workspace was entered again.
   */
  readonly cwd?: string | null
  readonly sessionId: string | null
  readonly remote: unknown
  isDestroyed(): boolean
  isMinimized(): boolean
  restore(): void
  show(): void
  focus(): void
}

export function activateWorkspaceView(views: readonly WorkspaceView[], workspace: string, sessionId?: string): boolean {
  const view = views.find(candidate => !candidate.remote && (candidate.workspace === workspace || candidate.cwd === workspace) &&
    (!sessionId || candidate.sessionId === sessionId) && !candidate.isDestroyed())
  if (!view) return false
  if (view.isMinimized()) view.restore()
  view.show()
  view.focus()
  return true
}
