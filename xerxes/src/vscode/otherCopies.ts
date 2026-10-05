// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** An installed extension as VS Code reports it: its ID and its parsed package.json. */
export interface InstalledExtension {
  readonly id: string
  readonly packageJSON: unknown
}

/**
 * IDs of other installed extensions that contribute the same view — another
 * copy of Xerxes under an earlier publisher ID or a side-loaded build, whose
 * title-bar buttons would otherwise show twice.
 */
export function otherCopies(self: string, view: string, installed: readonly InstalledExtension[]): string[] {
  return installed.filter(entry => entry.id.toLowerCase() !== self.toLowerCase() && contributesView(entry.packageJSON, view)).map(entry => entry.id)
}

function contributesView(manifest: unknown, view: string): boolean {
  if (!manifest || typeof manifest !== 'object') return false
  const views = (manifest as { contributes?: { views?: unknown } }).contributes?.views
  if (!views || typeof views !== 'object') return false
  return Object.values(views).some(list => Array.isArray(list) && list.some(entry => Boolean(entry) && typeof entry === 'object' && (entry as { id?: unknown }).id === view))
}
