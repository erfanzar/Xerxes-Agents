// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

type DraftStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>
const memory = new Map<string, string>()

export function draftKey(workspace: string, session: string): string {
  return `xerxes.desktop.draft.v1:${JSON.stringify([workspace, session])}`
}

function windowStorage(): DraftStorage | undefined {
  try { return window.sessionStorage } catch { return undefined }
}

/** Window-scoped drafts survive renderer reloads without becoming shared history. */
export function readDraft(key: string, storage = windowStorage()): string {
  if (memory.has(key)) return memory.get(key)!
  try { return storage?.getItem(key) ?? '' }
  catch { return memory.get(key) ?? '' }
}

export function writeDraft(key: string, text: string, storage = windowStorage()): void {
  memory.set(key, text)
  try {
    if (text) storage?.setItem(key, text)
    else storage?.removeItem(key)
  } catch {
    // A full or unavailable session store still retains the draft in this renderer.
  }
}
