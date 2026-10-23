// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** In-memory composer drafts keyed by task, so switching tabs or tasks keeps what was typed. */
export interface DraftStore {
  get(key: string): string
  set(key: string, text: string): void
  clear(key: string): void
}

export function createDraftStore(limit = 50): DraftStore {
  const drafts = new Map<string, string>()
  return {
    get: key => drafts.get(key) ?? '',
    set(key, text) {
      if (text === '') {
        drafts.delete(key)
        return
      }
      drafts.delete(key)
      drafts.set(key, text)
      while (drafts.size > limit) {
        const oldest = drafts.keys().next().value
        if (oldest === undefined) break
        drafts.delete(oldest)
      }
    },
    clear: key => { drafts.delete(key) },
  }
}
