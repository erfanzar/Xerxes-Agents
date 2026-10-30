// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Composer drafts keyed by task, so switching tabs or tasks keeps what was
 * typed. With a storage port (the browser's localStorage by default) drafts
 * also survive iOS evicting the home-screen app. Text only: images are never
 * persisted. Bounded by count (LRU) and by length per draft.
 */
export interface DraftStore {
  get(key: string): string
  set(key: string, text: string): void
  clear(key: string): void
}

/** The part of the Web Storage API the store uses; tests inject a fake. */
export interface DraftStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

export interface DraftStoreOptions {
  /** Explicit storage, `null` for memory only. Omitted: `globalThis.localStorage` when present. */
  readonly storage?: DraftStorage | null
  /** Longest draft kept, in characters. */
  readonly maxLength?: number
}

export const DRAFT_STORAGE_KEY = 'xerxes.drafts.v1'
export const DRAFT_MAX_LENGTH = 20_000

function browserStorage(): DraftStorage | null {
  try {
    const storage = (globalThis as { localStorage?: DraftStorage }).localStorage
    return storage ?? null
  } catch {
    // Reading localStorage throws in some private modes.
    return null
  }
}

function load(storage: DraftStorage | null, limit: number, maxLength: number): Map<string, string> {
  const drafts = new Map<string, string>()
  if (!storage) return drafts
  try {
    const raw = storage.getItem(DRAFT_STORAGE_KEY)
    if (!raw) return drafts
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) return drafts
    for (const entry of parsed) {
      if (!Array.isArray(entry) || typeof entry[0] !== 'string' || typeof entry[1] !== 'string' || entry[1] === '') continue
      drafts.set(entry[0], entry[1].slice(0, maxLength))
    }
  } catch {
    // An unreadable saved value is dropped, never guessed at.
    return new Map()
  }
  while (drafts.size > limit) drafts.delete(drafts.keys().next().value as string)
  return drafts
}

export function createDraftStore(limit = 50, options: DraftStoreOptions = {}): DraftStore {
  const storage = options.storage === undefined ? browserStorage() : options.storage
  const maxLength = options.maxLength ?? DRAFT_MAX_LENGTH
  const drafts = load(storage, limit, maxLength)
  const persist = (): void => {
    if (!storage) return
    try {
      if (drafts.size === 0) storage.removeItem(DRAFT_STORAGE_KEY)
      else storage.setItem(DRAFT_STORAGE_KEY, JSON.stringify([...drafts.entries()]))
    } catch {
      // Quota exceeded or storage disabled: the in-memory copy still works for this visit.
    }
  }
  return {
    get: key => drafts.get(key) ?? '',
    set(key, text) {
      const value = text.slice(0, maxLength)
      if (value === '') {
        if (drafts.delete(key)) persist()
        return
      }
      drafts.delete(key)
      drafts.set(key, value)
      while (drafts.size > limit) {
        const oldest = drafts.keys().next().value
        if (oldest === undefined) break
        drafts.delete(oldest)
      }
      persist()
    },
    clear: key => { if (drafts.delete(key)) persist() },
  }
}

/**
 * What stays in the composer once the Hub accepted `sent`: text typed while
 * the send was in flight is kept. If the person only appended, the appended
 * part stays; if they changed the sent text, everything they now have stays.
 */
export function draftAfterAccepted(current: string, sent: string): string {
  if (current === sent || current.trim() === sent.trim()) return ''
  if (current.startsWith(sent)) return current.slice(sent.length).replace(/^\s+/, '')
  return current
}
