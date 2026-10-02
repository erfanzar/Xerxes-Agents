// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { mkdir, rename, rm } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'

/**
 * Durable map from a channel conversation key (`telegram:private:<id>`) to
 * the daemon session id that conversation continues.
 *
 * Channel keys are not session ids, so the runtime cannot resume them by
 * itself: without this map every restart, runtime update or idle eviction
 * silently started the conversation over.
 */
export interface ChannelSessionIndex {
  get(sessionKey: string): Promise<string | undefined>
  set(sessionKey: string, sessionId: string): Promise<void>
}

/** Raised when the persisted index exists but is not a valid index file. */
export class ChannelSessionIndexError extends Error {
  constructor(path: string, reason: string, options?: ErrorOptions) {
    super(`channel session index '${path}' ${reason}; fix or remove it`, options)
    this.name = new.target.name
  }
}

/**
 * JSON-file {@link ChannelSessionIndex}.
 *
 * The file is read once and cached. Writes are serialized and replace the
 * file atomically, so a crash mid-write keeps the previous index intact.
 */
export class JsonChannelSessionIndex implements ChannelSessionIndex {
  readonly path: string
  private entries: Promise<Map<string, string>> | undefined
  private writes: Promise<void> = Promise.resolve()

  constructor(path: string) {
    if (!path.trim()) throw new TypeError('channel session index path must not be empty')
    this.path = resolve(path)
  }

  async get(sessionKey: string): Promise<string | undefined> {
    return (await this.load()).get(sessionKey)
  }

  set(sessionKey: string, sessionId: string): Promise<void> {
    const write = this.writes.then(async () => {
      const entries = await this.load()
      if (entries.get(sessionKey) === sessionId) return
      entries.set(sessionKey, sessionId)
      await this.persist(entries)
    })
    this.writes = write.catch(() => undefined)
    return write
  }

  private load(): Promise<Map<string, string>> {
    if (this.entries) return this.entries
    const loading = readIndex(this.path)
    this.entries = loading
    // A failed read must be retried on the next call, not cached forever.
    loading.catch(() => {
      if (this.entries === loading) this.entries = undefined
    })
    return loading
  }

  private async persist(entries: ReadonlyMap<string, string>): Promise<void> {
    await mkdir(dirname(this.path), { recursive: true })
    const temporary = `${this.path}.${randomUUID()}.tmp`
    try {
      await Bun.write(temporary, JSON.stringify({ sessions: Object.fromEntries(entries) }, null, 2) + '\n')
      await rename(temporary, this.path)
    } catch (error) {
      await rm(temporary, { force: true })
      throw error
    }
  }
}

async function readIndex(path: string): Promise<Map<string, string>> {
  const file = Bun.file(path)
  if (!await file.exists()) return new Map()
  let parsed: unknown
  try {
    parsed = JSON.parse(await file.text())
  } catch (error) {
    throw new ChannelSessionIndexError(path, 'is not valid JSON', { cause: error })
  }
  const sessions = isRecord(parsed) ? parsed.sessions : undefined
  if (!isRecord(sessions)) throw new ChannelSessionIndexError(path, 'has no sessions object')
  const entries = new Map<string, string>()
  for (const [key, value] of Object.entries(sessions)) {
    if (typeof value !== 'string' || !value) {
      throw new ChannelSessionIndexError(path, `maps '${key}' to a value that is not a session id`)
    }
    entries.set(key, value)
  }
  return entries
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
