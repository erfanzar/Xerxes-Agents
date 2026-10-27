// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { constants } from 'node:fs'
import { mkdir, open, readFile, rename, rm } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { REPO_KEY_PATTERN } from './sharedLessonWire.js'

/** A replacement never follows the destination symlink, nor exposes partial JSON. */
export async function atomicLessonWrite(path: string, text: string, mode = 0o600): Promise<void> {
  const temporary = join(dirname(path), `.lesson-sharing-${crypto.randomUUID()}.tmp`)
  const handle = await open(temporary, 'wx', mode)
  try {
    await handle.writeFile(text, 'utf8')
    await handle.sync()
    await handle.close()
    await rename(temporary, path)
  } finally {
    await handle.close()
    await rm(temporary, { force: true }) // Only our exclusive-created temporary file.
  }
}

/** Bound bytes and reject links with the opened descriptor, not a prior lstat. */
export async function readLessonFile(path: string): Promise<string> {
  const handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
  try {
    const stat = await handle.stat()
    if (!stat.isFile() || stat.size > 64 * 1024) throw new Error('not a regular file or exceeds 64 KiB')
    const buffer = Buffer.alloc(64 * 1024 + 1)
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0)
    if (bytesRead > 64 * 1024) throw new Error('exceeds 64 KiB')
    return buffer.subarray(0, bytesRead).toString('utf8')
  } finally { await handle.close() }
}

interface SharingSettings { repos: Record<string, { enabled_at: string }> }
const settingsQueues = new Map<string, Promise<unknown>>()

export class LessonSharingSettings {
  constructor(readonly path: string) {}
  private async load(): Promise<SharingSettings> {
    let text: string
    try { text = await readFile(this.path, 'utf8') } catch (error) { if (isMissingLessonFile(error)) return { repos: {} }; throw error }
    const value: unknown = JSON.parse(text)
    if (!object(value) || !object(value.repos) || Object.keys(value).some(key => key !== 'repos')) throw new Error('Invalid local lesson-sharing settings; refusing repository sharing.')
    const repos: SharingSettings['repos'] = {}
    for (const [key, row] of Object.entries(value.repos)) {
      if (!REPO_KEY_PATTERN.test(key) || !object(row) || Object.keys(row).some(key => key !== 'enabled_at') || typeof row.enabled_at !== 'string' || !Number.isFinite(Date.parse(row.enabled_at))) throw new Error('Invalid local lesson-sharing repository opt-in.')
      repos[key] = { enabled_at: row.enabled_at }
    }
    return { repos }
  }
  async enabled(repo: string): Promise<boolean> { return Object.hasOwn((await this.load()).repos, repo) }
  async enable(repo: string): Promise<void> {
    if (!REPO_KEY_PATTERN.test(repo)) throw new Error('Cannot opt in an invalid repository identity.')
    const pending = (settingsQueues.get(this.path) ?? Promise.resolve()).then(async () => {
      const settings = await this.load()
      if (Object.hasOwn(settings.repos, repo)) return
      settings.repos[repo] = { enabled_at: new Date().toISOString() }
      await mkdir(dirname(this.path), { recursive: true, mode: 0o700 })
      await atomicLessonWrite(this.path, JSON.stringify(settings, null, 2) + '\n')
    })
    settingsQueues.set(this.path, pending.catch(() => {}))
    await pending
  }
}
function object(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value) }
export function isMissingLessonFile(error: unknown): boolean { return error instanceof Error && 'code' in error && error.code === 'ENOENT' }
