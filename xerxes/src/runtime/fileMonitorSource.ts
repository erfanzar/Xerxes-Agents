// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { watch, type FSWatcher } from 'node:fs'
import { lstat, realpath, stat } from 'node:fs/promises'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

const MAX_PATH_CHARS = 4_096
const DEBOUNCE_MS = 50

export interface FileMonitorChange {
  readonly text: string
  readonly identity: string
}

export interface FileMonitorSource {
  open(
    workspace: string,
    path: string,
    onChange: (event: FileMonitorChange) => void,
    onError: (error: unknown) => void,
    signal?: AbortSignal,
  ): Promise<{ readonly path: string; readonly workspace: string; readonly close: () => void }>
}

interface FileState {
  readonly signature: string
  readonly size: number
  readonly mtimeMs: number
}

const nativeFileMonitorSource: FileMonitorSource = {
  async open(workspace, path, onChange, onError, signal) {
    if (signal?.aborted) throw signal.reason ?? new Error('File monitor was aborted before opening')
    const roots = await canonicalRoot(workspace)
    const target = await canonicalTarget(roots, path)
    const parent = dirname(target.path)
    const name = basename(target.path)
    const parentInfo = await stat(parent)
    const parentIdentity = `${parentInfo.dev}:${parentInfo.ino}`
    let state: FileState | undefined = await readValidatedState(roots.workspace, target.path)
    if (state === undefined) throw new Error(`Monitored file is not a regular file: ${path}`)
    const watchers: FSWatcher[] = []
    let poll: ReturnType<typeof setInterval> | undefined
    let timer: ReturnType<typeof setTimeout> | undefined
    let closed = false
    let pending = false
    let processing = false

    const fail = (error: unknown): void => {
      if (closed) return
      try { onError(error) } catch { /* Consumer callbacks cannot keep the watcher alive or escape as an async error. */ } finally { close() }
    }
    const emit = async (): Promise<void> => {
      if (closed || !pending || processing) return
      pending = false
      processing = true
      try {
        const currentParent = await stat(parent)
        if (`${currentParent.dev}:${currentParent.ino}` !== parentIdentity) {
          throw new Error('Monitored parent directory was replaced; closing file monitor')
        }
        const next = await readValidatedState(roots.workspace, target.path)
        if (closed) return
        const previous = state
        if (next === undefined) {
          if (previous !== undefined) {
            state = undefined
            onChange(eventFor('deleted', roots.workspace, target.path, previous))
          }
          return
        }
        const kind = previous === undefined ? 'recreated' : previous.signature === next.signature ? undefined : 'changed'
        state = next
        if (kind !== undefined) onChange(eventFor(kind, roots.workspace, target.path, next))
      } finally {
        processing = false
        if (!closed && pending && timer === undefined) {
          timer = setTimeout(() => { timer = undefined; void emit().catch(fail) }, DEBOUNCE_MS)
        }
      }
    }
    const schedule = (): void => {
      if (closed) return
      pending = true
      if (timer === undefined) {
        timer = setTimeout(() => {
          timer = undefined
          void emit().catch(fail)
        }, DEBOUNCE_MS)
      }
    }
    const close = (): void => {
      if (closed) return
      closed = true
      pending = false
      if (timer !== undefined) clearTimeout(timer)
      if (poll !== undefined) clearInterval(poll)
      timer = undefined
      for (const watcher of watchers) watcher.close()
      watchers.length = 0
      signal?.removeEventListener('abort', abort)
    }
    const abort = (): void => close()
    try {
      // Bun 1.3.12's Linux directory watcher can open unrelated Unix sockets
      // and fail with ENXIO. Poll only this file and its parent identity; never
      // enumerate or open siblings, and retain the same scope validation.
      if (process.platform === 'linux') {
        poll = setInterval(schedule, 250)
        poll.unref()
      } else {
      const onParentEvent = (_event: string, filename: string | Buffer | null): void => {
        if (closed) return
        if (filename === null || filename.toString() === name) schedule()
      }
      const parentWatcher = watch(parent, { persistent: false }, onParentEvent)
      parentWatcher.on('error', fail)
      watchers.push(parentWatcher)
      if (dirname(parent) !== parent) {
        const grandparentWatcher = watch(dirname(parent), { persistent: false }, (_event, filename) => {
          if (closed) return
          if (filename === null || filename.toString() === basename(parent)) schedule()
        })
        grandparentWatcher.on('error', fail)
        watchers.push(grandparentWatcher)
      }
      }
      signal?.addEventListener('abort', abort, { once: true })
      if (signal?.aborted) {
        close()
        throw signal.reason ?? new Error('File monitor was aborted while opening')
      }
      // Compare once after subscriptions are live so changes in the setup window are observed.
      schedule()
    } catch (error) {
      close()
      throw error
    }
    return Object.freeze({ path: target.path, workspace: roots.workspace, close })
  },
}

export { nativeFileMonitorSource }

async function canonicalRoot(value: string): Promise<{ readonly workspace: string }> {
  boundedPath(value, 'workspace')
  const workspace = await realpath(value)
  const info = await stat(workspace)
  if (!info.isDirectory()) throw new Error(`Workspace is not a directory: ${value}`)
  return { workspace }
}

async function canonicalTarget(
  roots: { readonly workspace: string },
  value: string,
): Promise<{ readonly path: string }> {
  boundedPath(value, 'path')
  const requested = isAbsolute(value) ? resolve(value) : resolve(roots.workspace, value)
  const parent = await realpath(dirname(requested))
  assertInside(roots.workspace, parent, 'Monitored path must stay inside the workspace')
  const lexical = join(parent, basename(requested))
  const info = await lstat(lexical)
  if (!info.isFile() && !info.isSymbolicLink()) throw new Error(`Monitored path is not a regular file: ${value}`)
  const canonical = await realpath(lexical)
  assertInside(roots.workspace, canonical, 'Monitored path resolves outside the workspace')
  const regular = await stat(canonical)
  if (!regular.isFile()) throw new Error(`Monitored path is not a regular file: ${value}`)
  // Resolve accepted symlinks once at attachment time; monitor the canonical file
  // and its parent so direct target changes remain observable.
  return { path: canonical }
}

async function readValidatedState(workspace: string, lexical: string): Promise<FileState | undefined> {
  try {
    const canonical = await realpath(lexical)
    assertInside(workspace, canonical, 'Monitored path resolves outside the workspace')
    return readState(canonical)
  } catch (error) {
    if (isMissing(error)) return undefined
    throw error
  }
}

async function readState(path: string): Promise<FileState | undefined> {
  try {
    const info = await stat(path)
    if (!info.isFile()) return undefined
    const mtime = typeof info.mtimeMs === 'number' ? info.mtimeMs : 0
    return { signature: `${info.dev}:${info.ino}:${info.size}:${mtime}`, size: info.size, mtimeMs: mtime }
  } catch (error) {
    if (isMissing(error)) return undefined
    throw error
  }
}

function eventFor(kind: 'changed' | 'deleted' | 'recreated', workspace: string, path: string, state: FileState): FileMonitorChange {
  const relativePath = relative(workspace, path) || basename(path)
  const text = JSON.stringify({ event: kind, path: relativePath, size: kind === 'deleted' ? null : state.size, mtime_ms: kind === 'deleted' ? null : state.mtimeMs })
  return { text, identity: `${workspace}\u0000${relativePath}\u0000${kind}\u0000${state.signature}` }
}

function boundedPath(value: string, field: string): void {
  if (typeof value !== 'string' || !value || value.length > MAX_PATH_CHARS || value.includes('\u0000') || value.includes('\n') || value.includes('\r')) {
    throw new Error(`${field} must be a bounded path without control characters`)
  }
  if (!isAbsolute(value) && value.split(/[\\/]+/).includes('..')) throw new Error(`${field} must not contain parent traversal`)
}

function assertInside(root: string, candidate: string, message: string): void {
  const rest = relative(root, candidate)
  if (isAbsolute(rest) || rest === '..' || rest.startsWith(`..${sep}`)) throw new Error(message)
}

function isMissing(error: unknown): boolean {
  return Boolean(error && typeof error === 'object' && 'code' in error && ((error as { code?: unknown }).code === 'ENOENT' || (error as { code?: unknown }).code === 'ENOTDIR'))
}
