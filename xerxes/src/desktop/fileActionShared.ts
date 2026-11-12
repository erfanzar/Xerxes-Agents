// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * "Open in default app" and "Show in Finder" for a workspace file the Files
 * panel shows. Shared by the bridge (renderer side) and the hosts (Electron
 * main, the VS Code extension), which validate the same request again.
 */

import { isAbsolutePath } from './chatMediaShared.js'

export const FILE_ACTION_CHANNEL = 'desktop:file-action'

export type FileAction = 'open' | 'reveal'

export interface FileActionRequest {
  readonly action: FileAction
  /** Workspace-relative path, as the Files panel names it. */
  readonly path: string
  /** The task's absolute folder the path resolves in. */
  readonly cwd: string
}

export type FileActionResult = { readonly ok: true } | { readonly ok: false; readonly error: string }

const MAX_PATH = 4096
const plainText = (value: unknown): value is string => typeof value === 'string' && value.length > 0 && value.length <= MAX_PATH && !/[\x00-\x1f\x7f]/.test(value)

/** The request, checked; or why it is refused. */
export function fileActionRequestOf(value: unknown): FileActionRequest | { readonly error: string } {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return { error: 'the file request is not valid' }
  const { action, path, cwd } = value as Record<string, unknown>
  if (action !== 'open' && action !== 'reveal') return { error: 'the file action is not valid' }
  if (!plainText(path)) return { error: 'the file path is not valid' }
  if (!plainText(cwd) || !isAbsolutePath(cwd)) return { error: 'the task folder is not an absolute path' }
  return { action, path, cwd }
}

/** A host's reply, as the renderer may trust it. */
export function asFileActionResult(value: unknown): FileActionResult {
  if (value && typeof value === 'object' && (value as { ok?: unknown }).ok === true) return { ok: true }
  const error = value && typeof value === 'object' ? (value as { error?: unknown }).error : undefined
  return { ok: false, error: typeof error === 'string' && error ? error.slice(0, 300) : 'the file could not be opened' }
}
