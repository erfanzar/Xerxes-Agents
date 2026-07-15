// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readdir } from 'node:fs/promises'

import {
  MAX_CONTEXT_FILE_BYTES,
  MarkdownAgentWorkspace,
  isDefaultWorkspaceFile,
} from './workspace.js'
import {
  WorkspaceFilesystemError,
  existingWorkspaceDirectory,
  inspectWorkspaceChild,
  normalizeWorkspacePath,
  readWorkspaceFile,
  requireWorkspaceFileSize,
  writeWorkspaceFile,
} from './workspaceFilesystem.js'

/** Canonical root-level Markdown files accepted from compatible agent workspaces. */
export const IMPORTABLE_FILES = ['AGENTS.md', 'SOUL.md', 'USER.md', 'MEMORY.md', 'TOOLS.md', 'IDENTITY.md'] as const

const LEGACY_DEFAULT_FILE_BYTES = 600

/** Raised when an external Markdown workspace cannot be safely imported. */
export class WorkspaceImportError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'WorkspaceImportError'
  }
}

export interface ImportResult {
  readonly conflicts: string[]
  readonly copied: string[]
  readonly skipped: string[]
  readonly source: string
  readonly target: string
}

export type WorkspaceImportResult = ImportResult

export interface ImportWorkspaceOptions {
  readonly dryRun?: boolean
  readonly overwrite?: boolean
  readonly targetWorkspace?: MarkdownAgentWorkspace
}

/**
 * Copy compatible workspace Markdown into a channel workspace.
 *
 * Imported text remains raw on disk to preserve the source layout; it is always passed
 * through the context prompt scanner later by {@link MarkdownAgentWorkspace.loadContext}.
 */
export async function importWorkspace(sourceDir: string, options: ImportWorkspaceOptions = {}): Promise<ImportResult> {
  try {
    return await importWorkspaceSafely(sourceDir, options)
  } catch (error) {
    if (error instanceof WorkspaceImportError) throw error
    const detail = error instanceof Error ? error.message : String(error)
    throw new WorkspaceImportError(`workspace import failed: ${detail}`)
  }
}

async function importWorkspaceSafely(sourceDir: string, options: ImportWorkspaceOptions): Promise<ImportResult> {
  const normalizedSource = normalizeWorkspacePath(sourceDir)
  const source = await existingWorkspaceDirectory(normalizedSource, 'workspace source directory')
  if (source === undefined) {
    throw new WorkspaceImportError(`workspace source directory not found: ${normalizedSource}`)
  }

  const workspace = options.targetWorkspace ?? new MarkdownAgentWorkspace()
  const dryRun = options.dryRun ?? false
  const overwrite = options.overwrite ?? false
  let targetRoot: string | undefined
  if (dryRun) {
    targetRoot = await existingWorkspaceDirectory(workspace.path, 'workspace target directory')
  } else {
    await workspace.ensure()
    targetRoot = await existingWorkspaceDirectory(workspace.path, 'workspace target directory')
    if (targetRoot === undefined) {
      throw new WorkspaceImportError(`workspace target directory was not created: ${workspace.path}`)
    }
  }

  const result: ImportResult = {
    source,
    target: workspace.path,
    copied: [],
    skipped: [],
    conflicts: [],
  }

  for (const name of IMPORTABLE_FILES) {
    const sourceEntry = await inspectWorkspaceChild(source, name, `source workspace file ${name}`)
    if (sourceEntry.kind !== 'file') {
      result.skipped.push(name)
      continue
    }
    const content = await importableContent(sourceEntry.file, `source workspace file ${name}`)
    if (targetRoot !== undefined) {
      const targetEntry = await inspectWorkspaceChild(targetRoot, name, `target workspace file ${name}`)
      if (targetEntry.kind === 'directory' || targetEntry.kind === 'other') {
        throw new WorkspaceFilesystemError(targetEntry.path, `target workspace file ${name} must be a regular file`)
      }
      if (targetEntry.kind === 'file' && !overwrite && !(await fileLooksDefault(targetEntry.file, name))) {
        result.conflicts.push(name)
        continue
      }
    }

    if (!dryRun) {
      if (targetRoot === undefined) {
        throw new WorkspaceImportError(`workspace target directory is unavailable: ${workspace.path}`)
      }
      await writeWorkspaceFile(targetRoot, name, content, `target workspace file ${name}`)
    }
    result.copied.push(name)
  }

  await importDailyNotes(source, targetRoot, result, { dryRun, overwrite, workspace })
  return result
}

async function importDailyNotes(
  sourceRoot: string,
  targetRoot: string | undefined,
  result: ImportResult,
  options: { readonly dryRun: boolean; readonly overwrite: boolean; readonly workspace: MarkdownAgentWorkspace },
): Promise<void> {
  const sourceMemory = await inspectWorkspaceChild(sourceRoot, 'memory', 'source workspace memory directory')
  if (sourceMemory.kind !== 'directory') return

  const targetMemory = await targetMemoryDirectory(targetRoot, options.dryRun)
  let entries
  try {
    entries = await readdir(sourceMemory.path, { withFileTypes: true })
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    throw new WorkspaceImportError(`cannot list source workspace memory directory: ${detail}`)
  }

  const names = entries
    .map(entry => entry.name)
    .filter(name => name.endsWith('.md'))
    .sort((left, right) => left.localeCompare(right))
  for (const name of names) {
    const sourceEntry = await inspectWorkspaceChild(sourceMemory.path, name, `source workspace memory file ${name}`)
    if (sourceEntry.kind !== 'file') continue
    const relativeName = `memory/${name}`
    const content = await importableContent(sourceEntry.file, `source workspace memory file ${name}`)

    if (targetMemory !== undefined) {
      const targetEntry = await inspectWorkspaceChild(targetMemory, name, `target workspace memory file ${name}`)
      if (targetEntry.kind === 'directory' || targetEntry.kind === 'other') {
        throw new WorkspaceFilesystemError(
          targetEntry.path,
          `target workspace memory file ${name} must be a regular file`,
        )
      }
      if (targetEntry.kind === 'file' && !options.overwrite) {
        result.conflicts.push(relativeName)
        continue
      }
    }

    if (!options.dryRun) {
      if (targetMemory === undefined) {
        throw new WorkspaceImportError(`workspace target memory directory is unavailable: ${options.workspace.path}`)
      }
      await writeWorkspaceFile(targetMemory, name, content, `target workspace memory file ${name}`)
    }
    result.copied.push(relativeName)
  }
}

async function targetMemoryDirectory(targetRoot: string | undefined, dryRun: boolean): Promise<string | undefined> {
  if (targetRoot === undefined) return undefined
  const targetMemory = await inspectWorkspaceChild(targetRoot, 'memory', 'target workspace memory directory')
  if (targetMemory.kind === 'missing' && dryRun) return undefined
  if (targetMemory.kind !== 'directory') {
    const path = targetMemory.kind === 'file'
      ? targetMemory.file.path
      : targetMemory.kind === 'missing'
        ? targetRoot
        : targetMemory.path
    throw new WorkspaceFilesystemError(path, 'target workspace memory directory must be a directory')
  }
  return targetMemory.path
}

async function fileLooksDefault(file: { readonly path: string; readonly size: number }, name: string): Promise<boolean> {
  if (file.size < LEGACY_DEFAULT_FILE_BYTES) return true
  if (file.size > MAX_CONTEXT_FILE_BYTES) return false
  return isDefaultWorkspaceFile(name, await readWorkspaceFile(file, `target workspace file ${name}`))
}

async function importableContent(
  file: { readonly path: string; readonly size: number },
  label: string,
): Promise<string> {
  requireWorkspaceFileSize(file, MAX_CONTEXT_FILE_BYTES, label)
  return readWorkspaceFile(file, label)
}
