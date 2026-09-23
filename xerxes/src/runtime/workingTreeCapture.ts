// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { realpath, unlink } from 'node:fs/promises'
import { join } from 'node:path'

export type CaptureGit = (args: readonly string[], directory: string, environment?: Record<string, string>, raw?: boolean) => Promise<string>

/** Snapshot nested repositories as ordinary files, never share their Git directories. */
export async function captureWorkingTree(options: {
  git: CaptureGit; directory: string; indexDirectory: string; base: string
  importObjects: (directory: string, tree: string, destination: string) => Promise<void>
  depth?: number
}): Promise<{ tree: string; flattened: boolean }> {
  const { git, directory, indexDirectory, base, importObjects } = options
  const depth = options.depth ?? 0
  if (depth > 16) throw new Error('Nested repository capture exceeds 16 levels')
  const index = join(indexDirectory, crypto.randomUUID() + '.capture-index')
  const environment = { GIT_INDEX_FILE: index }
  try {
    await git(['read-tree', base], directory, environment)
    await git(['add', '-A', '--', '.'], directory, environment)
    const snapshot = await git(['write-tree'], directory, environment)
    const entries = await git(['ls-tree', '-r', '-z', snapshot], directory, {}, true)
    const links = entries.split('\0').filter(entry => entry.startsWith('160000 ')).map(entry => entry.slice(entry.indexOf('\t') + 1))
    // Child dirty files are verified by their own temporary index below.
    await git(['diff', '--quiet', '--ignore-submodules=dirty', '--'], directory, environment)
    if (await git(['ls-files', '--others', '--exclude-standard'], directory, environment)) throw new Error('Working tree changed during capture; retry when edits settle')
    for (const path of links) {
      const nested = join(directory, path)
      if (await realpath(nested) !== nested || await git(['rev-parse', '--show-toplevel'], nested) !== nested) throw new Error(`Initialize the nested repository at ${path} before capturing this workspace`)
      const childBase = await git(['rev-parse', 'HEAD'], nested)
      const child = await captureWorkingTree({ ...options, directory: nested, base: childBase, depth: depth + 1 })
      await importObjects(nested, child.tree, directory)
      await git(['update-index', '--force-remove', '--', path], directory, environment)
      await git(['read-tree', `--prefix=${path}/`, child.tree], directory, environment)
    }
    if (await git(['rev-parse', 'HEAD'], directory) !== base) throw new Error('HEAD changed during working-tree capture; retry when edits settle')
    const tree = await git(['write-tree'], directory, environment)
    await git(['read-tree', snapshot], directory, environment)
    await git(['add', '-A', '--', '.'], directory, environment)
    if (await git(['write-tree'], directory, environment) !== snapshot) throw new Error('Working tree changed during capture; retry when edits settle')
    return { tree, flattened: links.length > 0 }
  } finally {
    await unlink(index).catch(error => { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error })
  }
}
