// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readdir, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

export interface DeliveryTarget {
  readonly platform: string
  readonly recipient?: string
}

export type DeliverySender = (
  platform: string,
  recipient: string,
  content: string,
) => void | Promise<void>

export interface ArchiveOptions {
  /** Maximum archived outputs kept per job; oldest files are pruned. Defaults to 50. */
  readonly retention?: number
}

const DEFAULT_RETENTION = 50

export async function archiveOutput(
  baseDirectory: string,
  jobId: string,
  content: string,
  now = new Date(),
  options: ArchiveOptions = {},
): Promise<string> {
  const directory = join(baseDirectory, jobId)
  await mkdir(directory, { recursive: true })
  // Millisecond resolution plus a UUID keeps same-second runs from overwriting
  // each other while preserving chronological filename ordering.
  const timestamp = now.toISOString().replace(/[-:.]/g, '').replace('Z', '')
  const path = join(directory, `${timestamp}-${crypto.randomUUID()}.md`)
  await writeFile(path, content, 'utf8')
  await pruneArchives(directory, options.retention ?? DEFAULT_RETENTION)
  return path
}

/** Archive every result and forward it only for a real channel target. */
export async function routeOutput(
  target: DeliveryTarget,
  content: string,
  options: {
    readonly archiveDirectory: string
    readonly jobId: string
    readonly retention?: number
    readonly sender?: DeliverySender
  },
): Promise<string> {
  const path = await archiveOutput(
    options.archiveDirectory,
    options.jobId,
    content,
    new Date(),
    options.retention === undefined ? {} : { retention: options.retention },
  )
  if (
    target.platform !== 'none' &&
    target.platform !== 'workspace' &&
    options.sender
  ) {
    await options.sender(target.platform, target.recipient ?? '', content)
  }
  return path
}

/** Keep the newest `retention` archives (filenames sort chronologically). */
async function pruneArchives(directory: string, retention: number): Promise<void> {
  const keep = Math.max(1, Math.floor(retention))
  const archives = (await readdir(directory))
    .filter((name) => name.endsWith('.md'))
    .sort()
  const excess = archives.length - keep
  for (let index = 0; index < excess; index += 1) {
    const stale = archives[index]
    if (stale) await rm(join(directory, stale), { force: true })
  }
}
