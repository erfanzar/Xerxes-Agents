// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, writeFile } from 'node:fs/promises'
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

export async function archiveOutput(
  baseDirectory: string,
  jobId: string,
  content: string,
  now = new Date(),
): Promise<string> {
  const directory = join(baseDirectory, jobId)
  await mkdir(directory, { recursive: true })
  const timestamp = now
    .toISOString()
    .replace(/[-:]/g, '')
    .replace(/\.\d{3}Z$/, '')
  const path = join(directory, `${timestamp}.md`)
  await writeFile(path, content, 'utf8')
  return path
}

/** Archive every result and forward it only for a real channel target. */
export async function routeOutput(
  target: DeliveryTarget,
  content: string,
  options: {
    readonly archiveDirectory: string
    readonly jobId: string
    readonly sender?: DeliverySender
  },
): Promise<string> {
  const path = await archiveOutput(
    options.archiveDirectory,
    options.jobId,
    content,
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
