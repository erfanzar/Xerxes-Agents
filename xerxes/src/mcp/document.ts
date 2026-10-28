// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { closeSync, constants, fstatSync, openSync, readSync } from 'node:fs'
import { openRegularFileSync } from '../security/regularFile.js'

/** Startup permits linked configuration; editing requires a uniquely owned file.
 * Both paths bound reads and reject special files without waiting for a writer.
 */
export function readMcpDocument(path: string, forEditing = false): string | undefined {
  let fd: number
  try { fd = (forEditing ? openRegularFileSync : openSync)(path, constants.O_RDONLY | (constants.O_NONBLOCK ?? 0)) }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined
    throw error
  }
  try {
    const stat = fstatSync(fd)
    if (!stat.isFile() || (forEditing && stat.nlink !== 1) || stat.size > 1_048_576) {
      throw new Error('MCP settings must be a regular file of at most 1 MiB; editing requires one hard link')
    }
    // A writer can grow the file after stat, so limit the actual bytes read too.
    const buffer = Buffer.alloc(1_048_577)
    let size = 0
    while (size < buffer.length) {
      const count = readSync(fd, buffer, size, buffer.length - size, null)
      if (!count) break
      size += count
    }
    if (size === buffer.length) throw new Error('MCP settings exceed 1 MiB')
    try { return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(buffer.subarray(0, size)) }
    catch { throw new Error('MCP settings must contain valid UTF-8') }
  } finally { closeSync(fd) }
}
