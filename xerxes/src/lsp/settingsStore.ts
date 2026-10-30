// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { createHash, randomUUID } from 'node:crypto'
import { closeSync, constants, fstatSync, fsyncSync, linkSync, openSync, readSync, renameSync, unlinkSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { parseLspConfig, type LspServerConfig } from './config.js'
import { openRegularFileSync } from '../security/regularFile.js'

const digest = (source: Uint8Array | string) => createHash('sha256').update(source).digest('hex')
const MISSING = digest('xerxes:lsp:missing')
export interface LspSettingsSnapshot {
  readonly revision: string
  readonly servers: readonly LspServerConfig[]
  readonly warnings?: readonly string[]
}

/** Host-only snapshots contain executable settings. Never serialize them to transcript output. */
export class LspSettingsStore {
  constructor(readonly path: string) {}
  read(): LspSettingsSnapshot {
    let fd: number
    try { fd = openRegularFileSync(this.path, constants.O_RDONLY | (constants.O_NONBLOCK ?? 0)) }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { revision: MISSING, servers: [] }
      throw new Error('Cannot open user lsp.json; use a readable regular file')
    }
    try {
      const stat = fstatSync(fd)
      if (!stat.isFile() || stat.nlink !== 1) throw new Error('LSP settings require a regular file with one hard link')
      const buffer = Buffer.alloc(1_048_577); let size = 0
      while (size < buffer.length) {
        const count = readSync(fd, buffer, size, buffer.length - size, null)
        if (!count) break
        size += count
      }
      if (size === buffer.length) throw new Error('LSP settings exceed 1 MiB')
      const bytes = buffer.subarray(0, size)
      let value: unknown
      try { value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)) }
      catch { throw new Error('User lsp.json must contain valid UTF-8 JSON') }
      return { revision: digest(bytes), servers: parseLspConfig(value) }
    } finally { closeSync(fd) }
  }

  /** Revision checks coordinate editors; arbitrary external writers are not locked out. */
  save(value: unknown, revision: string): LspSettingsSnapshot {
    const servers = parseLspConfig(value)
    if (!/^[a-f0-9]{64}$/.test(revision)) throw new Error('Invalid LSP settings revision')
    const source = JSON.stringify({ servers }, null, 2) + '\n'
    if (Buffer.byteLength(source) > 1_048_576) throw new Error('LSP settings exceed 1 MiB')
    const lockPath = this.path + '.settings-lock'
    let lock: number
    try { lock = openSync(lockPath, 'wx', 0o600) }
    catch { throw new Error('Cannot acquire LSP settings lock; inspect file permissions or an interrupted writer before retrying') }
    const temporary = join(dirname(this.path), '.lsp-settings-' + randomUUID() + '.tmp')
    let exists = false
    let committed: { revision: string; servers: readonly LspServerConfig[]; warnings: string[] } | undefined
    try {
      writeFileSync(lock, JSON.stringify({ pid: process.pid, createdAt: Date.now() })); fsyncSync(lock)
      if (this.read().revision !== revision) throw new Error('LSP settings changed; reload before saving')
      const fd = openSync(temporary, 'wx', 0o600); exists = true
      try { writeFileSync(fd, source); fsyncSync(fd) } finally { closeSync(fd) }
      if (this.read().revision !== revision) throw new Error('LSP settings changed; reload before saving')
      if (revision === MISSING) linkSync(temporary, this.path)
      else { renameSync(temporary, this.path); exists = false }
      committed = { revision: digest(source), servers, warnings: [] }
      let directory: number | undefined
      try { if (process.platform !== 'win32') { directory = openSync(dirname(this.path), constants.O_RDONLY); fsyncSync(directory) } }
      catch { committed.warnings.push('Settings saved; directory synchronization failed and crash durability is uncertain') }
      finally { if (directory !== undefined) { try { closeSync(directory) } catch { committed.warnings.push('Settings saved; directory handle cleanup failed') } } }
      return committed
    } finally {
      const failures: string[] = []
      try { closeSync(lock) } catch { failures.push('lock handle') }
      try { if (exists) unlinkSync(temporary) } catch { failures.push('temporary file') }
      try { unlinkSync(lockPath) } catch { failures.push('settings lock') }
      if (failures.length) {
        const warning = 'LSP settings cleanup failed: ' + failures.join(', ')
        if (committed) committed.warnings.push(warning)
        else throw new Error(warning)
      }
    }
  }
}
