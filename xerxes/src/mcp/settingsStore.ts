// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { createHash, randomUUID } from 'node:crypto'
import { closeSync, constants, fsyncSync, linkSync, openSync, renameSync, unlinkSync, writeFileSync } from 'node:fs'
import { readMcpDocument } from './document.js'
import { dirname, join } from 'node:path'
import { parseMcpConfigDocument, parseMcpServerConfig } from './config.js'
import type { MCPServerConfig } from './types.js'
import type { MCPManager } from './manager.js'
const MISSING_REVISION = createHash('sha256').update('xerxes:mcp:missing').digest('hex')

export interface McpSettingsSnapshot {
  readonly revision: string
  readonly servers: readonly MCPServerConfig[]
  readonly warnings?: readonly string[]
}

/** Existing user-file editor. Callers must not expose credential-bearing snapshots
 * in transcript events. Locking coordinates cooperating editors; revisions also
 * detect external changes before rename, but cannot exclude arbitrary writers.
 */
export class McpSettingsStore {
  constructor(readonly path: string) {}

  read(): McpSettingsSnapshot {
    const source = readMcpDocument(this.path, true)
    if (source === undefined) return { revision: MISSING_REVISION, servers: [] }
    const parsed = parseMcpConfigDocument(source, this.path)
    if (parsed.warnings.length) throw new Error('MCP settings contain invalid entries; repair the file before editing settings')
    if (new Set(parsed.servers.map(server => server.name)).size !== parsed.servers.length) throw new Error('MCP settings contain duplicate names; repair the file before editing settings')
    return { revision: createHash('sha256').update(source).digest('hex'), servers: parsed.servers }
  }

  /** Synchronous commit suitable for a live-registry swap boundary. The lock is
   * never stolen: interruption leaves an actionable lock requiring host recovery.
   */
  replace(value: unknown, revision: string, create = false): McpSettingsSnapshot {
    const parsed = parseMcpServerConfig(value)
    if (!parsed.ok) throw new TypeError(parsed.error)
    if (!/^[a-f0-9]{64}$/.test(revision)) throw new Error('Invalid MCP settings revision')
    const lockPath = `${this.path}.settings-lock`
    let lock: number
    try { lock = openSync(lockPath, 'wx', 0o600) } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'EEXIST') throw new Error('MCP settings are locked by another edit or an interrupted writer; inspect the settings lock before retrying')
      throw error
    }
    const temporary = join(dirname(this.path), `.mcp-settings-${randomUUID()}.tmp`)
    let temporaryExists = false
    let committed: { revision: string; servers: readonly MCPServerConfig[]; warnings?: string[] } | undefined
    try {
      writeFileSync(lock, JSON.stringify({ pid: process.pid, createdAt: Date.now() }))
      fsyncSync(lock)
      const current = this.read()
      if (current.revision !== revision) throw new Error('MCP settings changed; reload before saving')
      const exists = current.servers.some(server => server.name === parsed.config.name)
      if (create ? exists : !exists) throw new Error(create ? 'MCP server name already exists' : 'MCP server is not in this settings file; reload before saving')
      const servers = create ? [...current.servers, parsed.config] : current.servers.map(server => server.name === parsed.config.name ? parsed.config : server)
      const source = `${JSON.stringify({ servers }, null, 2)}\n`
      if (Buffer.byteLength(source) > 1_048_576) throw new Error('MCP settings exceed 1 MiB')
      const fd = openSync(temporary, 'wx', 0o600)
      temporaryExists = true
      try { writeFileSync(fd, source); fsyncSync(fd) } finally { closeSync(fd) }
      if (this.read().revision !== revision) throw new Error('MCP settings changed during save; reload before saving')
      const result = { revision: createHash('sha256').update(source).digest('hex'), servers }
      // Creating a missing file must never overwrite a concurrent external creator.
      if (revision === MISSING_REVISION) linkSync(temporary, this.path)
      else renameSync(temporary, this.path)
      committed = result
      temporaryExists = revision === MISSING_REVISION
      // Failure after rename must not make the live host retain old settings.
      // Report durability/cleanup uncertainty on the successful commit instead.
      let directory: number | undefined
      try { if (process.platform !== 'win32') { directory = openSync(dirname(this.path), constants.O_RDONLY); fsyncSync(directory) } }
      catch { committed.warnings = ['MCP settings saved, but directory synchronization failed; crash durability is uncertain'] }
      finally { if (directory !== undefined) { try { closeSync(directory) } catch { (committed.warnings ??= []).push('MCP settings directory handle cleanup failed') } } }
      return committed
    } finally {
      const cleanupErrors: string[] = []
      try { closeSync(lock) } catch { cleanupErrors.push('lock handle') }
      try { if (temporaryExists) unlinkSync(temporary) } catch { cleanupErrors.push('temporary file') }
      try { unlinkSync(lockPath) } catch { cleanupErrors.push('settings lock') }
      if (cleanupErrors.length) {
        const warning = `MCP settings cleanup failed for ${cleanupErrors.join(', ')}; inspect before retrying`
        if (committed) (committed.warnings ??= []).push(warning)
        else throw new Error(warning)
      }
    }
  }
}

/** Stage the connection first; a stale file or failed write leaves live state intact.
 * A process killed after file rename loads the new settings on its next startup.
 */
export async function replaceMcpSettings(
  manager: MCPManager,
  store: McpSettingsStore,
  value: unknown,
  revision: string,
  signal?: AbortSignal,
  create = false,
): Promise<McpSettingsSnapshot> {
  const parsed = parseMcpServerConfig(value)
  if (!parsed.ok) throw new TypeError(parsed.error)
  const settings = store.read()
  if (settings.revision !== revision) throw new Error('MCP settings changed; reload before saving')
  const exists = settings.servers.some(server => server.name === parsed.config.name)
  if (create ? exists : !exists) throw new Error(create ? 'MCP server name already exists' : 'MCP server is not in this settings file')
  let saved: McpSettingsSnapshot | undefined
  const install = create ? manager.createServer.bind(manager) : manager.replaceServer.bind(manager)
  await install(parsed.config, signal, () => {
    saved = store.replace(parsed.config, revision, create)
    return undefined
  })
  if (!saved) throw new Error('MCP settings were not committed')
  return saved
}
