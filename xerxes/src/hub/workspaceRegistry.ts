// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmod, lstat, mkdir, open, readFile, rename, rm } from 'node:fs/promises'
import { dirname, isAbsolute, join } from 'node:path'
import { isSshTarget } from '../daemon/sshTarget.js'
import { withFileLock } from '../session/daemonTranscript.js'
import { hubPaths } from './paths.js'

export type HubWorkspaceEndpoint =
  | { readonly kind: 'local'; readonly socketPath?: string }
  | {
      readonly kind: 'ssh'
      readonly sshAlias: string
      readonly remoteProjectDir: string
      /** Absolute path of the daemon socket on the remote host; the tunnel forwards exactly this. */
      readonly remoteSocketPath?: string
    }

export interface HubWorkspaceInput {
  readonly label: string
  readonly endpoint: HubWorkspaceEndpoint
  /** For a `discover` machine entry this is always `/`: its projects are found by asking the daemon. */
  readonly projectDir: string
  readonly connection: { readonly autoConnect: boolean; readonly discover?: boolean }
  readonly consent: { readonly bootstrap: boolean; readonly update: boolean }
}

export interface HubWorkspace extends HubWorkspaceInput {
  readonly id: string
  readonly createdAt: string
  readonly updatedAt: string
}

export interface HubWorkspaceRegistrySnapshot {
  readonly version: 1
  readonly revision: number
  readonly workspaces: readonly HubWorkspace[]
}

export class HubRegistryError extends Error {
  constructor(readonly code: 'corrupt_registry' | 'invalid_input' | 'not_found', message: string) {
    super(message)
    this.name = 'HubRegistryError'
  }
}

export class HubRegistryConflictError extends Error {
  readonly code = 'revision_conflict'
  constructor(readonly expectedRevision: number, readonly actualRevision: number) {
    super(`Hub registry revision conflict: expected ${expectedRevision}, found ${actualRevision}; reload before retrying`)
    this.name = 'HubRegistryConflictError'
  }
}

const invalid = (): never => { throw new HubRegistryError('invalid_input', 'Invalid workspace settings; use absolute paths and SSH aliases, never passwords or key material') }
function object(value: unknown, keys: readonly string[]): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return invalid()
  const row = value as Record<string, unknown>
  if (Object.keys(row).some(key => !keys.includes(key))) return invalid()
  return row
}
function text(value: unknown, max = 4096): string {
  if (typeof value !== 'string' || !value.trim() || value.length > max || /[\x00-\x1f\x7f]/.test(value)) return invalid()
  return value
}
function absolute(value: unknown): string {
  const path = text(value)
  if (!isAbsolute(path)) return invalid()
  return path
}
function bool(value: unknown): boolean { return typeof value === 'boolean' ? value : invalid() }
function rejectSecrets(value: unknown, depth = 0): void {
  if (depth > 12) return invalid()
  if (typeof value === 'string') {
    if (/-----BEGIN|(?:ssh-(?:rsa|dss|ed25519)|ecdsa-sha2-\S+)\s+[A-Za-z0-9+/]{16,}|(?:password|passphrase|private[_ -]?key)\s*[:=]|\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}/i.test(value)) return invalid()
  } else if (value && typeof value === 'object') {
    for (const [key, child] of Object.entries(value)) {
      if (/password|passphrase|secret|token|credential|private.?key|identity.?file/i.test(key)) return invalid()
      rejectSecrets(child, depth + 1)
    }
  }
}
function input(value: unknown): HubWorkspaceInput {
  rejectSecrets(value)
  const row = object(value, ['label', 'endpoint', 'projectDir', 'connection', 'consent'])
  const connection = object(row.connection, ['autoConnect', 'discover'])
  const discover = connection.discover === undefined ? false : bool(connection.discover)
  const rawEndpoint = object(row.endpoint, ['kind', 'socketPath', 'sshAlias', 'remoteProjectDir', 'remoteSocketPath'])
  let endpoint: HubWorkspaceEndpoint
  if (rawEndpoint.kind === 'local') {
    object(rawEndpoint, ['kind', 'socketPath'])
    endpoint = { kind: 'local', ...(rawEndpoint.socketPath === undefined ? {} : { socketPath: absolute(rawEndpoint.socketPath) }) }
  } else if (rawEndpoint.kind === 'ssh') {
    object(rawEndpoint, ['kind', 'sshAlias', 'remoteProjectDir', 'remoteSocketPath'])
    if (!isSshTarget(rawEndpoint.sshAlias)) return invalid()
    let remoteSocketPath: string | undefined
    if (rawEndpoint.remoteSocketPath !== undefined) {
      remoteSocketPath = absolute(rawEndpoint.remoteSocketPath)
      // ssh -L takes local:remote; a colon in the remote socket path would change the forward.
      if (remoteSocketPath.includes(':')) return invalid()
    }
    const remoteProjectDir = absolute(rawEndpoint.remoteProjectDir)
    endpoint = {
      kind: 'ssh', sshAlias: rawEndpoint.sshAlias, remoteProjectDir: discover ? '/' : remoteProjectDir,
      ...(remoteSocketPath === undefined ? {} : { remoteSocketPath }),
    }
  } else return invalid()
  const consent = object(row.consent, ['bootstrap', 'update'])
  // A machine's projectDir is ignored for routing; it is stored as `/` so nothing depends on what the caller sent.
  if (discover && row.projectDir !== undefined) absolute(row.projectDir)
  const projectDir = discover ? '/' : absolute(row.projectDir)
  if (endpoint.kind === 'ssh' && endpoint.remoteProjectDir !== projectDir) return invalid()
  return {
    label: text(row.label, 200), endpoint, projectDir,
    connection: { autoConnect: bool(connection.autoConnect), ...(discover ? { discover: true } : {}) },
    consent: { bootstrap: bool(consent.bootstrap), update: bool(consent.update) },
  }
}
function timestamp(value: unknown): string {
  const result = text(value, 40)
  if (!Number.isFinite(Date.parse(result))) return invalid()
  return result
}
function entry(value: unknown): HubWorkspace {
  rejectSecrets(value)
  const row = object(value, ['id', 'label', 'endpoint', 'projectDir', 'connection', 'consent', 'createdAt', 'updatedAt'])
  const id = text(row.id, 100)
  if (!/^[a-zA-Z0-9_-]+$/.test(id)) return invalid()
  return { ...input({ label: row.label, endpoint: row.endpoint, projectDir: row.projectDir, connection: row.connection, consent: row.consent }), id, createdAt: timestamp(row.createdAt), updatedAt: timestamp(row.updatedAt) }
}
function revision(value: unknown): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0 || value >= Number.MAX_SAFE_INTEGER) return invalid()
  return value
}
function missing(error: unknown): boolean { return !!error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT' }

export interface WorkspaceRegistryOptions {
  readonly now?: () => Date
  readonly generateId?: () => string
}

/** Cross-process revision checks happen inside the same lock as the atomic replacement. */
export class WorkspaceRegistry {
  constructor(readonly path = hubPaths().workspacesPath, private readonly options: WorkspaceRegistryOptions = {}) {}

  async load(): Promise<HubWorkspaceRegistrySnapshot> {
    let raw: string
    try {
      const info = await lstat(this.path)
      if (!info.isFile() || info.size > 1024 * 1024) throw new HubRegistryError('corrupt_registry', 'Hub registry must be a regular file no larger than 1 MiB; preserve it and repair manually')
      raw = await readFile(this.path, 'utf8')
    } catch (error) {
      if (missing(error)) return { version: 1, revision: 0, workspaces: [] }
      throw error
    }
    try {
      const row = object(JSON.parse(raw) as unknown, ['version', 'revision', 'workspaces'])
      if (row.version !== 1 || !Array.isArray(row.workspaces) || row.workspaces.length > 1000) return invalid()
      const workspaces = row.workspaces.map(entry)
      if (new Set(workspaces.map(workspace => workspace.id)).size !== workspaces.length) return invalid()
      return { version: 1, revision: revision(row.revision), workspaces }
    } catch {
      // Never echo the corrupt input: it may contain a secret from a manual edit.
      throw new HubRegistryError('corrupt_registry', 'Corrupt Hub workspace registry; original file preserved. Repair it manually before writing')
    }
  }

  async add(value: unknown, expectedRevision: number): Promise<HubWorkspaceRegistrySnapshot> {
    const settings = input(value)
    return this.mutate(expectedRevision, snapshot => {
      if (snapshot.workspaces.length >= 1000) return invalid()
      const now = this.now()
      return [...snapshot.workspaces, entry({ ...settings, id: this.options.generateId?.() ?? crypto.randomUUID(), createdAt: now, updatedAt: now })]
    })
  }

  async update(id: string, patch: unknown, expectedRevision: number): Promise<HubWorkspaceRegistrySnapshot> {
    rejectSecrets(patch)
    const changes = object(patch, ['label', 'endpoint', 'projectDir', 'connection', 'consent'])
    return this.mutate(expectedRevision, snapshot => {
      if (!snapshot.workspaces.some(workspace => workspace.id === id)) throw new HubRegistryError('not_found', 'Unknown Hub workspace id')
      return snapshot.workspaces.map(workspace => workspace.id === id ? entry({ ...workspace, ...changes, updatedAt: this.now() }) : workspace)
    })
  }

  /**
   * Register ONE machine (a `discover` entry) for a daemon socket, idempotently by socket path. Older
   * per-project local entries that point at the same socket are removed in the same write, so re-linking a
   * machine replaces them instead of listing every project twice. Retries on a concurrent registry write.
   */
  async upsertMachine(machine: { readonly label: string; readonly socketPath: string }): Promise<{ readonly snapshot: HubWorkspaceRegistrySnapshot; readonly id: string; readonly created: boolean; readonly removed: readonly string[] }> {
    const settings = input({
      label: machine.label, endpoint: { kind: 'local', socketPath: machine.socketPath }, projectDir: '/',
      connection: { autoConnect: true, discover: true }, consent: { bootstrap: false, update: false },
    })
    const socketPath = settings.endpoint.kind === 'local' ? settings.endpoint.socketPath : undefined
    for (let attempt = 0; ; attempt += 1) {
      const current = await this.load()
      let id = ''
      let created = false
      let removed: string[] = []
      try {
        const snapshot = await this.mutate(current.revision, snap => {
          removed = []
          const existing = snap.workspaces.find(row => row.connection.discover === true && row.endpoint.kind === 'local' && row.endpoint.socketPath === socketPath)
          const kept = snap.workspaces.filter(row => {
            if (row.connection.discover === true || row.endpoint.kind !== 'local' || row.endpoint.socketPath !== socketPath) return true
            removed.push(row.id)
            return false
          })
          if (existing) {
            id = existing.id; created = false
            return kept.map(row => row.id === id && row.label !== settings.label ? entry({ ...row, label: settings.label, updatedAt: this.now() }) : row)
          }
          id = this.options.generateId?.() ?? crypto.randomUUID(); created = true
          const now = this.now()
          return [...kept, entry({ ...settings, id, createdAt: now, updatedAt: now })]
        })
        return { snapshot, id, created, removed }
      } catch (error) {
        if (error instanceof HubRegistryConflictError && attempt < 4) continue
        throw error
      }
    }
  }

  async remove(id: string, expectedRevision: number): Promise<HubWorkspaceRegistrySnapshot> {
    return this.mutate(expectedRevision, snapshot => {
      if (!snapshot.workspaces.some(workspace => workspace.id === id)) throw new HubRegistryError('not_found', 'Unknown Hub workspace id')
      return snapshot.workspaces.filter(workspace => workspace.id !== id)
    })
  }

  /** Opt-in, read-only access to the daemon's {machines:[{alias,target,workspacePath}]} format. */
  async importMachines(expectedRevision: number, machinesPath = join(dirname(dirname(this.path)), 'machines.json')): Promise<HubWorkspaceRegistrySnapshot> {
    let raw: string
    try { raw = await readFile(machinesPath, 'utf8') }
    catch (error) { if (missing(error)) raw = '{"machines":[]}'; else throw error }
    let imported: HubWorkspaceInput[]
    try {
      if (Buffer.byteLength(raw) > 1024 * 1024) return invalid()
      const row = object(JSON.parse(raw) as unknown, ['machines'])
      if (!Array.isArray(row.machines) || row.machines.length > 100) return invalid()
      const aliases = new Set<string>()
      imported = row.machines.map(value => {
        rejectSecrets(value)
        const machine = object(value, ['alias', 'target', 'workspacePath'])
        const label = text(machine.alias, 64)
        if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(label) || aliases.has(label)) return invalid()
        aliases.add(label)
        return input({ label, endpoint: { kind: 'ssh', sshAlias: machine.target, remoteProjectDir: machine.workspacePath }, projectDir: machine.workspacePath, connection: { autoConnect: false }, consent: { bootstrap: false, update: false } })
      })
    } catch { throw new HubRegistryError('invalid_input', 'Invalid machines configuration; nothing imported and source preserved') }
    return this.mutate(expectedRevision, snapshot => {
      const workspaces = [...snapshot.workspaces]
      for (const settings of imported) {
        if (workspaces.some(workspace => workspace.projectDir === settings.projectDir && workspace.endpoint.kind === 'ssh' && settings.endpoint.kind === 'ssh' && workspace.endpoint.sshAlias === settings.endpoint.sshAlias)) continue
        const now = this.now()
        workspaces.push(entry({ ...settings, id: this.options.generateId?.() ?? crypto.randomUUID(), createdAt: now, updatedAt: now }))
      }
      return workspaces
    })
  }

  private now(): string { return (this.options.now?.() ?? new Date()).toISOString() }

  private async mutate(expected: number, change: (snapshot: HubWorkspaceRegistrySnapshot) => readonly HubWorkspace[]): Promise<HubWorkspaceRegistrySnapshot> {
    revision(expected)
    await mkdir(dirname(this.path), { recursive: true, mode: 0o700 })
    await chmod(dirname(this.path), 0o700)
    const lock = `${this.path}.lock`
    return withFileLock(lock, async () => {
      await chmod(lock, 0o600)
      const current = await this.load()
      if (current.revision !== expected) throw new HubRegistryConflictError(expected, current.revision)
      const workspaces = change(current)
      if (workspaces.length > 1000 || new Set(workspaces.map(workspace => workspace.id)).size !== workspaces.length) return invalid()
      if (JSON.stringify(workspaces) === JSON.stringify(current.workspaces)) return current
      const next: HubWorkspaceRegistrySnapshot = { version: 1, revision: current.revision + 1, workspaces }
      const content = `${JSON.stringify(next, null, 2)}\n`
      if (Buffer.byteLength(content) > 1024 * 1024) return invalid()
      const temporary = `${this.path}.${crypto.randomUUID()}.tmp`
      try {
        const file = await open(temporary, 'wx', 0o600)
        try { await file.writeFile(content, 'utf8'); await file.sync() }
        finally { await file.close() }
        await rename(temporary, this.path)
      } finally { await rm(temporary, { force: true }) }
      return next
    }, { label: 'Hub workspace registry', staleMs: 30000, waitMs: 5000 })
  }
}
