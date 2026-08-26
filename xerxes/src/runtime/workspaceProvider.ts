// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { tmpdir } from 'node:os'
import { dirname, join, resolve, sep } from 'node:path'

export type WorkspaceKind = 'local' | 'docker' | 'ssh' | 'daytona' | 'modal'

export interface WorkspaceSpec {
  readonly id: string
  readonly kind: WorkspaceKind
  readonly image?: string
  readonly env?: Readonly<Record<string, string>>
  readonly cpus?: number
  readonly memoryMb?: number
  readonly workingDir?: string
  readonly repository?: string
}

export interface WorkspaceConnection {
  readonly id: string
  readonly kind: WorkspaceKind
  readonly workingDir: string
  readonly env: Readonly<Record<string, string>>
}

export interface WorkspaceExecResult {
  readonly exitCode: number
  readonly stdout: string
  readonly stderr: string
}

/** Host-injected boundary for a workspace provider. */
export interface WorkspaceProviderPort {
  readonly kind: WorkspaceKind
  prepare(spec: WorkspaceSpec): Promise<WorkspaceConnection>
  exec(connection: WorkspaceConnection, command: readonly string[]): Promise<WorkspaceExecResult>
  readFile(connection: WorkspaceConnection, path: string): Promise<string>
  writeFile(connection: WorkspaceConnection, path: string, content: string): Promise<void>
  destroy(connection: WorkspaceConnection): Promise<void>
}

export interface WorkspaceProviderRegistryOptions {
  readonly fallback?: WorkspaceProviderPort
}

export class WorkspaceProviderRegistry {
  private readonly providers = new Map<WorkspaceKind, WorkspaceProviderPort>()
  private readonly fallback?: WorkspaceProviderPort

  constructor(options: WorkspaceProviderRegistryOptions = {}) {
    if (options.fallback !== undefined) {
      this.fallback = options.fallback
    }
  }

  register(port: WorkspaceProviderPort): void {
    this.providers.set(port.kind, port)
  }

  get(kind: WorkspaceKind): WorkspaceProviderPort {
    const registered = this.providers.get(kind)
    if (registered !== undefined) return registered
    if (this.fallback !== undefined) return this.fallback
    throw new Error(`workspace provider ${kind} is not registered and no fallback is configured`)
  }

  has(kind: WorkspaceKind): boolean {
    return this.providers.has(kind) || this.fallback !== undefined
  }
}

export interface LocalWorkspaceHostPort {
  readonly spawn: (command: readonly string[], options: { cwd: string; env?: Record<string, string> }) => Promise<WorkspaceExecResult>
  readonly readFile: (path: string) => Promise<string>
  readonly writeFile: (path: string, content: string) => Promise<void>
  readonly mkdir: (path: string) => Promise<void>
}

/**
 * Where a workspace lives when the caller does not say.
 *
 * One definition, because this literal was repeated in five places — four in
 * workspaceCommand and one here — and `exec` reconstructing it independently of
 * `prepare` means one edit puts them in different directories. `tmpdir()`
 * rather than a hard-coded `/tmp`, which does not exist on Windows.
 */
export function defaultWorkspaceDir(id: string): string {
  return join(tmpdir(), `xerxes-workspace-${id}`)
}

/**
 * Confine a guest-supplied path to the workspace it belongs to.
 *
 * The provider joined `${workingDir}/${path}` raw, so `../../../../etc/passwd`
 * read and `../../.ssh/authorized_keys` wrote straight through — an escape
 * offered by the very abstraction whose stated job is isolation. Resolution has
 * to happen before the containment check, because `a/../../b` only reveals
 * itself as an escape once normalized.
 */
export function resolveInsideWorkspace(workingDir: string, path: string): string {
  const root = resolve(workingDir)
  const target = resolve(root, path)
  if (target !== root && !target.startsWith(`${root}${sep}`)) {
    throw new WorkspacePathEscapeError(path, workingDir)
  }
  return target
}

/** Raised when a workspace-relative path resolves outside its workspace. */
export class WorkspacePathEscapeError extends Error {
  constructor(readonly path: string, readonly workingDir: string) {
    super(`path ${JSON.stringify(path)} escapes workspace ${JSON.stringify(workingDir)}`)
    this.name = 'WorkspacePathEscapeError'
  }
}

export function createLocalWorkspaceProvider(host: LocalWorkspaceHostPort): WorkspaceProviderPort {
  return {
    kind: 'local',
    async prepare(spec) {
      const workingDir = spec.workingDir ?? defaultWorkspaceDir(spec.id)
      await host.mkdir(workingDir)
      return { id: spec.id, kind: 'local', workingDir, env: spec.env ?? {} }
    },
    async exec(connection, command) {
      return host.spawn(command, { cwd: connection.workingDir, env: { ...connection.env } })
    },
    async readFile(connection, path) {
      return host.readFile(resolveInsideWorkspace(connection.workingDir, path))
    },
    async writeFile(connection, path, content) {
      const target = resolveInsideWorkspace(connection.workingDir, path)
      // A nested path is the ordinary case for writing into a workspace; without
      // this every `a/b/c.txt` failed with ENOENT.
      await host.mkdir(dirname(target))
      return host.writeFile(target, content)
    },
    async destroy(_connection) {
      // Local directories are intentionally left for inspection unless the host cleans up.
    },
  }
}
