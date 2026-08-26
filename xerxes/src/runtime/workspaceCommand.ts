// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { localWorkspaceHost } from './workspaceLocalHost.js'
import { WorkspaceProviderRegistry, createLocalWorkspaceProvider, defaultWorkspaceDir } from './workspaceProvider.js'

export type WorkspaceCommandAction = 'create' | 'exec' | 'read' | 'write' | 'destroy'

export interface WorkspaceCommandOptions {
  readonly action: WorkspaceCommandAction
  readonly id?: string
  readonly workingDir?: string
  readonly command?: readonly string[]
  readonly path?: string
  readonly content?: string
}

export interface WorkspaceCommandResult {
  readonly ok: boolean
  readonly connection?: { id: string; kind: string; workingDir: string }
  readonly exitCode?: number
  readonly stdout?: string
  readonly stderr?: string
  readonly content?: string
  readonly error?: string
}

export async function runWorkspaceCommand(options: WorkspaceCommandOptions): Promise<WorkspaceCommandResult> {
  const registry = new WorkspaceProviderRegistry()
  registry.register(createLocalWorkspaceProvider(localWorkspaceHost))
  const provider = registry.get('local')

  try {
    switch (options.action) {
      case 'create': {
        if (!options.id) return { ok: false, error: 'workspace create requires --id' }
        const connection = await provider.prepare({ id: options.id, kind: 'local', workingDir: options.workingDir })
        return { ok: true, connection: { id: connection.id, kind: connection.kind, workingDir: connection.workingDir } }
      }
      case 'exec': {
        if (!options.id || !options.command?.length) return { ok: false, error: 'workspace exec requires --id and a command' }
        const connection = { id: options.id, kind: 'local' as const, workingDir: options.workingDir ?? defaultWorkspaceDir(options.id), env: {} }
        const result = await provider.exec(connection, options.command)
        return { ok: result.exitCode === 0, exitCode: result.exitCode, stdout: result.stdout, stderr: result.stderr }
      }
      case 'read': {
        if (!options.id || !options.path) return { ok: false, error: 'workspace read requires --id and --path' }
        const connection = { id: options.id, kind: 'local' as const, workingDir: options.workingDir ?? defaultWorkspaceDir(options.id), env: {} }
        const content = await provider.readFile(connection, options.path)
        return { ok: true, content }
      }
      case 'write': {
        if (!options.id || !options.path || options.content === undefined) return { ok: false, error: 'workspace write requires --id, --path, and --content' }
        const connection = { id: options.id, kind: 'local' as const, workingDir: options.workingDir ?? defaultWorkspaceDir(options.id), env: {} }
        await provider.writeFile(connection, options.path, options.content)
        return { ok: true }
      }
      case 'destroy': {
        if (!options.id) return { ok: false, error: 'workspace destroy requires --id' }
        const connection = { id: options.id, kind: 'local' as const, workingDir: options.workingDir ?? defaultWorkspaceDir(options.id), env: {} }
        await provider.destroy(connection)
        return { ok: true }
      }
    }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}
