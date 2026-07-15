// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomUUID } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'

import { xerxesHome } from '../daemon/paths.js'
export interface SnapshotRecord {
  readonly commitSha: string
  readonly createdAt: string
  readonly id: string
  readonly label: string
  readonly workspaceDir: string
}

/**
 * Creates git snapshots in a bare shadow repository without modifying the
 * workspace's own git metadata or history.
 */
export class SnapshotManager {
  readonly workspaceDirectory: string
  private readonly recordsPath: string
  private readonly shadowRoot: string

  constructor(workspaceDirectory: string, options: SnapshotManagerOptions = {}) {
    this.workspaceDirectory = resolve(workspaceDirectory)
    this.shadowRoot = resolve(options.shadowRoot ?? join(xerxesHome(), 'snapshots'))
    this.recordsPath = join(this.shadowDirectory, '_records.txt')
  }

  get shadowDirectory(): string {
    return join(this.shadowRoot, workspaceHash(this.workspaceDirectory))
  }

  get(ref: string): SnapshotRecord | undefined {
    return this.list().find(record => record.id === ref || record.label === ref || record.commitSha.startsWith(ref))
  }

  list(): SnapshotRecord[] {
    if (!existsSync(this.recordsPath)) return []
    return readFileSync(this.recordsPath, 'utf8').split(/\r?\n/).flatMap(line => {
      if (!line.trim()) return []
      const parts = line.split('\t')
      if (parts.length !== 5) return []
      const [id, label, commitSha, createdAt, workspaceDir] = parts
      if (!id || label === undefined || !commitSha || !createdAt || !workspaceDir) return []
      return [{ id, label, commitSha, createdAt, workspaceDir }]
    })
  }

  prune(options: SnapshotPruneOptions = {}): number {
    const keep = options.keep ?? 100
    if (!Number.isInteger(keep) || keep < 0) throw new RangeError('keep must be a non-negative integer')
    const records = this.list()
    if (records.length <= keep) return 0
    const retained = keep === 0 ? [] : records.slice(-keep)
    this.writeRecords(retained)
    return records.length - retained.length
  }

  reset(): void {
    rmSync(this.shadowDirectory, { recursive: true, force: true })
  }

  rollback(ref: string): SnapshotRecord {
    const record = this.get(ref)
    if (!record) throw new Error(`snapshot not found: ${ref}`)
    this.runGit(['checkout', record.commitSha, '--', '.'])
    return record
  }

  snapshot(label = ''): SnapshotRecord {
    this.ensureRepository()
    this.runGit(['add', '-A'])
    const message = label || `snapshot-${new Date().toISOString()}`
    this.runGit(['commit', '--allow-empty', '-m', message])
    const commitSha = this.runGit(['rev-parse', 'HEAD']).trim()
    const record: SnapshotRecord = {
      id: randomUUID().replaceAll('-', '').slice(0, 12),
      label,
      commitSha,
      createdAt: new Date().toISOString(),
      workspaceDir: this.workspaceDirectory,
    }
    this.appendRecord(record)
    return record
  }

  /** Run a command against the shadow repository for snapshot-diff consumers. */
  runGit(args: readonly string[]): string {
    const result = spawnSync('git', [...args], {
      cwd: this.workspaceDirectory,
      env: {
        ...process.env,
        GIT_DIR: join(this.shadowDirectory, '.git'),
        GIT_WORK_TREE: this.workspaceDirectory,
        GIT_AUTHOR_NAME: 'xerxes-snapshot',
        GIT_AUTHOR_EMAIL: 'snapshots@xerxes',
        GIT_COMMITTER_NAME: 'xerxes-snapshot',
        GIT_COMMITTER_EMAIL: 'snapshots@xerxes',
      },
      encoding: 'utf8',
    })
    const stdout = typeof result.stdout === 'string' ? result.stdout : ''
    const stderr = typeof result.stderr === 'string' ? result.stderr : ''
    if (result.status !== 0) {
      throw new Error(`git ${args.join(' ')} failed (exit ${result.status ?? 'unknown'}): ${stderr.trim()}`)
    }
    return stdout
  }

  private appendRecord(record: SnapshotRecord): void {
    const existing = existsSync(this.recordsPath) ? readFileSync(this.recordsPath, 'utf8') : ''
    const label = record.label.replaceAll(/[\t\r\n]/g, ' ')
    const line = [record.id, label, record.commitSha, record.createdAt, record.workspaceDir].join('\t')
    const content = `${existing}${existing && !existing.endsWith('\n') ? '\n' : ''}${line}\n`
    this.writeTextAtomically(this.recordsPath, content)
  }

  private ensureRepository(): void {
    const gitDirectory = join(this.shadowDirectory, '.git')
    if (existsSync(gitDirectory)) return
    mkdirSync(this.shadowDirectory, { recursive: true })
    const result = spawnSync('git', ['init', '--bare', '--quiet', '--initial-branch', 'main', gitDirectory], {
      cwd: this.workspaceDirectory,
      encoding: 'utf8',
    })
    if (result.status !== 0) {
      const stderr = typeof result.stderr === 'string' ? result.stderr : ''
      throw new Error(`snapshot repository initialization failed: ${stderr.trim()}`)
    }
    const infoDirectory = join(gitDirectory, 'info')
    mkdirSync(infoDirectory, { recursive: true })
    writeFileSync(join(infoDirectory, 'exclude'), '.xerxes/snapshots/**\n', 'utf8')
  }

  private writeRecords(records: readonly SnapshotRecord[]): void {
    const content = records.map(record => [
      record.id,
      record.label.replaceAll(/[\t\r\n]/g, ' '),
      record.commitSha,
      record.createdAt,
      record.workspaceDir,
    ].join('\t')).join('\n')
    this.writeTextAtomically(this.recordsPath, content)
  }

  private writeTextAtomically(path: string, content: string): void {
    mkdirSync(dirname(path), { recursive: true })
    const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`
    try {
      writeFileSync(temporary, content, 'utf8')
      renameSync(temporary, path)
    } catch (error) {
      rmSync(temporary, { force: true })
      throw error
    }
  }
}

export interface SnapshotManagerOptions {
  readonly shadowRoot?: string
}

export interface SnapshotPruneOptions {
  readonly keep?: number
}

function workspaceHash(workspaceDirectory: string): string {
  return createHash('sha1').update(workspaceDirectory).digest('hex').slice(0, 12)
}
