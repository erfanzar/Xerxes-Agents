// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomUUID } from 'node:crypto'
import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'

const GIT_COMMAND_TIMEOUT_MS = 30_000

/** Paths the shadow repository never tracks or deletes: Xerxes state and common secret files. */
const SHADOW_EXCLUDE_PATTERNS = [
  '.xerxes/snapshots/**',
  '.env*',
  '*.pem',
  '*.key',
  '*credentials*',
  '*secret*',
  'id_rsa*',
  '.ssh/**',
  '.npmrc',
  '.netrc',
  '*.p12',
  '*.keystore',
  'kubeconfig*',
] as const

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
 *
 * Git commands run asynchronously with a bounded lifetime so snapshot work
 * never blocks the daemon's event loop, and the shadow directory is created
 * private (0o700) because it mirrors workspace contents.
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
    if (ref.length === 0) return undefined
    const records = this.list()
    const exact = records.find(record => record.id === ref || record.label === ref)
    if (exact) return exact
    // Empty or short SHA prefixes silently matched the first record, letting
    // rollback('') restore an arbitrary snapshot. Require enough entropy and
    // refuse ambiguous prefixes instead.
    if (ref.length < 4) return undefined
    const matches = records.filter(record => record.commitSha.startsWith(ref))
    if (matches.length > 1) {
      throw new Error(`ambiguous snapshot ref: ${ref} matches ${matches.length} snapshots`)
    }
    return matches[0]
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

  async prune(options: SnapshotPruneOptions = {}): Promise<number> {
    const keep = options.keep ?? 100
    if (!Number.isInteger(keep) || keep < 0) throw new RangeError('keep must be a non-negative integer')
    const records = this.list()
    if (records.length <= keep) return 0
    const retained = keep === 0 ? [] : records.slice(-keep)
    if (retained.length === 0) {
      this.reset()
      return records.length
    }
    // Re-anchor retained history on a fresh root commit so the pruned commits
    // become unreachable and `git gc` can collect them; otherwise the bare
    // repo would grow with every snapshot forever. Retained records keep their
    // ids and labels while their rewritten commit SHAs are stored back.
    const rewritten: SnapshotRecord[] = []
    let parent: string | undefined
    for (const record of retained) {
      const tree = (await this.runGit(['rev-parse', `${record.commitSha}^{tree}`])).trim()
      const args = ['commit-tree', tree, '-m', record.label || `snapshot-${record.createdAt}`]
      if (parent) args.push('-p', parent)
      parent = (await this.runGit(args)).trim()
      rewritten.push({ ...record, commitSha: parent })
    }
    if (parent) await this.runGit(['update-ref', 'HEAD', parent])
    // Bare repositories normally keep no reflogs; expiry is best-effort.
    await this.runGit(['reflog', 'expire', '--expire=now', '--all']).catch(() => '')
    await this.runGit(['gc', '--prune=now', '--quiet'])
    this.writeRecords(rewritten)
    return records.length - retained.length
  }

  reset(): void {
    rmSync(this.shadowDirectory, { recursive: true, force: true })
  }

  async rollback(ref: string): Promise<SnapshotRecord> {
    const record = this.get(ref)
    if (!record) throw new Error(`snapshot not found: ${ref}`)
    // checkout-index overwrites modified files without a backup, so capture
    // the current tree first; the pre-rollback snapshot can itself be
    // rolled back to undo a mistaken restore.
    await this.snapshot(`pre-rollback:${record.id}`)
    // Full-tree restore: point the index at the snapshot tree, rewrite every
    // tracked file, then delete files the snapshot does not track. `-x` also
    // removes ignored build outputs created after the snapshot (plain `-fd`
    // would honor the workspace .gitignore and leave a mixed tree), while the
    // explicit `-e` patterns keep shadow-excluded secrets and Xerxes state.
    await this.runGit(['read-tree', record.commitSha])
    await this.runGit(['checkout-index', '-f', '-a'])
    await this.runGit(['clean', '-fdx', ...SHADOW_EXCLUDE_PATTERNS.flatMap(pattern => ['-e', pattern])])
    return record
  }

  async snapshot(label = ''): Promise<SnapshotRecord> {
    await this.ensureRepository()
    await this.runGit(['add', '-A'])
    const message = label || `snapshot-${new Date().toISOString()}`
    await this.runGit(['commit', '--allow-empty', '-m', message])
    const commitSha = (await this.runGit(['rev-parse', 'HEAD'])).trim()
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
  async runGit(args: readonly string[]): Promise<string> {
    return runGitProcess(args, {
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
    })
  }

  private appendRecord(record: SnapshotRecord): void {
    const existing = existsSync(this.recordsPath) ? readFileSync(this.recordsPath, 'utf8') : ''
    const label = record.label.replaceAll(/[\t\r\n]/g, ' ')
    const line = [record.id, label, record.commitSha, record.createdAt, record.workspaceDir].join('\t')
    const content = `${existing}${existing && !existing.endsWith('\n') ? '\n' : ''}${line}\n`
    this.writeTextAtomically(this.recordsPath, content)
  }

  private async ensureRepository(): Promise<void> {
    const gitDirectory = join(this.shadowDirectory, '.git')
    if (!existsSync(gitDirectory)) {
      mkdirSync(this.shadowDirectory, { recursive: true, mode: 0o700 })
      // Normalize permissions even when the directory already existed.
      chmodSync(this.shadowDirectory, 0o700)
      await runGitProcess(['init', '--bare', '--quiet', '--initial-branch', 'main', gitDirectory], {
        cwd: this.workspaceDirectory,
        env: { ...process.env },
      })
    }
    this.ensureExcludePatterns(join(gitDirectory, 'info'))
  }

  private ensureExcludePatterns(infoDirectory: string): void {
    mkdirSync(infoDirectory, { recursive: true })
    const path = join(infoDirectory, 'exclude')
    const existing = existsSync(path) ? readFileSync(path, 'utf8') : ''
    const present = new Set(existing.split(/\r?\n/))
    const missing = SHADOW_EXCLUDE_PATTERNS.filter(pattern => !present.has(pattern))
    if (missing.length === 0) return
    const separator = existing.length > 0 && !existing.endsWith('\n') ? '\n' : ''
    writeFileSync(path, `${existing}${separator}${missing.join('\n')}\n`, 'utf8')
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

/** Run one git invocation with a hard timeout, killing the process when it overruns. */
async function runGitProcess(
  args: readonly string[],
  options: { readonly cwd: string; readonly env: Record<string, string | undefined> },
): Promise<string> {
  const child = Bun.spawn(['git', ...args], {
    cwd: options.cwd,
    env: options.env,
    stdout: 'pipe',
    stderr: 'pipe',
  })
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    child.kill()
  }, GIT_COMMAND_TIMEOUT_MS)
  try {
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])
    if (timedOut) throw new Error(`git ${args.join(' ')} timed out after ${GIT_COMMAND_TIMEOUT_MS}ms`)
    if (exitCode !== 0) {
      throw new Error(`git ${args.join(' ')} failed (exit ${exitCode}): ${stderr.trim()}`)
    }
    return stdout
  } finally {
    clearTimeout(timer)
  }
}

function workspaceHash(workspaceDirectory: string): string {
  return createHash('sha1').update(workspaceDirectory).digest('hex').slice(0, 12)
}
