// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomUUID } from 'node:crypto'
import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { lstat, mkdir, open, readFile, rename, rm, stat } from 'node:fs/promises'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'

const GIT_COMMAND_TIMEOUT_MS = 30_000
const REPOSITORY_LOCK_STALE_MS = 60_000
const REPOSITORY_LOCK_WAIT_MS = 2

/** Mutating shadow-Git operations share an index and record log across manager instances. */
const repositoryOperations = new Map<string, Promise<void>>()

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

/** Number of tab-separated fields written before the session/turn link existed. */
const LEGACY_RECORD_FIELDS = 5

export interface SnapshotRecord {
  readonly commitSha: string
  readonly createdAt: string
  readonly id: string
  readonly label: string
  /** Session that owns this snapshot; absent for manual or pre-link records. */
  readonly sessionId?: string
  /**
   * Index of the turn this snapshot precedes. Without it "take me back to
   * before turn 7" is unexpressible: a bare timestamp cannot be matched to a
   * point in the conversation the user actually remembers.
   */
  readonly turnIndex?: number
  readonly workspaceDir: string
}

export interface SnapshotRestoreAttempt {
  readonly id: string
  readonly targetId: string
  readonly backupId: string
  readonly path: string | null
  readonly phase: 'prepared' | 'completed' | 'failed' | 'recovered' | 'reverted'
  readonly updatedAt: string
  readonly error?: string
}

/** Conversation coordinates attached to an automatic snapshot. */
export interface SnapshotLink {
  readonly sessionId?: string
  readonly turnIndex?: number
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

  /**
   * Read the record log, tolerating rows written before snapshots carried a
   * session/turn link.
   *
   * Those rows have five fields instead of seven. Requiring the new width
   * would silently discard every snapshot a user had already taken, so a short
   * row is read as an unlinked record and a longer one keeps only the fields
   * this version understands.
   */
  list(): SnapshotRecord[] {
    if (!existsSync(this.recordsPath)) return []
    return readFileSync(this.recordsPath, 'utf8').split(/\r?\n/).flatMap(line => {
      if (!line.trim()) return []
      const parts = line.split('\t')
      if (parts.length < LEGACY_RECORD_FIELDS) return []
      const [id, label, commitSha, createdAt, workspaceDir, sessionId, turnIndex] = parts
      if (!id || label === undefined || !commitSha || !createdAt || !workspaceDir) return []
      const turn = parseTurnIndex(turnIndex)
      return [{
        id,
        label,
        commitSha,
        createdAt,
        workspaceDir,
        ...(sessionId ? { sessionId } : {}),
        ...(turn === undefined ? {} : { turnIndex: turn }),
      }]
    })
  }

  /** Snapshots taken for one session, oldest first. */
  listForSession(sessionId: string): SnapshotRecord[] {
    if (!sessionId) return []
    return this.list().filter(record => record.sessionId === sessionId)
  }

  /**
   * The snapshot capturing the workspace as it stood before a given turn.
   *
   * The newest match wins: a retried turn snapshots the same index again, and
   * the later capture is the one that precedes the attempt still in the
   * transcript.
   */
  getForTurn(sessionId: string, turnIndex: number): SnapshotRecord | undefined {
    return this.listForSession(sessionId).filter(record => record.turnIndex === turnIndex).at(-1)
  }

  async prune(options: SnapshotPruneOptions = {}): Promise<number> {
    return this.serializeRepositoryOperation(() => this.pruneUnlocked(options))
  }

  private async pruneUnlocked(options: SnapshotPruneOptions): Promise<number> {
    const keep = options.keep ?? 100
    if (!Number.isInteger(keep) || keep < 0) throw new RangeError('keep must be a non-negative integer')
    const records = this.list()
    if (records.length <= keep) return 0
    const attempts = await this.restoreAttempts()
    const pinned = new Set(attempts.filter(attempt => (attempt.phase === 'prepared' || attempt.phase === 'failed')).flatMap(attempt => [attempt.targetId, attempt.backupId]))
    const ordinary = new Set((keep === 0 ? [] : records.slice(-keep)).map(record => record.id))
    const retained = records.filter(record => ordinary.has(record.id) || pinned.has(record.id))
    if (retained.length === 0) {
      this.resetUnlocked()
      return records.length
    }
    // Re-anchor retained history on a fresh root commit so the pruned commits
    // become unreachable and `git gc` can collect them; otherwise the bare
    // repo would grow with every snapshot forever. Retained records keep their
    // ids and labels while their rewritten commit SHAs are stored back.
    const rewritten: SnapshotRecord[] = []
    let parent: string | undefined
    for (const record of retained) {
      const tree = (await this.runGitUnlocked(['rev-parse', `${record.commitSha}^{tree}`])).trim()
      const args = ['commit-tree', tree, '-m', record.label || `snapshot-${record.createdAt}`]
      if (parent) args.push('-p', parent)
      parent = (await this.runGitUnlocked(args)).trim()
      rewritten.push({ ...record, commitSha: parent })
    }
    if (parent) await this.runGitUnlocked(['update-ref', 'HEAD', parent])
    // Bare repositories normally keep no reflogs; expiry is best-effort.
    await this.runGitUnlocked(['reflog', 'expire', '--expire=now', '--all']).catch(() => '')
    await this.runGitUnlocked(['gc', '--prune=now', '--quiet'])
    this.writeRecords(rewritten)
    return records.length - retained.length
  }

  async reset(): Promise<void> {
    await this.serializeRepositoryOperation(async () => this.resetUnlocked())
  }

  private resetUnlocked(): void {
    rmSync(this.shadowDirectory, { recursive: true, force: true })
  }

  async rollback(ref: string, revision?: string): Promise<SnapshotRecord> {
    return this.serializeRepositoryOperation(() => this.rollbackUnlocked(ref, revision))
  }

  private async rollbackUnlocked(ref: string, revision?: string): Promise<SnapshotRecord> {
    const record = this.get(ref)
    if (!record) throw new Error(`snapshot not found: ${ref}`)
    await this.ensureRepository()
    const targetPaths = (await this.runGitUnlocked(['ls-tree', '-r', '--name-only', '-z', record.commitSha])).split('\0').filter(Boolean)
    await this.preflightRestorePaths(targetPaths)
    // checkout-index overwrites modified files without a backup, so capture
    // the current tree first; the pre-rollback snapshot can itself be
    // rolled back to undo a mistaken restore.
    const previous = await this.snapshotUnlocked(`pre-rollback:${record.id}`)
    if (revision !== undefined) {
      const tree = (await this.runGitUnlocked(['rev-parse', `${previous.commitSha}^{tree}`])).trim()
      if (revision !== restoreRevision(record.commitSha, tree)) {
        throw new Error('Snapshot preview is stale or belongs to another target; preview again before restoring. Workspace files were not changed.')
      }
    }
    // Only remove paths captured by the backup. Ignored files may contain work
    // that neither snapshot tracks; a workspace-wide clean -x would lose it.
    const removed = (await this.runGitUnlocked(['diff', '--name-only', '--no-renames', '--diff-filter=D', '-z', previous.commitSha, record.commitSha, '--', '.']))
      .split('\0').filter(Boolean)
    await this.preflightRestorePaths(removed)
    await this.withRestoreAttempt(record, previous, null, async () => {
    await this.runGitUnlocked(['read-tree', record.commitSha])
    await this.runGitUnlocked(['checkout-index', '-f', '-a'])
    for (const path of removed) {
      const absolute = join(this.workspaceDirectory, this.workspaceRelativePath(path))
      const current = await lstat(absolute).catch(error => {
        if (hasErrorCode(error, 'ENOENT')) return undefined
        throw error
      })
      if (!current) continue
      if (current.isDirectory()) throw new Error(`Restore cleanup refused a directory at ${path}; backup ${previous.id} preserves the prior files`)
      // Honor the restored ignore rules too: excluded output is not part of
      // the requested snapshot state, even if an intermediate backup saw it.
      await this.runGitUnlocked(['clean', '-f', '--', `:(literal)${path}`])
    }
    })
    return record
  }

  async snapshot(label = '', link: SnapshotLink = {}): Promise<SnapshotRecord> {
    return this.serializeRepositoryOperation(() => this.snapshotUnlocked(label, link))
  }

  private async snapshotUnlocked(label = '', link: SnapshotLink = {}): Promise<SnapshotRecord> {
    await this.ensureRepository()
    await this.runGitUnlocked(['add', '-A'])
    const message = label || `snapshot-${new Date().toISOString()}`
    await this.runGitUnlocked(['commit', '--allow-empty', '-m', message])
    const commitSha = (await this.runGitUnlocked(['rev-parse', 'HEAD'])).trim()
    const turnIndex = link.turnIndex
    const record: SnapshotRecord = {
      id: randomUUID().replaceAll('-', '').slice(0, 12),
      label,
      commitSha,
      createdAt: new Date().toISOString(),
      workspaceDir: this.workspaceDirectory,
      ...(link.sessionId ? { sessionId: link.sessionId } : {}),
      ...(turnIndex === undefined || !Number.isInteger(turnIndex) || turnIndex < 0 ? {} : { turnIndex }),
    }
    this.appendRecord(record)
    return record
  }

  /**
   * Restore one file from a snapshot without touching the rest of the tree.
   *
   * A full rollback is the wrong tool when a single file was damaged: it also
   * discards every unrelated edit made since. Like rollback, this captures the
   * current tree first, because `git checkout` overwrites the target with no
   * backup of its own.
   */
  async restoreFile(ref: string, filePath: string, revision?: string): Promise<SnapshotRestoreResult> {
    return this.serializeRepositoryOperation(() => this.restoreFileUnlocked(ref, filePath, revision))
  }

  private async restoreFileUnlocked(ref: string, filePath: string, revision?: string): Promise<SnapshotRestoreResult> {
    const record = this.get(ref)
    if (!record) throw new Error(`snapshot not found: ${ref}`)
    const path = this.workspaceRelativePath(filePath)
    await this.ensureRepository()
    const targetEntry = await this.fileEntry(record.commitSha, path)
    if (!targetEntry && revision === undefined) throw new Error(`snapshot ${record.id} does not track ${path}`)
    await this.preflightRestorePaths([path])
    const previous = await this.snapshotUnlocked(`pre-restore:${record.id}`)
    if (revision !== undefined) {
      const currentEntry = await this.fileEntry(previous.commitSha, path)
      if (revision !== fileRestoreRevision(record.commitSha, path, currentEntry)) throw new Error('Snapshot file preview is stale; preview this file again. Workspace files were not changed.')
      if (!targetEntry && !currentEntry) throw new Error('File is absent from both captured trees')
    }
    await this.withRestoreAttempt(record, previous, path, async () => {
    if (targetEntry) {
      await this.runGitUnlocked(['checkout', record.commitSha, '--', `:(literal)${path}`])
    } else {
      // A guarded deletion only removes this backed-up leaf; rm is not recursive.
      await this.preflightRestorePaths([path])
      await rm(join(this.workspaceDirectory, path), { force: true })
      await this.runGitUnlocked(['update-index', '--force-remove', '--', path])
    }
    })
    return { path, previous, snapshot: record }
  }

  /** Compare through a private index, including new files without changing the workspace index. */
  async diff(ref: string, reverse = false): Promise<string> {
    return (await this.preview(ref, reverse)).diff
  }

  async preview(ref: string, reverse = true, filePath?: string): Promise<{ snapshot: SnapshotRecord; revision: string; diff: string; files: string[]; action?: 'restore' | 'remove' }> {
    return this.serializeRepositoryOperation(async () => {
      const record = this.get(ref)
      if (!record) throw new Error(`snapshot not found: ${ref}`)
      await this.ensureRepository()
      const index = join(this.shadowDirectory, `preview-${randomUUID()}.index`)
      try {
        await this.runGitUnlocked(['read-tree', 'HEAD'], index)
        await this.runGitUnlocked(['add', '-A'], index)
        const tree = (await this.runGitUnlocked(['write-tree'], index)).trim()
        const path = filePath === undefined ? undefined : this.workspaceRelativePath(filePath)
        const files = (await this.runGitUnlocked(['diff', '--no-renames', '--name-only', '-z', record.commitSha, tree, '--', '.'], index)).split('\0').filter(Boolean)
        const targetEntry = path === undefined ? undefined : await this.fileEntry(record.commitSha, path)
        const currentEntry = path === undefined ? undefined : await this.fileEntry(tree, path)
        if (path !== undefined && !targetEntry && !currentEntry) throw new Error('File is absent from both captured trees')
        const diff = await this.runGitUnlocked(['diff', '--no-ext-diff', '--no-textconv', '--no-renames', '--binary',
          ...(reverse ? ['-R'] : []), record.commitSha, tree, '--', path === undefined ? '.' : `:(literal)${path}`], index, 2 * 1024 * 1024)
        return { snapshot: record, revision: path === undefined ? restoreRevision(record.commitSha, tree) : fileRestoreRevision(record.commitSha, path, currentEntry!), diff, files,
          ...(path === undefined ? {} : { action: targetEntry ? 'restore' as const : 'remove' as const }) }
      } finally {
        await rm(index, { force: true })
        await rm(index + '.lock', { force: true })
      }
    })
  }

  async restoreAttempts(): Promise<SnapshotRestoreAttempt[]> {
    const filename = join(this.shadowDirectory, '_restore-attempts.json')
    const info = await stat(filename).catch(error => { if (hasErrorCode(error, 'ENOENT')) return undefined; throw error })
    if (!info) return []
    if (info.size > 512_000) throw new Error('Restore recovery journal exceeds its size limit')
    const parsed: unknown = JSON.parse(await readFile(filename, 'utf8'))
    if (!Array.isArray(parsed) || parsed.length > 256) throw new Error('Invalid restore recovery journal')
    return parsed.map((value: unknown) => {
      if (!value || typeof value !== 'object') throw new Error('Invalid restore recovery record')
      const row = value as Record<string, unknown>
      if (typeof row.id !== 'string' || typeof row.targetId !== 'string' || typeof row.backupId !== 'string' || typeof row.updatedAt !== 'string'
        || (row.path !== null && typeof row.path !== 'string') || !['prepared', 'completed', 'failed', 'recovered', 'reverted'].includes(String(row.phase))
        || (row.error !== undefined && typeof row.error !== 'string')) throw new Error('Invalid restore recovery record')
      return row as unknown as SnapshotRestoreAttempt
    })
  }

  private async saveRestoreAttempts(rows: SnapshotRestoreAttempt[]): Promise<void> {
    const filename = join(this.shadowDirectory, '_restore-attempts.json')
    const temporary = filename + '.' + randomUUID()
    const body = JSON.stringify(rows)
    if (rows.length > 256 || Buffer.byteLength(body) > 512_000) throw new Error('Restore recovery journal exceeds its size limit')
    try {
      const file = await open(temporary, 'wx', 0o600)
      try { await file.writeFile(body); await file.sync() } finally { await file.close() }
      await rename(temporary, filename)
      if (process.platform !== 'win32') {
        const directory = await open(this.shadowDirectory, 'r')
        try { await directory.sync() } finally { await directory.close() }
      }
    } finally { await rm(temporary, { force: true }) }
  }

  private async withRestoreAttempt(target: SnapshotRecord, backup: SnapshotRecord, path: string | null, mutation: () => Promise<void>): Promise<void> {
    const history = await this.restoreAttempts()
    const unresolved = history.filter(row => row.phase === 'prepared' || row.phase === 'failed')
    const recovering = unresolved.some(row => row.backupId === target.id && (path === null || row.path === path))
    if (unresolved.length >= 128 && !recovering) throw new Error('Too many unresolved restore attempts; review recovery records before restoring')
    // Keep the original recovery anchor and the latest retry for this target/scope.
    // Otherwise failed recovery retries eventually make their own journal unreadable.
    const pending = recovering ? unresolved.filter(row => row.targetId !== target.id || row.path !== path) : unresolved
    const completed = history.filter(row => row.phase === 'completed' || row.phase === 'recovered' || row.phase === 'reverted')
    const keepCompleted = Math.max(0, Math.min(100, 255 - pending.length))
    const rows = [...(keepCompleted ? completed.slice(-keepCompleted) : []), ...pending]
    const attempt: SnapshotRestoreAttempt = { id: randomUUID(), targetId: target.id, backupId: backup.id, path, phase: 'prepared', updatedAt: new Date().toISOString() }
    await this.saveRestoreAttempts([...rows, attempt])
    try {
      await mutation()
      await this.saveRestoreAttempts([...rows.map(row => row.backupId === target.id && (path === null || row.path === path) ? { ...row, phase: 'recovered' as const, updatedAt: new Date().toISOString() } : row), { ...attempt, phase: 'completed', updatedAt: new Date().toISOString() }])
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause)
      let recoveryError = ''
      try {
        const conflicts = await this.reverseFailedRestore(target, backup, path)
        if (conflicts.length) recoveryError = `${conflicts.length} file(s) need review: ${conflicts.slice(0, 20).join(', ')}`
      } catch (error) { recoveryError = error instanceof Error ? error.message : String(error) }
      const detail = recoveryError ? `${message}; automatic reversal incomplete: ${recoveryError}` : `${message}; original captured files restored automatically`
      try { await this.saveRestoreAttempts([...rows, { ...attempt, phase: recoveryError ? 'failed' : 'reverted', updatedAt: new Date().toISOString(), error: detail.slice(0, 2000) }]) }
      catch (journalError) { throw new AggregateError([cause, journalError], `Restore outcome uncertain; backup ${backup.id}, attempt ${attempt.id}. Recovery journal update also failed.`) }
      throw new Error(`Restore failed; ${recoveryError ? 'workspace may be partially changed' : 'original captured files restored automatically'}. Review backup ${backup.id} (attempt ${attempt.id}): ${detail}`, { cause })
    }
  }

  /** Reverse only files still matching this restore's target, preserving unknown edits. */
  private async reverseFailedRestore(target: SnapshotRecord, backup: SnapshotRecord, path: string | null): Promise<string[]> {
    const paths = path === null
      ? (await this.runGitUnlocked(['diff', '--name-only', '--no-renames', '-z', backup.commitSha, target.commitSha, '--', '.'])).split('\0').filter(Boolean)
      : [path]
    const index = join(this.shadowDirectory, `reversal-${randomUUID()}.index`)
    const conflicts: string[] = []
    const deadline = Date.now() + 10_000
    try {
      for (const candidate of paths) {
        if (Date.now() > deadline) { conflicts.push(candidate); continue }
        try {
          await this.preflightRestorePaths([candidate])
          const before = await this.fileEntry(backup.commitSha, candidate)
          const intended = await this.fileEntry(target.commitSha, candidate)
          const current = await this.currentFileEntry(backup.commitSha, candidate, index)
          if (current === before) continue
          if (current !== intended) { conflicts.push(candidate); continue }
          await this.preflightRestorePaths([candidate])
          if (before) await this.runGitUnlocked(['checkout', backup.commitSha, '--', `:(literal)${candidate}`])
          else {
            await rm(join(this.workspaceDirectory, candidate), { force: true })
            await this.runGitUnlocked(['update-index', '--force-remove', '--', candidate])
          }
          if (await this.currentFileEntry(backup.commitSha, candidate, index) !== before) conflicts.push(candidate)
        } catch { conflicts.push(candidate) }
      }
      return conflicts
    } finally { await rm(index, { force: true }); await rm(index + '.lock', { force: true }) }
  }

  private async currentFileEntry(base: string, path: string, index: string): Promise<string> {
    const info = await lstat(join(this.workspaceDirectory, path)).catch(error => { if (hasErrorCode(error, 'ENOENT')) return undefined; throw error })
    if (!info) return ''
    await this.runGitUnlocked(['read-tree', base], index)
    await this.runGitUnlocked(['add', '-A', '--', `:(literal)${path}`], index)
    const tree = (await this.runGitUnlocked(['write-tree'], index)).trim()
    return this.fileEntry(tree, path)
  }

  private async fileEntry(tree: string, path: string): Promise<string> {
    const entry = await this.runGitUnlocked(['ls-tree', '-z', tree, '--', `:(literal)${path}`])
    if (entry && !/^(100644|100755|120000) blob /.test(entry)) throw new Error('Selected snapshot path is not a file or symlink')
    return entry
  }

  /** Run a command against the shadow repository for snapshot-diff consumers. */
  async runGit(args: readonly string[]): Promise<string> {
    return this.serializeRepositoryOperation(() => this.runGitUnlocked(args))
  }

  private async runGitUnlocked(args: readonly string[], indexFile?: string, maxOutputBytes?: number): Promise<string> {
    return runGitProcess(args, {
      ...(maxOutputBytes === undefined ? {} : { maxOutputBytes }),
      cwd: this.workspaceDirectory,
      env: {
        ...process.env,
        GIT_INDEX_FILE: indexFile,
        GIT_COMMON_DIR: undefined,
        GIT_OBJECT_DIRECTORY: undefined,
        GIT_ALTERNATE_OBJECT_DIRECTORIES: undefined,
        GIT_DIR: join(this.shadowDirectory, '.git'),
        GIT_WORK_TREE: this.workspaceDirectory,
        GIT_AUTHOR_NAME: 'xerxes-snapshot',
        GIT_AUTHOR_EMAIL: 'snapshots@xerxes',
        GIT_COMMITTER_NAME: 'xerxes-snapshot',
        GIT_COMMITTER_EMAIL: 'snapshots@xerxes',
      },
    })
  }

  private async serializeRepositoryOperation<T>(operation: () => Promise<T>): Promise<T> {
    const key = this.shadowDirectory
    const previous = repositoryOperations.get(key) ?? Promise.resolve()
    let release!: () => void
    const current = new Promise<void>(resolveOperation => { release = resolveOperation })
    repositoryOperations.set(key, current)
    await previous.catch(() => undefined)
    try {
      return await withRepositoryLock(`${key}.lock`, operation)
    } finally {
      release()
      if (repositoryOperations.get(key) === current) repositoryOperations.delete(key)
    }
  }

  private appendRecord(record: SnapshotRecord): void {
    const existing = existsSync(this.recordsPath) ? readFileSync(this.recordsPath, 'utf8') : ''
    const content = `${existing}${existing && !existing.endsWith('\n') ? '\n' : ''}${recordLine(record)}\n`
    this.writeTextAtomically(this.recordsPath, content)
  }

  /** Refuse known checkout obstructions before any workspace files are written. */
  private async preflightRestorePaths(paths: readonly string[]): Promise<void> {
    const checkedParents = new Set<string>()
    for (const path of paths) {
      const parts = path.split('/')
      if (isAbsolute(path) || parts.some(part => !part || part === '.' || part === '..')) {
        throw new Error(`Unsafe snapshot path: ${path}`)
      }
      let currentPath = this.workspaceDirectory
      for (let index = 0; index < parts.length; index++) {
        currentPath = join(currentPath, parts[index]!)
        const leaf = index === parts.length - 1
        if (!leaf && checkedParents.has(currentPath)) continue
        const current = await lstat(currentPath).catch(error => {
          if (hasErrorCode(error, 'ENOENT')) return undefined
          throw error
        })
        if (!current) break
        if (!leaf && !current.isDirectory()) {
          throw new Error(`Restore refused: ancestor of ${path} is not a real directory. Workspace files were not changed.`)
        }
        if (leaf && (!current.isFile() && !current.isSymbolicLink())) {
          throw new Error(`Restore refused: ${path} is a directory or special file. Workspace files were not changed.`)
        }
        if (!leaf) checkedParents.add(currentPath)
      }
    }
  }

  /** Resolve a caller-supplied path to a workspace-relative, git-usable path. */
  private workspaceRelativePath(candidate: string): string {
    const trimmed = candidate.trim()
    if (!trimmed) throw new Error('a file path is required')
    const relativePath = relative(this.workspaceDirectory, resolve(this.workspaceDirectory, candidate))
    // A `../` path would let a restore write anywhere the daemon can write,
    // driven by nothing more than a snapshot ref and an attacker-chosen path.
    if (!relativePath || relativePath === '..' || relativePath.startsWith(`..${sep}`) || isAbsolute(relativePath)) {
      throw new Error(`path escapes the snapshot workspace: ${candidate}`)
    }
    return relativePath.split(sep).join('/')
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
    // The state home can live inside the workspace (notably when started
    // from ~). Never snapshot credentials, transcripts, or the shadow repo
    // itself just because their directory has a nonstandard name.
    const statePatterns = [this.shadowRoot, xerxesHome()].flatMap(root => {
      const path = relative(this.workspaceDirectory, root)
      if (!path || path === '..' || path.startsWith(`..${sep}`) || isAbsolute(path)) return []
      return ['/' + path.split(sep).join('/').replace(/[\\*?\[\]]/gu, '\\$&') + '/']
    })
    const missing = [...SHADOW_EXCLUDE_PATTERNS, ...statePatterns].filter(pattern => !present.has(pattern))
    if (missing.length === 0) return
    const separator = existing.length > 0 && !existing.endsWith('\n') ? '\n' : ''
    writeFileSync(path, `${existing}${separator}${missing.join('\n')}\n`, 'utf8')
  }

  private writeRecords(records: readonly SnapshotRecord[]): void {
    this.writeTextAtomically(this.recordsPath, records.map(recordLine).join('\n'))
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

/** What a single-file restore replaced, and the snapshot that can undo it. */
export interface SnapshotRestoreResult {
  readonly path: string
  readonly previous: SnapshotRecord
  readonly snapshot: SnapshotRecord
}

/** One tab-separated record row; every field is scrubbed of the row separators. */
function recordLine(record: SnapshotRecord): string {
  return [
    record.id,
    record.label,
    record.commitSha,
    record.createdAt,
    record.workspaceDir,
    record.sessionId ?? '',
    record.turnIndex === undefined ? '' : String(record.turnIndex),
  ].map(field => field.replaceAll(/[\t\r\n]/g, ' ')).join('\t')
}

/** A turn index is only trusted when it survives a round trip as a non-negative integer. */
function parseTurnIndex(value: string | undefined): number | undefined {
  if (value === undefined || value.trim() === '') return undefined
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : undefined
}

/** Serialize one shadow repository across Xerxes processes and recover abandoned locks. */
async function withRepositoryLock<T>(path: string, operation: () => Promise<T>): Promise<T> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 })
  let handle: Awaited<ReturnType<typeof open>>
  for (;;) {
    try {
      handle = await open(path, 'wx', 0o600)
      break
    } catch (error) {
      if (!hasErrorCode(error, 'EEXIST')) throw error
      await recoverStaleRepositoryLock(path)
      await Bun.sleep(REPOSITORY_LOCK_WAIT_MS)
    }
  }
  try {
    await handle.writeFile(`${process.pid}\n`, 'utf8')
    return await operation()
  } finally {
    await handle.close()
    await rm(path, { force: true })
  }
}

async function recoverStaleRepositoryLock(path: string): Promise<void> {
  let lockStat: Awaited<ReturnType<typeof stat>>
  let pid: number
  try {
    lockStat = await stat(path)
    pid = Number.parseInt((await readFile(path, 'utf8')).trim(), 10)
  } catch (error) {
    if (hasErrorCode(error, 'ENOENT')) return
    return
  }
  if (Number.isSafeInteger(pid) && pid > 0) {
    try {
      process.kill(pid, 0)
      return
    } catch (error) {
      if (!hasErrorCode(error, 'ESRCH')) return
    }
  } else if (Date.now() - lockStat.mtimeMs < REPOSITORY_LOCK_STALE_MS) return
  // Rename first so a delayed owner cannot unlink a replacement lock created
  // after recovery. If another waiter won the rename, simply retry normally.
  const abandoned = `${path}.stale-${process.pid}-${randomUUID()}`
  try {
    await rename(path, abandoned)
    await rm(abandoned, { force: true })
  } catch (error) {
    if (!hasErrorCode(error, 'ENOENT')) throw error
  }
}

function hasErrorCode(error: unknown, code: string): boolean {
  return typeof error === 'object'
    && error !== null
    && 'code' in error
    && (error as { readonly code?: unknown }).code === code
}

/** Run one git invocation with a hard timeout, killing the process when it overruns. */
async function runGitProcess(
  args: readonly string[],
  options: { readonly cwd: string; readonly env: Record<string, string | undefined>; readonly maxOutputBytes?: number },
): Promise<string> {
  const child = Bun.spawn(['git', ...args], {
    cwd: options.cwd,
    env: options.env,
    stdout: 'pipe',
    stderr: 'pipe',
  })
  const readers = [child.stdout.getReader(), child.stderr.getReader()]
  const collect = async (reader: typeof readers[number], limit: number): Promise<string> => {
    const chunks: Uint8Array[] = []
    let size = 0
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      size += value.byteLength
      if (size > limit) throw new Error(`git output exceeded ${limit} bytes; narrow the snapshot changes before retrying`)
      chunks.push(value)
    }
    return Buffer.concat(chunks).toString('utf8')
  }
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    child.kill('SIGKILL')
    for (const reader of readers) void reader.cancel().catch(() => {})
  }, GIT_COMMAND_TIMEOUT_MS)
  try {
    const [stdout, stderr, exitCode] = await Promise.all([
      collect(readers[0]!, options.maxOutputBytes ?? 16 * 1024 * 1024),
      collect(readers[1]!, 64 * 1024),
      child.exited,
    ])
    if (timedOut) throw new Error(`git ${args.join(' ')} timed out after ${GIT_COMMAND_TIMEOUT_MS}ms`)
    if (exitCode !== 0) {
      throw new Error(`git ${args.join(' ')} failed (exit ${exitCode}): ${stderr.trim()}`)
    }
    return stdout
  } finally {
    clearTimeout(timer)
    if (child.exitCode === null) child.kill('SIGKILL')
    await Promise.allSettled(readers.map(reader => reader.cancel()))
  }
}

function workspaceHash(workspaceDirectory: string): string {
  return createHash('sha1').update(workspaceDirectory).digest('hex').slice(0, 12)
}

function restoreRevision(target: string, current: string): string {
  return createHash('sha256').update(target + ':' + current).digest('hex')
}

function fileRestoreRevision(target: string, path: string, current: string): string {
  return createHash('sha256').update(JSON.stringify([target, path, current])).digest('hex')
}
