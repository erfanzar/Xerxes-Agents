// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { createHash, randomUUID } from 'node:crypto'
import { constants } from 'node:fs'
import { chmod, lstat, link, mkdir, open, readdir, readlink, realpath, rename, symlink, unlink } from 'node:fs/promises'
import { dirname, isAbsolute, join } from 'node:path'
import { hostname } from 'node:os'
import { Database } from 'bun:sqlite'
import { openRegularFile } from '../security/regularFile.js'

export type WorkspaceGit = (args: readonly string[], cwd?: string, env?: Record<string, string>, raw?: boolean) => Promise<string>
interface FileState { path: string; kind: 'missing' | 'file' | 'symlink'; hash?: string; mode?: number; target?: string }
interface CapturedFile { state: FileState; bytes?: Uint8Array }
export interface WorkspaceApplyResult { id: string; status: 'applied'; destination: string; backupPath: string }
const hash = (value: string | Uint8Array) => createHash('sha256').update(value).digest('hex')
const missing = (error: unknown) => (error as NodeJS.ErrnoException).code === 'ENOENT'

async function safePath(root: string, path: string) {
  const parts = path.split('/')
  if (isAbsolute(path) || path.includes('\\') || parts.some(part => !part || part === '.' || part === '..' || part.toLowerCase() === '.git' || part.includes(':'))) throw new Error('Unsupported integration path: ' + path)
  let parent = root
  for (const part of parts.slice(0, -1)) {
    parent = join(parent, part)
    const info = await lstat(parent).catch(error => { if (missing(error)) return null; throw error })
    if (info && !info.isDirectory()) throw new Error('Integration path has a non-directory ancestor: ' + path)
  }
  return join(root, ...parts)
}
async function capture(root: string, paths: readonly string[]): Promise<CapturedFile[]> {
  let bytes = 0
  const result: CapturedFile[] = []
  for (const path of paths) {
    const target = await safePath(root, path)
    const info = await lstat(target).catch(error => { if (missing(error)) return null; throw error })
    if (!info) { result.push({ state: { path, kind: 'missing' } }); continue }
    if (info.isSymbolicLink()) { result.push({ state: { path, kind: 'symlink', target: await readlink(target) } }); continue }
    if (!info.isFile() || info.nlink > 1) throw new Error('Integration requires an ordinary file without hard links: ' + path)
    const file = await openRegularFile(target, constants.O_RDONLY)
    const chunks: Uint8Array[] = []
    try {
      for (;;) {
        const buffer = new Uint8Array(65536)
        const read = await file.read(buffer)
        if (!read.bytesRead) break
        bytes += read.bytesRead
        if (bytes > 64 * 1024 * 1024) throw new Error('Integration backup exceeds 64 MiB')
        chunks.push(buffer.subarray(0, read.bytesRead))
      }
    } finally { await file.close() }
    const content = Buffer.concat(chunks)
    result.push({ state: { path, kind: 'file', hash: hash(content), mode: info.mode & 0o777 }, bytes: content })
  }
  return result
}
function stateId(destination: string, head: string, files: readonly CapturedFile[]) {
  return hash(JSON.stringify({ destination, head, files: files.map(file => file.state) }))
}
async function durableWrite(path: string, bytes: string | Uint8Array) {
  const file = await openRegularFile(path, constants.O_WRONLY | constants.O_CREAT | constants.O_TRUNC)
  try { await file.writeFile(bytes); await file.sync() } finally { await file.close() }
}
async function syncDirectory(path: string) {
  if (process.platform === 'win32') return
  const directory = await open(path, 'r')
  try { await directory.sync() } finally { await directory.close() }
}
async function writeRecord(directory: string, value: unknown) {
  const temporary = join(directory, 'record-' + randomUUID() + '.tmp')
  await durableWrite(temporary, JSON.stringify(value))
  await rename(temporary, join(directory, 'record.json'))
  await syncDirectory(directory)
}
async function publishLock(directory: string, lockPath: string, owner: unknown) {
  const candidate = join(directory, 'lock-owner-' + randomUUID() + '.json')
  await durableWrite(candidate, JSON.stringify(owner))
  try { await link(candidate, lockPath) }
  catch (error) { await unlink(candidate); throw error }
  await unlink(candidate)
  await syncDirectory(dirname(lockPath))
}
async function patchPaths(git: WorkspaceGit, destination: string, patch: string) {
  const output = await git(['apply', '--numstat', '-z', patch], destination, {}, true)
  const paths = output.split('\0').filter(Boolean).map(entry => {
    const match = entry.match(/^(?:\d+|-)\t(?:\d+|-)\t([\s\S]+)$/)
    if (!match) throw new Error('Unsupported integration patch record')
    return match[1]!
  })
  if (!paths.length || paths.length > 1000) throw new Error('Integration requires between 1 and 1000 changed paths')
  return [...new Set(paths)].sort()
}
export async function inspectDestination(git: WorkspaceGit, destination: string, patch: string) {
  const root = await realpath(destination)
  const head = await git(['rev-parse', 'HEAD'], root)
  const paths = await patchPaths(git, root, patch)
  const files = await capture(root, paths)
  return { destination: root, head, paths, files, state: stateId(root, head, files) }
}

interface IntegrationRecord {
  id: string; reviewId: string; destination: string; head: string
  original: FileState[]; expected: FileState[]
  status: 'preparing' | 'abandoned' | 'prepared' | 'applied' | 'needs-recovery' | 'rolled-back'
}
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
const SHA256 = /^[0-9a-f]{64}$/

async function readPrivateFile(path: string, limit: number, lockRecord = false): Promise<Buffer> {
  const file = await openRegularFile(path, constants.O_RDONLY | (constants.O_NONBLOCK ?? 0))
  try {
    const info = await file.stat()
    if (!info.isFile() || (!lockRecord && info.nlink !== 1) || info.size > limit) throw new Error('Invalid integration artifact: ' + path)
    const bytes = Buffer.alloc(info.size + 1)
    let offset = 0
    while (offset < bytes.length) {
      const read = await file.read(bytes, offset, bytes.length - offset)
      if (!read.bytesRead) break
      offset += read.bytesRead
    }
    if (offset !== info.size) throw new Error('Integration artifact changed while reading: ' + path)
    return bytes.subarray(0, offset)
  } finally { await file.close() }
}

async function readIntegration(storage: string, id: string, destination: string, verifyBackups = true) {
  if (!UUID.test(id)) throw new Error('Invalid integration id')
  const directory = join(storage, 'integration-' + id)
  if (await realpath(directory) !== directory) throw new Error('Integration artifact directory is redirected')
  const value: unknown = JSON.parse((await readPrivateFile(join(directory, 'record.json'), 1024 * 1024)).toString())
  if (!value || typeof value !== 'object') throw new Error('Invalid integration record')
  const row = value as Record<string, unknown>
  const root = await realpath(destination)
  if (row.id !== id || row.destination !== root || typeof row.reviewId !== 'string' || !SHA256.test(row.reviewId) ||
    typeof row.head !== 'string' || !/^[0-9a-f]{40,64}$/.test(row.head) ||
    !['preparing', 'abandoned', 'prepared', 'applied', 'needs-recovery', 'rolled-back'].includes(String(row.status))) throw new Error('Invalid integration ownership or state')
  const untouched = row.status === 'preparing' || row.status === 'abandoned'
  for (const key of ['original', 'expected']) {
    const states = row[key]
    if (!Array.isArray(states) || (untouched ? states.length !== 0 : !states.length) || states.length > 1000) throw new Error('Invalid integration file states')
    const paths = new Set<string>()
    for (const state of states) {
      if (!state || typeof state !== 'object' || typeof state.path !== 'string' || paths.has(state.path)) throw new Error('Invalid integration file state')
      await safePath(root, state.path)
      paths.add(state.path)
      if (state.kind === 'file') {
        if (typeof state.hash !== 'string' || !SHA256.test(state.hash) || !Number.isInteger(state.mode) || state.mode < 0 || state.mode > 0o777) throw new Error('Invalid integration file fingerprint')
      } else if (state.kind === 'symlink') {
        if (typeof state.target !== 'string' || state.target.includes('\0')) throw new Error('Invalid integration link')
      } else if (state.kind !== 'missing') throw new Error('Invalid integration file kind')
    }
  }
  const record = row as unknown as IntegrationRecord
  if (record.original.length !== record.expected.length || record.original.some((state, i) => state.path !== record.expected[i]!.path)) throw new Error('Integration paths do not match')
  let size = 0
  const original: CapturedFile[] = []
  for (const [index, state] of record.original.entries()) {
    if (!verifyBackups) { original.push({ state }); continue }
    const bytes = state.kind === 'file' ? await readPrivateFile(join(directory, `original-${index}`), 64 * 1024 * 1024 - size) : undefined
    if (bytes) {
      size += bytes.length
      if (hash(bytes) !== state.hash) throw new Error('Integration backup checksum mismatch: ' + state.path)
    }
    original.push(bytes ? { state, bytes } : { state })
  }
  return { directory, record, original }
}

export interface WorkspaceIntegrationRecord {
  id: string; backupPath: string; destination?: string; status?: IntegrationRecord['status']; paths?: string[]; error?: string
}
export async function inspectWorkspaceIntegration(git: WorkspaceGit, storage: string, destination: string, id: string) {
  const { record, directory } = await readIntegration(storage, id, destination)
  const currentHead = await git(['rev-parse', 'HEAD'], record.destination)
  const terminal = ['applied', 'rolled-back', 'abandoned'].includes(record.status)
  const files: Array<{ path: string; action: 'preserve' | 'restore' | 'unchanged' | 'conflict'; reason: string }> = []
  for (const [index, before] of record.original.entries()) {
    if (terminal) { files.push({ path: before.path, action: 'preserve', reason: 'Operation completed; recovery only releases its lock' }); continue }
    if (currentHead !== record.head) { files.push({ path: before.path, action: 'conflict', reason: 'Destination HEAD changed' }); continue }
    try {
      const current = (await capture(record.destination, [before.path]))[0]!.state
      const action = JSON.stringify(current) === JSON.stringify(before) ? 'unchanged' : JSON.stringify(current) === JSON.stringify(record.expected[index]) ? 'restore' : 'conflict'
      files.push({ path: before.path, action, reason: action === 'unchanged' ? 'Already matches original' : action === 'restore' ? 'Matches prepared result' : 'Contains newer or different content' })
    } catch (error) { files.push({ path: before.path, action: 'conflict', reason: String(error) }) }
  }
  return { id, destination: record.destination, status: record.status, backupPath: directory, files, checkedAt: new Date().toISOString(), headChanged: currentHead !== record.head }
}
export async function listWorkspaceIntegrations(storage: string, destination: string, after?: string) {
  if (after !== undefined && !UUID.test(after)) throw new Error('Invalid integration cursor')
  const entries = await readdir(storage).catch(error => { if (missing(error)) return []; throw error })
  if (entries.length > 4096) throw new Error('Workspace artifact inventory exceeds 4096 entries')
  const ids = entries.filter(name => name.startsWith('integration-') && UUID.test(name.slice(12))).map(name => name.slice(12)).sort().filter(id => after === undefined || id > after)
  const records: WorkspaceIntegrationRecord[] = []
  for (const id of ids.slice(0, 100)) {
    const backupPath = join(storage, 'integration-' + id)
    try {
      const { record } = await readIntegration(storage, id, destination, false)
      records.push({ id, backupPath, destination: record.destination, status: record.status, paths: record.original.map(file => file.path) })
    } catch (error) { records.push({ id, backupPath, error: String(error) }) }
  }
  return { records, ...(ids.length > 100 ? { next: ids[99]! } : {}) }
}

async function restoreOriginal(git: WorkspaceGit, destination: string, head: string, original: readonly CapturedFile[], expected: readonly FileState[]) {
  const conflicts: string[] = []
  const headChanged = await git(['rev-parse', 'HEAD'], destination).then(value => value !== head).catch(() => true)
  for (const [index, before] of original.entries()) {
    if (headChanged) { conflicts.push(before.state.path); continue }
    try {
      const current = (await capture(destination, [before.state.path]))[0]!
      if (JSON.stringify(current.state) === JSON.stringify(before.state)) continue
      if (JSON.stringify(current.state) !== JSON.stringify(expected[index])) { conflicts.push(before.state.path); continue }
      const target = await safePath(destination, before.state.path)
      if (before.state.kind === 'missing') await unlink(target)
      else {
        await mkdir(dirname(target), { recursive: true })
        const temporary = join(dirname(target), '.xerxes-restore-' + randomUUID())
        if (before.state.kind === 'file') { await durableWrite(temporary, before.bytes!); await chmod(temporary, before.state.mode!) }
        else await symlink(before.state.target!, temporary)
        await rename(temporary, target)
      }
      await syncDirectory(dirname(target))
    } catch { conflicts.push(before.state.path) }
  }
  return conflicts
}

/** Explicit rollback of an interrupted apply. Never steals a live or unverifiable owner's lock. */
export async function recoverWorkspaceApply(options: { git: WorkspaceGit; destination: string; storage: string; id: string }) {
  // Validate every backup before acquiring ownership or writing any destination file.
  await readIntegration(options.storage, options.id, options.destination)
  const lockPath = join(options.storage, 'integration.lock')
  // SQLite's OS lock is released on process death. Never delete this database:
  // replacing its inode would let two recovering processes hold different locks.
  const legacy = await lstat(join(options.storage, 'integration-recovery.lock')).catch(error => { if (missing(error)) return null; throw error })
  if (legacy) throw new Error('Legacy recovery guard requires owner inspection before migration')
  const guardPath = join(options.storage, 'integration-recovery.sqlite')
  const guardFile = await openRegularFile(guardPath, constants.O_RDWR | constants.O_CREAT | (constants.O_NONBLOCK ?? 0), 0o600)
  try {
    const info = await guardFile.stat()
    if (!info.isFile() || info.nlink !== 1) throw new Error('Invalid recovery guard file')
  } finally { await guardFile.close() }
  const guard = new Database(guardPath, { strict: true })
  try { guard.exec('PRAGMA busy_timeout=0; BEGIN IMMEDIATE') }
  catch (error) { guard.close(); throw new Error('Another recovery is active or its guard is unavailable', { cause: error }) }
  let owned = false
  const token = randomUUID()
  try {
    const held = await readPrivateFile(lockPath, 4096, true).catch(error => { if (missing(error)) return null; throw error })
    if (held) {
      const owner = JSON.parse(held.toString()) as { id?: unknown; pid?: unknown; host?: unknown }
      if (owner.id !== options.id || owner.host !== hostname() || typeof owner.pid !== 'number' || !Number.isSafeInteger(owner.pid) || owner.pid <= 0) throw new Error('Integration lock owner cannot be verified')
      try { process.kill(owner.pid, 0); throw new Error('Integration owner is still running') }
      catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ESRCH') throw error }
      // The old integration lock remains in place throughout ownership transfer.
      const temporary = join(options.storage, 'recovery-owner-' + token)
      await durableWrite(temporary, JSON.stringify({ id: options.id, token, pid: process.pid, host: hostname() }))
      await rename(temporary, lockPath)
    } else {
      await publishLock(join(options.storage, 'integration-' + options.id), lockPath, { id: options.id, token, pid: process.pid, host: hostname() })
    }
    owned = true
    // Re-read after acquiring ownership: an apply may have completed since
    // initial inspection. Never turn a successful apply into a rollback.
    const { directory, record, original } = await readIntegration(options.storage, options.id, options.destination)
    if (record.status === 'applied' || record.status === 'rolled-back') return { id: options.id, status: record.status, conflicts: [], backupPath: directory }
    if (record.status === 'preparing' || record.status === 'abandoned') {
      await writeRecord(directory, { ...record, status: 'abandoned', recoveredAt: new Date().toISOString() })
      return { id: options.id, status: 'abandoned', conflicts: [], backupPath: directory }
    }
    const conflicts = await restoreOriginal(options.git, record.destination, record.head, original, record.expected)
    const status = conflicts.length ? 'needs-recovery' : 'rolled-back'
    await writeRecord(directory, { ...record, status, conflicts, recoveredAt: new Date().toISOString() })
    return { id: options.id, status, conflicts, backupPath: directory }
  } finally {
    try {
      if (owned) {
        const owner = JSON.parse((await readPrivateFile(lockPath, 4096, true)).toString()) as { token?: unknown }
        if (owner.token !== token) throw new Error('Recovery lock ownership changed')
        await unlink(lockPath); await syncDirectory(options.storage)
      }
    } finally { guard.close() }
  }
}

/** Applies a checked patch with durable original/expected file states and conservative rollback. */
export async function applyWorkspacePatch(options: {
  git: WorkspaceGit; destination: string; storage: string; patch: string; destinationState: string; reviewId: string
}): Promise<WorkspaceApplyResult> {
  const { git } = options
  const id = randomUUID(), directory = join(options.storage, 'integration-' + id)
  if (!SHA256.test(options.reviewId) || !SHA256.test(options.destinationState)) throw new Error('Invalid checked integration identity')
  const destination = await realpath(options.destination)
  const head = await git(['rev-parse', 'HEAD'], destination)
  // Publish the no-writes state before acquiring a lock. A crash at any later
  // preparation boundary leaves an identifiable, safely abandonable operation.
  await mkdir(directory, { mode: 0o700 })
  await writeRecord(directory, { id, reviewId: options.reviewId, destination, head, original: [], expected: [], status: 'preparing', at: new Date().toISOString() })
  await syncDirectory(options.storage)
  const lockPath = join(options.storage, 'integration.lock')
  await publishLock(directory, lockPath, { id, pid: process.pid, host: hostname() }).catch(error => { throw new Error('Workspace integration is locked; inspect any interrupted integration before retrying', { cause: error }) })
  try {
    const preparation = await readIntegration(options.storage, id, destination)
    if (preparation.record.status !== 'preparing') throw new Error('Integration preparation was already recovered; check again')
    const patch = join(directory, 'changes.patch')
    await durableWrite(patch, options.patch)
    const original = await inspectDestination(git, options.destination, patch)
    if (original.state !== options.destinationState) throw new Error('Destination changed since integration check; check again')
    await git(['apply', '--check', '--whitespace=nowarn', patch], original.destination)
    const stage = join(directory, 'stage')
    await mkdir(stage, { mode: 0o700 })
    for (let index = 0; index < original.files.length; index++) {
      const file = original.files[index]!
      const target = await safePath(stage, file.state.path)
      if (file.state.kind === 'missing') continue
      await mkdir(dirname(target), { recursive: true })
      if (file.state.kind === 'file') {
        await durableWrite(join(directory, `original-${index}`), file.bytes!)
        await Bun.write(target, file.bytes!); await chmod(target, file.state.mode!)
      } else await symlink(file.state.target!, target)
    }
    // Unchanged attributes still control touched files. Copy them into the
    // prediction tree, including attributes in ancestor directories.
    const attributes = new Set<string>(['.gitattributes'])
    for (const path of original.paths) {
      let parent = dirname(path)
      while (parent !== '.') {
        attributes.add(join(parent, '.gitattributes'))
        const next = dirname(parent)
        if (next === parent) break
        parent = next
      }
    }
    for (const path of attributes) {
      if (original.paths.includes(path)) continue
      const source = await safePath(original.destination, path)
      let bytes: Uint8Array
      try {
        const handle = await openRegularFile(source, constants.O_RDONLY)
        try { bytes = await handle.readFile() } finally { await handle.close() }
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') continue
        throw error
      }
      const target = await safePath(stage, path)
      await mkdir(dirname(target), { recursive: true })
      await durableWrite(target, bytes)
    }
    await git(['-c', 'init.templateDir=', 'init', stage], original.destination)
    // Git's working-tree conversions are repository-local. A fresh staging
    // repository must use the destination's EOL policy before predicting bytes.
    for (const key of ['core.autocrlf', 'core.eol']) {
      const value = await git(['config', '--get', key], original.destination).catch(() => '')
      if (value) await git(['config', key, value], stage)
    }
    await git(['apply', '--whitespace=nowarn', patch], stage)
    const expected = await capture(stage, original.paths)
    const record = { id, reviewId: options.reviewId, destination: original.destination, head: original.head, original: original.files.map(file => file.state), expected: expected.map(file => file.state), status: 'prepared', at: new Date().toISOString() }
    await writeRecord(directory, record)
    if ((await Bun.file(lockPath).json()).id !== id) throw new Error('Integration lock ownership changed')
    const unchanged = await inspectDestination(git, original.destination, patch)
    if (unchanged.state !== original.state) throw new Error('Destination changed during preparation; no files applied')
    try {
      await git(['apply', '--whitespace=nowarn', patch], original.destination)
      const actual = await capture(original.destination, original.paths)
      if (JSON.stringify(actual.map(file => file.state)) !== JSON.stringify(expected.map(file => file.state))) throw new Error('Destination differs from the prepared result')
      if (await git(['rev-parse', 'HEAD'], original.destination) !== original.head) throw new Error('Destination HEAD changed during integration')
      for (const file of actual) {
        const path = await safePath(original.destination, file.state.path)
        if (file.state.kind === 'file') { const handle = await openRegularFile(path, constants.O_RDWR); try { await handle.sync() } finally { await handle.close() } }
        await syncDirectory(dirname(path))
      }
      await writeRecord(directory, { ...record, status: 'applied' })
      return { id, status: 'applied', destination: original.destination, backupPath: directory }
    } catch (failure) {
      const conflicts = await restoreOriginal(git, original.destination, original.head, original.files, expected.map(file => file.state))
      const status = conflicts.length ? 'needs-recovery' : 'rolled-back'
      await writeRecord(directory, { ...record, status, conflicts, error: String(failure) })
      throw new Error(`Integration failed (${status}); backup: ${directory}${conflicts.length ? '; preserve concurrent changes in: ' + conflicts.join(', ') : ''}`, { cause: failure })
    }
  } finally {
    const held = await Bun.file(lockPath).json() as { id?: string }
    if (held.id !== id) throw new Error('Integration lock ownership changed; preserving the current lock')
    await unlink(lockPath)
  }
}
