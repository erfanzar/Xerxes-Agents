// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { copyFile, lstat, mkdir, readFile, readdir, realpath, rename, rmdir, rm, unlink } from 'node:fs/promises'
import { homedir } from 'node:os'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

import { xerxesSubdir } from '../core/paths.js'
import { PathEscape, resolveWithin } from '../security/pathSecurity.js'
import { scanContextContent } from '../security/promptScanner.js'

/** Repositories whose supplied skill content is accepted as a trusted source. */
export const TRUSTED_REPOS = new Set<string>(['erfanzar/xerxes'])

export const SKILLS_DIR = xerxesSubdir('skills')
export const HUB_DIR = join(SKILLS_DIR, '.hub')
export const QUARANTINE_DIR = join(HUB_DIR, 'quarantine')
export const TRUSTED_HASHES_PATH = join(HUB_DIR, 'trusted_hashes.json')

/** Raised when a guard operation would follow a symlink or escape its configured skill roots. */
export class SkillGuardPathError extends Error {
  readonly path: string

  constructor(path: string, reason: string) {
    super(`unsafe skill path ${JSON.stringify(path)}: ${reason}`)
    this.name = 'SkillGuardPathError'
    this.path = path
  }
}

export interface SkillGuardPaths {
  /** Explicit active-skills root, primarily for isolated hosts and tests. */
  readonly skillsDirectory?: string
  /** Explicit quarantine root. Defaults to <skillsDirectory>/.hub/quarantine. */
  readonly quarantineDirectory?: string
}

export interface ScanSkillOptions {
  readonly sourceRepo?: string
  readonly trustedHashes?: Readonly<Record<string, string>>
  readonly trustedRepos?: ReadonlySet<string>
}

interface ScanResultOptions {
  readonly hashMismatch?: boolean
  readonly injectionDetected?: boolean
  readonly reasons?: readonly string[]
  readonly untrustedSource?: boolean
}

/** Outcome of scanning a local skill for injected content and trust violations. */
export class ScanResult {
  readonly hashMismatch: boolean
  readonly injectionDetected: boolean
  readonly isSafe: boolean
  readonly reasons: readonly string[]
  readonly untrustedSource: boolean

  constructor(options: ScanResultOptions = {}) {
    this.reasons = [...(options.reasons ?? [])]
    this.hashMismatch = options.hashMismatch ?? false
    this.injectionDetected = options.injectionDetected ?? false
    this.untrustedSource = options.untrustedSource ?? false
    this.isSafe = this.reasons.length === 0
  }

  get summary(): string {
    return this.isSafe ? 'Safe' : this.reasons.join('; ') || 'Unsafe'
  }
}

/** Return the SHA-256 digest of one regular local file without following symlinks. */
export async function hashSkillFile(path: string): Promise<string> {
  const file = await requireRegularFile(normalizePath(path), 'skill file')
  return digest(await readFile(file))
}

/**
 * Return a deterministic SHA-256 digest for every regular file below a skill directory.
 *
 * Relative POSIX paths and bytes are fed in lexical order, matching the Python guard's
 * directory digest while rejecting symlink traversal.
 */
export async function hashSkillDirectory(path: string): Promise<string> {
  const root = await requireDirectory(normalizePath(path), 'skill directory')
  const hash = createHash('sha256')
  await hashDirectoryTree(root, root, hash)
  return hash.digest('hex')
}

/** Load the local trusted-hash database, returning an empty map for absent or malformed JSON. */
export async function loadTrustedHashes(paths: SkillGuardPaths = {}): Promise<Record<string, string>> {
  const skillsDirectory = normalizePath(paths.skillsDirectory ?? SKILLS_DIR)
  const root = await existingDirectory(skillsDirectory, 'skills directory')
  if (root === undefined) return {}
  const hub = await existingChildDirectory(root, '.hub', 'skills hub directory')
  if (hub === undefined) return {}
  const hashFile = await existingRegularFile(hub, 'trusted_hashes.json', 'trusted hash database')
  if (hashFile === undefined) return {}

  try {
    return parseTrustedHashes(await readFile(hashFile, 'utf8'))
  } catch {
    return {}
  }
}

/** Persist trusted content digests atomically below the configured skills hub. */
export async function saveTrustedHashes(
  data: Readonly<Record<string, string>>,
  paths: SkillGuardPaths = {},
): Promise<void> {
  const normalized = normalizeTrustedHashes(data)
  const root = await ensureDirectory(normalizePath(paths.skillsDirectory ?? SKILLS_DIR), 'skills directory')
  const hub = await ensureChildDirectory(root, '.hub', 'skills hub directory')
  await writeDirectFile(hub, 'trusted_hashes.json', JSON.stringify(normalized, null, 2), 'trusted hash database')
}

/** Run prompt-injection, trusted-hash, and source-repository checks for one skill. */
export async function scanSkill(skillPath: string, options: ScanSkillOptions = {}): Promise<ScanResult> {
  let skill: SkillMarkdown | undefined
  try {
    skill = await skillMarkdownPath(skillPath)
  } catch (error) {
    return unreadableSkillResult(error)
  }
  if (skill === undefined) {
    return new ScanResult({ reasons: ['Missing SKILL.md'] })
  }

  let content: string
  try {
    content = await readFile(skill.path, 'utf8')
  } catch (error) {
    return unreadableSkillResult(error)
  }

  const reasons: string[] = []
  let injectionDetected = false
  let hashMismatch = false
  let untrustedSource = false
  if (scanContextContent(content, skill.path).includes('[BLOCKED:')) {
    injectionDetected = true
    reasons.push('Prompt injection detected in SKILL.md')
  }

  if (options.trustedHashes !== undefined) {
    try {
      const currentHash = await hashSkillFile(skill.path)
      const expected = options.trustedHashes[skill.trustedHashKey] ?? options.trustedHashes[skill.path]
      if (expected !== undefined && currentHash !== expected) {
        hashMismatch = true
        reasons.push('Content hash mismatch')
      }
    } catch (error) {
      return unreadableSkillResult(error)
    }
  }

  if (options.sourceRepo !== undefined && !(options.trustedRepos ?? TRUSTED_REPOS).has(options.sourceRepo)) {
    untrustedSource = true
    reasons.push(`Source repo '${options.sourceRepo}' not in trusted list`)
  }

  return new ScanResult({ reasons, hashMismatch, injectionDetected, untrustedSource })
}

/** Move an active direct-child skill directory into the contained quarantine directory. */
export async function quarantineSkill(skillPath: string, paths: SkillGuardPaths = {}): Promise<string> {
  const skillsRoot = await requireDirectory(normalizePath(paths.skillsDirectory ?? SKILLS_DIR), 'skills directory')
  const skillName = await skillNameFromPath(skillsRoot, skillPath)
  const source = await requireChildDirectory(skillsRoot, skillName, 'active skill directory')
  const quarantineRoot = await ensureQuarantineDirectory(skillsRoot, paths)
  const destination = await containedChildPath(quarantineRoot, skillName, 'quarantined skill directory')
  if (isWithin(source, quarantineRoot) || isWithin(quarantineRoot, source)) {
    throw new SkillGuardPathError(quarantineRoot, 'quarantine directory must not overlap the skill being quarantined')
  }
  await replaceChildSkillDirectory(quarantineRoot, skillName, 'quarantined skill directory')
  await moveDirectory(source, destination, 'quarantine skill')
  return destination
}

/** Restore a named quarantined skill to the active skills root. */
export async function approveSkill(skillName: string, paths: SkillGuardPaths = {}): Promise<string> {
  const name = validateSkillName(skillName)
  const skillsRoot = await ensureDirectory(normalizePath(paths.skillsDirectory ?? SKILLS_DIR), 'skills directory')
  const quarantineRoot = await existingQuarantineDirectory(skillsRoot, paths)
  if (quarantineRoot === undefined) {
    return `[Error] Skill '${name}' not found in quarantine.`
  }

  const quarantined = await existingChildDirectory(quarantineRoot, name, 'quarantined skill directory')
  if (quarantined === undefined) {
    return `[Error] Skill '${name}' not found in quarantine.`
  }
  const destination = await containedChildPath(skillsRoot, name, 'active skill directory')
  if (isWithin(destination, quarantined) || isWithin(quarantined, destination)) {
    throw new SkillGuardPathError(quarantined, 'quarantined skill must not overlap its active destination')
  }
  // Set any active sibling aside first so a failed move can restore it instead of losing it.
  const replaced = await existingChildDirectory(skillsRoot, name, 'active skill directory')
  let backup: { readonly path: string; readonly restoreTo: string } | undefined
  if (replaced !== undefined) {
    const backupPath = await containedChildPath(skillsRoot, `.${name}.approve-${crypto.randomUUID()}.bak`, 'replaced skill backup')
    await moveDirectory(replaced, backupPath, 'set aside active skill')
    backup = { path: backupPath, restoreTo: replaced }
  }
  try {
    await moveDirectory(quarantined, destination, 'approve skill')
  } catch (error) {
    if (backup !== undefined) await moveDirectory(backup.path, backup.restoreTo, 'restore active skill')
    throw error
  }
  if (backup !== undefined) await removeDirectoryTree(skillsRoot, backup.path)
  return `Approved and activated skill '${name}'`
}

interface SkillMarkdown {
  readonly path: string
  readonly trustedHashKey: string
}

async function skillMarkdownPath(input: string): Promise<SkillMarkdown | undefined> {
  const normalized = normalizePath(input)
  let metadata
  try {
    metadata = await lstat(normalized)
  } catch (error) {
    if (isMissing(error)) return undefined
    throw pathError(normalized, 'cannot inspect skill path', error)
  }
  if (metadata.isSymbolicLink()) {
    throw new SkillGuardPathError(normalized, 'skill path must not be a symbolic link')
  }
  if (metadata.isFile()) {
    if (basename(normalized) !== 'SKILL.md') return undefined
    return { path: await requireRegularFile(normalized, 'SKILL.md'), trustedHashKey: normalized }
  }
  if (!metadata.isDirectory()) return undefined
  const directory = await requireDirectory(normalized, 'skill directory')
  const path = await existingRegularFile(directory, 'SKILL.md', 'SKILL.md')
  return path === undefined ? undefined : { path, trustedHashKey: join(normalized, 'SKILL.md') }
}

async function ensureQuarantineDirectory(skillsRoot: string, paths: SkillGuardPaths): Promise<string> {
  if (paths.quarantineDirectory !== undefined) {
    return ensureDirectory(normalizePath(paths.quarantineDirectory), 'quarantine directory')
  }
  const hub = await ensureChildDirectory(skillsRoot, '.hub', 'skills hub directory')
  return ensureChildDirectory(hub, 'quarantine', 'quarantine directory')
}

async function existingQuarantineDirectory(skillsRoot: string, paths: SkillGuardPaths): Promise<string | undefined> {
  if (paths.quarantineDirectory !== undefined) {
    return existingDirectory(normalizePath(paths.quarantineDirectory), 'quarantine directory')
  }
  const hub = await existingChildDirectory(skillsRoot, '.hub', 'skills hub directory')
  if (hub === undefined) return undefined
  return existingChildDirectory(hub, 'quarantine', 'quarantine directory')
}

async function hashDirectoryTree(root: string, directory: string, hash: ReturnType<typeof createHash>): Promise<void> {
  let entries
  try {
    entries = await readdir(directory, { withFileTypes: true })
  } catch (error) {
    throw pathError(directory, 'cannot list skill directory', error)
  }

  const names = entries.map(entry => entry.name).sort(compareLexical)
  for (const name of names) {
    const candidate = join(directory, name)
    let metadata
    try {
      metadata = await lstat(candidate)
    } catch (error) {
      throw pathError(candidate, 'cannot inspect skill entry', error)
    }
    if (metadata.isSymbolicLink()) {
      throw new SkillGuardPathError(candidate, 'skill directory must not contain symbolic links')
    }
    if (metadata.isDirectory()) {
      const child = await containedPath(root, candidate, 'skill directory entry')
      await hashDirectoryTree(root, child, hash)
      continue
    }
    if (!metadata.isFile()) continue
    const file = await containedPath(root, candidate, 'skill file')
    const relativeName = relative(root, candidate).split(sep).join('/')
    hash.update(relativeName, 'utf8')
    try {
      hash.update(await readFile(file))
    } catch (error) {
      throw pathError(file, 'cannot read skill file', error)
    }
  }
}

async function requireDirectory(path: string, label: string): Promise<string> {
  let metadata
  try {
    metadata = await lstat(path)
  } catch (error) {
    if (isMissing(error)) throw new SkillGuardPathError(path, `${label} does not exist`)
    throw pathError(path, `cannot inspect ${label}`, error)
  }
  if (metadata.isSymbolicLink()) {
    throw new SkillGuardPathError(path, `${label} must not be a symbolic link`)
  }
  if (!metadata.isDirectory()) {
    throw new SkillGuardPathError(path, `${label} must be a directory`)
  }
  try {
    return await realpath(path)
  } catch (error) {
    throw pathError(path, `cannot resolve ${label}`, error)
  }
}

async function existingDirectory(path: string, label: string): Promise<string | undefined> {
  try {
    await lstat(path)
  } catch (error) {
    if (isMissing(error)) return undefined
    throw pathError(path, `cannot inspect ${label}`, error)
  }
  return requireDirectory(path, label)
}

async function ensureDirectory(path: string, label: string): Promise<string> {
  const existing = await existingDirectory(path, label)
  if (existing !== undefined) return existing
  try {
    await mkdir(path, { recursive: true })
  } catch (error) {
    throw pathError(path, `cannot create ${label}`, error)
  }
  return requireDirectory(path, label)
}

async function requireChildDirectory(root: string, name: string, label: string): Promise<string> {
  const directory = await existingChildDirectory(root, name, label)
  if (directory === undefined) {
    throw new SkillGuardPathError(join(root, name), `${label} does not exist`)
  }
  return directory
}

async function existingChildDirectory(root: string, name: string, label: string): Promise<string | undefined> {
  const candidate = directChildPath(root, name, label)
  let metadata
  try {
    metadata = await lstat(candidate)
  } catch (error) {
    if (isMissing(error)) return undefined
    throw pathError(candidate, `cannot inspect ${label}`, error)
  }
  if (metadata.isSymbolicLink()) {
    throw new SkillGuardPathError(candidate, `${label} must not be a symbolic link`)
  }
  if (!metadata.isDirectory()) {
    throw new SkillGuardPathError(candidate, `${label} must be a directory`)
  }
  return containedChildPath(root, name, label)
}

async function ensureChildDirectory(root: string, name: string, label: string): Promise<string> {
  const existing = await existingChildDirectory(root, name, label)
  if (existing !== undefined) return existing
  const candidate = directChildPath(root, name, label)
  try {
    await mkdir(candidate)
  } catch (error) {
    if (!isAlreadyExists(error)) {
      throw pathError(candidate, `cannot create ${label}`, error)
    }
  }
  return requireChildDirectory(root, name, label)
}

async function existingRegularFile(root: string, name: string, label: string): Promise<string | undefined> {
  const candidate = directChildPath(root, name, label)
  let metadata
  try {
    metadata = await lstat(candidate)
  } catch (error) {
    if (isMissing(error)) return undefined
    throw pathError(candidate, `cannot inspect ${label}`, error)
  }
  if (metadata.isSymbolicLink()) {
    throw new SkillGuardPathError(candidate, `${label} must not be a symbolic link`)
  }
  if (!metadata.isFile()) {
    throw new SkillGuardPathError(candidate, `${label} must be a regular file`)
  }
  return containedChildPath(root, name, label)
}

async function requireRegularFile(path: string, label: string): Promise<string> {
  let metadata
  try {
    metadata = await lstat(path)
  } catch (error) {
    if (isMissing(error)) throw new SkillGuardPathError(path, `${label} does not exist`)
    throw pathError(path, `cannot inspect ${label}`, error)
  }
  if (metadata.isSymbolicLink()) {
    throw new SkillGuardPathError(path, `${label} must not be a symbolic link`)
  }
  if (!metadata.isFile()) {
    throw new SkillGuardPathError(path, `${label} must be a regular file`)
  }
  try {
    return await realpath(path)
  } catch (error) {
    throw pathError(path, `cannot resolve ${label}`, error)
  }
}

async function replaceChildSkillDirectory(root: string, name: string, label: string): Promise<void> {
  const existing = await existingChildDirectory(root, name, label)
  if (existing === undefined) return
  await removeDirectoryTree(root, existing)
}

async function removeDirectoryTree(root: string, directory: string): Promise<void> {
  const safeDirectory = await containedPath(root, directory, 'skill directory')
  let entries
  try {
    entries = await readdir(safeDirectory, { withFileTypes: true })
  } catch (error) {
    throw pathError(safeDirectory, 'cannot list skill directory for replacement', error)
  }
  for (const entry of entries) {
    const child = join(safeDirectory, entry.name)
    let metadata
    try {
      metadata = await lstat(child)
    } catch (error) {
      throw pathError(child, 'cannot inspect skill entry for replacement', error)
    }
    if (metadata.isSymbolicLink()) {
      throw new SkillGuardPathError(child, 'skill directory replacement refuses symbolic links')
    }
    if (metadata.isDirectory()) {
      await removeDirectoryTree(root, child)
      continue
    }
    if (!metadata.isFile()) {
      throw new SkillGuardPathError(child, 'skill directory replacement refuses non-file entries')
    }
    try {
      await unlink(child)
    } catch (error) {
      throw pathError(child, 'cannot remove skill file for replacement', error)
    }
  }
  try {
    await rmdir(safeDirectory)
  } catch (error) {
    throw pathError(safeDirectory, 'cannot remove skill directory for replacement', error)
  }
}

async function writeDirectFile(root: string, name: string, content: string, label: string): Promise<void> {
  const existing = await existingRegularFile(root, name, label)
  const target = existing ?? await containedChildPath(root, name, label)
  const temporary = join(dirname(target), `.${basename(target)}.${crypto.randomUUID()}.tmp`)
  try {
    await Bun.write(temporary, content)
    await rename(temporary, target)
  } catch (error) {
    throw pathError(target, `cannot write ${label}`, error)
  } finally {
    try {
      await rm(temporary, { force: true })
    } catch {
      // The primary write error is more useful, and a failed temporary-file cleanup cannot expose trusted content.
    }
  }
}

async function moveDirectory(source: string, destination: string, label: string): Promise<void> {
  try {
    await rename(source, destination)
    return
  } catch (error) {
    if (!isCrossDevice(error)) throw pathError(source, `cannot ${label}`, error)
  }
  // Rename cannot cross filesystem boundaries; fall back to a guarded copy, then remove the source.
  await copyDirectoryTree(source, destination, label)
  await removeDirectoryTree(dirname(source), source)
}

async function copyDirectoryTree(source: string, destination: string, label: string): Promise<void> {
  let entries
  try {
    entries = await readdir(source, { withFileTypes: true })
  } catch (error) {
    throw pathError(source, `cannot list ${label} source`, error)
  }
  try {
    await mkdir(destination, { recursive: true })
  } catch (error) {
    throw pathError(destination, `cannot create ${label} destination`, error)
  }
  for (const entry of entries) {
    const childSource = join(source, entry.name)
    const childDestination = join(destination, entry.name)
    if (entry.isSymbolicLink()) {
      throw new SkillGuardPathError(childSource, `${label} refuses symbolic links`)
    }
    if (entry.isDirectory()) {
      await copyDirectoryTree(childSource, childDestination, label)
      continue
    }
    if (!entry.isFile()) {
      throw new SkillGuardPathError(childSource, `${label} refuses non-file entries`)
    }
    try {
      await copyFile(childSource, childDestination)
    } catch (error) {
      throw pathError(childSource, `cannot copy ${label} file`, error)
    }
  }
}

async function containedChildPath(root: string, name: string, label: string): Promise<string> {
  directChildPath(root, name, label)
  try {
    return await resolveWithin(root, name)
  } catch (error) {
    if (error instanceof PathEscape) {
      throw new SkillGuardPathError(name, `${label} escapes root ${root}`)
    }
    throw pathError(name, `cannot resolve ${label}`, error)
  }
}

async function containedPath(root: string, candidate: string, label: string): Promise<string> {
  if (!isWithin(root, candidate)) {
    throw new SkillGuardPathError(candidate, `${label} escapes root ${root}`)
  }
  const name = relative(root, candidate)
  if (!name || name.includes(`..${sep}`) || name === '..') {
    throw new SkillGuardPathError(candidate, `${label} must be below root ${root}`)
  }
  try {
    return await resolveWithin(root, name)
  } catch (error) {
    if (error instanceof PathEscape) {
      throw new SkillGuardPathError(candidate, `${label} escapes root ${root}`)
    }
    throw pathError(candidate, `cannot resolve ${label}`, error)
  }
}

function directChildPath(root: string, name: string, label: string): string {
  if (!name || name.includes('\0') || basename(name) !== name || name === '.' || name === '..') {
    throw new SkillGuardPathError(name, `${label} must use one path segment`)
  }
  const candidate = resolve(root, name)
  if (!isWithin(root, candidate)) {
    throw new SkillGuardPathError(candidate, `${label} escapes root ${root}`)
  }
  return candidate
}

async function skillNameFromPath(skillsRoot: string, skillPath: string): Promise<string> {
  if (typeof skillPath !== 'string' || !skillPath.trim() || skillPath.includes('\0')) {
    throw new SkillGuardPathError(String(skillPath), 'skill path must be a non-empty string without null bytes')
  }
  const expanded = expandHome(skillPath.trim())
  const candidate = isAbsolute(expanded) ? resolve(expanded) : resolve(skillsRoot, expanded)
  try {
    const metadata = await lstat(candidate)
    if (metadata.isSymbolicLink()) {
      throw new SkillGuardPathError(candidate, 'skill path must not be a symbolic link')
    }
    if (metadata.isDirectory()) {
      let canonical: string
      try {
        canonical = await realpath(candidate)
      } catch (error) {
        throw pathError(candidate, 'cannot resolve skill path', error)
      }
      return validateContainedSkillName(skillsRoot, canonical, skillPath)
    }
  } catch (error) {
    if (error instanceof SkillGuardPathError) throw error
    if (!isMissing(error)) throw pathError(candidate, 'cannot inspect skill path', error)
  }
  return validateContainedSkillName(skillsRoot, candidate, skillPath)
}

function validateContainedSkillName(skillsRoot: string, candidate: string, original: string): string {
  if (!isWithin(skillsRoot, candidate)) {
    throw new SkillGuardPathError(original, `skill must be a direct child of ${skillsRoot}`)
  }
  return validateSkillName(relative(skillsRoot, candidate))
}

function validateSkillName(name: string): string {
  const isPlainName = name
    && name !== '.'
    && name !== '..'
    && name !== '.hub'
    && !name.includes('\0')
    && basename(name) === name
    && !name.includes('..')
  if (!isPlainName) {
    throw new SkillGuardPathError(name, 'skill name must be a plain active-skill directory name')
  }
  return name
}

function normalizePath(path: string): string {
  if (typeof path !== 'string') {
    throw new TypeError('skill path must be a string')
  }
  const trimmed = path.trim()
  if (!trimmed || trimmed.includes('\0')) {
    throw new SkillGuardPathError(path, 'path must be non-empty and contain no null bytes')
  }
  return resolve(expandHome(trimmed))
}

function expandHome(path: string): string {
  if (path === '~') return homedir()
  if (path.startsWith('~/') || path.startsWith('~\\')) return join(homedir(), path.slice(2))
  return path
}

function parseTrustedHashes(value: string): Record<string, string> {
  try {
    return normalizeTrustedHashes(JSON.parse(value))
  } catch {
    return {}
  }
}

function normalizeTrustedHashes(value: unknown): Record<string, string> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return {}
  const normalized: Record<string, string> = {}
  for (const [key, entry] of Object.entries(value)) {
    // Skip only the malformed entry; one bad record must not void the whole database.
    if (typeof entry !== 'string') continue
    normalized[key] = entry
  }
  return normalized
}

function unreadableSkillResult(error: unknown): ScanResult {
  const detail = error instanceof Error ? error.message : String(error)
  return new ScanResult({ reasons: [`Unreadable SKILL.md: ${detail}`] })
}

function digest(content: Uint8Array): string {
  return createHash('sha256').update(content).digest('hex')
}

function isWithin(root: string, candidate: string): boolean {
  const pathFromRoot = relative(root, candidate)
  return pathFromRoot === ''
    || (!pathFromRoot.startsWith(`..${sep}`) && pathFromRoot !== '..' && !isAbsolute(pathFromRoot))
}

function compareLexical(left: string, right: string): number {
  if (left < right) return -1
  if (left > right) return 1
  return 0
}

function pathError(path: string, action: string, error: unknown): SkillGuardPathError {
  const detail = error instanceof Error ? error.message : String(error)
  return new SkillGuardPathError(path, `${action}: ${detail}`)
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function isAlreadyExists(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'EEXIST'
}

function isCrossDevice(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'EXDEV'
}
