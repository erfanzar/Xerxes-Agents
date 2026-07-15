// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'
import { lstat, mkdir, readdir, realpath, rename, rm, writeFile } from 'node:fs/promises'
import { isAbsolute, join, relative, resolve, sep } from 'node:path'

const SKILL_MARKDOWN_FILE = 'SKILL.md'
// The adjacent skill guard owns its metadata and quarantine beneath this directory.
const RESERVED_SKILL_DIRECTORY_NAMES = new Set(['.hub'])

/** A locally installable SKILL.md payload returned by a configured source. */
export interface SkillSyncBundle {
  readonly bodyMarkdown: string
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly name: string
  readonly sourceName?: string
  readonly version: string
}

/**
 * Host-provided source boundary for manifest reconciliation.
 *
 * This module intentionally does not implement network fetching. A host can use
 * a local catalog, a cache, or a separately authorized remote client here.
 */
export interface SkillSyncSource {
  fetch(identifier: string): SkillSyncBundle | Promise<SkillSyncBundle>
}

/** One desired `(source, identifier)` row in a skills manifest. */
export interface SkillSyncManifestEntry {
  readonly identifier: string
  readonly source: string
}

/** Source lookup table accepted by {@link syncSkillManifest}. */
export type SkillSyncSources = Readonly<Record<string, SkillSyncSource>> | ReadonlyMap<string, SkillSyncSource>

/** Configuration for a manifest reconciliation pass. */
export interface SkillSyncOptions {
  /** Directory containing one direct child directory per installed skill. */
  readonly targetDirectory: string
  /** Remove direct child skill directories that are not named in the manifest. */
  readonly prune?: boolean
}

/** One non-fatal entry-level reconciliation error. */
export interface SkillSyncFailure {
  readonly identifier: string
  readonly reason: string
}

/** Stable outcome of one manifest reconciliation pass. */
export interface SkillSyncResult {
  readonly failed: readonly SkillSyncFailure[]
  readonly installed: readonly string[]
  readonly removed: readonly string[]
  readonly skipped: readonly string[]
}

/** Raised when a bundle or filesystem target cannot be handled safely. */
export class SkillSyncError extends Error {
  constructor(message: string, options: { readonly cause?: unknown } = {}) {
    super(message, options)
    this.name = 'SkillSyncError'
  }
}

/** Install a bundle below `targetDirectory/<bundle.name>/SKILL.md` atomically. */
export async function installSkillBundle(bundle: SkillSyncBundle, targetDirectory: string): Promise<string> {
  const root = await ensureSkillRoot(targetDirectory)
  return installBundleInRoot(normalizeBundle(bundle), root)
}

/**
 * Reconcile a local skill directory with a manifest in stable identifier order.
 *
 * Duplicate rows for the same source and identifier are collapsed. Declaring the
 * same identifier from different sources is reported as a failure rather than
 * selecting a source based on manifest iteration order.
 */
export async function syncSkillManifest(
  manifest: Iterable<SkillSyncManifestEntry>,
  sources: SkillSyncSources,
  options: SkillSyncOptions,
): Promise<SkillSyncResult> {
  const root = await ensureSkillRoot(options.targetDirectory)
  const result = newMutableResult()
  const normalized = normalizeManifest(manifest, result)

  for (const entry of normalized.entries) {
    if (entry.conflictingSources) {
      result.failed.push({
        identifier: entry.identifier,
        reason: `manifest identifier is declared by multiple sources: ${entry.conflictingSources.join(', ')}`,
      })
      continue
    }

    try {
      if (await isInstalledSkill(root, entry.identifier)) {
        result.skipped.push(entry.identifier)
        continue
      }
    } catch (error) {
      result.failed.push({ identifier: entry.identifier, reason: errorMessage(error) })
      continue
    }

    const source = sourceFor(sources, entry.source)
    if (!source) {
      result.failed.push({ identifier: entry.identifier, reason: `unknown source: ${entry.source}` })
      continue
    }

    try {
      const bundle = normalizeBundle(await source.fetch(entry.identifier))
      if (bundle.name !== entry.identifier) {
        const message = [
          'source returned bundle name',
          JSON.stringify(bundle.name),
          'for identifier',
          JSON.stringify(entry.identifier),
        ].join(' ')
        throw new SkillSyncError(
          message,
        )
      }
      await installBundleInRoot(bundle, root)
      result.installed.push(entry.identifier)
    } catch (error) {
      result.failed.push({ identifier: entry.identifier, reason: errorMessage(error) })
    }
  }

  if (options.prune) {
    await pruneUnrequestedSkills(root, normalized.requested, result)
  }
  return finalizeResult(result)
}

interface NormalizedBundle {
  readonly bodyMarkdown: string
  readonly name: string
}

interface NormalizedManifest {
  readonly entries: readonly NormalizedManifestEntry[]
  readonly requested: Set<string>
}

interface NormalizedManifestEntry {
  readonly conflictingSources?: readonly string[]
  readonly identifier: string
  readonly source: string
}

interface MutableSkillSyncResult {
  readonly failed: SkillSyncFailure[]
  readonly installed: string[]
  readonly removed: string[]
  readonly skipped: string[]
}

async function installBundleInRoot(bundle: NormalizedBundle, root: string): Promise<string> {
  const skillDirectory = await ensureSkillDirectory(root, bundle.name)
  const output = containedChildPath(skillDirectory, SKILL_MARKDOWN_FILE, 'skill file')
  await assertRegularFileOrMissing(skillDirectory, SKILL_MARKDOWN_FILE, 'skill file')
  const temporaryName = `.${SKILL_MARKDOWN_FILE}.${randomUUID()}.tmp`
  const temporary = containedChildPath(skillDirectory, temporaryName, 'temporary skill file')
  try {
    await writeFile(temporary, bundle.bodyMarkdown, { encoding: 'utf8', flag: 'wx' })
    await rename(temporary, output)
  } catch (error) {
    const message = `cannot write skill ${JSON.stringify(bundle.name)}: ${errorMessage(error)}`
    throw new SkillSyncError(message, { cause: error })
  } finally {
    try {
      await rm(temporary, { force: true })
    } catch {
      // A cleanup failure must not obscure an installation failure or written skill path.
    }
  }
  return output
}

async function ensureSkillRoot(targetDirectory: string): Promise<string> {
  if (typeof targetDirectory !== 'string' || !targetDirectory.trim()) {
    throw new SkillSyncError('targetDirectory must be a non-empty path')
  }
  if (targetDirectory.includes('\0')) {
    throw new SkillSyncError('targetDirectory must not contain a null byte')
  }
  const requested = resolve(targetDirectory)
  try {
    await mkdir(requested, { recursive: true })
    const canonical = await realpath(requested)
    const metadata = await lstat(canonical)
    if (!metadata.isDirectory()) {
      throw new SkillSyncError(`targetDirectory must be a directory: ${JSON.stringify(targetDirectory)}`)
    }
    return canonical
  } catch (error) {
    if (error instanceof SkillSyncError) {
      throw error
    }
    const message = `cannot prepare targetDirectory ${JSON.stringify(targetDirectory)}: ${errorMessage(error)}`
    throw new SkillSyncError(message, {
      cause: error,
    })
  }
}

async function ensureSkillDirectory(root: string, name: string): Promise<string> {
  const existing = await existingSkillDirectory(root, name)
  if (existing) {
    return existing
  }
  const path = containedChildPath(root, name, 'skill directory')
  try {
    await mkdir(path)
  } catch (error) {
    if (!isAlreadyExists(error)) {
      const message = `cannot create skill directory ${JSON.stringify(name)}: ${errorMessage(error)}`
      throw new SkillSyncError(message, { cause: error })
    }
  }
  const created = await existingSkillDirectory(root, name)
  if (!created) {
    throw new SkillSyncError(`skill directory was not created: ${JSON.stringify(name)}`)
  }
  return created
}

async function existingSkillDirectory(root: string, name: string): Promise<string | undefined> {
  const path = containedChildPath(root, name, 'skill directory')
  const metadata = await optionalLstat(path)
  if (!metadata) {
    return undefined
  }
  if (metadata.isSymbolicLink()) {
    throw new SkillSyncError(`skill directory must not be a symbolic link: ${JSON.stringify(name)}`)
  }
  if (!metadata.isDirectory()) {
    throw new SkillSyncError(`skill directory must be a directory: ${JSON.stringify(name)}`)
  }
  const canonical = await canonicalContainedPath(root, path, 'skill directory')
  return canonical
}

async function isInstalledSkill(root: string, name: string): Promise<boolean> {
  const directory = await existingSkillDirectory(root, name)
  if (!directory) {
    return false
  }
  return (await assertRegularFileOrMissing(directory, SKILL_MARKDOWN_FILE, 'skill file')) !== undefined
}

async function assertRegularFileOrMissing(root: string, name: string, label: string): Promise<string | undefined> {
  const path = containedChildPath(root, name, label)
  const metadata = await optionalLstat(path)
  if (!metadata) {
    return undefined
  }
  if (metadata.isSymbolicLink()) {
    throw new SkillSyncError(`${label} must not be a symbolic link: ${JSON.stringify(name)}`)
  }
  if (!metadata.isFile()) {
    throw new SkillSyncError(`${label} must be a regular file: ${JSON.stringify(name)}`)
  }
  return canonicalContainedPath(root, path, label)
}

async function pruneUnrequestedSkills(
  root: string,
  requested: ReadonlySet<string>,
  result: MutableSkillSyncResult,
): Promise<void> {
  let entries
  try {
    entries = await readdir(root, { encoding: 'utf8', withFileTypes: true })
  } catch (error) {
    result.failed.push({ identifier: '<target>', reason: `cannot list target directory: ${errorMessage(error)}` })
    return
  }

  const directories = entries
    .filter(entry => entry.isDirectory() && !RESERVED_SKILL_DIRECTORY_NAMES.has(entry.name))
    .sort((left, right) => compareStrings(left.name, right.name))
  for (const entry of directories) {
    if (requested.has(entry.name)) {
      continue
    }
    try {
      const path = await existingSkillDirectory(root, entry.name)
      if (!path) {
        continue
      }
      await rm(path, { force: false, recursive: true })
      result.removed.push(entry.name)
    } catch (error) {
      result.failed.push({ identifier: entry.name, reason: errorMessage(error) })
    }
  }
}

function normalizeManifest(
  manifest: Iterable<SkillSyncManifestEntry>,
  result: MutableSkillSyncResult,
): NormalizedManifest {
  const groups = new Map<string, Set<string>>()
  for (const entry of manifest) {
    const label = typeof entry?.identifier === 'string' ? entry.identifier : '<invalid>'
    try {
      const identifier = requiredSkillName(entry.identifier, 'manifest identifier')
      const source = requiredText(entry.source, 'manifest source')
      const sources = groups.get(identifier) ?? new Set<string>()
      sources.add(source)
      groups.set(identifier, sources)
    } catch (error) {
      result.failed.push({ identifier: label, reason: errorMessage(error) })
    }
  }

  const requested = new Set<string>(groups.keys())
  const entries = [...groups].map(([identifier, sources]) => {
    const names = [...sources].sort(compareStrings)
    return {
      identifier,
      source: names[0] ?? '',
      ...(names.length > 1 ? { conflictingSources: names } : {}),
    }
  }).sort((left, right) => compareStrings(left.identifier, right.identifier))
  return { entries, requested }
}

function normalizeBundle(bundle: SkillSyncBundle): NormalizedBundle {
  if (typeof bundle !== 'object' || bundle === null) {
    throw new SkillSyncError('source returned an invalid skill bundle')
  }
  const name = requiredSkillName(bundle.name, 'bundle name')
  if (typeof bundle.version !== 'string' || !bundle.version.trim()) {
    throw new SkillSyncError('bundle version must be a non-empty string')
  }
  if (typeof bundle.bodyMarkdown !== 'string') {
    throw new SkillSyncError('bundle bodyMarkdown must be a string')
  }
  return { bodyMarkdown: bundle.bodyMarkdown, name }
}

function sourceFor(sources: SkillSyncSources, name: string): SkillSyncSource | undefined {
  if (sources instanceof Map) {
    return sources.get(name)
  }
  const table = sources as Readonly<Record<string, SkillSyncSource>>
  if (!Object.prototype.hasOwnProperty.call(table, name)) {
    return undefined
  }
  const source = table[name]
  return source && typeof source.fetch === 'function' ? source : undefined
}

function containedChildPath(root: string, name: string, label: string): string {
  const child = requiredChildName(name, label)
  const path = resolve(root, child)
  assertContained(root, path, label)
  return path
}

function requiredChildName(value: string, label: string): string {
  const name = requiredText(value, label)
  if (name === '.' || name === '..' || name.includes('/') || name.includes('\\') || name.includes('\0')) {
    throw new SkillSyncError(`${label} must be a single contained directory name`)
  }
  return name
}

function requiredSkillName(value: string, label: string): string {
  const name = requiredChildName(value, label)
  if (RESERVED_SKILL_DIRECTORY_NAMES.has(name)) {
    throw new SkillSyncError(`${label} is reserved for skill-system state`)
  }
  return name
}

function requiredText(value: string, label: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new SkillSyncError(`${label} must be a non-empty string`)
  }
  if (value.includes('\0')) {
    throw new SkillSyncError(`${label} must not contain a null byte`)
  }
  return value.trim()
}

async function canonicalContainedPath(root: string, path: string, label: string): Promise<string> {
  try {
    const canonical = await realpath(path)
    assertContained(root, canonical, label)
    return canonical
  } catch (error) {
    if (error instanceof SkillSyncError) {
      throw error
    }
    throw new SkillSyncError(`cannot resolve ${label}: ${errorMessage(error)}`, { cause: error })
  }
}

function assertContained(root: string, candidate: string, label: string): void {
  const pathFromRoot = relative(root, candidate)
  const isContained = pathFromRoot === ''
    || (!pathFromRoot.startsWith(`..${sep}`) && pathFromRoot !== '..' && !isAbsolute(pathFromRoot))
  if (isContained) {
    return
  }
  throw new SkillSyncError(`${label} resolves outside targetDirectory`)
}

async function optionalLstat(path: string) {
  try {
    return await lstat(path)
  } catch (error) {
    if (isMissing(error)) {
      return undefined
    }
    throw new SkillSyncError(`cannot inspect ${JSON.stringify(path)}: ${errorMessage(error)}`, { cause: error })
  }
}

function newMutableResult(): MutableSkillSyncResult {
  return { failed: [], installed: [], removed: [], skipped: [] }
}

function finalizeResult(result: MutableSkillSyncResult): SkillSyncResult {
  return Object.freeze({
    failed: Object.freeze([...result.failed]
      .sort((left, right) => {
        return compareStrings(left.identifier, right.identifier) || compareStrings(left.reason, right.reason)
      })
      .map(failure => Object.freeze({ ...failure }))),
    installed: Object.freeze([...result.installed].sort(compareStrings)),
    removed: Object.freeze([...result.removed].sort(compareStrings)),
    skipped: Object.freeze([...result.skipped].sort(compareStrings)),
  })
}

function compareStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}

function isAlreadyExists(error: unknown): boolean {
  return hasCode(error, 'EEXIST')
}

function isMissing(error: unknown): boolean {
  return hasCode(error, 'ENOENT')
}

function hasCode(error: unknown, code: string): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === code
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
