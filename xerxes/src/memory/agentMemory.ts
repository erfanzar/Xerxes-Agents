// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { appendFile, mkdir, open, readFile, readdir, realpath, rename, rm, stat } from 'node:fs/promises'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { xerxesHome } from '../daemon/paths.js'
import { scanContextContent } from '../security/promptScanner.js'
import { MemoryItem } from './base.js'
import { buildMemoryContextBlock } from './contextFencing.js'
import { HybridRetriever } from './retrieval.js'

export const AgentMemoryScope = {
  GLOBAL: 'global',
  PROJECT: 'project',
} as const

export type AgentMemoryScope = (typeof AgentMemoryScope)[keyof typeof AgentMemoryScope]

export const CANONICAL_AGENT_MEMORY_FILES = Object.freeze([
  'IDENTITY.md',
  'SOUL.md',
  'USER.md',
  'MEMORY.md',
  'KNOWLEDGE.md',
  'INSIGHTS.md',
  'EXPERIENCES.md',
] as const)

/** The one canonical file that indexes topic files instead of holding their bodies. */
export const MEMORY_INDEX_FILE = 'MEMORY.md'

/** Per-file ceiling for a canonical file injected in full. */
export const MAX_MEMORY_FILE_PROMPT_BYTES = 4_000
/** Topic manifest lines kept before the renderer names the remainder as omitted. */
export const MAX_MEMORY_INDEX_ENTRIES = 48
/** Byte ceiling for the topic manifest, independent of the entry count. */
export const MAX_MEMORY_INDEX_BYTES = 8 * 1024
/**
 * Ceiling for the whole memory section. The canonical files were previously
 * bounded only per file, so N canonical files plus every topic body could
 * still dwarf the turn's real context; this is the total the section may cost.
 */
export const MAX_MEMORY_SECTION_BYTES = 32 * 1024
/** Lines read from each file when collecting metadata: enough for frontmatter and a title. */
export const MEMORY_METADATA_HEAD_LINES = 30
const MEMORY_METADATA_HEAD_BYTES = 4 * 1024

const DEFAULT_CONTENT: Readonly<Record<(typeof CANONICAL_AGENT_MEMORY_FILES)[number], string>> = Object.freeze({
  'IDENTITY.md': '# Identity\n\nYou are Xerxes. Track durable notes about your working identity.\n',
  'SOUL.md': '# Soul\n\nDirect, pragmatic, technically careful, and evidence-led.\n',
  'USER.md': '# User profile\n\nTrack stable user preferences across sessions.\n',
  // Seeded as an index, not a container: this file is injected into every turn
  // in full, so a body written here is a permanent per-turn tax. Topic files
  // are the container and reach the prompt as one manifest line each.
  'MEMORY.md': [
    '# Memory index',
    '',
    '<!-- One list entry per topic file: `- topics/<name>.md - what it holds`.',
    'Bodies belong in the topic file, which carries name/description/type frontmatter. -->',
    '',
  ].join('\n'),
  'KNOWLEDGE.md': '# Knowledge\n\nCumulative mental models and explanations.\n',
  'INSIGHTS.md': '# Insights\n\nShort aha-moments and anti-patterns.\n',
  'EXPERIENCES.md': '# Experiences\n\nAppend successes, failures, and lessons before repeating risky work.\n',
})

export interface AgentMemoryFile {
  readonly bytes: number
  /** Frontmatter `description`, empty when the file declares none. */
  readonly description: string
  readonly modifiedAt: Date
  readonly path: string
  readonly scope: AgentMemoryScope
  /** Frontmatter `name`, falling back to the file's basename. */
  readonly title: string
  /** Frontmatter `type`, empty when the file declares none. */
  readonly type: string
}

export interface MemoryPromptOptions {
  /** Text this conversation has already shown; matching topics are dropped before ranking. */
  readonly alreadySurfaced?: string
  readonly maxBytesPerFile?: number
  readonly maxIndexBytes?: number
  readonly maxIndexEntries?: number
  readonly maxTotalBytes?: number
  /** Ranking query; without one the manifest keeps its deterministic scope/path order. */
  readonly query?: string
  /** Tools used successfully in recent turns; their reference topics rank down. */
  readonly recentSuccessfulTools?: readonly string[]
}

export interface MemoryRelevanceOptions {
  readonly alreadySurfaced?: string
  readonly limit?: number
  readonly now?: Date
  readonly query?: string
  readonly recentSuccessfulTools?: readonly string[]
}

export interface AgentMemoryOptions {
  readonly globalDirectory?: string
  readonly projectDirectory?: string
  readonly projectRoot?: string
  readonly projectSalt?: string
}

export interface AgentMemoryWriteResult {
  readonly bytes: number
  readonly path: string
  readonly scope: AgentMemoryScope
}

/**
 * Persistent global plus project-scoped agent memory with path containment.
 *
 * The storage boundary is asynchronous and Bun-native. All paths are
 * scope-relative; absolute paths, traversal, and existing symlink escapes are
 * rejected before reads or writes.
 */
export class AgentMemory {
  readonly globalDirectory: string
  readonly projectDirectory: string | undefined
  readonly projectRoot: string | undefined
  private readonly appendLocks = new Map<string, Promise<void>>()

  constructor(options: AgentMemoryOptions = {}) {
    this.projectRoot = options.projectRoot ? resolve(options.projectRoot) : undefined
    this.globalDirectory = resolve(options.globalDirectory ?? defaultGlobalMemoryDirectory())
    this.projectDirectory = options.projectDirectory
      ? resolve(options.projectDirectory)
      : this.projectRoot
        ? projectMemoryDirectoryFor(this.projectRoot, options.projectSalt)
        : undefined
  }

  hasProjectScope(): boolean {
    return this.projectDirectory !== undefined
  }

  scopeDirectory(scope: AgentMemoryScope | string): string {
    const normalized = normalizeScope(scope)
    if (normalized === AgentMemoryScope.GLOBAL) return this.globalDirectory
    if (!this.projectDirectory) {
      throw new ValidationError('scope', 'project memory scope is unavailable because no project root is configured', scope)
    }
    return this.projectDirectory
  }

  async ensure(): Promise<void> {
    await this.ensureScope(AgentMemoryScope.GLOBAL)
    if (this.projectDirectory) await this.ensureScope(AgentMemoryScope.PROJECT)
  }

  async read(scope: AgentMemoryScope | string, path: string): Promise<string> {
    const target = await this.resolveTarget(scope, path)
    try {
      const info = await stat(target)
      if (!info.isFile()) throw new ValidationError('path', 'must refer to a regular memory file', path)
      return await readFile(target, 'utf8')
    } catch (error) {
      if (isMissing(error)) {
        throw new ValidationError('path', 'does not exist in the selected memory scope', path)
      }
      throw error
    }
  }

  async write(scope: AgentMemoryScope | string, path: string, body: string): Promise<AgentMemoryWriteResult> {
    if (typeof body !== 'string') throw new ValidationError('body', 'must be a string', body)
    const target = await this.resolveTarget(scope, path)
    await mkdir(dirname(target), { recursive: true })
    await this.assertExistingPathInsideScope(scope, target)
    const temporary = join(dirname(target), '.' + basename(target) + '.' + crypto.randomUUID() + '.tmp')
    try {
      await Bun.write(temporary, body)
      await rename(temporary, target)
    } finally {
      await rm(temporary, { force: true })
    }
    return {
      scope: normalizeScope(scope),
      path: this.relativePath(scope, target),
      bytes: Buffer.byteLength(body),
    }
  }

  async append(
    scope: AgentMemoryScope | string,
    path: string,
    body: string,
    options: { readonly section?: string; readonly timestamp?: boolean } = {},
  ): Promise<{ readonly appendedBytes: number; readonly path: string; readonly scope: AgentMemoryScope }> {
    if (typeof body !== 'string' || !body.trim()) throw new ValidationError('body', 'must be a non-empty string', body)
    const target = await this.resolveTarget(scope, path)
    let addition = body.trim()
    if (options.section?.trim()) addition = '## ' + options.section.trim() + '\n\n' + addition
    if (options.timestamp ?? true) addition = '<!-- ' + new Date().toISOString() + ' -->\n' + addition

    return this.withAppendLock(target, async () => {
      // Append-only write through O_APPEND instead of read-modify-write: the
      // per-instance promise lock cannot serialize two AgentMemory instances
      // (daemon + CLI, parallel agents) sharing one directory, and their
      // interleaved read-modify-write cycles silently lost journal entries.
      // Kernel-level append positioning keeps every entry durable; at worst
      // concurrent writers interleave separator blank lines.
      await mkdir(dirname(target), { recursive: true })
      const tail = await readTail(target)
      const chunk = tail === undefined || tail.length === 0
        ? addition + '\n'
        : (tail.endsWith('\n') ? '\n' : '\n\n') + addition + '\n'
      await appendFile(target, chunk, 'utf8')
      return {
        scope: normalizeScope(scope),
        path: this.relativePath(scope, target),
        appendedBytes: Buffer.byteLength(addition),
      }
    })
  }

  async journal(
    scope: AgentMemoryScope | string,
    note: string,
    when = new Date(),
  ): Promise<{ readonly appendedBytes: number; readonly path: string; readonly scope: AgentMemoryScope }> {
    if (Number.isNaN(when.valueOf())) throw new ValidationError('when', 'must be a valid date')
    const day = when.toISOString().slice(0, 10)
    const time = when.toISOString().slice(11, 19)
    return this.append(scope, 'journal/' + day + '.md', '- ' + time + '  ' + note.trim(), { timestamp: false })
  }

  async search(
    query: string,
    options: { readonly limit?: number; readonly scope?: AgentMemoryScope | string } = {},
  ): Promise<Array<{ readonly path: string; readonly scope: AgentMemoryScope; readonly snippet: string }>> {
    const needle = query.trim().toLowerCase()
    if (!needle) return []
    const limit = validateLimit(options.limit ?? 20)
    const hits: Array<{ path: string; scope: AgentMemoryScope; snippet: string }> = []
    for (const file of await this.listFiles(options.scope)) {
      let content: string
      try {
        content = await this.read(file.scope, file.path)
      } catch {
        continue
      }
      const lowered = content.toLowerCase()
      let offset = 0
      for (let count = 0; count < 3; count += 1) {
        const index = lowered.indexOf(needle, offset)
        if (index < 0) break
        hits.push({
          scope: file.scope,
          path: file.path,
          snippet: content.slice(Math.max(0, index - 60), Math.min(content.length, index + needle.length + 60))
            .replaceAll('\n', ' / '),
        })
        if (hits.length >= limit) return hits
        offset = index + needle.length
      }
    }
    return hits
  }

  async listFiles(scope?: AgentMemoryScope | string): Promise<AgentMemoryFile[]> {
    const scopes =
      scope === undefined
        ? this.projectDirectory
          ? [AgentMemoryScope.GLOBAL, AgentMemoryScope.PROJECT]
          : [AgentMemoryScope.GLOBAL]
        : [normalizeScope(scope)]
    const files: AgentMemoryFile[] = []
    for (const selected of scopes) {
      try {
        files.push(...(await this.collectFiles(selected)))
      } catch (error) {
        if (!isMissing(error)) throw error
      }
    }
    return files
  }

  /**
   * Render the memory section injected into every turn.
   *
   * Two passes, because memory is a container and only its index belongs in
   * the prompt: the canonical files are injected in full but individually
   * clipped, and every other topic file contributes a single manifest line.
   * Concatenating every topic body — the previous behaviour — grew without
   * bound and pushed the actual task out of the useful context window.
   */
  async toPromptSection(options: MemoryPromptOptions = {}): Promise<string> {
    const maxBytesPerFile = validateLimit(options.maxBytesPerFile ?? MAX_MEMORY_FILE_PROMPT_BYTES)
    const maxIndexBytes = validateLimit(options.maxIndexBytes ?? MAX_MEMORY_INDEX_BYTES)
    const maxIndexEntries = validateLimit(options.maxIndexEntries ?? MAX_MEMORY_INDEX_ENTRIES)
    const maxTotalBytes = validateLimit(options.maxTotalBytes ?? MAX_MEMORY_SECTION_BYTES)
    await this.ensure()
    const order = new Map(
      ['SOUL.md', 'IDENTITY.md', 'USER.md', 'EXPERIENCES.md', 'MEMORY.md', 'KNOWLEDGE.md', 'INSIGHTS.md'].map(
        (name, index) => [name, index],
      ),
    )
    const entries = (await this.listFiles()).filter(
      entry => entry.path.endsWith('.md') && entry.bytes > 0 && this.shouldIncludeInPrompt(entry.path),
    )
    const canonical = entries.filter(entry => isCanonicalMemoryFile(entry.path))
    canonical.sort((left, right) => {
      const scopeOrder = left.scope === right.scope ? 0 : left.scope === AgentMemoryScope.PROJECT ? -1 : 1
      if (scopeOrder !== 0) return scopeOrder
      const fileOrder = (order.get(left.path) ?? 99) - (order.get(right.path) ?? 99)
      return fileOrder !== 0 ? fileOrder : left.path.localeCompare(right.path)
    })

    const sections = [
      '# Persistent agent memory',
      'Use global memory for cross-project facts and project memory for this codebase. Read before risky work and record durable decisions, stable user preferences, recurring failures, and reusable wins before ending a substantive turn.',
      MEMORY_CONTAINER_RULE,
      MEMORY_STALENESS_RULE,
      MEMORY_EXCLUSION_RULE,
      'Available tools: agent_memory_read, agent_memory_write, agent_memory_append, agent_memory_journal, agent_memory_search, agent_memory_list, and agent_memory_status.',
      '## Current memory contents',
    ]
    for (const entry of canonical) {
      let body: string
      try {
        body = (await this.read(entry.scope, entry.path)).trim()
      } catch {
        continue
      }
      if (!body) continue
      if (Buffer.byteLength(body) > maxBytesPerFile) {
        const tail = entry.path === 'EXPERIENCES.md'
        const shortened = tail ? body.slice(-maxBytesPerFile) : body.slice(0, maxBytesPerFile)
        body = shortened + '\n\n[Memory file truncated; use agent_memory_read for full text.]'
      }
      // Memory file bodies are agent-written data derived from untrusted
      // tool and web output. They flow straight into the system prompt
      // (cli.ts, daemon/turnRunner.ts), so neutralise embedded hostile
      // instructions and fence them as data, never as instructions.
      const fenced = buildMemoryContextBlock(scanContextContent(body, `agent memory: ${entry.path}`))
      sections.push('### [' + entry.scope + '] ' + entry.path + '\n\n' + fenced)
    }

    const topics = entries.filter(entry => !isCanonicalMemoryFile(entry.path))
    const selectionOptions: MemoryRelevanceOptions = {
      ...(options.alreadySurfaced === undefined ? {} : { alreadySurfaced: options.alreadySurfaced }),
      ...(options.query === undefined ? {} : { query: options.query }),
      ...(options.recentSuccessfulTools === undefined ? {} : { recentSuccessfulTools: options.recentSuccessfulTools }),
    }
    const manifest = renderMemoryManifest(selectRelevantMemoryFiles(topics, selectionOptions), {
      maxBytes: maxIndexBytes,
      maxEntries: maxIndexEntries,
    })
    if (manifest) sections.push(manifest)

    sections.push(
      '## Before ending the turn',
      'Only if this substantive turn produced durable new information, write it to the appropriate memory file or journal now. Otherwise do not call a memory-writing tool.',
    )
    return clipMemoryText(sections.join('\n\n').trimEnd(), maxTotalBytes, 'agent memory section') + '\n'
  }

  async status(): Promise<{
    readonly filesByScope: Readonly<Record<string, number>>
    readonly globalDirectory: string
    readonly projectDirectory: string | undefined
    readonly totalFiles: number
  }> {
    const filesByScope: Record<string, number> = {}
    const files = await this.listFiles()
    for (const file of files) filesByScope[file.scope] = (filesByScope[file.scope] ?? 0) + 1
    return {
      globalDirectory: this.globalDirectory,
      projectDirectory: this.projectDirectory,
      filesByScope,
      totalFiles: files.length,
    }
  }

  private async ensureScope(scope: AgentMemoryScope): Promise<void> {
    const directory = this.scopeDirectory(scope)
    await mkdir(directory, { recursive: true })
    await mkdir(join(directory, 'journal'), { recursive: true })
    for (const name of CANONICAL_AGENT_MEMORY_FILES) {
      const target = join(directory, name)
      try {
        await stat(target)
      } catch (error) {
        if (!isMissing(error)) throw error
        await Bun.write(target, DEFAULT_CONTENT[name])
      }
    }
  }

  private async resolveTarget(scope: AgentMemoryScope | string, path: string): Promise<string> {
    const normalized = normalizeScope(scope)
    if (typeof path !== 'string' || !path.trim()) {
      throw new ValidationError('path', 'must be a non-empty relative path', path)
    }
    if (path.includes('\0') || isAbsolute(path)) {
      throw new ValidationError('path', 'must be a safe relative path', path)
    }
    await this.ensureScope(normalized)
    const root = this.scopeDirectory(normalized)
    const target = resolve(root, path)
    if (!isWithin(root, target)) {
      throw new ValidationError('path', 'escapes the selected memory scope', path)
    }
    await this.assertExistingPathInsideScope(normalized, target)
    return target
  }

  private async assertExistingPathInsideScope(scope: AgentMemoryScope | string, target: string): Promise<void> {
    const root = await realpath(this.scopeDirectory(scope))
    let existing = target
    while (true) {
      try {
        const resolvedExisting = await realpath(existing)
        if (!isWithin(root, resolvedExisting)) {
          throw new ValidationError('path', 'resolves outside the selected memory scope', target)
        }
        return
      } catch (error) {
        if (!isMissing(error)) throw error
      }
      const parent = dirname(existing)
      if (parent === existing) {
        throw new ValidationError('path', 'cannot resolve an ancestor inside the selected memory scope', target)
      }
      existing = parent
    }
  }

  private relativePath(scope: AgentMemoryScope | string, absolutePath: string): string {
    const result = relative(this.scopeDirectory(scope), absolutePath)
    if (!result || result.startsWith('..' + sep) || result === '..' || isAbsolute(result)) {
      throw new ValidationError('path', 'is outside the selected memory scope', absolutePath)
    }
    return result.replaceAll('\\', '/')
  }

  private async collectFiles(scope: AgentMemoryScope): Promise<AgentMemoryFile[]> {
    const root = this.scopeDirectory(scope)
    const result: AgentMemoryFile[] = []
    const visit = async (directory: string): Promise<void> => {
      const entries = await readdir(directory, { withFileTypes: true })
      entries.sort((left, right) => left.name.localeCompare(right.name))
      for (const entry of entries) {
        const fullPath = join(directory, entry.name)
        if (entry.isSymbolicLink()) continue
        if (entry.isDirectory()) {
          await visit(fullPath)
          continue
        }
        if (!entry.isFile()) continue
        try {
          const info = await stat(fullPath)
          const path = this.relativePath(scope, fullPath)
          // Only the head is read: collection runs on every prompt build, and
          // frontmatter plus a title always live in the first few lines, so a
          // 2 MB journal must not be paged in to learn its description.
          const metadata = path.endsWith('.md')
            ? parseMemoryFrontmatter(await readHead(fullPath, MEMORY_METADATA_HEAD_LINES, MEMORY_METADATA_HEAD_BYTES))
            : { description: '', name: '', type: '' }
          result.push({
            scope,
            path,
            bytes: info.size,
            modifiedAt: info.mtime,
            title: metadata.name || basename(path),
            description: metadata.description,
            type: metadata.type,
          })
        } catch (error) {
          if (!isMissing(error)) throw error
        }
      }
    }
    await visit(root)
    return result
  }

  private shouldIncludeInPrompt(path: string): boolean {
    if (!path.startsWith('journal/')) return true
    const name = path.slice('journal/'.length).replace(/\.md$/, '')
    const timestamp = Date.parse(name + 'T00:00:00.000Z')
    return Number.isFinite(timestamp) && Date.now() - timestamp <= 7 * 24 * 60 * 60 * 1000
  }

  private async withAppendLock<T>(path: string, operation: () => Promise<T>): Promise<T> {
    const previous = this.appendLocks.get(path) ?? Promise.resolve()
    let release: (() => void) | undefined
    const current = new Promise<void>((resolveLock) => {
      release = resolveLock
    })
    this.appendLocks.set(path, current)
    await previous
    try {
      return await operation()
    } finally {
      release?.()
      if (this.appendLocks.get(path) === current) this.appendLocks.delete(path)
    }
  }
}

/** Instruction that keeps MEMORY.md an index and pushes bodies into topic files. */
export const MEMORY_CONTAINER_RULE = [
  'MEMORY.md is an index, not a container: it holds one list entry per topic file and nothing else.',
  'Write a body to topics/<name>.md with `name`, `description`, and `type` frontmatter, then link it from MEMORY.md.',
  'Only the canonical files above are injected in full; topic files reach you as one manifest line each, so a useful description is what makes one findable.',
].join(' ')

/**
 * Reminder that memory records the past, not the present.
 *
 * Memory is written once and read for months, so its file paths, line numbers
 * and symbol names describe a repository that has since moved on. Acting on one
 * directly produces a confident recommendation about code that no longer exists.
 */
export const MEMORY_STALENESS_RULE =
  'Every memory is a point-in-time observation, not current state: any file, line, or symbol it names may have '
  + 'moved, been renamed, or been deleted since it was written, so confirm it with a read or a search before '
  + 'acting on it or recommending it.'

/**
 * What never belongs in memory, scoped to what Xerxes can re-derive on demand.
 *
 * The previous single sentence excluded only "facts already present", which
 * left the largest category — everything the repo itself already answers —
 * looking like fair game, and memory filled up with restated code.
 */
export const MEMORY_EXCLUSION_RULE = [
  'Do not write memory for anything this workspace can re-derive:',
  'code patterns or structure (the repo map already covers them),',
  'file paths and symbol locations (grep and read find them on demand),',
  'git history (git log has it),',
  'or content already in the project\'s own agent markdown files (AGENTS.md, XERXES.md, CLAUDE.md).',
  'Do not write memory for routine questions, arithmetic, transient test prompts, raw tool output, or facts already recorded.',
  'A turn with no durable new information should perform no memory write.',
].join(' ')

/** Topics whose value is knowing a tool fails, which recent success does not retire. */
export const MEMORY_GOTCHA_PATTERN = /gotcha|warning|caveat|pitfall|footgun|failure|regression|breaks/i

/** Multiplier applied to a topic that documents a tool the agent just used successfully. */
const RECENT_TOOL_PENALTY = 0.35

/** Whether a memory path is one of the canonical files injected in full. */
export function isCanonicalMemoryFile(path: string): boolean {
  return (CANONICAL_AGENT_MEMORY_FILES as readonly string[]).includes(path)
}

/**
 * Tolerant `name`/`description`/`type` frontmatter parse for a topic file.
 *
 * Deliberately the same shape as the SKILL.md parser in extensions/skills.ts —
 * quote stripping, null-prototype record, prototype keys refused — so a topic
 * file and a skill declare themselves the same way. It is duplicated rather
 * than imported only because that parser is module-private today.
 */
export function parseMemoryFrontmatter(content: string): {
  readonly description: string
  readonly name: string
  readonly type: string
} {
  const match = content.match(/^---\s*\r?\n([\s\S]*?)(?:\r?\n---\s*(?:\r?\n|$)|$)/)
  if (!match) return { description: '', name: '', type: '' }
  const fields: Record<string, string> = Object.create(null)
  for (const rawLine of (match[1] ?? '').split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line.startsWith('#')) continue
    const separator = line.indexOf(':')
    if (separator < 0) continue
    const key = line.slice(0, separator).trim().toLowerCase()
    if (key === '__proto__' || key === 'constructor' || key === 'prototype') continue
    fields[key] = stripFrontmatterQuotes(line.slice(separator + 1).trim())
  }
  return { name: fields.name ?? '', description: fields.description ?? '', type: fields.type ?? '' }
}

/**
 * Relative-day label for a dated journal entry, or undefined for anything else.
 *
 * The day comes from the filename, and only journal files get a label at all.
 * An mtime-derived "3 days ago" would be worse than no label: MEMORY.md and a
 * topic file are edited long after the claims inside them were true, so it
 * would advertise a document full of year-old assertions as fresh. A
 * `journal/YYYY-MM-DD.md` name is the one case where the file genuinely is
 * about the day it names.
 */
export function journalDayLabel(path: string, now = new Date()): string | undefined {
  const match = /^journal\/(\d{4}-\d{2}-\d{2})\.md$/.exec(path.replaceAll('\\', '/'))
  if (!match?.[1]) return undefined
  const entryDay = Date.parse(match[1] + 'T00:00:00.000Z')
  if (!Number.isFinite(entryDay)) return undefined
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())
  const days = Math.round((today - entryDay) / (24 * 60 * 60 * 1000))
  // A future-dated entry means a clock disagreement, not freshness; say nothing
  // rather than render "-1 days ago".
  if (days < 0) return undefined
  if (days === 0) return 'today'
  if (days === 1) return 'yesterday'
  return `${days} days ago`
}

/** Render one bounded, injection-scanned manifest line for a memory topic file. */
export function memoryManifestLine(entry: AgentMemoryFile, now = new Date()): string {
  const path = inertMemoryField(entry.path, `path in ${entry.scope} memory`, 120) || 'unnamed.md'
  const description = inertMemoryField(
    entry.description || entry.title || 'No description',
    `description for ${path}`,
    200,
  )
  const type = inertMemoryField(entry.type, `type for ${path}`, 40)
  const day = journalDayLabel(entry.path, now)
  return `  - [${entry.scope}] ${path}: ${description}${type ? ` [${type}]` : ''}${day ? ` (${day})` : ''}`
}

/**
 * Render the topic manifest under both an entry and a byte ceiling, naming how
 * many topics were dropped so a missing file reads as budgeted-out rather than
 * as nonexistent.
 */
export function renderMemoryManifest(
  entries: readonly AgentMemoryFile[],
  options: { readonly maxBytes?: number; readonly maxEntries?: number; readonly now?: Date } = {},
): string {
  if (entries.length === 0) return ''
  const maxBytes = options.maxBytes ?? MAX_MEMORY_INDEX_BYTES
  const maxEntries = options.maxEntries ?? MAX_MEMORY_INDEX_ENTRIES
  const now = options.now ?? new Date()
  const shown = entries.slice(0, maxEntries)
  const header = '## Memory topic files (metadata only; read one with agent_memory_read before relying on it)'
  // Not `shown.map(memoryManifestLine)`: Array#map passes the index as the
  // second argument, which would arrive as the renderer's `now`.
  const lines = [header, ...shown.map(entry => memoryManifestLine(entry, now))]
  let omitted = entries.length - shown.length
  if (omitted > 0) lines.push(memoryOmissionMarker(omitted))

  const complete = lines.join('\n')
  if (Buffer.byteLength(complete, 'utf8') <= maxBytes) return complete

  const candidates = lines.slice(1, omitted > 0 ? -1 : undefined)
  const kept = [header]
  for (let index = 0; index < candidates.length; index += 1) {
    const line = candidates[index]
    if (line === undefined) continue
    const remaining = candidates.length - index + omitted
    const attempt = [...kept, line, memoryOmissionMarker(remaining)].join('\n')
    if (Buffer.byteLength(attempt, 'utf8') > maxBytes) break
    kept.push(line)
  }
  omitted = candidates.length - (kept.length - 1) + omitted
  if (omitted > 0) kept.push(memoryOmissionMarker(omitted))
  return kept.join('\n')
}

/**
 * Rank topic files locally with the hybrid weights the memory subsystem
 * already uses. No model call: a per-turn cheap-model ranking would add a
 * network round trip to every prompt build for a decision BM25 plus recency
 * already makes well.
 *
 * Two signals no similarity function can express are applied around the
 * ranker: topics the conversation has already shown are removed before the
 * top-N cut, and a topic documenting a tool the agent just used successfully
 * is demoted — unless it is the file recording how that tool goes wrong,
 * which is exactly what success does not retire.
 */
export function selectRelevantMemoryFiles(
  entries: readonly AgentMemoryFile[],
  options: MemoryRelevanceOptions = {},
): AgentMemoryFile[] {
  const limit = options.limit ?? entries.length
  const surfaced = (options.alreadySurfaced ?? '').toLowerCase()
  const tools = (options.recentSuccessfulTools ?? []).map(tool => tool.trim().toLowerCase()).filter(Boolean)
  const candidates = entries.filter(entry => !alreadySurfaced(entry, surfaced))
  const query = options.query?.trim() ?? ''
  if (!query) return candidates.slice(0, limit)

  const items = new Map<MemoryItem, AgentMemoryFile>()
  for (const entry of candidates) {
    items.set(new MemoryItem({ content: memorySearchText(entry), timestamp: entry.modifiedAt }), entry)
  }
  const ranked = new HybridRetriever().rank(query, [...items.keys()], items.size, options.now ?? new Date())
  return ranked
    .map(result => {
      const entry = items.get(result.item)
      return entry === undefined ? undefined : { entry, score: result.score * recentToolPenalty(entry, tools) }
    })
    .filter((scored): scored is { entry: AgentMemoryFile; score: number } => scored !== undefined)
    .sort((left, right) => right.score - left.score)
    .slice(0, limit)
    .map(scored => scored.entry)
}

/**
 * Explain why a body may not be written to MEMORY.md, or undefined when it may.
 *
 * MEMORY.md is injected in full on every turn, so a prose body there is a tax
 * paid forever; index entries are cheap and point at the topic file that holds
 * the prose.
 */
export function memoryIndexBodyIssue(path: string, body: string): string | undefined {
  if (path.replaceAll('\\', '/').replace(/^\.\//, '') !== MEMORY_INDEX_FILE) return undefined
  const withoutComments = body.replace(/<!--[\s\S]*?-->/g, '')
  const prose = withoutComments.split(/\r?\n/).find(line => isProseLine(line))
  if (prose === undefined) return undefined
  return `${MEMORY_INDEX_FILE} is the memory index and accepts only headings and one-line entries; `
    + 'write the body to topics/<name>.md with name/description/type frontmatter and link it here '
    + `(offending line: ${JSON.stringify(prose.trim().slice(0, 80))})`
}

/**
 * Clip prompt-bound memory text by UTF-8 bytes without splitting a code point.
 *
 * Same contract as clipBootstrapContext in runtime/bootstrap.ts, reimplemented
 * rather than imported so the memory subsystem does not depend on the runtime
 * bootstrap module graph.
 */
export function clipMemoryText(content: string, maxBytes: number, label: string): string {
  if (Buffer.byteLength(content, 'utf8') <= maxBytes) return content
  const marker = `\n\n[truncated: ${label} exceeded ${maxBytes} UTF-8 bytes; use agent_memory_read or agent_memory_search for the rest]`
  const budget = Math.max(0, maxBytes - Buffer.byteLength(marker, 'utf8'))
  let clipped = ''
  let usedBytes = 0
  for (const character of content) {
    const size = Buffer.byteLength(character, 'utf8')
    if (usedBytes + size > budget) break
    clipped += character
    usedBytes += size
  }
  return clipped.trimEnd() + marker
}

function memoryOmissionMarker(count: number): string {
  return `  ... ${count} more memory topic file${count === 1 ? '' : 's'} omitted; use agent_memory_list or agent_memory_search to reach them`
}

function memorySearchText(entry: AgentMemoryFile): string {
  return [entry.path, entry.title, entry.description, entry.type].filter(Boolean).join(' ')
}

function alreadySurfaced(entry: AgentMemoryFile, surfaced: string): boolean {
  if (!surfaced) return false
  if (surfaced.includes(entry.path.toLowerCase())) return true
  // Short titles ("notes", "api") collide with ordinary prose, so only a title
  // specific enough to be a real reference counts as already shown.
  return entry.title.length >= 8 && surfaced.includes(entry.title.toLowerCase())
}

function recentToolPenalty(entry: AgentMemoryFile, tools: readonly string[]): number {
  if (tools.length === 0) return 1
  const text = memorySearchText(entry).toLowerCase()
  if (!tools.some(tool => text.includes(tool))) return 1
  return MEMORY_GOTCHA_PATTERN.test(text) ? 1 : RECENT_TOOL_PENALTY
}

function isProseLine(line: string): boolean {
  const trimmed = line.trim()
  if (!trimmed) return false
  return !/^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||---|```|<)/.test(trimmed)
}

function inertMemoryField(value: string, label: string, maximumCharacters: number): string {
  const singleLine = value.replace(/[\r\n\t]+/g, ' ').replace(/\s+/g, ' ').trim()
  const scanned = scanContextContent(singleLine, `agent memory ${label}`)
  if (scanned.length <= maximumCharacters) return scanned
  return scanned.slice(0, Math.max(0, maximumCharacters - 3)).trimEnd() + '...'
}

function stripFrontmatterQuotes(value: string): string {
  return value.replace(/^(?:"([\s\S]*)"|'([\s\S]*)')$/, '$1$2').trim()
}

/** Return the cross-project memory directory under the configured Xerxes home. */
export function defaultGlobalMemoryDirectory(): string {
  return join(xerxesHome(), 'memory')
}

/** Return the deterministic project-specific memory directory for a workspace. */
export function projectMemoryDirectoryFor(projectRoot: string, salt = process.env.XERXES_PROJECT_SALT): string {
  const effectiveSalt = salt?.trim() || 'xerxes-project-salt'
  const canonicalRoot = resolve(projectRoot)
  const digest = createHash('sha256').update(effectiveSalt + '|' + canonicalRoot, 'utf8').digest('hex').slice(0, 12)
  return join(xerxesHome(), 'projects', digest, 'memory')
}

export function normalizeScope(scope: AgentMemoryScope | string): AgentMemoryScope {
  if (scope === AgentMemoryScope.GLOBAL || scope === AgentMemoryScope.PROJECT) return scope
  throw new ValidationError('scope', 'must be global or project', scope)
}

/** Return whether a user-provided relative path has no absolute or traversal escape. */
export function isSafeMemoryRelativePath(path: string): boolean {
  return typeof path === 'string'
    && Boolean(path.trim())
    && !path.includes('\0')
    && !isAbsolute(path)
    && !path.split(/[\\/]+/).includes('..')
}

function isWithin(root: string, target: string): boolean {
  const difference = relative(resolve(root), resolve(target))
  return difference === '' || (!difference.startsWith('..' + sep) && difference !== '..' && !isAbsolute(difference))
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

/**
 * Read the trailing bytes of a file to pick an append separator without
 * loading the whole file. Returns undefined when the file does not exist.
 * A single byte suffices to detect a trailing newline: 0x0A never appears
 * as a UTF-8 continuation byte.
 */
async function readTail(path: string, bytes = 1): Promise<string | undefined> {
  let handle: Awaited<ReturnType<typeof open>> | undefined
  try {
    handle = await open(path, 'r')
    const info = await handle.stat()
    if (!info.isFile() || info.size === 0) return ''
    const length = Math.min(bytes, info.size)
    const buffer = Buffer.alloc(length)
    await handle.read(buffer, 0, length, info.size - length)
    return buffer.toString('utf8')
  } catch (error) {
    if (isMissing(error)) return undefined
    throw error
  } finally {
    await handle?.close()
  }
}

/**
 * Read at most `maxLines` lines, and never more than `maxBytes`, from the head
 * of a file. Metadata collection touches every memory file on every prompt
 * build, so an oversized journal must cost a bounded read, not its full size.
 */
async function readHead(path: string, maxLines: number, maxBytes: number): Promise<string> {
  let handle: Awaited<ReturnType<typeof open>> | undefined
  try {
    handle = await open(path, 'r')
    const info = await handle.stat()
    if (!info.isFile() || info.size === 0) return ''
    const length = Math.min(maxBytes, info.size)
    const buffer = Buffer.alloc(length)
    await handle.read(buffer, 0, length, 0)
    return buffer.toString('utf8').split(/\r?\n/).slice(0, maxLines).join('\n')
  } catch (error) {
    if (isMissing(error)) return ''
    throw error
  } finally {
    await handle?.close()
  }
}

function validateLimit(limit: number): number {
  if (!Number.isInteger(limit) || limit < 1 || limit > 1_000_000) {
    throw new ValidationError('limit', 'must be an integer between 1 and 1000000', limit)
  }
  return limit
}
