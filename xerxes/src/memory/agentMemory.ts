// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { mkdir, readFile, readdir, realpath, rename, rm, stat } from 'node:fs/promises'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { xerxesHome } from '../daemon/paths.js'

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

const DEFAULT_CONTENT: Readonly<Record<(typeof CANONICAL_AGENT_MEMORY_FILES)[number], string>> = Object.freeze({
  'IDENTITY.md': '# Identity\n\nYou are Xerxes. Track durable notes about your working identity.\n',
  'SOUL.md': '# Soul\n\nDirect, pragmatic, technically careful, and evidence-led.\n',
  'USER.md': '# User profile\n\nTrack stable user preferences across sessions.\n',
  'MEMORY.md': '# Memory\n\nDurable facts, decisions, and project context.\n',
  'KNOWLEDGE.md': '# Knowledge\n\nCumulative mental models and explanations.\n',
  'INSIGHTS.md': '# Insights\n\nShort aha-moments and anti-patterns.\n',
  'EXPERIENCES.md': '# Experiences\n\nAppend successes, failures, and lessons before repeating risky work.\n',
})

export interface AgentMemoryFile {
  readonly bytes: number
  readonly path: string
  readonly scope: AgentMemoryScope
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
      let existing = ''
      try {
        existing = await this.read(scope, path)
      } catch (error) {
        if (!(error instanceof ValidationError)) throw error
      }
      const next = existing ? existing.replace(/\n?$/, '\n') + '\n' + addition + '\n' : addition + '\n'
      await this.write(scope, path, next)
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

  async toPromptSection(options: { readonly maxBytesPerFile?: number } = {}): Promise<string> {
    const maxBytesPerFile = validateLimit(options.maxBytesPerFile ?? 4_000)
    await this.ensure()
    const order = new Map(
      ['SOUL.md', 'IDENTITY.md', 'USER.md', 'EXPERIENCES.md', 'MEMORY.md', 'KNOWLEDGE.md', 'INSIGHTS.md'].map(
        (name, index) => [name, index],
      ),
    )
    const entries = await this.listFiles()
    entries.sort((left, right) => {
      const scopeOrder = left.scope === right.scope ? 0 : left.scope === AgentMemoryScope.PROJECT ? -1 : 1
      if (scopeOrder !== 0) return scopeOrder
      const fileOrder = (order.get(left.path) ?? 99) - (order.get(right.path) ?? 99)
      return fileOrder !== 0 ? fileOrder : left.path.localeCompare(right.path)
    })

    const sections = [
      '# Persistent agent memory',
      'Use global memory for cross-project facts and project memory for this codebase. Read before risky work and record durable decisions, stable user preferences, recurring failures, and reusable wins before ending a substantive turn.',
      'Do not write memory for routine questions, arithmetic, transient test prompts, raw tool output, or facts already present. A turn with no durable new information should perform no memory write.',
      'Available tools: agent_memory_read, agent_memory_write, agent_memory_append, agent_memory_journal, agent_memory_search, agent_memory_list, and agent_memory_status.',
      '## Current memory contents',
    ]
    for (const entry of entries) {
      if (!entry.path.endsWith('.md') || entry.bytes === 0 || !this.shouldIncludeInPrompt(entry.path)) continue
      let body: string
      try {
        body = (await this.read(entry.scope, entry.path)).trim()
      } catch {
        continue
      }
      if (!body) continue
      if (Buffer.byteLength(body) > maxBytesPerFile) {
        const tail = entry.path.startsWith('journal/') || entry.path === 'EXPERIENCES.md'
        const shortened = tail ? body.slice(-maxBytesPerFile) : body.slice(0, maxBytesPerFile)
        body = shortened + '\n\n[Memory file truncated; use agent_memory_read for full text.]'
      }
      sections.push('### [' + entry.scope + '] ' + entry.path + '\n\n' + body)
    }
    sections.push(
      '## Before ending the turn',
      'Only if this substantive turn produced durable new information, write it to the appropriate memory file or journal now. Otherwise do not call a memory-writing tool.',
    )
    return sections.join('\n\n').trimEnd() + '\n'
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
          result.push({
            scope,
            path: this.relativePath(scope, fullPath),
            bytes: (await stat(fullPath)).size,
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

function validateLimit(limit: number): number {
  if (!Number.isInteger(limit) || limit < 1 || limit > 1_000_000) {
    throw new ValidationError('limit', 'must be an integer between 1 and 1000000', limit)
  }
  return limit
}
