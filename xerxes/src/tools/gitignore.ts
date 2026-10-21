// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lstat } from 'node:fs/promises'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

import { WorkspacePathError, type WorkspacePathResolver } from './pathSafety.js'

interface IgnoreRule {
  readonly directory: string
  readonly expression: RegExp
  readonly basenameOnly: boolean
  readonly directoryOnly: boolean
  readonly negated: boolean
}

interface DirectoryRules {
  readonly ignored: boolean
  readonly rules: readonly IgnoreRule[]
}

/** One search's ignore state. No parsed files or directory decisions survive the walk. */
export class GitignoreMatcher {
  private readonly directories = new Map<string, Promise<DirectoryRules>>()
  private readonly files = new Map<string, Promise<string | undefined>>()

  private constructor(
    private readonly root: string,
    private readonly paths: WorkspacePathResolver,
    private readonly excludeRules: readonly IgnoreRule[],
  ) {}

  /** Search for a work-tree marker without inspecting any ancestor outside the workspace. */
  static async create(searchDirectory: string, paths: WorkspacePathResolver): Promise<GitignoreMatcher | undefined> {
    const boundary = await paths.resolve('.')
    let current = searchDirectory
    while (within(boundary, current)) {
      try {
        const marker = await lstat(join(current, '.git'))
        if (marker.isDirectory() || marker.isFile()) {
          const matcher = new GitignoreMatcher(current, paths, [])
          let gitDirectory: string | undefined = join(current, '.git')
          if (marker.isFile()) {
            const pointer = await matcher.readOptional(gitDirectory)
            const gitdir = /^gitdir: (.+)\r?$/m.exec(pointer ?? '')?.[1]?.trim()
            gitDirectory = gitdir ? resolve(current, gitdir) : undefined
          }
          if (gitDirectory) {
            // Linked worktrees keep info/exclude in their common Git directory.
            // Both pointers are untrusted and readOptional enforces the workspace boundary.
            const common = await matcher.readOptional(join(gitDirectory, 'commondir'))
            const commonDirectory = common?.trim() ? resolve(gitDirectory, common.trim()) : gitDirectory
            const exclude = await matcher.readOptional(join(commonDirectory, 'info/exclude'))
            return new GitignoreMatcher(current, paths, parseRules(exclude ?? '', current))
          }
          return matcher
        }
      } catch (error) {
        if (!missing(error)) throw error
      }
      if (current === boundary) break
      current = dirname(current)
    }
    return undefined
  }

  /** Ancestors are checked first: a negation cannot resurrect a child of an excluded directory. */
  async isIgnored(candidate: string, directory: boolean): Promise<boolean> {
    if (!within(this.root, candidate)) return false
    const path = relative(this.root, candidate)
    if (path.split(sep).includes('.git')) return true
    if (!path) return false
    if (directory) return (await this.directoryRules(candidate)).ignored
    const parent = await this.directoryRules(dirname(candidate))
    return parent.ignored || ignoredBy(parent.rules, candidate, false)
  }

  private directoryRules(directory: string): Promise<DirectoryRules> {
    const cached = this.directories.get(directory)
    if (cached) return cached
    const pending = this.loadDirectory(directory)
    this.directories.set(directory, pending)
    return pending
  }

  private async loadDirectory(directory: string): Promise<DirectoryRules> {
    const parent = directory === this.root
      ? { ignored: false, rules: this.excludeRules }
      : await this.directoryRules(dirname(directory))
    if (parent.ignored || (directory !== this.root && ignoredBy(parent.rules, directory, true))) {
      return { ignored: true, rules: parent.rules }
    }
    const text = await this.readOptional(join(directory, '.gitignore'))
    return { ignored: false, rules: [...parent.rules, ...parseRules(text ?? '', directory)] }
  }

  private readOptional(candidate: string): Promise<string | undefined> {
    const cached = this.files.get(candidate)
    if (cached) return cached
    const pending = this.readBounded(candidate)
    this.files.set(candidate, pending)
    return pending
  }

  private async readBounded(candidate: string): Promise<string | undefined> {
    try {
      const checked = await this.paths.resolve(candidate)
      // Git does not follow symlinked .gitignore files. Also avoid reading devices/directories.
      if (!(await lstat(candidate)).isFile()) return undefined
      return await Bun.file(checked).text()
    } catch (error) {
      if (error instanceof WorkspacePathError || missing(error)) return undefined
      throw error
    }
  }
}

function ignoredBy(rules: readonly IgnoreRule[], candidate: string, directory: boolean): boolean {
  let ignored = false
  for (const rule of rules) {
    if (rule.directoryOnly && !directory) continue
    if (!within(rule.directory, candidate)) continue
    const path = relative(rule.directory, candidate).split(sep).join('/')
    if (rule.expression.test(utf8Bytes(rule.basenameOnly ? basename(candidate) : path))) ignored = !rule.negated
  }
  return ignored
}

function parseRules(text: string, directory: string): IgnoreRule[] {
  const rules: IgnoreRule[] = []
  for (let line of text.replace(/^\uFEFF/, '').split('\n')) {
    line = trimTrailingSpaces(line.replace(/\r$/, ''))
    if (!line || line.startsWith('#')) continue
    const negated = line.startsWith('!')
    if (negated) line = line.slice(1)
    if (!line) continue
    const directoryOnly = line.endsWith('/')
    if (directoryOnly) line = line.slice(0, -1)
    const basenameOnly = !line.includes('/')
    if (line.startsWith('/')) line = line.slice(1)
    if (!line) continue
    const source = wildmatch(utf8Bytes(line))
    if (source === undefined) continue
    rules.push({ directory, expression: new RegExp(`^${source}$(?![\\s\\S])`), basenameOnly, directoryOnly, negated })
  }
  return rules
}

/** Unescaped trailing spaces disappear; an escaped final space (and earlier spaces) stays. */
function trimTrailingSpaces(line: string): string {
  let end = line.length
  while (end > 0 && line[end - 1] === ' ') {
    let backslashes = 0
    for (let index = end - 2; index >= 0 && line[index] === '\\'; index -= 1) backslashes += 1
    if (backslashes % 2 === 1) break
    end -= 1
  }
  return line.slice(0, end)
}

/** Compile git's pathname wildcards, not Bun globs (braces and leading ! are literal here). */
function wildmatch(pattern: string): string | undefined {
  let source = ''
  for (let index = 0; index < pattern.length; index += 1) {
    const character = pattern[index]!
    if (character === '\\') {
      if (index + 1 === pattern.length) return undefined
      source += literal(pattern[++index]!)
    } else if (character === '*') {
      let end = index + 1
      while (pattern[end] === '*') end += 1
      const segmentStart = index === 0 || pattern[index - 1] === '/'
      const segmentEnd = end === pattern.length || pattern[end] === '/'
      if (end - index >= 2 && segmentStart && segmentEnd) {
        if (pattern[end] === '/') {
          source += '(?:[^/]+/)*'
          end += 1
        } else {
          source += '[\\s\\S]*'
        }
      } else {
        source += '[^/]*'
      }
      index = end - 1
    } else if (character === '?') {
      source += '[^/]'
    } else if (character === '[') {
      const bracket = characterClass(pattern, index)
      if (!bracket) return undefined
      source += bracket.source
      index = bracket.end
    } else {
      source += literal(character)
    }
  }
  return source
}

const POSIX_CLASSES: Readonly<Record<string, string>> = {
  alnum: 'A-Za-z0-9', alpha: 'A-Za-z', blank: ' \\t', cntrl: '\\x00-\\x1f\\x7f',
  digit: '0-9', graph: '\\x21-\\x7e', lower: 'a-z', print: '\\x20-\\x7e',
  punct: '\\x21-\\x2f\\x3a-\\x40\\x5b-\\x60\\x7b-\\x7e',
  space: ' \\t\\r\\n\\v\\f', upper: 'A-Z', xdigit: 'A-Fa-f0-9',
}

function characterClass(pattern: string, start: number): { source: string; end: number } | undefined {
  let index = start + 1
  const negated = pattern[index] === '!' || pattern[index] === '^'
  if (negated) index += 1
  let body = ''
  let previous: string | undefined
  if (pattern[index] === ']') {
    body = '\\]'
    previous = ']'
    index += 1
  }
  for (; index < pattern.length; index += 1) {
    const character = pattern[index]!
    if (character === ']') {
      if (!body) return undefined
      const source = `(?=[^/])[${negated ? '^' : ''}${body}]`
      try {
        new RegExp(source)
        return { source, end: index }
      } catch {
        return undefined
      }
    }
    if (character === '-' && previous !== undefined && pattern[index + 1] !== ']') {
      if (++index === pattern.length) return undefined
      if (pattern[index] === '\\' && ++index === pattern.length) return undefined
      const end = pattern[index]!
      // Git retains the first literal in a descending range; JS rejects the whole class.
      if (previous <= end) body += `-${classLiteral(end)}`
      previous = undefined
    } else if (character === '\\') {
      if (++index === pattern.length) return undefined
      previous = pattern[index]!
      body += classLiteral(previous)
    } else if (character === '[' && pattern[index + 1] === ':') {
      const end = pattern.indexOf(':]', index + 2)
      const expansion = POSIX_CLASSES[pattern.slice(index + 2, end)]
      if (end === -1 || !expansion) return undefined
      body += expansion
      previous = undefined
      index = end + 1
    } else {
      body += classLiteral(character)
      previous = character
    }
  }
  return undefined
}

const UTF8_ENCODER = new TextEncoder()

/** Git wildmatch operates on bytes, so ? consumes one UTF-8 byte, not one JS code unit. */
function utf8Bytes(text: string): string {
  if (!/[^\x00-\x7f]/.test(text)) return text
  let bytes = ''
  for (const byte of UTF8_ENCODER.encode(text)) bytes += String.fromCharCode(byte)
  return bytes
}

function literal(character: string): string {
  return /[\\^$.*+?()[\]{}|]/.test(character) ? `\\${character}` : character
}

function classLiteral(character: string): string {
  return /[\\\]\[\^-]/.test(character) ? `\\${character}` : character
}

function within(root: string, candidate: string): boolean {
  const path = relative(root, candidate)
  return path === '' || (path !== '..' && !path.startsWith(`..${sep}`) && !isAbsolute(path))
}

function missing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error
    && (error.code === 'ENOENT' || error.code === 'ENOTDIR')
}
