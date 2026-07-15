// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lstat, readdir, readFile, realpath } from 'node:fs/promises'
import { basename, extname, resolve } from 'node:path'

const CHARS_PER_TOKEN = 4
const DEFAULT_MAX_FILE_BYTES = 1_000_000
const DEFAULT_MAX_FILES = 200
const DEFAULT_MAX_SYMBOLS_PER_FILE = 15
const DEFAULT_TOKEN_BUDGET = 2_048
const MAX_GITIGNORE_BYTES = 128_000
const PYTHON_FUNCTION = /^(async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))?\s*(?:->\s*([^:]+))?\s*:/

const IGNORED_DIRECTORIES = new Set([
  '.git', '.hg', '.idea', '.mypy_cache', '.pytest_cache', '.ruff_cache', '.svn', '.tox', '.venv', '.vscode',
  '__pycache__', 'build', 'dist', 'env', 'node_modules', 'venv',
])
const IGNORED_SUFFIXES = [
  '.7z', '.a', '.bmp', '.bz2', '.dll', '.dylib', '.gif', '.gz', '.ico', '.jpeg', '.jpg', '.lock', '.min.css',
  '.min.js', '.o', '.pdf', '.png', '.pyo', '.pyc', '.rar', '.so', '.svg', '.tar', '.tiff', '.wasm', '.zip',
] as const
const SOURCE_SUFFIXES = new Set([
  '.bash', '.c', '.cc', '.cpp', '.go', '.h', '.hpp', '.java', '.js', '.jsx', '.kt', '.lua', '.php', '.py', '.rb',
  '.rs', '.scala', '.sh', '.swift', '.ts', '.tsx', '.zsh',
])

export interface RepoMapSymbol {
  readonly file: string
  readonly kind: string
  readonly line: number
  readonly name: string
  readonly signature: string
}

/** Python-compatible name for a symbol entry. */
export type Symbol = RepoMapSymbol

export interface RepoMapConfigOptions {
  readonly maxFileBytes?: number
  readonly maxFiles?: number
  readonly maxSymbolsPerFile?: number
  readonly recencyWeight?: number
  readonly referenceWeight?: number
  readonly tokenBudget?: number
}

/** Validated knobs for deterministic repository-map construction. */
export class RepoMapConfig {
  readonly maxFileBytes: number
  readonly maxFiles: number
  readonly maxSymbolsPerFile: number
  readonly recencyWeight: number
  readonly referenceWeight: number
  readonly tokenBudget: number

  constructor(options: RepoMapConfigOptions = {}) {
    this.tokenBudget = nonNegativeInteger(options.tokenBudget ?? DEFAULT_TOKEN_BUDGET, 'tokenBudget')
    this.maxFiles = positiveInteger(options.maxFiles ?? DEFAULT_MAX_FILES, 'maxFiles')
    this.maxSymbolsPerFile = positiveInteger(
      options.maxSymbolsPerFile ?? DEFAULT_MAX_SYMBOLS_PER_FILE,
      'maxSymbolsPerFile',
    )
    this.maxFileBytes = positiveInteger(options.maxFileBytes ?? DEFAULT_MAX_FILE_BYTES, 'maxFileBytes')
    this.referenceWeight = nonNegativeNumber(options.referenceWeight ?? 3, 'referenceWeight')
    this.recencyWeight = nonNegativeNumber(options.recencyWeight ?? 1, 'recencyWeight')
  }
}

export interface RepoMapResult {
  readonly estimatedTokens: number
  readonly fileCount: number
  readonly includedCount: number
  readonly symbolCount: number
  readonly text: string
}

export interface RepoMapBuildOptions {
  /** Re-extract all symbols instead of reusing unchanged file cache entries. */
  readonly force?: boolean
}

interface CachedFile {
  readonly mtimeMs: number
  readonly size: number
  readonly symbols: readonly RepoMapSymbol[]
}

interface SourceFile {
  readonly absolutePath: string
  readonly displayPath: string
  readonly mtimeMs: number
  readonly relativePath: string
  readonly size: number
}

interface ScoredSymbol {
  readonly references: number
  readonly score: number
  readonly symbol: RepoMapSymbol
}

interface Declaration {
  readonly kind: string
  readonly name: string
  readonly signature: string
}

type SourceText = readonly [string, string]

/**
 * Builds a compact, ranked map from one or more workspace roots.
 *
 * Source scanning deliberately uses a declaration-line extractor rather than claiming
 * complete parsers for every language. It recognizes public Python definitions/classes
 * and common TS/JS/Go/Rust-style declaration forms deterministically; ambiguous or
 * multiline grammar is skipped rather than guessed.
 */
export class RepoMapper {
  readonly config: RepoMapConfig
  private readonly fileCache = new Map<string, CachedFile>()

  constructor(config: RepoMapConfig | RepoMapConfigOptions = {}) {
    this.config = config instanceof RepoMapConfig ? config : new RepoMapConfig(config)
  }

  /** Scan supplied workspace root(s), rank extracted declarations, and render within the token budget. */
  async build(root: string | readonly string[], options: RepoMapBuildOptions = {}): Promise<RepoMapResult> {
    const roots = await workspaceRoots(root)
    if (!roots.length) {
      return emptyResult()
    }
    if (options.force) {
      this.fileCache.clear()
    }

    const sourceFiles = await collectSourceFiles(roots, this.config)
    const currentPaths = new Set(sourceFiles.map(file => file.absolutePath))
    for (const path of this.fileCache.keys()) {
      if (!currentPaths.has(path)) {
        this.fileCache.delete(path)
      }
    }

    const sourceTexts: SourceText[] = []
    const symbols: RepoMapSymbol[] = []
    for (const file of sourceFiles) {
      const source = await readSource(file.absolutePath, this.config.maxFileBytes)
      if (source === undefined) {
        continue
      }
      sourceTexts.push([file.displayPath, source])
      const cached = this.fileCache.get(file.absolutePath)
      const relativeSymbols = !options.force
        && cached?.mtimeMs === file.mtimeMs
        && cached.size === file.size
        ? cached.symbols
        : extractSymbols(source, file.relativePath)
      if (!cached || options.force || cached.mtimeMs !== file.mtimeMs || cached.size !== file.size) {
        this.fileCache.set(file.absolutePath, {
          mtimeMs: file.mtimeMs,
          size: file.size,
          symbols: relativeSymbols,
        })
      }
      symbols.push(...relativeSymbols.map(symbol => ({ ...symbol, file: file.displayPath })))
    }

    if (!symbols.length) {
      return {
        ...emptyResult(),
        fileCount: sourceTexts.length,
      }
    }

    const referenceCounts = countReferences(symbols, sourceTexts)
    const mtimes = new Map(sourceFiles.map(file => [file.displayPath, file.mtimeMs]))
    const maxMtime = Math.max(1, ...mtimes.values())
    const scored = symbols.map(symbol => {
      const baseName = symbolBaseName(symbol.name)
      const references = referenceCounts.get(baseName) ?? 1
      const recency = (mtimes.get(symbol.file) ?? 0) / maxMtime
      return {
        symbol,
        references,
        score: references * this.config.referenceWeight + recency * this.config.recencyWeight,
      }
    }).sort(compareScoredSymbols)

    return renderRepoMap(scored, sourceTexts.length, this.config)
  }

  /** Drop cached symbols for one absolute source path, or clear every cache entry. */
  invalidate(filePath?: string): void {
    if (filePath === undefined) {
      this.fileCache.clear()
      return
    }
    this.fileCache.delete(resolve(filePath))
  }
}

/** Convenience wrapper that returns only prompt-ready repo-map text. */
export async function buildRepoMap(
  root: string | readonly string[],
  config: RepoMapConfig | RepoMapConfigOptions = {},
): Promise<string> {
  return (await new RepoMapper(config).build(root)).text
}

/** Extract declaration-line symbols without traversing the filesystem. */
export function extractRepoMapSymbols(source: string, file: string): readonly RepoMapSymbol[] {
  return extractSymbols(source, file)
}

async function workspaceRoots(root: string | readonly string[]): Promise<string[]> {
  const candidates = typeof root === 'string' ? [root] : [...root]
  const resolved: string[] = []
  for (const candidate of candidates) {
    try {
      const canonical = await realpath(candidate)
      if (!(await lstat(canonical)).isDirectory() || resolved.includes(canonical)) {
        continue
      }
      resolved.push(canonical)
    } catch {
      // A missing or unreadable supplied workspace has no map rather than aborting a turn.
    }
  }
  return resolved.sort(compareText)
}

async function collectSourceFiles(roots: readonly string[], config: RepoMapConfig): Promise<SourceFile[]> {
  const labels = rootLabels(roots)
  const filesByPath = new Map<string, SourceFile>()
  for (const root of roots) {
    const patterns = await gitignorePatterns(root)
    const label = labels.get(root) ?? basename(root)
    const candidates = await walkRoot(root, patterns)
    for (const candidate of candidates) {
      const displayPath = roots.length === 1 ? candidate.relativePath : `${label}/${candidate.relativePath}`
      if (!filesByPath.has(candidate.absolutePath)) {
        filesByPath.set(candidate.absolutePath, { ...candidate, displayPath })
      }
    }
  }
  const files = [...filesByPath.values()]
  files.sort((left, right) => {
    return compareText(left.displayPath, right.displayPath)
      || compareText(left.absolutePath, right.absolutePath)
  })
  return files.slice(0, config.maxFiles)
}

async function walkRoot(root: string, patterns: readonly string[]): Promise<Omit<SourceFile, 'displayPath'>[]> {
  const files: Array<Omit<SourceFile, 'displayPath'>> = []
  const pending: Array<{ readonly absolutePath: string; readonly relativePath: string }> = [{
    absolutePath: root,
    relativePath: '',
  }]
  while (pending.length) {
    const current = pending.pop()
    if (!current) {
      continue
    }
    let entries
    try {
      entries = await readdir(current.absolutePath, { withFileTypes: true })
    } catch {
      continue
    }
    entries.sort((left, right) => compareText(left.name, right.name))
    for (const entry of entries) {
      const relativePath = current.relativePath ? `${current.relativePath}/${entry.name}` : entry.name
      const absolutePath = resolve(current.absolutePath, entry.name)
      if (entry.isSymbolicLink() || ignoredPath(relativePath, entry.name, entry.isDirectory(), patterns)) {
        continue
      }
      if (entry.isDirectory()) {
        pending.push({ absolutePath, relativePath })
        continue
      }
      if (!entry.isFile() || !sourceFile(entry.name)) {
        continue
      }
      try {
        const status = await lstat(absolutePath)
        if (!status.isFile() || status.isSymbolicLink()) {
          continue
        }
        files.push({
          absolutePath,
          relativePath,
          mtimeMs: status.mtimeMs,
          size: status.size,
        })
      } catch {
        // Files can disappear or become unreadable while a workspace is scanned.
      }
    }
  }
  return files
}

function rootLabels(roots: readonly string[]): Map<string, string> {
  const labels = new Map<string, string>()
  const used = new Set<string>()
  for (const root of roots) {
    const base = basename(root) || 'workspace'
    let label = base
    let suffix = 2
    while (used.has(label)) {
      label = `${base}-${suffix}`
      suffix += 1
    }
    used.add(label)
    labels.set(root, label)
  }
  return labels
}

async function gitignorePatterns(root: string): Promise<string[]> {
  try {
    const path = resolve(root, '.gitignore')
    if ((await lstat(path)).size > MAX_GITIGNORE_BYTES) {
      return []
    }
    const source = await readFile(path, 'utf8')
    return source.split(/\r?\n/).flatMap(line => {
      const pattern = line.trim()
      if (!pattern || pattern.startsWith('#') || pattern.startsWith('!')) {
        return []
      }
      return [pattern.replace(/^\/+/, '').replace(/\/+$/, '')]
    })
  } catch {
    return []
  }
}

function ignoredPath(relativePath: string, name: string, directory: boolean, patterns: readonly string[]): boolean {
  if (directory && (IGNORED_DIRECTORIES.has(name) || name.endsWith('.egg-info'))) {
    return true
  }
  if (!directory && IGNORED_SUFFIXES.some(suffix => name.toLowerCase().endsWith(suffix))) {
    return true
  }
  return patterns.some(pattern => matchesGitignore(relativePath, pattern))
}

function matchesGitignore(relativePath: string, pattern: string): boolean {
  if (!pattern) {
    return false
  }
  const segments = relativePath.split('/')
  if (pattern.startsWith('*.') && relativePath.endsWith(pattern.slice(1))) {
    return true
  }
  if (!pattern.includes('/')) {
    return segments.includes(pattern)
  }
  return relativePath === pattern || relativePath.startsWith(`${pattern}/`)
}

function sourceFile(name: string): boolean {
  return SOURCE_SUFFIXES.has(extname(name).toLowerCase())
}

async function readSource(path: string, maximumBytes: number): Promise<string | undefined> {
  try {
    const status = await lstat(path)
    if (!status.isFile() || status.size > maximumBytes) {
      return undefined
    }
    return await readFile(path, 'utf8')
  } catch {
    return undefined
  }
}

function extractSymbols(source: string, file: string): RepoMapSymbol[] {
  return extname(file).toLowerCase() === '.py'
    ? extractPythonSymbols(source, file)
    : extractDeclarationSymbols(source, file)
}

function extractPythonSymbols(source: string, file: string): RepoMapSymbol[] {
  const symbols: RepoMapSymbol[] = []
  const scopes: Array<{
    readonly includeMethods: boolean
    readonly indent: number
    readonly kind: 'class' | 'function'
    readonly name: string
  }> = []
  const lines = source.split(/\r?\n/)
  for (let index = 0; index < lines.length; index += 1) {
    const raw = lines[index] ?? ''
    const content = raw.trim()
    if (!content || content.startsWith('#')) {
      continue
    }
    const indent = raw.length - raw.trimStart().length
    while (scopes.length && indent <= (scopes.at(-1)?.indent ?? -1)) {
      scopes.pop()
    }
    const classMatch = /^(?:class)\s+([A-Za-z_][A-Za-z0-9_]*)/.exec(content)
    if (classMatch) {
      const name = classMatch[1] ?? ''
      const publicTopLevel = indent === 0 && !name.startsWith('_')
      scopes.push({ kind: 'class', name, indent, includeMethods: publicTopLevel })
      if (publicTopLevel) {
        symbols.push(symbol(file, name, 'class', index + 1, `class ${name}`))
      }
      continue
    }
    const functionMatch = PYTHON_FUNCTION.exec(content)
    if (functionMatch) {
      const prefix = functionMatch[1] ?? 'def'
      const name = functionMatch[2] ?? ''
      if (name.startsWith('_')) {
        scopes.push({ kind: 'function', name, indent, includeMethods: false })
        continue
      }
      const enclosingScope = scopes.at(-1)
      if (indent === 0) {
        symbols.push(symbol(
          file,
          name,
          prefix === 'async def' ? 'async_function' : 'function',
          index + 1,
          pythonSignature(functionMatch),
        ))
      } else if (enclosingScope?.kind === 'class' && enclosingScope.includeMethods && indent > enclosingScope.indent) {
        symbols.push(symbol(file, `${enclosingScope.name}.${name}`, 'method', index + 1))
      }
      scopes.push({ kind: 'function', name, indent, includeMethods: false })
      continue
    }
    const constantMatch = /^([A-Z][A-Z0-9_]*)\s*(?::[^=]+)?=/.exec(content)
    if (indent === 0 && constantMatch?.[1]) {
      symbols.push(symbol(file, constantMatch[1], 'constant', index + 1))
    }
  }
  return symbols
}

function pythonSignature(match: RegExpExecArray): string {
  const prefix = match[1] ?? 'def'
  const name = match[2] ?? ''
  const parameters = (match[3] ?? '()').replace(/\s+/g, ' ')
  const returns = match[4]?.trim()
  return `${prefix} ${name}${parameters}${returns ? ` -> ${returns}` : ''}`
}

function extractDeclarationSymbols(source: string, file: string): RepoMapSymbol[] {
  const symbols: RepoMapSymbol[] = []
  const seen = new Set<string>()
  const lines = source.split(/\r?\n/)
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index] ?? ''
    const declaration = declarationFromLine(line)
    if (!declaration || declaration.name.startsWith('_')) {
      continue
    }
    const key = `${declaration.kind}:${declaration.name}`
    if (seen.has(key)) {
      continue
    }
    seen.add(key)
    symbols.push(symbol(file, declaration.name, declaration.kind, index + 1, declaration.signature))
  }
  return symbols
}

function declarationFromLine(line: string): Declaration | undefined {
  const patterns: ReadonlyArray<readonly [string, RegExp, (name: string) => string]> = [
    [
      'function',
      /^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)/,
      name => `function ${name}`,
    ],
    [
      'const_arrow',
      /^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>/,
      name => name,
    ],
    ['class', /^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)/, name => `class ${name}`],
    ['interface', /^\s*(?:export\s+)?(?:declare\s+)?interface\s+([A-Za-z_$][\w$]*)/, name => `interface ${name}`],
    ['type', /^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*=/, name => `type ${name}`],
    ['fn', /^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)/, name => `fn ${name}`],
    ['func', /^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)/, name => `func ${name}`],
    ['struct', /^\s*(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)/, name => `struct ${name}`],
    ['constant', /^\s*(?:export\s+)?const\s+([A-Z][A-Z0-9_]*)\s*=/, name => name],
  ]
  for (const [kind, pattern, signature] of patterns) {
    const match = pattern.exec(line)
    if (match?.[1]) {
      return { kind, name: match[1], signature: signature(match[1]) }
    }
  }
  return undefined
}

function symbol(file: string, name: string, kind: string, line: number, signature = ''): RepoMapSymbol {
  return { file, name, kind, line, signature }
}

function countReferences(symbols: readonly RepoMapSymbol[], sources: readonly SourceText[]): Map<string, number> {
  const names = new Set(symbols.map(symbol => symbolBaseName(symbol.name)))
  const counts = new Map<string, number>()
  for (const [, source] of sources) {
    const tokens = new Set(source.match(/[A-Za-z_$][A-Za-z0-9_$]*/g) ?? [])
    for (const name of names) {
      if (tokens.has(name)) {
        counts.set(name, (counts.get(name) ?? 0) + 1)
      }
    }
  }
  return counts
}

function renderRepoMap(scored: readonly ScoredSymbol[], fileCount: number, config: RepoMapConfig): RepoMapResult {
  const perFile = new Map<string, ScoredSymbol[]>()
  for (const item of scored) {
    const symbols = perFile.get(item.symbol.file) ?? []
    symbols.push(item)
    perFile.set(item.symbol.file, symbols)
  }
  const files = [...perFile.keys()].sort((left, right) => {
    const leftScore = perFile.get(left)?.[0]?.score ?? 0
    const rightScore = perFile.get(right)?.[0]?.score ?? 0
    return rightScore - leftScore || compareText(left, right)
  })
  let includedCount = 0
  let text = ''
  for (const file of files) {
    const withHeader = text ? `${text}\n\n${file}` : file
    if (estimateTokens(withHeader) > config.tokenBudget) {
      break
    }
    text = withHeader
    const symbols = perFile.get(file)?.slice(0, config.maxSymbolsPerFile) ?? []
    for (const item of symbols) {
      const refs = item.references > 1 ? ` (${item.references} refs)` : ''
      const signature = item.symbol.signature || item.symbol.name
      const entry = `  ${item.symbol.kind}: ${signature}${refs}`
      const withEntry = `${text}\n${entry}`
      if (estimateTokens(withEntry) > config.tokenBudget) {
        const withEllipsis = `${text}\n  ...`
        if (estimateTokens(withEllipsis) <= config.tokenBudget) {
          text = withEllipsis
        }
        break
      }
      text = withEntry
      includedCount += 1
    }
  }
  return {
    text,
    fileCount,
    symbolCount: scored.length,
    includedCount,
    estimatedTokens: text ? estimateTokens(text) : 0,
  }
}

function compareScoredSymbols(left: ScoredSymbol, right: ScoredSymbol): number {
  return right.score - left.score
    || compareText(left.symbol.file, right.symbol.file)
    || left.symbol.line - right.symbol.line
    || compareText(left.symbol.name, right.symbol.name)
}

function symbolBaseName(name: string): string {
  return name.slice(name.lastIndexOf('.') + 1)
}

function estimateTokens(text: string): number {
  return text ? Math.max(1, Math.ceil(text.length / CHARS_PER_TOKEN)) : 0
}

function emptyResult(): RepoMapResult {
  return { text: '', fileCount: 0, symbolCount: 0, includedCount: 0, estimatedTokens: 0 }
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer`)
  }
  return value
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative safe integer`)
  }
  return value
}

function nonNegativeNumber(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative finite number`)
  }
  return value
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}
