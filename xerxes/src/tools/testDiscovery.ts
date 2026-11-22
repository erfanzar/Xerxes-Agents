// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync, readdirSync, readFileSync, realpathSync, statSync } from 'node:fs'
import { basename, dirname, extname, isAbsolute, join, relative, resolve, sep } from 'node:path'

export type TestRunnerKind = 'bun' | 'vitest' | 'jest' | 'pytest' | 'go' | 'cargo'

export interface DiscoverRelatedTestsOptions {
  /** Maximum related test paths returned (default: 8). */
  readonly maxResults?: number
}

export interface RelatedTestsResult {
  /** Deterministic sorted workspace-relative POSIX paths to existing related test files. */
  readonly relatedTests: readonly string[]
  /** Detected native workspace test runner, when identifiable. */
  readonly runner?: TestRunnerKind
  /** Suggested non-interactive test command scoped to the discovered test targets. */
  readonly runnerCommand?: string
  /** Concise single-line `[related_tests]` hint suitable for appending to edit/patch summaries. */
  readonly hint?: string
}

type LanguageFamily = 'js' | 'python' | 'go' | 'rust'

const JS_SOURCE_EXTENSIONS = new Set(['.ts', '.tsx', '.mts', '.cts', '.js', '.jsx', '.mjs', '.cjs'])
const JS_TEST_SUFFIXES = [
  '.test.ts',
  '.test.tsx',
  '.spec.ts',
  '.spec.tsx',
  '.test.js',
  '.test.jsx',
  '.spec.js',
  '.spec.jsx',
  '.test.mts',
  '.spec.mts',
  '.test.mjs',
  '.spec.mjs',
] as const

const IGNORED_SCAN_DIRS = new Set([
  '.git',
  '.hg',
  '.svn',
  'node_modules',
  'dist',
  'build',
  'out',
  'coverage',
  'target',
  '.venv',
  'venv',
  '__pycache__',
  '.next',
  '.turbo',
])

const MAX_TEST_DIR_SCAN_FILES = 600
const DEFAULT_MAX_RELATED_TESTS = 8

/**
 * Deterministically map edited workspace source files to existing test files and
 * detect the workspace's native test runner command without spawning processes.
 */
export function discoverRelatedTests(
  workspaceRoot: string,
  changedFilePaths: readonly string[],
  options: DiscoverRelatedTestsOptions = {},
): RelatedTestsResult {
  const maxResults = options.maxResults !== undefined && Number.isInteger(options.maxResults) && options.maxResults > 0
    ? options.maxResults
    : DEFAULT_MAX_RELATED_TESTS
  const canonicalRoot = safeRealpath(resolve(workspaceRoot))
  if (!canonicalRoot || !isDirectorySync(canonicalRoot)) {
    return { relatedTests: [] }
  }

  const found = new Set<string>()
  const touchedFamilies = new Set<LanguageFamily>()
  const packageRoots = new Set<string>([canonicalRoot])
  const scanCache = new Map<string, readonly string[]>()

  for (const rawPath of changedFilePaths) {
    if (!rawPath.trim() || rawPath.includes('\0')) continue
    const relPosix = normalizeWorkspaceRelative(canonicalRoot, rawPath)
    if (!relPosix) continue
    if (isTestFilePath(relPosix)) continue
    const family = detectLanguageFamily(relPosix)
    if (!family) continue

    touchedFamilies.add(family)
    const pkgRoot = findNearestPackageRoot(canonicalRoot, relPosix, family)
    packageRoots.add(pkgRoot)

    const candidates = candidateTestPathsForSource(canonicalRoot, pkgRoot, relPosix, family, scanCache)
    for (const candidate of candidates) {
      if (found.size >= maxResults) break
      found.add(candidate)
    }
  }

  const relatedTests = [...found].sort((left, right) => left.localeCompare(right)).slice(0, maxResults)
  const primaryFamily = pickPrimaryFamily(relatedTests, touchedFamilies)
  const runnerInfo = primaryFamily === undefined
    ? undefined
    : detectTestRunner(canonicalRoot, [...packageRoots], primaryFamily, relatedTests)

  const hint = formatRelatedTestsLine(relatedTests, runnerInfo?.runnerCommand)
  return {
    relatedTests,
    ...(runnerInfo?.runner === undefined ? {} : { runner: runnerInfo.runner }),
    ...(runnerInfo?.runnerCommand === undefined ? {} : { runnerCommand: runnerInfo.runnerCommand }),
    ...(hint === undefined ? {} : { hint }),
  }
}

/** Format a concise `[related_tests]` suffix line when related tests were discovered. */
export function formatRelatedTestsHint(result: RelatedTestsResult): string | undefined {
  return result.hint ?? formatRelatedTestsLine(result.relatedTests, result.runnerCommand)
}

function formatRelatedTestsLine(relatedTests: readonly string[], runnerCommand: string | undefined): string | undefined {
  if (relatedTests.length === 0) return undefined
  const list = relatedTests.join(', ')
  return runnerCommand ? `[related_tests] ${list} (run: ${runnerCommand})` : `[related_tests] ${list}`
}

function detectLanguageFamily(relPosix: string): LanguageFamily | undefined {
  const lower = relPosix.toLowerCase()
  if (lower.endsWith('.d.ts') || lower.endsWith('.d.mts') || lower.endsWith('.d.cts')) return undefined
  const ext = extname(lower)
  if (JS_SOURCE_EXTENSIONS.has(ext)) return 'js'
  if (ext === '.py') return 'python'
  if (ext === '.go') return 'go'
  if (ext === '.rs') return 'rust'
  return undefined
}

export function isTestFilePath(relPosix: string): boolean {
  const normalized = relPosix.replaceAll('\\', '/')
  const lower = normalized.toLowerCase()
  const base = basename(lower)
  const segments = lower.split('/')

  if (JS_TEST_SUFFIXES.some(suffix => lower.endsWith(suffix))) return true
  if (segments.includes('__tests__')) return true
  if (lower.endsWith('_test.go')) return true
  if (lower.endsWith('.py') && (base.startsWith('test_') || base.endsWith('_test.py') || base === 'conftest.py')) {
    return true
  }
  if (lower.endsWith('.rs') && (segments.includes('tests') || base.endsWith('_test.rs') || base === 'tests.rs')) {
    return true
  }
  return false
}

function candidateTestPathsForSource(
  workspaceRoot: string,
  pkgRoot: string,
  relPosix: string,
  family: LanguageFamily,
  scanCache: Map<string, readonly string[]>,
): string[] {
  const pkgRelPrefix = toPosix(relative(workspaceRoot, pkgRoot))
  const relFromPkg = pkgRelPrefix && relPosix.startsWith(`${pkgRelPrefix}/`)
    ? relPosix.slice(pkgRelPrefix.length + 1)
    : relPosix

  const rootsToCheck = pkgRoot === workspaceRoot ? [workspaceRoot] : [pkgRoot, workspaceRoot]
  const hits: string[] = []
  const seen = new Set<string>()

  const pushIfExistingTest = (baseDir: string, candidateRelToBase: string): void => {
    const abs = resolve(baseDir, candidateRelToBase)
    const canonical = safeExistingRegularFileInside(workspaceRoot, abs)
    if (!canonical) return
    const workspaceRel = toPosix(relative(workspaceRoot, canonical))
    if (!workspaceRel || workspaceRel === relPosix || seen.has(workspaceRel)) return
    seen.add(workspaceRel)
    hits.push(workspaceRel)
  }

  for (const rootDir of rootsToCheck) {
    const localRel = rootDir === pkgRoot ? relFromPkg : relPosix
    const dir = dirname(localRel) === '.' ? '' : dirname(localRel)
    const ext = extname(localRel)
    const stem = basename(localRel, ext)
    if (!stem) continue

    const strippedSubdir = stripSourcePrefix(dir)

    if (family === 'js') {
      if (stem === 'index' && dir) {
        const parentName = basename(dir)
        for (const suffix of JS_TEST_SUFFIXES) {
          pushIfExistingTest(rootDir, joinPosix(dir, '__tests__', `${parentName}${suffix}`))
          pushIfExistingTest(rootDir, joinPosix('test', `${parentName}${suffix}`))
          pushIfExistingTest(rootDir, joinPosix('tests', `${parentName}${suffix}`))
        }
      }
      for (const suffix of JS_TEST_SUFFIXES) {
        // Co-located & __tests__
        pushIfExistingTest(rootDir, joinPosix(dir, `${stem}${suffix}`))
        pushIfExistingTest(rootDir, joinPosix(dir, '__tests__', `${stem}${suffix}`))
        // Mirrored under test/ and tests/
        if (strippedSubdir) {
          pushIfExistingTest(rootDir, joinPosix('test', strippedSubdir, `${stem}${suffix}`))
          pushIfExistingTest(rootDir, joinPosix('tests', strippedSubdir, `${stem}${suffix}`))
        }
        if (dir && dir !== strippedSubdir) {
          pushIfExistingTest(rootDir, joinPosix('test', dir, `${stem}${suffix}`))
          pushIfExistingTest(rootDir, joinPosix('tests', dir, `${stem}${suffix}`))
        }
        // Flat test/<stem>.test.ts (e.g. Xerxes convention: src/foo/bar.ts -> test/bar.test.ts)
        pushIfExistingTest(rootDir, joinPosix('test', `${stem}${suffix}`))
        pushIfExistingTest(rootDir, joinPosix('tests', `${stem}${suffix}`))
      }
      if (stem !== 'index') {
        for (const testFolder of ['test', 'tests', '__tests__']) {
          for (const entry of scanTestDirectory(workspaceRoot, join(rootDir, testFolder), scanCache)) {
            const lowerBase = basename(entry).toLowerCase()
            const lowerStem = stem.toLowerCase()
            if (JS_TEST_SUFFIXES.some(suffix => lowerBase === `${lowerStem}${suffix}`)) {
              pushIfExistingTest(workspaceRoot, entry)
            }
          }
        }
      }
    } else if (family === 'python') {
      const targetNames = stem === '__init__' && dir
        ? [`test_${basename(dir)}.py`, `${basename(dir)}_test.py`]
        : [`test_${stem}.py`, `${stem}_test.py`]

      for (const testName of targetNames) {
        pushIfExistingTest(rootDir, joinPosix(dir, testName))
        pushIfExistingTest(rootDir, joinPosix(dir, 'tests', testName))
        if (strippedSubdir) {
          pushIfExistingTest(rootDir, joinPosix('tests', strippedSubdir, testName))
          pushIfExistingTest(rootDir, joinPosix('test', strippedSubdir, testName))
        }
        if (dir && dir !== strippedSubdir) {
          pushIfExistingTest(rootDir, joinPosix('tests', dir, testName))
          pushIfExistingTest(rootDir, joinPosix('test', dir, testName))
        }
        // Also strip top-level package folder (e.g. pkg/sub/bar.py -> tests/sub/test_bar.py)
        const withoutTopPkg = stripFirstSegment(strippedSubdir || dir)
        if (withoutTopPkg) {
          pushIfExistingTest(rootDir, joinPosix('tests', withoutTopPkg, testName))
          pushIfExistingTest(rootDir, joinPosix('test', withoutTopPkg, testName))
        }
        pushIfExistingTest(rootDir, joinPosix('tests', testName))
        pushIfExistingTest(rootDir, joinPosix('test', testName))
      }
      if (stem !== '__init__') {
        const lowerTargets = new Set(targetNames.map(name => name.toLowerCase()))
        for (const testFolder of ['tests', 'test']) {
          for (const entry of scanTestDirectory(workspaceRoot, join(rootDir, testFolder), scanCache)) {
            if (lowerTargets.has(basename(entry).toLowerCase())) {
              pushIfExistingTest(workspaceRoot, entry)
            }
          }
        }
      }
    } else if (family === 'go') {
      pushIfExistingTest(rootDir, joinPosix(dir, `${stem}_test.go`))
      pushIfExistingTest(rootDir, joinPosix(dir, `${stem}_internal_test.go`))
      if (strippedSubdir) {
        pushIfExistingTest(rootDir, joinPosix('test', strippedSubdir, `${stem}_test.go`))
      }
      pushIfExistingTest(rootDir, joinPosix('test', `${stem}_test.go`))
    } else if (family === 'rust') {
      const targetStems = (stem === 'lib' || stem === 'main' || stem === 'mod') && dir && dir !== 'src'
        ? [basename(dir), stem]
        : [stem]
      for (const candidateStem of targetStems) {
        pushIfExistingTest(rootDir, joinPosix('tests', `${candidateStem}.rs`))
        pushIfExistingTest(rootDir, joinPosix('tests', `test_${candidateStem}.rs`))
        pushIfExistingTest(rootDir, joinPosix('tests', `${candidateStem}_test.rs`))
        if (strippedSubdir) {
          pushIfExistingTest(rootDir, joinPosix('tests', strippedSubdir, `${candidateStem}.rs`))
          pushIfExistingTest(rootDir, joinPosix('tests', strippedSubdir, `test_${candidateStem}.rs`))
        }
        pushIfExistingTest(rootDir, joinPosix(dir, `${candidateStem}_test.rs`))
        pushIfExistingTest(rootDir, joinPosix(dir, candidateStem, 'tests.rs'))
      }
      if (stem === 'lib' || stem === 'mod') {
        pushIfExistingTest(rootDir, joinPosix(dir, 'tests.rs'))
      }
    }
  }

  return hits
}

function scanTestDirectory(
  workspaceRoot: string,
  testDirAbs: string,
  cache: Map<string, readonly string[]>,
): readonly string[] {
  const cached = cache.get(testDirAbs)
  if (cached !== undefined) return cached
  const canonicalDir = safeRealpath(testDirAbs)
  if (!canonicalDir || !isWithinPath(workspaceRoot, canonicalDir) || !isDirectorySync(canonicalDir)) {
    cache.set(testDirAbs, [])
    return []
  }

  const collected: string[] = []
  const queue: string[] = [canonicalDir]
  while (queue.length > 0 && collected.length < MAX_TEST_DIR_SCAN_FILES) {
    const current = queue.shift()!
    let entries
    try {
      entries = readdirSync(current, { withFileTypes: true })
    } catch {
      continue
    }
    entries.sort((a, b) => a.name.localeCompare(b.name))
    for (const entry of entries) {
      if (collected.length >= MAX_TEST_DIR_SCAN_FILES) break
      if (entry.name.startsWith('.') || IGNORED_SCAN_DIRS.has(entry.name)) continue
      const childAbs = join(current, entry.name)
      if (entry.isSymbolicLink()) continue
      if (entry.isDirectory()) {
        queue.push(childAbs)
      } else if (entry.isFile()) {
        const rel = toPosix(relative(workspaceRoot, childAbs))
        if (rel && !rel.startsWith('..')) collected.push(rel)
      }
    }
  }

  cache.set(testDirAbs, collected)
  return collected
}

function pickPrimaryFamily(
  relatedTests: readonly string[],
  touchedFamilies: ReadonlySet<LanguageFamily>,
): LanguageFamily | undefined {
  const firstTest = relatedTests[0]
  if (firstTest !== undefined) {
    const lower = firstTest.toLowerCase()
    if (lower.endsWith('.go')) return 'go'
    if (lower.endsWith('.py')) return 'python'
    if (lower.endsWith('.rs')) return 'rust'
    return 'js'
  }
  for (const preferred of ['js', 'python', 'go', 'rust'] as const) {
    if (touchedFamilies.has(preferred)) return preferred
  }
  return undefined
}

function detectTestRunner(
  workspaceRoot: string,
  packageRoots: readonly string[],
  family: LanguageFamily,
  relatedTests: readonly string[],
): { readonly runner: TestRunnerKind; readonly runnerCommand: string } | undefined {
  const searchDirs = [...new Set([...packageRoots, workspaceRoot])]
  const argsSuffix = relatedTests.length > 0 ? ` ${relatedTests.join(' ')}` : ''

  if (family === 'js') {
    for (const dir of searchDirs) {
      if (hasAnyFile(dir, ['vitest.config.ts', 'vitest.config.mts', 'vitest.config.js', 'vitest.config.mjs'])) {
        return { runner: 'vitest', runnerCommand: `vitest run${argsSuffix}` }
      }
      const pkg = readPackageJson(dir)
      if (pkg) {
        const testScript = pkg.scripts?.test ?? ''
        if (/\bvitest\b/.test(testScript) || hasDep(pkg, 'vitest')) {
          return { runner: 'vitest', runnerCommand: `vitest run${argsSuffix}` }
        }
        if (/\bbun\s+test\b/.test(testScript) || hasAnyFile(dir, ['bun.lock', 'bun.lockb', 'bunfig.toml'])) {
          return { runner: 'bun', runnerCommand: `bun test${argsSuffix}` }
        }
        if (/\bjest\b/.test(testScript) || hasDep(pkg, 'jest') || hasAnyFile(dir, ['jest.config.js', 'jest.config.ts', 'jest.config.mjs', 'jest.config.cjs'])) {
          return { runner: 'jest', runnerCommand: `npx jest${argsSuffix}` }
        }
        if (hasDep(pkg, '@types/bun') || hasDep(pkg, 'bun-types')) {
          return { runner: 'bun', runnerCommand: `bun test${argsSuffix}` }
        }
      }
      if (hasAnyFile(dir, ['bun.lock', 'bun.lockb', 'bunfig.toml'])) {
        return { runner: 'bun', runnerCommand: `bun test${argsSuffix}` }
      }
    }
    return { runner: 'bun', runnerCommand: `bun test${argsSuffix}` }
  }

  if (family === 'python') {
    const hasPyMarker = searchDirs.some(dir =>
      hasAnyFile(dir, ['pyproject.toml', 'pytest.ini', 'setup.cfg', 'tox.ini', 'conftest.py', 'requirements.txt', 'setup.py']),
    )
    if (hasPyMarker || relatedTests.length > 0) {
      return { runner: 'pytest', runnerCommand: `pytest${argsSuffix}` }
    }
    return undefined
  }

  if (family === 'go') {
    const hasGoMod = searchDirs.some(dir => hasAnyFile(dir, ['go.mod', 'go.work']))
    if (hasGoMod || relatedTests.length > 0) {
      if (relatedTests.length === 0) {
        return { runner: 'go', runnerCommand: 'go test ./...' }
      }
      const pkgDirs = [...new Set(relatedTests.map(testPath => {
        const dir = dirname(testPath)
        return dir === '.' ? '.' : `./${dir}`
      }))]
      return { runner: 'go', runnerCommand: `go test ${pkgDirs.join(' ')}` }
    }
    return undefined
  }

  if (family === 'rust') {
    const hasCargo = searchDirs.some(dir => hasAnyFile(dir, ['Cargo.toml']))
    if (hasCargo || relatedTests.length > 0) {
      if (relatedTests.length === 1) {
        const single = relatedTests[0]!
        const segments = single.split('/')
        const testsIdx = segments.indexOf('tests')
        if (testsIdx >= 0 && testsIdx === segments.length - 2) {
          const targetName = basename(single, '.rs')
          return { runner: 'cargo', runnerCommand: `cargo test --test ${targetName}` }
        }
      }
      return { runner: 'cargo', runnerCommand: 'cargo test' }
    }
    return undefined
  }

  return undefined
}

interface ParsedPackageJson {
  readonly scripts?: Readonly<Record<string, string>>
  readonly dependencies?: Readonly<Record<string, string>>
  readonly devDependencies?: Readonly<Record<string, string>>
}

function readPackageJson(dir: string): ParsedPackageJson | undefined {
  const pkgPath = join(dir, 'package.json')
  try {
    if (!isRegularFileSync(pkgPath)) return undefined
    const raw = JSON.parse(readFileSync(pkgPath, 'utf8')) as unknown
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return undefined
    return raw as ParsedPackageJson
  } catch {
    return undefined
  }
}

function hasDep(pkg: ParsedPackageJson, name: string): boolean {
  return Boolean(pkg.devDependencies?.[name] ?? pkg.dependencies?.[name])
}

function hasAnyFile(dir: string, names: readonly string[]): boolean {
  return names.some(name => existsSync(join(dir, name)))
}

function findNearestPackageRoot(workspaceRoot: string, relPosix: string, family: LanguageFamily): string {
  const markers = family === 'js'
    ? ['package.json', 'tsconfig.json', 'vitest.config.ts', 'bunfig.toml']
    : family === 'python'
      ? ['pyproject.toml', 'pytest.ini', 'setup.py', 'setup.cfg']
      : family === 'go'
        ? ['go.mod']
        : ['Cargo.toml']

  let current = dirname(resolve(workspaceRoot, relPosix))
  while (isWithinPath(workspaceRoot, current)) {
    if (hasAnyFile(current, markers)) return current
    if (current === workspaceRoot) break
    const parent = dirname(current)
    if (parent === current) break
    current = parent
  }
  return workspaceRoot
}

function stripSourcePrefix(dir: string): string {
  if (!dir) return ''
  const segments = dir.split('/')
  if (segments[0] === 'src' || segments[0] === 'lib' || segments[0] === 'app') {
    return segments.slice(1).join('/')
  }
  return dir
}

function stripFirstSegment(dir: string): string {
  if (!dir) return ''
  const idx = dir.indexOf('/')
  return idx < 0 ? '' : dir.slice(idx + 1)
}

function joinPosix(...parts: readonly string[]): string {
  return parts.filter(Boolean).join('/')
}

function normalizeWorkspaceRelative(workspaceRoot: string, candidate: string): string | undefined {
  const lexical = isAbsolute(candidate) ? resolve(candidate) : resolve(workspaceRoot, candidate)
  if (!isWithinPath(workspaceRoot, lexical)) return undefined
  const rel = toPosix(relative(workspaceRoot, lexical))
  return !rel || rel === '.' || rel.startsWith('..') ? undefined : rel
}

function safeExistingRegularFileInside(workspaceRoot: string, candidateAbs: string): string | undefined {
  if (!isWithinPath(workspaceRoot, candidateAbs)) return undefined
  const canonical = safeRealpath(candidateAbs)
  if (!canonical || !isWithinPath(workspaceRoot, canonical)) return undefined
  return isRegularFileSync(canonical) ? canonical : undefined
}

function safeRealpath(target: string): string | undefined {
  try {
    return realpathSync.native(target)
  } catch {
    return undefined
  }
}

function isRegularFileSync(target: string): boolean {
  try {
    return statSync(target).isFile()
  } catch {
    return false
  }
}

function isDirectorySync(target: string): boolean {
  try {
    return statSync(target).isDirectory()
  } catch {
    return false
  }
}

function isWithinPath(root: string, candidate: string): boolean {
  const rel = relative(root, candidate)
  return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel))
}

function toPosix(value: string): string {
  return sep === '/' ? value : value.replaceAll('\\', '/')
}
