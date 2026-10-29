// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { realpathSync } from 'node:fs'
import { lstat, readdir } from 'node:fs/promises'
import { dirname, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const LEGACY_ROOT_FILES = new Set([
  '.pre-commit-config.yaml',
  '.python-version',
  'pyproject.toml',
  'requirements.txt',
  'setup.cfg',
  'setup.py',
  'uv.lock',
])

const IGNORED_DIRECTORIES = new Set([
  '.git',
  '.venv',
  'node_modules',
])

const IGNORED_RELATIVE_DIRECTORIES = new Set([
  'docs/_bun',
])

export interface LegacyPythonArtifact {
  readonly path: string
  readonly reason: 'legacy-root-metadata' | 'python-bytecode' | 'python-source'
}

/**
 * Find repository-owned Python artifacts that would reintroduce the retired
 * Xerxes Python runtime. Local environments and generated Bun docs are not
 * part of the source tree and are deliberately ignored.
 *
 * In a git work tree the candidates are what git would commit (tracked files
 * plus untracked files that are not ignored), so a gitignored local clone of
 * another project does not fail the check. Outside git the directory is walked.
 */
export async function findLegacyPythonArtifacts(
  repositoryDirectory: string,
): Promise<readonly LegacyPythonArtifact[]> {
  const root = resolve(repositoryDirectory)
  const findings: LegacyPythonArtifact[] = []
  const listed = gitCandidatePaths(root)
  if (listed) {
    for (const path of listed) {
      if (insideIgnoredDirectory(path)) continue
      const reason = legacyReason(path)
      if (reason) findings.push({ path, reason })
    }
  } else {
    await walk(root, root, findings)
  }
  return Object.freeze(findings.sort((left, right) => left.path.localeCompare(right.path)))
}

/** Paths git would commit, relative to `root`; undefined when `root` is not the top of a git work tree. */
function gitCandidatePaths(root: string): readonly string[] | undefined {
  const top = Bun.spawnSync(['git', '-C', root, 'rev-parse', '--show-toplevel'], { stdout: 'pipe', stderr: 'pipe' })
  // Compare real paths: on macOS a temp dir under /var is reported by git as /private/var.
  if (top.exitCode !== 0 || realpathSync(top.stdout.toString().trim()) !== realpathSync(root)) return undefined
  const listed = Bun.spawnSync(
    ['git', '-C', root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
    { stdout: 'pipe', stderr: 'pipe' },
  )
  if (listed.exitCode !== 0) {
    throw new Error(`git ls-files failed in ${root}: ${listed.stderr.toString().trim()}`)
  }
  return listed.stdout.toString().split('\0').filter(Boolean)
}

function insideIgnoredDirectory(path: string): boolean {
  const parts = path.split('/')
  for (let index = 0; index < parts.length - 1; index += 1) {
    if (IGNORED_DIRECTORIES.has(parts[index]!)) return true
    if (IGNORED_RELATIVE_DIRECTORIES.has(parts.slice(0, index + 1).join('/'))) return true
  }
  return false
}

/** Throw a readable error when a repository is not Bun-only yet. */
export async function assertBunOnlyRepository(
  repositoryDirectory: string,
): Promise<void> {
  const findings = await findLegacyPythonArtifacts(repositoryDirectory)
  if (!findings.length) return
  const lines = findings.map(finding => `- ${finding.path} (${finding.reason})`)
  throw new Error(`Legacy Python artifacts remain in the repository:\n${lines.join('\n')}`)
}

async function walk(
  root: string,
  directory: string,
  findings: LegacyPythonArtifact[],
): Promise<void> {
  const entries = await readdir(directory, { withFileTypes: true })
  for (const entry of entries) {
    const path = resolve(directory, entry.name)
    const relativePath = relative(root, path).replaceAll('\\', '/')
    if (entry.isDirectory()) {
      if (IGNORED_DIRECTORIES.has(entry.name) || IGNORED_RELATIVE_DIRECTORIES.has(relativePath)) {
        continue
      }
      await walk(root, path, findings)
      continue
    }
    if (entry.isSymbolicLink()) {
      const stats = await lstat(path)
      if (!stats.isFile()) continue
    }
    const reason = legacyReason(relativePath)
    if (reason) findings.push({ path: relativePath, reason })
  }
}

function legacyReason(path: string): LegacyPythonArtifact['reason'] | undefined {
  const name = path.split('/').at(-1) ?? path
  if (!path.includes('/') && LEGACY_ROOT_FILES.has(name)) return 'legacy-root-metadata'
  if (name.endsWith('.pyc')) return 'python-bytecode'
  if (name.endsWith('.py')) return 'python-source'
  return undefined
}

if (import.meta.main) {
  const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..')
  const repositoryDirectory = resolve(packageDirectory, '..')
  await assertBunOnlyRepository(repositoryDirectory)
  console.log('Bun-only repository check passed.')
}
