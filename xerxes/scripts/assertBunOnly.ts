// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

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
 */
export async function findLegacyPythonArtifacts(
  repositoryDirectory: string,
): Promise<readonly LegacyPythonArtifact[]> {
  const root = resolve(repositoryDirectory)
  const findings: LegacyPythonArtifact[] = []
  await walk(root, root, findings)
  return Object.freeze(findings.sort((left, right) => left.path.localeCompare(right.path)))
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
