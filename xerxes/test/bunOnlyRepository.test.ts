// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  assertBunOnlyRepository,
  findLegacyPythonArtifacts,
} from '../scripts/assertBunOnly.js'

test('Bun-only repository check rejects legacy source, bytecode, and packaging while ignoring local environments', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-only-'))
  try {
    await mkdir(join(root, '.venv', 'lib'), { recursive: true })
    await mkdir(join(root, 'docs', '_bun'), { recursive: true })
    await mkdir(join(root, 'src'), { recursive: true })
    await writeFile(join(root, '.venv', 'lib', 'ignored.py'), 'pass\n')
    await writeFile(join(root, 'docs', '_bun', 'ignored.py'), 'pass\n')
    await writeFile(join(root, 'src', 'runtime.py'), 'pass\n')
    await writeFile(join(root, 'cache.pyc'), 'bytecode')
    await writeFile(join(root, 'pyproject.toml'), '[project]\n')

    await expect(findLegacyPythonArtifacts(root)).resolves.toEqual([
      { path: 'cache.pyc', reason: 'python-bytecode' },
      { path: 'pyproject.toml', reason: 'legacy-root-metadata' },
      { path: 'src/runtime.py', reason: 'python-source' },
    ])
    await expect(assertBunOnlyRepository(root)).rejects.toThrow('Legacy Python artifacts remain')

    await rm(join(root, 'src', 'runtime.py'))
    await rm(join(root, 'cache.pyc'))
    await rm(join(root, 'pyproject.toml'))
    await expect(assertBunOnlyRepository(root)).resolves.toBeUndefined()
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('in a git work tree, gitignored local clones are skipped but tracked and untracked Python is still refused', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-only-git-'))
  const git = (...args: string[]) => {
    const run = Bun.spawnSync(['git', '-C', root, ...args], { stdout: 'pipe', stderr: 'pipe' })
    if (run.exitCode !== 0) throw new Error(run.stderr.toString())
  }
  try {
    git('init', '-q')
    await mkdir(join(root, '.xerxes', 'review', 'clone'), { recursive: true })
    await mkdir(join(root, 'src'), { recursive: true })
    await mkdir(join(root, 'node_modules', 'pkg'), { recursive: true })
    await writeFile(join(root, '.gitignore'), '.xerxes/\n')
    await writeFile(join(root, '.xerxes', 'review', 'clone', 'tool.py'), 'pass\n')
    await writeFile(join(root, 'node_modules', 'pkg', 'helper.py'), 'pass\n')
    await writeFile(join(root, 'src', 'tracked.py'), 'pass\n')
    await writeFile(join(root, 'src', 'untracked.py'), 'pass\n')
    git('add', '.gitignore', 'src/tracked.py')

    // node_modules is not gitignored here, so git lists it; the built-in directory skip still applies.
    await expect(findLegacyPythonArtifacts(root)).resolves.toEqual([
      { path: 'src/tracked.py', reason: 'python-source' },
      { path: 'src/untracked.py', reason: 'python-source' },
    ])

    git('rm', '-q', '--cached', 'src/tracked.py')
    await rm(join(root, 'src'), { recursive: true })
    await expect(assertBunOnlyRepository(root)).resolves.toBeUndefined()
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})
