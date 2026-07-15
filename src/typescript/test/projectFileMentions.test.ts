// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { dirname, join, resolve } from 'node:path'
import { mkdir, mkdtemp, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'

import {
  PROJECT_FILE_INDEX_CACHE_TTL_MS,
  PROJECT_FILE_INDEX_LIMIT,
  PROJECT_FILE_MENTION_LIMIT,
  ProjectFileMentionIndexCache,
  resolveProjectFileMentionRoot,
} from '../src/daemon/projectFileMentions.js'

const BINARY_EXTENSIONS = [
  '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.bmp', '.tiff',
  '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.flac', '.ogg',
  '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar', '.xz',
  '.woff', '.woff2', '.ttf', '.eot', '.otf',
  '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
  '.exe', '.dll', '.so', '.dylib', '.o', '.a', '.pyc', '.class', '.wasm',
] as const

test('git project mentions search tracked and untracked files from the repository root', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-file-mentions-git-'))
  try {
    await runGit(root, ['init', '--quiet'])
    await writeFixture(root, '.gitignore', '.private/\nignored/\n*.log\n')
    await writeFixture(root, 'notes/alpha', 'exact')
    await writeFixture(root, 'src/alpha.ts', 'prefix')
    await writeFixture(root, 'alpha-notes.md', 'prefix')
    await writeFixture(root, 'docs/find-alpha-guide.md', 'contains')
    await writeFixture(root, 'alpha-area/zeta.ts', 'path prefix')
    await writeFixture(root, 'nested/use-alpha/config.ts', 'path contains')
    await writeFixture(root, 'tracked/omega.txt', 'tracked')
    await writeFixture(root, 'tracked/deleted-alpha.ts', 'staged then deleted')
    await writeFixture(root, '.private/alpha-secret.ts', 'ignored hidden directory')
    await writeFixture(root, 'ignored/alpha-secret.ts', 'ignored directory')
    await writeFixture(root, 'debug-alpha.log', 'ignored extension')
    await Promise.all(BINARY_EXTENSIONS.map(async (extension, index) => {
      const normalized = index === 0 ? extension.toUpperCase() : extension
      await writeFixture(root, `binary/alpha-${index}${normalized}`, 'binary extension')
    }))
    await mkdir(join(root, 'alpha-empty'))
    await mkdir(join(root, 'packages', 'app'), { recursive: true })
    await runGit(root, ['add', 'tracked/omega.txt', 'tracked/deleted-alpha.ts'])
    await rm(join(root, 'tracked', 'deleted-alpha.ts'))

    const index = new ProjectFileMentionIndexCache()
    const result = await index.search(join(root, 'packages', 'app'), '@ALPHA')
    const canonicalRoot = await realpath(root)

    expect(result.rootDirectory).toBe(canonicalRoot)
    expect(await resolveProjectFileMentionRoot(join(root, 'packages', 'app'))).toBe(canonicalRoot)
    expect(result.source).toBe('git')
    expect(result.matches.map(match => match.relativePath)).toEqual([
      'notes/alpha',
      'src/alpha.ts',
      'alpha-notes.md',
      'docs/find-alpha-guide.md',
      'alpha-area/zeta.ts',
      'nested/use-alpha/config.ts',
    ])
    expect(result.matches[0]).toEqual({
      absolutePath: resolve(canonicalRoot, 'notes', 'alpha'),
      basename: 'alpha',
      relativePath: 'notes/alpha',
    })
    expect(result.matches.every(match => !match.relativePath.includes('\\'))).toBe(true)
    expect(result.matches.some(match => match.relativePath.includes('ignored'))).toBe(false)
    expect(result.matches.some(match => match.relativePath.startsWith('binary/'))).toBe(false)
    expect(result.matches.some(match => match.relativePath === 'alpha-empty')).toBe(false)

    const tracked = await index.search(join(root, 'packages', 'app'), 'omega')
    expect(tracked.matches.map(match => match.relativePath)).toEqual(['tracked/omega.txt'])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('filesystem fallback skips dependency and hidden directories, returns files only, and caches indexes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-file-mentions-fs-'))
  try {
    await writeFixture(root, 'target-root.txt', 'root')
    await writeFixture(root, 'visible/target-nested.txt', 'nested')
    await writeFixture(root, 'visible/target-folder/file.txt', 'path match')
    await writeFixture(root, '.cache/target-hidden.ts', 'hidden')
    await writeFixture(root, '.git/target-metadata', 'git metadata')
    await writeFixture(root, 'node_modules/pkg/target-dependency.ts', 'dependency')
    await Promise.all(BINARY_EXTENSIONS.map(async (extension, index) => {
      await writeFixture(root, `visible/binary/target-${index}${extension}`, 'binary extension')
    }))
    await mkdir(join(root, 'visible', 'target-directory'))

    const index = new ProjectFileMentionIndexCache({ cacheTtlMs: 60_000 })
    const result = await index.search(root, 'TARGET')
    expect(result).toMatchObject({ rootDirectory: resolve(root), source: 'filesystem' })
    expect(await resolveProjectFileMentionRoot(root)).toBe(resolve(root))
    expect(result.matches.map(match => match.relativePath)).toEqual([
      'target-root.txt',
      'visible/target-nested.txt',
      'visible/target-folder/file.txt',
    ])
    expect(result.matches.some(match => match.relativePath === 'visible/target-directory')).toBe(false)
    expect(result.matches.every(match => !match.relativePath.startsWith('.'))).toBe(true)
    expect(result.matches.every(match => !match.relativePath.startsWith('node_modules/'))).toBe(true)
    expect(result.matches.some(match => match.relativePath.includes('/binary/'))).toBe(false)

    await writeFixture(root, 'fresh-target.ts', 'created after indexing')
    expect((await index.search(root, 'fresh')).matches).toEqual([])
    index.invalidate(root)
    expect((await index.search(root, 'fresh')).matches.map(match => match.relativePath)).toEqual([
      'fresh-target.ts',
    ])

    const limited = new ProjectFileMentionIndexCache({ maxFiles: 2 })
    expect((await limited.search(root, '')).matches).toHaveLength(2)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('project mention results use fixed production ceilings and return only the top 50', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-file-mentions-limit-'))
  try {
    await Promise.all(
      Array.from({ length: 55 }, async (_, index) => {
        await writeFixture(root, `needle-${String(index).padStart(2, '0')}.txt`, 'match')
      }),
    )

    const result = await new ProjectFileMentionIndexCache().search(root, 'needle')
    expect(PROJECT_FILE_INDEX_LIMIT).toBe(5_000)
    expect(PROJECT_FILE_MENTION_LIMIT).toBe(50)
    expect(PROJECT_FILE_INDEX_CACHE_TTL_MS).toBe(30_000)
    expect(result.matches).toHaveLength(50)
    expect(result.matches[0]?.relativePath).toBe('needle-00.txt')
    expect(result.matches.at(-1)?.relativePath).toBe('needle-49.txt')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

async function writeFixture(root: string, relativePath: string, content: string): Promise<void> {
  const absolutePath = join(root, ...relativePath.split('/'))
  await mkdir(dirname(absolutePath), { recursive: true })
  await Bun.write(absolutePath, content)
}

async function runGit(cwd: string, args: readonly string[]): Promise<void> {
  const process = Bun.spawn(['git', ...args], {
    cwd,
    stdin: 'ignore',
    stdout: 'pipe',
    stderr: 'pipe',
  })
  const [exitCode, stdout, stderr] = await Promise.all([
    process.exited,
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
  ])
  if (exitCode !== 0) {
    throw new Error(`git ${args.join(' ')} failed (${exitCode}): ${stderr || stdout}`)
  }
}
