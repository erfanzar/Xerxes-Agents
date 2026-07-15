// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, realpath, rm, stat, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { tmpdir } from 'node:os'

import {
  SkillSyncError,
  installSkillBundle,
  syncSkillManifest,
  type SkillSyncBundle,
} from '../src/extensions/skillsSync.js'

test('skill bundle installation writes only a contained SKILL.md child', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-skills-sync-'))
  const escapedName = `escape-${crypto.randomUUID()}`
  try {
    const output = await installSkillBundle(bundle('review', '# review instructions'), root)
    expect(output).toBe(join(await realpath(root), 'review', 'SKILL.md'))
    expect(await readFile(output, 'utf8')).toBe('# review instructions')

    await expect(installSkillBundle(bundle(`../${escapedName}`, 'outside'), root)).rejects.toBeInstanceOf(SkillSyncError)
    await expect(installSkillBundle(bundle('.hub', 'guard state'), root)).rejects.toBeInstanceOf(SkillSyncError)
    expect(await pathExists(join(dirname(root), escapedName, 'SKILL.md'))).toBeFalse()
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test(
  'skill manifest sync is deterministic, preserves installed entries, and prunes only stray directories',
  async () => {
    const root = await mkdtemp(join(tmpdir(), 'xerxes-skills-sync-'))
    const calls: string[] = []
    try {
      await installSkillBundle(bundle('alpha', 'already installed'), root)
      await installSkillBundle(bundle('zeta', 'stray'), root)
      await installSkillBundle(bundle('gamma', 'stray'), root)
      await mkdir(join(root, '.hub'), { recursive: true })
      await writeFile(join(root, '.hub', 'trusted_hashes.json'), '{}', 'utf8')
      await writeFile(join(root, 'notes.txt'), 'not a skill directory', 'utf8')

      const result = await syncSkillManifest([
        { identifier: 'beta', source: 'catalog' },
        { identifier: 'missing', source: 'unknown' },
        { identifier: 'alpha', source: 'catalog' },
        { identifier: 'beta', source: 'catalog' },
      ], {
        catalog: {
          fetch(identifier): SkillSyncBundle {
            calls.push(identifier)
            return bundle(identifier, `instructions for ${identifier}`)
          },
        },
      }, { prune: true, targetDirectory: root })

      expect(calls).toEqual(['beta'])
      expect(result).toEqual({
        failed: [{ identifier: 'missing', reason: 'unknown source: unknown' }],
        installed: ['beta'],
        removed: ['gamma', 'zeta'],
        skipped: ['alpha'],
      })
      expect(await readFile(join(root, 'beta', 'SKILL.md'), 'utf8')).toBe('instructions for beta')
      expect(await pathExists(join(root, 'gamma'))).toBeFalse()
      expect(await pathExists(join(root, 'zeta'))).toBeFalse()
      expect(await readFile(join(root, '.hub', 'trusted_hashes.json'), 'utf8')).toBe('{}')
      expect(await readFile(join(root, 'notes.txt'), 'utf8')).toBe('not a skill directory')
    } finally {
      await rm(root, { force: true, recursive: true })
    }
  },
)

test(
  'invalid, conflicting, and mismatched manifest entries fail without escaping or pruning requested skills',
  async () => {
    const root = await mkdtemp(join(tmpdir(), 'xerxes-skills-sync-'))
    const escapedName = `escape-${crypto.randomUUID()}`
    const calls: string[] = []
    try {
      await installSkillBundle(bundle('wanted', 'preserve this skill'), root)
      const result = await syncSkillManifest([
        { identifier: 'wanted', source: 'source-z' },
        { identifier: `../${escapedName}`, source: 'catalog' },
        { identifier: 'mismatch', source: 'catalog' },
        { identifier: 'wanted', source: 'source-a' },
      ], {
        catalog: {
          fetch(identifier): SkillSyncBundle {
            calls.push(identifier)
            return bundle('returned-name', 'unexpected name')
          },
        },
      }, { prune: true, targetDirectory: root })

      expect(calls).toEqual(['mismatch'])
      expect(result.installed).toEqual([])
      expect(result.skipped).toEqual([])
      expect(result.removed).toEqual([])
      expect(result.failed.map(failure => failure.identifier)).toEqual([`../${escapedName}`, 'mismatch', 'wanted'])
      expect(result.failed[0]?.reason).toContain('single contained directory name')
      expect(result.failed[1]?.reason).toContain('returned bundle name')
      expect(result.failed[2]?.reason).toBe('manifest identifier is declared by multiple sources: source-a, source-z')
      expect(await readFile(join(root, 'wanted', 'SKILL.md'), 'utf8')).toBe('preserve this skill')
      expect(await pathExists(join(root, 'returned-name'))).toBeFalse()
      expect(await pathExists(join(dirname(root), escapedName, 'SKILL.md'))).toBeFalse()
    } finally {
      await rm(root, { force: true, recursive: true })
    }
  },
)

function bundle(name: string, bodyMarkdown: string): SkillSyncBundle {
  return { bodyMarkdown, name, version: '1.0.0' }
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path)
    return true
  } catch (error) {
    if (typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT') {
      return false
    }
    throw error
  }
}
