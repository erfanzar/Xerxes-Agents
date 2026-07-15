// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import {
  InjectedRemoteSkillSource,
  LocalSkillSource,
  OfficialSkillSource,
  SkillsHub,
} from '../src/extensions/skillsHub.js'

async function inTemporaryDirectory(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-skills-hub-'))
  try {
    await run(directory)
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
}

test('local and official sources read safe local SKILL.md files and search only active local skills', async () => {
  await inTemporaryDirectory(async directory => {
    const localSkill = join(directory, 'local-source')
    const installed = join(directory, 'installed')
    const official = join(directory, 'official')
    await mkdir(localSkill)
    await mkdir(join(installed, 'searchable'), { recursive: true })
    await mkdir(join(installed, '.hub', 'quarantine', 'hidden'), { recursive: true })
    await mkdir(join(official, 'bundled'), { recursive: true })
    await writeFile(join(localSkill, 'SKILL.md'), '---\nname: local-name\n---\nDo local work.', 'utf8')
    await writeFile(
      join(installed, 'searchable', 'SKILL.md'),
      '---\nname: searchable\n---\nFind the useful needle.',
      'utf8',
    )
    await writeFile(join(installed, '.hub', 'quarantine', 'hidden', 'SKILL.md'), 'secret needle', 'utf8')
    await writeFile(join(official, 'bundled', 'SKILL.md'), '# Bundled\nOfficial workflow.', 'utf8')

    const local = new LocalSkillSource({ skillsDirectory: installed })
    const bundle = await local.fetch(localSkill)
    expect(bundle.name).toBe('local-name')
    expect(bundle.content).toContain('Do local work.')
    const localMatches = await local.search('needle')
    expect(localMatches).toHaveLength(1)
    expect(localMatches[0]).toMatchObject({ name: 'searchable', source: 'local' })

    const officialSource = new OfficialSkillSource({ directory: official })
    expect((await officialSource.fetch('bundled')).metadata).toEqual({ official: true })
    const officialMatches = await officialSource.search('workflow')
    expect(officialMatches).toHaveLength(1)
    expect(officialMatches[0]).toMatchObject({ name: 'bundled', source: 'official' })
    await expect(officialSource.fetch('../outside')).rejects.toThrow('plain directory name')
  })
})

test('skills hub installs, force-replaces, records lock and audit state, and uninstalls safely', async () => {
  await inTemporaryDirectory(async directory => {
    const skillsDirectory = join(directory, 'skills')
    const source = join(directory, 'source')
    const installedAt = new Date(2026, 4, 8, 15, 4, 5).valueOf() / 1000
    await mkdir(source)
    await writeFile(join(source, 'SKILL.md'), '---\nname: deploy\n---\nVersion one.', 'utf8')
    const hub = new SkillsHub({
      skillsDirectory,
      now: () => new Date(installedAt * 1000),
    })

    expect(await hub.install(`local:${source}`)).toBe(`Installed skill 'deploy' from local:${source}`)
    expect(await hub.install(`local:${source}`)).toBe(
      "[Error] Skill 'deploy' already installed. Use force=true to overwrite.",
    )
    expect(await readFile(join(skillsDirectory, 'deploy', 'SKILL.md'), 'utf8')).toContain('Version one.')
    expect(await hub.listInstalled()).toEqual([
      expect.objectContaining({ name: 'deploy', source: 'local', identifier: source, installedAt }),
    ])
    expect(JSON.parse(await readFile(join(skillsDirectory, '.hub', 'lock.json'), 'utf8'))).toMatchObject({
      deploy: { source: 'local', identifier: source, installedAt },
    })
    expect(await readFile(join(skillsDirectory, '.hub', 'audit.log'), 'utf8')).toContain('install')

    await writeFile(join(source, 'SKILL.md'), '---\nname: deploy\n---\nVersion two.', 'utf8')
    expect(await hub.install(`local:${source}`, { force: true })).toBe(`Installed skill 'deploy' from local:${source}`)
    expect(await readFile(join(skillsDirectory, 'deploy', 'SKILL.md'), 'utf8')).toContain('Version two.')

    expect(await hub.uninstall('deploy')).toBe("Uninstalled skill 'deploy'")
    expect(await hub.listInstalled()).toEqual([])
    expect(await readFile(join(skillsDirectory, '.hub', 'audit.log'), 'utf8')).toContain('uninstall')
    expect(await hub.uninstall('deploy')).toBe("[Error] Skill 'deploy' is not installed.")
  })
})

test('injected remote sources are the only remote boundary and participate in install and search', async () => {
  await inTemporaryDirectory(async directory => {
    const skillsDirectory = join(directory, 'skills')
    const calls: string[] = []
    const remote = new InjectedRemoteSkillSource('catalog', {
      async fetch(identifier) {
        calls.push(`fetch:${identifier}`)
        return {
          name: 'remote-skill',
          content: '# Remote\nConfigured transport only.',
          metadata: { repo: 'trusted/catalog' },
        }
      },
      async search(query, limit) {
        calls.push(`search:${query}:${limit}`)
        return [{ name: 'remote-skill', identifier: 'remote-id', source: 'ignored-by-hub' }]
      },
    })
    const hub = new SkillsHub({ skillsDirectory, sources: [remote] })

    expect(await hub.install('catalog:remote-id')).toBe("Installed skill 'remote-skill' from catalog:remote-id")
    expect(await hub.search('remote', 3)).toEqual([
      { name: 'remote-skill', identifier: 'remote-id', source: 'catalog' },
    ])
    expect(calls).toEqual(['fetch:remote-id', 'search:remote:3'])
    expect(hub.getSource('github')).toBeUndefined()
    expect(await hub.install('github:owner/repo/path')).toBe('[Error] Unknown source: github')
  })
})

test('hub refuses symlinked destinations and delegates contained quarantine approval to skills guard', async () => {
  await inTemporaryDirectory(async directory => {
    const skillsDirectory = join(directory, 'skills')
    const source = join(directory, 'source')
    const outside = join(directory, 'outside')
    await mkdir(source)
    await mkdir(outside)
    await writeFile(join(source, 'SKILL.md'), '---\nname: unsafe\n---\nNo overwrite.', 'utf8')
    await writeFile(join(outside, 'SKILL.md'), 'outside stays intact', 'utf8')
    await mkdir(skillsDirectory)
    await symlink(outside, join(skillsDirectory, 'unsafe'))

    const hub = new SkillsHub({ skillsDirectory })
    const failed = await hub.install(`local:${source}`, { force: true })
    expect(failed).toContain('[Error] Failed to install')
    expect(await readFile(join(outside, 'SKILL.md'), 'utf8')).toBe('outside stays intact')
    expect(await hub.uninstall('unsafe')).toContain('[Error] Failed to uninstall')

    const pending = join(skillsDirectory, '.hub', 'quarantine', 'approved')
    await mkdir(pending, { recursive: true })
    await writeFile(join(pending, 'SKILL.md'), '# Approved', 'utf8')
    expect(await hub.approveSkill('approved')).toBe("Approved and activated skill 'approved'")
    expect(await readFile(join(skillsDirectory, 'approved', 'SKILL.md'), 'utf8')).toBe('# Approved')
    expect(await readFile(join(skillsDirectory, '.hub', 'audit.log'), 'utf8')).toContain('approve')
  })
})
