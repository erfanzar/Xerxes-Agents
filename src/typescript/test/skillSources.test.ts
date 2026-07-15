// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import {
  AgentskillsIOSource,
  GitHubSkillSource,
  LocalSkillSource,
  OfficialSkillSource,
  SkillRegistryHttpError,
  SkillSourceConfigurationError,
  SkillSourceError,
  SkillSourceNotFoundError,
  type SkillRegistryFetchResponse,
} from '../src/extensions/skillSources/index.js'

test('local skill source finds nested markdown, extracts a version, and rejects unsafe identifiers', async () => {
  await inTemporaryDirectory(async root => {
    await mkdir(join(root, 'nested', 'alpha'), { recursive: true })
    await mkdir(join(root, 'beta'), { recursive: true })
    await writeFile(
      join(root, 'nested', 'alpha', 'SKILL.md'),
      '---\nversion: "1.2.3"\n---\n\n# Alpha\nAlpha helps with reviews.\n',
      'utf8',
    )
    await writeFile(join(root, 'beta', 'SKILL.md'), '# Beta\nBeta does useful things.\n', 'utf8')

    const source = new LocalSkillSource({ root })
    expect(await source.search('alpha')).toEqual([
      expect.objectContaining({ description: 'Alpha helps with reviews.', name: 'alpha', version: '1.2.3' }),
    ])
    expect(await source.search('does useful')).toEqual([expect.objectContaining({ name: 'beta' })])
    expect(await source.search('anything', 0)).toEqual([])
    await expect(source.fetch('alpha')).resolves.toMatchObject({
      bodyMarkdown: expect.stringContaining('Alpha helps'),
      metadata: {},
      name: 'alpha',
      sourceName: 'local',
      version: '1.2.3',
    })
    await expect(source.fetch('missing')).rejects.toBeInstanceOf(SkillSourceNotFoundError)
    await expect(source.fetch('../outside')).rejects.toBeInstanceOf(SkillSourceError)
  })
})

test('GitHub source delegates only through explicit host ports', async () => {
  const calls: unknown[] = []
  const source = new GitHubSkillSource({
    repository: 'owner/catalogue',
    ports: {
      async fetchSkillMarkdown(request) {
        calls.push(request)
        return '# Native only\n'
      },
      async searchSkills(request) {
        calls.push(request)
        return [{ name: 'review', description: 'Review a patch', version: '7', tags: ['code'] }]
      },
    },
  })

  await expect(source.fetch('review')).resolves.toEqual({
    bodyMarkdown: '# Native only\n',
    metadata: { repository: 'owner/catalogue' },
    name: 'review',
    sourceName: 'github',
    version: 'github',
  })
  await expect(source.search('review', 3)).resolves.toEqual([
    { description: 'Review a patch', name: 'review', sourceName: 'github', tags: ['code'], version: '7' },
  ])
  expect(calls).toEqual([
    { identifier: 'review', repository: 'owner/catalogue' },
    { limit: 3, query: 'review', repository: 'owner/catalogue' },
  ])

  const unconfigured = new GitHubSkillSource()
  await expect(unconfigured.fetch('review')).rejects.toBeInstanceOf(SkillSourceConfigurationError)
  await expect(unconfigured.search('review')).resolves.toEqual([])
})

test(
  'official source builds registry requests through an injected fetch transport without a global fallback',
  async () => {
    const requests: string[] = []
    const source = new OfficialSkillSource({
      baseUrl: 'https://example.test/catalog',
      transport: {
        async fetch(url) {
          requests.push(url)
          if (url.includes('/search')) {
            return response(200, JSON.stringify([{ name: 'review', description: 'Review code', tags: ['code'] }]))
          }
          return response(200, '# Review\n')
        },
      },
    })

    await expect(source.search('review code', 2)).resolves.toEqual([
      { description: 'Review code', name: 'review', sourceName: 'official', tags: ['code'], version: '' },
    ])
    await expect(source.fetch('review')).resolves.toEqual({
      bodyMarkdown: '# Review\n',
      metadata: {},
      name: 'review',
      sourceName: 'official',
      version: 'official',
    })

    const searchUrl = new URL(requests[0] ?? '')
    expect(searchUrl.pathname).toBe('/catalog/search')
    expect(searchUrl.searchParams.get('q')).toBe('review code')
    expect(searchUrl.searchParams.get('limit')).toBe('2')
    expect(new URL(requests[1] ?? '').pathname).toBe('/catalog/skills/review/SKILL.md')
  },
)

test('official source makes fetch failures explicit but treats failed searches as no results', async () => {
  const source = new OfficialSkillSource({
    transport: {
      async fetch(url) {
        if (url.includes('/search')) throw new Error('offline')
        return response(503, 'unavailable')
      },
    },
  })

  await expect(source.search('review')).resolves.toEqual([])
  await expect(source.fetch('review')).rejects.toBeInstanceOf(SkillRegistryHttpError)
})

test('agentskills.io preset preserves its registry path while keeping transport injection explicit', async () => {
  const requests: string[] = []
  const source = new AgentskillsIOSource({
    transport: {
      async fetch(url) {
        requests.push(url)
        return response(200, '[]')
      },
    },
  })

  expect(source.name).toBe('agentskills.io')
  await expect(source.search('native')).resolves.toEqual([])
  expect(new URL(requests[0] ?? '').pathname).toBe('/api/v1/search')
})

function response(status: number, body: string): SkillRegistryFetchResponse {
  return {
    ok: status >= 200 && status < 300,
    status,
    async text(): Promise<string> {
      return body
    },
  }
}

async function inTemporaryDirectory(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-skill-sources-'))
  try {
    await run(directory)
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
}
