// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { createHash } from 'node:crypto'
import { mkdir, mkdtemp, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createRepoIdentityResolver, normalizeRepoRemote } from '../src/memory/repoIdentity.js'
import { GovernedMemoryStore } from '../src/memory/governedMemoryStore.js'
import { LessonStore } from '../src/memory/lessons.js'

const directories: string[] = []
afterEach(async () => { for (const path of directories.splice(0)) await rm(path, { recursive: true, force: true }) })
async function directory(): Promise<string> {
  const path = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-repo-identity-')))
  directories.push(path); return path
}
async function git(path: string, ...args: string[]): Promise<string> {
  const child = Bun.spawn(['git', '-C', path, ...args], { stdout: 'pipe', stderr: 'pipe', env: { ...Bun.env, GIT_TERMINAL_PROMPT: '0' } })
  const [code, text, error] = await Promise.all([child.exited, new Response(child.stdout).text(), new Response(child.stderr).text()])
  if (code) throw new Error(error)
  return text.trim()
}
const key = (text: string) => `git:${createHash('sha256').update(text).digest('hex').slice(0, 32)}`

test('origin normalization ignores transport, user, port, host case and git suffix but preserves path case', () => {
  for (const remote of ['git@GitHub.com:erfanzar/Xerxes-Agents.git', 'https://GitHub.com/erfanzar/Xerxes-Agents/', 'ssh://git@github.com:2222/erfanzar/Xerxes-Agents.git/', 'git://github.com:9418/erfanzar/Xerxes-Agents']) {
    expect(normalizeRepoRemote(remote)).toBe('github.com/erfanzar/Xerxes-Agents')
  }
  expect(normalizeRepoRemote('not a remote')).toBeUndefined()
})

test('real repositories fall back to the smallest root commit and never invent keys for empty/non-git folders', async () => {
  const root = await directory()
  const resolver = createRepoIdentityResolver()
  expect(await resolver(root)).toBeUndefined()
  await git(root, 'init', '--quiet')
  expect(await resolver(root)).toBeUndefined()
  // Use a fresh resolver after changing an initially unborn repository.
  await git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '--quiet', '--allow-empty', '-m', 'root')
  const sha = await git(root, 'rev-list', '--max-parents=0', 'HEAD')
  expect(await createRepoIdentityResolver()(root)).toBe(key(`root:${sha}`))
  const nested = join(root, 'nested'); await mkdir(nested)
  const resolved = createRepoIdentityResolver()
  expect(await resolved(nested)).toBe(await resolved(root))
})

test('multiple root histories choose the lexicographically smallest root; identity resolution is cached', async () => {
  const root = await directory(); await git(root, 'init', '--quiet')
  const author = ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid']
  await git(root, ...author, 'commit', '--quiet', '--allow-empty', '-m', 'first root')
  const second = await git(root, ...author, 'commit-tree', 'HEAD^{tree}', '-m', 'second root')
  await git(root, ...author, 'merge', '--quiet', '--allow-unrelated-histories', '--no-ff', '-m', 'join', second)
  const roots = (await git(root, 'rev-list', '--max-parents=0', 'HEAD')).split('\n').sort()
  expect(roots).toHaveLength(2)
  const resolver = createRepoIdentityResolver()
  expect(await resolver(root)).toBe(key(`root:${roots[0]}`))
  await git(root, 'remote', 'add', 'origin', 'https://github.com/fixture/new-origin')
  expect(await resolver(root)).toBe(key(`root:${roots[0]}`))
  expect(await createRepoIdentityResolver()(root)).toBe(key('github.com/fixture/new-origin'))
})

test('a missing git executable returns no identity', async () => {
  const root = await directory()
  const original = process.env.PATH
  try {
    process.env.PATH = root
    expect(await createRepoIdentityResolver()(root)).toBeUndefined()
  } finally { if (original === undefined) delete process.env.PATH; else process.env.PATH = original }
})

test('undefined repository answers retry after a 60-second backoff instead of being cached forever', async () => {
  const root = await directory(); let clock = 0
  const resolver = createRepoIdentityResolver({ now: () => clock })
  await git(root, 'init', '--quiet') // Both workspace and root caches receive an undefined answer.
  expect(await resolver(root)).toBeUndefined()
  await git(root, 'remote', 'add', 'origin', 'https://github.com/fixture/recovered')
  clock = 59_999
  expect(await resolver(root)).toBeUndefined()
  clock = 60_001
  expect(await resolver(root)).toBe(key('github.com/fixture/recovered'))
})

test('relative and local remote URLs fall back to root-commit identity, never a made-up host', async () => {
  const root = await directory(); await git(root, 'init', '--quiet')
  await git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '--quiet', '--allow-empty', '-m', 'root')
  const sha = await git(root, 'rev-list', '--max-parents=0', 'HEAD')
  await git(root, 'remote', 'add', 'origin', '../another-clone.git')
  for (const remote of ['../another-clone.git', 'local/another-clone.git', '/tmp/another-clone.git', 'file:///tmp/another-clone.git']) {
    expect(normalizeRepoRemote(remote)).toBeUndefined()
    await git(root, 'remote', 'set-url', 'origin', remote)
    expect(await createRepoIdentityResolver()(root)).toBe(key(`root:${sha}`))
  }
})

test('lessons are visible across two clone paths with equivalent origins; old path-only records stay local', async () => {
  const first = await directory(); const second = await directory(); const other = await directory()
  for (const path of [first, second, other]) await git(path, 'init', '--quiet')
  await git(first, 'remote', 'add', 'origin', 'git@github.com:owner/Project.git')
  await git(second, 'remote', 'add', 'origin', 'https://github.com/owner/Project')
  await git(other, 'remote', 'add', 'origin', 'https://github.com/owner/Other')
  const memory = new GovernedMemoryStore({ directory: join(first, 'memory') })
  const legacy = new LessonStore({ memory })
  const input = { workspace: first, cue: 'fixtures', lesson: 'Use temporary fixtures.', confidence: 0.9, agentId: 'fixture', evidence: { kind: 'test' as const, command: 'bun test', exitCode: 0, summary: 'Passed' } }
  const old = await legacy.record(input)
  const lessons = new LessonStore({ memory, resolveRepo: createRepoIdentityResolver() })
  const shared = await lessons.record(input)
  expect(shared.repo).toBe(key('github.com/owner/Project'))
  expect((await lessons.list(first)).map(row => row.id).sort()).toEqual([old.id, shared.id].sort())
  expect((await lessons.recall('fixtures', second)).map(row => row.id)).toEqual([shared.id])
  expect((await lessons.get(shared.id, second))?.id).toBe(shared.id)
  expect(await lessons.get(old.id, second)).toBeUndefined()
  expect(await lessons.list(other)).toEqual([])
})
