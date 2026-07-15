// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import {
  buildRepoMap,
  extractRepoMapSymbols,
  RepoMapConfig,
  RepoMapper,
} from '../src/context/repoMap.js'

test('repo mapper scans supplied workspaces deterministically and honors ignores', async () => {
  await inTemporaryDirectory(async root => {
    await writeFile(join(root, 'app.py'), [
      'MAX_RETRIES = 3',
      '',
      'def public_function(left, right):',
      '    return left + right',
      '',
      'async def async_fetch(url: str) -> str:',
      '    return url',
      '',
      'def _private_helper():',
      '    return None',
      '',
      'class Processor:',
      '    def run(self, value):',
      '        return value',
      '',
      '    def _hidden(self):',
      '        return None',
    ].join('\n'))
    await writeFile(join(root, 'client.ts'), [
      'export function fetchData(url: string) { return url }',
      'export const handler = async (value: string) => value',
      'export class ApiClient {}',
      'export interface ClientOptions { retries: number }',
    ].join('\n'))
    await mkdir(join(root, 'subdir'))
    await writeFile(join(root, 'subdir', 'utils.py'), 'def helper_func():\n    return 42\n')
    await mkdir(join(root, 'ignored_dir'))
    await writeFile(join(root, 'ignored_dir', 'secret.py'), 'SECRET = "hidden"\n')
    await mkdir(join(root, 'node_modules', 'pkg'), { recursive: true })
    await writeFile(join(root, 'node_modules', 'pkg', 'index.ts'), 'export const DEPENDENCY = true\n')
    await writeFile(join(root, '.gitignore'), 'ignored_dir/\n*.lock\n')
    await writeFile(join(root, 'generated.lock'), 'not source')

    const mapper = new RepoMapper(new RepoMapConfig({ tokenBudget: 2_000 }))
    const first = await mapper.build(root)
    const second = await mapper.build(root)
    expect(second).toEqual(first)
    expect(first.fileCount).toBe(3)
    expect(first.text).toContain('MAX_RETRIES')
    expect(first.text).toContain('public_function')
    expect(first.text).toContain('async_fetch')
    expect(first.text).toContain('Processor.run')
    expect(first.text).toContain('fetchData')
    expect(first.text).toContain('ApiClient')
    expect(first.text).not.toContain('_private_helper')
    expect(first.text).not.toContain('_hidden')
    expect(first.text).not.toContain('ignored_dir')
    expect(first.text).not.toContain('DEPENDENCY')
    expect(await buildRepoMap(root)).toBe(first.text)
  })
})

test('repo mapper ranks cross-file references, respects its budget, and labels multiple workspace roots', async () => {
  await inTemporaryDirectory(async temporary => {
    const firstRoot = join(temporary, 'first-workspace')
    const secondRoot = join(temporary, 'second-workspace')
    await mkdir(firstRoot)
    await mkdir(secondRoot)
    await writeFile(join(firstRoot, 'core.ts'), [
      'export function popularFunc() {}',
      'export function obscureFunc() {}',
    ].join('\n'))
    await writeFile(join(firstRoot, 'a.ts'), 'import { popularFunc } from "./core"\npopularFunc()\n')
    await writeFile(join(firstRoot, 'b.ts'), 'import { popularFunc } from "./core"\npopularFunc()\n')
    await writeFile(join(secondRoot, 'other.ts'), 'export function otherWorkspaceSymbol() {}\n')

    const large = await new RepoMapper({ tokenBudget: 2_000 }).build(firstRoot)
    const smallBudget = 20
    const small = await new RepoMapper({ tokenBudget: smallBudget }).build(firstRoot)
    expect(large.text.indexOf('popularFunc')).toBeLessThan(large.text.indexOf('obscureFunc'))
    expect(small.estimatedTokens).toBeLessThanOrEqual(smallBudget)
    expect(small.includedCount).toBeLessThanOrEqual(large.includedCount)

    const multiple = await new RepoMapper().build([secondRoot, firstRoot])
    expect(multiple.text).toContain('first-workspace/core.ts')
    expect(multiple.text).toContain('second-workspace/other.ts')
  })
})

test('repo-map declaration scanning is deliberately scoped and invalid roots return an empty result', async () => {
  const symbols = extractRepoMapSymbols([
    'export type Config = { enabled: boolean }',
    'export const run = (value: string) => value',
    'class Service {}',
  ].join('\n'), 'sample.ts')
  expect(symbols).toEqual(expect.arrayContaining([
    expect.objectContaining({ kind: 'type', name: 'Config' }),
    expect.objectContaining({ kind: 'const_arrow', name: 'run' }),
    expect.objectContaining({ kind: 'class', name: 'Service' }),
  ]))

  const result = await new RepoMapper().build(join(tmpdir(), 'xerxes-no-such-workspace'))
  expect(result).toEqual({ text: '', fileCount: 0, symbolCount: 0, includedCount: 0, estimatedTokens: 0 })
})

async function inTemporaryDirectory(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-repo-map-'))
  try {
    await run(directory)
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
}
