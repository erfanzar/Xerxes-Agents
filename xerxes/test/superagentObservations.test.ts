// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { WorkspaceObservations } from '../src/hub/supervisor/abilities/observations.js'

const roots: string[] = []
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }) })
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-observe-')); roots.push(root)
  await writeFile(join(root, 'a.txt'), 'first\nsecond needle\nthird\n')
  await mkdir(join(root, 'nested')); await writeFile(join(root, 'nested', 'b.txt'), 'needle again\n')
  return { root, observer: new WorkspaceObservations(root) }
}

test('file pages are byte-bounded and carry a continuation offset', async () => {
  const { root, observer } = await fixture()
  await writeFile(join(root, 'big.txt'), 'a'.repeat(70000))
  const first = await observer.read('big.txt', 0, 100)
  expect(first.text).toHaveLength(100); expect(first.next_offset).toBe(100)
  if (first.next_offset === null) throw new Error('Expected another page')
  expect((await observer.read('big.txt', first.next_offset, 100)).next_offset).toBe(200)
})

test('file reads refuse traversal, symlink escape, secrets and nonregular files', async () => {
  const { root, observer } = await fixture()
  const outside = await mkdtemp(join(tmpdir(), 'xerxes-outside-')); roots.push(outside)
  await writeFile(join(outside, 'secret'), 'do not disclose')
  await symlink(join(outside, 'secret'), join(root, 'escape'))
  for (const path of ['../outside', '/etc/passwd', 'escape', '.env', 'nested']) {
    await expect(observer.read(path)).rejects.toThrow()
  }
})

test('directory listing and literal search stay inside the workspace and page results', async () => {
  const { observer } = await fixture()
  expect((await observer.list('.')).entries).toContainEqual({ name: 'a.txt', kind: 'file' })
  const page = await observer.search('needle', 0, 1)
  expect(page.hits).toEqual([{ path: 'a.txt', line: 2, text: 'second needle' }])
  expect(page.next_offset).toBe(1)
  expect((await observer.search('needle', 1, 1)).hits).toEqual([{ path: 'nested/b.txt', line: 1, text: 'needle again' }])
})

test('binary files are refused and grep treats metacharacters literally', async () => {
  const { root, observer } = await fixture()
  await writeFile(join(root, 'binary'), Buffer.from([0, 1, 2]))
  await expect(observer.read('binary')).rejects.toThrow('binary')
  expect((await observer.search('.*')).hits).toEqual([])
})

test('GitHub and UI observations explicitly refuse without configured host ports', async () => {
  const { observer } = await fixture()
  await expect(observer.github('issues')).rejects.toThrow('GitHub')
  await expect(observer.screenshot(390)).rejects.toThrow('screenshot')
  await expect(observer.screenshot(999)).rejects.toThrow('width')
})

test('injected screenshot host receives only approved Xerxes widths', async () => {
  const { root } = await fixture()
  const widths: number[] = []
  const observer = new WorkspaceObservations(root, { screenshot: async width => { widths.push(width); return { artifact: 'xerxes-ui.png', width } } })
  expect(await observer.screenshot(1280)).toEqual({ artifact: 'xerxes-ui.png', width: 1280 })
  expect(widths).toEqual([1280])
})

test('Git observations stay in the workspace subtree and exclude credential diffs', async () => {
  const { root } = await fixture()
  await mkdir(join(root, 'project-a')); await mkdir(join(root, 'project-b'))
  await writeFile(join(root, 'project-a', 'safe.txt'), 'before')
  await writeFile(join(root, 'project-a', 'credentials.json'), 'old credential')
  await writeFile(join(root, 'project-b', 'hidden.txt'), 'before')
  for (const args of [['init', '-q', '--template='], ['add', '.']]) {
    const result = Bun.spawnSync(['git', ...args], { cwd: root, stdout: 'pipe', stderr: 'pipe' })
    expect(result.exitCode).toBe(0)
  }
  await writeFile(join(root, 'project-a', 'safe.txt'), 'after')
  await writeFile(join(root, 'project-a', 'credentials.json'), 'DO-NOT-DISCLOSE')
  await writeFile(join(root, 'project-b', 'hidden.txt'), 'sibling change')
  const observer = new WorkspaceObservations(join(root, 'project-a'))
  expect((await observer.git('status')).text).not.toContain('hidden.txt')
  const diff = (await observer.git('diff')).text
  expect(diff).toContain('safe.txt')
  expect(diff).not.toContain('DO-NOT-DISCLOSE')
})
