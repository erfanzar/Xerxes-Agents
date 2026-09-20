// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { collectGitDiff } from '../src/workspace/gitDiff.js'

test('untracked files are navigable diff sections including empty files and safe symlinks', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xr-diff-'))
  try {
    await Bun.spawn(['git', 'init', '-q', dir]).exited
    await Bun.write(join(dir, 'new file.ts'), 'export const answer = 42\n')
    await Bun.write(join(dir, 'empty.txt'), '')
    await symlink('/etc/passwd', join(dir, 'external-link'))
    const result = await collectGitDiff({ cwd: dir, includeUntracked: true })
    expect(result.kind).toBe('ok')
    if (result.kind !== 'ok') throw new Error('missing diff')
    expect(result.diff.lines.filter(line => line.kind === 'file').map(line => line.text)).toEqual(expect.arrayContaining(['new file.ts', 'empty.txt', 'external-link']))
    expect(result.diff.lines.some(line => line.text === '+export const answer = 42')).toBe(true)
    expect(result.diff.lines.some(line => line.text === 'Empty untracked file')).toBe(true)
    expect(result.diff.lines.map(line => line.text).join('\n')).not.toContain('root:')
    const bounded = await collectGitDiff({ cwd: dir, includeUntracked: true, maxLines: 4 })
    expect(bounded.kind === 'ok' && bounded.diff.truncated).toBe(true)
  } finally { await rm(dir, { recursive: true, force: true }) }
})

test('a selected untracked file has its own budget after the overview was truncated', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xr-large-diff-'))
  try {
    await Bun.spawn(['git', 'init', '-q', dir]).exited
    await Bun.write(join(dir, 'tracked.ts'), 'initial\n')
    await Bun.spawn(['git', '-C', dir, 'add', 'tracked.ts']).exited
    await Bun.spawn(['git', '-C', dir, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test', 'commit', '-qm', 'fixture']).exited
    await Bun.write(join(dir, 'tracked.ts'), 'changed\n'.repeat(5000))
    await Bun.write(join(dir, 'new 空 file.ts'), 'export const visible = true\n')
    const overview = await collectGitDiff({ cwd: dir, includeUntracked: true })
    expect(overview.kind === 'ok' && overview.diff.truncated).toBe(true)
    const selected = await collectGitDiff({ cwd: dir, includeUntracked: true, path: 'new 空 file.ts' })
    expect(selected.kind === 'ok' && selected.diff.lines).toEqual(expect.arrayContaining([
      { kind: 'file', text: 'new 空 file.ts' }, { kind: 'add', text: '+export const visible = true', newLine: 1 },
    ]))
    expect(selected.kind === 'ok' && selected.diff.untracked).toEqual(['new 空 file.ts'])
    for (const path of ['../outside', '/etc/passwd', 'a/../b', 'bad\npath']) {
      expect((await collectGitDiff({ cwd: dir, includeUntracked: true, path })).kind).toBe('error')
    }
    expect((await collectGitDiff({ cwd: dir, includeUntracked: true, path: 'missing.ts' })).kind).toBe('clean')
  } finally { await rm(dir, { recursive: true, force: true }) }
})
