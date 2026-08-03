// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { collectGitDiff, parseUnifiedDiff, type GitDiffRunner } from '../lib/gitDiff.js'

const SAMPLE_DIFF = [
  'diff --git a/src/a.ts b/src/a.ts',
  'index 1111111..2222222 100644',
  '--- a/src/a.ts',
  '+++ b/src/a.ts',
  '@@ -1,3 +1,4 @@',
  ' context line',
  '-removed line',
  '+added line',
  '+another add',
  'diff --git a/src/b.ts b/src/b.ts',
  'index 3333333..4444444 100644',
  '--- a/src/b.ts',
  '+++ b/src/b.ts',
  '@@ -10,1 +10,1 @@',
  '-old',
  '+new',
  ''
].join('\n')

describe('parseUnifiedDiff', () => {
  it('classifies rows and counts files, insertions, and deletions', () => {
    const parsed = parseUnifiedDiff(SAMPLE_DIFF)

    expect(parsed.files).toBe(2)
    expect(parsed.insertions).toBe(3)
    expect(parsed.deletions).toBe(2)
    expect(parsed.truncated).toBe(false)

    const kinds = parsed.lines.map(line => line.kind)
    expect(kinds).toContain('file')
    expect(kinds).toContain('hunk')
    expect(kinds).toContain('add')
    expect(kinds).toContain('del')
    expect(kinds).toContain('context')
    expect(parsed.lines[0]).toEqual({ kind: 'file', text: 'src/a.ts' })
    expect(parsed.lines.find(line => line.text === ' context line')).toMatchObject({ oldLine: 1, newLine: 1 })
    expect(parsed.lines.find(line => line.text === '-removed line')).toMatchObject({ oldLine: 2 })
    expect(parsed.lines.find(line => line.text === '-removed line')?.newLine).toBeUndefined()
    expect(parsed.lines.find(line => line.text === '+added line')).toMatchObject({ newLine: 2 })
    expect(parsed.lines.find(line => line.text === '+added line')?.oldLine).toBeUndefined()
    // ---/+++ file markers are dropped; the diff --git row names the file.
    expect(parsed.lines.some(line => line.text.startsWith('---'))).toBe(false)
    expect(parsed.lines.some(line => line.text.startsWith('+++'))).toBe(false)
  })

  it('truncates at the line cap with an explicit marker', () => {
    const parsed = parseUnifiedDiff(SAMPLE_DIFF, { maxLines: 3 })

    expect(parsed.lines).toHaveLength(3)
    expect(parsed.truncated).toBe(true)
  })

  it('truncates at the byte cap', () => {
    const parsed = parseUnifiedDiff(SAMPLE_DIFF, { maxBytes: 30 })

    expect(parsed.truncated).toBe(true)
    expect(parsed.lines.length).toBeLessThan(5)
  })
})

function fakeRunner(handlers: Record<string, { code: number; stdout?: string; stderr?: string }>): GitDiffRunner {
  return async args => {
    const key = args.join(' ')
    for (const [prefix, result] of Object.entries(handlers)) {
      if (key.startsWith(prefix)) {
        return { code: result.code, stderr: result.stderr ?? '', stdout: result.stdout ?? '' }
      }
    }
    throw new Error(`unexpected git argv: ${key}`)
  }
}

describe('collectGitDiff', () => {
  it('reports an honest error outside a git repository', async () => {
    const result = await collectGitDiff({
      cwd: '/tmp/nope',
      run: fakeRunner({ 'rev-parse --is-inside-work-tree': { code: 128, stderr: 'fatal: not a git repository' } })
    })

    expect(result).toEqual({ kind: 'error', message: 'not a git repository — the diff viewer needs a git worktree' })
  })

  it('reports a clean worktree when there is no diff and no untracked files', async () => {
    const result = await collectGitDiff({
      cwd: '/repo',
      run: fakeRunner({
        'rev-parse --is-inside-work-tree': { code: 0, stdout: 'true\n' },
        'rev-parse --verify HEAD': { code: 0, stdout: 'abc123\n' },
        'diff --no-color --no-ext-diff HEAD --': { code: 0, stdout: '' },
        'ls-files --others --exclude-standard': { code: 0, stdout: '' }
      })
    })

    expect(result).toEqual({ kind: 'clean' })
  })

  it('returns parsed diff plus capped untracked files', async () => {
    const result = await collectGitDiff({
      cwd: '/repo',
      maxUntracked: 1,
      run: fakeRunner({
        'rev-parse --is-inside-work-tree': { code: 0, stdout: 'true\n' },
        'rev-parse --verify HEAD': { code: 0, stdout: 'abc123\n' },
        'diff --no-color --no-ext-diff HEAD --': { code: 0, stdout: SAMPLE_DIFF },
        'ls-files --others --exclude-standard': { code: 0, stdout: 'new-one.ts\nnew-two.ts\n' }
      })
    })

    expect(result.kind).toBe('ok')
    if (result.kind !== 'ok') return
    expect(result.diff.files).toBe(2)
    expect(result.diff.insertions).toBe(3)
    expect(result.diff.deletions).toBe(2)
    expect(result.diff.untracked).toEqual(['new-one.ts'])
    expect(result.diff.untrackedTruncated).toBe(true)
  })

  it('falls back to the empty tree when the repository has no HEAD commit', async () => {
    const seen: string[] = []
    const run: GitDiffRunner = async args => {
      seen.push(args.join(' '))
      const key = args.join(' ')
      if (key.startsWith('rev-parse --is-inside-work-tree')) return { code: 0, stderr: '', stdout: 'true\n' }
      if (key.startsWith('rev-parse --verify HEAD')) return { code: 1, stderr: '', stdout: '' }
      if (key.startsWith('diff ')) return { code: 0, stderr: '', stdout: '' }
      if (key.startsWith('ls-files ')) return { code: 0, stderr: '', stdout: 'first.ts\n' }
      throw new Error(`unexpected: ${key}`)
    }

    const result = await collectGitDiff({ cwd: '/repo', run })

    expect(seen.some(argv => argv.includes('4b825dc642cb6eb9a060e54bf8d69288fbee4904'))).toBe(true)
    expect(result.kind).toBe('ok')
  })

  it('surfaces a git diff failure with stderr context', async () => {
    const result = await collectGitDiff({
      cwd: '/repo',
      run: fakeRunner({
        'rev-parse --is-inside-work-tree': { code: 0, stdout: 'true\n' },
        'rev-parse --verify HEAD': { code: 0, stdout: 'abc\n' },
        'diff --no-color --no-ext-diff HEAD --': { code: 1, stderr: 'fatal: bad revision' }
      })
    })

    expect(result).toEqual({ kind: 'error', message: 'git diff failed: fatal: bad revision' })
  })
})
