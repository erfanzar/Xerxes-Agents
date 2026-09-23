// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile, readFile, mkdir } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { GitScm, assertScmPath, cleanCommitMessage, commitMessagePrompt, parseScmStatus } from '../src/workspace/gitScm.js'

const dirs: string[] = []
afterEach(async () => { await Promise.all(dirs.splice(0).map(dir => rm(dir, { recursive: true, force: true }))) })

async function git(cwd: string, ...args: string[]): Promise<string> {
  const proc = Bun.spawn(['git', ...args], { cwd, stdout: 'pipe', stderr: 'pipe', env: { ...process.env, GIT_AUTHOR_NAME: 't', GIT_AUTHOR_EMAIL: 't@t', GIT_COMMITTER_NAME: 't', GIT_COMMITTER_EMAIL: 't@t' } })
  const out = await new Response(proc.stdout).text()
  if ((await proc.exited) !== 0) throw new Error(`git ${args.join(' ')}: ${await new Response(proc.stderr).text()}`)
  return out
}

async function repo(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'xr-scm-'))
  dirs.push(dir)
  await git(dir, 'init', '-q', '-b', 'main')
  await git(dir, 'config', 'user.name', 'Test')
  await git(dir, 'config', 'user.email', 'test@example.com')
  await git(dir, 'config', 'commit.gpgsign', 'false')
  return dir
}

test('porcelain v2 parsing splits index from worktree and keeps paths with spaces intact', () => {
  const out = [
    '# branch.oid 1234567890abcdef',
    '# branch.head feature/login',
    '# branch.upstream origin/feature/login',
    '# branch.ab +2 -1',
    '1 M. N... 100644 100644 100644 aaa bbb src/staged only.ts',
    '1 .M N... 100644 100644 100644 aaa aaa src/unstaged.ts',
    '1 MM N... 100644 100644 100644 aaa bbb both.ts',
    '2 R. N... 100644 100644 100644 aaa aaa R100 new name.ts',
    'old name.ts',
    'u UU N... 100644 100644 100644 100644 a b c conflict.ts',
    '? fresh file.md',
    '! ignored.log',
    '',
  ].join('\0')
  const status = parseScmStatus(out, '/repo')
  expect(status).toMatchObject({ branch: 'feature/login', detached: false, hasHead: true, upstream: 'origin/feature/login', ahead: 2, behind: 1, truncated: false })
  expect(status.staged).toEqual([
    { path: 'src/staged only.ts', status: 'M' },
    { path: 'both.ts', status: 'M' },
    { path: 'new name.ts', status: 'R', origPath: 'old name.ts' },
  ])
  expect(status.unstaged).toEqual([{ path: 'src/unstaged.ts', status: 'M' }, { path: 'both.ts', status: 'M' }])
  expect(status.conflicts).toEqual([{ path: 'conflict.ts', status: 'U' }])
  expect(status.untracked).toEqual([{ path: 'fresh file.md', status: '?' }])
})

test('an unborn branch and a detached head are reported as such', () => {
  expect(parseScmStatus('# branch.oid (initial)\0# branch.head main\0', '/r')).toMatchObject({ hasHead: false, branch: 'main', detached: false })
  expect(parseScmStatus('# branch.oid abc\0# branch.head (detached)\0', '/r')).toMatchObject({ branch: null, detached: true })
})

test('groups beyond the render budget keep exact counts', () => {
  const out = Array.from({ length: 5 }, (_, i) => `? f${i}.txt`).join('\0')
  const status = parseScmStatus(out, '/r', 2)
  expect(status.untracked).toHaveLength(2)
  expect(status.counts.untracked).toBe(5)
  expect(status.truncated).toBe(true)
})

test('paths must stay inside the repository', () => {
  for (const bad of ['', '/etc/passwd', '../x', 'a/../../x', 'C:\\x', 'a\nb', 'a\0b']) expect(() => assertScmPath(bad)).toThrow()
  expect(assertScmPath('src/a b.ts')).toBe('src/a b.ts')
})

test('a directory outside any repository opens as null', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xr-scm-none-'))
  dirs.push(dir)
  expect(await GitScm.open(dir)).toBeNull()
})

test('stage, unstage, discard and commit round-trip against a real repository', async () => {
  const dir = await repo()
  await writeFile(join(dir, 'a.ts'), 'one\n')
  await writeFile(join(dir, '*.ts'), 'literal star\n')
  await mkdir(join(dir, 'sub'))
  const scm = (await GitScm.open(join(dir, 'sub')))!
  expect(scm.root.endsWith(dir.split('/').pop()!)).toBe(true)

  let status = await scm.status()
  expect(status.hasHead).toBe(false)
  expect(status.untracked.map(f => f.path).sort()).toEqual(['*.ts', 'a.ts'])

  // `*.ts` is a file name, not a glob: staging it must not stage a.ts.
  await scm.stage(['*.ts'])
  status = await scm.status()
  expect(status.staged.map(f => f.path)).toEqual(['*.ts'])
  await scm.unstage(['*.ts'])
  expect((await scm.status()).staged).toEqual([])

  await scm.stageAll()
  const first = await scm.commit('Add files')
  expect(first.subject).toBe('Add files')
  status = await scm.status()
  expect(status.hasHead).toBe(true)
  expect(status.counts).toEqual({ staged: 0, unstaged: 0, untracked: 0, conflicts: 0 })

  await writeFile(join(dir, 'a.ts'), 'one\ntwo\n')
  await scm.stage(['a.ts'])
  await writeFile(join(dir, 'a.ts'), 'one\ntwo\nthree\n')
  status = await scm.status()
  expect(status.staged).toEqual([{ path: 'a.ts', status: 'M' }])
  expect(status.unstaged).toEqual([{ path: 'a.ts', status: 'M' }])

  const staged = await scm.diff('a.ts', { staged: true })
  expect(staged.lines.filter(l => l.kind === 'add').map(l => l.text)).toEqual(['+two'])
  const worktree = await scm.diff('a.ts')
  expect(worktree.lines.filter(l => l.kind === 'add').map(l => l.text)).toEqual(['+three'])

  // Discard drops only the unstaged part; the staged "two" survives.
  await writeFile(join(dir, 'scratch.txt'), 'temp\n')
  await scm.discard(['a.ts', 'scratch.txt'])
  expect(await readFile(join(dir, 'a.ts'), 'utf8')).toBe('one\ntwo\n')
  expect(existsSync(join(dir, 'scratch.txt'))).toBe(false)
  status = await scm.status()
  expect(status.staged).toEqual([{ path: 'a.ts', status: 'M' }])
  expect(status.unstaged).toEqual([])

  const context = await scm.commitContext()
  expect(context.diff).toContain('+two')
  expect(context.files).toBe('M\ta.ts')
  expect(context.recentSubjects).toEqual(['Add files'])

  await expect(scm.commit('   ')).rejects.toThrow('commit message')
  const second = await scm.commit('Extend a')
  expect((await scm.log()).map(c => c.subject)).toEqual(['Extend a', 'Add files'])
  expect(second.short.length).toBeGreaterThan(3)
})

test('untracked files diff against nothing, and concurrent mutations do not collide on the index lock', async () => {
  const dir = await repo()
  await writeFile(join(dir, 'base.txt'), 'x\n')
  await git(dir, 'add', '.')
  await git(dir, 'commit', '-q', '-m', 'base')
  const scm = (await GitScm.open(dir))!
  const names = Array.from({ length: 12 }, (_, i) => `f${i}.txt`)
  await Promise.all(names.map(name => writeFile(join(dir, name), `${name}\n`)))
  const preview = await scm.diff('f0.txt', { untracked: true })
  expect(preview.lines.some(l => l.kind === 'add' && l.text === '+f0.txt')).toBe(true)
  await Promise.all(names.map(name => scm.stage([name])))
  expect((await scm.status()).counts.staged).toBe(12)
})

test('branches: create, list, switch; push sets an upstream on first push; pull fast-forwards', async () => {
  const dir = await repo()
  const remote = await mkdtemp(join(tmpdir(), 'xr-scm-remote-'))
  dirs.push(remote)
  await git(remote, 'init', '-q', '--bare', '-b', 'main')
  await writeFile(join(dir, 'r.txt'), 'r\n')
  await git(dir, 'add', '.')
  await git(dir, 'commit', '-q', '-m', 'root')
  const scm = (await GitScm.open(dir))!

  await expect(scm.push()).rejects.toThrow('no remote')
  await git(dir, 'remote', 'add', 'origin', remote)

  await expect(scm.switchBranch('bad..name', { create: true })).rejects.toThrow('not a valid branch name')
  await scm.switchBranch('feature', { create: true })
  let branches = await scm.branches()
  expect(branches.find(b => b.current)?.name).toBe('feature')
  expect(branches.map(b => b.name).sort()).toEqual(['feature', 'main'])

  await scm.push()
  let status = await scm.status()
  expect(status.upstream).toBe('origin/feature')
  expect(status.ahead).toBe(0)

  // Someone else pushes to the same branch; fetch shows we are behind, pull catches up.
  const other = await mkdtemp(join(tmpdir(), 'xr-scm-other-'))
  dirs.push(other)
  await git(other, 'clone', '-q', '-b', 'feature', remote, '.')
  await git(other, 'config', 'user.name', 'O')
  await git(other, 'config', 'user.email', 'o@o')
  await writeFile(join(other, 'o.txt'), 'o\n')
  await git(other, 'add', '.')
  await git(other, 'commit', '-q', '-m', 'from elsewhere')
  await git(other, 'push', '-q')
  await scm.fetch()
  status = await scm.status()
  expect(status.behind).toBe(1)
  await scm.pull()
  status = await scm.status()
  expect(status.behind).toBe(0)
  expect((await scm.log(1))[0]?.subject).toBe('from elsewhere')

  await scm.switchBranch('main')
  branches = await scm.branches()
  expect(branches.find(b => b.current)?.name).toBe('main')
})

test('the commit-message prompt carries the house style, and replies are unwrapped', () => {
  const prompt = commitMessagePrompt({ stat: ' a.ts | 2 +-', files: 'M\ta.ts\nD\told.ts', diff: '+x', truncated: true, recentSubjects: ['fix(ui): align rows', 'feat(daemon): add git'] })
  expect(prompt).toContain('- fix(ui): align rows')
  expect(prompt).toContain('Diff (truncated')
  expect(prompt).toContain('Changed files (name-status; everything this commit will include):\nM\ta.ts\nD\told.ts')
  // Detailed by default: a body with an overview and grouped bullets.
  expect(prompt).toContain('bulleted list')
  expect(prompt).toContain('Only a truly trivial one-line change may omit the body')
  expect(cleanCommitMessage('```\nfix(ui): tidy\n\nBody.\n```')).toBe('fix(ui): tidy\n\nBody.')
  expect(cleanCommitMessage('Commit message: "Add git panel"')).toBe('Add git panel')
  expect(cleanCommitMessage('Fix "quoted" thing')).toBe('Fix "quoted" thing')
})

test('a commit opens to its message and changed files, and each file shows that commit\'s diff', async () => {
  const dir = await repo()
  await writeFile(join(dir, 'keep.ts'), 'one\n')
  await writeFile(join(dir, 'old name.ts'), 'a\nb\nc\nd\ne\nf\n')
  await git(dir, 'add', '.')
  await git(dir, 'commit', '-q', '-m', 'root')
  await writeFile(join(dir, 'keep.ts'), 'one\ntwo\n')
  await writeFile(join(dir, 'fresh.md'), 'hi\n')
  await git(dir, 'mv', 'old name.ts', 'new name.ts')
  await git(dir, 'add', '.')
  await git(dir, 'commit', '-q', '-m', 'Rework files', '-m', 'Why: tidy the layout.')
  const scm = (await GitScm.open(dir))!
  const [head, root] = await scm.log(2)
  const shown = await scm.showCommit(head!.short)
  expect(shown.commit).toMatchObject({ subject: 'Rework files', body: 'Why: tidy the layout.', parents: 1 })
  expect(shown.files).toEqual(expect.arrayContaining([
    { path: 'keep.ts', status: 'M' },
    { path: 'fresh.md', status: 'A' },
    { path: 'new name.ts', origPath: 'old name.ts', status: 'R' },
  ]))
  const diff = await scm.commitDiff(head!.hash, 'keep.ts')
  expect(diff.lines.filter(l => l.kind === 'add').map(l => l.text)).toEqual(['+two'])
  // The first commit has no parent; its files still list.
  expect((await scm.showCommit(root!.hash)).files.map(f => f.path).sort()).toEqual(['keep.ts', 'old name.ts'])
  await expect(scm.showCommit('HEAD~1')).rejects.toThrow('hash')
  await expect(scm.showCommit('--output=/tmp/x')).rejects.toThrow('hash')
})

test('generate and commit cover every change — unstaged and new files too — but never ignored ones', async () => {
  const dir = await repo()
  await writeFile(join(dir, '.gitignore'), '*.log\n')
  await writeFile(join(dir, 'tracked.ts'), 'one\n')
  await writeFile(join(dir, 'gone.ts'), 'bye\n')
  await git(dir, 'add', '.')
  await git(dir, 'commit', '-q', '-m', 'base')
  await writeFile(join(dir, 'tracked.ts'), 'one\ntwo\n')
  await rm(join(dir, 'gone.ts'))
  await writeFile(join(dir, 'new.md'), '# fresh\n')
  await writeFile(join(dir, 'debug.log'), 'noise\n')
  const scm = (await GitScm.open(dir))!

  const context = await scm.commitContext()
  expect(context.files.split('\n').sort()).toEqual(['A\tnew.md', 'D\tgone.ts', 'M\ttracked.ts'])
  expect(context.diff).toContain('+two')
  expect(context.diff).toContain('+# fresh')
  expect(context.diff).not.toContain('noise')
  // The scratch index left the real staging area exactly as it was.
  expect((await scm.status()).counts.staged).toBe(0)

  const commit = await scm.commit('Everything at once', { all: true })
  expect(commit.subject).toBe('Everything at once')
  const after = await scm.status()
  expect(after.counts).toEqual({ staged: 0, unstaged: 0, untracked: 0, conflicts: 0 })
  expect((await git(dir, 'ls-files')).split('\n').filter(Boolean).sort()).toEqual(['.gitignore', 'new.md', 'tracked.ts'])
})
