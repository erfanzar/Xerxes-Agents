// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, realpath, rm, readdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { nativeSubagentWorktrees } from '../src/runtime/subagentWorktrees.js'
import { SubAgentManager } from '../src/agents/subagentManager.js'

async function git(cwd: string, ...args: string[]) {
  const child = Bun.spawn(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', ...args], { cwd, stdout: 'pipe', stderr: 'pipe' })
  const [code, stdout, stderr] = await Promise.all([child.exited, new Response(child.stdout).text(), new Response(child.stderr).text()])
  if (code !== 0) throw new Error(stderr)
  return stdout.trim()
}

test('owned agent worktrees isolate edits and survive reload without deleting reviewable work', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-agent-worktrees-')))
  try {
    await git(root, 'init')
    await Bun.write(join(root, 'file.txt'), 'original')
    await Bun.write(join(root, '.gitignore'), 'ignored.txt\n')
    await git(root, 'add', '.')
    await git(root, 'commit', '-m', 'fixture baseline')
    await Bun.write(join(root, 'file.txt'), 'parent edits')
    const port = nativeSubagentWorktrees(root)
    const first = await port.create({ taskId: 'one', taskName: 'One' })
    const second = await port.create({ taskId: 'two', taskName: 'Two' })
    expect(first.path).not.toBe(second.path)
    expect(await Bun.file(join(first.path, 'file.txt')).text()).toBe('original')
    await Bun.write(join(first.path, 'file.txt'), 'first agent')
    expect(await Bun.file(join(second.path, 'file.txt')).text()).toBe('original')
    expect(await Bun.file(join(root, 'file.txt')).text()).toBe('parent edits')
    expect(await port.isClean(first)).toBe(false)
    await expect(port.remove(first)).rejects.toThrow('review')
    await git(first.path, 'add', 'file.txt')
    await git(first.path, 'commit', '-m', 'fixture agent result')
    const reloaded = nativeSubagentWorktrees(root)
    expect(await reloaded.isClean(first)).toBe(false)
    await expect(reloaded.remove(first)).rejects.toThrow('review')
    await Bun.write(join(second.path, 'ignored.txt'), 'setup evidence')
    expect(await port.isClean(second)).toBe(false)
    await expect(port.remove(second)).rejects.toThrow('review')
    await rm(join(second.path, 'ignored.txt'))
    expect(await reloaded.isClean(second)).toBe(true)
    await reloaded.remove(second)
    expect(await Bun.file(join(second.path, 'file.txt')).exists()).toBe(false)
    expect(await Bun.file(join(first.path, 'file.txt')).text()).toBe('first agent')
    await expect(port.remove({ path: root, branch: first.branch })).rejects.toThrow('owned')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('agent isolation rejects a non-Git workspace instead of using it as a fallback', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-no-git-'))
  try { await expect(nativeSubagentWorktrees(root).create({ taskId: 'one', taskName: 'One' })).rejects.toThrow('Git operation failed') }
  finally { await rm(root, { recursive: true, force: true }) }
})

test('manager cancellation retains a real checkout until the running agent settles', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-agent-worktree-cancel-')))
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  const manager = new SubAgentManager({ worktree: nativeSubagentWorktrees(root), runner: async request => {
    expect(request.prompt).not.toContain('Commit your changes before finishing')
    expect(request.worktree?.path).not.toBe(root)
    await gate
    expect(await Bun.file(join(request.worktree!.path, 'file.txt')).exists()).toBe(true)
    return 'settled'
  } })
  try {
    await git(root, 'init')
    await Bun.write(join(root, 'file.txt'), 'original')
    await git(root, 'add', '.')
    await git(root, 'commit', '-m', 'fixture baseline')
    const task = await manager.spawn({ prompt: 'isolated task', isolation: 'worktree' })
    await manager.waitFor(() => task.status === 'running', { timeoutMs: 1000 })
    expect(manager.cancel(task.id)).toBe(true)
    expect(await Bun.file(join(task.worktreePath, 'file.txt')).exists()).toBe(true)
    release()
    await manager.waitFor(() => manager.peekMailbox().some(event => event.taskId === task.id && event.type === 'worktree_removed'), { timeoutMs: 1000 })
    expect(await Bun.file(join(task.worktreePath, 'file.txt')).exists()).toBe(false)
  } finally { release(); await manager.close(); await rm(root, { recursive: true, force: true }) }
})

test('explicit worktree revisions select committed content and invalid revisions allocate nothing', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-worktree-ref-')))
  try {
    await git(root, 'init')
    await Bun.write(join(root, 'file.txt'), 'first')
    await git(root, 'add', '.')
    await git(root, 'commit', '-m', 'first fixture')
    const first = await git(root, 'rev-parse', 'HEAD')
    await git(root, 'tag', 'baseline')
    await Bun.write(join(root, 'file.txt'), 'second')
    await git(root, 'commit', '-am', 'second fixture')
    const port = nativeSubagentWorktrees(root)
    for (const ref of ['baseline', first, 'HEAD~1']) {
      const tree = await port.create({ taskId: ref, taskName: ref, config: { _nativeSubagentWorktreeRef: ref } })
      expect(await Bun.file(join(tree.path, 'file.txt')).text()).toBe('first')
      expect(await git(tree.path, 'rev-parse', 'HEAD')).toBe(first)
      await port.remove(tree)
    }
    const before = await git(root, 'worktree', 'list', '--porcelain')
    for (const ref of ['missing-ref', '--help', 'HEAD\nother', true, '', 'x'.repeat(1025)]) {
      await expect(port.create({ taskId: 'invalid', taskName: 'invalid', config: { _nativeSubagentWorktreeRef: ref } })).rejects.toThrow()
    }
    expect(await git(root, 'worktree', 'list', '--porcelain')).toBe(before)
    expect(await Bun.file(join(root, 'file.txt')).text()).toBe('second')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('working-tree capture includes edits and untracked files without changing the parent index', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-worktree-capture-')))
  try {
    await git(root, 'init')
    await Bun.write(join(root, 'file.txt'), 'original')
    await Bun.write(join(root, 'deleted.txt'), 'delete me')
    await Bun.write(join(root, '.gitignore'), 'ignored.txt\n')
    await git(root, 'add', '.')
    await git(root, 'commit', '-m', 'capture fixture')
    await Bun.write(join(root, 'file.txt'), 'staged')
    await git(root, 'add', 'file.txt')
    await Bun.write(join(root, 'file.txt'), 'unstaged after staged')
    await rm(join(root, 'deleted.txt'))
    await Bun.write(join(root, 'new.bin'), new Uint8Array([0, 255, 42, 0]))
    await Bun.write(join(root, 'ignored.txt'), 'local secret')
    const index = await Bun.file(join(root, '.git/index')).bytes()
    const port = nativeSubagentWorktrees(root)
    const tree = await port.create({ taskId: 'capture', taskName: 'Capture', config: { _nativeSubagentWorktreeSource: 'working-tree' } })
    expect(await Bun.file(join(tree.path, 'file.txt')).text()).toBe('unstaged after staged')
    expect(await Bun.file(join(tree.path, 'new.bin')).bytes()).toEqual(new Uint8Array([0, 255, 42, 0]))
    expect(await Bun.file(join(tree.path, 'deleted.txt')).exists()).toBe(false)
    expect(await Bun.file(join(tree.path, 'ignored.txt')).exists()).toBe(false)
    expect(await git(tree.path, 'diff', '--cached', '--name-only')).toBe('')
    expect(await git(tree.path, 'diff', '--name-only')).toContain('file.txt')
    expect(await Bun.file(join(root, '.git/index')).bytes()).toEqual(index)
    expect(await git(root, 'show', ':file.txt')).toBe('staged')
    expect(await Bun.file(join(root, 'file.txt')).text()).toBe('unstaged after staged')
    expect(await port.isClean(tree)).toBe(false)
    await expect(port.remove(tree)).rejects.toThrow('review')
    await expect(port.create({ taskId: 'conflict', taskName: 'Conflict', config: { _nativeSubagentWorktreeSource: 'working-tree', _nativeSubagentWorktreeRef: 'HEAD' } })).rejects.toThrow('mutually exclusive')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('retained workspace inspection survives reload and separates captured edits from agent changes', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-worktree-review-')))
  try {
    await git(root, 'init')
    await Bun.write(join(root, 'file.txt'), 'original')
    await git(root, 'add', '.')
    await git(root, 'commit', '-m', 'review fixture')
    const port = nativeSubagentWorktrees(root)
    expect((await port.list()).records).toHaveLength(0)
    expect(await readdir(join(root, '.git'))).not.toContain('xerxes-agent-worktrees')
    await Bun.write(join(root, 'file.txt'), 'inherited')
    await Bun.write(join(root, 'inherited-new.txt'), 'inherited untracked')
    const tree = await port.create({ taskId: 'review-task', taskName: 'Review', config: { _nativeSubagentWorktreeSource: 'working-tree' } })
    const id = tree.branch.replace('xerxes/agent-', '')
    const reopened = nativeSubagentWorktrees(root)
    expect((await reopened.list()).records).toMatchObject([{ id, taskId: 'review-task', path: tree.path }])
    expect((await reopened.inspect(id)).diff).toBe('')
    await Bun.write(join(tree.path, 'file.txt'), 'agent change')
    await Bun.write(join(tree.path, 'new-agent.txt'), 'new agent file')
    const childIndexPath = await git(tree.path, 'rev-parse', '--path-format=absolute', '--git-path', 'index')
    const childIndex = await Bun.file(childIndexPath).bytes()
    const review = await reopened.inspect(id)
    expect(await Bun.file(childIndexPath).bytes()).toEqual(childIndex)
    expect(review.diff).toContain('+new agent file')
    expect(review.diff).toContain('-inherited')
    expect(review.diff).toContain('+agent change')
    expect(review.snapshotTree).toBeDefined()
    expect(await Bun.file(join(root, 'file.txt')).text()).toBe('inherited')
    await expect(reopened.inspect('../foreign')).rejects.toThrow('UUID')
    await rm(tree.path, { recursive: true, force: true })
    expect((await reopened.list()).records[0]?.error).toBeDefined()
    await expect(reopened.inspect(id)).rejects.toThrow()
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('inherited Git routing cannot redirect an agent workspace into another repository', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-worktree-git-env-')))
  try {
    const selected = join(root, 'selected'), foreign = join(root, 'foreign')
    for (const directory of [selected, foreign]) {
      await git(root, 'init', directory)
      await Bun.write(join(directory, 'file.txt'), directory)
      await git(directory, 'add', '.')
      await git(directory, 'commit', '-m', 'environment fixture')
    }
    const index = await Bun.file(join(foreign, '.git/index')).bytes()
    const module = new URL('../src/runtime/subagentWorktrees.ts', import.meta.url).href
    const script = `import { nativeSubagentWorktrees } from ${JSON.stringify(module)}; const port = nativeSubagentWorktrees(process.argv[1]); const tree = await port.create({ taskId: 'isolated', taskName: 'Isolated' }); console.log(JSON.stringify(tree));`
    const child = Bun.spawn([process.execPath, '-e', script, selected], { env: { ...Bun.env, GIT_DIR: join(foreign, '.git'), GIT_WORK_TREE: foreign, GIT_INDEX_FILE: join(foreign, '.git/index'), GIT_COMMON_DIR: join(foreign, '.git') }, stdout: 'pipe', stderr: 'pipe' })
    const [code, output, error] = await Promise.all([child.exited, new Response(child.stdout).text(), new Response(child.stderr).text()])
    expect(error).toBe('')
    expect(code).toBe(0)
    const tree = JSON.parse(output) as { path: string; branch: string }
    expect(await Bun.file(join(tree.path, 'file.txt')).text()).toBe(selected)
    expect(await Bun.file(join(foreign, '.git/index')).bytes()).toEqual(index)
    expect(await git(foreign, 'worktree', 'list', '--porcelain')).not.toContain(tree.path)
    await nativeSubagentWorktrees(selected).remove(tree)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('integration check binds reviewed text and binary content and preserves both workspaces', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-worktree-apply-check-')))
  try {
    await git(root, 'init')
    await Bun.write(join(root, 'file.txt'), 'original\n')
    await Bun.write(join(root, 'binary.bin'), new Uint8Array([0, 1, 2, 255]))
    await git(root, 'add', '.')
    await git(root, 'commit', '-m', 'check fixture')
    const port = nativeSubagentWorktrees(root)
    const tree = await port.create({ taskId: 'check', taskName: 'Check' })
    const id = tree.branch.replace('xerxes/agent-', '')
    await Bun.write(join(tree.path, 'file.txt'), 'agent result  \n')
    await Bun.write(join(tree.path, 'binary.bin'), new Uint8Array([0, 3, 4, 255]))
    const review = await port.inspect(id)
    expect(review.diff).toContain('GIT binary patch')
    expect((await port.checkApply(id, review.reviewId)).canApply).toBe(true)
    expect(await Bun.file(join(root, 'file.txt')).text()).toBe('original\n')
    expect(await Bun.file(join(root, 'binary.bin')).bytes()).toEqual(new Uint8Array([0, 1, 2, 255]))
    await Bun.write(join(root, 'file.txt'), 'conflicting parent edit\n')
    expect((await port.checkApply(id, review.reviewId)).canApply).toBe(false)
    expect(await Bun.file(join(root, 'file.txt')).text()).toBe('conflicting parent edit\n')
    expect(await Bun.file(join(tree.path, 'file.txt')).text()).toBe('agent result  \n')
    await Bun.write(join(tree.path, 'binary.bin'), new Uint8Array([0, 5, 6, 255]))
    await expect(port.checkApply(id, review.reviewId)).rejects.toThrow('changed since review')
    expect((await port.inspect(id)).reviewId).not.toBe(review.reviewId)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('working-tree capture isolates nested repositories, excludes secrets, and preserves source indexes', async () => {
  const root=await realpath(await mkdtemp(join(tmpdir(),'xerxes-gitlink-capture-')))
  try {
    await git(root,'init');await Bun.write(join(root,'file.txt'),'parent');await git(root,'add','.');await git(root,'commit','-m','parent')
    const nested=join(root,'nested')
    await Bun.write(join(nested,'child.txt'),'child');await git(nested,'init');await git(nested,'add','.');await git(nested,'commit','-m','nested')
    const before=await Bun.file(join(root,'.git/index')).bytes()
    const childIndex=await Bun.file(join(nested,'.git/index')).bytes()
    await Bun.write(join(nested,'child.txt'),'dirty child')
    await Bun.write(join(nested,'.gitignore'),'secret.txt\n')
    await Bun.write(join(nested,'secret.txt'),'must stay local')
    await Bun.write(join(nested,'new.txt'),'new child')
    const port=nativeSubagentWorktrees(root)
    const tree=await port.create({taskId:'nested',taskName:'Nested',config:{_nativeSubagentWorktreeSource:'working-tree'}})
    expect(await Bun.file(join(tree.path,'nested/child.txt')).text()).toBe('dirty child')
    expect(await Bun.file(join(tree.path,'nested/new.txt')).text()).toBe('new child')
    expect(await Bun.file(join(tree.path,'nested/secret.txt')).exists()).toBe(false)
    expect(await Bun.file(join(tree.path,'nested/.git/config')).exists()).toBe(false)
    const id=tree.branch.replace('xerxes/agent-','')
    expect((await port.inspect(id)).diff).toBe('')
    await Bun.write(join(tree.path,'nested/child.txt'),'agent child')
    expect((await port.inspect(id)).diff).toContain('+agent child')
    expect(await Bun.file(join(nested,'child.txt')).text()).toBe('dirty child')
    expect(await Bun.file(join(nested,'.git/index')).bytes()).toEqual(childIndex)
    expect(await Bun.file(join(root,'.git/index')).bytes()).toEqual(before)
  } finally {await rm(root,{recursive:true,force:true})}
})

test('initialized submodules and recursive children capture current contents without network or shared Git state', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-submodule-capture-')))
  try {
    const source = join(root, 'source'), parent = join(root, 'parent')
    for (const dir of [source, parent]) {
      await git(root, 'init', dir)
      await Bun.write(join(dir, 'file.txt'), 'baseline')
      await git(dir, 'add', '.')
      await git(dir, 'commit', '-m', 'baseline')
    }
    await git(parent, '-c', 'protocol.file.allow=always', 'submodule', 'add', source, 'module')
    await git(parent, 'commit', '-am', 'submodule')
    const child = join(parent, 'module'), grandchild = join(child, 'nested')
    await git(child, 'init', grandchild)
    await Bun.write(join(grandchild, 'nested.txt'), 'nested baseline')
    await git(grandchild, 'add', '.')
    await git(grandchild, 'commit', '-m', 'nested')
    await Bun.write(join(child, 'file.txt'), 'dirty submodule')
    const parentIndex = await Bun.file(join(parent, '.git/index')).bytes()
    const port = nativeSubagentWorktrees(parent)
    const tree = await port.create({ taskId: 'submodule', taskName: 'Submodule', config: { _nativeSubagentWorktreeSource: 'working-tree' } })
    expect(await Bun.file(join(tree.path, 'module/file.txt')).text()).toBe('dirty submodule')
    expect(await Bun.file(join(tree.path, 'module/nested/nested.txt')).text()).toBe('nested baseline')
    expect(await Bun.file(join(tree.path, 'module/.git')).exists()).toBe(false)
    expect(await Bun.file(join(parent, '.git/index')).bytes()).toEqual(parentIndex)
    expect((await port.inspect(tree.branch.replace('xerxes/agent-', ''))).diff).toBe('')
    const controller = new AbortController(); controller.abort()
    await expect(port.create({ taskId: 'cancel', taskName: 'Cancel', signal: controller.signal, config: { _nativeSubagentWorktreeSource: 'working-tree' } })).rejects.toThrow()
    expect((await port.list()).records).toHaveLength(1)
    await git(parent, 'submodule', 'deinit', '-f', 'module')
    await expect(port.create({ taskId: 'absent', taskName: 'Absent', config: { _nativeSubagentWorktreeSource: 'working-tree' } })).rejects.toThrow('Initialize the nested repository')
    expect((await port.list()).records).toHaveLength(1)
  } finally { await rm(root, { recursive: true, force: true }) }
})
