// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, realpath, rm, readdir, rename } from 'node:fs/promises'
import { tmpdir, hostname } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { nativeSubagentWorktrees } from '../src/runtime/subagentWorktrees.js'
import { applyWorkspacePatch, recoverWorkspaceApply, type WorkspaceGit } from '../src/runtime/workspaceApply.js'

async function fixture(autocrlf = false, attributes = '') {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-apply-')))
  const git: WorkspaceGit = async (args, cwd = root, env = {}, raw = false) => {
    const child = Bun.spawn(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null', ...args], { cwd, env: { ...Bun.env, GIT_OPTIONAL_LOCKS: '0', ...env }, stdout: 'pipe', stderr: 'pipe' })
    const [code, output, error] = await Promise.all([child.exited, new Response(child.stdout).text(), new Response(child.stderr).text()])
    if (code) throw new Error(error)
    return raw ? output : output.trim()
  }
  await git(['init'])
  await git(['config', 'core.autocrlf', String(autocrlf)])
  if (attributes) await Bun.write(join(root, '.gitattributes'), attributes)
  await Bun.write(join(root, 'a.txt'), 'original a\n')
  await Bun.write(join(root, 'b.txt'), 'original b\n')
  await Bun.write(join(root, 'binary.bin'), new Uint8Array([0, 1, 255]))
  await git(['add', '.']); await git(['commit', '-m', 'apply fixture'])
  const port = nativeSubagentWorktrees(root)
  const tree = await port.create({ taskId: 'apply', taskName: 'Apply' })
  const id = tree.branch.replace('xerxes/agent-', '')
  await Bun.write(join(tree.path, 'a.txt'), 'agent a\n')
  await Bun.write(join(tree.path, 'b.txt'), 'agent b\n')
  return { root, git, port, tree, id, storage: join(root, '.git/xerxes-agent-worktrees') }
}

test.each(['', '*.txt text eol=lf\n'])('integration predicts destination EOL conversions with autocrlf and attributes (%s)', async attributes => {
  const f = await fixture(true, attributes)
  try {
    const review = await f.port.inspect(f.id), check = await f.port.checkApply(f.id, review.reviewId)
    const applied = await f.port.apply(f.id, review.reviewId, check.destinationState!)
    expect(applied.status).toBe('applied')
    expect(await Bun.file(join(f.root, 'a.txt')).text()).toBe(attributes ? 'agent a\n' : 'agent a\r\n')
  } finally { await rm(f.root, { recursive: true, force: true }) }
})

test('checked integration applies text, binary, rename and new files while preserving the parent index and source', async () => {
  const f = await fixture()
  try {
    await rename(join(f.tree.path, 'a.txt'), join(f.tree.path, 'renamed.txt'))
    await Bun.write(join(f.tree.path, 'binary.bin'), new Uint8Array([0, 42, 255]))
    await Bun.write(join(f.tree.path, 'new.txt'), 'new agent file\n')
    await Bun.write(join(f.root, 'staged.txt'), 'user staged file')
    await f.git(['add', 'staged.txt'])
    const index = await Bun.file(join(f.root, '.git/index')).bytes()
    const review = await f.port.inspect(f.id)
    const check = await f.port.checkApply(f.id, review.reviewId)
    expect(check.canApply).toBe(true)
    const applied = await f.port.apply(f.id, review.reviewId, check.destinationState!)
    expect(applied.status).toBe('applied')
    expect(await Bun.file(join(f.root, 'a.txt')).exists()).toBe(false)
    expect(await Bun.file(join(f.root, 'renamed.txt')).text()).toBe('agent a\n')
    expect(await Bun.file(join(f.root, 'new.txt')).text()).toBe('new agent file\n')
    expect(await Bun.file(join(f.root, 'binary.bin')).bytes()).toEqual(new Uint8Array([0, 42, 255]))
    expect(await Bun.file(join(f.root, '.git/index')).bytes()).toEqual(index)
    expect(await Bun.file(join(f.tree.path, 'renamed.txt')).text()).toBe('agent a\n')
    expect(await Bun.file(join(applied.backupPath, 'record.json')).json()).toMatchObject({ status: 'applied', reviewId: review.reviewId })
    await expect(f.port.apply(f.id, review.reviewId, check.destinationState!)).rejects.toThrow('Destination changed')
  } finally { await rm(f.root, { recursive: true, force: true }) }
})

test.each([false, true])('partial apply failure restores owned writes and preserves concurrent changes (concurrent=%s)', async concurrent => {
  const f = await fixture()
  try {
    const review = await f.port.inspect(f.id)
    const check = await f.port.checkApply(f.id, review.reviewId)
    const injected: WorkspaceGit = async (args, cwd, env, raw) => {
      if (cwd === f.root && args[0] === 'apply' && !args.includes('--check') && !args.includes('--numstat')) {
        await Bun.write(join(f.root, 'a.txt'), 'agent a\n')
        if (concurrent) await Bun.write(join(f.root, 'b.txt'), 'concurrent writer\n')
        throw new Error('injected partial write failure')
      }
      return f.git(args, cwd, env, raw)
    }
    await expect(applyWorkspacePatch({ git: injected, storage: f.storage, destination: f.root, patch: review.diff, reviewId: review.reviewId, destinationState: check.destinationState! })).rejects.toThrow(concurrent ? 'needs-recovery' : 'rolled-back')
    expect(await Bun.file(join(f.root, 'a.txt')).text()).toBe('original a\n')
    expect(await Bun.file(join(f.root, 'b.txt')).text()).toBe(concurrent ? 'concurrent writer\n' : 'original b\n')
    const id = (await readdir(f.storage)).find(name => name.startsWith('integration-'))!
    const record = await Bun.file(join(f.storage, id, 'record.json')).json()
    expect(record.status).toBe(concurrent ? 'needs-recovery' : 'rolled-back')
    expect(await Bun.file(join(f.storage, id, 'original-0')).text()).toBe('original a\n')
  } finally { await rm(f.root, { recursive: true, force: true }) }
})

test('destination drift after checking rejects integration without replacing the newer file', async () => {
  const f = await fixture()
  try {
    const review = await f.port.inspect(f.id), check = await f.port.checkApply(f.id, review.reviewId)
    await Bun.write(join(f.root, 'a.txt'), 'new parent work\n')
    await expect(f.port.apply(f.id, review.reviewId, check.destinationState!)).rejects.toThrow('Destination changed')
    expect(await Bun.file(join(f.root, 'a.txt')).text()).toBe('new parent work\n')
    expect(await Bun.file(join(f.root, 'b.txt')).text()).toBe('original b\n')
  } finally { await rm(f.root, { recursive: true, force: true }) }
})

test.each(['clean', 'concurrent', 'corrupt', 'live'] as const)('interrupted integration recovery: %s', async scenario => {
  const f = await fixture()
  try {
    const review = await f.port.inspect(f.id), check = await f.port.checkApply(f.id, review.reviewId)
    const applied = await f.port.apply(f.id, review.reviewId, check.destinationState!)
    const recordPath = join(applied.backupPath, 'record.json')
    const record = await Bun.file(recordPath).json()
    // A durable prepared record with a partly applied destination is the crash boundary.
    await Bun.write(recordPath, JSON.stringify({ ...record, status: 'prepared' }))
    await Bun.write(join(f.root, 'b.txt'), scenario === 'concurrent' ? 'newer work\n' : 'original b\n')
    if (scenario === 'corrupt') await Bun.write(join(applied.backupPath, 'original-0'), 'damaged backup')
    const worker = Bun.spawn([process.execPath, '-e', 'process.exit(0)'], { stdout: 'ignore', stderr: 'ignore' })
    await worker.exited
    const lockPath = join(f.storage, 'integration.lock')
    await Bun.write(lockPath, JSON.stringify({ id: applied.id, pid: scenario === 'live' ? process.pid : worker.pid, host: hostname() }))
    const recover = () => recoverWorkspaceApply({ git: f.git, destination: f.root, storage: f.storage, id: applied.id })
    if (scenario === 'corrupt' || scenario === 'live') {
      await expect(recover()).rejects.toThrow(scenario === 'corrupt' ? 'checksum mismatch' : 'still running')
      expect(await Bun.file(join(f.root, 'a.txt')).text()).toBe('agent a\n')
      expect(await Bun.file(lockPath).exists()).toBe(true)
    } else {
      expect(await recover()).toMatchObject({ status: scenario === 'clean' ? 'rolled-back' : 'needs-recovery', conflicts: scenario === 'clean' ? [] : ['b.txt'] })
      expect(await Bun.file(join(f.root, 'a.txt')).text()).toBe('original a\n')
      expect(await Bun.file(join(f.root, 'b.txt')).text()).toBe(scenario === 'clean' ? 'original b\n' : 'newer work\n')
      expect(await Bun.file(lockPath).exists()).toBe(false)
      expect(await recover()).toMatchObject({ status: scenario === 'clean' ? 'rolled-back' : 'needs-recovery' })
    }
  } finally { await rm(f.root, { recursive: true, force: true }) }
})

test.each(['preparation', 'apply', 'recovery'])('killed workers leave recoverable ownership (%s)', async phase => {
  const f = await fixture()
  try {
    const review = await f.port.inspect(f.id), check = await f.port.checkApply(f.id, review.reviewId)
    const modulePath = fileURLToPath(new URL('../src/runtime/workspaceApply.ts', import.meta.url))
    const workerPath = join(f.storage, 'crash-worker.ts')
    await Bun.write(workerPath, `
      import { applyWorkspacePatch } from ${JSON.stringify(modulePath)};
      const options = ${JSON.stringify({ storage: f.storage, destination: f.root, patch: review.diff, destinationState: check.destinationState, reviewId: review.reviewId })};
      await applyWorkspacePatch({ ...options, git: async (args, cwd, env = {}, raw = false) => {
        if (${JSON.stringify(phase)} === 'preparation' && args.includes('--numstat')) {
          process.kill(process.pid, 'SIGKILL'); await new Promise(() => {});
        }
        if (cwd === options.destination && args[0] === 'apply' && !args.includes('--check') && !args.includes('--numstat')) {
          await Bun.write(cwd + '/a.txt', 'agent a\\n');
          process.kill(process.pid, 'SIGKILL');
          await new Promise(() => {});
        }
        const child = Bun.spawn(['git', '-c', 'core.hooksPath=/dev/null', ...args], { cwd, env: { ...Bun.env, ...env }, stdout: 'pipe', stderr: 'pipe' });
        const [code, output, error] = await Promise.all([child.exited, new Response(child.stdout).text(), new Response(child.stderr).text()]);
        if (code) throw new Error(error);
        return raw ? output : output.trim();
      }});
    `)
    const worker = Bun.spawn([process.execPath, workerPath], { stdout: 'pipe', stderr: 'pipe' })
    const [code, , error] = await Promise.all([worker.exited, new Response(worker.stdout).text(), new Response(worker.stderr).text()])
    expect(code).not.toBe(0)
    expect(error).toBe('')
    const lockPath = join(f.storage, 'integration.lock')
    const owner = await Bun.file(lockPath).json()
    expect(owner.pid).toBe(worker.pid)
    expect(await Bun.file(join(f.root, 'a.txt')).text()).toBe(phase === 'preparation' ? 'original a\n' : 'agent a\n')
    if (phase === 'recovery') {
      await Bun.write(workerPath, `
        import { recoverWorkspaceApply } from ${JSON.stringify(modulePath)};
        await recoverWorkspaceApply({
          ...${JSON.stringify({ storage: f.storage, destination: f.root, id: owner.id })},
          git: async () => { process.kill(process.pid, 'SIGKILL'); await new Promise(() => {}); return ''; }
        });
      `)
      const recovery = Bun.spawn([process.execPath, workerPath], { stdout: 'pipe', stderr: 'pipe' })
      const [exitCode, , stderr] = await Promise.all([recovery.exited, new Response(recovery.stdout).text(), new Response(recovery.stderr).text()])
      expect(exitCode).not.toBe(0)
      expect(stderr).toBe('')
      expect((await Bun.file(lockPath).json()).pid).toBe(recovery.pid)
    }
    expect((await f.port.integrations()).records).toEqual(expect.arrayContaining([expect.objectContaining({ id: owner.id, status: phase === 'preparation' ? 'preparing' : 'prepared' })]))
    expect(await f.port.recoverIntegration(owner.id)).toMatchObject({ status: phase === 'preparation' ? 'abandoned' : 'rolled-back', conflicts: [] })
    expect(await Bun.file(join(f.root, 'a.txt')).text()).toBe('original a\n')
    expect(await Bun.file(lockPath).exists()).toBe(false)
    if (phase === 'preparation') expect(await f.port.apply(f.id, review.reviewId, check.destinationState!)).toMatchObject({ status: 'applied' })
  } finally { await rm(f.root, { recursive: true, force: true }) }
})

test('recovery serializes callers without blocking the event loop and leaves the live owner intact', async () => {
  const f = await fixture()
  const entered = Promise.withResolvers<void>(), release = Promise.withResolvers<void>()
  let running: Promise<unknown> | undefined
  try {
    const review = await f.port.inspect(f.id), check = await f.port.checkApply(f.id, review.reviewId)
    const applied = await f.port.apply(f.id, review.reviewId, check.destinationState!)
    const recordPath = join(applied.backupPath, 'record.json')
    await Bun.write(recordPath, JSON.stringify({ ...await Bun.file(recordPath).json(), status: 'prepared' }))
    running = recoverWorkspaceApply({ storage: f.storage, destination: f.root, id: applied.id, git: async (...args) => { entered.resolve(); await release.promise; return f.git(...args) } })
    await entered.promise
    await expect(f.port.recoverIntegration(applied.id)).rejects.toThrow('Another recovery is active')
    expect((await Bun.file(join(f.storage, 'integration.lock')).json()).pid).toBe(process.pid)
    release.resolve()
    expect(await running).toMatchObject({ status: 'rolled-back' })
  } finally { release.resolve(); await running; await rm(f.root, { recursive: true, force: true }) }
})

test.each(['applied', 'rolled-back', 'abandoned'])('terminal integration %s releases a dead lock without changing newer files', async status => {
  const f = await fixture()
  try {
    const review = await f.port.inspect(f.id), check = await f.port.checkApply(f.id, review.reviewId)
    const applied = await f.port.apply(f.id, review.reviewId, check.destinationState!)
    const recordPath = join(applied.backupPath, 'record.json')
    const record = await Bun.file(recordPath).json()
    await Bun.write(recordPath, JSON.stringify({ ...record, status, ...(status === 'abandoned' ? { original: [], expected: [] } : {}) }))
    await Bun.write(join(f.root, 'a.txt'), 'newer work after completion\n')
    const worker = Bun.spawn([process.execPath, '-e', 'process.exit(0)'], { stdout: 'ignore', stderr: 'ignore' })
    await worker.exited
    const lockPath = join(f.storage, 'integration.lock')
    await Bun.write(lockPath, JSON.stringify({ id: applied.id, pid: worker.pid, host: hostname() }))
    expect(await f.port.recoverIntegration(applied.id)).toMatchObject({ status, conflicts: [] })
    expect(await Bun.file(join(f.root, 'a.txt')).text()).toBe('newer work after completion\n')
    expect(await Bun.file(join(f.root, 'b.txt')).text()).toBe('agent b\n')
    expect(await Bun.file(lockPath).exists()).toBe(false)
    expect(await Bun.file(recordPath).json()).toMatchObject({ status })
  } finally { await rm(f.root, { recursive: true, force: true }) }
})

test('recovery inspection classifies original, applied and concurrent content without writing', async () => {
  const f = await fixture()
  try {
    const review = await f.port.inspect(f.id), check = await f.port.checkApply(f.id, review.reviewId)
    const applied = await f.port.apply(f.id, review.reviewId, check.destinationState!)
    const recordPath = join(applied.backupPath, 'record.json')
    await Bun.write(recordPath, JSON.stringify({ ...await Bun.file(recordPath).json(), status: 'prepared' }))
    await Bun.write(join(f.root, 'b.txt'), 'original b\n')
    expect((await f.port.inspectIntegration(applied.id)).files).toMatchObject([{ path: 'a.txt', action: 'restore' }, { path: 'b.txt', action: 'unchanged' }])
    await Bun.write(join(f.root, 'b.txt'), 'concurrent change\n')
    expect((await f.port.inspectIntegration(applied.id)).files).toMatchObject([{ action: 'restore' }, { action: 'conflict' }])
    expect(await Bun.file(join(f.root, 'a.txt')).text()).toBe('agent a\n')
    expect(await Bun.file(join(f.root, 'b.txt')).text()).toBe('concurrent change\n')
    await Bun.write(join(applied.backupPath, 'original-0'), 'corrupt')
    await expect(f.port.inspectIntegration(applied.id)).rejects.toThrow('checksum mismatch')
  } finally { await rm(f.root, { recursive: true, force: true }) }
})
