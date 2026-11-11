// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { nativeSubagentWorktrees } from '../src/runtime/subagentWorktrees.js'
import { SubAgentManager } from '../src/agents/subagentManager.js'
import { loadWorkspaceSetup, runWorkspaceSetup, type WorkspaceSetupResult } from '../src/runtime/workspaceSetup.js'

test('configured setup runs in its checkout and retained failure output is inspectable', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-setup-')))
  const git = async (...args: string[]) => {
    const p = Bun.spawn(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', ...args], { cwd: root, stdout: 'pipe', stderr: 'pipe' })
    const [code, , error] = await Promise.all([p.exited, new Response(p.stdout).text(), new Response(p.stderr).text()]); if (code) throw new Error(error)
  }
  try {
    await git('init'); await Bun.write(join(root, 'file.txt'), 'parent'); await git('add', '.'); await git('commit', '-m', 'fixture')
    const config = join(root, 'setup-policy.json')
    await Bun.write(config, JSON.stringify({ workspaces: { [root]: { command: [process.execPath, '-e', 'await Bun.write("generated.txt",process.cwd()); console.log("setup ready")'] } } }))
    const port = nativeSubagentWorktrees(root, { setupConfigPath: config })
    const tree = await port.create({ taskId: 'ok', taskName: 'Setup' })
    expect(await Bun.file(join(tree.path, 'generated.txt')).text()).toBe(tree.path)
    expect(await Bun.file(join(root, 'generated.txt')).exists()).toBe(false)
    expect((await port.inspect(tree.branch.slice('xerxes/agent-'.length))).setup).toContain('setup ready')
    await Bun.write(config, JSON.stringify({ workspaces: { [root]: { command: [process.execPath, '-e', 'console.error("setup failure evidence");process.exit(7)'] } } }))
    await expect(port.create({ taskId: 'failed', taskName: 'Failure' })).rejects.toThrow('retained checkout')
    const failed = (await port.list()).records.find(record => record.taskId === 'failed')!
    expect((await port.inspect(failed.id)).setup).toContain('setup failure evidence')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('setup configuration rejects invalid commands and timeout before execution', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-setup-config-'))
  try {
    const path = join(root, 'config.json')
    for (const policy of [{ command: ['bun;echo'] }, { command: ['bun'], timeout_ms: 0 }]) {
      await Bun.write(path, JSON.stringify({ workspaces: { [root]: policy } }))
      await expect(loadWorkspaceSetup(root, path)).rejects.toThrow()
    }
    expect(await loadWorkspaceSetup('/unconfigured', path)).toBeUndefined()
  } finally { await rm(root, { recursive: true, force: true }) }
})

test.each(['timeout', 'cancel'])('setup %s records failure and never backgrounds', async mode => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-setup-stop-'))
  const controller = new AbortController(), records: WorkspaceSetupResult[] = []
  try {
    const running = runWorkspaceSetup({ command: [process.execPath, '-e', 'await Bun.write("started","yes");setInterval(()=>{},1000)'], timeoutMs: mode === 'timeout' ? 30 : 2000 }, root, controller.signal, async result => { records.push(result) })
    if (mode === 'cancel') {
      const deadline = Date.now() + 1000
      while (!await Bun.file(join(root, 'started')).exists() && Date.now() < deadline) await Bun.sleep(10)
      expect(await Bun.file(join(root, 'started')).exists()).toBe(true)
      controller.abort(new Error('cancel fixture'))
    }
    await expect(running).rejects.toThrow()
    expect(records.at(-1)?.status).toBe('failed')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('failed setup can retry the same task in a fresh checkout while preserving failure evidence', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-setup-retry-')))
  let manager: InstanceType<typeof SubAgentManager> | undefined
  try {
    const git = Bun.spawn(['git', 'init', root], { stdout: 'pipe', stderr: 'pipe' })
    await Promise.all([git.exited, new Response(git.stdout).text(), new Response(git.stderr).text()])
    const commit = Bun.spawn(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-m', 'fixture'], { cwd: root, stdout: 'pipe', stderr: 'pipe' })
    const [code] = await Promise.all([commit.exited, new Response(commit.stdout).text(), new Response(commit.stderr).text()])
    expect(code).toBe(0)
    const config = join(root, 'setup.json')
    const configure = (script: string) => Bun.write(config, JSON.stringify({ workspaces: { [root]: { command: [process.execPath, '-e', script] } } }))
    await configure('await Bun.write("failed-marker","keep"); process.exit(2)')
    const port = nativeSubagentWorktrees(root, { setupConfigPath: config })
    const runs: string[] = []
    manager = new SubAgentManager({ worktree: port, runner: async request => { runs.push(request.worktree!.path); return 'done' } })
    const task = await manager.spawn({ name: 'setup-retry', prompt: 'work', isolation: 'worktree' })
    expect(task.status).toBe('failed')
    expect(runs).toHaveLength(0)
    const failedPath = task.worktreePath
    expect(await Bun.file(join(failedPath, 'failed-marker')).text()).toBe('keep')
    await expect(manager.retry(task.id)).rejects.toThrow('Workspace setup failed')
    expect(task.status).toBe('failed')
    expect(task.worktreePath).not.toBe(failedPath)
    expect(task.error).toContain(task.worktreePath)
    expect(runs).toHaveLength(0)
    await configure('await Bun.write("ready-marker","ready")')
    const retried = await manager.retry(task.id)
    expect(retried?.id).toBe(task.id)
    await manager.wait(task.id, 1000)
    expect(task.status).toBe('completed')
    expect(runs).toHaveLength(1)
    expect(runs[0]).not.toBe(failedPath)
    expect(await Bun.file(join(runs[0]!, 'ready-marker')).text()).toBe('ready')
    expect(await Bun.file(join(failedPath, 'failed-marker')).text()).toBe('keep')
  } finally { await manager?.close(); await rm(root, { recursive: true, force: true }) }
})

test('sandbox-bound worktree allocation disables repository setup and never parses its command policy', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-setup-disabled-')))
  try {
    for (const args of [['init', '--template='], ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-m', 'fixture']]) {
      expect(Bun.spawnSync(['git', ...args], { cwd: root, stdout: 'pipe', stderr: 'pipe' }).exitCode).toBe(0)
    }
    const config = join(root, 'setup-policy.json')
    await Bun.write(config, JSON.stringify({ workspaces: { [root]: { command: [process.execPath, '-e', 'await Bun.write("UNSANDBOXED-SETUP","forbidden")'] } } }))
    const port = nativeSubagentWorktrees(root, { setupConfigPath: config, setupPolicy: 'disabled' })
    const tree = await port.create({ taskId: 'scoped-execution', taskName: 'sandbox worktree' })
    expect(await Bun.file(join(tree.path, 'UNSANDBOXED-SETUP')).exists()).toBe(false)
    expect(await port.isClean(tree)).toBe(true)
    await Bun.write(config, 'invalid policy must not even be parsed')
    const second = await port.create({ taskId: 'second-scoped-execution', taskName: 'sandbox worktree' })
    expect(await port.isClean(second)).toBe(true)
  } finally { await rm(root, { recursive: true, force: true }) }
})
