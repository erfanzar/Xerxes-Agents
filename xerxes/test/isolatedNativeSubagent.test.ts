// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, realpath, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { getActiveSession, runWithActiveSession } from '../src/runtime/sessionContext.js'
import { nativeSubagentWorktrees } from '../src/runtime/subagentWorktrees.js'
import { registerFileTools } from '../src/tools/fileTools.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import { BUILTIN_AGENTS } from '../src/agents/definitions.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'

async function git(cwd: string, ...args: string[]) {
  const child = Bun.spawn(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', ...args], { cwd, stdout: 'pipe', stderr: 'pipe' })
  const [code, error] = await Promise.all([child.exited, new Response(child.stderr).text(), new Response(child.stdout).text()])
  if (code !== 0) throw new Error(error)
}

test('concurrent native allocations keep the shared Git worktree registry consistent across adapters', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-native-allocation-')))
  try {
    // A wider fan-out exposes Git reading a sibling's partially registered
    // worktree before its commondir file exists. Keep real Git and separate
    // adapters: the registry belongs to the repository, not one host instance.
    for (let round = 0; round < 5; round++) {
      const repository = join(root, String(round))
      await git(root, 'init', repository)
      await Bun.write(join(repository, 'parent.txt'), 'parent content')
      await git(repository, 'add', '.')
      await git(repository, 'commit', '-m', 'fixture baseline')
      const ports = [nativeSubagentWorktrees(repository), nativeSubagentWorktrees(repository)]
      const results = await Promise.allSettled(Array.from({ length: 32 }, (_, child) => ports[child % ports.length]!.create({ taskId: `child-${child}`, taskName: `Child ${child}` })))
      expect(results.flatMap(result => result.status === 'rejected' ? [String(result.reason)] : [])).toEqual([])
      const trees = results.flatMap(result => result.status === 'fulfilled' ? [result.value] : [])
      expect(new Set(trees.map(tree => tree.path)).size).toBe(32)
      for (const tree of trees) expect(await Bun.file(join(tree.path, 'parent.txt')).text()).toBe('parent content')
    }
  } finally { await rm(root, { recursive: true, force: true }) }
}, 15_000)

test('concurrent native children bind real file tools to their own worktrees', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-native-isolation-')))
  const tools = new ToolRegistry()
  registerFileTools(tools, new WorkspacePathResolver(root, () => getActiveSession<{ cwd: string }>()?.cwd))
  const calls = new Map<string, number>()
  const bothChildrenStarted = Promise.withResolvers<void>()
  const transcripts = new DaemonTranscriptStore({ directory: join(root, 'sessions'), currentProjectDirectory: root })
  const host = createNativeSubagentHost({ transcriptStore: transcripts, agentDefinitions: BUILTIN_AGENTS, cwd: root, worktree: nativeSubagentWorktrees(root), model: 'gpt-4o', permissionMode: 'accept-all', eventBus: new DaemonSubagentEventBus(), tools: tools.definitions(), toolExecutor: tools,
    llm: { async *stream() {
      const cwd = getActiveSession<{ cwd: string }>()?.cwd
      expect(cwd).toBeDefined()
      expect(cwd).not.toBe(root)
      expect(await Bun.file(join(cwd!, 'parent.txt')).text()).toBe('parent content')
      const round = calls.get(cwd!) ?? 0
      calls.set(cwd!, round + 1)
      // Registry allocation may serialize, but both real child turns must
      // still overlap before either starts exercising its bound file tools.
      if (calls.size >= 2) bothChildrenStarted.resolve()
      await bothChildrenStarted.promise
      if (round < 2) yield { toolCalls: [{ id: 'write-' + round, type: 'function', function: { name: 'WriteFile', arguments: { file_path: round === 0 ? 'child.txt' : join(root, 'escaped.txt'), content: cwd! } } }] }
      else yield { content: 'Isolated file created; parent path denied.' }
    } } })
  try {
    await git(root, 'init')
    await Bun.write(join(root, 'parent.txt'), 'parent content')
    await git(root, 'add', '.')
    await git(root, 'commit', '-m', 'fixture baseline')
    await git(root, 'tag', 'baseline')
    await Bun.write(join(root, 'parent.txt'), 'new parent content')
    await git(root, 'commit', '-am', 'new parent fixture')
    await runWithActiveSession({ cwd: root }, async () => {
      const tasks = await Promise.all(['one', 'two'].map(name => host.managerPort.spawn({ message: 'Exercise isolated file tools', nickname: name, isolation: 'worktree', worktreeRef: 'baseline' })))
      for (const task of tasks) {
        const settled = await host.manager.wait(task.id, 3000)
        expect(settled?.error).toBe('')
        expect(settled?.status).toBe('completed')
      }
      expect(calls.size).toBe(2)
      for (const snapshot of tasks) {
        expect(snapshot.rules).toContain('isolation:worktree')
        expect(snapshot.rules).toContain('worktree-ref:baseline')
        const task = host.manager.listTasks().find(task => task.id === snapshot.id)!
        expect(task.status).toBe('completed')
        expect(await Bun.file(join(task.worktreePath, 'child.txt')).text()).toBe(task.worktreePath)
      }
      expect(getActiveSession<{ cwd: string }>()?.cwd).toBe(root)
    })
    expect(await Bun.file(join(root, 'child.txt')).exists()).toBe(false)
    expect(await Bun.file(join(root, 'escaped.txt')).exists()).toBe(false)
    expect(await Bun.file(join(root, 'parent.txt')).text()).toBe('new parent content')
    const saved = await transcripts.list()
    expect(saved).toHaveLength(2)
    expect(saved.map(transcript => transcript.cwd).sort()).toEqual([...calls.keys()].sort())
    expect(saved.every(transcript => transcript.metadata.project_root === root)).toBe(true)
    const first = host.managerPort.listHandles()[0]!
    const originalPaths = [...calls.keys()]
    await host.retry(first.id, { message: 'Retry isolated work' })
    await host.manager.wait(first.id, 3000)
    expect(calls.size).toBe(3)
    const retried = host.manager.listTasks().find(task => task.id === first.id)!
    expect(retried.status).toBe('completed')
    expect(originalPaths).not.toContain(retried.worktreePath)
    expect(await Bun.file(join(retried.worktreePath, 'child.txt')).text()).toBe(retried.worktreePath)
    for (const path of originalPaths) expect(await Bun.file(join(path, 'child.txt')).exists()).toBe(true)
  } finally { bothChildrenStarted.resolve(); await host.manager.shutdown(); await rm(root, { recursive: true, force: true }) }
})

test('project switches during allocation and later retries keep their original worktree owner', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-isolation-projects-')))
  const a = join(root, 'a'), b = join(root, 'b')
  const entered = Promise.withResolvers<void>(), release = Promise.withResolvers<void>()
  let allocations = 0
  const factory = (cwd: string) => {
    const native = nativeSubagentWorktrees(cwd)
    return { ...native, async create(request: Parameters<typeof native.create>[0]) {
      if (++allocations === 1) { entered.resolve(); await release.promise }
      return native.create(request)
    } }
  }
  const bus = new DaemonSubagentEventBus()
  const observations: string[] = []
  const options = (cwd: string) => ({ cwd, worktreeForWorkspace: factory, agentDefinitions: BUILTIN_AGENTS, model: 'gpt-4o', permissionMode: 'accept-all' as const, eventBus: bus, tools: [], toolExecutor: new ToolRegistry(), llm: { async *stream() {
    const child = getActiveSession<{ cwd: string }>()!.cwd
    observations.push(await Bun.file(join(child, 'owner.txt')).text())
    yield { content: 'done' }
  } } })
  const host = createNativeSubagentHost(options(a))
  try {
    for (const cwd of [a, b]) {
      await mkdir(cwd)
      await git(cwd, 'init')
      await Bun.write(join(cwd, 'owner.txt'), cwd)
      await git(cwd, 'add', '.')
      await git(cwd, 'commit', '-m', 'fixture baseline')
      await Bun.write(join(cwd, 'owner.txt'), cwd + '-dirty')
    }
    const first = host.managerPort.spawn({ message: 'check owner', isolation: 'worktree', worktreeSource: 'working-tree' })
    await entered.promise
    host.reconfigure(options(b))
    release.resolve()
    const old = await first
    await host.manager.wait(old.id, 3000)
    expect(observations).toEqual([a + '-dirty'])
    const next = await host.managerPort.spawn({ message: 'check owner', isolation: 'worktree', worktreeSource: 'working-tree' })
    await host.manager.wait(next.id, 3000)
    expect(observations).toEqual([a + '-dirty', b + '-dirty'])
    host.reconfigure(options(b))
    await host.retry(old.id, { message: 'check original owner again' })
    await host.manager.wait(old.id, 3000)
    expect(observations).toEqual([a + '-dirty', b + '-dirty', a + '-dirty'])
    expect(host.manager.listTasks().every(task => task.status === 'completed')).toBe(true)
  } finally { release.resolve(); await host.manager.shutdown(); await rm(root, { recursive: true, force: true }) }
})
