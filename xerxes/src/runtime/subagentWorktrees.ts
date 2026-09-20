// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { applyWorkspacePatch, inspectDestination, recoverWorkspaceApply, listWorkspaceIntegrations, inspectWorkspaceIntegration, type WorkspaceApplyResult } from './workspaceApply.js'
import { createHash } from 'node:crypto'
import { parseWorktreeRef, parseWorktreeSource, WorktreeSetupError } from '../agents/worktreeOptions.js'
import { mkdir, realpath, unlink, readdir, lstat, rename } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import type { SubagentWorktree, SubagentWorktreePort } from '../agents/subagentManager.js'
import { loadWorkspaceSetup, runWorkspaceSetup } from './workspaceSetup.js'

interface Ownership { id: string; taskId: string; path: string; branch: string; base: string; snapshotTree?: string }

export interface AgentWorkspaceRecord {
  readonly id: string
  readonly taskId?: string
  readonly path: string
  readonly branch: string
  readonly error?: string
}
export interface AgentWorkspaceReview extends AgentWorkspaceRecord {
  readonly setup?: string
  readonly reviewId: string
  readonly base: string
  readonly snapshotTree?: string
  readonly head: string
  readonly status: string
  readonly diff: string
}
export interface AgentWorkspaceApplyCheck {
  readonly destinationState?: string
  readonly reviewId: string
  readonly destination: string
  readonly destinationHead: string
  readonly canApply: boolean
  readonly checkedAt: string
  readonly error?: string
}
export interface ManagedAgentWorktrees extends SubagentWorktreePort {
  list(after?: string): Promise<{ records: readonly AgentWorkspaceRecord[]; next?: string }>
  inspect(id: string): Promise<AgentWorkspaceReview>
  checkApply(id: string, reviewId: string): Promise<AgentWorkspaceApplyCheck>
  apply(id: string, reviewId: string, destinationState: string): Promise<WorkspaceApplyResult>
  recoverIntegration(id: string): ReturnType<typeof recoverWorkspaceApply>
  integrations(after?: string): ReturnType<typeof listWorkspaceIntegrations>
  inspectIntegration(id: string): ReturnType<typeof inspectWorkspaceIntegration>
}

/** Git-only isolation adapter. No shared-checkout fallback and no automatic commits. */
export function nativeSubagentWorktrees(repository: string, options: { setupConfigPath?: string } = {}): ManagedAgentWorktrees {
  const cwd = resolve(repository)
  const gitEnvironment: Record<string, string | undefined> = { ...Bun.env, GIT_OPTIONAL_LOCKS: '0' }
  for (const name of ['GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE', 'GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_PREFIX']) delete gitEnvironment[name]
  const git = async (args: readonly string[], directory = cwd, environment: Record<string, string> = {}, raw = false, signal?: AbortSignal): Promise<string> => {
    signal?.throwIfAborted()
    const process = Bun.spawn(['git', '-c', 'core.hooksPath=/dev/null', ...args], { cwd: directory, env: { ...gitEnvironment, ...environment }, stdout: 'pipe', stderr: 'pipe', signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(60_000)]) : AbortSignal.timeout(60_000) })
    const collect = async (stream: ReadableStream<Uint8Array>) => {
      const reader = stream.getReader()
      const decoder = new TextDecoder()
      let bytes = 0, text = ''
      try {
        for (;;) {
          const chunk = await reader.read()
          if (chunk.done) return text + decoder.decode()
          bytes += chunk.value.byteLength
          if (bytes > 1_048_576) throw new Error('Agent worktree Git output exceeds 1 MiB; preserve the checkout for inspection')
          text += decoder.decode(chunk.value, { stream: true })
        }
      } finally { reader.releaseLock() }
    }
    let result: [number, string, string]
    try { result = await Promise.all([process.exited, collect(process.stdout), collect(process.stderr)]) }
    catch (error) { process.kill(); await process.exited; throw error }
    const [code, output, error] = result
    if (code !== 0) throw new Error(`Agent worktree Git operation failed: ${error.trim().slice(0, 4000) || `exit ${code}`}`)
    return raw ? output : output.trim()
  }
  const storage = async (create = true) => {
    const common = await git(['rev-parse', '--path-format=absolute', '--git-common-dir'])
    const root = join(await realpath(common), 'xerxes-agent-worktrees')
    if (create) await mkdir(root, { recursive: true, mode: 0o700 })
    // A redirected ownership directory cannot authorize cleanup elsewhere.
    const canonical = await realpath(root).catch(error => { if (!create && (error as NodeJS.ErrnoException).code === 'ENOENT') return root; throw error })
    if (canonical !== root) throw new Error('Agent worktree ownership directory must not be a symlink')
    return root
  }
  const ownership = async (tree: SubagentWorktree): Promise<{ record: Ownership; manifest: string }> => {
    const root = await storage(false)
    const id = tree.branch.replace(/^xerxes\/agent-/, '')
    if (!/^[a-f0-9-]{36}$/.test(id) || tree.branch !== `xerxes/agent-${id}` || tree.path !== join(root, id)) throw new Error('Worktree is not owned by this agent host')
    const manifest = join(root, `${id}.json`)
    const info = await lstat(manifest)
    if (!info.isFile() || info.size > 65536) throw new Error('Invalid agent worktree manifest file')
    const record = await Bun.file(manifest).json() as Partial<Ownership>
    if (record.id !== id || record.path !== tree.path || record.branch !== tree.branch || typeof record.taskId !== 'string' || typeof record.base !== 'string' || !/^[a-f0-9]{40,64}$/.test(record.base)) throw new Error('Invalid agent worktree ownership record')
    if (record.snapshotTree !== undefined && (typeof record.snapshotTree !== 'string' || !/^[a-f0-9]{40,64}$/.test(record.snapshotTree))) throw new Error('Invalid captured workspace tree')
    if (await realpath(tree.path) !== tree.path || await git(['rev-parse', '--show-toplevel'], tree.path) !== tree.path || await git(['branch', '--show-current'], tree.path) !== tree.branch) throw new Error('Agent worktree identity changed; preserve it for review')
    return { record: record as Ownership, manifest }
  }
  const clean = async (tree: SubagentWorktree, record: Ownership) => {
    if (await git(['rev-parse', 'HEAD'], tree.path) !== record.base) return false
    // Ignored files may be valuable setup/results; they also prevent removal.
    return (await git(['status', '--porcelain', '--untracked-files=all', '--ignored'], tree.path)) === ''
  }
  const service: ManagedAgentWorktrees = {
    async list(after) {
      if (after !== undefined && !/^[a-f0-9-]{36}$/.test(after)) throw new Error('Invalid workspace cursor')
      const root = await storage(false)
      const names = await readdir(root).catch(error => { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return [] as string[]; throw error })
      if (names.length > 4096) throw new Error('Workspace inventory exceeds 4096 entries; inspect a specific workspace ID')
      const results: AgentWorkspaceRecord[] = []
      const selected = names.filter(name => /^[a-f0-9-]{36}\.json$/.test(name) && (after === undefined || name.slice(0, -5) > after)).sort()
      for (const name of selected.slice(0, 100)) {
        if (!/^[a-f0-9-]{36}\.json$/.test(name)) continue
        const id = name.slice(0, -5)
        const tree = { path: join(root, id), branch: 'xerxes/agent-' + id }
        try {
          const { record } = await ownership(tree)
          results.push({ id, taskId: record.taskId, ...tree })
        } catch (error) { results.push({ id, ...tree, error: error instanceof Error ? error.message : String(error) }) }
      }
      return { records: results, ...(selected.length > 100 ? { next: results.at(-1)!.id } : {}) }
    },
    async inspect(id) {
      if (!/^[a-f0-9-]{36}$/.test(id)) throw new Error('Workspace ID must be the allocation UUID from /workspaces')
      const root = await storage(false)
      const tree = { path: join(root, id), branch: 'xerxes/agent-' + id }
      const { record } = await ownership(tree)
      const baseline = record.snapshotTree ?? record.base
      const head = await git(['rev-parse', 'HEAD'], tree.path)
      const status = await git(['status', '--porcelain', '--untracked-files=all', '--ignored'], tree.path)
      const index = join(root, id + '.review-index-' + crypto.randomUUID())
      const environment = { GIT_INDEX_FILE: index }
      let diff: string
      try {
        await git(['read-tree', baseline], tree.path, environment)
        await git(['add', '--intent-to-add', '-A', '--', '.'], tree.path, environment)
        diff = await git(['diff', '--no-renames', '--binary', '--no-color', '--no-ext-diff', '--no-textconv', baseline, '--'], tree.path, environment, true)
      } finally {
        await unlink(index).catch(error => { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error })
      }
      const reviewId = createHash('sha256').update(JSON.stringify({ id, base: record.base, snapshotTree: record.snapshotTree, head, diff })).digest('hex')
      const setupFile = Bun.file(join(root, `${id}.setup.json`))
      let setup: string | undefined
      if (await setupFile.exists()) {
        if (setupFile.size > 65536) throw new Error('Workspace setup record exceeds 64 KiB')
        const result = await setupFile.json() as Record<string, unknown>
        setup = ['status', 'error', 'stdout', 'stderr'].filter(key => typeof result[key] === 'string').map(key => `${key}: ${key === 'status' && result[key] === 'running' ? 'incomplete; completion not recorded' : String(result[key]).slice(0, 4000)}`).join('\n')
      }
      return { id, reviewId, taskId: record.taskId, ...tree, base: record.base, ...(record.snapshotTree ? { snapshotTree: record.snapshotTree } : {}), ...(setup === undefined ? {} : { setup }), head, status, diff }
    },
    async checkApply(id, reviewId) {
      if (!/^[a-f0-9]{64}$/.test(reviewId)) throw new Error('Refresh the workspace review before checking integration')
      const review = await service.inspect(id)
      if (review.reviewId !== reviewId) throw new Error('Agent changes changed since review; refresh before checking integration')
      const destination = await git(['rev-parse', '--show-toplevel'])
      const destinationHead = await git(['rev-parse', 'HEAD'])
      const result = { reviewId, destination, destinationHead, checkedAt: new Date().toISOString() }
      if (destination === review.path) throw new Error('Cannot integrate a workspace into itself')
      if (!review.diff) return { ...result, canApply: false, error: 'No changes from starting state' }
      const root = await storage(false)
      const patch = join(root, id + '.apply-check-' + crypto.randomUUID() + '.patch')
      try {
        await Bun.write(patch, review.diff, { mode: 0o600 })
        try {
          await git(['apply', '--check', '--whitespace=nowarn', patch], destination)
          const snapshot = await inspectDestination(git, destination, patch)
          return { ...result, destinationHead: snapshot.head, destinationState: snapshot.state, canApply: true }
        } catch (error) {
          return { ...result, canApply: false, error: error instanceof Error ? error.message : String(error) }
        }
      } finally { await unlink(patch).catch(error => { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }) }
    },
    async apply(id, reviewId, destinationState) {
      const review = await service.inspect(id)
      if (review.reviewId !== reviewId) throw new Error('Agent changes changed since review; refresh before applying')
      if (!/^[a-f0-9]{64}$/.test(destinationState)) throw new Error('Check integration before applying changes')
      const destination = await git(['rev-parse', '--show-toplevel'])
      if (destination === review.path) throw new Error('Cannot integrate a workspace into itself')
      return applyWorkspacePatch({ git, destination, storage: await storage(false), patch: review.diff, destinationState, reviewId })
    },
    async recoverIntegration(id) {
      return recoverWorkspaceApply({ git, destination: await git(['rev-parse', '--show-toplevel']), storage: await storage(false), id })
    },
    async integrations(after) {
      return listWorkspaceIntegrations(await storage(false), await git(['rev-parse', '--show-toplevel']), after)
    },
    async inspectIntegration(id) {
      return inspectWorkspaceIntegration(git, await storage(false), await git(['rev-parse', '--show-toplevel']), id)
    },
    async create(request) {
      request.signal?.throwIfAborted()
      const allocateGit = (args: readonly string[], directory = cwd, environment: Record<string, string> = {}, raw = false) => git(args, directory, environment, raw, request.signal)
      if (!request.taskId.trim()) throw new Error('Agent worktree requires a task identity')
      const selectedRef = parseWorktreeRef(request.config?._nativeSubagentWorktreeRef)
      const source = parseWorktreeSource(request.config?._nativeSubagentWorktreeSource)
      if (source && selectedRef) throw new Error('worktree_source and worktree_ref are mutually exclusive')
      const ref = selectedRef ?? 'HEAD'
      const base = await allocateGit(['rev-parse', '--verify', '--end-of-options', `${ref}^{commit}`])
      const setup = await loadWorkspaceSetup(await allocateGit(['rev-parse', '--show-toplevel']), options.setupConfigPath)
      const root = await storage()
      const id = crypto.randomUUID()
      const record: Ownership = { id, taskId: request.taskId, path: join(root, id), branch: `xerxes/agent-${id}`, base }
      if (source) {
        const sourceRoot = await allocateGit(['rev-parse', '--show-toplevel'])
        const index = join(root, id + '.capture-index')
        const environment = { GIT_INDEX_FILE: index }
        try {
          await allocateGit(['read-tree', base], sourceRoot, environment)
          await allocateGit(['add', '-A', '--', '.'], sourceRoot, environment)
          const snapshotTree = await allocateGit(['write-tree'], sourceRoot, environment)
          // The NUL-delimited standard format also works with older Git on SSH hosts.
          const entries = await allocateGit(['ls-tree', '-r', '-z', snapshotTree], sourceRoot)
          if (entries.split('\0').some(entry => entry.startsWith('160000 '))) throw new Error('Working-tree capture does not support submodules or nested Git repositories; use a committed revision')
          await allocateGit(['diff', '--quiet', '--ignore-submodules=none', '--'], sourceRoot, environment)
          if (await allocateGit(['ls-files', '--others', '--exclude-standard'], sourceRoot, environment)) throw new Error('Working tree changed during capture; retry when edits settle')
          if (await allocateGit(['rev-parse', 'HEAD'], sourceRoot) !== base) throw new Error('HEAD changed during working-tree capture; retry when edits settle')
          record.snapshotTree = snapshotTree
        } finally {
          // Only this unique temporary index is touched; the parent's index is never used for writes.
          await unlink(index).catch(error => { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error })
        }
      }
      // Write ownership before Git can create resources. Failed setup leaves
      // its record and any partial checkout available for explicit recovery.
      await Bun.write(join(root, `${id}.json`), JSON.stringify(record), { mode: 0o600 })
      await allocateGit(['worktree', 'add', '-b', record.branch, record.path, base])
      if (record.snapshotTree) {
        await allocateGit(['read-tree', '--reset', '-u', record.snapshotTree], record.path)
        // Copied edits remain ordinary working files, visible to git diff; do not
        // import the parent staging selection into an independent agent index.
        await allocateGit(['read-tree', base], record.path)
      }
      if (setup) {
        try { await runWorkspaceSetup(setup, record.path, request.signal, async result => {
          const temporary = join(root, `${id}.setup-${crypto.randomUUID()}.tmp`)
          await Bun.write(temporary, JSON.stringify(result), { mode: 0o600 })
          await rename(temporary, join(root, `${id}.setup.json`))
        }) }
        catch (error) { throw new WorktreeSetupError(`Workspace setup failed; retained checkout ${record.path}; details ${join(root, `${id}.setup.json`)}`, { path: record.path, branch: record.branch }, error) }
      }
      return { path: record.path, branch: record.branch }
    },
    async isClean(tree) { const { record } = await ownership(tree); return clean(tree, record) },
    async remove(tree) {
      const { record, manifest } = await ownership(tree)
      if (!await clean(tree, record)) throw new Error('Agent worktree has files or commits to review; refusing cleanup')
      await git(['worktree', 'remove', tree.path])
      await unlink(manifest)
      // Keep the branch reference. Cleanup never deletes unmerged history.
    },
  }
  return service
}
