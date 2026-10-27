// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { realpath } from 'node:fs/promises'
import { isAbsolute } from 'node:path'

/** No credentials, protocol or transport-specific port participate in identity. */
export function normalizeRepoRemote(remote: string): string | undefined {
  const text = remote.trim().replace(/\/+$/u, '')
  if (!text || text.startsWith('.') || isAbsolute(text)) return undefined
  let host: string; let path: string
  if (/^[a-z][a-z\d+.-]*:\/\//iu.test(text)) {
    try {
      const url = new URL(text)
      if (!url.hostname) return undefined
      host = url.hostname.toLowerCase(); path = url.pathname
    } catch { return undefined }
  } else {
    const match = /^(?:[^/@]+@)?(\[[^\]]+\]|[^/:]+)(?::\d+)?:([\s\S]+)$/u.exec(text)
    if (!match) return undefined
    host = match[1]!.toLowerCase(); path = match[2]!
  }
  path = path.replace(/^\/+|\/+$/gu, '').replace(/\.git$/u, '')
  return host && path ? `${host}/${path}` : undefined
}

/** Bounded, non-interactive git invocation. Missing git/repositories return no result. */
export async function lessonGit(root: string, args: readonly string[]): Promise<{ code: number; text: string } | undefined> {
  let process: Bun.Subprocess<'ignore', 'pipe', 'pipe'>
  try {
    process = Bun.spawn(['git', '-C', root, ...args], {
      stdin: 'ignore', stdout: 'pipe', stderr: 'pipe',
      env: { ...Bun.env, GIT_TERMINAL_PROMPT: '0', GIT_OPTIONAL_LOCKS: '0' },
    })
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') return undefined
    throw error
  }
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<undefined>(resolve => {
    timer = setTimeout(() => { process.kill('SIGKILL'); resolve(undefined) }, 5_000)
  })
  try {
    const result = Promise.all([process.exited, new Response(process.stdout).text(), new Response(process.stderr).text()])
      .then(([code, text]) => ({ code, text: text.trim() }))
    return await Promise.race([result, timeout])
  } finally { clearTimeout(timer) }
}

/** Cache by canonical git root, not clone path spelling or nested working directory. */
export function createRepoIdentityResolver(options: { readonly now?: () => number } = {}): (workspace: string) => Promise<string | undefined> {
  const now = options.now ?? Date.now
  const roots = new Map<string, CacheEntry>()
  const workspaces = new Map<string, CacheEntry>()
  const cached = (cache: Map<string, CacheEntry>, key: string, load: () => Promise<string | undefined>): Promise<string | undefined> => {
    const existing = cache.get(key)
    if (existing && existing.retryAt > now()) return existing.pending
    const entry: CacheEntry = { pending: Promise.resolve(undefined), retryAt: Infinity }
    entry.pending = load().then(result => { entry.retryAt = result === undefined ? now() + 60_000 : Infinity; return result }, error => { cache.delete(key); throw error })
    cache.set(key, entry)
    return entry.pending
  }
  return async workspace => {
    const canonical = await realpath(workspace)
    return cached(workspaces, canonical, async () => {
      const root = await lessonGitRoot(canonical)
      return root === undefined ? undefined : cached(roots, root, () => identityForRoot(root))
    })
  }
}

interface CacheEntry { pending: Promise<string | undefined>; retryAt: number }

/** Folder sharing always targets the Git toplevel, even from a nested session cwd. */
export async function lessonGitRoot(workspace: string): Promise<string | undefined> {
  const found = await lessonGit(await realpath(workspace), ['rev-parse', '--show-toplevel'])
  return found?.code === 0 && isAbsolute(found.text) ? realpath(found.text) : undefined
}

async function identityForRoot(root: string): Promise<string | undefined> {
  const remote = await lessonGit(root, ['config', '--get', 'remote.origin.url'])
  if (!remote || (remote.code !== 0 && remote.code !== 1)) return undefined
  let normalized = remote.code === 0 ? normalizeRepoRemote(remote.text) : undefined
  // Relative/local remotes have no host identity: use the repository's root history.
  if (remote.code === 0 && !normalized && /^(?:[a-z][a-z\d+.-]*:\/\/|[^/]+:)/iu.test(remote.text)) {
    if (!remote.text.startsWith('file://')) return undefined
  }
  if (!normalized) {
    const commits = await lessonGit(root, ['rev-list', '--max-parents=0', 'HEAD'])
    if (!commits || commits.code !== 0) return undefined
    const first = commits.text.split(/\s+/u).filter(sha => /^[0-9a-f]{40,64}$/u.test(sha)).sort()[0]
    if (!first) return undefined
    normalized = `root:${first}`
  }
  return `git:${createHash('sha256').update(normalized).digest('hex').slice(0, 32)}`
}
