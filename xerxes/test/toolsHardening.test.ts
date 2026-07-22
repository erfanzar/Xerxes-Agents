// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { ContextualMemory } from '../src/memory/contextualMemory.js'
import { LongTermMemory } from '../src/memory/longTermMemory.js'
import { SimpleStorage } from '../src/memory/storage.js'
import { findAndReplace, gitApplyPatch } from '../src/tools/codingTools.js'
import { deleteMemory, registerMemoryTools, saveMemory, searchMemory } from '../src/tools/memoryTools.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import { PublicWebClient, type WebFetch } from '../src/tools/webTools.js'
import type { JsonObject, ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

function call(name: string, arguments_: JsonObject): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name, arguments: arguments_ } }
}

function definition(name: string): ToolDefinition {
  return {
    type: 'function',
    function: { name, description: name + ' test double', parameters: { properties: {}, type: 'object' } },
  }
}

function contextualMemory(): ContextualMemory {
  return new ContextualMemory({
    longTerm: new LongTermMemory({ storage: new SimpleStorage() }),
  })
}

function publicClient(fetcher: WebFetch): PublicWebClient {
  // Deterministic public DNS answers keep these tests offline.
  return new PublicWebClient({ fetcher, urlSafety: { dnsLookup: async () => ['93.184.216.34'] } })
}

async function inWorkspace(run: (workspace: string, paths: WorkspacePathResolver) => Promise<void>): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-tools-hardening-'))
  try {
    await run(workspace, new WorkspacePathResolver(workspace))
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
}

async function git(cwd: string, arguments_: readonly string[]): Promise<void> {
  const child = Bun.spawn(['git', ...arguments_], { cwd, stdin: 'ignore', stderr: 'pipe', stdout: 'pipe' })
  const [code, stderr] = await Promise.all([child.exited, new Response(child.stderr).text()])
  if (code !== 0) {
    throw new Error(stderr)
  }
}

test('PublicWebClient strips caller headers on cross-origin redirects and rejects scheme downgrades', async () => {
  const seen: Array<{ headers: unknown; url: string }> = []
  const redirecting = publicClient(async (url, init) => {
    seen.push({ headers: init.headers, url })
    if (url === 'https://one.example/start') {
      return new Response(null, { headers: { location: 'https://two.example/next' }, status: 302 })
    }
    if (url === 'https://two.example/next') {
      return new Response(null, { headers: { location: 'https://one.example/back' }, status: 302 })
    }
    return new Response('ok', { status: 200 })
  })
  const result = await redirecting.fetch('https://one.example/start', {
    headers: { authorization: 'Bearer secret', cookie: 'sid=1', 'x-custom': 'value' },
  })
  expect(result.url).toBe('https://one.example/back')
  // The same-origin first hop keeps the caller headers…
  expect(seen[0]?.headers).toEqual({ authorization: 'Bearer secret', cookie: 'sid=1', 'x-custom': 'value' })
  // …the cross-origin hop drops authorization, cookie, and custom headers…
  expect(seen[1]?.headers).toBeUndefined()
  // …and they stay dropped even when a later hop returns to the original origin.
  expect(seen[2]?.headers).toBeUndefined()

  const sameOrigin = publicClient(async (url, init) => {
    seen.push({ headers: init.headers, url })
    if (url.endsWith('/start')) {
      return new Response(null, { headers: { location: '/end' }, status: 302 })
    }
    return new Response('ok', { status: 200 })
  })
  await sameOrigin.fetch('https://three.example/start', { headers: { authorization: 'Bearer keep' } })
  expect(seen[seen.length - 1]).toEqual({ headers: { authorization: 'Bearer keep' }, url: 'https://three.example/end' })

  let downgradeFetches = 0
  const downgrading = publicClient(async url => {
    downgradeFetches += 1
    return new Response(null, { headers: { location: 'http://one.example/insecure' }, status: 302 })
  })
  await expect(downgrading.fetch('https://one.example/start')).rejects.toThrow('downgrade')
  expect(downgradeFetches).toBe(1)
})

test('memory tools scope agent_id to the execution context unless a privileged port authorizes cross-agent access', async () => {
  const memory = contextualMemory()
  const registry = new ToolRegistry()
  registerMemoryTools(registry, { resolveContext: () => ({ memory }) })
  const execution = { agentId: 'agent-a', metadata: {} }

  // The execution-context agent wins over a model-supplied cross-agent agent_id.
  const deniedSave = JSON.parse(await registry.execute(
    call('save_memory', { agent_id: 'agent-b', content: 'poisoned note' }),
    execution,
  )) as { message: string; status: string }
  expect(deniedSave.status).toBe('error')
  expect(deniedSave.message).toContain('privileged host port')
  expect(memory.shortTerm.size + memory.longTerm.size).toBe(0)

  const deniedSearch = JSON.parse(await registry.execute(
    call('search_memory', { agent_id: 'agent-b', query: 'poisoned' }),
    execution,
  )) as { status: string }
  expect(deniedSearch.status).toBe('error')

  // Same-agent and default agent_id keep working.
  const ownSave = JSON.parse(await registry.execute(
    call('save_memory', { agent_id: 'agent-a', content: 'own note' }),
    execution,
  )) as { status: string }
  expect(ownSave.status).toBe('success')
  const ownSearch = JSON.parse(await registry.execute(
    call('search_memory', { query: 'own' }),
    execution,
  )) as { count: number; status: string }
  expect(ownSearch).toMatchObject({ status: 'success', count: 1 })

  // An explicit privileged host port re-enables intentional cross-agent access.
  const privileged = new ToolRegistry()
  registerMemoryTools(privileged, { resolveContext: () => ({ memory }), allowCrossAgent: () => true })
  const allowedSave = JSON.parse(await privileged.execute(
    call('save_memory', { agent_id: 'agent-b', content: 'shared note' }),
    execution,
  )) as { status: string }
  expect(allowedSave.status).toBe('success')
  const crossSearch = JSON.parse(await privileged.execute(
    call('search_memory', { agent_id: 'agent-b', query: 'shared' }),
    execution,
  )) as { count: number; status: string }
  expect(crossSearch).toMatchObject({ status: 'success', count: 1 })
})

test('delete_memory scopes bulk and by-ID deletion to the calling agent', () => {
  const memory = contextualMemory()
  const contextA = { agentId: 'agent-a', memory }
  const contextB = { agentId: 'agent-b', memory }
  saveMemory({ content: 'alpha shared note', memoryType: 'long_term', tags: ['shared'] }, contextA)
  const savedB = saveMemory(
    { content: 'beta shared note', memoryType: 'long_term', tags: ['shared'] },
    contextB,
  ) as { memory_id: string; status: string }

  // A tags-only bulk delete removes the calling agent's entries, never another agent's.
  const bulk = deleteMemory({ tags: ['shared'] }, contextA) as { deleted_count: number; status: string }
  expect(bulk).toMatchObject({ status: 'success', deleted_count: 1 })
  expect((searchMemory({ query: 'beta' }, contextB) as { count: number }).count).toBe(1)

  // A by-ID delete of another agent's memory is rejected and observable.
  const byId = deleteMemory({ memoryId: savedB.memory_id }, contextA) as { message: string; status: string }
  expect(byId.status).toBe('error')
  expect(byId.message).toContain('another agent')
  expect((searchMemory({ query: 'beta' }, contextB) as { count: number }).count).toBe(1)

  // A cross-agent agent_id criterion is rejected without a privileged port…
  const crossBulk = deleteMemory({ agentId: 'agent-b', tags: ['shared'] }, contextA) as { status: string }
  expect(crossBulk.status).toBe('error')
  // …and honored with one.
  const privilegedBulk = deleteMemory({ agentId: 'agent-b', tags: ['shared'] }, contextA, () => true) as {
    deleted_count: number
    status: string
  }
  expect(privilegedBulk).toMatchObject({ status: 'success', deleted_count: 1 })
  expect((searchMemory({ query: 'beta' }, contextB) as { count: number }).count).toBe(0)
})

test('find_and_replace rejects catastrophic-backtracking regex shapes and over-long patterns', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'subject.txt'), 'a'.repeat(19) + 'b\nhttps://example.com\n')
    const reject = (search: string, message: string) => expect(findAndReplace({
      backup: false,
      file_path: 'subject.txt',
      regex: true,
      replace: 'x',
      search,
    }, paths)).rejects.toThrow(message)

    // Nested quantifiers under an unbounded outer quantifier.
    await reject('(a+)+$', 'unbounded quantifier')
    await reject('(a?)+$', 'unbounded quantifier')
    // Adjacent unbounded wildcard quantifiers.
    await reject('.*.*b', 'adjacent unbounded wildcard')
    // Pattern-length cap.
    await reject('a'.repeat(257), 'at most 256')

    // Bounded nested quantifiers and ordinary patterns stay legal.
    const bounded = await findAndReplace({
      backup: false,
      file_path: 'subject.txt',
      regex: true,
      replace: '',
      search: 'https?://',
    }, paths)
    expect(bounded).toContain('Replaced 1')
    const grouped = await findAndReplace({
      backup: false,
      file_path: 'subject.txt',
      regex: true,
      replace: 'b',
      search: '(a?a)?b',
    }, paths)
    expect(grouped).toContain('Replaced 1')
    expect(await Bun.file(join(workspace, 'subject.txt')).text()).toBe('a'.repeat(17) + 'b\nexample.com\n')
  })
})

test('git_apply_patch rejects patch paths that escape the workspace through an in-repo symlink', async () => {
  await inWorkspace(async (workspace, paths) => {
    const outside = await mkdtemp(join(tmpdir(), 'xerxes-git-apply-outside-'))
    try {
      await git(workspace, ['init'])
      await git(workspace, ['config', 'user.email', 'test@example.invalid'])
      await git(workspace, ['config', 'user.name', 'Xerxes Test'])
      await Bun.write(join(workspace, 'tracked.txt'), 'before\n')
      await git(workspace, ['add', 'tracked.txt'])
      await git(workspace, ['commit', '-m', 'initial'])
      await Bun.write(join(outside, 'secret.txt'), 'original\n')
      await symlink(outside, join(workspace, 'link'), 'dir')

      const escapePatch = [
        '--- a/link/secret.txt',
        '+++ b/link/secret.txt',
        '@@ -1 +1 @@',
        '-original',
        '+pwned',
        '',
      ].join('\n')
      await expect(gitApplyPatch({ patch_content: escapePatch }, paths)).rejects.toThrow('workspace root')
      expect(await Bun.file(join(outside, 'secret.txt')).text()).toBe('original\n')

      const traversalPatch = [
        '--- a/tracked.txt',
        '+++ b/../outside.txt',
        '@@ -1 +1 @@',
        '-before',
        '+pwned',
        '',
      ].join('\n')
      await expect(gitApplyPatch({ patch_content: traversalPatch }, paths)).rejects.toThrow('workspace root')

      // A normal in-workspace patch still validates and applies.
      const safePatch = [
        '--- a/tracked.txt',
        '+++ b/tracked.txt',
        '@@ -1 +1 @@',
        '-before',
        '+after',
        '',
      ].join('\n')
      expect(await gitApplyPatch({ patch_content: safePatch }, paths)).toBe('Patch applied successfully')
      expect(await Bun.file(join(workspace, 'tracked.txt')).text()).toBe('after\n')
    } finally {
      await rm(outside, { force: true, recursive: true })
    }
  })
})

test('ToolRegistry warns on duplicate registration while replace() stays the explicit override path', () => {
  const warnings: string[] = []
  const registry = new ToolRegistry({
    onDuplicateRegistration: (name, agentId) => warnings.push(name + '@' + agentId),
  })
  registry.register(definition('dup'), () => 'first')
  registry.register(definition('dup'), () => 'second')
  expect(warnings).toEqual(['dup@default'])
  // First handler still wins; the duplicate does not silently take over.
  expect(registry.get('dup')?.({}, { metadata: {} })).toBe('first')

  registry.replace(definition('dup'), () => 'replaced')
  expect(registry.get('dup')?.({}, { metadata: {} })).toBe('replaced')
  expect(warnings).toHaveLength(1)

  // Registering a variant for a different agent is not a duplicate.
  registry.register(definition('dup'), () => 'agent-a-variant', 'agent-a')
  expect(warnings).toHaveLength(1)
  expect(registry.get('dup', 'agent-a')?.({}, { metadata: {} })).toBe('agent-a-variant')
})
