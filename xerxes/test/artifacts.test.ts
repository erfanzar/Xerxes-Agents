// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { artifactsRpc } from '../src/daemon/artifactsRpc.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  ArtifactStore,
  MAX_ARTIFACT_CONTENT_BYTES,
  MAX_ARTIFACT_TITLE_CHARS,
  MAX_ARTIFACTS_PER_SESSION,
} from '../src/session/artifacts.js'
import { CLAUDE_WORKFLOW_TOOL_GUIDANCE, registerClaudeWorkflowTools } from '../src/tools/claudeTools/workflow.js'
import type { JsonObject } from '../src/types/toolCalls.js'

async function withRoot(run: (root: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-artifacts-'))
  try {
    await run(root)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

/** A clock that advances one second per reading, so ordering by updated_at is deterministic. */
function steppingClock(): () => Date {
  let tick = 0
  return () => new Date(Date.UTC(2026, 0, 1, 0, 0, tick++))
}

test('store creates, updates, lists, reads and pins per session', async () => {
  await withRoot(async root => {
    const store = new ArtifactStore({ root, now: steppingClock() })
    const report = await store.create('abc123', { title: '  Q3 report ', content: '# Report\nbody' })
    expect(report).toMatchObject({ title: 'Q3 report', kind: 'markdown', pinned: false, version: 1 })
    expect(report.id).toMatch(/^[a-z0-9]{10}$/)
    expect(report.created_at).toBe(report.updated_at)
    const code = await store.create('abc123', { title: 'Script', kind: 'code', language: 'typescript', content: 'let x = 1' })

    // Newest revision first.
    expect((await store.list('abc123')).map(row => row.id)).toEqual([code.id, report.id])

    const revised = await store.update('abc123', report.id, { title: 'Q3 report v2', content: 'é' })
    expect(revised).toMatchObject({ id: report.id, title: 'Q3 report v2', version: 2, content: 'é', created_at: report.created_at })
    expect(revised.updated_at > report.updated_at).toBe(true)
    // Title kept when an update omits it.
    expect((await store.update('abc123', report.id, { content: 'again' })).title).toBe('Q3 report v2')

    const listed = await store.list('abc123')
    expect(listed.map(row => row.id)).toEqual([report.id, code.id])
    expect(listed[0]).toMatchObject({ size: 5, version: 3 })
    expect(listed[1]).toMatchObject({ kind: 'code', language: 'typescript', size: 9 })
    expect('content' in listed[0]!).toBe(false)

    // Pinning moves a row first without touching version or updated_at.
    const before = await store.read('abc123', code.id)
    const pinned = await store.pin('abc123', code.id, true)
    expect(pinned).toMatchObject({ id: code.id, pinned: true, version: 1, updated_at: before?.updated_at })
    expect((await store.list('abc123')).map(row => row.id)).toEqual([code.id, report.id])
    expect(await store.pin('abc123', 'zzzzzzzzzz', true)).toBeUndefined()

    // Sessions are isolated.
    expect(await store.list('other')).toEqual([])
    expect(await store.read('other', code.id)).toBeUndefined()
    await expect(store.update('other', code.id, { content: 'x' })).rejects.toThrow('no artifact')

    // Atomic writes leave no temporary files behind.
    expect((await readdir(join(root, 'abc123'))).sort()).toEqual([`${code.id}.json`, `${report.id}.json`].sort())
  })
})

test('store enforces bounds and refuses unsafe session ids', async () => {
  await withRoot(async root => {
    const store = new ArtifactStore({ root })
    await expect(store.create('s1', { title: '', content: 'x' })).rejects.toThrow('title')
    await expect(store.create('s1', { title: 'x'.repeat(MAX_ARTIFACT_TITLE_CHARS + 1), content: 'x' })).rejects.toThrow('at most 200')
    await expect(store.create('s1', { title: 't', content: '   ' })).rejects.toThrow('content')
    await expect(store.create('s1', { title: 't', content: 'a'.repeat(MAX_ARTIFACT_CONTENT_BYTES + 1) })).rejects.toThrow('2 MiB')
    expect((await store.create('s1', { title: 't', content: 'a'.repeat(MAX_ARTIFACT_CONTENT_BYTES) })).version).toBe(1)
    await expect(store.create('../escape', { title: 't', content: 'x' })).rejects.toThrow('session id')
    await expect(store.list('a/b')).rejects.toThrow('session id')

    const full = new ArtifactStore({ root: join(root, 'full') })
    await Promise.all(Array.from({ length: MAX_ARTIFACTS_PER_SESSION }, (_, i) => full.create('s2', { title: `a${i}`, content: 'x' })))
    expect(await full.list('s2')).toHaveLength(MAX_ARTIFACTS_PER_SESSION)
    await expect(full.create('s2', { title: 'one more', content: 'x' })).rejects.toThrow('maximum of 200')
  })
})

test('store skips corrupt or foreign files and never deletes them', async () => {
  await withRoot(async root => {
    const store = new ArtifactStore({ root })
    const good = await store.create('s1', { title: 'Good', content: 'ok' })
    const directory = join(root, 's1')
    await writeFile(join(directory, 'aaaaaaaaaa.json'), '{not json')
    await writeFile(join(directory, 'bbbbbbbbbb.json'), JSON.stringify({ ...good, id: 'bbbbbbbbbb', kind: 'pdf' }))
    // The id inside must match the file name.
    await writeFile(join(directory, 'cccccccccc.json'), JSON.stringify({ ...good }))
    await writeFile(join(directory, 'dddddddddd.json'), JSON.stringify({ ...good, id: 'dddddddddd', version: 0 }))
    await writeFile(join(directory, 'notes.txt'), 'stray')
    await mkdir(join(directory, 'eeeeeeeeee.json'))

    expect((await store.list('s1')).map(row => row.id)).toEqual([good.id])
    expect(await store.read('s1', 'aaaaaaaaaa')).toBeUndefined()
    expect(await store.read('s1', 'cccccccccc')).toBeUndefined()
    expect(await store.read('s1', '../s1/x')).toBeUndefined()
    expect((await readdir(directory)).length).toBe(7)
  })
})

async function runArtifactTool(registry: ToolRegistry, args: JsonObject, sessionId?: string): Promise<string> {
  return registry.execute(
    { id: crypto.randomUUID(), type: 'function', function: { name: 'ArtifactTool', arguments: args } },
    { metadata: {}, ...(sessionId === undefined ? {} : { sessionId }) },
  )
}

test('ArtifactTool creates and updates through the registry with the session id', async () => {
  await withRoot(async root => {
    const store = new ArtifactStore({ root })
    const registry = new ToolRegistry()
    const registered = registerClaudeWorkflowTools(registry, { artifactStore: store })
    expect(registered.map(tool => tool.function.name)).toContain('ArtifactTool')
    expect(registry.hasDeclaredCapabilities('ArtifactTool')).toBe(true)
    expect(registry.capabilities('ArtifactTool')).toMatchObject({ destructive: false, openWorld: false, readOnly: false })
    expect(registry.guidanceForTools(['ArtifactTool'])[0]?.text).toBe(CLAUDE_WORKFLOW_TOOL_GUIDANCE.ArtifactTool)

    const created = await runArtifactTool(registry, { action: 'create', title: 'Migration plan', content: '# Plan\n- step' }, 'sess01')
    expect(created).toContain('Created artifact "Migration plan"')
    const [row] = await store.list('sess01')
    expect(row).toMatchObject({ title: 'Migration plan', kind: 'markdown', version: 1 })
    expect(created).toContain(`id ${row!.id}`)

    const updated = await runArtifactTool(registry, { action: 'update', id: row!.id, content: '# Plan v2', kind: 'text' }, 'sess01')
    expect(updated).toBe(`Updated artifact "Migration plan" (id ${row!.id}, version 2).`)
    expect(await store.read('sess01', row!.id)).toMatchObject({ content: '# Plan v2', kind: 'text', version: 2 })
    expect(await store.list('sess01')).toHaveLength(1)

    await expect(runArtifactTool(registry, { action: 'update', content: 'x' }, 'sess01')).rejects.toThrow('id')
    await expect(runArtifactTool(registry, { action: 'update', id: 'zzzzzzzzzz', content: 'x' }, 'sess01')).rejects.toThrow('no artifact')
    await expect(runArtifactTool(registry, { action: 'create', content: 'x' }, 'sess01')).rejects.toThrow('title')
  })
})

test('ArtifactTool refuses a call without a session instead of inventing one', async () => {
  await withRoot(async root => {
    const store = new ArtifactStore({ root })
    const registry = new ToolRegistry()
    registerClaudeWorkflowTools(registry, { artifactStore: store })
    await expect(runArtifactTool(registry, { action: 'create', title: 'Report', content: 'body' })).rejects.toThrow('no session is attached')
    expect(await readdir(root)).toEqual([])
  })
})

test('artifacts RPC lists without content, reads, pins and reports unknown ids', async () => {
  await withRoot(async root => {
    const store = new ArtifactStore({ root, now: steppingClock() })
    const first = await store.create('sess01', { title: 'First', content: 'one' })
    const second = await store.create('sess01', { title: 'Code', kind: 'code', language: 'rust', content: 'fn main() {}' })

    const listed = await artifactsRpc(store, 'artifacts.list', 'sess01', {})
    expect(listed).toEqual({
      ok: true,
      artifacts: [
        { id: second.id, title: 'Code', kind: 'code', language: 'rust', size: 12, created_at: second.created_at, updated_at: second.updated_at, pinned: false, version: 1 },
        { id: first.id, title: 'First', kind: 'markdown', size: 3, created_at: first.created_at, updated_at: first.updated_at, pinned: false, version: 1 },
      ],
    })

    expect(await artifactsRpc(store, 'artifacts.read', 'sess01', { id: first.id })).toMatchObject({ ok: true, artifact: { id: first.id, content: 'one', size: 3 } })
    expect(await artifactsRpc(store, 'artifacts.read', 'sess01', { id: 'zzzzzzzzzz' })).toMatchObject({ ok: false })
    expect(await artifactsRpc(store, 'artifacts.read', 'sess01', {})).toEqual({ ok: false, error: 'id is required' })

    const pinned = await artifactsRpc(store, 'artifacts.pin', 'sess01', { id: first.id, pinned: true })
    expect(pinned).toMatchObject({ ok: true, artifact: { id: first.id, pinned: true, version: 1 } })
    expect(JSON.stringify(pinned)).not.toContain('content')
    const reordered = await artifactsRpc(store, 'artifacts.list', 'sess01', {}) as { artifacts: { id: string }[] }
    expect(reordered.artifacts.map(row => row.id)).toEqual([first.id, second.id])
    expect(await artifactsRpc(store, 'artifacts.pin', 'sess01', { id: first.id })).toEqual({ ok: false, error: 'pinned must be a boolean' })
    expect(await artifactsRpc(store, 'artifacts.pin', 'sess01', { id: 'zzzzzzzzzz', pinned: true })).toMatchObject({ ok: false })
    expect(await artifactsRpc(store, 'artifacts.list', 'other', {})).toEqual({ ok: true, artifacts: [] })
  })
})

test('artifacts.update saves a person\'s edit, bumps the version and validates params', async () => {
  await withRoot(async root => {
    const store = new ArtifactStore({ root, now: steppingClock() })
    const created = await store.create('sess02', { title: 'Plan', kind: 'markdown', content: '# old' })
    const updated = await artifactsRpc(store, 'artifacts.update', 'sess02', { id: created.id, content: '# new' })
    expect(updated).toMatchObject({ ok: true, artifact: { id: created.id, content: '# new', version: 2, title: 'Plan' } })
    expect(await artifactsRpc(store, 'artifacts.read', 'sess02', { id: created.id })).toMatchObject({ ok: true, artifact: { content: '# new', version: 2 } })
    expect(await artifactsRpc(store, 'artifacts.update', 'sess02', { id: created.id })).toEqual({ ok: false, error: 'content must be a string' })
    expect(await artifactsRpc(store, 'artifacts.update', 'sess02', { id: created.id, content: 'x', title: 5 })).toEqual({ ok: false, error: 'title must be a string' })
    expect(await artifactsRpc(store, 'artifacts.update', 'sess02', { id: 'zzzzzzzzzz', content: 'x' })).toMatchObject({ ok: false })
  })
})
