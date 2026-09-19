// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { InMemoryDaemonRuntime, type TurnRunner } from '../src/daemon/runtime.js'
import { WorkspaceTurnRunner } from '../src/daemon/workspaceTurnRunner.js'
import { DaemonWorkspaces } from '../src/daemon/workspaceResources.js'
import { parseSkillMarkdown } from '../src/extensions/skills.js'

test('parallel workspaces retain separate capabilities, runners and cancellation', async () => {
  const home = await mkdtemp(join(tmpdir(), 'xws-'))
  const resources = new DaemonWorkspaces({ home, allowWorkspace: false, report() {} })
  const runtime = new InMemoryDaemonRuntime(undefined, { sessionDirectory: join(home, 'sessions') })
  try {
    const [a, b] = await Promise.all([runtime.openSession('a', 'default', { cwd: join(home, 'a') }), runtime.openSession('b', 'default', { cwd: join(home, 'b') })])
    const [ra, rb] = await Promise.all([resources.get(a.cwd), resources.get(b.cwd)])
    expect(await resources.get(a.cwd)).toBe(ra)
    expect(ra.mcpManager).not.toBe(rb.mcpManager)
    expect(ra.agentPresetRoster).not.toBe(rb.agentPresetRoster)
    for (const [resource, label] of [[ra, 'alpha'], [rb, 'beta']] as const) resource.skillRegistry.register(parseSkillMarkdown(`---\nname: isolated\ndescription: test\n---\n${label}`, join(home, label, 'SKILL.md')))
    let created = 0
    const runner = new WorkspaceTurnRunner(resources, (cwd, resource): TurnRunner => {
      created++
      return { async *run(session, _text, signal) {
        yield { type: 'text_delta', payload: { text: resource.skillRegistry.get('isolated')?.instructions, cwd, session: session.id } }
        await Promise.resolve()
        signal.throwIfAborted()
        yield { type: 'text_delta', payload: { text: resource.skillRegistry.get('isolated')?.instructions } }
      } }
    })
    const ca = new AbortController(), cb = new AbortController()
    const aa = runner.run(a, 'run', ca.signal)[Symbol.asyncIterator]()
    const bb = runner.run(b, 'run', cb.signal)[Symbol.asyncIterator]()
    expect((await aa.next()).value?.payload.text).toBe('alpha')
    expect((await bb.next()).value?.payload.text).toBe('beta')
    ca.abort(new Error('stop alpha'))
    await expect(aa.next()).rejects.toThrow('stop alpha')
    expect((await bb.next()).value?.payload.text).toBe('beta')
    await bb.next()
    expect(created).toBe(2)
    for await (const _event of runner.run(b, 'again', cb.signal)) { /* consume */ }
    expect(created).toBe(2)
  } finally { await resources.close(); await runtime.shutdown(); await rm(home, { recursive: true, force: true }) }
})

test('workspace initialization failures are observable and retryable', async () => {
  let attempts = 0
  const resources = new DaemonWorkspaces({ home: '/tmp', allowWorkspace: false, report() {}, beforeOpen: async () => { attempts++; throw new Error('ownership unavailable') } })
  await expect(resources.get('/not-owned')).rejects.toThrow('ownership unavailable')
  await expect(resources.get('/not-owned')).rejects.toThrow('ownership unavailable')
  expect(attempts).toBe(2)
  await resources.close()
  await expect(resources.get('/not-owned')).rejects.toThrow('closed')
})

test('a released load finishing late cannot replace a reopened workspace', async () => {
  const home = await mkdtemp(join(tmpdir(), 'xws-reopen-'))
  let unblock!: () => void
  const gate = new Promise<void>(resolve => { unblock = resolve })
  let opens = 0
  const resources = new DaemonWorkspaces({
    home, allowWorkspace: false, report() {},
    beforeOpen: async () => { if (++opens === 1) await gate },
  })
  try {
    const oldLoad = resources.get(home)
    const release = resources.release(home)
    const reopened = await resources.get(home)
    expect(resources.peek(home)).toBe(reopened)
    unblock()
    expect(await oldLoad).not.toBe(reopened)
    await release
    expect(resources.peek(home)).toBe(reopened)
    expect(await resources.get(home)).toBe(reopened)
  } finally {
    unblock()
    await resources.close()
    await rm(home, { recursive: true, force: true })
  }
})
