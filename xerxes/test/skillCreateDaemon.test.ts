// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm, stat } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { SkillCreateFlow, type SkillCreateTransition } from '../src/daemon/skillCreate.js'

interface Frame {
  readonly id?: number
  readonly method?: string
  readonly params?: { readonly payload?: Record<string, unknown>; readonly type?: string }
  readonly result?: Record<string, unknown>
}

test('daemon runs the native skill-create interview and submits its authored draft as a real turn', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-skill-create-daemon-'))
  const socketPath = join(directory, 'daemon.sock')
  const skillsDirectory = join(directory, 'skills')
  const server = new DaemonServer({
    socketPath,
    skillDirectory: skillsDirectory,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, 'sessions'),
    }),
  })
  await server.start()
  const client = await SocketTestClient.connect(socketPath)
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { session_key: 'skill-create' } })
    expect((await client.next(frame => frame.id === 1)).result).toMatchObject({ ok: true })
    await client.next(eventFrame('init_done'))
    await client.next(eventFrame('status_update'))

    client.send({ jsonrpc: '2.0', id: 2, method: 'slash', params: { command: '/skill-create release-notes' } })
    expect((await client.next(frame => frame.id === 2)).result).toEqual({ ok: true })
    expect((await client.next(eventFrame('notification'))).params?.payload?.body).toContain('What should this skill do?')
    expect((await stat(join(skillsDirectory, 'release-notes'))).isDirectory()).toBeTrue()

    client.send({ jsonrpc: '2.0', id: 3, method: 'turn.submit', params: { text: 'Draft release notes from merged pull requests.' } })
    expect((await client.next(frame => frame.id === 3)).result).toEqual({ ok: true, consumed_for: 'skill-create' })
    expect((await client.next(eventFrame('notification'))).params?.payload?.body).toContain('When should a future session activate')
    await client.next(eventFrame('turn_begin'))
    await client.next(eventFrame('turn_end'))

    client.send({ jsonrpc: '2.0', id: 4, method: 'turn.submit', params: { text: '' } })
    expect((await client.next(frame => frame.id === 4)).result).toEqual({ ok: true, consumed_for: 'skill-create' })
    expect((await client.next(eventFrame('notification'))).params?.payload?.body).toContain('That field is required')
    await client.next(eventFrame('turn_begin'))
    await client.next(eventFrame('turn_end'))

    client.send({ jsonrpc: '2.0', id: 5, method: 'turn.submit', params: { text: 'auto' } })
    expect((await client.next(frame => frame.id === 5)).result).toEqual({ ok: true, consumed_for: 'skill-create' })
    expect((await client.next(eventFrame('notification'))).params?.payload?.body).toContain('Drafting skill `release-notes`')
    await client.next(eventFrame('turn_begin'))
    await client.next(eventFrame('turn_end'))
    const draftBegin = await client.next(frame => frame.method === 'event'
      && frame.params?.type === 'turn_begin'
      && typeof frame.params.payload?.text === 'string'
      && frame.params.payload.text.includes('Write a reusable agent skill called'))
    expect(draftBegin.params?.payload?.text).toContain('SKILL.md')
    await client.next(eventFrame('text_part'))
    await client.next(eventFrame('turn_end'))
  } finally {
    client.close()
    await server.stop()
    await rm(directory, { recursive: true, force: true })
  }
})

/** Read a transition's user-facing text regardless of which arm it is. */
function say(transition: SkillCreateTransition | undefined): string {
  if (transition?.kind === 'prompt' || transition?.kind === 'cancelled') return transition.message
  if (transition?.kind === 'draft') return transition.draft.announcement
  return ''
}

test('the interview asks where to save only when both roots exist, and settles the directory with it', async () => {
  const created: string[] = []
  const ensureDirectory = async (path: string): Promise<void> => {
    created.push(path)
  }
  const flow = new SkillCreateFlow({
    skillsDirectory: '/home/dev/.xerxes/skills',
    projectSkillsDirectory: '/repo/.agents/skills',
    ensureDirectory,
  })

  expect(say(await flow.start('release-notes', 'session-1'))).toContain('Where should this skill be saved?')
  // The scope decides the directory, so nothing may be created before it is answered.
  expect(created).toEqual([])
  expect(say(await flow.answer('session-1', 'somewhere else'))).toContain('Answer `project` or `user`')
  expect(say(await flow.answer('session-1', 'project'))).toContain('What should this skill do?')
  expect(created).toEqual([join('/repo/.agents/skills', 'release-notes')])

  const draft = await flow.answer('session-1', 'auto')
  expect(draft?.kind).toBe('draft')
  if (draft?.kind !== 'draft') throw new Error('expected a draft')
  expect(draft.draft.scope).toBe('project')
  expect(draft.draft.targetPath).toBe(join('/repo/.agents/skills', 'release-notes', 'SKILL.md'))
  // Success is checkable per step, irreversible steps are called out, and the
  // trailing section is the end-to-end signal rather than the only one.
  expect(draft.draft.prompt).toContain('the observable signal that it worked')
  expect(draft.draft.prompt).toContain('`**Irreversible.**`')
  expect(draft.draft.prompt).toContain('Do not restate the per-step signals here')

  // An empty answer takes the root that discovery already trusts.
  const defaulted = new SkillCreateFlow({
    skillsDirectory: '/home/dev/.xerxes/skills',
    projectSkillsDirectory: '/repo/.agents/skills',
    ensureDirectory,
  })
  await defaulted.start('notes', 'session-2')
  expect(say(await defaulted.answer('session-2', ''))).toContain('What should this skill do?')
  expect(created.at(-1)).toBe(join('/home/dev/.xerxes/skills', 'notes'))

  // One writable root means there is no question to ask; the flow is unchanged.
  const single = new SkillCreateFlow({ skillsDirectory: '/home/dev/.xerxes/skills', ensureDirectory })
  expect(say(await single.start('solo', 'session-3'))).toContain('What should this skill do?')
  expect(created.at(-1)).toBe(join('/home/dev/.xerxes/skills', 'solo'))
})

function eventFrame(type: string): (frame: Frame) => boolean {
  return frame => frame.method === 'event' && frame.params?.type === type
}

class SocketTestClient {
  private buffer = ''
  private readonly frames: Frame[] = []
  private readonly waiters: Array<{ predicate: (frame: Frame) => boolean; resolve: (frame: Frame) => void }> = []

  private constructor(private readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', chunk => this.receive(typeof chunk === 'string' ? chunk : new TextDecoder().decode(chunk)))
  }

  static async connect(socketPath: string): Promise<SocketTestClient> {
    const socket = connect({ path: socketPath })
    await new Promise<void>((resolve, reject) => {
      socket.once('connect', resolve)
      socket.once('error', reject)
    })
    return new SocketTestClient(socket)
  }

  close(): void {
    this.socket.destroy()
  }

  next(predicate: (frame: Frame) => boolean): Promise<Frame> {
    const index = this.frames.findIndex(predicate)
    if (index >= 0) {
      const frame = this.frames.splice(index, 1)[0]
      if (frame) return Promise.resolve(frame)
    }
    return new Promise(resolve => this.waiters.push({ predicate, resolve }))
  }

  send(frame: Record<string, unknown>): void {
    this.socket.write(`${JSON.stringify(frame)}\n`)
  }

  private receive(chunk: string): void {
    this.buffer += chunk
    let newline = this.buffer.indexOf('\n')
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline)
      this.buffer = this.buffer.slice(newline + 1)
      if (line.trim()) this.handle(JSON.parse(line) as Frame)
      newline = this.buffer.indexOf('\n')
    }
  }

  private handle(frame: Frame): void {
    const index = this.waiters.findIndex(waiter => waiter.predicate(frame))
    const waiter = index < 0 ? undefined : this.waiters.splice(index, 1)[0]
    if (waiter) {
      waiter.resolve(frame)
    } else {
      this.frames.push(frame)
    }
  }
}
