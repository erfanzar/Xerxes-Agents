// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DaemonServer } from '../src/daemon/server.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { ManagedPlugins } from '../src/extensions/managedPlugins.js'
import { PluginSecretStore } from '../src/extensions/pluginSecrets.js'

test('daemon installs local skills with assets and manages native plugin modules through slash', async () => {
  const root = await mkdtemp(join(tmpdir(), 'extension-rpc-'))
  const skillDirectory = join(root, 'skills')
  const manager = new ManagedPlugins(join(root, 'plugins.json'))
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
  const server = new DaemonServer({ socketPath: join(root, 'daemon.sock'), runtime, skillDirectory, skillDirectories: [skillDirectory, join(root, ".xerxes/skills")], managedPlugins: manager })
  await mkdir(join(root, 'new skill/references'), { recursive: true })
  await writeFile(join(root, 'new skill/SKILL.md'), '---\nname: local-guide\ndescription: Local guide\n---\nRead references/guide.md.')
  await writeFile(join(root, 'new skill/references/guide.md'), 'Useful guide')
  await writeFile(join(root, 'plugin file.ts'), `export function register(r) { r.registerTool('greeting', () => 'hello', {name: 'greeting'}); }`)
  await mkdir(join(root, ".xerxes/skills/project-guide"), { recursive: true })
  await writeFile(join(root, ".xerxes/skills/project-guide/SKILL.md"), "---\nname: project-guide\ndescription: Project review guide\n---\nReview this project.")
  await server.start()
  const client = await DaemonParityClient.connect(join(root, 'daemon.sock'))
  let id = 0
  const request = async (method: string, params: Record<string, unknown>) => {
    const current = ++id
    client.send({ jsonrpc: '2.0', id: current, method, params })
    return (await client.next(frame => frame.id === current)).result
  }
  try {
    await request('initialize', { session_key: 'extensions', project_dir: root })
    expect(await request('slash', { command: '/skills install "new skill"' })).toMatchObject({ ok: true })
    expect(await Bun.file(join(skillDirectory, 'local-guide/references/guide.md')).text()).toBe('Useful guide')
    expect(await request('slash', { command: '/skills install "new skill"' })).toMatchObject({ ok: false })
    expect(await request('slash', { command: '/skills search project-guide' })).toMatchObject({ ok: true, results: [expect.objectContaining({ name: 'project-guide', description: 'Project review guide' })] })
    expect(await request('slash', { command: '/skills search local-guide' })).toMatchObject({ ok: true, results: [expect.objectContaining({ name: 'local-guide' })] })
    expect(await request('complete', { text: '/local-guide' })).toMatchObject({ completions: [expect.objectContaining({ value: '/local-guide ' })] })
    expect(await request('slash', { command: '/plugins install "plugin file.ts"' })).toMatchObject({ ok: true })
    expect(manager.inventory()[0]).toMatchObject({ name: 'greeting', enabled: true })
    expect(await request('slash', { command: '/plugins disable greeting' })).toMatchObject({ ok: true })
    expect(manager.inventory()[0]).toMatchObject({ enabled: false })
    expect(await request('slash', { command: '/plugins enable greeting' })).toMatchObject({ ok: true })
    expect(await request('complete', { text: '/plugins en' })).toMatchObject({ completions: [expect.objectContaining({ value: '/plugins enable ' })] })
    expect(await request('complete', { text: '/plugins re' })).toMatchObject({ completions: [expect.objectContaining({ value: '/plugins reload ' })] })
    // A plugin command receives its argument text and the session's directory.
    // The plugin runs in its own process, so it reports what it received through its result.
    await writeFile(join(root, 'command.ts'), `export function register(r) { r.registerCommand('xdaemonecho', (args, context) => 'echoed ' + JSON.stringify({ args, cwd: context.cwd }), { description: 'Echo' }, { name: 'daemon-echo' }) }`)
    expect(await request('slash', { command: '/plugins install command.ts' })).toMatchObject({ ok: true })
    expect(await request('slash', { command: '/xdaemonecho one two' })).toMatchObject({ ok: true, plugin: 'xdaemonecho' })
    const echoed = await client.next(frame => frame.params?.type === 'notification' && String(frame.params.payload?.body ?? '').startsWith('echoed '))
    expect(JSON.parse(String(echoed.params?.payload?.body).slice('echoed '.length))).toEqual({ args: 'one two', cwd: expect.any(String) })
    expect(await request('slash', { command: '/plugins reload daemon-echo' })).toMatchObject({ ok: true })
    expect(await request('slash', { command: '/plugins uninstall daemon-echo' })).toMatchObject({ ok: true })
    expect(manager.inventory().map(item => item.name)).toEqual(['greeting'])
    expect(await request('slash', { command: '/xdaemonecho again' })).not.toMatchObject({ plugin: 'xdaemonecho' })
  } finally { manager.clearCommands(); client.close(); await server.stop(); await rm(root, { recursive: true, force: true }) }
})

test('plugin secrets are set over RPC without the value ever appearing in a reply or notification', async () => {
  const root = await mkdtemp(join(tmpdir(), 'plugin-secret-rpc-'))
  const secretsPath = join(root, 'home', 'plugin-secrets.json')
  const manager = new ManagedPlugins(join(root, 'home', 'plugins.json'), {
    secrets: new PluginSecretStore(secretsPath, { key: async () => Buffer.alloc(32, 9) }),
    projectTrustFile: join(root, 'home', 'trust.json'),
    sourcesDirectory: join(root, 'home', 'sources'),
  })
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
  const server = new DaemonServer({ socketPath: join(root, 'daemon.sock'), runtime, managedPlugins: manager, exitLogPath: join(root, 'exits.jsonl') })
  await writeFile(join(root, 'vault.ts'), `export function register(r) {
    r.registerPlugin({ name: 'vault', permissions: { secrets: ['API_TOKEN'] } })
    r.registerTool('vault_peek', { description: 'peek', parameters: { type: 'object' }, execute: (_input, context) => context.secrets?.API_TOKEN === 'super-secret-value' ? 'match' : 'missing' }, undefined, 'vault')
  }`)
  await server.start()
  const client = await DaemonParityClient.connect(join(root, 'daemon.sock'))
  const frames: string[] = []
  let id = 0
  const request = async (method: string, params: Record<string, unknown>) => {
    const current = ++id
    client.send({ jsonrpc: '2.0', id: current, method, params })
    const frame = await client.next(candidate => candidate.id === current)
    frames.push(JSON.stringify(frame))
    return frame.result
  }
  try {
    await request('initialize', { session_key: 'secrets', project_dir: root })
    expect(await request('slash', { command: '/plugins install vault.ts' })).toMatchObject({ ok: true })
    expect(await request('plugins.secret.status', { plugin: 'vault' })).toEqual({ ok: true, plugin: 'vault', declared: ['API_TOKEN'], set: [] })
    expect(await request('plugins.secret.set', { plugin: 'vault', name: 'OTHER', value: 'x' })).toMatchObject({ ok: false, error: expect.stringContaining('does not declare') })
    expect(await request('plugins.secret.set', { plugin: 'vault', name: 'API_TOKEN', value: '' })).toMatchObject({ ok: false })
    expect(await request('plugins.secret.set', { plugin: 'vault', name: 'API_TOKEN', value: 'super-secret-value' })).toEqual({ ok: true, plugin: 'vault', name: 'API_TOKEN', stored: true })
    expect(await request('plugins.secret.status', { plugin: 'vault' })).toMatchObject({ set: ['API_TOKEN'] })
    expect(frames.join('\n')).not.toContain('super-secret-value')
    expect(await Bun.file(secretsPath).text()).not.toContain('super-secret-value')
  } finally { manager.close(); client.close(); await server.stop(); await rm(root, { recursive: true, force: true }) }
})

interface Frame {
  readonly id?: number | string
  readonly method?: string
  readonly params?: {
    readonly payload?: Record<string, unknown>
    readonly type?: string
  }
  readonly result?: Record<string, unknown>
}

class DaemonParityClient {
  private buffer = ''
  private readonly frames: Frame[] = []
  private readonly waiters: Array<{ predicate: (frame: Frame) => boolean; resolve: (frame: Frame) => void }> = []

  private constructor(private readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', chunk => this.receive(typeof chunk === 'string' ? chunk : new TextDecoder().decode(chunk)))
  }

  static async connect(socketPath: string): Promise<DaemonParityClient> {
    const socket = connect({ path: socketPath })
    await new Promise<void>((resolveConnection, rejectConnection) => {
      socket.once('connect', resolveConnection)
      socket.once('error', rejectConnection)
    })
    return new DaemonParityClient(socket)
  }

  close(): void {
    this.socket.destroy()
  }

  next(predicate: (frame: Frame) => boolean): Promise<Frame> {
    const index = this.frames.findIndex(predicate)
    if (index >= 0) {
      const frame = this.frames.splice(index, 1)[0]
      if (frame) {
        return Promise.resolve(frame)
      }
    }
    return new Promise(resolveFrame => this.waiters.push({ predicate, resolve: resolveFrame }))
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
      if (line.trim()) {
        this.handle(JSON.parse(line) as Frame)
      }
      newline = this.buffer.indexOf('\n')
    }
  }

  private handle(frame: Frame): void {
    const index = this.waiters.findIndex(waiter => waiter.predicate(frame))
    const waiter = index >= 0 ? this.waiters.splice(index, 1)[0] : undefined
    if (waiter) {
      waiter.resolve(frame)
      return
    }
    this.frames.push(frame)
  }
}
