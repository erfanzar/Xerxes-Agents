// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ConfigurationError, ValidationError } from '../src/core/errors.js'
import { createLlmClient } from '../src/llms/client.js'
import { EditDiagnostics } from '../src/runtime/editDiagnostics.js'
import { enableLocalOnly, isLocalUrl, localOnlyActive, parseLocalOnlyConfig } from '../src/runtime/localOnly.js'

const CLI = join(import.meta.dir, '../src/cli.ts')

test('local_only comes from XERXES_LOCAL_ONLY, then runtime.local_only, and a bad value is refused', () => {
  expect(parseLocalOnlyConfig({}, {})).toBe(false)
  expect(parseLocalOnlyConfig({ local_only: true }, {})).toBe(true)
  expect(parseLocalOnlyConfig({ local_only: true }, { XERXES_LOCAL_ONLY: 'off' })).toBe(false)
  expect(parseLocalOnlyConfig({}, { XERXES_LOCAL_ONLY: 'Yes' })).toBe(true)
  expect(() => parseLocalOnlyConfig({ local_only: 'yes' }, {})).toThrow(ValidationError)
  expect(() => parseLocalOnlyConfig({}, { XERXES_LOCAL_ONLY: 'maybe' })).toThrow(ValidationError)
})

test('the guard lets fetch and WebSocket reach only this machine, and restore puts everything back', async () => {
  const reached: string[] = []
  const platformFetch = globalThis.fetch
  globalThis.fetch = Object.assign(async (input: Parameters<typeof fetch>[0]) => {
    reached.push(String(input instanceof Request ? input.url : input))
    return new Response('ok')
  }, platformFetch) as typeof fetch
  const env: Record<string, string | undefined> = { XERXES_MODELS_DEV: '1' }
  const restore = enableLocalOnly(env)
  try {
    expect(localOnlyActive()).toBe(true)
    expect(env).toMatchObject({ XERXES_LOCAL_ONLY: '1', XERXES_MODELS_DEV: '0', XERXES_CLAUDE_CODE_AUTOUPDATE: '0', XERXES_AUTO_INSTALL_CLAUDE_CODE: '0' })
    expect(await (await fetch('http://127.0.0.1:9/v1/models')).text()).toBe('ok')
    expect(await (await fetch(new URL('http://localhost:9/x'))).text()).toBe('ok')
    const refused = await fetch('https://api.example.com/v1/chat?key=secret').catch((error: unknown) => error)
    expect(refused).toBeInstanceOf(ConfigurationError)
    expect(String((refused as Error).message)).toContain('https://api.example.com')
    // The path and query, which may carry a token, are not repeated.
    expect(String((refused as Error).message)).not.toContain('secret')
    expect(reached).toEqual(['http://127.0.0.1:9/v1/models', 'http://localhost:9/x'])
    expect(() => new WebSocket('wss://chatgpt.example.com/socket')).toThrow(ConfigurationError)
  } finally {
    restore()
    globalThis.fetch = platformFetch
  }
  expect(localOnlyActive()).toBe(false)
  expect(env).toEqual({ XERXES_MODELS_DEV: '1' })
  expect(isLocalUrl('unix:///tmp/engine.sock')).toBe(true)
  expect(isLocalUrl('https://models.dev/api.json')).toBe(false)
})

test('local mode builds a model client only for a model served on this machine', () => {
  const restore = enableLocalOnly({})
  try {
    expect(() => createLlmClient('gpt-4o', { provider: 'openai', api_key: 'k' })).toThrow(/fully local mode refuses the model gpt-4o/)
    expect(() => createLlmClient('claude-code/sonnet', {})).toThrow(ConfigurationError)
    expect(() => createLlmClient('bonsai', { provider: 'openai', base_url: 'http://127.0.0.1:1234/v1', api_key: 'lm-studio' })).not.toThrow()
  } finally {
    restore()
  }
})

test('edit diagnostics install nothing in local mode', async () => {
  const commands: string[][] = []
  const diagnostics = new EditDiagnostics('/project', {
    checker: { source: 'tsc', command: ['bunx', 'tsc', '--noEmit'] },
    commandRunner: async command => {
      commands.push([...command])
      return { exitCode: 0, stdout: '', stderr: '' }
    },
  })
  const restore = enableLocalOnly({})
  try {
    await diagnostics.noteFileWillChange('/project/a.ts')
  } finally {
    restore()
  }
  expect(commands[0]).toEqual(['bunx', '--no-install', 'tsc', '--noEmit'])
})

/**
 * Loaded with `bun --preload` under the CLI: records every attempt to reach
 * another machine, through fetch, WebSocket, sockets, DNS or a network
 * command. Fetch, WebSocket and network commands are refused, not sent.
 */
const RECORDER = `
import { appendFileSync } from 'node:fs'
import dns from 'node:dns'
import net from 'node:net'
import tls from 'node:tls'
import childProcess from 'node:child_process'
const log = process.env.NETWORK_LOG
const record = what => appendFileSync(log, what + '\\n')
const local = host => !host || host === 'localhost' || host === '::1' || host === '[::1]' || /^127\\./.test(host) || host.endsWith('.localhost')
const hostOf = value => { try { return new URL(String(value)).hostname.replace(/^\\[|\\]$/g, '') } catch { return '' } }
const platformFetch = globalThis.fetch
globalThis.fetch = Object.assign((input, init) => {
  const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
  if (!url.startsWith('unix:') && !local(hostOf(url))) { record('fetch ' + url); return Promise.reject(new Error('recorder refused ' + url)) }
  return platformFetch(input, init)
}, platformFetch)
const PlatformWebSocket = globalThis.WebSocket
globalThis.WebSocket = class extends PlatformWebSocket {
  constructor(url, protocols) { if (!local(hostOf(url))) { record('websocket ' + url); throw new Error('recorder refused') } super(url, protocols) }
}
const target = args => {
  const first = args[0]
  if (typeof first === 'object' && first !== null) return first.path ? undefined : first.host ?? 'localhost'
  if (typeof first === 'string' && !/^\\d+$/.test(first)) return undefined
  return typeof args[1] === 'string' ? args[1] : 'localhost'
}
for (const [module, names] of [[net, ['connect', 'createConnection']], [tls, ['connect']]]) {
  for (const name of names) {
    const original = module[name]
    try { module[name] = (...args) => { const host = target(args); if (host !== undefined && !local(host)) record(name + ' ' + host); return original.apply(module, args) } } catch {}
  }
}
for (const name of ['lookup']) {
  const original = dns[name]
  try { dns[name] = (host, ...rest) => { if (!local(host)) record('dns ' + host); return original.call(dns, host, ...rest) } } catch {}
}
const NETWORK_COMMANDS = /(^|\\/)(claude|curl|wget|npx)$/
const gitRemote = argv => /(^|\\/)git$/.test(argv[0] ?? '') && argv.some(arg => ['fetch', 'pull', 'push', 'clone', 'ls-remote'].includes(arg))
const platformSpawn = Bun.spawn
try {
  Bun.spawn = (...args) => {
    const argv = Array.isArray(args[0]) ? args[0] : args[0]?.cmd ?? []
    if (NETWORK_COMMANDS.test(argv[0] ?? '') || gitRemote(argv) || (/(^|\\/)bash$/.test(argv[0] ?? '') && argv.join(' ').includes('curl'))) {
      record('spawn ' + argv.join(' '))
      throw new Error('recorder refused ' + argv[0])
    }
    return platformSpawn(...args)
  }
} catch {}
const platformChildSpawn = childProcess.spawn
try {
  childProcess.spawn = (command, ...rest) => {
    const argv = [command, ...(Array.isArray(rest[0]) ? rest[0] : [])]
    if (NETWORK_COMMANDS.test(command) || gitRemote(argv)) {
      record('spawn ' + argv.join(' '))
      throw new Error('recorder refused ' + command)
    }
    return platformChildSpawn.call(childProcess, command, ...rest)
  }
} catch {}
`

function sse(chunks: readonly Record<string, unknown>[]): Response {
  return new Response(chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n', {
    headers: { 'content-type': 'text/event-stream' },
  })
}

function bounded<T>(promise: Promise<T>, milliseconds = 15_000): Promise<T> {
  return Promise.race([promise, Bun.sleep(milliseconds).then(() => { throw new Error(`timed out after ${milliseconds} ms`) })])
}

type Frame = { readonly id?: number; readonly result?: Record<string, unknown>; readonly error?: unknown; readonly params?: { readonly type?: string } }

test.each(['oneshot', 'daemon'] as const)('fully local mode reaches no other machine: %s start, model list and a turn', async mode => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-local-only-'))
  const home = join(root, 'home')
  const log = join(root, 'network.log')
  const toolNames: string[][] = []
  const provider = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch(request) {
      if (new URL(request.url).pathname.endsWith('/models')) return Response.json({ data: [{ id: 'local-model' }] })
      const body = (await request.json()) as { tools?: Array<{ function?: { name?: string } }>; messages?: Array<{ role?: string }> }
      toolNames.push((body.tools ?? []).map(tool => tool.function?.name ?? ''))
      const answered = body.messages?.some(message => message.role === 'tool') ?? false
      return answered || mode === 'daemon'
        ? sse([{ choices: [{ delta: { content: 'stayed local' }, finish_reason: 'stop' }] }])
        // One-shot asks for the model list, which looks at every profile.
        : sse([{ choices: [{ delta: { tool_calls: [{ index: 0, id: 'models', type: 'function', function: { name: 'list_available_models', arguments: '{}' } }] }, finish_reason: 'tool_calls' }] }])
    },
  })
  let child: ReturnType<typeof Bun.spawn> | undefined
  let socket: Socket | undefined
  try {
    await mkdir(join(home, 'daemon'), { recursive: true })
    await writeFile(join(home, 'daemon', 'config.json'), JSON.stringify({ project_directory: root, runtime: {
      model: 'local-model', provider: 'openai', base_url: `${provider.url}v1`, api_key: 'local-key', permission_mode: 'accept-all',
      local_only: true,
    } }))
    const recorder = join(root, 'recorder.ts')
    await writeFile(recorder, RECORDER)
    await writeFile(log, '')
    const args = mode === 'daemon'
      ? ['daemon', '--project-dir', root, '--socket', join(root, 's.sock')]
      : ['List the models you can use.']
    const spawned = Bun.spawn([process.execPath, '--preload', recorder, CLI, ...args], {
      cwd: root,
      env: {
        ...process.env,
        XERXES_HOME: home,
        NETWORK_LOG: log,
        // On, so local mode is what turns them off.
        XERXES_MODELS_DEV: '1',
        XERXES_CLAUDE_CODE_AUTOUPDATE: '1',
        XERXES_AUTO_INSTALL_CLAUDE_CODE: '1',
      },
      stdin: 'pipe',
      stdout: 'pipe',
      stderr: 'pipe',
    })
    child = spawned
    const stderr = new Response(spawned.stderr).text()
    if (mode === 'oneshot') {
      const stdout = new Response(spawned.stdout).text()
      expect(await bounded(spawned.exited, 60_000)).toBe(0)
      expect(await stdout).toContain('stayed local')
    } else {
      const socketPath = join(root, 's.sock')
      for (let attempt = 0; attempt < 400 && !socket; attempt += 1) {
        try {
          socket = await new Promise<Socket>((resolve, reject) => {
            const candidate = connect(socketPath)
            candidate.once('error', reject)
            candidate.once('connect', () => resolve(candidate))
          })
        } catch {
          if (spawned.exitCode !== null) throw new Error(await stderr)
          await Bun.sleep(25)
        }
      }
      if (!socket) throw new Error('daemon did not open its socket')
      const open = socket
      let buffer = ''
      let nextId = 0
      let turnEnded = false
      const pending = new Map<number, (frame: Frame) => void>()
      open.setEncoding('utf8')
      open.on('data', chunk => {
        buffer += String(chunk)
        for (let end = buffer.indexOf('\n'); end >= 0; end = buffer.indexOf('\n')) {
          const line = buffer.slice(0, end)
          buffer = buffer.slice(end + 1)
          if (!line) continue
          const frame = JSON.parse(line) as Frame
          if (frame.params?.type === 'turn_end') turnEnded = true
          if (frame.id !== undefined) {
            pending.get(frame.id)?.(frame)
            pending.delete(frame.id)
          }
        }
      })
      const rpc = (method: string, params: Record<string, unknown> = {}) => bounded(new Promise<Frame>(resolve => {
        const id = ++nextId
        pending.set(id, resolve)
        open.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
      }))
      // initialize discovers the active profile's models; fetch_models is the picker.
      await rpc('initialize', { session_key: 'local-only', project_dir: root })
      await rpc('fetch_models', {})
      expect((await rpc('turn.submit', { text: 'Say where you ran.' })).error).toBeUndefined()
      for (let attempt = 0; attempt < 1000 && !turnEnded; attempt += 1) await Bun.sleep(10)
      expect(turnEnded).toBe(true)
    }
    // Nothing left this machine, and no web tools were offered.
    expect((await readFile(log, 'utf8')).trim()).toBe('')
    expect(toolNames.length).toBeGreaterThan(0)
    for (const names of toolNames) expect(names).not.toContain('WebScraper')
  } finally {
    socket?.destroy()
    if (child?.exitCode === null) {
      child.kill('SIGTERM')
      await child.exited
    }
    provider.stop(true)
    await rm(root, { recursive: true, force: true })
  }
}, 90_000)
