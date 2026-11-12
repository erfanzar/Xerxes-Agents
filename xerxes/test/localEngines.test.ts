// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { createHash } from 'node:crypto'
import { appendFileSync } from 'node:fs'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { BUILTIN_MODELS, GIB, type CatalogEntry } from '../src/local/catalog.js'
import { NO_ENGINE_HELP, runLocalCommand, type LocalCommandPorts, type SaveLocalProfileInput } from '../src/local/command.js'
import { checkLlamaCpp, checkMlxLm, compareVersions, parseLlamaCppVersion, parseMlxLmVersion } from '../src/local/engineVersion.js'
import type { HardwareProfile } from '../src/local/hardware.js'
import type { LmStudioEngine } from '../src/local/lmstudio.js'
import { mlxAttention } from '../src/local/mlxConfig.js'
import { createLocalServer, type LocalServer, type ManagedEngine, type ServerPorts, type ServerStatus, type StartOptions } from '../src/local/server.js'

let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'xerxes-local-engines-'))
})
afterEach(async () => {
  await rm(dir, { recursive: true, force: true })
})

const mac24: HardwareProfile = { platform: 'darwin', arch: 'arm64', totalMemBytes: 24 * GIB, appleSilicon: true, gpu: { vendor: 'apple' } }
const linux24: HardwareProfile = { platform: 'linux', arch: 'x64', totalMemBytes: 24 * GIB, appleSilicon: false, gpu: { vendor: 'none' } }

describe('engine versions', () => {
  test('llama.cpp releases report a version and a build; older builds only a build', () => {
    expect(parseLlamaCppVersion('version: 0.9.0 (build 11683, commit a1b2c3d)\nbuilt with clang for arm64')).toEqual({ version: '0.9.0', build: 11683 })
    expect(parseLlamaCppVersion('version: 11236 (a1b2c3d)\nbuilt with cc')).toEqual({ build: 11236 })
    // Built without git history: the build number is 0, so only the release version counts.
    expect(parseLlamaCppVersion('version: 0.9.0 (build 0, commit unknown)')).toEqual({ version: '0.9.0' })
    expect(parseLlamaCppVersion('llama-server: unknown option')).toEqual({})
  })

  test('the minimum is checked by build, else by release version, else it is unknown', () => {
    const minimum = { build: 11_555, version: '0.7.0' }
    expect(checkLlamaCpp({ build: 11_683, version: '0.9.0' }, minimum)).toMatchObject({ ok: true })
    expect(checkLlamaCpp({ build: 11_236 }, minimum)).toEqual({ ok: false, reason: 'too-old', found: 'build 11236' })
    expect(checkLlamaCpp({ version: '0.6.0' }, minimum)).toEqual({ ok: false, reason: 'too-old', found: '0.6.0' })
    expect(checkLlamaCpp({ version: '0.7.0' }, minimum)).toEqual({ ok: true, found: '0.7.0' })
    expect(checkLlamaCpp({}, minimum)).toEqual({ ok: false, reason: 'unknown' })
  })

  test('mlx-lm prints a bare version, compared numerically', () => {
    expect(parseMlxLmVersion('0.32.2\n')).toBe('0.32.2')
    expect(parseMlxLmVersion('error: no module')).toBeUndefined()
    expect(compareVersions('0.32.10', '0.32.2')).toBeGreaterThan(0)
    expect(checkMlxLm('0.32.0', '0.32.2')).toEqual({ ok: false, reason: 'too-old', found: '0.32.0' })
    expect(checkMlxLm('0.32.2', '0.32.2')).toEqual({ ok: true, found: '0.32.2' })
    expect(checkMlxLm(undefined, '0.32.2')).toEqual({ ok: false, reason: 'unknown' })
  })
})

describe('MLX model shape', () => {
  test("Ternary Bonsai 2 27B keeps a cache in one layer in four: 64 KiB per token", () => {
    const layerTypes = Array.from({ length: 64 }, (_, layer) => ((layer + 1) % 4 === 0 ? 'full_attention' : 'linear_attention'))
    const config = {
      model_type: 'prism_hadamard_qwen35',
      text_config: { num_hidden_layers: 64, full_attention_interval: 4, num_key_value_heads: 4, head_dim: 256, max_position_embeddings: 262_144, layer_types: layerTypes },
    }
    expect(mlxAttention('/m', config)).toEqual({
      architecture: 'prism_hadamard_qwen35', blockCount: 64, cachedLayers: 16, kvBytesPerToken: 65_536, contextLength: 262_144,
    })
  })

  test('a plain model caches every layer, and a config without a shape is an error', () => {
    expect(mlxAttention('/m', { model_type: 'llama', num_hidden_layers: 2, num_attention_heads: 8, hidden_size: 512 }).kvBytesPerToken).toBe(2 * 8 * 64 * 2 * 2)
    expect(() => mlxAttention('/m', { model_type: 'x' })).toThrow('num_hidden_layers is missing')
  })
})

interface FakeProcess {
  readonly spawned: string[][]
  readonly ran: string[][]
  readonly ports: ServerPorts
  versionOutput: string | undefined
  existing: Set<string>
}

function fakeProcesses(binaries: Record<string, string>): FakeProcess {
  const spawned: string[][] = []
  const ran: string[][] = []
  const alive = new Map<number, string>()
  let nextPid = 500
  const f: FakeProcess = {
    spawned,
    ran,
    versionOutput: undefined,
    existing: new Set(Object.values(binaries)),
    ports: undefined as unknown as ServerPorts,
  }
  const ports: ServerPorts = {
    which: name => binaries[name],
    exists: async path => f.existing.has(path),
    spawn(argv, { logPath }) {
      spawned.push([...argv])
      appendFileSync(logPath, 'starting\n')
      const pid = nextPid++
      alive.set(pid, argv.join(' '))
      return { pid, exited: new Promise<number>(() => undefined) }
    },
    fetch: async () => ({ status: 200 }),
    isPidAlive: pid => alive.has(pid),
    commandOf: async pid => alive.get(pid),
    signal: pid => { alive.delete(pid) },
    portInUse: async () => false,
    run: async argv => {
      ran.push([...argv])
      if (f.versionOutput === undefined) throw new Error('cannot run')
      return { exitCode: 0, output: f.versionOutput }
    },
    sleep: async () => undefined,
    now: () => Date.now(),
  }
  return Object.assign(f, { ports })
}

describe('createLocalServer for mlx_lm.server', () => {
  test('starts the folder with one request at a time and two prompt caches, in its own state file', async () => {
    const f = fakeProcesses({ 'mlx_lm.server': '/venv/bin/mlx_lm.server' })
    const server = createLocalServer(f.ports, { engine: 'mlx', stateDir: dir, pollMs: 1 })
    expect(server.engine).toBe('mlx')
    const status = await server.start({ modelPath: '/models/Ternary-Bonsai-2-27B-mlx-2bit', port: 11581, ctxSize: 40_000 })
    expect(f.spawned).toEqual([[
      '/venv/bin/mlx_lm.server', '--model', '/models/Ternary-Bonsai-2-27B-mlx-2bit', '--host', '127.0.0.1', '--port', '11581',
      '--decode-concurrency', '1', '--prompt-concurrency', '1', '--prompt-cache-size', '2',
    ]])
    expect(status).toMatchObject({ running: true, healthy: true, model: '/models/Ternary-Bonsai-2-27B-mlx-2bit' })
    const state = JSON.parse(await readFile(join(dir, 'mlx-server.json'), 'utf8'))
    expect(state).toMatchObject({ bin: '/venv/bin/mlx_lm.server', ctxSize: 40_000, logPath: join(dir, 'mlx-server.log') })
    // llama-server's record is separate, so the two never touch each other's process.
    expect(await createLocalServer(f.ports, { stateDir: dir }).status()).toEqual({ running: false, healthy: false, stale: false })
    expect(await server.stop()).toMatchObject({ stopped: true })
  })

  test('the version comes from mlx_lm beside the server, and is unknown when that is missing', async () => {
    const f = fakeProcesses({ 'mlx_lm.server': '/venv/bin/mlx_lm.server' })
    f.versionOutput = '0.32.2\n'
    const server = createLocalServer(f.ports, { engine: 'mlx', stateDir: dir })
    expect(await server.version()).toBeUndefined()
    f.existing.add('/venv/bin/mlx_lm')
    expect(await server.version()).toBe('0.32.2')
    expect(f.ran).toEqual([['/venv/bin/mlx_lm', '--version']])
  })

  test("llama-server's version is its own --version, and a missing binary names mlx-lm's install", async () => {
    const f = fakeProcesses({ 'llama-server': '/opt/bin/llama-server' })
    f.versionOutput = 'version: 0.9.0 (build 11683, commit a1b2c3d)'
    expect(await createLocalServer(f.ports, { stateDir: dir }).version()).toContain('build 11683')
    expect(f.ran).toEqual([['/opt/bin/llama-server', '--version']])
    const missing = await createLocalServer(fakeProcesses({}).ports, { engine: 'mlx', stateDir: dir })
      .start({ modelPath: '/m', port: 1, ctxSize: 1 }).catch((error: unknown) => error)
    expect(String(missing)).toContain('uv tool install mlx-lm')
  })
})

// A tiny MLX build so setup can download and verify it in a test.
const mlxFiles = { 'config.json': '{"model_type":"llama","num_hidden_layers":2,"num_attention_heads":8,"hidden_size":512}', 'model.safetensors': 'weights'.repeat(64) }
const pinned = Object.entries(mlxFiles).map(([path, content]) => ({
  path, sha256: createHash('sha256').update(content).digest('hex'), bytes: Buffer.byteLength(content),
}))
const tinyMlx: CatalogEntry = {
  id: 'ternary-bonsai-2-27b',
  label: 'Tiny',
  fileName: 'tiny.gguf',
  minMemoryBytes: 1 * GIB,
  ctxSize: 8192,
  nativeContext: 262_144,
  mlx: { repo: 'prism-ml/tiny-mlx', revision: 'abc123', dirName: 'tiny-mlx', files: pinned, minMemoryBytes: 1 * GIB, minMlxLm: '0.32.2' },
}

interface EngineHarness {
  readonly ports: LocalCommandPorts
  readonly out: string[]
  readonly err: string[]
  readonly profiles: SaveLocalProfileInput[]
  readonly fetched: string[]
  readonly starts: Record<ManagedEngine | 'lmstudio', StartOptions[]>
  binaries: { llama: string | undefined; mlx: string | undefined; lms: string | undefined }
  versions: { llama: string | undefined; mlx: string | undefined }
  statuses: Record<ManagedEngine, ServerStatus>
}

function engineHarness(hardware: HardwareProfile, options: { catalog?: readonly CatalogEntry[] } = {}): EngineHarness {
  const out: string[] = []
  const err: string[] = []
  const profiles: SaveLocalProfileInput[] = []
  const fetched: string[] = []
  const starts: EngineHarness['starts'] = { 'llama-server': [], mlx: [], lmstudio: [] }
  const h = {
    binaries: { llama: '/opt/bin/llama-server', mlx: '/venv/bin/mlx_lm.server', lms: '/Users/me/.lmstudio/bin/lms' } as EngineHarness['binaries'],
    versions: { llama: 'version: 0.9.0 (build 11683, commit a1b2c3d)', mlx: '0.32.2' } as EngineHarness['versions'],
    statuses: { 'llama-server': { running: false, healthy: false, stale: false }, mlx: { running: false, healthy: false, stale: false } } as Record<ManagedEngine, ServerStatus>,
  }
  const server = (engine: ManagedEngine): LocalServer => ({
    engine,
    findBinary: async () => (engine === 'mlx' ? h.binaries.mlx : h.binaries.llama),
    version: async () => (engine === 'mlx' ? h.versions.mlx : h.versions.llama),
    async start(start) {
      starts[engine].push(start)
      h.statuses[engine] = { running: true, healthy: true, stale: false, pid: 77, port: start.port, model: start.modelPath }
      return h.statuses[engine]
    },
    async stop() {
      if (!h.statuses[engine].running) return { stopped: false, reason: 'not-running' }
      h.statuses[engine] = { running: false, healthy: false, stale: false }
      return { stopped: true, pid: 77, forced: false }
    },
    status: async () => h.statuses[engine],
    recover: async () => ({ recovered: false, reason: 'not-managed' as const }),
  })
  const lmstudio: LmStudioEngine = {
    findBinary: async () => h.binaries.lms,
    listModels: async () => [{ modelKey: 'ternary-bonsai-2-27b-mlx', maxContextLength: 262_144 }],
    serverInfo: async () => ({ running: false, port: 1234 }),
    async start(start) {
      starts.lmstudio.push({ modelPath: start.modelKey, port: 1234, ctxSize: start.contextLength ?? 0 })
      return { running: true, healthy: true, stale: false, port: 1234, model: start.identifier, contextLength: start.contextLength ?? 16_384 }
    },
    stop: async () => ({ stopped: false, reason: 'not-running' }),
    status: async () => ({ running: false, healthy: false, stale: false }),
    recover: async () => ({ recovered: false, reason: 'not-managed' as const }),
  }
  const ports: LocalCommandPorts = {
    hardware: async () => hardware,
    fetch: async url => {
      fetched.push(url)
      const name = url.split('/').at(-1) ?? ''
      const content = (mlxFiles as Record<string, string>)[name] ?? 'not the real weights'
      return new Response(content, { status: 200 })
    },
    server: server('llama-server'),
    mlx: server('mlx'),
    lmstudio,
    ...(options.catalog === undefined ? {} : { catalog: () => options.catalog! }),
    saveProfile: async input => { profiles.push(input) },
    stdout: text => out.push(text),
    stderr: text => err.push(text),
    config: { local: { models_dir: join(dir, 'models') } },
    env: { XERXES_HOME: dir },
  }
  return Object.assign(h, { ports, out, err, profiles, fetched, starts })
}

const body = (h: EngineHarness): Record<string, any> => JSON.parse(h.out.at(-1) ?? '{}')

describe('xerxes local picks engines in the order llama-server, mlx, LM Studio', () => {
  test('with llama-server installed it is used, and the 27B downloads from the pinned Hugging Face file', async () => {
    const h = engineHarness(mac24)
    expect(await runLocalCommand(['setup', '--json'], h.ports)).toBe(1)
    expect(h.err.join('')).toContain('Engine: llama-server')
    expect(h.err.join('')).toContain('Model: ternary-bonsai-2-27b')
    // The fake serves other bytes, so the pinned size refuses them and nothing starts.
    expect(h.fetched).toEqual([BUILTIN_MODELS.find(entry => entry.id === 'ternary-bonsai-2-27b')!.source!.url])
    expect(body(h).error).toMatchObject({ code: 'SIZE_MISMATCH' })
    expect(h.starts['llama-server']).toEqual([])
    expect(h.profiles).toEqual([])
  })

  test('without llama-server, Apple silicon uses mlx; elsewhere it falls through to LM Studio', async () => {
    const mac = engineHarness(mac24, { catalog: [tinyMlx] })
    mac.binaries.llama = undefined
    expect(await runLocalCommand(['setup', '--json'], mac.ports)).toBe(0)
    expect(body(mac)).toMatchObject({ engine: 'mlx' })

    const linux = engineHarness(linux24)
    linux.binaries.llama = undefined
    expect(await runLocalCommand(['setup', '--json'], linux.ports)).toBe(0)
    expect(body(linux)).toMatchObject({ engine: 'lmstudio' })
    expect(linux.starts.mlx).toEqual([])
  })

  test('with no engine installed, setup names all three and downloads nothing', async () => {
    const h = engineHarness(mac24)
    h.binaries = { llama: undefined, mlx: undefined, lms: undefined }
    expect(await runLocalCommand(['setup', '--json'], h.ports)).toBe(1)
    expect(body(h).error).toEqual({ code: 'BINARY_NOT_FOUND', message: NO_ENGINE_HELP })
    expect(h.fetched).toEqual([])
  })

  test('a llama.cpp build older than the 27B needs is refused before any download; --yes goes on', async () => {
    const h = engineHarness(mac24)
    h.versions.llama = 'version: 11236 (deadbee)'
    expect(await runLocalCommand(['setup', '--json'], h.ports)).toBe(1)
    expect(body(h).error.code).toBe('ENGINE_TOO_OLD')
    expect(body(h).error.message).toContain('b11555')
    expect(body(h).error.message).toContain('gibberish')
    expect(h.fetched).toEqual([])

    const unknown = engineHarness(mac24)
    unknown.versions.llama = undefined
    expect(await runLocalCommand(['setup', '--json'], unknown.ports)).toBe(1)
    expect(body(unknown).error.code).toBe('ENGINE_VERSION_UNKNOWN')

    const forced = engineHarness(mac24)
    forced.versions.llama = 'version: 11236 (deadbee)'
    expect(await runLocalCommand(['setup', '--yes', '--json'], forced.ports)).toBe(1)
    expect(forced.err.join('')).toContain('Continuing because of --yes')
    expect(forced.fetched).toHaveLength(1)
  })
})

describe('xerxes local with --engine mlx', () => {
  test('setup downloads each pinned file, starts the folder and saves it as the model name', async () => {
    const h = engineHarness(mac24, { catalog: [tinyMlx] })
    expect(await runLocalCommand(['setup', '--engine', 'mlx', '--ctx-size', '32768', '--json'], h.ports)).toBe(0)
    const folder = join(dir, 'models', 'tiny-mlx')
    expect(h.fetched).toEqual([
      'https://huggingface.co/prism-ml/tiny-mlx/resolve/abc123/config.json',
      'https://huggingface.co/prism-ml/tiny-mlx/resolve/abc123/model.safetensors',
    ])
    expect(await readFile(join(folder, 'config.json'), 'utf8')).toBe(mlxFiles['config.json'])
    expect(h.starts.mlx).toEqual([{ modelPath: folder, port: 11581, ctxSize: 32768 }])
    expect(h.profiles).toEqual([{
      name: 'bonsai-local', baseUrl: 'http://127.0.0.1:11581/v1', model: folder, provider: 'custom', dataTrust: 'local', promptProfile: 'lean', contextLimit: 32768,
    }])
    expect(body(h)).toMatchObject({ ok: true, engine: 'mlx', model: folder, model_id: 'ternary-bonsai-2-27b', downloaded: true, started: true })

    // status and stop find the mlx record without being told the engine.
    h.out.length = 0
    expect(await runLocalCommand(['status', '--json'], h.ports)).toBe(0)
    expect(body(h)).toMatchObject({ running: true, engine: 'mlx', model: folder })
    h.out.length = 0
    expect(await runLocalCommand(['stop', '--json'], h.ports)).toBe(0)
    expect(body(h)).toMatchObject({ stopped: true, engine: 'mlx' })
  })

  test('the window is sized from free memory and the config when ctx_size is unset', async () => {
    const h = engineHarness(mac24, { catalog: [tinyMlx] })
    const read: string[] = []
    const ports: LocalCommandPorts = {
      ...h.ports,
      memoryFit: {
        freeBytes: async () => 12 * GIB,
        attention: async () => { throw new Error('not a gguf') },
        weightsBytes: () => 0,
        mlxAttention: async folder => {
          read.push(folder)
          return { architecture: 'x', blockCount: 1, cachedLayers: 1, kvBytesPerToken: 65_536 }
        },
      },
    }
    expect(await runLocalCommand(['setup', '--engine', 'mlx', '--json'], ports)).toBe(0)
    expect(read).toEqual([join(dir, 'models', 'tiny-mlx')])
    const window = h.starts.mlx[0]!.ctxSize
    expect(window).toBeGreaterThan(8192)
    expect(h.profiles[0]?.contextLimit).toBe(window)
    expect(h.err.join('')).toContain('KiB of cache per token')
  })

  test('mlx-lm older than the model needs, a non-Apple machine and missing files are refused', async () => {
    const old = engineHarness(mac24, { catalog: [tinyMlx] })
    old.versions.mlx = '0.32.0'
    expect(await runLocalCommand(['setup', '--engine', 'mlx', '--json'], old.ports)).toBe(1)
    expect(body(old).error.code).toBe('ENGINE_TOO_OLD')
    expect(old.fetched).toEqual([])

    const linux = engineHarness(linux24, { catalog: [tinyMlx] })
    expect(await runLocalCommand(['setup', '--engine', 'mlx', '--json'], linux.ports)).toBe(1)
    expect(body(linux).error.message).toContain('Apple silicon only')

    const notDownloaded = engineHarness(mac24, { catalog: [tinyMlx] })
    expect(await runLocalCommand(['start', '--engine', 'mlx', '--json'], notDownloaded.ports)).toBe(1)
    expect(body(notDownloaded).error.code).toBe('MODEL_MISSING')
    expect(notDownloaded.starts.mlx).toEqual([])
  })

  test('a changed file is not started', async () => {
    const h = engineHarness(mac24, { catalog: [tinyMlx] })
    const folder = join(dir, 'models', 'tiny-mlx')
    await mkdir(folder, { recursive: true })
    await writeFile(join(folder, 'config.json'), mlxFiles['config.json'])
    await writeFile(join(folder, 'model.safetensors'), 'tampered')
    expect(await runLocalCommand(['start', '--engine', 'mlx', '--json'], h.ports)).toBe(1)
    expect(body(h).error.code).toBe('MODEL_NOT_VERIFIED')
    expect(h.starts.mlx).toEqual([])
  })
})
