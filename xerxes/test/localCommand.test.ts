// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { createHash } from 'node:crypto'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ProfileStore, resolvedProfileContextLimit } from '../src/bridge/profiles.js'
import { GIB } from '../src/local/catalog.js'
import { saveLocalProfile } from '../src/local/cliEntry.js'
import { LOCAL_USAGE, runLocalCommand, type LocalCommandPorts, type SaveLocalProfileInput } from '../src/local/command.js'
import type { FetchPort } from '../src/local/download.js'
import type { HardwareProfile } from '../src/local/hardware.js'
import type { LmStudioEngine, LmStudioStartOptions } from '../src/local/lmstudio.js'
import type { LocalServer, ServerStatus, StartOptions } from '../src/local/server.js'
import { ErrorKind, classifyError } from '../src/runtime/errorClassifier.js'

let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'xerxes-local-cmd-'))
})
afterEach(async () => {
  await rm(dir, { recursive: true, force: true })
})

const hw16: HardwareProfile = {
  platform: 'linux',
  arch: 'x64',
  totalMemBytes: 16 * GIB,
  appleSilicon: false,
  gpu: { vendor: 'none' },
}
const hw8: HardwareProfile = { ...hw16, totalMemBytes: 8 * GIB }

const payload = new TextEncoder().encode('GGUF-model'.repeat(50))
const sha = createHash('sha256').update(payload).digest('hex')

interface Harness {
  readonly ports: LocalCommandPorts
  readonly out: string[]
  readonly err: string[]
  readonly profiles: SaveLocalProfileInput[]
  readonly starts: StartOptions[]
  readonly fetched: string[]
  status: ServerStatus
  binary: string | undefined
}

function harness(hardware: HardwareProfile, runtime: Record<string, unknown> = {}): Harness {
  const out: string[] = []
  const err: string[] = []
  const profiles: SaveLocalProfileInput[] = []
  const starts: StartOptions[] = []
  const fetched: string[] = []
  const h = { status: { running: false, healthy: false, stale: false } as ServerStatus, binary: '/opt/bin/llama-server' as string | undefined }
  const server: LocalServer = {
    findBinary: async () => h.binary,
    async start(options) {
      starts.push(options)
      h.status = { running: true, healthy: true, stale: false, pid: 4242, port: options.port, model: options.modelPath, startedAt: 'now', logPath: '/log' }
      return h.status
    },
    stop: async () => {
      if (!h.status.running) return { stopped: false, reason: 'not-running' }
      h.status = { running: false, healthy: false, stale: false }
      return { stopped: true, pid: 4242, forced: false }
    },
    status: async () => h.status,
  }
  const fetchPort: FetchPort = async (url) => {
    fetched.push(url)
    return new Response(payload, { status: 200 })
  }
  const ports: LocalCommandPorts = {
    hardware: async () => hardware,
    fetch: fetchPort,
    server,
    saveProfile: async (input) => {
      profiles.push(input)
    },
    stdout: (s) => out.push(s),
    stderr: (s) => err.push(s),
    config: { local: { models_dir: join(dir, 'models'), ...(runtime.local as object | undefined) }, ...runtime, ...(runtime.local === undefined ? {} : { local: { models_dir: join(dir, 'models'), ...(runtime.local as object) } }) },
    env: { XERXES_HOME: dir },
  }
  return Object.assign(h, { ports, out, err, profiles, starts, fetched }) as unknown as Harness
}

const json = (h: Harness): Record<string, unknown> => {
  expect(h.out).toHaveLength(1)
  return JSON.parse(h.out[0] ?? '') as Record<string, unknown>
}

describe('runLocalCommand', () => {
  test('no arguments prints usage', async () => {
    const h = harness(hw16)
    expect(await runLocalCommand([], h.ports)).toBe(0)
    expect(h.out.join('')).toBe(LOCAL_USAGE)
  })

  test('unknown command and unknown option exit 2', async () => {
    const h = harness(hw16)
    expect(await runLocalCommand(['bogus'], h.ports)).toBe(2)
    expect(await runLocalCommand(['status', '--wat'], h.ports)).toBe(2)
  })

  test('setup with no configured source exits non-zero with guidance and fakes nothing', async () => {
    const h = harness(hw16)
    const code = await runLocalCommand(['setup'], h.ports)
    expect(code).toBe(1)
    expect(h.err.join('')).toContain('sha256')
    expect(h.err.join('')).toContain('--model-file')
    expect(h.starts).toEqual([])
    expect(h.profiles).toEqual([])
    expect(h.fetched).toEqual([])
  })

  test('setup --json with a missing source prints one error object', async () => {
    const h = harness(hw16)
    expect(await runLocalCommand(['setup', '--json'], h.ports)).toBe(1)
    const body = json(h)
    expect(body.ok).toBe(false)
    expect(body.error).toMatchObject({ code: 'NO_SOURCE' })
  })

  test('setup below 12 GiB suggests a cloud profile and does nothing', async () => {
    const h = harness(hw8)
    expect(await runLocalCommand(['setup'], h.ports)).toBe(1)
    expect(h.err.join('')).toContain('cloud provider profile')
    expect(h.starts).toEqual([])
  })

  test('setup downloads, verifies, starts, and saves the lean local profile', async () => {
    const h = harness(hw16, {
      local: { models: { 'bonsai-9b': { url: 'https://models.test/b.gguf', sha256: sha, bytes: payload.byteLength } } },
    })
    expect(await runLocalCommand(['setup', '--json'], h.ports)).toBe(0)
    const body = json(h)
    expect(body).toMatchObject({
      ok: true,
      model: 'bonsai-9b',
      port: 11581,
      base_url: 'http://127.0.0.1:11581/v1',
      profile: 'bonsai-local',
      downloaded: true,
      started: true,
    })
    expect(h.fetched).toEqual(['https://models.test/b.gguf'])
    expect(h.starts).toEqual([{ modelPath: join(dir, 'models', 'bonsai-9b.gguf'), port: 11581, ctxSize: 32768 }])
    expect(h.profiles).toEqual([
      {
        name: 'bonsai-local',
        baseUrl: 'http://127.0.0.1:11581/v1',
        model: 'bonsai-9b',
        provider: 'custom',
        dataTrust: 'local',
        promptProfile: 'lean',
        contextLimit: 32768,
      },
    ])
    // progress went to stderr, never stdout
    expect(h.err.join('')).toContain('Downloading')
  })

  test('setup --model-file skips the download entirely', async () => {
    const file = join(dir, 'mine.gguf')
    await writeFile(file, Buffer.concat([Buffer.from('GGUF'), Buffer.alloc(8)]))
    const h = harness(hw16)
    expect(await runLocalCommand(['setup', '--model-file', file, '--json'], h.ports)).toBe(0)
    expect(json(h)).toMatchObject({ ok: true, model_path: file, downloaded: false })
    expect(h.fetched).toEqual([])
    expect(h.starts[0]?.modelPath).toBe(file)
    expect(h.profiles).toHaveLength(1)
  })

  test('setup with a hash mismatch fails and saves no profile', async () => {
    const h = harness(hw16, {
      local: { models: { 'bonsai-9b': { url: 'https://models.test/b.gguf', sha256: 'f'.repeat(64) } } },
    })
    expect(await runLocalCommand(['setup'], h.ports)).toBe(1)
    expect(h.err.join('')).toContain('sha256 mismatch')
    expect(h.starts).toEqual([])
    expect(h.profiles).toEqual([])
  })

  test('setup --model for a model larger than memory needs --yes', async () => {
    const file = join(dir, 'mine.gguf')
    await writeFile(file, Buffer.concat([Buffer.from('GGUF'), Buffer.alloc(8)]))
    const refused = harness(hw16)
    expect(await runLocalCommand(['setup', '--model', 'ternary-bonsai-2-27b', '--model-file', file], refused.ports)).toBe(1)
    expect(refused.err.join('')).toContain('--yes')
    const allowed = harness(hw16)
    expect(await runLocalCommand(['setup', '--model', 'ternary-bonsai-2-27b', '--model-file', file, '--yes'], allowed.ports)).toBe(0)
    expect(allowed.starts[0]?.ctxSize).toBe(16384)
  })

  test('an invalid local config exits 2 naming the key', async () => {
    const h = harness(hw16)
    const bad: LocalCommandPorts = { ...h.ports, config: { local: { port: 'nope' } } }
    expect(await runLocalCommand(['status', '--json'], bad)).toBe(2)
    expect(JSON.stringify(JSON.parse(h.out[0] ?? ''))).toContain('local.port')
  })

  test('status --json prints exactly one object; exit code follows running', async () => {
    const h = harness(hw16)
    expect(await runLocalCommand(['status', '--json'], h.ports)).toBe(1)
    expect(json(h)).toEqual({ ok: true, running: false, healthy: false, stale: false })

    const r = harness(hw16)
    r.status = { running: true, healthy: true, stale: false, pid: 7, port: 11581, model: '/m.gguf', startedAt: 't', logPath: '/l' }
    expect(await runLocalCommand(['status', '--json'], r.ports)).toBe(0)
    expect(json(r)).toMatchObject({ ok: true, running: true, healthy: true, pid: 7, port: 11581, started_at: 't' })
  })

  test('stop reports what happened', async () => {
    const h = harness(hw16)
    expect(await runLocalCommand(['stop', '--json'], h.ports)).toBe(0)
    expect(json(h)).toMatchObject({ ok: true, stopped: false, reason: 'not-running' })
    const r = harness(hw16)
    r.status = { running: true, healthy: true, stale: false, pid: 4242, port: 11581, model: '/m' }
    expect(await runLocalCommand(['stop', '--json'], r.ports)).toBe(0)
    expect(json(r)).toMatchObject({ ok: true, stopped: true, pid: 4242 })
  })

  test('start refuses when the model is not downloaded', async () => {
    const h = harness(hw16)
    expect(await runLocalCommand(['start'], h.ports)).toBe(1)
    expect(h.err.join('')).toContain('No model file')
    expect(h.starts).toEqual([])
  })

  test('start uses a downloaded, unconfigured-source model file and a running server is a no-op', async () => {
    const h = harness(hw16)
    await writeFile(join(dir, 'models', 'bonsai-9b.gguf'), Buffer.concat([Buffer.from('GGUF'), Buffer.alloc(8)])).catch(async () => {
      const { mkdir } = await import('node:fs/promises')
      await mkdir(join(dir, 'models'), { recursive: true })
      await writeFile(join(dir, 'models', 'bonsai-9b.gguf'), Buffer.concat([Buffer.from('GGUF'), Buffer.alloc(8)]))
    })
    expect(await runLocalCommand(['start', '--json'], h.ports)).toBe(0)
    expect(h.starts).toHaveLength(1)
    expect(json(h)).toMatchObject({ ok: true, already_running: false, running: true })

    h.out.length = 0
    expect(await runLocalCommand(['start', '--json'], h.ports)).toBe(0)
    expect(h.starts).toHaveLength(1)
    expect(json(h)).toMatchObject({ already_running: true })
  })

  test('doctor --json reports hardware, selection, binary, model and server; not ready exits 1', async () => {
    const h = harness(hw16)
    h.binary = undefined
    expect(await runLocalCommand(['doctor', '--json'], h.ports)).toBe(1)
    const body = json(h) as Record<string, any>
    expect(body.ready).toBe(false)
    expect(body.hardware).toMatchObject({ platform: 'linux', total_mem_bytes: 16 * GIB, gpu: { vendor: 'none' } })
    expect(body.selection).toMatchObject({ ok: true, model: 'bonsai-9b', basis: 'system-memory' })
    expect(body.llama_server).toEqual({ found: false })
    expect(body.model).toMatchObject({ id: 'bonsai-9b', present: false, source_configured: false })
    expect(body.server).toMatchObject({ running: false })
  })

  test('doctor verifies a configured model file and is ready when everything is present', async () => {
    const h = harness(hw16, {
      local: { models: { 'bonsai-9b': { url: 'https://models.test/b.gguf', sha256: sha, bytes: payload.byteLength } } },
    })
    const { mkdir } = await import('node:fs/promises')
    await mkdir(join(dir, 'models'), { recursive: true })
    await writeFile(join(dir, 'models', 'bonsai-9b.gguf'), payload)
    expect(await runLocalCommand(['doctor', '--json'], h.ports)).toBe(0)
    const body = json(h) as Record<string, any>
    expect(body.ready).toBe(true)
    expect(body.model).toMatchObject({ present: true, verified: true, source_configured: true })
    expect(body.llama_server).toEqual({ found: true, path: '/opt/bin/llama-server' })
  })
})

test('the saved local profile carries the llama-server window, so the runtime can compact before overflowing', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xerxes-local-profile-'))
  try {
    const store = new ProfileStore(join(dir, 'profiles.json'))
    saveLocalProfile(store, {
      name: 'bonsai-local',
      baseUrl: 'http://127.0.0.1:11581/v1',
      model: 'ternary-bonsai-2-27b',
      provider: 'custom',
      dataTrust: 'local',
      promptProfile: 'lean',
      contextLimit: 16384,
    })
    const saved = store.get('bonsai-local')
    expect(resolvedProfileContextLimit(saved, 'ternary-bonsai-2-27b')).toBe(16384)
    expect(saved?.prompt_profile).toBe('lean')
    expect(store.active()?.name).not.toBe('bonsai-local')
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test("llama.cpp's context overflow error is classified as an overflow, so compaction can rescue the turn", () => {
  const error = new Error('400 {"error":{"code":400,"message":"the request exceeds the available context size, try increasing it","type":"exceed_context_size_error"}}')
  expect(classifyError(error).kind).toBe(ErrorKind.CONTEXT_OVERFLOW)
})

describe('runLocalCommand with LM Studio', () => {
  const KEY = 'ternary-bonsai-2-27b-mlx'
  const mac24: HardwareProfile = {
    platform: 'darwin',
    arch: 'arm64',
    totalMemBytes: 24 * GIB,
    appleSilicon: true,
    gpu: { vendor: 'apple', name: 'Apple M4 Pro' },
  }

  interface LmHarness {
    readonly base: Harness
    readonly ports: LocalCommandPorts
    readonly starts: LmStudioStartOptions[]
    status: ServerStatus
    library: string[]
    lms: string | undefined
    /** Window LM Studio reports after a load. */
    loadedWindow: number
  }

  function lmHarness(hardware: HardwareProfile, runtime: Record<string, unknown> = {}): LmHarness {
    const base = harness(hardware, runtime)
    const starts: LmStudioStartOptions[] = []
    const h = {
      status: { running: false, healthy: false, stale: false } as ServerStatus,
      library: [KEY, 'qwen/qwen3-8b'],
      lms: '/Users/me/.lmstudio/bin/lms' as string | undefined,
      loadedWindow: 16384,
    }
    const lmstudio: LmStudioEngine = {
      findBinary: async () => h.lms,
      listModels: async () => h.library.map((modelKey) => ({ modelKey, maxContextLength: 262144 })),
      serverInfo: async () => ({ running: false, port: 1234 }),
      async start(options) {
        starts.push(options)
        h.status = { running: true, healthy: true, stale: false, port: options.port ?? 1234, model: options.identifier, startedAt: 'now', contextLength: h.loadedWindow }
        return h.status
      },
      async stop() {
        if (!h.status.running) return { stopped: false, reason: 'not-running' }
        h.status = { running: false, healthy: false, stale: false }
        return { stopped: true, model: KEY, unloaded: true, serverStopped: true }
      },
      status: async () => h.status,
    }
    return Object.assign(h, { base, starts, ports: { ...base.ports, lmstudio } })
  }

  test('setup --engine lmstudio loads the 27B MLX pack with one slot and saves the window LM Studio loaded', async () => {
    const h = lmHarness(mac24)
    h.loadedWindow = 16384
    expect(await runLocalCommand(['setup', '--engine', 'lmstudio', '--json'], h.ports)).toBe(0)
    expect(h.starts).toHaveLength(1)
    expect(h.starts[0]).toMatchObject({ modelKey: KEY, identifier: KEY, contextLength: 16384 })
    // No configured port: LM Studio keeps its own.
    expect(h.starts[0]?.port).toBeUndefined()
    expect(json(h.base)).toMatchObject({
      ok: true,
      engine: 'lmstudio',
      model: KEY,
      model_id: 'ternary-bonsai-2-27b',
      base_url: 'http://127.0.0.1:1234/v1',
      context_length: 16384,
      started: true,
    })
    expect(h.base.profiles).toEqual([
      {
        name: 'bonsai-local',
        baseUrl: 'http://127.0.0.1:1234/v1',
        model: KEY,
        provider: 'custom',
        dataTrust: 'local',
        promptProfile: 'lean',
        contextLimit: 16384,
      },
    ])
    // Nothing went through llama-server or the downloader.
    expect(h.base.starts).toEqual([])
    expect(h.base.fetched).toEqual([])
  })

  test('the runtime.local.engine key, ctx_size and port select the same path without flags', async () => {
    const h = lmHarness(mac24, { local: { engine: 'lmstudio', ctx_size: 42752, port: 1234 } })
    h.loadedWindow = 42752
    expect(await runLocalCommand(['setup', '--json'], h.ports)).toBe(0)
    expect(h.starts[0]).toMatchObject({ modelKey: KEY, contextLength: 42752, port: 1234 })
    expect(h.base.profiles[0]?.contextLimit).toBe(42752)
  })

  test('--ctx-size wins over the config and is bounded by the model', async () => {
    const h = lmHarness(mac24, { local: { engine: 'lmstudio', ctx_size: 8192 } })
    expect(await runLocalCommand(['start', '--ctx-size', '32768', '--json'], h.ports)).toBe(0)
    expect(h.starts[0]?.contextLength).toBe(32768)
    const tooBig = lmHarness(mac24)
    expect(await runLocalCommand(['setup', '--engine', 'lmstudio', '--ctx-size', '300000'], tooBig.ports)).toBe(2)
    expect(tooBig.starts).toEqual([])
  })

  test('--model-key loads any LM Studio model; a key not in the library is refused with the keys that are', async () => {
    const h = lmHarness(mac24)
    expect(await runLocalCommand(['setup', '--engine', 'lmstudio', '--model-key', 'qwen/qwen3-8b', '--json'], h.ports)).toBe(0)
    expect(h.starts[0]).toMatchObject({ modelKey: 'qwen/qwen3-8b', identifier: 'qwen/qwen3-8b' })
    // Unknown to the catalog and no ctx_size: LM Studio chooses, and its window is recorded.
    expect(h.starts[0]?.contextLength).toBeUndefined()
    expect(h.base.profiles[0]).toMatchObject({ model: 'qwen/qwen3-8b', contextLimit: 16384 })

    const missing = lmHarness(mac24)
    expect(await runLocalCommand(['setup', '--engine', 'lmstudio', '--model-key', 'nope'], missing.ports)).toBe(1)
    expect(missing.base.err.join('')).toContain(`LM Studio has no model with key "nope"`)
    expect(missing.base.err.join('')).toContain(KEY)
    expect(missing.starts).toEqual([])
    expect(missing.base.profiles).toEqual([])
  })

  test('a model with no LM Studio build, and llama-only or LM-Studio-only flags, are refused', async () => {
    const h = lmHarness(mac24)
    expect(await runLocalCommand(['setup', '--engine', 'lmstudio', '--model', 'bonsai-9b'], h.ports)).toBe(1)
    expect(h.base.err.join('')).toContain('--model-key')
    expect(await runLocalCommand(['setup', '--engine', 'lmstudio', '--model-file', '/m.gguf'], h.ports)).toBe(2)
    expect(await runLocalCommand(['setup', '--model-key', KEY], h.ports)).toBe(2)
    expect(await runLocalCommand(['setup', '--engine', 'ollama'], h.ports)).toBe(2)
    expect(h.starts).toEqual([])
  })

  test('without lms the guidance names LM Studio and nothing starts', async () => {
    const h = lmHarness(mac24)
    h.lms = undefined
    expect(await runLocalCommand(['setup', '--engine', 'lmstudio', '--json'], h.ports)).toBe(1)
    expect(json(h.base)).toMatchObject({ ok: false, error: { code: 'BINARY_NOT_FOUND' } })
    expect(h.starts).toEqual([])
  })

  test('a 16 GB Mac is below the MLX build budget unless --yes', async () => {
    const mac16: HardwareProfile = { ...mac24, totalMemBytes: 16 * GIB }
    const refused = lmHarness(mac16)
    expect(await runLocalCommand(['setup', '--engine', 'lmstudio'], refused.ports)).toBe(1)
    expect(refused.base.err.join('')).toContain('--model-key')
    const allowed = lmHarness(mac16)
    expect(await runLocalCommand(['setup', '--engine', 'lmstudio', '--model', 'ternary-bonsai-2-27b', '--yes'], allowed.ports)).toBe(0)
  })

  test('status and stop find the LM Studio load without --engine', async () => {
    const h = lmHarness(mac24)
    h.status = { running: true, healthy: true, stale: false, port: 1234, model: KEY, startedAt: 't', contextLength: 16384 }
    expect(await runLocalCommand(['status', '--json'], h.ports)).toBe(0)
    expect(json(h.base)).toMatchObject({ ok: true, running: true, engine: 'lmstudio', model: KEY, context_length: 16384 })
    h.base.out.length = 0
    expect(await runLocalCommand(['status'], h.ports)).toBe(0)
    expect(h.base.out.join('')).toContain('LM Studio, port 1234')
    h.base.out.length = 0
    expect(await runLocalCommand(['stop', '--json'], h.ports)).toBe(0)
    expect(json(h.base)).toMatchObject({ ok: true, stopped: true, engine: 'lmstudio', unloaded: true, server_stopped: true })
  })

  test('setup while Xerxes holds another LM Studio load is refused', async () => {
    const h = lmHarness(mac24)
    h.status = { running: true, healthy: true, stale: false, port: 1234, model: 'qwen/qwen3-8b', contextLength: 8192 }
    expect(await runLocalCommand(['setup', '--engine', 'lmstudio'], h.ports)).toBe(1)
    expect(h.base.err.join('')).toContain('xerxes local stop --engine lmstudio')
    expect(h.starts).toEqual([])
  })

  test('doctor --engine lmstudio reports lms, the library entry and readiness', async () => {
    const h = lmHarness(mac24)
    expect(await runLocalCommand(['doctor', '--engine', 'lmstudio', '--json'], h.ports)).toBe(0)
    const body = json(h.base) as Record<string, any>
    expect(body).toMatchObject({
      engine: 'lmstudio',
      ready: true,
      selection: { ok: true, model_key: KEY, model: 'ternary-bonsai-2-27b', context_length: 16384 },
      lms: { found: true },
      lmstudio_server: { running: false, port: 1234 },
      model: { key: KEY, present: true, max_context_length: 262144 },
    })
  })
})

test('llama-server setup records the window the server reports, not just the launch setting', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xerxes-local-window-'))
  try {
    const file = join(dir, 'mine.gguf')
    await writeFile(file, Buffer.concat([Buffer.from('GGUF'), Buffer.alloc(8)]))
    const out: string[] = []
    const err: string[] = []
    const profiles: SaveLocalProfileInput[] = []
    const probed: string[] = []
    const server: LocalServer = {
      findBinary: async () => '/opt/bin/llama-server',
      start: async (options) => ({ running: true, healthy: true, stale: false, pid: 1, port: options.port, model: options.modelPath }),
      stop: async () => ({ stopped: false, reason: 'not-running' }),
      status: async () => ({ running: false, healthy: false, stale: false }),
    }
    const ports: LocalCommandPorts = {
      hardware: async () => hw16,
      fetch: async () => new Response(''),
      server,
      loadedWindow: async (baseUrl) => {
        probed.push(baseUrl)
        return { ok: true, engine: 'llama-server', window: 30000 }
      },
      saveProfile: async (input) => {
        profiles.push(input)
      },
      stdout: (s) => out.push(s),
      stderr: (s) => err.push(s),
      config: { local: { models_dir: join(dir, 'models') } },
      env: { XERXES_HOME: dir },
    }
    expect(await runLocalCommand(['setup', '--model-file', file, '--json'], ports)).toBe(0)
    expect(probed).toEqual(['http://127.0.0.1:11581/v1'])
    expect(profiles[0]?.contextLimit).toBe(30000)
    expect(JSON.parse(out[0] ?? '{}')).toMatchObject({ engine: 'llama-server', context_length: 30000 })

    // A probe that fails falls back to the --ctx-size and says why on stderr.
    profiles.length = 0
    err.length = 0
    const failing: LocalCommandPorts = { ...ports, loadedWindow: async () => ({ ok: false, reason: 'no answer' }) }
    expect(await runLocalCommand(['setup', '--model-file', file], failing)).toBe(0)
    expect(profiles[0]?.contextLimit).toBe(32768)
    expect(err.join('')).toContain('no answer')
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})
