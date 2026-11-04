// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { createHash } from 'node:crypto'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { GIB } from '../src/local/catalog.js'
import { LOCAL_USAGE, runLocalCommand, type LocalCommandPorts, type SaveLocalProfileInput } from '../src/local/command.js'
import type { FetchPort } from '../src/local/download.js'
import type { HardwareProfile } from '../src/local/hardware.js'
import type { LocalServer, ServerStatus, StartOptions } from '../src/local/server.js'

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
    expect(await runLocalCommand(['setup', '--model', 'ternary-bonsai-27b', '--model-file', file], refused.ports)).toBe(1)
    expect(refused.err.join('')).toContain('--yes')
    const allowed = harness(hw16)
    expect(await runLocalCommand(['setup', '--model', 'ternary-bonsai-27b', '--model-file', file, '--yes'], allowed.ports)).toBe(0)
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
