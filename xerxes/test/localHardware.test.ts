// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, test } from 'bun:test'

import { BUILTIN_MODELS, GIB, catalogWithConfig, parseLocalConfig, selectModel } from '../src/local/catalog.js'
import { parseNvidiaSmi, probeHardware, type HardwareProfile, type HardwarePorts } from '../src/local/hardware.js'

function ports(
  over: { platform: NodeJS.Platform; arch: string; mem?: number },
  answers: Record<string, { exitCode: number; stdout: string } | 'throw'>,
): HardwarePorts {
  return {
    platform: over.platform,
    arch: over.arch,
    totalMemBytes: () => over.mem ?? 8 * GIB,
    async run(argv) {
      const answer = answers[argv.join(' ')]
      if (answer === undefined || answer === 'throw') throw new Error(`spawn ${argv[0]} ENOENT`)
      return answer
    },
  }
}

describe('probeHardware', () => {
  test('Apple silicon reads memsize and brand from sysctl', async () => {
    const hw = await probeHardware(
      ports(
        { platform: 'darwin', arch: 'arm64' },
        {
          'sysctl -n hw.memsize': { exitCode: 0, stdout: `${64 * GIB}\n` },
          'sysctl -n machdep.cpu.brand_string': { exitCode: 0, stdout: 'Apple M3 Max\n' },
        },
      ),
    )
    expect(hw).toEqual({
      platform: 'darwin',
      arch: 'arm64',
      totalMemBytes: 64 * GIB,
      appleSilicon: true,
      gpu: { vendor: 'apple', name: 'Apple M3 Max' },
    })
  })

  test('Intel Mac is not Apple silicon and has no GPU entry', async () => {
    const hw = await probeHardware(
      ports(
        { platform: 'darwin', arch: 'x64' },
        {
          'sysctl -n hw.memsize': { exitCode: 0, stdout: `${16 * GIB}\n` },
          'sysctl -n machdep.cpu.brand_string': { exitCode: 0, stdout: 'Intel(R) Core(TM) i9\n' },
        },
      ),
    )
    expect(hw.appleSilicon).toBe(false)
    expect(hw.gpu.vendor).toBe('none')
    expect(hw.totalMemBytes).toBe(16 * GIB)
  })

  test('NVIDIA VRAM is parsed from MiB and the largest GPU wins', async () => {
    const hw = await probeHardware(
      ports(
        { platform: 'linux', arch: 'x64', mem: 32 * GIB },
        {
          'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits': {
            exitCode: 0,
            stdout: 'NVIDIA GeForce RTX 3060, 12288\nNVIDIA RTX 4090, 24564\n',
          },
        },
      ),
    )
    expect(hw.gpu).toEqual({ vendor: 'nvidia', name: 'NVIDIA RTX 4090', vramBytes: 24564 * 1024 * 1024 })
    expect(hw.totalMemBytes).toBe(32 * GIB)
  })

  test('a missing probe binary means no GPU and never throws', async () => {
    const hw = await probeHardware(ports({ platform: 'linux', arch: 'x64', mem: 16 * GIB }, {}))
    expect(hw.gpu).toEqual({ vendor: 'none' })
    expect(hw.totalMemBytes).toBe(16 * GIB)
  })

  test('failed or garbled sysctl falls back to the injected memory figure', async () => {
    const hw = await probeHardware(
      ports(
        { platform: 'darwin', arch: 'arm64', mem: 8 * GIB },
        {
          'sysctl -n hw.memsize': { exitCode: 1, stdout: '' },
          'sysctl -n machdep.cpu.brand_string': 'throw',
        },
      ),
    )
    expect(hw.totalMemBytes).toBe(8 * GIB)
    expect(hw.appleSilicon).toBe(true)
  })

  test('parseNvidiaSmi ignores junk lines', () => {
    expect(parseNvidiaSmi('garbage\n, 100\nGPU, abc\n')).toBeUndefined()
  })
})

function hw(over: Partial<HardwareProfile> & { totalMemBytes: number }): HardwareProfile {
  return { platform: 'linux', arch: 'x64', appleSilicon: false, gpu: { vendor: 'none' }, ...over }
}

describe('selectModel', () => {
  const catalog = BUILTIN_MODELS

  test('Apple silicon reserves 4 GiB for the OS', () => {
    const apple = (gib: number) => hw({ totalMemBytes: gib * GIB, appleSilicon: true, gpu: { vendor: 'apple' } })
    const at16 = selectModel(apple(16), catalog)
    expect(at16.ok && at16.entry.id).toBe('bonsai-9b')
    const at28 = selectModel(apple(28), catalog)
    expect(at28.ok && at28.entry.id).toBe('ternary-bonsai-27b')
    const at27 = selectModel(apple(27), catalog)
    expect(at27.ok && at27.entry.id).toBe('bonsai-9b')
    expect(selectModel(apple(15), catalog).ok).toBe(false)
  })

  test('NVIDIA uses VRAM, not system RAM', () => {
    const big = hw({ totalMemBytes: 128 * GIB, gpu: { vendor: 'nvidia', vramBytes: 8 * GIB } })
    const refused = selectModel(big, catalog)
    expect(refused.ok).toBe(false)
    expect(refused.basis).toBe('vram')
    const roomy = selectModel(hw({ totalMemBytes: 16 * GIB, gpu: { vendor: 'nvidia', vramBytes: 24 * GIB } }), catalog)
    expect(roomy.ok && roomy.entry.id).toBe('ternary-bonsai-27b')
  })

  test('other machines use system RAM and below 12 GiB suggest a cloud profile', () => {
    const fits = selectModel(hw({ totalMemBytes: 12 * GIB }), catalog)
    expect(fits.ok && fits.entry.id).toBe('bonsai-9b')
    const small = selectModel(hw({ totalMemBytes: 8 * GIB }), catalog)
    expect(small.ok).toBe(false)
    if (!small.ok) {
      expect(small.reason).toContain('cloud provider profile')
      expect(small.basis).toBe('system-memory')
    }
  })
})

describe('catalog', () => {
  test('built-ins carry no download source', () => {
    for (const entry of BUILTIN_MODELS) expect(entry.source).toBeUndefined()
    expect(BUILTIN_MODELS.map((m) => m.minMemoryBytes)).toEqual([12 * GIB, 24 * GIB])
  })

  test('config sources merge onto matching entries only', () => {
    const cfg = parseLocalConfig(
      { local: { models: { 'bonsai-9b': { url: 'https://example.test/b.gguf', sha256: 'A'.repeat(64), bytes: 10 } } } },
      { XERXES_HOME: '/tmp/xh' },
    )
    const merged = catalogWithConfig(cfg)
    expect(merged.find((m) => m.id === 'bonsai-9b')?.source).toEqual({
      url: 'https://example.test/b.gguf',
      sha256: 'a'.repeat(64),
      bytes: 10,
    })
    expect(merged.find((m) => m.id === 'ternary-bonsai-27b')?.source).toBeUndefined()
  })
})

describe('parseLocalConfig', () => {
  test('defaults', () => {
    const cfg = parseLocalConfig({}, { XERXES_HOME: '/tmp/xh' })
    expect(cfg.modelsDir).toBe('/tmp/xh/local/models')
    expect(cfg.port).toBe(11581)
    expect(cfg.llamaServerPath).toBeUndefined()
    expect(cfg.models).toEqual({})
  })

  test('env wins over the file', () => {
    const cfg = parseLocalConfig(
      { local: { port: 9000, llama_server_path: '/a/llama-server', models_dir: '/data/models' } },
      { XERXES_LOCAL_PORT: '9100', XERXES_LOCAL_LLAMA_SERVER: '/b/llama-server', XERXES_LOCAL_MODELS_DIR: '/env/models' },
    )
    expect(cfg).toMatchObject({ port: 9100, llamaServerPath: '/b/llama-server', modelsDir: '/env/models' })
  })

  test('invalid values are ValidationErrors naming the key', () => {
    const bad = (runtime: Record<string, unknown>) => () => parseLocalConfig(runtime, {})
    expect(bad({ local: { port: 99999 } })).toThrow('local.port')
    expect(bad({ local: { nope: 1 } })).toThrow('local.nope')
    expect(bad({ local: { models: { 'bonsai-9b': { url: 'http://x.test/m.gguf', sha256: 'a'.repeat(64) } } } })).toThrow(
      'local.models.bonsai-9b.url',
    )
    expect(bad({ local: { models: { 'bonsai-9b': { url: 'https://x.test/m.gguf', sha256: 'zz' } } } })).toThrow(
      'local.models.bonsai-9b.sha256',
    )
    expect(bad({ local: { models: { 'made-up': { url: 'https://x.test/m.gguf', sha256: 'a'.repeat(64) } } } })).toThrow(
      'local.models.made-up',
    )
    expect(bad({ local: { models_dir: 'relative/dir' } })).toThrow('local.models_dir')
  })
})
