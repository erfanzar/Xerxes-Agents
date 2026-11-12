// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, test } from 'bun:test'

import { BUILTIN_MODELS, GIB, catalogWithConfig, launchCtxSize, parseLocalConfig, selectModel } from '../src/local/catalog.js'
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
    // The 27B needs 20 GiB usable: a 24 GB Mac after the reserve.
    const at24 = selectModel(apple(24), catalog)
    expect(at24.ok && at24.entry.id).toBe('ternary-bonsai-2-27b')
    const at23 = selectModel(apple(23), catalog)
    expect(at23.ok && at23.entry.id).toBe('bonsai-9b')
    expect(selectModel(apple(15), catalog).ok).toBe(false)
  })

  test('NVIDIA uses VRAM, not system RAM', () => {
    const big = hw({ totalMemBytes: 128 * GIB, gpu: { vendor: 'nvidia', vramBytes: 8 * GIB } })
    const refused = selectModel(big, catalog)
    expect(refused.ok).toBe(false)
    expect(refused.basis).toBe('vram')
    const roomy = selectModel(hw({ totalMemBytes: 16 * GIB, gpu: { vendor: 'nvidia', vramBytes: 24 * GIB } }), catalog)
    expect(roomy.ok && roomy.entry.id).toBe('ternary-bonsai-2-27b')
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
  test('a built-in source is only a pinned Hugging Face file with its sha256 and size', () => {
    const big = BUILTIN_MODELS.find((m) => m.id === 'ternary-bonsai-2-27b')
    expect(big?.source).toEqual({
      url: 'https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/839bd57023c1558945afea70a74fa83c054f35f9/Ternary-Bonsai-2-27B-Q2_0.gguf',
      sha256: '4f99aed01b8a877e153f9aa6569a4440fe17c59701cc0953e36d7f460549e70e',
      bytes: 7_626_008_928,
    })
    expect(big?.minLlamaCpp).toEqual({ build: 11_555, version: '0.7.0' })
    expect(big?.mlx?.files.map((file) => file.path)).toContain('model.safetensors')
    for (const file of big?.mlx?.files ?? []) expect(file.sha256).toMatch(/^[0-9a-f]{64}$/)
    // A model without a published build has no source until the user configures one.
    expect(BUILTIN_MODELS.find((m) => m.id === 'bonsai-9b')?.source).toBeUndefined()
    expect(BUILTIN_MODELS.map((m) => m.minMemoryBytes)).toEqual([12 * GIB, 20 * GIB])
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
    // The 27B keeps its built-in source; a configured one would replace it.
    expect(merged.find((m) => m.id === 'ternary-bonsai-2-27b')?.source?.url).toStartWith('https://huggingface.co/prism-ml/')
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

test('local.ctx_size raises the launch window up to the model\'s own, and refuses more', () => {
  const twentySeven = BUILTIN_MODELS.find((entry) => entry.id === 'ternary-bonsai-2-27b')
  if (twentySeven === undefined) throw new Error('Ternary Bonsai 2 27B is missing from the catalog')
  // Ternary Bonsai 2 27B reads 262K tokens; the 16K default is a memory choice, not its limit.
  expect(twentySeven.nativeContext).toBe(262144)
  expect(launchCtxSize(twentySeven, {})).toBe(16384)
  const configured = parseLocalConfig({ local: { ctx_size: 131072 } }, {})
  expect(launchCtxSize(twentySeven, configured)).toBe(131072)
  expect(parseLocalConfig({ local: { ctx_size: 4096 } }, { XERXES_LOCAL_CTX_SIZE: '65536' }).ctxSize).toBe(65536)
  expect(() => launchCtxSize(twentySeven, { ctxSize: 524288 })).toThrow(/262144-token window/)
  expect(() => parseLocalConfig({ local: { ctx_size: 100 } }, {})).toThrow(/local\.ctx_size/)
  expect(() => parseLocalConfig({}, { XERXES_LOCAL_CTX_SIZE: 'big' })).toThrow(/XERXES_LOCAL_CTX_SIZE/)
})
