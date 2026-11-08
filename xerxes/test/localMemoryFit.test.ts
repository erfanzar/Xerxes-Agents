// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { GgufMetadataError, ggufAttention, readGgufMetadata } from '../src/local/ggufMetadata.js'
import type { HardwareProfile } from '../src/local/hardware.js'
import { FIT_RESERVE_BYTES, MIN_FIT_WINDOW, fitWindow, freeMemoryBytes, parseVmStat } from '../src/local/memoryFit.js'

const GIB = 1024 ** 3

/** A GGUF v3 header with the given key/values and no tensors. */
function gguf(entries: ReadonlyArray<readonly [string, number, unknown]>): Buffer {
  const parts: Buffer[] = [Buffer.from('GGUF', 'latin1')]
  const u32 = (value: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(value); return b }
  const u64 = (value: number) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(value)); return b }
  const str = (value: string) => Buffer.concat([u64(Buffer.byteLength(value)), Buffer.from(value, 'utf8')])
  const encode = (type: number, value: unknown): Buffer => {
    if (type === 4) return u32(value as number)
    if (type === 8) return str(value as string)
    if (type === 9) {
      const [elementType, items] = value as [number, unknown[]]
      return Buffer.concat([u32(elementType), u64(items.length), ...items.map(item => encode(elementType, item))])
    }
    throw new Error(`test encoder has no type ${type}`)
  }
  parts.push(u32(3), u64(0), u64(entries.length))
  for (const [key, type, value] of entries) parts.push(str(key), u32(type), encode(type, value))
  return Buffer.concat(parts)
}

test('a GGUF header gives the cache per token, counting only the full-attention layers of a hybrid model', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xerxes-gguf-'))
  try {
    const path = join(dir, 'hybrid.gguf')
    await writeFile(path, gguf([
      ['general.architecture', 8, 'qwen35'],
      // A large string array before the keys that matter, as tokenizers are.
      ['tokenizer.ggml.tokens', 9, [8, Array.from({ length: 5000 }, (_, index) => `tok${index}`)]],
      ['qwen35.block_count', 4, 8],
      ['qwen35.attention.head_count', 4, 4],
      ['qwen35.attention.head_count_kv', 4, 2],
      ['qwen35.attention.key_length', 4, 64],
      ['qwen35.attention.value_length', 4, 64],
      ['qwen35.full_attention_interval', 4, 4],
      ['qwen35.context_length', 4, 1000],
    ]))
    const shape = ggufAttention(path, await readGgufMetadata(path))
    // Layers 4 and 8 keep a cache: 2 layers x 2 heads x (64 + 64) x 2 bytes.
    expect(shape).toEqual({ architecture: 'qwen35', blockCount: 8, cachedLayers: 2, kvBytesPerToken: 1024, contextLength: 1000 })
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('a dense model without key_length uses embedding width over heads, and a bad file is a typed error', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xerxes-gguf-'))
  try {
    const dense = join(dir, 'dense.gguf')
    await writeFile(dense, gguf([
      ['general.architecture', 8, 'llama'],
      ['llama.block_count', 4, 2],
      ['llama.embedding_length', 4, 512],
      ['llama.attention.head_count', 4, 8],
      ['llama.attention.head_count_kv', 4, 2],
    ]))
    // 2 layers x 2 heads x (64 + 64) x 2 bytes.
    expect(ggufAttention(dense, await readGgufMetadata(dense)).kvBytesPerToken).toBe(1024)

    const notGguf = join(dir, 'model.gguf')
    await writeFile(notGguf, 'not a model')
    await expect(readGgufMetadata(notGguf)).rejects.toBeInstanceOf(GgufMetadataError)

    const noShape = join(dir, 'empty.gguf')
    await writeFile(noShape, gguf([['general.architecture', 8, 'llama']]))
    expect(() => ggufAttention(noShape, new Map([['general.architecture', 'llama']]))).toThrow('llama.block_count is missing')
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('the window is half the memory left after weights and reserve, capped at the model window', () => {
  // 16 GiB free, 8 GiB weights: (16 - 8 - 1) / 2 = 3.5 GiB for a 64 KiB/token cache.
  expect(fitWindow({ freeBytes: 16 * GIB, weightsBytes: 8 * GIB, kvBytesPerToken: 64 * 1024, maxWindow: 262_144 }))
    .toEqual({ fits: true, ctxSize: 57_344 })
  expect(fitWindow({ freeBytes: 200 * GIB, weightsBytes: 8 * GIB, kvBytesPerToken: 64 * 1024, maxWindow: 262_144 }))
    .toEqual({ fits: true, ctxSize: 262_144 })
  expect(fitWindow({ freeBytes: 8 * GIB + FIT_RESERVE_BYTES, weightsBytes: 8 * GIB, kvBytesPerToken: 64 * 1024, maxWindow: 262_144 }))
    .toEqual({ fits: false, ctxSize: MIN_FIT_WINDOW })
})

test('free memory comes from nvidia-smi, vm_stat or /proc/meminfo, and is undefined when unreadable', async () => {
  const base: HardwareProfile = { platform: 'darwin', arch: 'arm64', totalMemBytes: 24 * GIB, appleSilicon: true, gpu: { vendor: 'apple' } }
  const vmStat = 'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:  100.\nPages active:  999.\nPages inactive:  50.\nPages speculative:  25.\nPages purgeable:  25.\n'
  expect(parseVmStat(vmStat)).toBe(200 * 16384)
  expect(await freeMemoryBytes(base, async argv => ({ exitCode: argv[0] === 'vm_stat' ? 0 : 1, stdout: vmStat }))).toBe(200 * 16384)

  const nvidia: HardwareProfile = { ...base, platform: 'linux', appleSilicon: false, gpu: { vendor: 'nvidia', vramBytes: 24 * GIB } }
  expect(await freeMemoryBytes(nvidia, async () => ({ exitCode: 0, stdout: '1000\n20000\n' }))).toBe(20000 * 1024 ** 2)

  const linux: HardwareProfile = { ...nvidia, gpu: { vendor: 'none' } }
  expect(await freeMemoryBytes(linux, async () => ({ exitCode: 0, stdout: 'MemTotal: 999 kB\nMemAvailable:    2048 kB\n' }))).toBe(2048 * 1024)
  expect(await freeMemoryBytes(linux, async () => { throw new Error('no cat') })).toBeUndefined()
})
