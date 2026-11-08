// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * What a GGUF file's header says about the memory one token of context needs.
 *
 * llama-server allocates the whole KV cache for its --ctx-size up front, so
 * the window it can hold is the memory left after the weights divided by the
 * cache per token. That depends on the model: how many layers use full
 * attention (hybrid models such as Qwen 3.5 keep a cache in only every Nth
 * layer), how many key/value heads they have, and the head width. All of it is
 * in the GGUF header, which this module reads without loading the weights.
 */
import { open } from 'node:fs/promises'

/** llama-server's default cache type is f16: 2 bytes per element. */
const CACHE_BYTES_PER_ELEMENT = 2
const CHUNK_BYTES = 1 << 20
/** A header larger than this is not a GGUF file this reader trusts. */
const MAX_HEADER_BYTES = 256 << 20

export interface GgufAttention {
  readonly architecture: string
  readonly blockCount: number
  /** Layers that keep a KV cache (all of them unless the model is hybrid). */
  readonly cachedLayers: number
  /** KV cache bytes for one token of context, at f16. */
  readonly kvBytesPerToken: number
  /** The model's own window (`<arch>.context_length`), when the header has it. */
  readonly contextLength?: number
}

type GgufValue = number | bigint | boolean | string | readonly GgufValue[]

/** Thrown when the file is not a GGUF file or its header lacks the attention shape. */
export class GgufMetadataError extends Error {
  constructor(path: string, reason: string) {
    super(`Could not read the attention shape from ${path}: ${reason}`)
    this.name = 'GgufMetadataError'
  }
}

/** Read the header key/values of a GGUF file (not the tensors). */
export async function readGgufMetadata(path: string): Promise<ReadonlyMap<string, GgufValue>> {
  const handle = await open(path, 'r')
  try {
    const reader = new HeaderReader(async (position, length) => {
      const buffer = Buffer.alloc(length)
      const { bytesRead } = await handle.read(buffer, 0, length, position)
      return buffer.subarray(0, bytesRead)
    }, path)
    const magic = (await reader.bytes(4)).toString('latin1')
    if (magic !== 'GGUF') throw new GgufMetadataError(path, 'it does not start with the GGUF magic')
    const version = await reader.u32()
    if (version < 2 || version > 3) throw new GgufMetadataError(path, `GGUF version ${version} is not supported`)
    await reader.u64() // tensor count
    const count = Number(await reader.u64())
    const values = new Map<string, GgufValue>()
    for (let index = 0; index < count; index += 1) {
      const key = await reader.string()
      values.set(key, await reader.value(await reader.u32()))
    }
    return values
  } finally {
    await handle.close()
  }
}

/** The attention shape and cache size per token, from a GGUF header. */
export function ggufAttention(path: string, metadata: ReadonlyMap<string, GgufValue>): GgufAttention {
  const architecture = metadata.get('general.architecture')
  if (typeof architecture !== 'string' || architecture === '') throw new GgufMetadataError(path, 'general.architecture is missing')
  const field = (name: string): GgufValue | undefined => metadata.get(`${architecture}.${name}`)
  const blockCount = positive(field('block_count'))
  if (blockCount === undefined) throw new GgufMetadataError(path, `${architecture}.block_count is missing`)
  const headCount = perLayer(field('attention.head_count'), blockCount)
  const kvHeads = perLayer(field('attention.head_count_kv'), blockCount) ?? headCount
  if (kvHeads === undefined) throw new GgufMetadataError(path, `${architecture}.attention.head_count_kv is missing`)
  const embedding = positive(field('embedding_length'))
  const fallbackWidth = embedding !== undefined && headCount !== undefined && headCount[0]! > 0 ? embedding / headCount[0]! : undefined
  const keyWidth = positive(field('attention.key_length')) ?? fallbackWidth
  const valueWidth = positive(field('attention.value_length')) ?? keyWidth
  if (keyWidth === undefined || valueWidth === undefined) {
    throw new GgufMetadataError(path, `${architecture}.attention.key_length is missing`)
  }
  // Hybrid models keep a cache only in every Nth layer (the full-attention ones).
  const interval = positive(field('full_attention_interval'))
  let elements = 0
  let cachedLayers = 0
  for (let layer = 0; layer < blockCount; layer += 1) {
    if (interval !== undefined && (layer + 1) % interval !== 0) continue
    const heads = kvHeads[layer] ?? 0
    if (heads === 0) continue
    cachedLayers += 1
    elements += heads * (keyWidth + valueWidth)
  }
  const contextLength = positive(field('context_length'))
  return {
    architecture,
    blockCount,
    cachedLayers,
    kvBytesPerToken: elements * CACHE_BYTES_PER_ELEMENT,
    ...(contextLength === undefined ? {} : { contextLength }),
  }
}

function positive(value: GgufValue | undefined): number | undefined {
  const number = typeof value === 'bigint' ? Number(value) : value
  return typeof number === 'number' && Number.isFinite(number) && number > 0 ? number : undefined
}

/** A per-layer count: one number for every layer, or an array with one per layer. */
function perLayer(value: GgufValue | undefined, layers: number): number[] | undefined {
  if (Array.isArray(value)) {
    const counts = value.map(entry => (typeof entry === 'bigint' ? Number(entry) : typeof entry === 'number' ? entry : 0))
    return counts.length === layers ? counts : undefined
  }
  const single = positive(value as GgufValue | undefined)
  return single === undefined ? undefined : Array.from({ length: layers }, () => single)
}

type ReadAt = (position: number, length: number) => Promise<Buffer>

/** Sequential little-endian reader over a file, refilled one chunk at a time. */
class HeaderReader {
  #buffer: Buffer = Buffer.alloc(0)
  #start = 0
  #offset = 0

  constructor(private readonly readAt: ReadAt, private readonly path: string) {}

  async bytes(length: number): Promise<Buffer> {
    await this.#ensure(length)
    const at = this.#offset - this.#start
    this.#offset += length
    return this.#buffer.subarray(at, at + length)
  }

  async u32(): Promise<number> {
    return (await this.bytes(4)).readUInt32LE(0)
  }

  async u64(): Promise<bigint> {
    return (await this.bytes(8)).readBigUInt64LE(0)
  }

  async string(): Promise<string> {
    const length = Number(await this.u64())
    if (length > MAX_HEADER_BYTES) throw new GgufMetadataError(this.path, 'a header string is too long')
    return (await this.bytes(length)).toString('utf8')
  }

  async value(type: number): Promise<GgufValue> {
    switch (type) {
      case 0: return (await this.bytes(1)).readUInt8(0)
      case 1: return (await this.bytes(1)).readInt8(0)
      case 2: return (await this.bytes(2)).readUInt16LE(0)
      case 3: return (await this.bytes(2)).readInt16LE(0)
      case 4: return this.u32()
      case 5: return (await this.bytes(4)).readInt32LE(0)
      case 6: return (await this.bytes(4)).readFloatLE(0)
      case 7: return (await this.bytes(1)).readUInt8(0) !== 0
      case 8: return this.string()
      case 9: {
        const elementType = await this.u32()
        const count = Number(await this.u64())
        if (count > MAX_HEADER_BYTES) throw new GgufMetadataError(this.path, 'a header array is too long')
        const values: GgufValue[] = []
        for (let index = 0; index < count; index += 1) values.push(await this.value(elementType))
        return values
      }
      case 10: return this.u64()
      case 11: return (await this.bytes(8)).readBigInt64LE(0)
      case 12: return (await this.bytes(8)).readDoubleLE(0)
      default: throw new GgufMetadataError(this.path, `unknown header value type ${type}`)
    }
  }

  async #ensure(length: number): Promise<void> {
    const available = this.#start + this.#buffer.length - this.#offset
    if (available >= length) return
    if (this.#offset + length > MAX_HEADER_BYTES) throw new GgufMetadataError(this.path, 'the header is too large')
    const want = Math.max(length, CHUNK_BYTES)
    const chunk = await this.readAt(this.#offset, want)
    if (chunk.length < length) throw new GgufMetadataError(this.path, 'the file ends inside its header')
    this.#buffer = chunk
    this.#start = this.#offset
  }
}
