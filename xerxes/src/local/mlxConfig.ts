// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The attention shape of an MLX model folder, from its `config.json`, so the
 * window can be sized from free memory as it is for a GGUF file. mlx-lm grows
 * its cache as the context grows and has no window setting, so the size
 * computed here is the window recorded for the profile: compaction keeps the
 * conversation inside it.
 */

import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

import type { GgufAttention } from './ggufMetadata.js'

/** mlx-lm keeps its KV cache in the model's float type: 2 bytes per element. */
const CACHE_BYTES_PER_ELEMENT = 2

export class MlxConfigError extends Error {
  constructor(dir: string, reason: string) {
    super(`Could not read the attention shape from ${join(dir, 'config.json')}: ${reason}`)
    this.name = 'MlxConfigError'
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function positive(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : undefined
}

/**
 * The shape from a parsed `config.json`. Multimodal packs keep the language
 * model under `text_config`. A hybrid model (`full_attention_interval`, or
 * `layer_types`) keeps a cache only in its full-attention layers.
 */
export function mlxAttention(dir: string, config: unknown): GgufAttention {
  if (!isRecord(config)) throw new MlxConfigError(dir, 'it is not a JSON object')
  const text = isRecord(config.text_config) ? config.text_config : config
  const architecture = typeof config.model_type === 'string' ? config.model_type : 'unknown'
  const blockCount = positive(text.num_hidden_layers)
  if (blockCount === undefined) throw new MlxConfigError(dir, 'num_hidden_layers is missing')
  const heads = positive(text.num_attention_heads)
  const kvHeads = positive(text.num_key_value_heads) ?? heads
  if (kvHeads === undefined) throw new MlxConfigError(dir, 'num_key_value_heads is missing')
  const hidden = positive(text.hidden_size)
  const headDim = positive(text.head_dim) ?? (hidden !== undefined && heads !== undefined ? hidden / heads : undefined)
  if (headDim === undefined) throw new MlxConfigError(dir, 'head_dim is missing')
  const layerTypes = Array.isArray(text.layer_types) && text.layer_types.length === blockCount ? text.layer_types : undefined
  const interval = positive(text.full_attention_interval)
  let cachedLayers = 0
  for (let layer = 0; layer < blockCount; layer += 1) {
    const full = layerTypes !== undefined
      ? layerTypes[layer] === 'full_attention'
      : interval === undefined || (layer + 1) % interval === 0
    if (full) cachedLayers += 1
  }
  const contextLength = positive(text.max_position_embeddings)
  return {
    architecture,
    blockCount,
    cachedLayers,
    kvBytesPerToken: cachedLayers * kvHeads * headDim * 2 * CACHE_BYTES_PER_ELEMENT,
    ...(contextLength === undefined ? {} : { contextLength }),
  }
}

/** Read `config.json` from an MLX model folder and return its attention shape. */
export async function readMlxAttention(dir: string): Promise<GgufAttention> {
  let text: string
  try {
    text = await readFile(join(dir, 'config.json'), 'utf8')
  } catch (error) {
    throw new MlxConfigError(dir, error instanceof Error ? error.message : String(error))
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    throw new MlxConfigError(dir, 'it is not valid JSON')
  }
  return mlxAttention(dir, parsed)
}
