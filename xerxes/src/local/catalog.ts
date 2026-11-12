// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Local model catalog, model selection by hardware, and the `local` settings
 * parser. A built-in entry's download source is the one its model card pins:
 * a Hugging Face file at one commit with its SHA-256, so nothing here points at
 * an unverified artifact. `local.models.<id>` in the user's config replaces it.
 */
import { homedir } from 'node:os'
import { isAbsolute, join, resolve } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { xerxesHome } from '../daemon/paths.js'
import { huggingFaceUrl, localModelCards, type LmStudioBuild, type MlxBuild } from '../llms/modelCards.js'
import type { HardwareProfile } from './hardware.js'

export const GIB = 1024 ** 3
/** Memory kept back for the OS on Apple silicon (unified memory is shared). */
export const OS_RESERVE_BYTES = 4 * GIB
export const DEFAULT_LOCAL_PORT = 11581

export interface ModelSource {
  readonly url: string
  readonly sha256: string
  readonly bytes?: number
}

export interface CatalogEntry {
  readonly id: string
  readonly label: string
  /** File name under the models directory. */
  readonly fileName: string
  readonly minMemoryBytes: number
  /**
   * Default --ctx-size for llama-server: what fits in this entry's memory tier
   * next to the weights, not the model's limit. Override with local.ctx_size.
   */
  readonly ctxSize: number
  /** The model's own context window; local.ctx_size may not exceed it. */
  readonly nativeContext?: number
  readonly source?: ModelSource
  /** The oldest stock llama.cpp that runs the GGUF build. */
  readonly minLlamaCpp?: { readonly build: number; readonly version: string }
  /** The MLX build mlx_lm.server runs, when there is one. */
  readonly mlx?: MlxBuild
  /** How LM Studio knows this model, when it has a build LM Studio can run. */
  readonly lmstudio?: LmStudioBuild
}

/** In the order `xerxes local` tries them when no engine is chosen. */
export type LocalEngineName = 'llama-server' | 'mlx' | 'lmstudio'
export const LOCAL_ENGINES: readonly LocalEngineName[] = ['llama-server', 'mlx', 'lmstudio']

export type Catalog = readonly CatalogEntry[]

/**
 * The cards with a llama-server build, smallest first, in the shape local
 * setup uses. Every value comes from the model card (llms/modelCards.ts).
 */
export const BUILTIN_MODELS: Catalog = localModelCards()
  .flatMap((card): CatalogEntry[] => card.llamaServer === undefined ? [] : [{
    id: card.id,
    label: card.label,
    fileName: card.llamaServer.fileName,
    minMemoryBytes: card.llamaServer.minMemoryBytes,
    ctxSize: card.llamaServer.ctxSize,
    ...(card.nativeContext === undefined ? {} : { nativeContext: card.nativeContext }),
    ...(card.llamaServer.published === undefined ? {} : {
      source: {
        url: huggingFaceUrl(card.llamaServer.published, card.llamaServer.fileName),
        sha256: card.llamaServer.published.sha256,
        bytes: card.llamaServer.published.bytes,
      },
    }),
    ...(card.llamaServer.minLlamaCpp === undefined ? {} : { minLlamaCpp: card.llamaServer.minLlamaCpp }),
    ...(card.mlx === undefined ? {} : { mlx: card.mlx }),
    ...(card.lmstudio === undefined ? {} : { lmstudio: card.lmstudio }),
  }])
  .sort((a, b) => a.minMemoryBytes - b.minMemoryBytes)

export interface LocalConfig {
  /** local.engine / XERXES_LOCAL_ENGINE; undefined when unset (setup and start then use the first installed engine of llama-server, mlx and lmstudio). */
  readonly engine?: LocalEngineName
  readonly llamaServerPath?: string
  /** mlx-lm's `mlx_lm.server` (local.mlx_server_path / XERXES_LOCAL_MLX_SERVER). */
  readonly mlxServerPath?: string
  /** LM Studio's `lms` CLI (local.lms_path / XERXES_LOCAL_LMS). */
  readonly lmsPath?: string
  readonly modelsDir: string
  readonly port: number
  /** True when local.port or XERXES_LOCAL_PORT set the port, rather than the default. */
  readonly portExplicit: boolean
  readonly models: Readonly<Record<string, ModelSource>>
  /** --ctx-size override for llama-server (local.ctx_size, XERXES_LOCAL_CTX_SIZE). */
  readonly ctxSize?: number
}

type Env = Readonly<Record<string, string | undefined>>

const LOCAL_KEYS = new Set(['engine', 'llama_server_path', 'mlx_server_path', 'lms_path', 'models_dir', 'port', 'models', 'ctx_size'])
const MIN_CTX_SIZE = 2048
const MAX_CTX_SIZE = 1_048_576
const SHA256_PATTERN = /^[0-9a-f]{64}$/i

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function expandHome(path: string, env: Env): string {
  if (path === '~') return homedir()
  if (path.startsWith('~/') || path.startsWith('~\\')) return resolve(homedir(), path.slice(2))
  void env
  return path
}

function parsePort(value: unknown, field: string): number {
  const port = typeof value === 'string' && /^\d+$/.test(value.trim()) ? Number(value.trim()) : value
  if (typeof port !== 'number' || !Number.isInteger(port) || port < 1 || port > 65535) {
    throw new ValidationError(field, 'must be an integer from 1 to 65535', value)
  }
  return port
}

export function parseModelSource(value: unknown, field: string): ModelSource {
  if (!isRecord(value)) throw new ValidationError(field, 'must be an object { url, sha256 }', value)
  for (const key of Object.keys(value)) {
    if (key !== 'url' && key !== 'sha256' && key !== 'bytes') {
      throw new ValidationError(`${field}.${key}`, 'unknown key (allowed: url, sha256, bytes)', value[key])
    }
  }
  const { url, sha256, bytes } = value
  if (typeof url !== 'string') throw new ValidationError(`${field}.url`, 'must be an https URL string', url)
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    throw new ValidationError(`${field}.url`, 'is not a valid URL', url)
  }
  if (parsed.protocol !== 'https:') throw new ValidationError(`${field}.url`, 'must use https', url)
  if (typeof sha256 !== 'string' || !SHA256_PATTERN.test(sha256)) {
    throw new ValidationError(`${field}.sha256`, 'must be 64 hexadecimal characters', sha256)
  }
  if (bytes !== undefined && (typeof bytes !== 'number' || !Number.isSafeInteger(bytes) || bytes <= 0)) {
    throw new ValidationError(`${field}.bytes`, 'must be a positive integer', bytes)
  }
  return {
    url,
    sha256: sha256.toLowerCase(),
    ...(typeof bytes === 'number' ? { bytes } : {}),
  }
}

/**
 * Parse the `local` key of the daemon `runtime` settings. Env vars win over the
 * file: XERXES_LOCAL_LLAMA_SERVER, XERXES_LOCAL_MODELS_DIR, XERXES_LOCAL_PORT.
 */
export function parseLocalConfig(runtime: Record<string, unknown>, env: Env): LocalConfig {
  const raw = runtime.local
  if (raw !== undefined && !isRecord(raw)) throw new ValidationError('local', 'must be an object', raw)
  const section: Record<string, unknown> = raw ?? {}
  for (const key of Object.keys(section)) {
    if (!LOCAL_KEYS.has(key)) {
      throw new ValidationError(`local.${key}`, `unknown key (allowed: ${[...LOCAL_KEYS].join(', ')})`, section[key])
    }
  }

  const envServer = env.XERXES_LOCAL_LLAMA_SERVER?.trim()
  const fileServer = section.llama_server_path
  if (fileServer !== undefined && (typeof fileServer !== 'string' || fileServer.trim() === '')) {
    throw new ValidationError('local.llama_server_path', 'must be a non-empty string', fileServer)
  }
  const serverPath = envServer ? envServer : typeof fileServer === 'string' ? fileServer.trim() : undefined

  const envMlx = env.XERXES_LOCAL_MLX_SERVER?.trim()
  const fileMlx = section.mlx_server_path
  if (fileMlx !== undefined && (typeof fileMlx !== 'string' || fileMlx.trim() === '')) {
    throw new ValidationError('local.mlx_server_path', 'must be a non-empty string', fileMlx)
  }
  const mlxServerPath = envMlx ? envMlx : typeof fileMlx === 'string' ? expandHome(fileMlx.trim(), env) : undefined

  const envLms = env.XERXES_LOCAL_LMS?.trim()
  const fileLms = section.lms_path
  if (fileLms !== undefined && (typeof fileLms !== 'string' || fileLms.trim() === '')) {
    throw new ValidationError('local.lms_path', 'must be a non-empty string', fileLms)
  }
  const lmsPath = envLms ? envLms : typeof fileLms === 'string' ? expandHome(fileLms.trim(), env) : undefined

  const envEngine = env.XERXES_LOCAL_ENGINE?.trim()
  const engine = envEngine
    ? parseEngine(envEngine, 'XERXES_LOCAL_ENGINE')
    : section.engine === undefined
      ? undefined
      : parseEngine(section.engine, 'local.engine')

  const envDir = env.XERXES_LOCAL_MODELS_DIR?.trim()
  const fileDir = section.models_dir
  if (fileDir !== undefined && (typeof fileDir !== 'string' || fileDir.trim() === '')) {
    throw new ValidationError('local.models_dir', 'must be a non-empty string', fileDir)
  }
  const configuredDir = envDir ? envDir : typeof fileDir === 'string' ? fileDir.trim() : undefined
  let modelsDir: string
  if (configuredDir === undefined) {
    modelsDir = join(xerxesHome(env), 'local', 'models')
  } else {
    modelsDir = expandHome(configuredDir, env)
    if (!isAbsolute(modelsDir)) {
      throw new ValidationError('local.models_dir', 'must be an absolute path (or start with ~/)', configuredDir)
    }
  }

  const envPort = env.XERXES_LOCAL_PORT?.trim()
  const port = envPort
    ? parsePort(envPort, 'XERXES_LOCAL_PORT')
    : section.port === undefined
      ? DEFAULT_LOCAL_PORT
      : parsePort(section.port, 'local.port')

  const models: Record<string, ModelSource> = {}
  const rawModels = section.models
  if (rawModels !== undefined) {
    if (!isRecord(rawModels)) throw new ValidationError('local.models', 'must be an object keyed by model id', rawModels)
    const known = new Set(BUILTIN_MODELS.map((entry) => entry.id))
    for (const [id, value] of Object.entries(rawModels)) {
      if (!known.has(id)) {
        throw new ValidationError(
          `local.models.${id}`,
          `unknown model id (known: ${[...known].join(', ')})`,
          value,
        )
      }
      models[id] = parseModelSource(value, `local.models.${id}`)
    }
  }

  const envCtx = env.XERXES_LOCAL_CTX_SIZE?.trim()
  const ctxSize = envCtx
    ? parseCtxSize(envCtx, 'XERXES_LOCAL_CTX_SIZE')
    : section.ctx_size === undefined
      ? undefined
      : parseCtxSize(section.ctx_size, 'local.ctx_size')

  return {
    ...(engine === undefined ? {} : { engine }),
    ...(serverPath === undefined ? {} : { llamaServerPath: serverPath }),
    ...(mlxServerPath === undefined ? {} : { mlxServerPath }),
    ...(lmsPath === undefined ? {} : { lmsPath }),
    modelsDir,
    port,
    portExplicit: Boolean(envPort) || section.port !== undefined,
    models,
    ...(ctxSize === undefined ? {} : { ctxSize }),
  }
}

/** `llama-server`, `mlx` or `lmstudio`; anything else is refused naming the field. */
export function parseEngine(value: unknown, field: string): LocalEngineName {
  const name = typeof value === 'string' ? value.trim().toLowerCase() : value
  if (name === 'llama-server' || name === 'mlx' || name === 'lmstudio') return name
  throw new ValidationError(field, `must be one of ${LOCAL_ENGINES.join(', ')}`, value)
}

export function parseCtxSize(value: unknown, field: string): number {
  const size = typeof value === 'string' && /^\d+$/.test(value.trim()) ? Number(value.trim()) : value
  if (typeof size !== 'number' || !Number.isInteger(size) || size < MIN_CTX_SIZE || size > MAX_CTX_SIZE) {
    throw new ValidationError(field, `must be an integer from ${MIN_CTX_SIZE} to ${MAX_CTX_SIZE}`, value)
  }
  return size
}

/**
 * The --ctx-size to launch an entry with: the configured override, else the
 * entry's default. An override past the model's own window is refused rather
 * than clamped, so the setting never silently means something else.
 */
export function launchCtxSize(entry: CatalogEntry, config: Pick<LocalConfig, 'ctxSize'>): number {
  const size = config.ctxSize ?? entry.ctxSize
  if (entry.nativeContext !== undefined && size > entry.nativeContext) {
    throw new ValidationError(
      'local.ctx_size',
      `exceeds ${entry.label}'s ${entry.nativeContext}-token window`,
      size,
    )
  }
  return size
}

/** Built-in catalog with the user's `local.models` sources merged in. */
export function catalogWithConfig(config: { readonly models: Readonly<Record<string, ModelSource>> }): Catalog {
  return BUILTIN_MODELS.map((entry) => {
    const source = config.models[entry.id]
    return source === undefined ? entry : { ...entry, source }
  })
}

export type BudgetBasis = 'vram' | 'unified-memory' | 'system-memory'

export type ModelSelection =
  | { readonly ok: true; readonly entry: CatalogEntry; readonly budgetBytes: number; readonly basis: BudgetBasis }
  | { readonly ok: false; readonly reason: string; readonly budgetBytes: number; readonly basis: BudgetBasis }

/** Memory a model may use on this machine, and where that number came from. */
export function memoryBudget(hw: HardwareProfile): { budgetBytes: number; basis: BudgetBasis } {
  if (hw.gpu.vendor === 'nvidia' && hw.gpu.vramBytes !== undefined) {
    return { budgetBytes: hw.gpu.vramBytes, basis: 'vram' }
  }
  if (hw.appleSilicon) {
    return { budgetBytes: Math.max(0, hw.totalMemBytes - OS_RESERVE_BYTES), basis: 'unified-memory' }
  }
  return { budgetBytes: hw.totalMemBytes, basis: 'system-memory' }
}

export function formatGiB(bytes: number): string {
  return `${(bytes / GIB).toFixed(1)} GiB`
}

/** The largest catalog entry whose minimum memory fits; otherwise a typed refusal. */
export function selectModel(hw: HardwareProfile, catalog: Catalog): ModelSelection {
  const { budgetBytes, basis } = memoryBudget(hw)
  const bySize = [...catalog].sort((a, b) => b.minMemoryBytes - a.minMemoryBytes)
  const entry = bySize.find((candidate) => candidate.minMemoryBytes <= budgetBytes)
  if (entry !== undefined) return { ok: true, entry, budgetBytes, basis }
  const smallest = bySize[bySize.length - 1]
  const need = smallest === undefined ? 'a local model' : `${smallest.id} (needs ${formatGiB(smallest.minMemoryBytes)})`
  return {
    ok: false,
    budgetBytes,
    basis,
    reason:
      `This machine has ${formatGiB(budgetBytes)} of usable ${basis.replace('-', ' ')}; the smallest local model is ${need}. ` +
      'Use a cloud provider profile instead (run `xerxes setup` or `/provider` to add one).',
  }
}
