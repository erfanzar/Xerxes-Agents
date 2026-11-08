// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Verified model download. The body is streamed to a private temp file while it
 * is hashed (never buffered whole), checked against the expected size and
 * sha256, and only then renamed into place. Any failure removes the temp file,
 * so a bad download can never be published.
 */
import { createHash, randomBytes } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { mkdir, open, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { basename, dirname, join, resolve } from 'node:path'

import { XerxesError } from '../core/errors.js'
import type { ModelSource } from './catalog.js'

export type DownloadErrorCode =
  | 'BAD_SOURCE'
  | 'INSECURE_URL'
  | 'HTTP_STATUS'
  | 'NETWORK'
  | 'TOO_LARGE'
  | 'SIZE_MISMATCH'
  | 'HASH_MISMATCH'
  | 'ABORTED'
  | 'FILE_NOT_FOUND'
  | 'NOT_GGUF'

export class DownloadError extends XerxesError {
  readonly code: DownloadErrorCode

  constructor(code: DownloadErrorCode, message: string, details: Record<string, unknown> = {}, cause?: unknown) {
    super(message, { code, ...details }, cause === undefined ? {} : { cause })
    this.code = code
  }
}

export type FetchPort = (
  url: string,
  init: { readonly signal?: AbortSignal; readonly redirect: 'manual' },
) => Promise<Response>

export interface DownloadProgress {
  readonly receivedBytes: number
  readonly totalBytes?: number
}

export interface EnsureModelOptions {
  readonly destDir: string
  readonly fileName: string
  readonly source: ModelSource
  readonly fetch: FetchPort
  readonly onProgress?: (progress: DownloadProgress) => void
  readonly signal?: AbortSignal
  readonly maxRedirects?: number
}

export interface ModelFile {
  readonly path: string
  readonly bytes: number
  readonly sha256: string
  /** True when an already-verified file was reused and nothing was downloaded. */
  readonly reused: boolean
}

const SHA256_PATTERN = /^[0-9a-f]{64}$/i
const DEFAULT_MAX_REDIRECTS = 5

function assertHttps(url: string): URL {
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    throw new DownloadError('BAD_SOURCE', `Model URL is not a valid URL: ${url}`)
  }
  if (parsed.protocol !== 'https:') {
    throw new DownloadError(
      'INSECURE_URL',
      `Refusing to download a model over ${parsed.protocol.replace(':', '')}; the URL must use https (got ${parsed.origin}).`,
      { url: parsed.origin },
    )
  }
  return parsed
}

function sidecarPath(file: string): string {
  return `${file}.verified`
}

interface Sidecar {
  readonly sha256: string
  readonly size: number
  readonly mtimeMs: number
}

async function readSidecar(file: string): Promise<Sidecar | undefined> {
  try {
    const parsed: unknown = JSON.parse(await readFile(sidecarPath(file), 'utf8'))
    if (
      typeof parsed === 'object' &&
      parsed !== null &&
      'sha256' in parsed &&
      'size' in parsed &&
      'mtimeMs' in parsed &&
      typeof parsed.sha256 === 'string' &&
      typeof parsed.size === 'number' &&
      typeof parsed.mtimeMs === 'number'
    ) {
      return { sha256: parsed.sha256, size: parsed.size, mtimeMs: parsed.mtimeMs }
    }
  } catch {
    // no sidecar or unreadable: the file is simply re-hashed
  }
  return undefined
}

async function writeSidecar(file: string, sha256: string): Promise<void> {
  const info = await stat(file)
  const record: Sidecar = { sha256, size: info.size, mtimeMs: info.mtimeMs }
  await writeFile(sidecarPath(file), JSON.stringify(record) + '\n', { mode: 0o600 })
}

export async function hashFile(path: string): Promise<string> {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  return hash.digest('hex')
}

/**
 * Whether `path` exists and matches `source`. The file is hashed once; a sidecar
 * keyed by size and mtime lets later calls skip the hash until the file changes.
 */
export async function verifyModelFile(
  path: string,
  source: ModelSource,
): Promise<{ present: boolean; verified: boolean; bytes?: number }> {
  let info
  try {
    info = await stat(path)
  } catch {
    return { present: false, verified: false }
  }
  if (!info.isFile()) return { present: false, verified: false }
  const expected = source.sha256.toLowerCase()
  if (source.bytes !== undefined && info.size !== source.bytes) {
    return { present: true, verified: false, bytes: info.size }
  }
  const sidecar = await readSidecar(path)
  if (sidecar !== undefined && sidecar.sha256 === expected && sidecar.size === info.size && sidecar.mtimeMs === info.mtimeMs) {
    return { present: true, verified: true, bytes: info.size }
  }
  const actual = await hashFile(path)
  if (actual !== expected) return { present: true, verified: false, bytes: info.size }
  await writeSidecar(path, actual)
  return { present: true, verified: true, bytes: info.size }
}

async function openResponse(options: EnsureModelOptions): Promise<Response> {
  const max = options.maxRedirects ?? DEFAULT_MAX_REDIRECTS
  let url = assertHttps(options.source.url).toString()
  for (let hop = 0; hop <= max; hop += 1) {
    let response: Response
    try {
      response = await options.fetch(url, {
        redirect: 'manual',
        ...(options.signal === undefined ? {} : { signal: options.signal }),
      })
    } catch (error) {
      if (options.signal?.aborted) throw new DownloadError('ABORTED', 'Model download was cancelled.', {}, error)
      throw new DownloadError(
        'NETWORK',
        `Could not reach ${new URL(url).host}: ${error instanceof Error ? error.message : String(error)}. Check your connection and retry.`,
        { host: new URL(url).host },
        error,
      )
    }
    if (response.status >= 300 && response.status < 400) {
      const location = response.headers.get('location')
      await response.body?.cancel()
      if (location === null) {
        throw new DownloadError('HTTP_STATUS', `Redirect (${response.status}) from ${new URL(url).host} had no Location header.`)
      }
      url = assertHttps(new URL(location, url).toString()).toString()
      continue
    }
    if (response.status !== 200) {
      await response.body?.cancel()
      throw new DownloadError(
        'HTTP_STATUS',
        `Model server ${new URL(url).host} answered HTTP ${response.status}. Check local.models.<id>.url in your config.`,
        { status: response.status },
      )
    }
    return response
  }
  throw new DownloadError('HTTP_STATUS', `Too many redirects (more than ${max}) while downloading the model.`)
}

/**
 * Make sure a verified copy of the model exists at `destDir/fileName`.
 * Reuses a verified file; otherwise downloads, verifies and atomically publishes.
 */
export async function ensureModelFile(options: EnsureModelOptions): Promise<ModelFile> {
  const { source, destDir, fileName } = options
  assertHttps(source.url)
  if (!SHA256_PATTERN.test(source.sha256)) {
    throw new DownloadError('BAD_SOURCE', 'Model sha256 must be 64 hexadecimal characters.')
  }
  if (fileName === '' || basename(fileName) !== fileName) {
    throw new DownloadError('BAD_SOURCE', `Invalid model file name: ${fileName}`)
  }
  const expected = source.sha256.toLowerCase()
  const dest = join(destDir, fileName)
  await mkdir(destDir, { recursive: true, mode: 0o700 })

  const existing = await verifyModelFile(dest, source)
  if (existing.verified && existing.bytes !== undefined) {
    return { path: dest, bytes: existing.bytes, sha256: expected, reused: true }
  }

  const response = await openResponse(options)
  const declared = Number(response.headers.get('content-length') ?? '')
  const totalBytes = Number.isFinite(declared) && declared > 0 ? declared : source.bytes
  if (source.bytes !== undefined && Number.isFinite(declared) && declared > source.bytes) {
    await response.body?.cancel()
    throw new DownloadError(
      'TOO_LARGE',
      `Model server announced ${declared} bytes but the expected size is ${source.bytes}; refusing to download.`,
      { declared, expected: source.bytes },
    )
  }
  if (response.body === null) throw new DownloadError('NETWORK', 'Model server returned an empty response body.')

  const tmp = `${dest}.part-${process.pid}-${randomBytes(4).toString('hex')}`
  const handle = await open(tmp, 'wx', 0o600)
  const hash = createHash('sha256')
  let received = 0
  const reader = response.body.getReader()
  try {
    for (;;) {
      if (options.signal?.aborted) {
        await reader.cancel()
        throw new DownloadError('ABORTED', 'Model download was cancelled.')
      }
      let step: Awaited<ReturnType<typeof reader.read>>
      try {
        step = await reader.read()
      } catch (error) {
        throw new DownloadError(
          'NETWORK',
          `Download interrupted after ${received} bytes: ${error instanceof Error ? error.message : String(error)}. Retry to start again.`,
          { received },
          error,
        )
      }
      if (step.done) break
      received += step.value.byteLength
      if (source.bytes !== undefined && received > source.bytes) {
        await reader.cancel()
        throw new DownloadError(
          'TOO_LARGE',
          `Download exceeded the expected ${source.bytes} bytes; refusing to continue and discarding the partial file.`,
          { expected: source.bytes },
        )
      }
      hash.update(step.value)
      await handle.write(step.value)
      options.onProgress?.({ receivedBytes: received, ...(totalBytes === undefined ? {} : { totalBytes }) })
    }
    await handle.sync()
    await handle.close()

    if (source.bytes !== undefined && received !== source.bytes) {
      throw new DownloadError(
        'SIZE_MISMATCH',
        `Downloaded ${received} bytes but expected ${source.bytes}; the file was discarded.`,
        { received, expected: source.bytes },
      )
    }
    const actual = hash.digest('hex')
    if (actual !== expected) {
      throw new DownloadError(
        'HASH_MISMATCH',
        `sha256 mismatch: expected ${expected} but downloaded ${actual}. The file was discarded; check local.models.<id>.sha256 and url.`,
        { expected, actual },
      )
    }
    await rename(tmp, dest)
    await writeSidecar(dest, actual)
    return { path: dest, bytes: received, sha256: actual, reused: false }
  } catch (error) {
    await handle.close().catch(() => undefined)
    await rm(tmp, { force: true })
    throw error
  }
}

/** Accept a user-supplied model: must be an existing regular file that starts with the GGUF magic. */
export async function useExistingModelFile(path: string): Promise<{ path: string; bytes: number }> {
  const resolved = resolve(path)
  let info
  try {
    info = await stat(resolved)
  } catch {
    throw new DownloadError('FILE_NOT_FOUND', `Model file not found: ${resolved}`, { path: resolved })
  }
  if (!info.isFile()) throw new DownloadError('FILE_NOT_FOUND', `Not a regular file: ${resolved}`, { path: resolved })
  const handle = await open(resolved, 'r')
  try {
    const magic = Buffer.alloc(4)
    await handle.read(magic, 0, 4, 0)
    if (!resolved.toLowerCase().endsWith('.gguf') || magic.toString('latin1') !== 'GGUF') {
      throw new DownloadError(
        'NOT_GGUF',
        `${resolved} is not a GGUF model (needs a .gguf extension and GGUF header). llama-server only loads GGUF files.`,
        { path: resolved },
      )
    }
  } finally {
    await handle.close()
  }
  return { path: resolved, bytes: info.size }
}

export function modelPathFor(modelsDir: string, fileName: string): string {
  return join(modelsDir, fileName)
}

export function parentDirectory(path: string): string {
  return dirname(path)
}
