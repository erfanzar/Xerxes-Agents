// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { createHash } from 'node:crypto'
import { mkdtemp, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { DownloadError, ensureModelFile, useExistingModelFile, type FetchPort } from '../src/local/download.js'

let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'xerxes-local-dl-'))
})
afterEach(async () => {
  await rm(dir, { recursive: true, force: true })
})

const sha = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex')
const payload = new TextEncoder().encode('GGUF-fake-model-bytes'.repeat(100))

function chunked(bytes: Uint8Array, size = 256): ReadableStream<Uint8Array> {
  let offset = 0
  return new ReadableStream({
    pull(controller) {
      if (offset >= bytes.byteLength) return controller.close()
      controller.enqueue(bytes.slice(offset, offset + size))
      offset += size
    },
  })
}

function fetchOf(bytes: Uint8Array, headers: Record<string, string> = {}): { port: FetchPort; urls: string[] } {
  const urls: string[] = []
  return {
    urls,
    port: async (url) => {
      urls.push(url)
      return new Response(chunked(bytes), { status: 200, headers })
    },
  }
}

const files = async () => (await readdir(dir)).sort()

describe('ensureModelFile', () => {
  test('streams, verifies and publishes atomically, reporting progress', async () => {
    const { port } = fetchOf(payload, { 'content-length': String(payload.byteLength) })
    const seen: number[] = []
    const file = await ensureModelFile({
      destDir: dir,
      fileName: 'm.gguf',
      source: { url: 'https://models.test/m.gguf', sha256: sha(payload), bytes: payload.byteLength },
      fetch: port,
      onProgress: (p) => seen.push(p.receivedBytes),
    })
    expect(file).toMatchObject({ path: join(dir, 'm.gguf'), bytes: payload.byteLength, reused: false })
    expect((await stat(join(dir, 'm.gguf'))).size).toBe(payload.byteLength)
    expect(seen.length).toBeGreaterThan(1)
    expect(seen.at(-1)).toBe(payload.byteLength)
    // no temp files left behind
    expect((await files()).filter((name) => name.includes('.part-'))).toEqual([])
  })

  test('an already verified file is reused without any request', async () => {
    const source = { url: 'https://models.test/m.gguf', sha256: sha(payload), bytes: payload.byteLength }
    const first = fetchOf(payload)
    await ensureModelFile({ destDir: dir, fileName: 'm.gguf', source, fetch: first.port })
    const second = fetchOf(new Uint8Array([1]))
    const again = await ensureModelFile({ destDir: dir, fileName: 'm.gguf', source, fetch: second.port })
    expect(again.reused).toBe(true)
    expect(second.urls).toEqual([])
  })

  test('hash mismatch publishes nothing and leaves no temp file', async () => {
    const { port } = fetchOf(payload)
    const error = await ensureModelFile({
      destDir: dir,
      fileName: 'm.gguf',
      source: { url: 'https://models.test/m.gguf', sha256: 'f'.repeat(64) },
      fetch: port,
    }).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(DownloadError)
    expect((error as DownloadError).code).toBe('HASH_MISMATCH')
    expect(await files()).toEqual([])
  })

  test('a body larger than the declared size is refused mid-stream', async () => {
    const { port } = fetchOf(payload)
    const error = await ensureModelFile({
      destDir: dir,
      fileName: 'm.gguf',
      source: { url: 'https://models.test/m.gguf', sha256: sha(payload), bytes: 100 },
      fetch: port,
    }).catch((e: unknown) => e)
    expect((error as DownloadError).code).toBe('TOO_LARGE')
    expect(await files()).toEqual([])
  })

  test('an oversize content-length is refused before reading the body', async () => {
    const { port } = fetchOf(payload, { 'content-length': '999999999' })
    const error = await ensureModelFile({
      destDir: dir,
      fileName: 'm.gguf',
      source: { url: 'https://models.test/m.gguf', sha256: sha(payload), bytes: payload.byteLength },
      fetch: port,
    }).catch((e: unknown) => e)
    expect((error as DownloadError).code).toBe('TOO_LARGE')
    expect(await files()).toEqual([])
  })

  test('a short body is a size mismatch', async () => {
    const { port } = fetchOf(payload)
    const error = await ensureModelFile({
      destDir: dir,
      fileName: 'm.gguf',
      source: { url: 'https://models.test/m.gguf', sha256: sha(payload), bytes: payload.byteLength + 10 },
      fetch: port,
    }).catch((e: unknown) => e)
    expect((error as DownloadError).code).toBe('SIZE_MISMATCH')
    expect(await files()).toEqual([])
  })

  test('non-https sources are refused without any request', async () => {
    const { port, urls } = fetchOf(payload)
    const error = await ensureModelFile({
      destDir: dir,
      fileName: 'm.gguf',
      source: { url: 'http://models.test/m.gguf', sha256: sha(payload) },
      fetch: port,
    }).catch((e: unknown) => e)
    expect((error as DownloadError).code).toBe('INSECURE_URL')
    expect(urls).toEqual([])
  })

  test('a redirect to http is refused', async () => {
    const port: FetchPort = async () =>
      new Response(null, { status: 302, headers: { location: 'http://evil.test/m.gguf' } })
    const error = await ensureModelFile({
      destDir: dir,
      fileName: 'm.gguf',
      source: { url: 'https://models.test/m.gguf', sha256: sha(payload) },
      fetch: port,
    }).catch((e: unknown) => e)
    expect((error as DownloadError).code).toBe('INSECURE_URL')
  })

  test('an https redirect is followed and non-200 is an actionable error', async () => {
    const calls: string[] = []
    const port: FetchPort = async (url) => {
      calls.push(url)
      if (url === 'https://models.test/m.gguf') {
        return new Response(null, { status: 302, headers: { location: 'https://cdn.test/blob' } })
      }
      return new Response(chunked(payload), { status: 200 })
    }
    const file = await ensureModelFile({
      destDir: dir,
      fileName: 'm.gguf',
      source: { url: 'https://models.test/m.gguf', sha256: sha(payload) },
      fetch: port,
    })
    expect(calls).toEqual(['https://models.test/m.gguf', 'https://cdn.test/blob'])
    expect(file.bytes).toBe(payload.byteLength)

    const notFound: FetchPort = async () => new Response('no', { status: 404 })
    const error = await ensureModelFile({
      destDir: dir,
      fileName: 'other.gguf',
      source: { url: 'https://models.test/x.gguf', sha256: sha(payload) },
      fetch: notFound,
    }).catch((e: unknown) => e)
    expect((error as DownloadError).code).toBe('HTTP_STATUS')
    expect((error as DownloadError).message).toContain('local.models')
  })

  test('a network failure is typed', async () => {
    const port: FetchPort = async () => {
      throw new Error('ECONNRESET')
    }
    const error = await ensureModelFile({
      destDir: dir,
      fileName: 'm.gguf',
      source: { url: 'https://models.test/m.gguf', sha256: sha(payload) },
      fetch: port,
    }).catch((e: unknown) => e)
    expect((error as DownloadError).code).toBe('NETWORK')
  })
})

describe('useExistingModelFile', () => {
  test('accepts a real GGUF file', async () => {
    const path = join(dir, 'mine.gguf')
    await writeFile(path, Buffer.concat([Buffer.from('GGUF'), Buffer.alloc(16)]))
    expect(await useExistingModelFile(path)).toEqual({ path, bytes: 20 })
  })

  test('rejects missing files and non-GGUF content', async () => {
    const missing = await useExistingModelFile(join(dir, 'nope.gguf')).catch((e: unknown) => e)
    expect((missing as DownloadError).code).toBe('FILE_NOT_FOUND')
    const fake = join(dir, 'fake.gguf')
    await writeFile(fake, 'not a model')
    const wrong = await useExistingModelFile(fake).catch((e: unknown) => e)
    expect((wrong as DownloadError).code).toBe('NOT_GGUF')
  })
})
