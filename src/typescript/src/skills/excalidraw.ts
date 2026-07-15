// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { deflateSync } from 'node:zlib'

import { requireSkillText, skillFetchJson, skillJsonObject, type SkillFetch } from './http.js'

/** Excalidraw's public encrypted-collaboration upload endpoint. */
export const EXCALIDRAW_UPLOAD_URL = 'https://json.excalidraw.com/api/v2/post/'

/** Compression seam for callers that use a platform-specific native deflater. */
export type ExcalidrawCompressor = (payload: Uint8Array) => Promise<Uint8Array>

/** AES-GCM seam. The default implementation uses Bun's native Web Crypto implementation. */
export interface ExcalidrawCrypto {
  encrypt(key: Uint8Array, iv: Uint8Array, plaintext: Uint8Array): Promise<Uint8Array>
}

export interface ExcalidrawUploadOptions {
  readonly compressor?: ExcalidrawCompressor
  readonly crypto?: ExcalidrawCrypto
  readonly fetchImplementation?: SkillFetch
  readonly randomBytes?: (length: number) => Uint8Array
  readonly signal?: AbortSignal
  readonly uploadUrl?: string
}

export interface ExcalidrawEncryptedPayload {
  readonly encryptionKey: Uint8Array
  readonly iv: Uint8Array
  readonly payload: Uint8Array
}

/** Return the parsed JSON document or raise before an upload is attempted. */
export function parseExcalidrawDocument(excalidrawJson: string): unknown {
  let parsed: unknown
  try {
    parsed = JSON.parse(requireSkillText(excalidrawJson, 'excalidrawJson')) as unknown
  } catch (error) {
    throw new TypeError(`Excalidraw document is not valid JSON: ${error instanceof Error ? error.message : String(error)}`)
  }
  return parsed
}

/** Join buffers in Excalidraw's versioned, four-byte-big-endian upload framing. */
export function concatExcalidrawBuffers(...buffers: readonly Uint8Array[]): Uint8Array {
  let size = 4
  for (const buffer of buffers) {
    if (buffer.byteLength > 0xffff_ffff) throw new RangeError('Excalidraw buffer is too large')
    size += 4 + buffer.byteLength
  }
  if (!Number.isSafeInteger(size)) throw new RangeError('Excalidraw payload is too large')
  const output = new Uint8Array(size)
  const view = new DataView(output.buffer)
  view.setUint32(0, 1, false)
  let offset = 4
  for (const buffer of buffers) {
    view.setUint32(offset, buffer.byteLength, false)
    offset += 4
    output.set(buffer, offset)
    offset += buffer.byteLength
  }
  return output
}

/** Compress a payload with the zlib wrapper expected by Excalidraw's `pako@1` decoder. */
export async function compressExcalidrawPayload(payload: Uint8Array): Promise<Uint8Array> {
  return new Uint8Array(deflateSync(payload))
}

/** Native Bun Web Crypto adapter for Excalidraw's AES-GCM encryption scheme. */
export const nativeExcalidrawCrypto: ExcalidrawCrypto = {
  async encrypt(key: Uint8Array, iv: Uint8Array, plaintext: Uint8Array): Promise<Uint8Array> {
    const cryptoKey = await globalThis.crypto.subtle.importKey(
      'raw',
      new Uint8Array(key),
      { name: 'AES-GCM' },
      false,
      ['encrypt'],
    )
    const encrypted = await globalThis.crypto.subtle.encrypt(
      { name: 'AES-GCM', iv: new Uint8Array(iv) },
      cryptoKey,
      new Uint8Array(plaintext),
    )
    return new Uint8Array(encrypted)
  },
}

/** Construct, compress, and encrypt an Excalidraw collaboration payload without making a network request. */
export async function createExcalidrawEncryptedPayload(
  excalidrawJson: string,
  options: Pick<ExcalidrawUploadOptions, 'compressor' | 'crypto' | 'randomBytes'> = {},
): Promise<ExcalidrawEncryptedPayload> {
  parseExcalidrawDocument(excalidrawJson)
  const encode = new TextEncoder()
  const metadata = encode.encode('{}')
  const document = encode.encode(excalidrawJson)
  const innerPayload = concatExcalidrawBuffers(metadata, document)
  const compressed = await (options.compressor ?? compressExcalidrawPayload)(innerPayload)
  const encryptionKey = checkedRandomBytes(options.randomBytes ?? secureRandomBytes, 16, 'encryption key')
  const iv = checkedRandomBytes(options.randomBytes ?? secureRandomBytes, 12, 'initialization vector')
  const encrypted = await (options.crypto ?? nativeExcalidrawCrypto).encrypt(encryptionKey, iv, compressed)
  const encodingMetadata = encode.encode(JSON.stringify({
    version: 2,
    compression: 'pako@1',
    encryption: 'AES-GCM',
  }))
  return {
    encryptionKey,
    iv,
    payload: concatExcalidrawBuffers(encodingMetadata, iv, encrypted),
  }
}

/** Upload an encrypted `.excalidraw` JSON document and return its normal Excalidraw share URL. */
export async function uploadExcalidrawDocument(
  excalidrawJson: string,
  options: ExcalidrawUploadOptions = {},
): Promise<string> {
  const encrypted = await createExcalidrawEncryptedPayload(excalidrawJson, options)
  const fetchImplementation = options.fetchImplementation ?? fetch
  const uploadUrl = requireSkillText(options.uploadUrl ?? EXCALIDRAW_UPLOAD_URL, 'uploadUrl')
  const response = await skillFetchJson(fetchImplementation, uploadUrl, {
    method: 'POST',
    body: new Uint8Array(encrypted.payload).buffer,
    headers: { Accept: 'application/json', 'Content-Type': 'application/octet-stream' },
    ...(options.signal === undefined ? {} : { signal: options.signal }),
  })
  const result = skillJsonObject(response, 'Excalidraw upload response')
  const id = result.id
  if (typeof id !== 'string' || !id.trim()) {
    throw new Error(`Excalidraw upload returned no file ID. Response: ${JSON.stringify(result)}`)
  }
  return `https://excalidraw.com/#json=${id},${base64Url(encrypted.encryptionKey)}`
}

function secureRandomBytes(length: number): Uint8Array {
  const bytes = new Uint8Array(length)
  globalThis.crypto.getRandomValues(bytes)
  return bytes
}

function checkedRandomBytes(
  randomBytes: (length: number) => Uint8Array,
  length: number,
  purpose: string,
): Uint8Array {
  const bytes = randomBytes(length)
  if (!(bytes instanceof Uint8Array) || bytes.byteLength !== length) {
    throw new TypeError(`${purpose} generator must return exactly ${length} bytes`)
  }
  return new Uint8Array(bytes)
}

function base64Url(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString('base64').replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '')
}
