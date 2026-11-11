// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { deflateSync, inflateSync } from 'node:zlib'

const MAX_IMAGE_BYTES = 256 * 1024
const MAX_BASE64_CHARS = 4 * Math.ceil(MAX_IMAGE_BYTES / 3)
const WIDTHS: ReadonlySet<number> = new Set([390, 768, 1280, 1440])
const KEYS: ReadonlySet<string> = new Set(['kind', 'view_id', 'mime', 'width', 'height', 'captured_at', 'sha256', 'png_base64'])
const SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])

/** Validated pixels are data, never proof of a root, capture identity or disclosure consent. */
export interface CaptureImage {
  readonly viewId: string
  readonly mime: 'image/png'
  readonly width: number
  readonly height: number
  readonly capturedAt: number
  readonly sha256: string
  readonly sourceSha256: string
  readonly normalization: 'png-raster-v1'
  readonly byteLength: number
  readonly imageUrl: string
}

function invalid(): never { throw new Error('Invalid or unsupported bounded Xerxes capture image') }
function record(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value) }

function chunk(kind: string, body: Uint8Array): Buffer {
  const result = Buffer.alloc(body.length + 12)
  result.writeUInt32BE(body.length, 0); result.write(kind, 4, 4, 'ascii'); result.set(body, 8)
  result.writeUInt32BE(Bun.hash.crc32(result.subarray(4, -4)), result.length - 4)
  return result
}

/** Closed inline-image envelope. No remote/file URL, SVG, metadata chunks or animated PNG. */
export function validateCaptureImage(value: unknown): CaptureImage {
  if (!record(value) || Object.keys(value).some(key => !KEYS.has(key)) || value.kind !== 'xerxes-ui-capture/v1' || value.mime !== 'image/png'
    || typeof value.view_id !== 'string' || !/^[a-f0-9]{32}$/.test(value.view_id)
    || typeof value.width !== 'number' || !WIDTHS.has(value.width) || value.height !== 900
    || typeof value.captured_at !== 'number' || !Number.isSafeInteger(value.captured_at) || value.captured_at < 0
    || typeof value.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(value.sha256)
    || typeof value.png_base64 !== 'string' || value.png_base64.length > MAX_BASE64_CHARS
    || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value.png_base64)) invalid()
  const bytes = Buffer.from(value.png_base64, 'base64')
  if (bytes.length > MAX_IMAGE_BYTES || bytes.length < 57 || !bytes.subarray(0, 8).equals(SIGNATURE)
    || bytes.toString('base64') !== value.png_base64 || new Bun.CryptoHasher('sha256').update(bytes).digest('hex') !== value.sha256) invalid()
  let offset = 8; let header = false; let ended = false; let bytesPerPixel = 0
  const compressed: Uint8Array[] = []
  while (offset < bytes.length) {
    if (offset + 12 > bytes.length || ended) invalid()
    const length = bytes.readUInt32BE(offset); const end = offset + 12 + length
    if (end > bytes.length) invalid()
    const kind = bytes.toString('latin1', offset + 4, offset + 8)
    if (Bun.hash.crc32(bytes.subarray(offset + 4, end - 4)) !== bytes.readUInt32BE(end - 4)) invalid()
    const body = bytes.subarray(offset + 8, end - 4)
    if (!header) {
      if (kind !== 'IHDR' || length !== 13 || body.readUInt32BE(0) !== value.width || body.readUInt32BE(4) !== value.height
        || body[8] !== 8 || (body[9] !== 2 && body[9] !== 6) || body[10] !== 0 || body[11] !== 0 || body[12] !== 0) invalid()
      bytesPerPixel = body[9] === 6 ? 4 : 3; header = true
    } else if (kind === 'IDAT') {
      if (compressed.length >= 1024) invalid()
      compressed.push(body)
    } else if (kind === 'IEND' && length === 0 && compressed.length > 0) ended = true
    else invalid()
    offset = end
  }
  if (!ended) invalid()
  const stride = value.width * bytesPerPixel + 1
  const expectedBytes = stride * value.height
  let raster: Buffer
  try { raster = inflateSync(Buffer.concat(compressed), { maxOutputLength: expectedBytes }) } catch { invalid() }
  if (raster.length !== expectedBytes) invalid()
  for (let row = 0; row < value.height; row += 1) if ((raster[row * stride] ?? 255) > 4) invalid()
  // Forward only a canonical encoding of validated pixel data, never unused zlib bytes.
  const normalized = Buffer.concat([SIGNATURE, chunk('IHDR', bytes.subarray(16, 29)), chunk('IDAT', deflateSync(raster)), chunk('IEND', Buffer.alloc(0))])
  if (normalized.length > MAX_IMAGE_BYTES) invalid()
  return Object.freeze({ viewId: value.view_id, mime: 'image/png', width: value.width, height: value.height,
    capturedAt: value.captured_at, sourceSha256: value.sha256, normalization: 'png-raster-v1', byteLength: normalized.length,
    sha256: new Bun.CryptoHasher('sha256').update(normalized).digest('hex'), imageUrl: `data:image/png;base64,${normalized.toString('base64')}` })
}
