// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from './errors.js'

const DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024

export interface BinaryImage {
  readonly bytes: Uint8Array
  readonly mediaType: string
}

export interface DownloadImageOptions {
  readonly fetchImplementation?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>
  readonly maxBytes?: number
  readonly signal?: AbortSignal
}

export interface SerializeImageOptions {
  readonly addFormatPrefix?: boolean
  readonly maxImageB64Length?: number
}

/** Fetch an image into a portable byte representation without a Python/PIL dependency. */
export async function downloadImage(url: string, options: DownloadImageOptions = {}): Promise<BinaryImage> {
  let target: URL
  try {
    target = new URL(url)
  } catch {
    throw new ValidationError('url', 'must be an absolute image URL', url)
  }
  if (target.protocol !== 'http:' && target.protocol !== 'https:') {
    throw new ValidationError('url', 'must use http or https', url)
  }
  const maximum = options.maxBytes ?? DEFAULT_MAX_IMAGE_BYTES
  if (!Number.isInteger(maximum) || maximum < 1) {
    throw new ValidationError('maxBytes', 'must be a positive integer', maximum)
  }
  const response = await (options.fetchImplementation ?? fetch)(target, {
    headers: { 'User-Agent': 'Xerxes' },
    ...(options.signal ? { signal: options.signal } : {}),
  })
  if (!response.ok) {
    throw new Error('Error downloading image: HTTP ' + response.status)
  }
  const declaredLength = Number(response.headers.get('content-length') ?? 0)
  if (Number.isFinite(declaredLength) && declaredLength > maximum) {
    throw new Error('Image exceeds maximum size of ' + maximum + ' bytes')
  }
  const bytes = new Uint8Array(await response.arrayBuffer())
  if (bytes.byteLength > maximum) {
    throw new Error('Image exceeds maximum size of ' + maximum + ' bytes')
  }
  const mediaType = normalizeImageMediaType(response.headers.get('content-type')) ?? sniffImageMediaType(bytes)
  if (!mediaType) {
    throw new Error('Downloaded content is not a recognized image format')
  }
  return { bytes, mediaType }
}

/** Decode a data URL, a base64 image payload, or already-materialized bytes. */
export function coerceImage(value: BinaryImage | Uint8Array | string, mediaType = 'image/png'): BinaryImage {
  if (isBinaryImage(value)) {
    return { bytes: new Uint8Array(value.bytes), mediaType: normalizeRequiredImageMediaType(value.mediaType) }
  }
  if (value instanceof Uint8Array) {
    return { bytes: new Uint8Array(value), mediaType: normalizeRequiredImageMediaType(mediaType) }
  }
  if (typeof value !== 'string' || !value.trim()) {
    throw new ValidationError('image', 'must be non-empty bytes, base64, or a data URL', value)
  }
  const dataUrl = value.match(/^data:([^;,]+);base64,([A-Za-z0-9+/=\s]+)$/i)
  if (dataUrl) {
    return {
      mediaType: normalizeRequiredImageMediaType(dataUrl[1]!),
      bytes: decodeBase64(dataUrl[2]!),
    }
  }
  return {
    mediaType: normalizeRequiredImageMediaType(mediaType),
    bytes: decodeBase64(value),
  }
}

/** Encode image bytes for provider image_url payloads and durable JSON records. */
export function serializeImage(image: BinaryImage | Uint8Array | string, options: SerializeImageOptions = {}): string {
  const normalized = coerceImage(image)
  let encoded = Buffer.from(normalized.bytes).toString('base64')
  const maximum = options.maxImageB64Length
  if (maximum !== undefined) {
    if (!Number.isInteger(maximum) || maximum < 1) {
      throw new ValidationError('maxImageB64Length', 'must be a positive integer', maximum)
    }
    if (encoded.length > maximum) {
      encoded = encoded.slice(0, maximum) + '...'
    }
  }
  return options.addFormatPrefix ? 'data:' + normalized.mediaType + ';base64,' + encoded : encoded
}

function isBinaryImage(value: unknown): value is BinaryImage {
  return typeof value === 'object'
    && value !== null
    && 'bytes' in value
    && 'mediaType' in value
    && (value as BinaryImage).bytes instanceof Uint8Array
    && typeof (value as BinaryImage).mediaType === 'string'
}

function decodeBase64(value: string): Uint8Array {
  try {
    const normalized = value.replace(/\s/g, '')
    if (!/^[A-Za-z0-9+/]*={0,2}$/.test(normalized) || normalized.length % 4 === 1) {
      throw new Error('invalid base64')
    }
    return new Uint8Array(Buffer.from(normalized, 'base64'))
  } catch {
    throw new ValidationError('image', 'must contain valid base64 image data')
  }
}

function normalizeRequiredImageMediaType(value: string): string {
  const normalized = normalizeImageMediaType(value)
  if (!normalized) {
    throw new ValidationError('mediaType', 'must be a supported image media type', value)
  }
  return normalized
}

function normalizeImageMediaType(value: string | null): string | undefined {
  const normalized = value?.split(';')[0]?.trim().toLowerCase()
  return normalized && ['image/png', 'image/jpeg', 'image/gif', 'image/webp'].includes(normalized)
    ? normalized
    : undefined
}

function sniffImageMediaType(bytes: Uint8Array): string | undefined {
  if (bytes.length >= 8
    && bytes[0] === 0x89
    && bytes[1] === 0x50
    && bytes[2] === 0x4e
    && bytes[3] === 0x47) return 'image/png'
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'image/jpeg'
  if (bytes.length >= 6 && String.fromCharCode(...bytes.slice(0, 6)) === 'GIF87a') return 'image/gif'
  if (bytes.length >= 6 && String.fromCharCode(...bytes.slice(0, 6)) === 'GIF89a') return 'image/gif'
  if (bytes.length >= 12
    && String.fromCharCode(...bytes.slice(0, 4)) === 'RIFF'
    && String.fromCharCode(...bytes.slice(8, 12)) === 'WEBP') return 'image/webp'
  return undefined
}
