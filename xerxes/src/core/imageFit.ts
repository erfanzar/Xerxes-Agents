// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * An image scaled to what a model accepts without touching it again.
 *
 * Claude Code resizes any image wider than 2000px itself, and a resized image
 * stops its prompt cache from extending: a task holding three pasted retina
 * screenshots re-wrote its whole 680K conversation every round (3% cache hit).
 * The API also refuses an image over 5 MB of base64. Scaled here, the same
 * input always gives the same bytes, so the cache holds from round to round.
 *
 * Pure JavaScript (fast-png, jpeg-js): no native code, the same on every host.
 */
import { createHash } from 'node:crypto'
import { decode as decodePng } from 'fast-png'
import jpeg from 'jpeg-js'

/** Longest side Claude Code passes through unchanged. */
export const MAX_IMAGE_EDGE = 2000
/** Largest image the API accepts: 5 MB of base64 is 3.75 MB of bytes. */
export const MAX_IMAGE_BYTES = 3_750_000

export interface ImageData64 {
  readonly mediaType: string
  /** Base64, without a `data:` prefix. */
  readonly data: string
}

const fitted = new Map<string, ImageData64>()
const MAX_REMEMBERED = 64

/**
 * `image` within MAX_IMAGE_EDGE and MAX_IMAGE_BYTES, as a JPEG when it had to
 * change; unchanged when it already fits or is not a PNG or JPEG this can read.
 */
export function fitImage(image: ImageData64): ImageData64 {
  const bytes = Buffer.from(image.data, 'base64')
  const size = imageSize(bytes)
  if (!size) return image
  if (Math.max(size.width, size.height) <= MAX_IMAGE_EDGE && bytes.length <= MAX_IMAGE_BYTES) return image
  const key = createHash('sha256').update(image.data).digest('hex')
  const known = fitted.get(key)
  if (known) return known
  let result: ImageData64
  try {
    result = scaled(bytes, size.format)
  } catch {
    return image
  }
  fitted.set(key, result)
  if (fitted.size > MAX_REMEMBERED) fitted.delete(fitted.keys().next().value!)
  return result
}

/** Width, height and format from the file header, without decoding pixels. */
export function imageSize(bytes: Uint8Array): { width: number; height: number; format: 'png' | 'jpeg' } | undefined {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (bytes.length >= 24 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
    return { width: view.getUint32(16), height: view.getUint32(20), format: 'png' }
  }
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    let offset = 2
    while (offset + 9 < bytes.length) {
      if (bytes[offset] !== 0xff) { offset += 1; continue }
      const marker = bytes[offset + 1]!
      // Start-of-frame markers carry the dimensions (not DHT/JPG/DAC).
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        return { width: view.getUint16(offset + 7), height: view.getUint16(offset + 5), format: 'jpeg' }
      }
      offset += 2 + view.getUint16(offset + 2)
    }
  }
  return undefined
}

function scaled(bytes: Uint8Array, format: 'png' | 'jpeg'): ImageData64 {
  const source = format === 'png' ? pngRgba(bytes) : jpegRgba(bytes)
  const scale = Math.min(1, MAX_IMAGE_EDGE / Math.max(source.width, source.height))
  const target = scale < 1 ? boxScaled(source, Math.max(1, Math.round(source.width * scale)), Math.max(1, Math.round(source.height * scale))) : source
  for (const quality of [90, 80, 70, 60]) {
    const encoded = jpeg.encode({ data: Buffer.from(target.data.buffer, target.data.byteOffset, target.data.byteLength), width: target.width, height: target.height }, quality).data
    if (encoded.length <= MAX_IMAGE_BYTES || quality === 60) return { mediaType: 'image/jpeg', data: Buffer.from(encoded).toString('base64') }
  }
  throw new Error('unreachable')
}

interface Rgba { readonly width: number; readonly height: number; readonly data: Uint8Array }

function jpegRgba(bytes: Uint8Array): Rgba {
  const decoded = jpeg.decode(bytes, { useTArray: true, formatAsRGBA: true, maxMemoryUsageInMB: 2048, maxResolutionInMP: 400 })
  return { width: decoded.width, height: decoded.height, data: decoded.data }
}

/** Any PNG as 8-bit RGBA, transparency flattened onto white (JPEG has no alpha). */
function pngRgba(bytes: Uint8Array): Rgba {
  const png = decodePng(bytes)
  const { width, height, channels, depth } = png
  const pixels = width * height
  const out = new Uint8Array(pixels * 4)
  const palette = png.palette
  const sample = (index: number): number => {
    const value = png.data[index]!
    return depth === 16 ? value >> 8 : depth < 8 ? Math.round(value * 255 / ((1 << depth) - 1)) : value
  }
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    let r: number, g: number, b: number, a = 255
    if (palette) {
      const entry = palette[png.data[pixel]!] ?? [0, 0, 0]
      r = entry[0]!; g = entry[1]!; b = entry[2]!; a = entry[3] ?? 255
    } else if (channels >= 3) {
      r = sample(pixel * channels); g = sample(pixel * channels + 1); b = sample(pixel * channels + 2)
      if (channels === 4) a = sample(pixel * channels + 3)
    } else {
      r = g = b = sample(pixel * channels)
      if (channels === 2) a = sample(pixel * channels + 1)
    }
    const o = pixel * 4
    out[o] = Math.round((r * a + 255 * (255 - a)) / 255)
    out[o + 1] = Math.round((g * a + 255 * (255 - a)) / 255)
    out[o + 2] = Math.round((b * a + 255 * (255 - a)) / 255)
    out[o + 3] = 255
  }
  return { width, height, data: out }
}

/** Area-average downscale: each target pixel is the mean of the source pixels it covers. */
function boxScaled(source: Rgba, width: number, height: number): Rgba {
  const out = new Uint8Array(width * height * 4)
  const xRatio = source.width / width
  const yRatio = source.height / height
  for (let y = 0; y < height; y += 1) {
    const y0 = Math.floor(y * yRatio), y1 = Math.max(y0 + 1, Math.floor((y + 1) * yRatio))
    for (let x = 0; x < width; x += 1) {
      const x0 = Math.floor(x * xRatio), x1 = Math.max(x0 + 1, Math.floor((x + 1) * xRatio))
      let r = 0, g = 0, b = 0, count = 0
      for (let sy = y0; sy < y1 && sy < source.height; sy += 1) {
        let index = (sy * source.width + x0) * 4
        for (let sx = x0; sx < x1 && sx < source.width; sx += 1, index += 4) {
          r += source.data[index]!; g += source.data[index + 1]!; b += source.data[index + 2]!; count += 1
        }
      }
      const o = (y * width + x) * 4
      out[o] = Math.round(r / count); out[o + 1] = Math.round(g / count); out[o + 2] = Math.round(b / count); out[o + 3] = 255
    }
  }
  return { width, height, data: out }
}
