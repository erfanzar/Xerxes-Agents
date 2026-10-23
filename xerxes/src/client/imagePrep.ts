// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { HUB_LIMITS } from '../hub/protocol.js'

/**
 * Image attachments for the phone composer. Images travel inside the
 * `hub.command` frame, which the Hub caps at 1 MiB, so the limits here are
 * derived from that cap rather than from what a camera produces. Validation is
 * pure; `prepareImage` takes the browser primitives as arguments so it is
 * testable without a DOM. Browser-safe.
 */

export const IMAGE_LIMITS = Object.freeze({
  maxImages: 3,
  /** Longest edge after client-side resize. */
  maxEdge: 1600,
  /** Source files larger than this are refused before decoding. */
  maxSourceBytes: 25 * 1024 * 1024,
  /** One encoded JPEG. Three of these base64-encoded stay under the frame cap. */
  maxEncodedBytes: 200 * 1024,
  /** Everything attached to one message, as raw JPEG bytes. */
  maxTotalEncodedBytes: 600 * 1024,
  minEdge: 320,
})

export const ACCEPTED_IMAGE_TYPES: readonly string[] = ['image/jpeg', 'image/png', 'image/webp', 'image/gif']

export interface PreparedImage {
  readonly id: string
  readonly media_type: 'image/jpeg'
  /** Base64, no data-URL prefix. */
  readonly data: string
  readonly width: number
  readonly height: number
  /** Raw JPEG size. */
  readonly bytes: number
  /** Data URL for the preview thumbnail (blob: URLs are blocked by the Hub's CSP). */
  readonly previewUrl: string
}

export type ImageCheck = { readonly ok: true } | { readonly ok: false; readonly message: string }

export interface FileLike {
  readonly name: string
  readonly type: string
  readonly size: number
}

const HEIC = /\.(heic|heif)$/i

export function checkImageFile(file: FileLike, attached: number): ImageCheck {
  if (attached >= IMAGE_LIMITS.maxImages) {
    return { ok: false, message: `You can attach up to ${IMAGE_LIMITS.maxImages} images. Remove one to add another.` }
  }
  if (file.type === 'image/heic' || file.type === 'image/heif' || HEIC.test(file.name)) {
    return { ok: false, message: 'HEIC photos are not supported. Choose a JPEG or PNG, or set the camera to "Most Compatible".' }
  }
  if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
    const shown = file.type === '' ? 'This file type' : file.type
    return { ok: false, message: `${shown} is not supported. Choose a JPEG, PNG, WebP or GIF image.` }
  }
  if (file.size > IMAGE_LIMITS.maxSourceBytes) {
    return { ok: false, message: `That image is larger than ${Math.round(IMAGE_LIMITS.maxSourceBytes / 1024 / 1024)} MB. Choose a smaller one.` }
  }
  if (file.size === 0) return { ok: false, message: 'That image is empty.' }
  return { ok: true }
}

/** Scale so the longest edge is at most `maxEdge`; never upscales. */
export function fitWithin(width: number, height: number, maxEdge: number = IMAGE_LIMITS.maxEdge): { readonly width: number; readonly height: number } {
  const longest = Math.max(width, height)
  if (longest <= maxEdge) return { width, height }
  const scale = maxEdge / longest
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) }
}

/** What an attachment set costs on the wire, as JSON-embedded base64 bytes. */
export function attachmentWireBytes(images: readonly Pick<PreparedImage, 'bytes'>[]): number {
  return images.reduce((total, image) => total + Math.ceil(image.bytes / 3) * 4, 0)
}

export function fitsWireBudget(images: readonly Pick<PreparedImage, 'bytes'>[]): boolean {
  return images.reduce((total, image) => total + image.bytes, 0) <= IMAGE_LIMITS.maxTotalEncodedBytes
    && attachmentWireBytes(images) < HUB_LIMITS.maxFrameBytes - 64 * 1024
}

export interface ImageEngine {
  /** Decode and draw at the target size, then encode as JPEG at `quality`. */
  encode(file: Blob, target: { readonly width: number; readonly height: number }, quality: number): Promise<{ readonly bytes: Uint8Array; readonly previewUrl: string }>
  /** Natural size of the source after EXIF orientation. */
  measure(file: Blob): Promise<{ readonly width: number; readonly height: number }>
}

export function bytesToBase64(bytes: Uint8Array): string {
  let binary = ''
  const chunk = 0x8000
  for (let offset = 0; offset < bytes.length; offset += chunk) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunk))
  }
  return btoa(binary)
}

/**
 * Resize to at most 1600px and re-encode as JPEG, stepping quality and then
 * size down until the result fits `maxEncodedBytes`. Throws a user-readable
 * Error when it cannot be made small enough.
 */
export async function prepareImage(
  file: Blob,
  engine: ImageEngine,
  options: { readonly id: string; readonly onProgress?: (fraction: number) => void } = { id: 'image' },
): Promise<PreparedImage> {
  const natural = await engine.measure(file)
  let target = fitWithin(natural.width, natural.height)
  const qualities = [0.85, 0.72, 0.6, 0.48]
  let attempts = 0
  const total = qualities.length * 3
  for (let shrink = 0; shrink < 3; shrink += 1) {
    for (const quality of qualities) {
      attempts += 1
      options.onProgress?.(Math.min(0.95, attempts / total))
      const encoded = await engine.encode(file, target, quality)
      if (encoded.bytes.length <= IMAGE_LIMITS.maxEncodedBytes) {
        options.onProgress?.(1)
        return {
          id: options.id,
          media_type: 'image/jpeg',
          data: bytesToBase64(encoded.bytes),
          width: target.width,
          height: target.height,
          bytes: encoded.bytes.length,
          previewUrl: encoded.previewUrl,
        }
      }
    }
    const next = fitWithin(target.width, target.height, Math.round(Math.max(target.width, target.height) * 0.75))
    if (Math.max(next.width, next.height) < IMAGE_LIMITS.minEdge) break
    target = next
  }
  throw new Error('That image is too detailed to attach. Try a smaller or simpler image.')
}

/** Browser engine over createImageBitmap and a canvas. */
export function browserImageEngine(): ImageEngine {
  return {
    async measure(file) {
      const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' })
      const size = { width: bitmap.width, height: bitmap.height }
      bitmap.close()
      return size
    },
    async encode(file, target, quality) {
      const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' })
      try {
        const canvas = document.createElement('canvas')
        canvas.width = target.width
        canvas.height = target.height
        const context = canvas.getContext('2d')
        if (!context) throw new Error('This browser cannot resize images.')
        // JPEG has no alpha; flatten transparent PNGs onto white.
        context.fillStyle = '#ffffff'
        context.fillRect(0, 0, target.width, target.height)
        context.drawImage(bitmap, 0, 0, target.width, target.height)
        const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/jpeg', quality))
        if (!blob) throw new Error('This browser could not encode the image.')
        const bytes = new Uint8Array(await blob.arrayBuffer())
        // A data URL, not a blob URL: the Hub's CSP is `img-src 'self' data:`.
        return { bytes, previewUrl: `data:image/jpeg;base64,${bytesToBase64(bytes)}` }
      } finally {
        bitmap.close()
      }
    },
  }
}
