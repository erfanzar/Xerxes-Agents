// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { homedir } from 'node:os'
import { basename, isAbsolute, resolve } from 'node:path'

/**
 * Client-side image attachment loader for the TUI `/image` command.
 *
 * The daemon re-validates every attachment at the `turn.submit` boundary
 * (xerxes/src/daemon/images.ts); this loader exists so the user gets an
 * immediate, specific error before anything is queued as pending. The magic
 * byte table mirrors core/multimodal.ts — the UI bundle is rooted at src/ui
 * and cannot import runtime internals, so the small sniffer is kept local.
 */

export const MAX_IMAGE_ATTACHMENT_BYTES = 10 * 1024 * 1024
export const MAX_IMAGE_ATTACHMENTS_TOTAL_BYTES = 20 * 1024 * 1024

export interface LoadedImageAttachment {
  /** Canonical base64 of the file bytes. */
  readonly data: string
  /** Magic-byte-sniffed mime, never extension-derived. */
  readonly mediaType: string
  readonly name: string
  readonly path: string
  readonly size: number
}

export class ImageAttachmentError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ImageAttachmentError'
  }
}

/** Identify png/jpeg/gif/webp payloads from magic bytes; undefined otherwise. */
export function sniffAttachmentMediaType(bytes: Uint8Array): string | undefined {
  if (bytes.length >= 8 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
    return 'image/png'
  }

  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return 'image/jpeg'
  }

  const head6 = bytes.length >= 6 ? String.fromCharCode(...bytes.slice(0, 6)) : ''

  if (head6 === 'GIF87a' || head6 === 'GIF89a') {
    return 'image/gif'
  }

  if (
    bytes.length >= 12 &&
    String.fromCharCode(...bytes.slice(0, 4)) === 'RIFF' &&
    String.fromCharCode(...bytes.slice(8, 12)) === 'WEBP'
  ) {
    return 'image/webp'
  }

  return undefined
}

/**
 * Resolve a user-typed path the same way other TUI file flows do: surrounding
 * quotes stripped, `~` expanded, relative paths anchored at the session cwd.
 * No sandboxing is applied beyond resolution — attaching an explicit,
 * user-named file is the same trust decision as `@`-mentioning one.
 */
export function resolveAttachmentPath(input: string, cwd: string): string {
  let value = input.trim()

  if (
    (value.startsWith('"') && value.endsWith('"') && value.length >= 2) ||
    (value.startsWith("'") && value.endsWith("'") && value.length >= 2)
  ) {
    value = value.slice(1, -1)
  }

  if (value === '~') {
    value = homedir()
  } else if (value.startsWith('~/')) {
    value = resolve(homedir(), value.slice(2))
  }

  return isAbsolute(value) ? resolve(value) : resolve(cwd, value)
}

/** Read one image file into a validated, provider-ready attachment. */
export async function loadImageAttachment(path: string): Promise<LoadedImageAttachment> {
  const file = Bun.file(path)

  if (!(await file.exists())) {
    throw new ImageAttachmentError(`image not found: ${path}`)
  }

  if (file.size > MAX_IMAGE_ATTACHMENT_BYTES) {
    throw new ImageAttachmentError(
      `${basename(path)} is ${formatBytes(file.size)} — the per-image limit is ${formatBytes(MAX_IMAGE_ATTACHMENT_BYTES)}`
    )
  }

  const bytes = new Uint8Array(await file.arrayBuffer())

  if (!bytes.byteLength) {
    throw new ImageAttachmentError(`${basename(path)} is empty`)
  }

  if (bytes.byteLength > MAX_IMAGE_ATTACHMENT_BYTES) {
    throw new ImageAttachmentError(
      `${basename(path)} is ${formatBytes(bytes.byteLength)} — the per-image limit is ${formatBytes(MAX_IMAGE_ATTACHMENT_BYTES)}`
    )
  }

  const mediaType = sniffAttachmentMediaType(bytes)

  if (!mediaType) {
    throw new ImageAttachmentError(`${basename(path)} is not a png, jpeg, gif, or webp image`)
  }

  return {
    data: Buffer.from(bytes).toString('base64'),
    mediaType,
    name: basename(path),
    path,
    size: bytes.byteLength
  }
}

export function formatBytes(value: number): string {
  if (value >= 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(1)}MB`
  }

  return `${Math.max(1, Math.round(value / 1024))}KB`
}
