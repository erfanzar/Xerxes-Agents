// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * What the chat may show and open, shared by the bridge (preload and the VS
 * Code webview), the hosts (Electron main, the VS Code extension) and the
 * renderer, so every layer applies the same rules:
 *
 * - images are PNG, JPEG, GIF, WebP or SVG, at most 10 MB, and always reach
 *   the page as `data:` URLs (the renderer CSP stays `img-src 'self' data:`);
 * - links open outside the app only for http, https and mailto.
 *
 * Pure: no Node and no DOM beyond `URL`, because the sandboxed preload
 * bundles it.
 */

export const OPEN_EXTERNAL_CHANNEL = 'desktop:open-external'
export const IMAGE_DATA_CHANNEL = 'desktop:image-data'

export const CHAT_IMAGE_MAX_BYTES = 10 * 1024 * 1024
export const CHAT_IMAGE_PATH_MAX_LENGTH = 4096
export const EXTERNAL_URL_MAX_LENGTH = 2048

/** An image the chat can show, as a data URL, or the reason it cannot. */
export type ChatImageResult = { readonly ok: true; readonly url: string } | { readonly ok: false; readonly error: string }

const IMAGE_TYPES: Readonly<Record<string, string>> = {
  png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', gif: 'image/gif', webp: 'image/webp', svg: 'image/svg+xml',
}

export const UNSUPPORTED_IMAGE = 'only PNG, JPEG, GIF, WebP and SVG images can be shown'

/** The media type an image file's extension names, or undefined for anything else. */
export function chatImageTypeOf(path: string): string | undefined {
  const match = /\.([A-Za-z0-9]{1,5})$/.exec(path)
  return match?.[1] ? IMAGE_TYPES[match[1].toLowerCase()] : undefined
}

/** An image path as text, or the reason it is refused before any file is touched. */
export function checkChatImagePath(value: unknown): { readonly path: string } | { readonly error: string } {
  if (typeof value !== 'string' || !value) return { error: 'no image path was given' }
  if (value.length > CHAT_IMAGE_PATH_MAX_LENGTH) return { error: 'the image path is too long' }
  if (/[\x00-\x1f\x7f]/.test(value)) return { error: 'the image path contains control characters' }
  if (!chatImageTypeOf(value)) return { error: UNSUPPORTED_IMAGE }
  return { path: value }
}

const DATA_URL = /^data:(image\/(?:png|jpeg|gif|webp|svg\+xml));base64,([A-Za-z0-9+/]*={0,2})$/i

/** Check a data URL is a base64 image of an allowed type within the size cap. */
export function chatImageDataUrl(value: string): ChatImageResult {
  const match = DATA_URL.exec(value)
  if (!match?.[1]) {
    return { ok: false, error: /^data:image\//i.test(value) ? 'only base64 PNG, JPEG, GIF, WebP and SVG data URLs can be shown' : 'not an image data URL' }
  }
  const data = match[2] ?? ''
  if (!data || data.length % 4 !== 0) return { ok: false, error: 'the image data is not valid base64' }
  if (Math.floor((data.length * 3) / 4) > CHAT_IMAGE_MAX_BYTES) return { ok: false, error: 'the image is larger than 10 MB' }
  return { ok: true, url: `data:${match[1].toLowerCase()};base64,${data}` }
}

/** A host's answer to an image request, re-checked so the page only ever gets an image data URL. */
export function asChatImageResult(value: unknown): ChatImageResult {
  if (typeof value === 'object' && value !== null && 'ok' in value) {
    if (value.ok === true && 'url' in value && typeof value.url === 'string') return chatImageDataUrl(value.url)
    if (value.ok === false && 'error' in value && typeof value.error === 'string' && value.error) return { ok: false, error: value.error.slice(0, 300) }
  }
  return { ok: false, error: 'the app returned no image' }
}

/** One image request as it crosses the bridge: the path, and where a relative path is resolved. */
export interface ChatImageRequest {
  readonly path: string
  /** The task's folder; relative paths resolve against it (else the window's workspace). */
  readonly cwd?: string
  /** The session whose runtime reads the file in an SSH workspace. */
  readonly sessionKey?: string
}

/** The same request on the wire (snake_case, like the daemon's params). */
export interface ChatImageRequestWire {
  readonly path: string
  readonly cwd?: string
  readonly session_key?: string
}

/** A validated request in its wire shape. */
export function chatImageWire(request: ChatImageRequest): ChatImageRequestWire {
  return { path: request.path, ...(request.cwd ? { cwd: request.cwd } : {}), ...(request.sessionKey ? { session_key: request.sessionKey } : {}) }
}

const plainText = (value: string, max: number): boolean => value.length > 0 && value.length <= max && !/[\x00-\x1f\x7f]/.test(value)

/** Validate an untrusted image request (bridge and hosts both call this). */
export function chatImageRequestOf(value: unknown): ChatImageRequest | { readonly error: string } {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return { error: 'the image request is malformed' }
  const record = value as Record<string, unknown>
  for (const key of Object.keys(record)) if (key !== 'path' && key !== 'cwd' && key !== 'session_key') return { error: 'the image request is malformed' }
  const path = record.path
  if (typeof path !== 'string' || !plainText(path, CHAT_IMAGE_PATH_MAX_LENGTH)) return { error: typeof path === 'string' && path ? 'the image path is not valid' : 'no image path was given' }
  const cwd = record.cwd
  if (cwd !== undefined && cwd !== null && (typeof cwd !== 'string' || !plainText(cwd, CHAT_IMAGE_PATH_MAX_LENGTH) || !isAbsolutePath(cwd))) return { error: 'the task folder is not an absolute path' }
  const key = record.session_key
  if (key !== undefined && key !== null && (typeof key !== 'string' || !plainText(key, 256))) return { error: 'the session key is not valid' }
  return { path, ...(typeof cwd === 'string' ? { cwd } : {}), ...(typeof key === 'string' ? { sessionKey: key } : {}) }
}

/** Absolute on POSIX or Windows, without Node's path module (the renderer and preload have none). */
export function isAbsolutePath(path: string): boolean {
  return path.startsWith('/') || path.startsWith('\\\\') || /^[A-Za-z]:[\\/]/.test(path)
}

/** A `file://` URL's local path, or undefined for anything else (including a file URL on another host). */
export function fileUrlPath(value: string): string | undefined {
  if (!/^file:/i.test(value)) return undefined
  let url: URL
  try { url = new URL(value) } catch { return undefined }
  if (url.protocol !== 'file:' || (url.hostname && url.hostname !== 'localhost')) return undefined
  let path: string
  try { path = decodeURIComponent(url.pathname) } catch { return undefined }
  if (!path || path.includes('\0')) return undefined
  return /^\/[A-Za-z]:\//.test(path) ? path.slice(1) : path
}

/** The link to hand the system browser, or undefined unless it is an http, https or mailto URL. */
export function externalUrlOf(value: unknown): string | undefined {
  if (typeof value !== 'string' || !value || value.length > EXTERNAL_URL_MAX_LENGTH || /[\s\x00-\x1f\x7f]/.test(value)) return undefined
  let url: URL
  try { url = new URL(value) } catch { return undefined }
  const valid = url.protocol === 'http:' || url.protocol === 'https:' ? Boolean(url.hostname)
    : url.protocol === 'mailto:' ? Boolean(url.pathname)
    : false
  return valid && url.href.length <= EXTERNAL_URL_MAX_LENGTH ? url.href : undefined
}
