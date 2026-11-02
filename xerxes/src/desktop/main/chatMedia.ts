// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Reads an image the chat shows and hands it back as a data URL, so the page
 * keeps its `img-src 'self' data:` policy. Node only: the Electron main
 * process and the VS Code extension host both call it. Every request is
 * untrusted (the text comes from a model's reply), so a file is read only
 * when it is a regular file with an image extension, at most 10 MB, whose
 * bytes really are that kind of image. A relative path resolves inside the
 * workspace and must stay there, symlinks included.
 */

import { readFile, realpath, stat } from 'node:fs/promises'
import { isAbsolute, relative, resolve, sep } from 'node:path'

import { sniffImageMediaType } from '../../core/multimodal.js'
import {
  CHAT_IMAGE_MAX_BYTES, UNSUPPORTED_IMAGE, chatImageTypeOf, checkChatImagePath, fileUrlPath,
  type ChatImageRequest, type ChatImageResult,
} from '../chatMediaShared.js'

const fail = (error: string): ChatImageResult => ({ ok: false, error })

function fileSystemReason(error: unknown): string {
  const code = typeof error === 'object' && error !== null && 'code' in error ? error.code : undefined
  if (code === 'ENOENT' || code === 'ENOTDIR') return 'the file does not exist'
  if (code === 'EACCES' || code === 'EPERM') return 'permission to read the file was denied'
  if (code === 'ELOOP') return 'the path loops through symlinks'
  return 'the file could not be read'
}

const escapes = (root: string, target: string): boolean => {
  const inside = relative(root, target)
  return !inside || isAbsolute(inside) || inside.split(sep)[0] === '..'
}

const SVG_ROOT = /<svg[\s>]/i

/** The path a request names, as text: a `file://` URL becomes its local path. */
function requestedPath(value: string): string | undefined {
  return /^file:/i.test(value) ? fileUrlPath(value) : value
}

/**
 * A local image as a data URL. Absolute paths may point anywhere on this
 * machine (screenshots land in temporary folders); relative paths resolve
 * against `workspace` and are refused when they leave it.
 */
export async function readLocalChatImage(value: string, workspace: string | null): Promise<ChatImageResult> {
  const checked = checkChatImagePath(value)
  if ('error' in checked) return fail(checked.error)
  const path = requestedPath(value)
  if (path === undefined) return fail('only file URLs on this computer can be shown')
  let root: string | undefined
  let target: string
  if (isAbsolute(path)) target = resolve(path)
  else {
    if (!workspace || !isAbsolute(workspace)) return fail('a relative image path needs an open workspace')
    root = resolve(workspace)
    target = resolve(root, path)
    if (escapes(root, target)) return fail('a relative image path must stay inside the workspace')
  }
  let real: string
  try {
    real = await realpath(target)
    if (root !== undefined && escapes(await realpath(root), real)) return fail('a relative image path must stay inside the workspace')
  } catch (error) {
    return fail(fileSystemReason(error))
  }
  // A symlink named .png that points at a text file is refused by its target's name too.
  const named = chatImageTypeOf(real)
  if (!named) return fail(UNSUPPORTED_IMAGE)
  let bytes: Buffer
  try {
    const info = await stat(real)
    if (!info.isFile()) return fail('the path is not a regular file')
    if (info.size > CHAT_IMAGE_MAX_BYTES) return fail('the image is larger than 10 MB')
    if (info.size === 0) return fail('the image file is empty')
    bytes = await readFile(real)
  } catch (error) {
    return fail(fileSystemReason(error))
  }
  // The file may have grown between stat and read.
  if (bytes.length > CHAT_IMAGE_MAX_BYTES) return fail('the image is larger than 10 MB')
  if (named === 'image/svg+xml') {
    let text: string
    try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes) } catch { return fail('the SVG file is not UTF-8 text') }
    if (!SVG_ROOT.test(text)) return fail('the file is not an SVG image')
    return { ok: true, url: `data:image/svg+xml;base64,${bytes.toString('base64')}` }
  }
  // The bytes decide the type: a .png that holds HTML is refused, a .png that holds a JPEG still shows.
  const sniffed = sniffImageMediaType(bytes)
  if (!sniffed) return fail('the file is not a PNG, JPEG, GIF or WebP image')
  return { ok: true, url: `data:${sniffed};base64,${bytes.toString('base64')}` }
}

/** One runtime RPC (DaemonRpc.call satisfies it). */
export type RuntimeCall = (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>

/**
 * An image from an SSH workspace, read by the runtime on that host through
 * `workspace.filePreview` (the Files panel's read, with its workspace
 * containment and 128 KB bound). That read returns UTF-8 text only, so SVG is
 * the one kind it can carry; raster images are refused with that reason.
 */
export async function readRemoteChatImage(request: ChatImageRequest, call: RuntimeCall): Promise<ChatImageResult> {
  const checked = checkChatImagePath(request.path)
  if ('error' in checked) return fail(checked.error)
  const path = requestedPath(request.path)
  if (path === undefined) return fail('only file URLs on the workspace host can be shown')
  if (chatImageTypeOf(path) !== 'image/svg+xml') {
    return fail('the SSH workspace runtime sends text files only, so only SVG images from it can be shown')
  }
  let result: Record<string, unknown>
  try {
    result = await call('workspace.filePreview', { path, ...(request.sessionKey ? { session_key: request.sessionKey } : {}) })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return fail(message.replace(/^rpc -?\d+: /, '').slice(0, 300) || 'the runtime could not read the file')
  }
  if (result.ok !== true || typeof result.content !== 'string') {
    return fail(typeof result.error === 'string' && result.error ? result.error.slice(0, 300) : 'the runtime could not read the file')
  }
  if (result.truncated === true) return fail('the SVG is larger than the 128 KB the runtime sends')
  if (!SVG_ROOT.test(result.content)) return fail('the file is not an SVG image')
  return { ok: true, url: `data:image/svg+xml;base64,${Buffer.from(result.content, 'utf8').toString('base64')}` }
}
