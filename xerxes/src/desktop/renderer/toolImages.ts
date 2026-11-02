// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The pictures a tool step already carries, found in its recorded result:
 * computer_use screenshots (`image_url` data URLs), MCP and Anthropic-style
 * image content blocks, saved files from image generation, and a read of an
 * image file (shown from the path it read). Paths load through the host's
 * validated image read; data URLs are checked again where they are shown.
 */

import { chatImageTypeOf } from '../chatMediaShared.js'
import type { ToolItem } from './types.js'

export interface ToolImage {
  /** A `data:image/...` URL or a file path the host reads. */
  readonly src: string
  readonly alt: string
}

const MAX_TOOL_IMAGES = 8
const MAX_DEPTH = 8
const MAX_NODES = 4000
const BARE_DATA_IMAGE = /data:image\/(?:png|jpeg|gif|webp|svg\+xml);base64,[A-Za-z0-9+/]+={0,2}/gi
/** Tools whose argument names an image file they read or show. */
const IMAGE_READERS = new Set(['readfile', 'read', 'read_file', 'view_image', 'viewimage'])
const cache = new WeakMap<object, readonly ToolImage[]>()

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

const baseName = (path: string): string => path.split(/[\\/]/).filter(Boolean).at(-1) ?? path

function parsed(text: string): unknown {
  const trimmed = text.trim()
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) return undefined
  try { return JSON.parse(trimmed) as unknown } catch { return undefined }
}

const mediaTypeOf = (row: Record<string, unknown>): string | undefined => {
  for (const key of ['mime_type', 'mimeType', 'media_type', 'mediaType']) {
    const value = row[key]
    if (typeof value === 'string' && value.toLowerCase().startsWith('image/')) return value.toLowerCase()
  }
  return undefined
}

/** Every image a finished or running tool step carries, at most eight, in order. */
export function toolImagesOf(item: Pick<ToolItem, 'name' | 'input' | 'output'>): readonly ToolImage[] {
  const cached = cache.get(item)
  if (cached) return cached
  const out: ToolImage[] = []
  const seen = new Set<string>()
  const tool = item.name.split(/[.:]/).pop() ?? item.name
  const add = (src: string, alt: string): void => {
    if (out.length >= MAX_TOOL_IMAGES || seen.has(src)) return
    seen.add(src)
    out.push({ src, alt })
  }

  if (IMAGE_READERS.has(tool.toLowerCase())) {
    const args = parsed(item.input)
    const path = isRecord(args) ? (typeof args.file_path === 'string' ? args.file_path : typeof args.path === 'string' ? args.path : '') : ''
    if (path && chatImageTypeOf(path)) add(path, baseName(path))
  }

  const label = `Image from ${tool || 'the tool'}`
  let nodes = 0
  const walk = (value: unknown, depth: number): void => {
    if (out.length >= MAX_TOOL_IMAGES || depth > MAX_DEPTH || ++nodes > MAX_NODES) return
    if (typeof value === 'string') {
      if (value.startsWith('data:image/')) add(value.replaceAll(/\s/g, ''), label)
      return
    }
    if (Array.isArray(value)) { for (const entry of value) walk(entry, depth + 1); return }
    if (!isRecord(value)) return
    // MCP `{type:'image', data, mime_type}` and Anthropic `{type:'base64', media_type, data}` blocks.
    const source = isRecord(value.source) ? value.source : value
    const media = mediaTypeOf(source)
    if (media && typeof source.data === 'string' && source.data && (value.type === 'image' || source.type === 'base64')) {
      add(source.data.startsWith('data:') ? source.data : `data:${media};base64,${source.data.replaceAll(/\s/g, '')}`, label)
    }
    // A saved file (image generation's `saved: [{ path, media_type }]`).
    if (typeof value.path === 'string' && chatImageTypeOf(value.path) && mediaTypeOf(value)) add(value.path, baseName(value.path))
    for (const [key, entry] of Object.entries(value)) if (key !== 'data' || !media) walk(entry, depth + 1)
  }
  const result = parsed(item.output)
  if (result === undefined) {
    for (const match of item.output.matchAll(BARE_DATA_IMAGE)) add(match[0], label)
  } else walk(result, 0)

  cache.set(item, out)
  return out
}
