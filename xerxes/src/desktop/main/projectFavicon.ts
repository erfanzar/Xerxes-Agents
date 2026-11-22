// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * T3 Code-parity project logo / favicon discovery for local workspaces.
 *
 * Probes well-known favicon & brand logo paths inside the workspace root,
 * falls back to `<link rel="icon">` declarations in HTML / root route files,
 * validates real image bytes (strictly contained inside `cwd`, symlinks
 * resolved), optionally downscales oversized rasters to 64px PNGs, and returns
 * a safe `data:` URL (or `null` when the project has no logo on disk).
 */

import { open, readFile, realpath, stat } from 'node:fs/promises'
import { isAbsolute, join, relative, resolve, sep } from 'node:path'

import { sniffImageMediaType } from '../../core/multimodal.js'
import { PROJECT_ICON_MAX_DATA_BYTES, asProjectIconDataUrl } from '../chatMediaShared.js'

const MAX_CANDIDATE_BYTES = 6 * 1024 * 1024
const RESIZE_THRESHOLD_BYTES = 48 * 1024
const SOURCE_SCAN_BYTES = 128 * 1024
const POSITIVE_TTL_MS = 10 * 60_000
const NEGATIVE_TTL_MS = 60_000
const CACHE_CAPACITY = 512

const FAVICON_CANDIDATES = [
  'favicon.svg',
  'favicon.ico',
  'favicon.png',
  'public/favicon.svg',
  'public/favicon.ico',
  'public/favicon.png',
  'app/favicon.ico',
  'app/favicon.png',
  'app/icon.svg',
  'app/icon.png',
  'app/icon.ico',
  'src/favicon.svg',
  'src/favicon.ico',
  'src/favicon.png',
  'src/app/favicon.ico',
  'src/app/icon.svg',
  'src/app/icon.png',
  'assets/logo-128.png',
  'assets/icon.svg',
  'assets/icon.png',
  'assets/logo.svg',
  'assets/logo.png',
  'assets/logo.webp',
  'docs/assets/logo.svg',
  'docs/assets/logo.png',
  'docs/public/favicon.svg',
  'docs/public/favicon.png',
  'docs/public/logo.png',
  'images/logo.svg',
  'images/logo.png',
  'static/favicon.svg',
  'static/favicon.png',
  'static/favicon.ico',
  '.idea/icon.svg',
] as const

const ICON_SOURCE_FILES = [
  'index.html',
  'public/index.html',
  'src/index.html',
  'app/routes/__root.tsx',
  'src/routes/__root.tsx',
  'app/root.tsx',
  'src/root.tsx',
] as const

const FALLBACK_CANDIDATES = [
  'web/favicon.png',
  'logo.svg',
  'logo.png',
  'icon.svg',
  'icon.png',
] as const

const ICON_MIME_BY_EXT: Readonly<Record<string, string>> = {
  svg: 'image/svg+xml',
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  webp: 'image/webp',
  gif: 'image/gif',
  ico: 'image/x-icon',
}

const HTML_LINK_RE = /<link\b[^<>]*>/gi
const HTML_ICON_REL_RE = /\brel=["'](?:icon|shortcut icon|apple-touch-icon)["']/i
const HTML_ICON_HREF_RE = /\bhref=["']([^"'?#]+)/i
const OBJECT_ICON_REL_RE = /\brel\s*:\s*["'](?:icon|shortcut icon)["']/i
const OBJECT_ICON_HREF_RE = /\bhref\s*:\s*["']([^"'?#]+)/i
const SVG_ROOT_RE = /<svg[\s>]/i

function escapes(root: string, target: string): boolean {
  const inside = relative(root, target)
  return !inside || isAbsolute(inside) || inside.split(sep)[0] === '..'
}

function mimeFromExtension(filePath: string): string | undefined {
  const match = /\.([A-Za-z0-9]{1,5})$/.exec(filePath)
  return match?.[1] ? ICON_MIME_BY_EXT[match[1].toLowerCase()] : undefined
}

function isIcoBytes(bytes: Buffer): boolean {
  return bytes.length >= 6 && bytes[0] === 0 && bytes[1] === 0 && bytes[2] === 1 && bytes[3] === 0
}

export function extractDeclaredIconHref(source: string): string | null {
  for (const [tag] of source.matchAll(HTML_LINK_RE)) {
    if (!HTML_ICON_REL_RE.test(tag)) continue
    const match = tag.match(HTML_ICON_HREF_RE)
    if (match?.[1]) return match[1]
  }
  for (const chunk of source.split('}')) {
    if (!OBJECT_ICON_REL_RE.test(chunk)) continue
    const match = chunk.match(OBJECT_ICON_HREF_RE)
    if (match?.[1]) return match[1]
  }
  return null
}

async function resolveContainedFile(realRoot: string, relativePath: string): Promise<string | null> {
  if (!relativePath || /[\x00-\x1f\x7f]/.test(relativePath) || /^https?:/i.test(relativePath) || /^data:/i.test(relativePath)) {
    return null
  }
  const clean = relativePath.trim().replace(/^\/+/, '')
  if (!clean) return null
  const candidate = resolve(realRoot, clean)
  if (escapes(realRoot, candidate)) return null
  try {
    const realTarget = await realpath(candidate)
    if (escapes(realRoot, realTarget)) return null
    const info = await stat(realTarget)
    if (!info.isFile() || info.size === 0 || info.size > MAX_CANDIDATE_BYTES) return null
    return realTarget
  } catch {
    return null
  }
}

async function readHeadText(filePath: string): Promise<string | null> {
  try {
    const handle = await open(filePath, 'r')
    try {
      const buf = Buffer.alloc(SOURCE_SCAN_BYTES)
      const { bytesRead } = await handle.read(buf, 0, SOURCE_SCAN_BYTES, 0)
      return bytesRead > 0 ? buf.subarray(0, bytesRead).toString('utf8') : null
    } finally {
      await handle.close()
    }
  } catch {
    return null
  }
}

export async function findProjectIconPath(cwd: string): Promise<string | null> {
  if (!cwd || !isAbsolute(cwd) || /[\x00-\x1f\x7f]/.test(cwd)) return null
  let realRoot: string
  try {
    const info = await stat(cwd)
    if (!info.isDirectory()) return null
    realRoot = await realpath(cwd)
  } catch {
    return null
  }

  for (const candidate of FAVICON_CANDIDATES) {
    const hit = await resolveContainedFile(realRoot, candidate)
    if (hit && mimeFromExtension(hit)) return hit
  }

  for (const sourceFile of ICON_SOURCE_FILES) {
    const resolvedSource = await resolveContainedFile(realRoot, sourceFile)
    if (!resolvedSource) continue
    const text = await readHeadText(resolvedSource)
    if (!text) continue
    const href = extractDeclaredIconHref(text)
    if (!href) continue
    const clean = href.trim().replace(/^\/+/, '')
    for (const target of [join('public', clean), clean]) {
      const hit = await resolveContainedFile(realRoot, target)
      if (hit && mimeFromExtension(hit)) return hit
    }
  }

  for (const candidate of FALLBACK_CANDIDATES) {
    const hit = await resolveContainedFile(realRoot, candidate)
    if (hit && mimeFromExtension(hit)) return hit
  }

  return null
}

async function encodeIconFile(
  filePath: string,
  resizeToPng?: (bytes: Buffer) => Buffer | null,
): Promise<string | null> {
  const namedMime = mimeFromExtension(filePath)
  if (!namedMime) return null
  let bytes: Buffer
  try {
    bytes = await readFile(filePath)
  } catch {
    return null
  }
  if (bytes.length === 0 || bytes.length > MAX_CANDIDATE_BYTES) return null

  if (namedMime === 'image/svg+xml') {
    if (bytes.length > PROJECT_ICON_MAX_DATA_BYTES) return null
    let text: string
    try {
      text = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
    } catch {
      return null
    }
    if (!SVG_ROOT_RE.test(text)) return null
    return asProjectIconDataUrl(`data:image/svg+xml;base64,${bytes.toString('base64')}`)
  }

  if (namedMime === 'image/x-icon') {
    if (isIcoBytes(bytes) && bytes.length <= PROJECT_ICON_MAX_DATA_BYTES) {
      return asProjectIconDataUrl(`data:image/x-icon;base64,${bytes.toString('base64')}`)
    }
    // Some .ico files are actually PNGs — fall through to raster sniff.
  }

  const sniffed = sniffImageMediaType(bytes)
  if (!sniffed) return null

  if (resizeToPng && (bytes.length > RESIZE_THRESHOLD_BYTES || bytes.length > PROJECT_ICON_MAX_DATA_BYTES)) {
    try {
      const resized = resizeToPng(bytes)
      if (resized && resized.length > 0 && resized.length <= PROJECT_ICON_MAX_DATA_BYTES) {
        return asProjectIconDataUrl(`data:image/png;base64,${resized.toString('base64')}`)
      }
    } catch {
      // Fall through to raw encoding if small enough.
    }
  }

  if (bytes.length > PROJECT_ICON_MAX_DATA_BYTES) return null
  return asProjectIconDataUrl(`data:${sniffed};base64,${bytes.toString('base64')}`)
}

interface CacheEntry {
  readonly url: string | null
  readonly expiresAt: number
}

const iconCache = new Map<string, CacheEntry>()

export function clearProjectIconCacheForTests(): void {
  iconCache.clear()
}

/**
 * Resolves a project's real favicon/logo as a safe `data:` URL, or `null` if
 * the workspace has no logo file. Cached with TTL matching T3 Code.
 */
export async function resolveProjectIconDataUrl(
  cwd: string,
  resizeToPng?: (bytes: Buffer) => Buffer | null,
): Promise<string | null> {
  if (!cwd || !isAbsolute(cwd) || /[\x00-\x1f\x7f]/.test(cwd)) return null
  const now = Date.now()
  const cached = iconCache.get(cwd)
  if (cached && cached.expiresAt > now) return cached.url

  const iconPath = await findProjectIconPath(cwd)
  const url = iconPath ? await encodeIconFile(iconPath, resizeToPng) : null

  if (iconCache.size >= CACHE_CAPACITY) {
    const oldest = iconCache.keys().next().value
    if (oldest !== undefined) iconCache.delete(oldest)
  }
  iconCache.set(cwd, {
    url,
    expiresAt: now + (url ? POSITIVE_TTL_MS : NEGATIVE_TTL_MS),
  })
  return url
}
