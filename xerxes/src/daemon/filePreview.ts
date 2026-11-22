// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { lstat, realpath, stat } from 'node:fs/promises'
import { dirname, extname, isAbsolute, relative, resolve } from 'node:path'

import { sniffImageMediaType } from '../core/multimodal.js'
import { OfficeZipEntryTooLargeError, UnsupportedOfficeZipError } from '../skills/powerpoint/errors.js'
import { openOfficeZip } from '../skills/powerpoint/zip.js'
import { readDelimited } from './previews/delimited.js'
import { PREVIEW_FILE_LIMITS } from './previews/limits.js'
import { readWorkbook, WorkbookFormatError } from './previews/spreadsheet.js'
import { DocumentFormatError, readWordDocument } from './previews/wordDocument.js'
import { XmlTooComplexError } from './previews/xmlTree.js'

const MAX_BYTES = 128 * 1024

/** Preview kinds beyond plain text that a client can ask `workspace.filePreview` for. */
export const PREVIEW_KINDS = ['table', 'image', 'pdf', 'sheet', 'document', 'html'] as const
export type PreviewKind = (typeof PREVIEW_KINDS)[number]

/** The rich-preview fields of a request. Absent `kinds` keeps the text-only reply older clients expect. */
export interface PreviewOptions {
  readonly kinds: ReadonlySet<PreviewKind>
  /** Zero-based workbook sheet for the `sheet` kind. */
  readonly sheet: number
}

/** Why a rich preview has nothing to show. The file can still be opened in its own app. */
export type PreviewUnavailableReason = 'too_large' | 'unsupported' | 'unreadable'

const EXTENSION_KINDS: Readonly<Record<string, PreviewKind>> = {
  '.csv': 'table', '.tsv': 'table', '.tab': 'table',
  '.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.gif': 'image', '.webp': 'image', '.svg': 'image',
  '.pdf': 'pdf',
  '.xlsx': 'sheet', '.xlsm': 'sheet',
  '.docx': 'document', '.docm': 'document',
  '.html': 'html', '.htm': 'html', '.xhtml': 'html',
}

const KIND_LABELS: Readonly<Record<PreviewKind, string>> = {
  table: 'tables', image: 'images', pdf: 'PDFs', sheet: 'workbooks', document: 'documents', html: 'web pages',
}

/** Validate `kinds` and `sheet` from request params; undefined when the client asked for text only. */
export function previewOptionsOf(params: Readonly<Record<string, unknown>>): PreviewOptions | undefined {
  const { kinds, sheet } = params
  if (sheet !== undefined && sheet !== null && (typeof sheet !== 'number' || !Number.isSafeInteger(sheet) || sheet < 0 || sheet > 4095)) {
    throw new TypeError('sheet must be a sheet index from 0 to 4095')
  }
  if (kinds === undefined || kinds === null) return undefined
  if (!Array.isArray(kinds) || kinds.length > 16 || kinds.some(kind => typeof kind !== 'string' || kind.length > 32)) {
    throw new TypeError('kinds must be a list of at most 16 preview kind names')
  }
  // Names this daemon does not know are a newer client's; they are ignored, not refused.
  const known = new Set(kinds.filter((kind): kind is PreviewKind => (PREVIEW_KINDS as readonly string[]).includes(kind)))
  return { kinds: known, sheet: typeof sheet === 'number' ? sheet : 0 }
}

/** The version a save must match: a hash of the file's bytes. */
const versionOf = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex')

/** Result of resolving a preview path against the workspace and optional read-only roots. */
export interface ResolvedPreviewPath {
  /** Canonical real path on disk. */
  readonly file: string
  /** Workspace-relative path for workspace files, or canonical absolute path for external read-only roots. */
  readonly local: string
  /** True when the file resolved inside an external `allowedReadRoots` entry rather than `cwd`. */
  readonly readOnly: boolean
}

const SENSITIVE_EXTERNAL_DIR_SEGMENTS: ReadonlySet<string> = new Set([
  '.ssh',
  '.gnupg',
  '.aws',
  '.kube',
  '.docker',
  'credentials',
  'credential',
  'secrets',
  'secret',
  'keychain',
])

const SENSITIVE_SYSTEM_GENERATED_SEGMENTS: ReadonlySet<string> = new Set([
  'credentials',
  'credential',
  'secrets',
  'secret',
  'tokens',
  'token',
  'auth',
  'oauth',
  'keys',
  'key',
  'keychain',
  'cookies',
  'cookie',
  'session_tokens',
])

const SENSITIVE_EXTERNAL_BASENAMES: ReadonlySet<string> = new Set([
  '.env',
  '.netrc',
  '.npmrc',
  '.pypirc',
  'credentials.json',
  'secrets.json',
  'token.json',
  'tokens.json',
  'oauth_creds.json',
  'application_default_credentials.json',
  'id_rsa',
  'id_ed25519',
  'id_ecdsa',
  'id_dsa',
])

const SENSITIVE_EXTERNAL_EXTENSIONS: ReadonlySet<string> = new Set([
  '.pem',
  '.key',
  '.p12',
  '.pfx',
  '.jks',
  '.keystore',
])

/** Relative path from `root` to `target` only when `target` is strictly contained inside `root`. */
function relativeIfContained(root: string, target: string): string | undefined {
  const rel = relative(root, target)
  if (!rel || rel === '.' || isAbsolute(rel)) return undefined
  const segments = rel.split(/[\\/]+/)
  if (segments.length === 0 || segments[0] === '..' || segments.includes('..')) return undefined
  return rel
}

/** True when a relative path inside an external read-only root enters a sensitive credential/secret location. */
function isSensitiveExternalRelativePath(relPath: string): boolean {
  const segments = relPath.split(/[\\/]+/).filter(Boolean).map(segment => segment.toLowerCase())
  if (segments.length === 0) return true
  let insideSystemGenerated = false
  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i]!
    const isLast = i === segments.length - 1
    if (seg === '.system_generated') {
      insideSystemGenerated = true
      continue
    }
    if (insideSystemGenerated && SENSITIVE_SYSTEM_GENERATED_SEGMENTS.has(seg)) {
      return true
    }
    if (!isLast && SENSITIVE_EXTERNAL_DIR_SEGMENTS.has(seg)) {
      return true
    }
    if (isLast) {
      if (SENSITIVE_EXTERNAL_BASENAMES.has(seg) || seg.startsWith('.env.')) return true
      if (SENSITIVE_EXTERNAL_EXTENSIONS.has(extname(seg))) return true
    }
  }
  return false
}

/**
 * Resolve a file path for read-only preview.
 *
 * Workspace containment remains strictly primary: relative paths (and symlinks
 * inside `cwd`) resolve only inside `cwd`. Absolute paths outside `cwd` are
 * allowed read-only ONLY when their canonical `realpath` is strictly contained
 * within one of `allowedReadRoots` and does not touch sensitive internal dirs.
 */
export async function resolvePreviewPath(
  cwd: string,
  path: unknown,
  allowedReadRoots?: readonly string[],
): Promise<ResolvedPreviewPath> {
  if (typeof path !== 'string' || !path || path.length > 4096 || path.includes('\0')) {
    throw new TypeError('Choose a valid workspace file path')
  }
  const rawWorkspaceRoot = resolve(cwd)
  const root = await realpath(rawWorkspaceRoot)
  const rawIsAbsolute = isAbsolute(path)
  const candidate = resolve(root, path)

  const canonicalAllowedRoots: string[] = []
  const rawAllowedRoots: string[] = []
  if (rawIsAbsolute && allowedReadRoots && allowedReadRoots.length > 0) {
    for (const rawRoot of allowedReadRoots) {
      if (typeof rawRoot !== 'string' || !rawRoot.trim() || rawRoot.includes('\0')) continue
      const normalizedRaw = resolve(rawRoot)
      rawAllowedRoots.push(normalizedRaw)
      try {
        const realAllowed = await realpath(normalizedRaw)
        if ((await stat(realAllowed)).isDirectory() && !canonicalAllowedRoots.includes(realAllowed)) {
          canonicalAllowedRoots.push(realAllowed)
        }
      } catch {
        // Missing or unreadable read root cannot authorize access.
      }
    }
  }

  let file: string
  try {
    file = await realpath(candidate)
  } catch (error) {
    // Refuse out-of-bounds traversal paths and dangling symlink escapes cleanly
    // without revealing whether an external target exists.
    const lexicallyInWorkspace =
      relativeIfContained(root, candidate) !== undefined ||
      relativeIfContained(rawWorkspaceRoot, candidate) !== undefined
    const lexicallyInAllowed =
      rawIsAbsolute &&
      (canonicalAllowedRoots.some(allowed => relativeIfContained(allowed, candidate) !== undefined) ||
        rawAllowedRoots.some(allowed => relativeIfContained(allowed, candidate) !== undefined))
    if (!lexicallyInWorkspace && !lexicallyInAllowed) {
      throw new Error('File preview is limited to the selected workspace')
    }
    try {
      const linkInfo = await lstat(candidate)
      if (linkInfo.isSymbolicLink()) {
        throw new Error('File preview is limited to the selected workspace')
      }
    } catch (lstatError) {
      if (lstatError instanceof Error && lstatError.message === 'File preview is limited to the selected workspace') {
        throw lstatError
      }
      // If an intermediate directory is a symlink escaping all allowed roots, refuse as out-of-bounds.
      try {
        const realParent = await realpath(dirname(candidate))
        const parentInWorkspace = realParent === root || relativeIfContained(root, realParent) !== undefined
        const parentInAllowed =
          rawIsAbsolute &&
          canonicalAllowedRoots.some(allowed => realParent === allowed || relativeIfContained(allowed, realParent) !== undefined)
        if (!parentInWorkspace && !parentInAllowed) {
          throw new Error('File preview is limited to the selected workspace')
        }
      } catch (parentError) {
        if (parentError instanceof Error && parentError.message === 'File preview is limited to the selected workspace') {
          throw parentError
        }
      }
    }
    throw error
  }

  // 1. Workspace root is always primary.
  const local = relativeIfContained(root, file)
  if (local !== undefined) {
    if (!(await stat(file)).isFile()) throw new Error('Choose a regular file to preview')
    return { file, local, readOnly: false }
  }

  // 2. Relative paths (including workspace symlinks) never escape into external read roots.
  if (!rawIsAbsolute) {
    throw new Error('File preview is limited to the selected workspace')
  }

  // 3. External absolute paths are read-only and must stay strictly inside a verified allowed root.
  for (const allowedRoot of canonicalAllowedRoots) {
    const relInAllowed = relativeIfContained(allowedRoot, file)
    if (relInAllowed === undefined) continue
    if (isSensitiveExternalRelativePath(relInAllowed)) {
      throw new Error('File preview is limited to the selected workspace')
    }
    if (!(await stat(file)).isFile()) throw new Error('Choose a regular file to preview')
    return { file, local: file, readOnly: true }
  }

  throw new Error('File preview is limited to the selected workspace')
}

/** Canonical file strictly inside the workspace for writes; symlinks out of it and non-files are refused. */
async function workspaceFileForEdit(cwd: string, path: unknown): Promise<{ file: string; local: string }> {
  if (typeof path !== 'string' || !path || path.length > 4096 || path.includes('\0')) {
    throw new TypeError('Choose a valid workspace file path')
  }
  const rawWorkspaceRoot = resolve(cwd)
  const root = await realpath(rawWorkspaceRoot)
  const candidate = resolve(root, path)
  let file: string
  try {
    file = await realpath(candidate)
  } catch (error) {
    const lexicallyInWorkspace =
      relativeIfContained(root, candidate) !== undefined ||
      relativeIfContained(rawWorkspaceRoot, candidate) !== undefined
    if (!lexicallyInWorkspace) {
      throw new Error('File editing is limited to the selected workspace')
    }
    throw error
  }
  const local = relativeIfContained(root, file)
  if (local === undefined) {
    throw new Error('File editing is limited to the selected workspace')
  }
  if (!(await stat(file)).isFile()) throw new Error('Choose a regular file to edit')
  return { file, local }
}

/** A text preview could not be made; the legacy reply throws this message as an RPC error. */
class NotTextError extends Error {}

async function textPreview(file: string, local: string, readOnly = false) {
  const blob = Bun.file(file)
  const truncated = blob.size > MAX_BYTES
  const bytes = new Uint8Array(await blob.slice(0, MAX_BYTES).arrayBuffer())
  let content: string
  try { content = new TextDecoder('utf-8', { fatal: true }).decode(bytes, { stream: truncated }) }
  catch { throw new NotTextError('This file is not UTF-8 text and cannot be previewed') }
  if (content.includes('\0')) throw new NotTextError('This binary file cannot be previewed as text')
  // Only a whole editable workspace file has a version.
  return {
    ok: true as const,
    path: local,
    content,
    truncated,
    ...(readOnly ? { read_only: true as const } : {}),
    ...(truncated || readOnly ? {} : { version: versionOf(bytes) }),
  }
}

/**
 * Explicit workspace reads never follow a symlink outside the selected workspace
 * (or outside verified `allowedReadRoots` for absolute read-only previews).
 *
 * This is the text preview older clients understand. previewWorkspaceFileRich
 * serves a client that names the `kinds` it renders: a file whose kind the client renders comes back parsed
 * (`kind` says which), and anything that cannot be shown is a typed
 * `{ok:false,reason}` reply rather than an RPC error.
 */
export async function previewWorkspaceFile(
  cwd: string,
  path: unknown,
  allowedReadRoots?: readonly string[],
) {
  const { file, local, readOnly } = await resolvePreviewPath(cwd, path, allowedReadRoots)
  try { return await textPreview(file, local, readOnly) }
  catch (error) { throw error instanceof NotTextError ? new Error(error.message) : error }
}

/** A preview for a client that renders `options.kinds`: see previewWorkspaceFile. */
export async function previewWorkspaceFileRich(
  cwd: string,
  path: unknown,
  options: PreviewOptions,
  allowedReadRoots?: readonly string[],
) {
  const { file, local, readOnly } = await resolvePreviewPath(cwd, path, allowedReadRoots)
  const size = (await stat(file)).size
  const kind = EXTENSION_KINDS[extname(local).toLowerCase()]
  if (kind === undefined || !options.kinds.has(kind)) {
    try { return { ...(await textPreview(file, local, readOnly)), kind: 'text' as const, size } }
    catch (error) {
      if (!(error instanceof NotTextError)) throw error
      return unavailable(local, size, 'unsupported', 'This kind of file has no preview here. Open it in its default app to see it.')
    }
  }
  const limit = PREVIEW_FILE_LIMITS[kind]
  if (size > limit) {
    return { ...unavailable(local, size, 'too_large', `This file is ${formatBytes(size)}. Previews of ${KIND_LABELS[kind]} are limited to ${formatBytes(limit)}.`), limit }
  }
  // Read one byte past the limit: a file that grew since stat is caught, not read whole.
  const bytes = new Uint8Array(await Bun.file(file).slice(0, limit + 1).arrayBuffer())
  if (bytes.byteLength > limit) {
    return { ...unavailable(local, bytes.byteLength, 'too_large', `This file is ${formatBytes(bytes.byteLength)}. Previews of ${KIND_LABELS[kind]} are limited to ${formatBytes(limit)}.`), limit }
  }
  const base = {
    ok: true as const,
    path: local,
    size: bytes.byteLength,
    ...(readOnly ? { read_only: true as const } : {}),
  }
  switch (kind) {
    case 'html': {
      let content: string
      try { content = new TextDecoder('utf-8', { fatal: true }).decode(bytes) }
      catch { return unavailable(local, size, 'unreadable', 'This web page is not UTF-8 text, so it cannot be shown here.') }
      // Saving goes through the 128 KiB text editor, so only an editable workspace page that fits has a version.
      return { ...base, kind, content, truncated: false, ...(!readOnly && bytes.byteLength <= MAX_BYTES ? { version: versionOf(bytes) } : {}) }
    }
    case 'image': {
      if (extname(local).toLowerCase() === '.svg') {
        let text: string
        try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes) }
        catch { return unavailable(local, size, 'unreadable', 'This SVG file is not UTF-8 text.') }
        if (!/<svg[\s>]/i.test(text)) return unavailable(local, size, 'unreadable', 'This file is not an SVG image.')
        return { ...base, kind, media_type: 'image/svg+xml', data: Buffer.from(bytes).toString('base64') }
      }
      // The bytes decide the type: a .png holding HTML is refused, a .png holding a JPEG still shows.
      const mediaType = sniffImageMediaType(bytes)
      if (!mediaType) return unavailable(local, size, 'unreadable', 'This file is not a PNG, JPEG, GIF or WebP image.')
      return { ...base, kind, media_type: mediaType, data: Buffer.from(bytes).toString('base64') }
    }
    case 'pdf': {
      if (!new TextDecoder('latin1').decode(bytes.subarray(0, 1024)).includes('%PDF-')) return unavailable(local, size, 'unreadable', 'This file is not a PDF document.')
      return { ...base, kind, media_type: 'application/pdf', data: Buffer.from(bytes).toString('base64') }
    }
    case 'table': {
      const table = readDelimited(bytes, local)
      return {
        ...base, kind, delimiter: table.delimiter, encoding: table.encoding, header: table.header, rows: table.rows,
        total_rows: table.totalRows, total_columns: table.totalColumns, truncated: table.truncated,
      }
    }
    case 'sheet':
    case 'document': {
      const office = await officePreview(local, bytes, kind, options.sheet)
      return office.ok && readOnly ? { ...office, read_only: true as const } : office
    }
  }
}

/** A file whose bytes start like this is an OLE compound file: legacy .xls/.doc, or an encrypted OOXML file. */
const isCompoundFile = (bytes: Uint8Array): boolean => bytes[0] === 0xd0 && bytes[1] === 0xcf && bytes[2] === 0x11 && bytes[3] === 0xe0

async function officePreview(local: string, bytes: Uint8Array, kind: 'sheet' | 'document', sheet: number) {
  const noun = kind === 'sheet' ? 'workbook' : 'document'
  if (isCompoundFile(bytes)) {
    return unavailable(local, bytes.byteLength, 'unreadable', `This ${noun} is password-protected or saved in the old binary Office format, so it cannot be previewed.`)
  }
  try {
    const zip = openOfficeZip(bytes)
    const base = { ok: true as const, path: local, size: bytes.byteLength }
    if (kind === 'sheet') {
      const workbook = await readWorkbook(zip, sheet)
      return {
        ...base, kind, sheets: workbook.sheets, sheet: workbook.sheet, rows: workbook.rows,
        total_rows: workbook.totalRows, total_columns: workbook.totalColumns, truncated: workbook.truncated,
        ...(workbook.notice ? { notice: workbook.notice } : {}),
      }
    }
    const document = readWordDocument(zip)
    return { ...base, kind, blocks: document.blocks, truncated: document.truncated }
  } catch (error) {
    if (error instanceof OfficeZipEntryTooLargeError || error instanceof XmlTooComplexError) {
      return unavailable(local, bytes.byteLength, 'too_large', `This ${noun} holds more content than a preview reads. Open it in its default app to see it.`)
    }
    if (error instanceof UnsupportedOfficeZipError || error instanceof WorkbookFormatError || error instanceof DocumentFormatError) {
      return unavailable(local, bytes.byteLength, 'unreadable', `This ${noun} could not be read: ${error.message}.`)
    }
    throw error
  }
}

function unavailable(path: string, size: number, reason: PreviewUnavailableReason, error: string) {
  return { ok: false as const, path, reason, error, size }
}

/** Bytes as people read them: 512 B, 48.2 KB, 10 MB. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1 }
  return `${value >= 10 || Number.isInteger(value) ? Math.round(value) : value.toFixed(1)} ${units[unit]}`
}

/**
 * Save an edit made in the app. The file must already exist inside the
 * workspace, be text the preview showed whole, and still be the version the
 * editor opened: if anything (the agent, another editor) changed it since,
 * nothing is written and the caller is told to reload. Writing in place keeps
 * the file's mode and identity.
 */
export async function writeWorkspaceFile(cwd: string, path: unknown, content: unknown, baseVersion: unknown) {
  if (typeof content !== 'string') throw new TypeError('content must be text')
  if (typeof baseVersion !== 'string' || !/^[0-9a-f]{64}$/.test(baseVersion)) throw new TypeError('base_version must be the version the file was opened at')
  const encoded = new TextEncoder().encode(content)
  if (encoded.byteLength > MAX_BYTES) throw new Error('Files over 128 KiB cannot be edited here')
  if (content.includes('\0')) throw new Error('Text files cannot contain NUL characters')
  const { file, local } = await workspaceFileForEdit(cwd, path)
  const current = new Uint8Array(await Bun.file(file).arrayBuffer())
  if (versionOf(current) !== baseVersion) {
    return { ok: false as const, conflict: true, path: local, error: 'This file changed on disk since you opened it. Reload it to see the new version, then edit again.' }
  }
  await Bun.write(file, encoded)
  return { ok: true as const, path: local, version: versionOf(encoded) }
}
