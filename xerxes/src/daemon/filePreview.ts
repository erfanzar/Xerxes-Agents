// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { realpath, stat } from 'node:fs/promises'
import { extname, isAbsolute, relative, resolve } from 'node:path'

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

/** Canonical file inside the workspace; symlinks out of it and non-files are refused. */
async function workspaceFile(cwd: string, path: unknown, action: 'preview' | 'editing'): Promise<{ file: string; local: string }> {
  if (typeof path !== 'string' || !path || path.length > 4096 || path.includes('\0')) {
    throw new TypeError('Choose a valid workspace file path')
  }
  const root = await realpath(cwd)
  const file = await realpath(resolve(root, path))
  const local = relative(root, file)
  if (!local || local === '..' || local.startsWith('../') || isAbsolute(local)) {
    throw new Error(`File ${action} is limited to the selected workspace`)
  }
  if (!(await stat(file)).isFile()) throw new Error(`Choose a regular file to ${action === 'preview' ? 'preview' : 'edit'}`)
  return { file, local }
}

/** A text preview could not be made; the legacy reply throws this message as an RPC error. */
class NotTextError extends Error {}

async function textPreview(file: string, local: string) {
  const blob = Bun.file(file)
  const truncated = blob.size > MAX_BYTES
  const bytes = new Uint8Array(await blob.slice(0, MAX_BYTES).arrayBuffer())
  let content: string
  try { content = new TextDecoder('utf-8', { fatal: true }).decode(bytes, { stream: truncated }) }
  catch { throw new NotTextError('This file is not UTF-8 text and cannot be previewed') }
  if (content.includes('\0')) throw new NotTextError('This binary file cannot be previewed as text')
  // Only a whole file can be edited, so only a whole file has a version.
  return { ok: true as const, path: local, content, truncated, ...(truncated ? {} : { version: versionOf(bytes) }) }
}

/**
 * Explicit workspace reads never follow a symlink outside the selected workspace.
 *
 * This is the text preview older clients understand. previewWorkspaceFileRich
 * serves a client that names the `kinds` it renders: a file whose kind the client renders comes back parsed
 * (`kind` says which), and anything that cannot be shown is a typed
 * `{ok:false,reason}` reply rather than an RPC error.
 */
export async function previewWorkspaceFile(cwd: string, path: unknown) {
  const { file, local } = await workspaceFile(cwd, path, 'preview')
  try { return await textPreview(file, local) }
  catch (error) { throw error instanceof NotTextError ? new Error(error.message) : error }
}

/** A preview for a client that renders `options.kinds`: see previewWorkspaceFile. */
export async function previewWorkspaceFileRich(cwd: string, path: unknown, options: PreviewOptions) {
  const { file, local } = await workspaceFile(cwd, path, 'preview')
  const size = (await stat(file)).size
  const kind = EXTENSION_KINDS[extname(local).toLowerCase()]
  if (kind === undefined || !options.kinds.has(kind)) {
    try { return { ...(await textPreview(file, local)), kind: 'text' as const, size } }
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
  const base = { ok: true as const, path: local, size: bytes.byteLength }
  switch (kind) {
    case 'html': {
      let content: string
      try { content = new TextDecoder('utf-8', { fatal: true }).decode(bytes) }
      catch { return unavailable(local, size, 'unreadable', 'This web page is not UTF-8 text, so it cannot be shown here.') }
      // Saving goes through the 128 KiB text editor, so only a page that fits it has a version.
      return { ...base, kind, content, truncated: false, ...(bytes.byteLength <= MAX_BYTES ? { version: versionOf(bytes) } : {}) }
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
    case 'document':
      return officePreview(local, bytes, kind, options.sheet)
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
  const { file, local } = await workspaceFile(cwd, path, 'editing')
  const current = new Uint8Array(await Bun.file(file).arrayBuffer())
  if (versionOf(current) !== baseVersion) {
    return { ok: false as const, conflict: true, path: local, error: 'This file changed on disk since you opened it. Reload it to see the new version, then edit again.' }
  }
  await Bun.write(file, encoded)
  return { ok: true as const, path: local, version: versionOf(encoded) }
}
