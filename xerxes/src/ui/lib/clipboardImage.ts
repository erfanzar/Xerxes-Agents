// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { execFile } from 'node:child_process'
import { chmodSync, closeSync, mkdirSync, openSync, readdirSync, readSync, statSync, unlinkSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'

import {
  ImageAttachmentError,
  loadImageAttachment,
  type LoadedImageAttachment
} from './imageAttachment.js'

/**
 * Read an image from the system clipboard into a bounded per-session temp
 * file, validated with the same magic-byte sniff and size caps as `/image`.
 *
 * macOS only, with two honest backends tried in order:
 *   1. `pngpaste <file>` (brew install pngpaste) when present on PATH;
 *   2. an osascript one-liner writing the clipboard's «class PNGf» data.
 *
 * Every other platform returns an explicit `unsupported` result — the UI
 * never fakes an attachment. All process execution goes through the
 * injectable `run` runner so tests supply a fake and no real clipboard or
 * binary is touched outside production.
 */

/** Number of paste files retained per session temp directory; older files are swept. */
export const CLIPBOARD_PASTE_KEEP = 20

/** Bytes read back for dimension sniffing (JPEG SOF can sit behind large APP segments). */
const DIMENSION_HEAD_BYTES = 256 * 1024

export interface ClipboardImageRunResult {
  readonly stderr?: string
  readonly stdout?: string
}

/** Injectable process runner: resolves on exit code 0, rejects otherwise (or ENOENT). */
export type ClipboardImageRunner = (cmd: string, args: readonly string[]) => Promise<ClipboardImageRunResult>

const execFileAsync = promisify(execFile)

const defaultRunner: ClipboardImageRunner = async (cmd, args) => {
  const result = await execFileAsync(cmd, [...args], { maxBuffer: 1024 * 1024, windowsHide: true })

  return { stderr: result.stderr, stdout: result.stdout }
}

export type ClipboardImageRead =
  | { kind: 'image'; attachment: LoadedImageAttachment; height?: number; width?: number }
  | { kind: 'no-image' }
  | { kind: 'not-image' }
  | { kind: 'too-large'; size: number }
  | { kind: 'unsupported'; platform: NodeJS.Platform }
  | { kind: 'error'; message: string }

export interface ReadClipboardImageOptions {
  /** Paste files retained in the temp directory (default CLIPBOARD_PASTE_KEEP). */
  readonly keep?: number
  /** Clock override for deterministic temp filenames in tests. */
  readonly now?: () => number
  readonly platform?: NodeJS.Platform
  readonly run?: ClipboardImageRunner
  /** Live session id; namespaces the temp directory. */
  readonly sessionId?: null | string
  /** Temp root override (default: system temp dir). */
  readonly tmpBase?: string
}

// Writes the clipboard's PNG data to the path passed as argv[1]. osascript
// exits nonzero when the clipboard holds no «class PNGf» payload, which the
// caller treats as "no image".
const OSASCRIPT_PNG_TO_FILE = [
  'on run argv',
  '  set outPath to item 1 of argv',
  '  set pngData to the clipboard as «class PNGf»',
  '  set fp to open for access (POSIX file outPath) with write permission',
  '  write pngData to fp',
  '  close access fp',
  'end run'
].join('\n')

let pasteCounter = 0

const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error))

function pasteDirName(sessionId: null | string | undefined): string {
  const sid = (sessionId || 'shared').replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 48) || 'shared'

  return `xerxes-clipboard-${sid}`
}

function ensurePasteDir(tmpBase: string, sessionId: null | string | undefined): string {
  const dir = join(tmpBase, pasteDirName(sessionId))

  mkdirSync(dir, { mode: 0o700, recursive: true })
  // mkdir mode only applies to newly created leaf directories; chmod so a
  // pre-existing directory with looser perms is tightened too.
  chmodSync(dir, 0o700)

  return dir
}

/** Best-effort retention sweep: keep the newest `keep` paste files, delete the rest. */
export function sweepPasteFiles(dir: string, keep: number): void {
  let names: string[]

  try {
    names = readdirSync(dir).filter(name => name.startsWith('paste-'))
  } catch {
    return
  }

  names.sort()

  const excess = names.length - Math.max(0, keep)

  if (excess <= 0) {
    return
  }

  for (const name of names.slice(0, excess)) {
    try {
      unlinkSync(join(dir, name))
    } catch {
      // Best effort: a stuck file is retried on the next sweep.
    }
  }
}

function isNonEmptyFile(path: string): boolean {
  try {
    return statSync(path).size > 0
  } catch {
    return false
  }
}

function removeQuietly(path: string): void {
  try {
    unlinkSync(path)
  } catch {
    // Missing or locked file; nothing honest to do here.
  }
}

function readHead(path: string): Uint8Array {
  let fd: null | number = null

  try {
    fd = openSync(path, 'r')
    const size = Math.min(statSync(path).size, DIMENSION_HEAD_BYTES)
    const head = new Uint8Array(size)

    readSync(fd, head, 0, size, 0)

    return head
  } catch {
    return new Uint8Array(0)
  } finally {
    if (fd !== null) {
      try {
        closeSync(fd)
      } catch {
        // Already closed.
      }
    }
  }
}

/** Sniff pixel dimensions from png/jpeg/gif/webp headers; undefined when unknown. */
export function sniffImageDimensions(
  bytes: Uint8Array,
  mediaType: string
): { height: number; width: number } | undefined {
  if (!bytes.length) {
    return undefined
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)

  if (mediaType === 'image/png' && bytes.length >= 24) {
    return { height: view.getUint32(20), width: view.getUint32(16) }
  }

  if (mediaType === 'image/gif' && bytes.length >= 10) {
    return { height: view.getUint16(8, true), width: view.getUint16(6, true) }
  }

  if (mediaType === 'image/jpeg') {
    let offset = 2

    while (offset + 9 < bytes.length) {
      if (bytes[offset] !== 0xff) {
        break
      }

      const marker = bytes[offset + 1]!

      // Standalone markers without a length payload.
      if (marker === 0x01 || marker === 0xd8 || (marker >= 0xd0 && marker <= 0xd7)) {
        offset += 2
        continue
      }

      const length = view.getUint16(offset + 2)

      // Start-of-frame markers (excluding DHT/DAC/RST aliases in the Cx range).
      if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
        return { height: view.getUint16(offset + 5), width: view.getUint16(offset + 7) }
      }

      offset += 2 + length
    }

    return undefined
  }

  if (mediaType === 'image/webp' && bytes.length >= 30 && String.fromCharCode(...bytes.slice(12, 16)) === 'VP8X') {
    const width = 1 + (bytes[24]! | (bytes[25]! << 8) | (bytes[26]! << 16))
    const height = 1 + (bytes[27]! | (bytes[28]! << 8) | (bytes[29]! << 16))

    return { height, width }
  }

  return undefined
}

/**
 * Attempt to materialize the clipboard image as a temp file and validate it.
 *
 * Never throws: every failure mode is a typed `ClipboardImageRead` variant so
 * the paste flow can report exactly what happened.
 */
export async function readClipboardImage(options: ReadClipboardImageOptions = {}): Promise<ClipboardImageRead> {
  const platform = options.platform ?? process.platform

  if (platform !== 'darwin') {
    return { kind: 'unsupported', platform }
  }

  const run = options.run ?? defaultRunner
  const keep = options.keep ?? CLIPBOARD_PASTE_KEEP
  const now = options.now ?? Date.now

  let dir: string

  try {
    dir = ensurePasteDir(options.tmpBase ?? tmpdir(), options.sessionId)
  } catch (error) {
    return { kind: 'error', message: `could not create the clipboard temp directory: ${errorMessage(error)}` }
  }

  const target = join(dir, `paste-${now()}-${process.pid}-${pasteCounter++}.png`)

  let wrote = false

  try {
    await run('pngpaste', [target])
    wrote = isNonEmptyFile(target)
  } catch {
    // pngpaste missing from PATH or no image on the clipboard; fall back.
    wrote = false
  }

  if (!wrote) {
    removeQuietly(target)

    try {
      await run('osascript', ['-e', OSASCRIPT_PNG_TO_FILE, target])
      wrote = isNonEmptyFile(target)
    } catch {
      wrote = false
    }
  }

  if (!wrote) {
    removeQuietly(target)

    return { kind: 'no-image' }
  }

  try {
    const attachment = await loadImageAttachment(target)

    sweepPasteFiles(dir, keep)

    const dimensions = sniffImageDimensions(readHead(target), attachment.mediaType)

    return { kind: 'image', attachment, ...(dimensions ?? {}) }
  } catch (error) {
    const size = isNonEmptyFile(target) ? statSync(target).size : 0

    removeQuietly(target)

    if (error instanceof ImageAttachmentError) {
      if (/per-image limit/.test(error.message)) {
        return { kind: 'too-large', size }
      }

      if (/not a png, jpeg, gif, or webp image/.test(error.message)) {
        return { kind: 'not-image' }
      }

      // Missing or emptied file: indistinguishable from no clipboard image.
      return { kind: 'no-image' }
    }

    return { kind: 'error', message: errorMessage(error) }
  }
}

/** The per-session temp directory used for clipboard pastes (exported for tests/diagnostics). */
export function clipboardPasteDir(sessionId: null | string | undefined, tmpBase: string = tmpdir()): string {
  return join(tmpBase, pasteDirName(sessionId))
}
