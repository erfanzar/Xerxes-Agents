// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtempSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { attachClipboardImage } from '../app/clipboardPaste.js'
import { clearAttachments, getAttachments } from '../app/attachmentsStore.js'
import {
  CLIPBOARD_PASTE_KEEP,
  clipboardPasteDir,
  readClipboardImage,
  sniffImageDimensions,
  sweepPasteFiles,
  type ClipboardImageRunner
} from '../lib/clipboardImage.js'
import { MAX_IMAGE_ATTACHMENT_BYTES } from '../lib/imageAttachment.js'
import { readHotkeyClipboardText } from '../app/useComposerState.js'

const PNG_WIDTH = 1024
const PNG_HEIGHT = 768

/** Minimal valid PNG: signature + IHDR width/height (data bytes are irrelevant for sniffing). */
function pngBytes(width = PNG_WIDTH, height = PNG_HEIGHT, size = 64): Uint8Array {
  const bytes = new Uint8Array(Math.max(size, 24))

  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  new DataView(bytes.buffer).setUint32(16, width)
  new DataView(bytes.buffer).setUint32(20, height)

  return bytes
}

const GIF_BYTES = (() => {
  const bytes = new Uint8Array(16)

  bytes.set([0x47, 0x49, 0x46, 0x38, 0x39, 0x61])
  new DataView(bytes.buffer).setUint16(6, 320, true)
  new DataView(bytes.buffer).setUint16(8, 200, true)

  return bytes
})()

/** Fake runner that succeeds for `cmd` by writing `payload` to the trailing path argument. */
const runnerWriting =
  (succeedsFor: string, payload: Uint8Array): ClipboardImageRunner =>
  async (cmd, args) => {
    if (cmd !== succeedsFor) {
      const error = new Error(`${cmd}: command not found`) as Error & { code: string }
      error.code = 'ENOENT'
      throw error
    }

    const target = args.at(-1)!

    writeFileSync(target, payload)

    return {}
  }

/** Fake runner where every backend fails (missing binaries or empty clipboard). */
const runnerFailing: ClipboardImageRunner = async cmd => {
  throw new Error(`${cmd}: no image data found on the clipboard`)
}

describe('readClipboardImage', () => {
  let tmpBase: string

  beforeEach(() => {
    tmpBase = mkdtempSync(join(tmpdir(), 'xerxes-clip-test-'))
  })

  afterEach(() => {
    rmSync(tmpBase, { force: true, recursive: true })
  })

  it('materializes a clipboard image via pngpaste with sniffed mime and dimensions', async () => {
    const result = await readClipboardImage({
      now: () => 1_700_000_000_000,
      platform: 'darwin',
      run: runnerWriting('pngpaste', pngBytes()),
      sessionId: 'sess-a',
      tmpBase
    })

    expect(result.kind).toBe('image')

    if (result.kind !== 'image') {
      return
    }

    expect(result.attachment.mediaType).toBe('image/png')
    expect(result.attachment.size).toBe(64)
    expect(result.width).toBe(PNG_WIDTH)
    expect(result.height).toBe(PNG_HEIGHT)
    expect(result.attachment.path).toContain(clipboardPasteDir('sess-a', tmpBase))
    expect(result.attachment.data).toBe(Buffer.from(pngBytes()).toString('base64'))
  })

  it('creates a per-session temp directory with 0o700 permissions', async () => {
    const result = await readClipboardImage({
      platform: 'darwin',
      run: runnerWriting('pngpaste', pngBytes()),
      sessionId: 'sess-b',
      tmpBase
    })

    expect(result.kind).toBe('image')

    const mode = statSync(clipboardPasteDir('sess-b', tmpBase)).mode & 0o777

    expect(mode).toBe(0o700)
  })

  it('falls back to osascript when pngpaste is missing', async () => {
    const calls: string[] = []
    const run: ClipboardImageRunner = async (cmd, args) => {
      calls.push(cmd)

      return runnerWriting('osascript', pngBytes())(cmd, args)
    }

    const result = await readClipboardImage({ platform: 'darwin', run, sessionId: 'sess-c', tmpBase })

    expect(calls).toEqual(['pngpaste', 'osascript'])
    expect(result.kind).toBe('image')
  })

  it('reports no-image when both backends fail or produce nothing', async () => {
    const result = await readClipboardImage({ platform: 'darwin', run: runnerFailing, sessionId: 'sess-d', tmpBase })

    expect(result).toEqual({ kind: 'no-image' })

    // No stray temp files are left behind.
    expect(readdirSync(clipboardPasteDir('sess-d', tmpBase))).toEqual([])
  })

  it('rejects an oversized clipboard image with a typed too-large result', async () => {
    const big = pngBytes(PNG_WIDTH, PNG_HEIGHT, MAX_IMAGE_ATTACHMENT_BYTES + 1)
    const result = await readClipboardImage({
      platform: 'darwin',
      run: runnerWriting('pngpaste', big),
      sessionId: 'sess-e',
      tmpBase
    })

    expect(result.kind).toBe('too-large')

    if (result.kind !== 'too-large') {
      return
    }

    expect(result.size).toBe(MAX_IMAGE_ATTACHMENT_BYTES + 1)
    expect(readdirSync(clipboardPasteDir('sess-e', tmpBase))).toEqual([])
  })

  it('rejects non-image clipboard payloads as not-image', async () => {
    const result = await readClipboardImage({
      platform: 'darwin',
      run: runnerWriting('pngpaste', new Uint8Array([1, 2, 3, 4, 5])),
      sessionId: 'sess-f',
      tmpBase
    })

    expect(result).toEqual({ kind: 'not-image' })
  })

  it('returns an explicit unsupported result off macOS without running anything', async () => {
    const run = vi.fn()

    await expect(readClipboardImage({ platform: 'linux', run: run as never, tmpBase })).resolves.toEqual({
      kind: 'unsupported',
      platform: 'linux'
    })
    await expect(readClipboardImage({ platform: 'win32', run: run as never, tmpBase })).resolves.toEqual({
      kind: 'unsupported',
      platform: 'win32'
    })
    expect(run).not.toHaveBeenCalled()
  })

  it('sweeps paste files beyond the retention bound', async () => {
    const dir = clipboardPasteDir('sess-g', tmpBase)
    const run = runnerWriting('pngpaste', pngBytes())
    let tick = 1_700_000_000_000

    for (let index = 0; index < CLIPBOARD_PASTE_KEEP + 5; index += 1) {
      tick += 1000

      const result = await readClipboardImage({ now: () => tick, platform: 'darwin', run, sessionId: 'sess-g', tmpBase })

      expect(result.kind).toBe('image')
    }

    expect(readdirSync(dir).filter(name => name.startsWith('paste-'))).toHaveLength(CLIPBOARD_PASTE_KEEP)
  })

  it('sweepPasteFiles deletes the oldest files first', () => {
    const dir = mkdtempSync(join(tmpdir(), 'xerxes-clip-sweep-'))

    try {
      for (let index = 0; index < 5; index += 1) {
        writeFileSync(join(dir, `paste-${1_000 + index}-1-0.png`), pngBytes())
      }

      writeFileSync(join(dir, 'keep-me.txt'), 'x')
      sweepPasteFiles(dir, 2)

      expect(readdirSync(dir).sort()).toEqual(['keep-me.txt', 'paste-1003-1-0.png', 'paste-1004-1-0.png'])
    } finally {
      rmSync(dir, { force: true, recursive: true })
    }
  })
})

describe('sniffImageDimensions', () => {
  it('reads png and gif headers', () => {
    expect(sniffImageDimensions(pngBytes(640, 480), 'image/png')).toEqual({ height: 480, width: 640 })
    expect(sniffImageDimensions(GIF_BYTES, 'image/gif')).toEqual({ height: 200, width: 320 })
    expect(sniffImageDimensions(new Uint8Array(0), 'image/png')).toBeUndefined()
  })
})

describe('attachClipboardImage wiring', () => {
  let tmpBase: string

  beforeEach(() => {
    tmpBase = mkdtempSync(join(tmpdir(), 'xerxes-clip-wire-'))
    clearAttachments()
  })

  afterEach(() => {
    clearAttachments()
    rmSync(tmpBase, { force: true, recursive: true })
  })

  it('adds a clipboard image to the pending attachments store with a confirmation', async () => {
    const outcome = await attachClipboardImage({
      platform: 'darwin',
      run: runnerWriting('pngpaste', pngBytes()),
      sessionId: 'sess-h',
      tmpBase
    })

    expect(outcome.attached).toBe(true)
    expect(outcome.message).toBe('📎 attached clipboard image (1KB, 1024x768) — /image to review')
    expect(getAttachments()).toHaveLength(1)
    expect(getAttachments()[0]?.mediaType).toBe('image/png')
  })

  it('reports an honest nothing-to-paste message when the clipboard has no image', async () => {
    const outcome = await attachClipboardImage({ platform: 'darwin', run: runnerFailing, sessionId: 'sess-i', tmpBase })

    expect(outcome.attached).toBe(false)
    expect(outcome.message).toBe('nothing to paste — the clipboard has no usable text or image')
    expect(getAttachments()).toEqual([])
  })

  it('reports unsupported platforms honestly', async () => {
    const outcome = await attachClipboardImage({ platform: 'linux', run: runnerFailing, tmpBase })

    expect(outcome.attached).toBe(false)
    expect(outcome.message).toContain('not supported on linux')
    expect(outcome.message).toContain('/image <path>')
  })

  it('reports oversized clipboard images with the per-image limit', async () => {
    const outcome = await attachClipboardImage({
      platform: 'darwin',
      run: runnerWriting('pngpaste', pngBytes(PNG_WIDTH, PNG_HEIGHT, MAX_IMAGE_ATTACHMENT_BYTES + 1)),
      sessionId: 'sess-j',
      tmpBase
    })

    expect(outcome.attached).toBe(false)
    expect(outcome.message).toContain('per-image limit')
    expect(getAttachments()).toEqual([])
  })
})

describe('readHotkeyClipboardText (text wins over image fallback)', () => {
  it('returns usable text and never reaches the image fallback', async () => {
    await expect(readHotkeyClipboardText(null, {}, async () => 'hello clipboard', async () => null)).resolves.toBe(
      'hello clipboard'
    )
  })

  it('falls back to OSC52 text when the native read is unusable', async () => {
    await expect(readHotkeyClipboardText(null, {}, async () => '   ', async () => 'osc52 text')).resolves.toBe(
      'osc52 text'
    )
  })

  it('returns null when neither reader yields usable text (image fallback gate)', async () => {
    await expect(readHotkeyClipboardText(null, {}, async () => null, async () => null)).resolves.toBeNull()
    await expect(readHotkeyClipboardText(null, {}, async () => '  \n ', async () => '')).resolves.toBeNull()
  })

  it('prefers OSC52 first on remote shells but still lets text win', async () => {
    const env = { SSH_TTY: '/dev/ttys001' } as NodeJS.ProcessEnv

    await expect(readHotkeyClipboardText(null, env, async () => 'native text', async () => null)).resolves.toBe(
      'native text'
    )
    await expect(readHotkeyClipboardText(null, env, async () => null, async () => 'remote text')).resolves.toBe(
      'remote text'
    )
  })
})
