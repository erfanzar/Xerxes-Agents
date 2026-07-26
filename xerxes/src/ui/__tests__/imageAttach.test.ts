// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import {
  $attachments,
  addAttachment,
  attachmentsTotalBytes,
  clearAttachments,
  getAttachments,
  promptSubmitImages,
  restoreAttachments,
  takeAttachments,
  type PendingAttachment
} from '../app/attachmentsStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { GatewayClient } from '../gatewayClient.js'
import {
  formatBytes,
  ImageAttachmentError,
  loadImageAttachment,
  MAX_IMAGE_ATTACHMENT_BYTES,
  MAX_IMAGE_ATTACHMENTS_TOTAL_BYTES,
  resolveAttachmentPath,
  sniffAttachmentMediaType
} from '../lib/imageAttachment.js'

const PNG_BYTES = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3, 4])
const JPEG_BYTES = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 16, 7, 8])
const GIF_BYTES = new Uint8Array([0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 1, 2])
const WEBP_BYTES = new Uint8Array([0x52, 0x49, 0x46, 0x46, 4, 0, 0, 0, 0x57, 0x45, 0x42, 0x50])

const PNG_B64 = Buffer.from(PNG_BYTES).toString('base64')

const attachment = (name: string, size: number): PendingAttachment => ({
  data: PNG_B64,
  mediaType: 'image/png',
  name,
  path: `/tmp/${name}`,
  size
})

const flush = async () => {
  await Promise.resolve()
  await Promise.resolve()
  await Promise.resolve()
}

/** Wait until an async attach/load chain settles, with a hard deadline. */
const waitFor = async (condition: () => boolean, timeoutMs = 2_000) => {
  const deadline = Date.now() + timeoutMs

  while (!condition()) {
    if (Date.now() >= deadline) {
      throw new Error('timed out waiting for slash command side effects')
    }

    await new Promise(resolveWait => setTimeout(resolveWait, 5))
  }
}

describe('sniffAttachmentMediaType', () => {
  it('identifies supported formats from magic bytes', () => {
    expect(sniffAttachmentMediaType(PNG_BYTES)).toBe('image/png')
    expect(sniffAttachmentMediaType(JPEG_BYTES)).toBe('image/jpeg')
    expect(sniffAttachmentMediaType(GIF_BYTES)).toBe('image/gif')
    expect(sniffAttachmentMediaType(WEBP_BYTES)).toBe('image/webp')
    expect(sniffAttachmentMediaType(new Uint8Array([1, 2, 3]))).toBeUndefined()
  })
})

describe('resolveAttachmentPath', () => {
  it('strips quotes, expands ~, and anchors relative paths at the session cwd', () => {
    expect(resolveAttachmentPath('shot.png', '/work/dir')).toBe(resolve('/work/dir', 'shot.png'))
    expect(resolveAttachmentPath('"./a b.png"', '/work')).toBe(resolve('/work', 'a b.png'))
    expect(resolveAttachmentPath('/abs/pic.png', '/work')).toBe(resolve('/abs/pic.png'))
    expect(resolveAttachmentPath('~/pic.png', '/work')).toBe(resolve(homedir(), 'pic.png'))
  })
})

describe('loadImageAttachment', () => {
  let dir: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'xerxes-image-attach-'))
  })

  afterEach(() => {
    rmSync(dir, { force: true, recursive: true })
  })

  it('loads a real image with sniffed mime and base64 data', async () => {
    const path = join(dir, 'shot.png')
    writeFileSync(path, PNG_BYTES)

    const loaded = await loadImageAttachment(path)

    expect(loaded).toEqual({
      data: PNG_B64,
      mediaType: 'image/png',
      name: 'shot.png',
      path,
      size: PNG_BYTES.byteLength
    })
  })

  it('rejects a missing file, a non-image, and an oversized image with clear errors', async () => {
    await expect(loadImageAttachment(join(dir, 'missing.png'))).rejects.toThrow(/image not found/)

    const textPath = join(dir, 'notes.png')
    writeFileSync(textPath, 'just text, not an image')
    await expect(loadImageAttachment(textPath)).rejects.toThrow(/not a png, jpeg, gif, or webp image/)

    const bigPath = join(dir, 'huge.jpg')
    const big = new Uint8Array(MAX_IMAGE_ATTACHMENT_BYTES + 1)
    big.set(JPEG_BYTES)
    writeFileSync(bigPath, big)
    await expect(loadImageAttachment(bigPath)).rejects.toThrow(/per-image limit/)
  })
})

describe('attachmentsStore', () => {
  afterEach(() => clearAttachments())

  it('adds, totals, takes, and restores pending attachments', () => {
    expect(addAttachment(attachment('a.png', 100))).toBeNull()
    expect(addAttachment(attachment('b.png', 200))).toBeNull()
    expect(getAttachments().map(item => item.name)).toEqual(['a.png', 'b.png'])
    expect(attachmentsTotalBytes(getAttachments())).toBe(300)

    // Duplicates by path are rejected.
    expect(addAttachment(attachment('a.png', 100))).toMatch(/already attached/)

    const taken = takeAttachments()
    expect(taken.map(item => item.name)).toEqual(['a.png', 'b.png'])
    expect(getAttachments()).toEqual([])

    restoreAttachments(taken)
    expect(getAttachments().map(item => item.name)).toEqual(['a.png', 'b.png'])

    expect(promptSubmitImages(getAttachments())).toEqual([
      { data: PNG_B64, media_type: 'image/png' },
      { data: PNG_B64, media_type: 'image/png' }
    ])
  })

  it('enforces the combined per-message cap', () => {
    const each = MAX_IMAGE_ATTACHMENTS_TOTAL_BYTES - 100
    expect(addAttachment(attachment('big-1.png', each))).toBeNull()
    expect(addAttachment(attachment('big-2.png', 200))).toMatch(/per-message limit/)
    expect(getAttachments()).toHaveLength(1)
  })
})

function makeSlashContext(request: ReturnType<typeof vi.fn>) {
  const sys: string[] = []

  return {
    context: {
      composer: {
        enqueue: vi.fn(),
        hasSelection: false,
        paste: vi.fn(),
        queueRef: { current: [] },
        selection: {
          captureScrolledRows: vi.fn(),
          clearSelection: vi.fn(),
          copySelection: vi.fn(),
          copySelectionNoClear: vi.fn(),
          getState: vi.fn(),
          shiftAnchor: vi.fn(),
          shiftSelection: vi.fn(),
          version: vi.fn()
        },
        setInput: vi.fn()
      },
      gateway: { gw: { request }, rpc: request },
      local: {
        catalog: null,
        getHistoryItems: vi.fn(() => []),
        getLastUserMsg: vi.fn(() => ''),
        maybeWarn: vi.fn(),
        setCatalog: vi.fn()
      },
      session: {
        closeSession: vi.fn(),
        die: vi.fn(),
        dieWithCode: vi.fn(),
        guardBusySessionSwitch: vi.fn(),
        newLiveSession: vi.fn(),
        newSession: vi.fn(),
        resetVisibleHistory: vi.fn(),
        resumeById: vi.fn(),
        setSessionStartedAt: vi.fn()
      },
      slashFlightRef: { current: 0 },
      transcript: {
        page: vi.fn(),
        panel: vi.fn(),
        send: vi.fn(),
        setHistoryItems: vi.fn(),
        sys: (text: string) => sys.push(text),
        trimLastExchange: vi.fn((items: unknown) => items)
      },
      voice: { setVoiceEnabled: vi.fn(), setVoiceRecordKey: vi.fn(), setVoiceTts: vi.fn() }
    } as never,
    sys
  }
}

describe('/image slash command', () => {
  let dir: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'xerxes-image-cmd-'))
    clearAttachments()
  })

  afterEach(() => {
    clearAttachments()
    resetUiState()
    rmSync(dir, { force: true, recursive: true })
  })

  it('attaches existing image files, lists them, and clears them', async () => {
    patchUiState({ info: { cwd: dir, model: 'm', skills: {}, tools: {} } as never, sid: 's1' })
    writeFileSync(join(dir, 'one.png'), PNG_BYTES)
    writeFileSync(join(dir, 'two.jpg'), JPEG_BYTES)
    const request = vi.fn().mockResolvedValue({})
    const { context, sys } = makeSlashContext(request)
    const handler = createSlashHandler(context)

    handler('/image one.png two.jpg')
    await waitFor(() => getAttachments().length === 2)

    expect(getAttachments().map(item => item.name)).toEqual(['one.png', 'two.jpg'])
    expect(getAttachments()[1]?.mediaType).toBe('image/jpeg')
    expect(sys.some(line => line.includes('📎 attached one.png'))).toBe(true)
    expect(sys.some(line => line.includes('📎 attached two.jpg'))).toBe(true)

    handler('/image')
    expect(sys.at(-1)).toContain('pending image attachments')
    expect(sys.at(-1)).toContain('one.png')

    handler('/image clear')
    expect(getAttachments()).toEqual([])
    expect(sys.at(-1)).toContain('cleared 2 image attachments')

    // The local command never called the daemon.
    expect(request).not.toHaveBeenCalled()
  })

  it('reports unreadable paths without attaching', async () => {
    patchUiState({ info: { cwd: dir, model: 'm', skills: {}, tools: {} } as never, sid: 's1' })
    writeFileSync(join(dir, 'real.png'), PNG_BYTES)
    const request = vi.fn().mockResolvedValue({})
    const { context, sys } = makeSlashContext(request)

    createSlashHandler(context)('/image real.png notreal.png')
    await flush()

    // notreal.png does not exist, so the whole tail is treated as a
    // generation prompt and forwarded to the daemon instead.
    expect(request).toHaveBeenCalledWith('slash.exec', { command: 'image real.png notreal.png', session_id: 's1' })
    expect(getAttachments()).toEqual([])
    expect(sys.some(line => line.includes('📎 attached'))).toBe(false)
  })

  it('surfaces load failures for files that exist but are not images', async () => {
    patchUiState({ info: { cwd: dir, model: 'm', skills: {}, tools: {} } as never, sid: 's1' })
    writeFileSync(join(dir, 'fake.png'), 'definitely not an image')
    const request = vi.fn().mockResolvedValue({})
    const { context, sys } = makeSlashContext(request)

    createSlashHandler(context)('/image fake.png')
    await waitFor(() => sys.some(line => line.includes('not a png, jpeg, gif, or webp image')))

    expect(getAttachments()).toEqual([])
  })
})

describe('gateway prompt.submit images', () => {
  it('forwards attachments on turn.submit and omits the param for plain text', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:image-submit' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }
    privateClient.rawRequest = async (method, params = {}) => {
      calls.push({ method, params })
      return { ok: true }
    }

    await client.request('prompt.submit', {
      images: [{ data: PNG_B64, media_type: 'image/png' }],
      session_id: 'live-image',
      text: 'what is this?'
    })
    await client.request('prompt.submit', { images: [], session_id: 'live-image', text: 'plain' })

    expect(calls).toEqual([
      {
        method: 'turn.submit',
        params: {
          images: [{ data: PNG_B64, media_type: 'image/png' }],
          session_key: 'live-image',
          text: 'what is this?'
        }
      },
      { method: 'turn.submit', params: { session_key: 'live-image', text: 'plain' } }
    ])
  })

  it('image.attach reads and validates a local image file', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'xerxes-image-rpc-'))
    try {
      writeFileSync(join(dir, 'pic.png'), PNG_BYTES)
      const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:image-attach' })

      const attached = (await client.request('image.attach', { cwd: dir, path: 'pic.png' })) as Record<string, unknown>
      expect(attached).toMatchObject({
        attached: true,
        data: PNG_B64,
        media_type: 'image/png',
        name: 'pic.png',
        size: PNG_BYTES.byteLength
      })

      await expect(client.request('image.attach', { cwd: dir, path: 'missing.png' })).rejects.toThrow(/image not found/)
      await expect(client.request('image.attach', {})).rejects.toThrow(ImageAttachmentError)
    } finally {
      rmSync(dir, { force: true, recursive: true })
    }
  })
})

describe('composer indicator formatting', () => {
  it('formats byte totals for the pending line', () => {
    expect(formatBytes(512)).toBe('1KB')
    expect(formatBytes(2 * 1024 * 1024)).toBe('2.0MB')
  })
})
