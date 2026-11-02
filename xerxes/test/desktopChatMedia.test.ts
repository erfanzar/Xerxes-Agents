// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterAll, beforeAll, expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, symlink, truncate, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { createBridge, type BridgeTransport } from '../src/desktop/bridgeApi.js'
import {
  CHAT_IMAGE_MAX_BYTES, IMAGE_DATA_CHANNEL, OPEN_EXTERNAL_CHANNEL, asChatImageResult, chatImageDataUrl, chatImageRequestOf,
  checkChatImagePath, externalUrlOf, fileUrlPath,
} from '../src/desktop/chatMediaShared.js'
import { readLocalChatImage, readRemoteChatImage } from '../src/desktop/main/chatMedia.js'
import { PanelHost, type PanelDaemon, type PanelPorts } from '../src/vscode/panelHost.js'
import type { PushFrame, ResultFrame } from '../src/vscode/messages.js'

const PNG_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAACAAAAAQCAAAAABSayKFAAAAMElEQVR42mOQU9E2tLBz9Q6KiE/NLqqob+2ZNHP+0jWbdu4/dubyzQfPGEYVjCQFAOTZBBBEiOEqAAAAAElFTkSuQmCC'
const PNG = Buffer.from(PNG_BASE64, 'base64')
const SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><rect width="4" height="4"/></svg>'

let root = ''
let workspace = ''
let outside = ''

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), 'xerxes-chat-media-'))
  workspace = join(root, 'workspace')
  outside = join(root, 'outside')
  await mkdir(join(workspace, 'out'), { recursive: true })
  await mkdir(outside, { recursive: true })
  await writeFile(join(workspace, 'out', 'shot.png'), PNG)
  await writeFile(join(workspace, 'diagram.svg'), SVG)
  await writeFile(join(workspace, 'fake.svg'), 'not a picture')
  await writeFile(join(workspace, 'html.png'), '<html><script>alert(1)</script></html>')
  await writeFile(join(workspace, 'empty.png'), '')
  await writeFile(join(workspace, 'notes.txt'), 'hello')
  await mkdir(join(workspace, 'folder.png'))
  await writeFile(join(outside, 'secret.png'), PNG)
  await writeFile(join(outside, 'secret.txt'), 'token=123')
  await symlink(join(outside, 'secret.png'), join(workspace, 'escape.png'))
  await symlink(join(outside, 'secret.txt'), join(workspace, 'renamed.png'))
  // Sparse: the size check refuses it before a byte is read.
  await writeFile(join(workspace, 'huge.png'), PNG)
  await truncate(join(workspace, 'huge.png'), CHAT_IMAGE_MAX_BYTES + 1)
})

afterAll(async () => { await rm(root, { recursive: true, force: true }) })

test('links open outside the app only for http, https and mailto, within the length cap', () => {
  expect(externalUrlOf('https://example.com/a?b=1#c')).toBe('https://example.com/a?b=1#c')
  expect(externalUrlOf('http://example.com')).toBe('http://example.com/')
  expect(externalUrlOf('mailto:someone@example.com')).toBe('mailto:someone@example.com')
  // Credentials in the URL are kept as written, not stripped.
  expect(externalUrlOf('https://user:pw@example.com/')).toBe('https://user:pw@example.com/')
  for (const bad of ['javascript:alert(1)', 'file:///etc/passwd', 'ftp://example.com', 'data:text/html;base64,PGI+', 'vscode://x', 'xerxes://open',
    'https://', 'mailto:', 'https://exa mple.com', 'https://example.com/\n', '/relative', '', 'https://example.com/' + 'a'.repeat(2048)]) {
    expect(externalUrlOf(bad)).toBeUndefined()
  }
  expect(externalUrlOf(42)).toBeUndefined()
  expect(externalUrlOf({ href: 'https://example.com' })).toBeUndefined()
})

test('image paths and data URLs are checked before any file is touched', () => {
  expect(checkChatImagePath('a/b.PNG')).toEqual({ path: 'a/b.PNG' })
  for (const ok of ['x.jpg', 'x.jpeg', 'x.gif', 'x.webp', 'x.svg']) expect('path' in checkChatImagePath(ok)).toBe(true)
  for (const bad of ['x.txt', 'x.html', 'x.png.exe', 'x', 'x.svgz', 'x.png\0', 'x\n.png', '']) expect('error' in checkChatImagePath(bad)).toBe(true)
  expect(checkChatImagePath('x'.repeat(5000) + '.png')).toEqual({ error: 'the image path is too long' })

  expect(chatImageDataUrl(`data:image/png;base64,${PNG_BASE64}`)).toEqual({ ok: true, url: `data:image/png;base64,${PNG_BASE64}` })
  expect(chatImageDataUrl('data:IMAGE/PNG;base64,AAAA')).toEqual({ ok: true, url: 'data:image/png;base64,AAAA' })
  expect(chatImageDataUrl('data:text/html;base64,PGI+').ok).toBe(false)
  expect(chatImageDataUrl('data:image/bmp;base64,AAAA').ok).toBe(false)
  expect(chatImageDataUrl('data:image/png,<svg>').ok).toBe(false)
  expect(chatImageDataUrl('data:image/png;base64,abc')).toEqual({ ok: false, error: 'the image data is not valid base64' })
  expect(chatImageDataUrl(`data:image/png;base64,${'A'.repeat(Math.ceil((CHAT_IMAGE_MAX_BYTES + 3) / 3) * 4)}`)).toEqual({ ok: false, error: 'the image is larger than 10 MB' })
  expect(chatImageDataUrl('javascript:alert(1)').ok).toBe(false)

  expect(fileUrlPath('file:///tmp/a%20b.png')).toBe('/tmp/a b.png')
  expect(fileUrlPath('file://other-host/share/a.png')).toBeUndefined()
  expect(fileUrlPath('file:///C:/shots/a.png')).toBe('C:/shots/a.png')
})

test('an image request crossing the bridge is a closed, plain-text record', () => {
  expect(chatImageRequestOf({ path: 'a.png', cwd: '/w', session_key: 's1' })).toEqual({ path: 'a.png', cwd: '/w', sessionKey: 's1' })
  expect(chatImageRequestOf({ path: 'a.png' })).toEqual({ path: 'a.png' })
  expect(chatImageRequestOf({ path: 'a.png', extra: 1 })).toEqual({ error: 'the image request is malformed' })
  expect(chatImageRequestOf(['a.png'])).toEqual({ error: 'the image request is malformed' })
  expect(chatImageRequestOf('a.png')).toEqual({ error: 'the image request is malformed' })
  expect(chatImageRequestOf({ path: '' })).toEqual({ error: 'no image path was given' })
  expect(chatImageRequestOf({ path: 'a.png', cwd: 'relative' })).toEqual({ error: 'the task folder is not an absolute path' })
  expect(chatImageRequestOf({ path: 'a.png', session_key: 'x\n' })).toEqual({ error: 'the session key is not valid' })
  // The page only ever receives an image data URL, whatever the host answers.
  expect(asChatImageResult({ ok: true, url: 'javascript:alert(1)' }).ok).toBe(false)
  expect(asChatImageResult({ ok: true, url: 'file:///etc/passwd' }).ok).toBe(false)
  expect(asChatImageResult(null)).toEqual({ ok: false, error: 'the app returned no image' })
  expect(asChatImageResult({ ok: false, error: 'missing' })).toEqual({ ok: false, error: 'missing' })
})

test('a local image is read as a data URL: workspace-relative, absolute and file URLs', async () => {
  const expected = { ok: true as const, url: `data:image/png;base64,${PNG_BASE64}` }
  expect(await readLocalChatImage('out/shot.png', workspace)).toEqual(expected)
  expect(await readLocalChatImage('./out/../out/shot.png', workspace)).toEqual(expected)
  expect(await readLocalChatImage(join(workspace, 'out', 'shot.png'), null)).toEqual(expected)
  // Absolute paths may be anywhere on this computer (screenshots land in temporary folders).
  expect(await readLocalChatImage(join(outside, 'secret.png'), workspace)).toEqual(expected)
  expect(await readLocalChatImage(`file://${join(workspace, 'out', 'shot.png')}`, null)).toEqual(expected)
  const svg = await readLocalChatImage('diagram.svg', workspace)
  expect(svg).toEqual({ ok: true, url: `data:image/svg+xml;base64,${Buffer.from(SVG).toString('base64')}` })
})

test('traversal, symlink escapes, bad extensions, directories, oversize and fake images are refused with a reason', async () => {
  const reason = async (path: string, cwd: string | null = workspace) => {
    const result = await readLocalChatImage(path, cwd)
    if (result.ok) throw new Error(`${path} was read`)
    return result.error
  }
  expect(await reason('../outside/secret.png')).toBe('a relative image path must stay inside the workspace')
  expect(await reason('out/../../outside/secret.png')).toBe('a relative image path must stay inside the workspace')
  expect(await reason('escape.png')).toBe('a relative image path must stay inside the workspace')
  // A link named .png whose target is a text file is refused by the target's name.
  expect(await reason(join(workspace, 'renamed.png'))).toBe('only PNG, JPEG, GIF, WebP and SVG images can be shown')
  expect(await reason('notes.txt')).toBe('only PNG, JPEG, GIF, WebP and SVG images can be shown')
  expect(await reason(join(workspace, 'notes.txt'))).toBe('only PNG, JPEG, GIF, WebP and SVG images can be shown')
  expect(await reason('folder.png')).toBe('the path is not a regular file')
  expect(await reason('huge.png')).toBe('the image is larger than 10 MB')
  expect(await reason('empty.png')).toBe('the image file is empty')
  expect(await reason('html.png')).toBe('the file is not a PNG, JPEG, GIF or WebP image')
  expect(await reason('fake.svg')).toBe('the file is not an SVG image')
  expect(await reason('missing.png')).toBe('the file does not exist')
  expect(await reason('out/shot.png', null)).toBe('a relative image path needs an open workspace')
  expect(await reason('/dev/null.png')).toBe('the file does not exist')
  expect(await reason('file://elsewhere/a.png')).toBe('only file URLs on this computer can be shown')
})

test('an SSH workspace image is read by its runtime through workspace.filePreview, never from this disk', async () => {
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  const runtime = (answer: Record<string, unknown>) => async (method: string, params: Record<string, unknown>) => { calls.push({ method, params }); return answer }
  expect(await readRemoteChatImage({ path: 'docs/d.svg', sessionKey: 's1' }, runtime({ ok: true, content: SVG, truncated: false })))
    .toEqual({ ok: true, url: `data:image/svg+xml;base64,${Buffer.from(SVG).toString('base64')}` })
  expect(calls).toEqual([{ method: 'workspace.filePreview', params: { path: 'docs/d.svg', session_key: 's1' } }])
  expect(await readRemoteChatImage({ path: 'a.svg' }, runtime({ ok: true, content: SVG, truncated: true })))
    .toEqual({ ok: false, error: 'the SVG is larger than the 128 KB the runtime sends' })
  expect(await readRemoteChatImage({ path: 'a.svg' }, runtime({ ok: false, error: 'File preview is limited to the selected workspace' })))
    .toEqual({ ok: false, error: 'File preview is limited to the selected workspace' })
  expect((await readRemoteChatImage({ path: 'a.svg' }, runtime({ ok: true, content: 'plain text' }))).ok).toBe(false)
  expect(await readRemoteChatImage({ path: 'a.svg' }, async () => { throw new Error('rpc -32000: connection closed') }))
    .toEqual({ ok: false, error: 'connection closed' })
  calls.length = 0
  // Raster images cannot cross the text-only read: refused with that reason, no call made.
  const png = await readRemoteChatImage({ path: 'shot.png' }, runtime({ ok: true, content: '' }))
  expect(png).toEqual({ ok: false, error: 'the SSH workspace runtime sends text files only, so only SVG images from it can be shown' })
  expect((await readRemoteChatImage({ path: 'notes.txt' }, runtime({ ok: true, content: '' }))).ok).toBe(false)
  expect(calls).toEqual([])
})

function recordingTransport(answer: (channel: string, args: unknown[]) => unknown = () => true) {
  const invoked: Array<{ channel: string; args: unknown[] }> = []
  const transport: BridgeTransport = {
    async invoke(channel, ...args) { invoked.push({ channel, args }); return answer(channel, args) },
    on: () => () => {},
    setZoomFactor: () => {},
  }
  return { transport, invoked }
}

test('the bridge forwards only valid links and image requests to the host', async () => {
  const { transport, invoked } = recordingTransport((channel) => channel === IMAGE_DATA_CHANNEL ? { ok: true, url: 'data:image/png;base64,AAAA' } : true)
  const bridge = createBridge(transport)
  await expect(bridge.openExternal('javascript:alert(1)')).rejects.toThrow('Only http, https and mailto')
  await expect(bridge.openExternal('file:///etc/passwd')).rejects.toThrow('Only http, https and mailto')
  expect(await bridge.openExternal('https://example.com/docs')).toBe(true)
  expect(await bridge.imageData({ path: '' })).toEqual({ ok: false, error: 'no image path was given' })
  expect(await bridge.imageData({ path: 'a.png', cwd: '/w', session_key: 'k' })).toEqual({ ok: true, url: 'data:image/png;base64,AAAA' })
  expect(invoked).toEqual([
    { channel: OPEN_EXTERNAL_CHANNEL, args: ['https://example.com/docs'] },
    { channel: IMAGE_DATA_CHANNEL, args: [{ path: 'a.png', cwd: '/w', session_key: 'k' }] },
  ])
  // A host fault becomes a reason, never a rejection or a broken image.
  const failing = createBridge(recordingTransport(() => { throw new Error("Error invoking remote method 'desktop:image-data': Error: boom") }).transport)
  expect(await failing.imageData({ path: 'a.png' })).toEqual({ ok: false, error: 'boom' })
})

function panel(links = true) {
  const opened: string[] = []
  const ports: PanelPorts = {
    workspace, workspaceFolders: () => [workspace], remoteName: undefined,
    chooseFolder: async () => null, openFolder: async () => {}, openPanel: () => {}, revealPath: async () => true,
    backgroundData: async name => name, memory: { get: () => undefined, set: () => {} }, notify: () => {}, visible: () => true,
    openFile: async () => true, openDiff: async () => true, pickFiles: async () => [], showSourceControl: async () => true,
    setStyle: async () => true,
    ...(links ? { openExternal: async (url: string) => { opened.push(url); return true } } : {}),
  }
  const frames: Array<ResultFrame | PushFrame> = []
  const daemon: PanelDaemon = { call: async () => ({ ok: true }), onEvent: () => {}, offEvent: () => {}, dispose: () => {} }
  const host = new PanelHost(ports, frame => { frames.push(frame) }, () => daemon)
  let id = 0
  const invoke = async (channel: string, ...args: unknown[]) => {
    const frameId = ++id
    await host.receive({ kind: 'invoke', id: frameId, channel, args })
    const frame = frames.find((entry): entry is ResultFrame => entry.kind === 'result' && entry.id === frameId)
    if (!frame) throw new Error('no answer')
    return frame
  }
  return { invoke, opened }
}

test('the VS Code host opens links through env.openExternal and reads images from the workspace', async () => {
  const { invoke, opened } = panel()
  expect(await invoke(OPEN_EXTERNAL_CHANNEL, 'https://example.com/')).toMatchObject({ ok: true, value: true })
  expect(await invoke(OPEN_EXTERNAL_CHANNEL, 'javascript:alert(1)')).toMatchObject({ ok: false, error: 'Only http, https and mailto links can be opened' })
  expect(await invoke(OPEN_EXTERNAL_CHANNEL, 'command:workbench.action.terminal.new')).toMatchObject({ ok: false })
  expect(opened).toEqual(['https://example.com/'])
  expect(await invoke(IMAGE_DATA_CHANNEL, { path: 'out/shot.png' })).toMatchObject({ ok: true, value: { ok: true, url: `data:image/png;base64,${PNG_BASE64}` } })
  expect(await invoke(IMAGE_DATA_CHANNEL, { path: '../outside/secret.png' })).toMatchObject({ ok: true, value: { ok: false, error: 'a relative image path must stay inside the workspace' } })
  expect(await invoke(IMAGE_DATA_CHANNEL, { path: 'a.png', bogus: true })).toMatchObject({ ok: true, value: { ok: false, error: 'the image request is malformed' } })
  const old = panel(false)
  expect(await old.invoke(OPEN_EXTERNAL_CHANNEL, 'https://example.com/')).toMatchObject({ ok: false, error: 'This VS Code host cannot open links.' })
})
