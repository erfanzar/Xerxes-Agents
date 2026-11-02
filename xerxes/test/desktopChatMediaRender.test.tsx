// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { Markdown } from '../src/desktop/renderer/markdown.js'
import { ExecutionDetails, ToolCallRow } from '../src/desktop/renderer/Execution.js'
import { imageSourceOf, loadChatImage } from '../src/desktop/renderer/ImageView.js'
import { linkTargetOf, openExternalLink } from '../src/desktop/renderer/chatLinks.js'
import { toolImagesOf } from '../src/desktop/renderer/toolImages.js'
import type { ToolItem } from '../src/desktop/renderer/types.js'

const PNG = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAQCAAAAABSayKFAAAAMElEQVR42mOQU9E2tLBz9Q6KiE/NLqqob+2ZNHP+0jWbdu4/dubyzQfPGEYVjCQFAOTZBBBEiOEqAAAAAElFTkSuQmCC'
const render = (text: string): string => renderToStaticMarkup(createElement(Markdown, { text }))

const tool = (overrides: Partial<ToolItem>): ToolItem => ({
  id: 't1', verb: 'tool', arg: '', dur: '0.1s', state: 'done', name: 'computer_use', input: '{}', output: '', ...overrides,
})

test('a data-URL markdown image renders as a bounded, lazy, enlargeable image with its alt text', () => {
  const html = render(`Here is the chart:\n\n![Weekly chart](${PNG} "title")`)
  expect(html).toContain(`<img class="chatimg__img" src="${PNG}" alt="Weekly chart" loading="lazy"`)
  expect(html).toContain('aria-label="Enlarge image: Weekly chart"')
  expect(html).toContain('<button type="button" class="chatimg__open"')
})

test('a file-path image is never referenced by path: it waits for the host read', () => {
  for (const src of ['./out/shot.png', 'out/shot.png', '/tmp/shot.png', 'file:///tmp/shot.png']) {
    const html = render(`![shot](${src})`)
    expect(html).toContain('Loading image: shot')
    expect(html).not.toContain('<img')
    expect(html).not.toContain(`src="${src}"`)
  }
})

test('web images are not fetched: they show as a link', () => {
  const html = render('![logo](https://example.com/logo.png)')
  expect(html).not.toContain('<img')
  expect(html).toContain('Web image (not loaded)')
  expect(html).toContain('href="https://example.com/logo.png"')
})

test('refused images say why in one inline line', () => {
  expect(render('![x](data:image/png;base64,abc)')).toContain('Image unavailable: the image data is not valid base64')
  expect(render('![x](data:text/html;base64,PGI+)')).toContain('Image unavailable: not an image data URL')
  expect(render('![x](notes.txt)')).toContain('Image unavailable: only PNG, JPEG, GIF, WebP and SVG images can be shown')
  expect(render('![x](javascript:alert(1))')).toContain('Image unavailable')
  expect(render('![x](javascript:alert(1))')).not.toContain('<img')
})

test('links: web and mail links keep their URL as tooltip, file paths open in Files, others are inert', () => {
  const web = render('see [the docs](https://example.com/docs "Docs") and [mail](mailto:a@example.com)')
  expect(web).toContain('<a class="md__link" href="https://example.com/docs" title="https://example.com/docs" rel="noreferrer noopener">the docs</a>')
  expect(web).toContain('href="mailto:a@example.com"')
  const file = render('open [the runner](src/streaming/loop.ts:42)')
  expect(file).toContain('class="md__link md__link--file" role="link" tabindex="0" title="src/streaming/loop.ts:42">the runner</a>')
  expect(file).not.toContain('href=')
  const inert = render('[x](javascript:alert(1)) [y](#top) [z](ftp://host/file)')
  expect(inert).not.toContain('href=')
  expect(inert).toContain('md__link--inert')
})

test('bare URLs in prose become links, trailing punctuation stays prose, code is left alone', () => {
  const html = render('Read https://example.com/a_(b). Then (https://example.com/c), done.')
  expect(html).toContain('href="https://example.com/a_(b)"')
  expect(html).toContain('</a>. Then (')
  expect(html).toContain('href="https://example.com/c"')
  expect(html).toContain('</a>), done.')
  const code = render('run `curl https://example.com/x` now\n\n```\nhttps://example.com/y\n```')
  expect(code).not.toContain('md__link')
  expect(code).toContain('<code>curl https://example.com/x</code>')
  expect(render('bold **https://example.com/z** text')).toContain('<strong>https://example.com/z</strong>')
})

test('link and image targets are classified from text alone', () => {
  expect(linkTargetOf('https://example.com')).toEqual({ kind: 'external', url: 'https://example.com/' })
  expect(linkTargetOf('src/a.ts#L10')).toEqual({ kind: 'file', path: 'src/a.ts' })
  expect(linkTargetOf('/abs/My%20File.md:3:1')).toEqual({ kind: 'file', path: '/abs/My File.md' })
  expect(linkTargetOf('C:\\work\\a.ts')).toEqual({ kind: 'file', path: 'C:\\work\\a.ts' })
  expect(linkTargetOf('file:///etc/hosts')).toEqual({ kind: 'file', path: '/etc/hosts' })
  for (const inert of ['javascript:alert(1)', 'vbscript:x', '#anchor', '//evil.example', 'data:text/html,x', '']) expect(linkTargetOf(inert)).toEqual({ kind: 'inert' })
  expect(imageSourceOf(PNG)).toEqual({ kind: 'data', url: PNG })
  expect(imageSourceOf('https://example.com/a.png')).toEqual({ kind: 'web', url: 'https://example.com/a.png' })
  expect(imageSourceOf('out/a.webp')).toEqual({ kind: 'file', path: 'out/a.webp' })
  expect(imageSourceOf('ftp://x/a.png')).toEqual({ kind: 'refused', error: 'only file paths, data URLs and web links can be images' })
})

test('opening a link goes through the host and refuses other schemes before it', async () => {
  const opened: string[] = []
  await openExternalLink('https://example.com/x', { openExternal: async url => { opened.push(url); return true } })
  expect(opened).toEqual(['https://example.com/x'])
  await expect(openExternalLink('javascript:alert(1)', { openExternal: async () => true })).rejects.toThrow('Only http, https and mailto')
  await expect(openExternalLink('https://example.com/', { openExternal: async () => false })).rejects.toThrow('could not open')
})

test('image reads go through the bridge, never reject, and cache only successes', async () => {
  let reads = 0
  const bridge = { imageData: async () => { reads += 1; return reads === 1 ? { ok: false as const, error: 'the file does not exist' } : { ok: true as const, url: PNG } } }
  expect(await loadChatImage('pending.png', { cwd: '/w' }, bridge)).toEqual({ ok: false, error: 'the file does not exist' })
  expect(await loadChatImage('pending.png', { cwd: '/w' }, bridge)).toEqual({ ok: true, url: PNG })
  expect(await loadChatImage('pending.png', { cwd: '/w' }, bridge)).toEqual({ ok: true, url: PNG })
  expect(reads).toBe(2)
  expect(await loadChatImage('x.png', {}, { imageData: async () => { throw new Error('boom') } })).toEqual({ ok: false, error: 'boom' })
  expect(await loadChatImage('x.png', {}, {})).toEqual({ ok: false, error: 'this version of the app cannot read image files' })
})

test('tool steps carry their images: screenshots, MCP image blocks, saved files and image reads', () => {
  const screenshot = tool({ output: JSON.stringify({ _multimodal: true, content: [{ type: 'text', text: 'Screen capture: 32 x 16' }, { type: 'image_url', image_url: { url: PNG } }], text_summary: 'Screen capture: 32 x 16' }) })
  expect(toolImagesOf(screenshot)).toEqual([{ src: PNG, alt: 'Image from computer_use' }])
  const mcp = tool({ name: 'mcp.render', output: JSON.stringify({ content: [{ type: 'image', data: PNG.slice(22), mime_type: 'image/png' }] }) })
  expect(toolImagesOf(mcp)).toEqual([{ src: PNG, alt: 'Image from render' }])
  const generated = tool({ name: 'generate_image', output: JSON.stringify({ saved: [{ path: 'out/cat.png', media_type: 'image/png', bytes: 10 }] }) })
  expect(toolImagesOf(generated)).toEqual([{ src: 'out/cat.png', alt: 'cat.png' }])
  const read = tool({ name: 'ReadFile', input: JSON.stringify({ file_path: 'docs/diagram.svg' }), output: '<svg/>' })
  expect(toolImagesOf(read)).toEqual([{ src: 'docs/diagram.svg', alt: 'diagram.svg' }])
  expect(toolImagesOf(tool({ name: 'ReadFile', input: JSON.stringify({ file_path: 'a.ts' }), output: 'code' }))).toEqual([])
  expect(toolImagesOf(tool({ output: 'plain text, no picture' }))).toEqual([])
})

test('the opened tool step shows the pictures it carries, and a failed step does not', () => {
  const screenshot = tool({ output: JSON.stringify({ _multimodal: true, content: [{ type: 'image_url', image_url: { url: PNG } }], text_summary: 'capture' }) })
  const html = renderToStaticMarkup(createElement(ExecutionDetails, { item: screenshot }))
  expect(html).toContain('class="toolimages"')
  expect(html).toContain(`<img class="chatimg__img" src="${PNG}" alt="Image from computer_use"`)
  const failed = renderToStaticMarkup(createElement(ExecutionDetails, { item: { ...screenshot, state: 'failed', error: 'denied' } }))
  expect(failed).not.toContain('toolimages')
  // A read of an image file is a generic step (its picture, on open), not text lines.
  const row = renderToStaticMarkup(createElement(ToolCallRow, { item: tool({ name: 'ReadFile', input: JSON.stringify({ file_path: 'a.png' }), output: '' }), label: 'Read' }))
  expect(row).toContain('execution-row')
  expect(row).not.toContain('filetool')
})
