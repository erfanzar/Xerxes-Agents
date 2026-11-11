// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import {
  FileActions, PREVIEW_KINDS, previewReplyOf, renderedViewOf, requestKinds, requestPreview, RichFilePreview, shownNote,
  type RichPreview,
} from '../src/desktop/renderer/FilePreview.js'

const render = (preview: RichPreview, requestedSheet = 0): string =>
  renderToStaticMarkup(createElement(RichFilePreview, { preview, requestedSheet, chooseSheet: () => {}, actions: createElement('button', null, 'Open in default app') }))

test('every request names the kinds the panel draws; Source view leaves the rendered kind out', async () => {
  expect(requestKinds('out/discrepancies.csv', false)).toEqual([...PREVIEW_KINDS])
  expect(requestKinds('out/discrepancies.csv', true)).not.toContain('table')
  expect(requestKinds('logo.svg', true)).not.toContain('image')
  expect(requestKinds('page.html', true)).not.toContain('html')
  // Markdown renders from its text, so its Source view changes nothing on the wire.
  expect(requestKinds('README.md', true)).toEqual([...PREVIEW_KINDS])
  expect(renderedViewOf('a.tsv')).toBe('table')
  expect(renderedViewOf('a.xlsx')).toBeUndefined()
  const calls: unknown[] = []
  const reply = await requestPreview({ call: async (method, params) => { calls.push([method, params]); return { ok: true, kind: 'table', header: ['a'], rows: [['1']], total_rows: 1, total_columns: 1 } } }, 'key-1', 'b.xlsx', { source: false, sheet: 2 })
  expect(calls).toEqual([['workspace.filePreview', { path: 'b.xlsx', kinds: [...PREVIEW_KINDS], sheet: 2, session_key: 'key-1' }]])
  expect(reply.rich).toMatchObject({ kind: 'table', header: ['a'] })
})

test('replies are read defensively: text from older runtimes, typed unavailability, and real failures', () => {
  expect(previewReplyOf('a.ts', { ok: true, path: 'a.ts', content: 'x', truncated: false, version: 'v' })).toEqual({ file: { path: 'a.ts', content: 'x', version: 'v', truncated: false } })
  expect(previewReplyOf('p.html', { ok: true, kind: 'html', content: '<b>x</b>', truncated: false })).toMatchObject({ file: { content: '<b>x</b>', version: null }, rich: { kind: 'html', content: '<b>x</b>' } })
  expect(previewReplyOf('big.pdf', { ok: false, reason: 'too_large', error: 'This file is 48 MB.' })).toEqual({ rich: { kind: 'unavailable', path: 'big.pdf', reason: 'too_large', message: 'This file is 48 MB.' } })
  expect(() => previewReplyOf('x', { ok: false, error: 'File preview is limited to the selected workspace' })).toThrow('selected workspace')
  expect(() => previewReplyOf('x.png', { ok: true, kind: 'image', media_type: 'text/html', data: '' })).toThrow('image type')
  expect(previewReplyOf('x.new', { ok: true, kind: 'hologram' }).rich).toMatchObject({ kind: 'unavailable', message: expect.stringContaining('Update the app') })
  // Malformed cells become text; nothing is executed or trusted as markup.
  expect(previewReplyOf('a.csv', { ok: true, kind: 'table', header: ['h', 2], rows: [[null, '<b>']], total_rows: -4 }).rich).toMatchObject({ header: ['h', '2'], rows: [['', '<b>']], totalRows: 0 })
})

test('a capped table says how much it shows', () => {
  expect(shownNote(1000, 52340, 3, 3, true)).toBe('Showing 1,000 of 52,340 rows')
  expect(shownNote(5, 5, 100, 230, true)).toBe('Showing the first 100 of 230 columns')
  expect(shownNote(1000, 2000, 100, 230, true)).toBe('Showing 1,000 of 2,000 rows, first 100 of 230 columns')
  expect(shownNote(2, 2, 2, 2, true)).toBe('Long cells are shortened')
  expect(shownNote(2, 2, 2, 2, false)).toBe('')
  const html = render({ kind: 'table', path: 'out/discrepancies.csv', header: ['id', 'amount'], rows: [['1', '<script>x</script>']], totalRows: 4000, totalColumns: 2, truncated: true, encoding: 'windows-1252' })
  expect(html).toContain('Showing 1 of 4,000 rows. Read as Windows-1252 text')
  expect(html).toContain('<th scope="col">amount</th>')
  expect(html).toContain('&lt;script&gt;x&lt;/script&gt;')
  expect(html).toContain('aria-label="Table: discrepancies.csv"')
})

test('a workbook shows sheet tabs, column letters and row numbers; a chart sheet explains itself', () => {
  const sheet: RichPreview = { kind: 'sheet', path: 'book.xlsx', sheets: [{ name: 'Data', hidden: false }, { name: 'Secret', hidden: true }], sheet: 0, rows: [['Item', '12.50%'], ['', '']], totalRows: 2, totalColumns: 2, truncated: false, notice: '' }
  const html = render(sheet)
  expect(html).toContain('role="tablist"')
  expect(html).toContain('aria-selected="true"')
  expect(html).toContain('title="Secret (hidden sheet)"')
  expect(html).toContain('<th scope="col">A</th><th scope="col">B</th>')
  expect(html).toContain('<th class="file-grid__index" scope="row">2</th>')
  expect(html).toContain('class="is-number">12.50%')
  expect(render(sheet, 1)).toContain('Loading Secret…')
  expect(render({ ...sheet, rows: [], notice: 'This sheet is a chart. Open the workbook to see it.' })).toContain('This sheet is a chart')
})

test('a Word document draws headings, emphasis, lists with their markers and tables', () => {
  const html = render({ kind: 'document', path: 'report.docx', truncated: true, blocks: [
    { type: 'heading', level: 2, text: 'Findings' },
    { type: 'paragraph', runs: [{ text: 'Revenue grew ' }, { text: '12%', bold: true }, { text: ' fast', italic: true, underline: true }] },
    { type: 'list_item', ordered: true, level: 0, marker: '1.', runs: [{ text: 'Step one' }] },
    { type: 'list_item', ordered: true, level: 1, marker: '1.a)', runs: [{ text: 'Sub step' }] },
    { type: 'paragraph', runs: [{ text: 'After' }] },
    { type: 'table', rows: [['Region', 'Revenue'], ['EMEA', '1,200']] },
  ] })
  expect(html).toContain('only its beginning is shown')
  expect(html).toContain('<h2>Findings</h2>')
  expect(html).toContain('<strong>12%</strong>')
  expect(html).toContain('<em><u> fast</u></em>')
  // Consecutive items are one list; the paragraph after ends it.
  expect(html.match(/<ul class="file-docx__list">/g)?.length).toBe(1)
  expect(html).toContain('>1.a)</span>')
  expect(html).toContain('<td>EMEA</td>')
})

test('HTML renders in an empty sandbox (no scripts), images as data URLs, and unavailable files offer the actions', () => {
  const page = render({ kind: 'html', path: 'page.html', content: '<h1>Hi</h1><script>alert(1)</script>' })
  expect(page).toContain('sandbox=""')
  expect(page).toContain('srcDoc="&lt;h1&gt;Hi&lt;/h1&gt;&lt;script&gt;alert(1)&lt;/script&gt;"')
  expect(page).not.toContain('allow-scripts')
  expect(page).not.toContain('<script>')
  const image = render({ kind: 'image', path: 'logo.svg', mediaType: 'image/svg+xml', data: 'PHN2Zy8+' })
  expect(image).toContain('src="data:image/svg+xml;base64,PHN2Zy8+"')
  expect(image).toContain('alt="logo.svg"')
  const unavailable = render({ kind: 'unavailable', path: 'big.pdf', reason: 'too_large', message: 'This file is 48 MB. Previews of PDFs are limited to 10 MB.' })
  expect(unavailable).toContain('limited to 10 MB')
  expect(unavailable).toContain('Open in default app')
})

test('Open and Show in Finder are offered on every file, and disabled with a reason on an SSH workspace', () => {
  const local = renderToStaticMarkup(createElement(FileActions, { path: './out/a.xlsx', cwd: '/work', remote: false }))
  expect(local).toContain('Open in default app')
  expect(local).toMatch(/Show in (Finder|folder)/)
  const remote = renderToStaticMarkup(createElement(FileActions, { path: './out/a.xlsx', cwd: '/work', remote: true }))
  expect(remote).toContain('disabled=""')
  expect(remote).toContain('on the SSH host')
})
