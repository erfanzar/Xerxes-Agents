// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import { createOfficeZip } from '../src/skills/powerpoint/zip.js'
import { readFile } from '../src/tools/fileTools.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'

const encode = (text: string): Uint8Array => new TextEncoder().encode(text)

function docx(bodyXml: string): Uint8Array {
  return createOfficeZip([
    { name: '[Content_Types].xml', data: encode('<Types/>') },
    { name: 'word/document.xml', data: encode(`<w:document><w:body>${bodyXml}</w:body></w:document>`) },
  ])
}

async function inWorkspace(run: (workspace: string, paths: WorkspacePathResolver) => Promise<void>): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-office-text-'))
  try {
    await run(workspace, new WorkspacePathResolver(workspace))
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
}

test('ReadFile returns the paragraphs and table cells of a .docx, without field codes or deleted text', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'memo.docx'), docx([
      '<w:p><w:r><w:t>Quarterly &amp; annual</w:t></w:r><w:r><w:tab/><w:t xml:space="preserve"> review</w:t></w:r></w:p>',
      '<w:p><w:r><w:instrText>PAGE \\* MERGEFORMAT</w:instrText></w:r><w:r><w:delText>removed</w:delText></w:r><w:r><w:t>Kept</w:t></w:r></w:p>',
      '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Region</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Revenue</w:t></w:r></w:p></w:tc></w:tr></w:tbl>',
    ].join('')))
    const text = await readFile({ file_path: 'memo.docx' }, paths)
    expect(text).toContain('[ReadFile] Text extracted from a .docx document')
    expect(text).toContain('Quarterly & annual\t review')
    expect(text).toContain('Kept')
    expect(text).not.toContain('MERGEFORMAT')
    expect(text).not.toContain('removed')
    expect(text).toContain('Region')
    expect(text).toContain('Revenue')
  })
})

test('ReadFile returns .pptx slide text in slide order', async () => {
  await inWorkspace(async (workspace, paths) => {
    const slide = (text: string) => encode(`<p:sld><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>${text}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>`)
    await Bun.write(join(workspace, 'deck.pptx'), createOfficeZip([
      { name: 'ppt/slides/slide10.xml', data: slide('Ten') },
      { name: 'ppt/slides/slide2.xml', data: slide('Two') },
      { name: 'ppt/slides/slide1.xml', data: slide('One') },
    ]))
    const text = await readFile({ file_path: 'deck.pptx', limit: -1 }, paths)
    expect(text.indexOf('--- Slide 1 ---\nOne')).toBeGreaterThan(-1)
    expect(text.indexOf('Slide 1')).toBeLessThan(text.indexOf('Slide 2'))
    expect(text.indexOf('Slide 2')).toBeLessThan(text.indexOf('Slide 10'))
  })
})

test('ReadFile refuses binary files with a hint instead of returning noise', async () => {
  await inWorkspace(async (workspace, paths) => {
    await Bun.write(join(workspace, 'report.pdf'), new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x00, 0x01, 0x02]))
    await Bun.write(join(workspace, 'book.xlsx'), new Uint8Array([0x50, 0x4b, 0x03, 0x04, 0x00, 0x00]))
    await expect(readFile({ file_path: 'report.pdf' }, paths)).rejects.toThrow(/is a PDF/)
    await expect(readFile({ file_path: 'book.xlsx' }, paths)).rejects.toThrow(/XlsxReader/)
    await expect(readFile({ file_path: 'book.xlsx' }, paths)).rejects.toBeInstanceOf(ValidationError)
    // A broken .docx names the problem rather than throwing a zip internals error.
    await Bun.write(join(workspace, 'broken.docx'), encode('not a zip'))
    await expect(readFile({ file_path: 'broken.docx' }, paths)).rejects.toThrow(/is not a readable \.docx document/)
    // Text, including non-ASCII text, still reads verbatim.
    await Bun.write(join(workspace, 'notes.txt'), 'naïve café\nline two\n')
    expect(await readFile({ file_path: 'notes.txt' }, paths)).toBe('naïve café\nline two\n')
  })
})
