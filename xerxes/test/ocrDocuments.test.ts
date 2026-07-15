// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  assertMarkerDiskSpace,
  bunOcrDocumentFilesystem,
  checkMarkerDiskSpace,
  extractMarkerDocument,
  formatMarkerDiskSpaceCheck,
  formatMarkerExtraction,
  formatPdfMetadata,
  formatPdfTableExtraction,
  formatPdfTextExtraction,
  MARKER_MINIMUM_FREE_BYTES,
  MarkerDiskSpaceError,
  MarkerExtractionError,
  parsePdfPageSelection,
  PdfDocumentExtractor,
  PdfExtractionError,
  type OcrConversionPort,
  type OcrDocumentFilesystemPort,
  type PdfTextConversionPort,
} from '../src/skills/ocrDocuments/index.js'

test('PDF extraction preserves zero-based page selection, structured pages, and PyMuPDF-style text output', async () => {
  const { converter, markdownCalls, state } = createPdfConverter()
  const extractor = new PdfDocumentExtractor(converter)

  const text = await extractor.extractText(' report.pdf ', [0, 9, 1])
  expect(text).toEqual({
    metadata: {
      author: '', creator: 'PDF.js', format: 'PDF 1.7', pages: 2,
      producer: '', subject: '', title: 'Quarterly report',
    },
    pages: [
      { pageIndex: 0, pageNumber: 1, text: 'Revenue rose.' },
      { pageIndex: 1, pageNumber: 2, text: 'Expenses fell.' },
    ],
  })
  expect(formatPdfTextExtraction(text)).toBe(
    '\n--- Page 1/2 ---\n\nRevenue rose.\n\n--- Page 2/2 ---\n\nExpenses fell.',
  )
  expect(formatPdfMetadata(text.metadata)).toBe(JSON.stringify(text.metadata, null, 2))
  expect(state.closed).toBe(1)

  expect(parsePdfPageSelection('0-2')).toEqual([0, 1, 2])
  expect(parsePdfPageSelection('4')).toEqual([4])
  expect(() => parsePdfPageSelection('4-1')).toThrow('ascending')
  expect(() => parsePdfPageSelection('-1')).toThrow('zero-based')

  expect(await extractor.extractMarkdown('report.pdf', [1])).toBe('# Page two')
  expect(markdownCalls).toEqual([{ path: 'report.pdf', pages: [1] }])
})

test('PDF table and image modes retain headings, GitHub markdown, deterministic image names, and resource cleanup', async () => {
  const { converter, state } = createPdfConverter()
  const extractor = new PdfDocumentExtractor(converter)
  const filesystem = createFilesystem()

  const tables = await extractor.extractTables('tables.pdf')
  expect(tables.tables).toHaveLength(1)
  expect(formatPdfTableExtraction(tables)).toBe(
    '\n--- Page 1, Table 1 ---\n\n| Item | Notes |\n| --- | --- |\n| A\\|B | first<br>second |',
  )
  expect(state.closed).toBe(1)

  const images = await extractor.extractImages('images.pdf', 'saved-images', filesystem.port)
  expect(images).toEqual({
    images: [
      { pageIndex: 0, pageNumber: 1, path: 'saved-images/page1_img1.png' },
      { pageIndex: 1, pageNumber: 2, path: 'saved-images/page2_img1.png' },
    ],
    metadata: {
      author: '', creator: 'PDF.js', format: 'PDF 1.7', pages: 2,
      producer: '', subject: '', title: 'Quarterly report',
    },
    outputDirectory: 'saved-images',
  })
  expect(filesystem.directories).toEqual(['saved-images'])
  expect([...filesystem.files.entries()]).toEqual([
    ['saved-images/page1_img1.png', [1, 2]],
    ['saved-images/page2_img1.png', [3]],
  ])
  expect(state.closed).toBe(2)
})

test('PDF capability and malformed adapter failures stay explicit instead of falling back to Python', async () => {
  const unsupported = new PdfDocumentExtractor({
    open: async () => ({ metadata: {}, pageCount: 1, page: async () => ({ extractText: async () => 'text' }) }),
  })
  await expect(unsupported.extractMarkdown('report.pdf')).rejects.toThrow('does not support markdown')
  await expect(unsupported.extractTables('report.pdf')).rejects.toThrow('does not support table')
  await expect(unsupported.extractImages('report.pdf', 'images', createFilesystem().port)).rejects.toThrow(
    'does not support embedded-image',
  )

  const invalid = new PdfDocumentExtractor({
    open: async () => ({ metadata: {}, pageCount: -1, page: async () => ({ extractText: async () => 'text' }) }),
  })
  await expect(invalid.metadata('report.pdf')).rejects.toBeInstanceOf(PdfExtractionError)
})

test('marker-style OCR passes LLM intent into an injected converter and preserves markdown, metadata, pages, and images', async () => {
  const requests: unknown[] = []
  const converter: OcrConversionPort = {
    async convert(request) {
      requests.push(request)
      return {
        markdown: '# Scanned report\n\nRecovered text.',
        metadata: { language: 'en', confidence: 0.98, title: 'Scanned report' },
        pages: [{ markdown: '# Page 1', metadata: { confidence: 0.98 }, pageNumber: 1, text: 'Recovered text.' }],
        images: { 'page-1.png': new Uint8Array([7, 8, 9]) },
      }
    },
  }
  const filesystem = createFilesystem()

  const extraction = await extractMarkerDocument(' scan.pdf ', converter, {
    outputDirectory: 'ocr-images', outputFormat: 'json', useLlm: true,
  }, filesystem.port)
  expect(requests).toEqual([{ path: 'scan.pdf', useLlm: true }])
  expect(extraction).toEqual({
    images: [{ name: 'page-1.png', path: 'ocr-images/page-1.png' }],
    markdown: '# Scanned report\n\nRecovered text.',
    metadata: { language: 'en', confidence: 0.98, title: 'Scanned report' },
    pages: [{ markdown: '# Page 1', metadata: { confidence: 0.98 }, pageNumber: 1, text: 'Recovered text.' }],
  })
  expect(formatMarkerExtraction(extraction)).toBe('# Scanned report\n\nRecovered text.')
  expect(formatMarkerExtraction(extraction, 'json')).toBe(JSON.stringify({
    markdown: '# Scanned report\n\nRecovered text.',
    metadata: { language: 'en', confidence: 0.98, title: 'Scanned report' },
  }, null, 2))
  expect(filesystem.directories).toEqual(['ocr-images'])
  expect(filesystem.files.get('ocr-images/page-1.png')).toEqual([7, 8, 9])
})

test('marker disk requirements and image safety remain actionable through injected storage and filesystem ports', async () => {
  const passing = await checkMarkerDiskSpace({ freeBytes: async path => {
    expect(path).toBe('/workspace')
    return MARKER_MINIMUM_FREE_BYTES
  } }, '/workspace')
  expect(passing.meetsRequirement).toBeTrue()
  expect(formatMarkerDiskSpaceCheck(passing)).toBe('✓ 5.0GB free — sufficient for marker-pdf')

  await expect(assertMarkerDiskSpace({ freeBytes: async () => MARKER_MINIMUM_FREE_BYTES - 1 }, '/workspace'))
    .rejects.toBeInstanceOf(MarkerDiskSpaceError)

  const filesystem = createFilesystem()
  const unsafeConverter: OcrConversionPort = {
    async convert() {
      return { markdown: 'content', images: { '../escape.png': new Uint8Array([1]) } }
    },
  }
  await expect(extractMarkerDocument('scan.pdf', unsafeConverter, { outputDirectory: 'safe' }, filesystem.port))
    .rejects.toBeInstanceOf(MarkerExtractionError)
  await expect(extractMarkerDocument('scan.pdf', unsafeConverter, { outputDirectory: 'safe' }))
    .rejects.toThrow('filesystem adapter')
})

test('the opt-in Bun filesystem adapter writes image output without a subprocess', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-ocr-documents-'))
  try {
    const nested = bunOcrDocumentFilesystem.join(directory, 'nested')
    await bunOcrDocumentFilesystem.ensureDirectory(nested)
    const path = bunOcrDocumentFilesystem.join(nested, 'image.png')
    await bunOcrDocumentFilesystem.writeFile(path, new Uint8Array([4, 5]))
    expect([...await Bun.file(path).bytes()]).toEqual([4, 5])
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

interface PdfConverterState {
  closed: number
}

function createPdfConverter(): {
  readonly converter: PdfTextConversionPort
  readonly markdownCalls: { readonly path: string; readonly pages: readonly number[] | undefined }[]
  readonly state: PdfConverterState
} {
  const markdownCalls: { path: string; pages: readonly number[] | undefined }[] = []
  const state: PdfConverterState = { closed: 0 }
  return {
    converter: {
      async open() {
        return {
          metadata: { author: null, creator: 'PDF.js', format: 'PDF 1.7', title: 'Quarterly report' },
          pageCount: 2,
          async close() {
            state.closed += 1
          },
          async page(index) {
            if (index === 0) {
              return {
                async extractImages() { return [{ bytes: new Uint8Array([1, 2]) }] },
                async extractTables() {
                  return [{ headers: ['Item', 'Notes'], rows: [['A|B', 'first\nsecond']] }]
                },
                async extractText() { return 'Revenue rose.' },
              }
            }
            return {
              async extractImages() { return [{ bytes: new Uint8Array([3]) }] },
              async extractTables() { return [] },
              async extractText() { return 'Expenses fell.' },
            }
          },
        }
      },
      async toMarkdown(path, pages) {
        markdownCalls.push({ path, pages })
        return '# Page two'
      },
    },
    markdownCalls,
    state,
  }
}

function createFilesystem(): {
  readonly directories: string[]
  readonly files: Map<string, number[]>
  readonly port: OcrDocumentFilesystemPort
} {
  const directories: string[] = []
  const files = new Map<string, number[]>()
  return {
    directories,
    files,
    port: {
      async ensureDirectory(path) {
        directories.push(path)
      },
      join(directory, name) {
        return `${directory}/${name}`
      },
      async writeFile(path, bytes) {
        files.set(path, [...bytes])
      },
    },
  }
}
