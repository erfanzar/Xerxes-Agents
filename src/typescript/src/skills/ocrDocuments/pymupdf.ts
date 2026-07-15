// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type {
  OcrDocumentFilesystemPort,
  PdfEmbeddedImage,
  PdfExtractedTable,
  PdfSourceMetadata,
  PdfTextConversionPort,
  PdfTextDocumentPort,
  PdfTextPagePort,
} from './types.js'

/** Failure raised when an injected PDF conversion engine cannot satisfy a requested operation. */
export class PdfExtractionError extends Error {
  constructor(message: string, options: { readonly cause?: unknown } = {}) {
    super(message, options)
    this.name = 'PdfExtractionError'
  }
}

/** Stable metadata shape printed by the former PDF-helper metadata mode. */
export interface PdfDocumentMetadata {
  readonly author: string
  readonly creator: string
  readonly format: string
  readonly pages: number
  readonly producer: string
  readonly subject: string
  readonly title: string
}

/** A text page with both zero-based source index and one-based display number. */
export interface ExtractedPdfTextPage {
  readonly pageIndex: number
  readonly pageNumber: number
  readonly text: string
}

/** Structured result for plain-text PDF extraction. */
export interface PdfTextExtraction {
  readonly metadata: PdfDocumentMetadata
  readonly pages: readonly ExtractedPdfTextPage[]
}

/** A detected table together with its source page and one-based table number. */
export interface ExtractedPdfTable {
  readonly pageIndex: number
  readonly pageNumber: number
  readonly table: PdfExtractedTable
  readonly tableNumber: number
}

/** Structured result for table extraction. */
export interface PdfTableExtraction {
  readonly metadata: PdfDocumentMetadata
  readonly tables: readonly ExtractedPdfTable[]
}

/** An image persisted by a caller-provided filesystem adapter. */
export interface WrittenPdfImage {
  readonly pageIndex: number
  readonly pageNumber: number
  readonly path: string
}

/** Structured result for embedded-image extraction. */
export interface PdfImageExtraction {
  readonly images: readonly WrittenPdfImage[]
  readonly metadata: PdfDocumentMetadata
  readonly outputDirectory: string
}

/** Parse the zero-based page selector accepted by the former helper script. */
export function parsePdfPageSelection(value: string): readonly number[] {
  const text = requirePdfPath(value, 'page selection')
  const range = /^(\d+)\s*-\s*(\d+)$/.exec(text)
  if (range) {
    const start = Number(range[1])
    const end = Number(range[2])
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || end < start) {
      throw new RangeError('page selection range must contain ascending non-negative safe integers')
    }
    const size = end - start + 1
    if (!Number.isSafeInteger(size) || size > 100_000) {
      throw new RangeError('page selection range is too large')
    }
    return Array.from({ length: size }, (_value, offset) => start + offset)
  }
  if (!/^\d+$/.test(text)) {
    throw new TypeError('page selection must be a zero-based page number or inclusive range such as "0-4"')
  }
  const page = Number(text)
  if (!Number.isSafeInteger(page)) {
    throw new RangeError('page selection must be a non-negative safe integer')
  }
  return [page]
}

/** Render the text-mode output printed by the former PDF helper. */
export function formatPdfTextExtraction(extraction: PdfTextExtraction): string {
  return extraction.pages
    .map(page => `\n--- Page ${page.pageNumber}/${extraction.metadata.pages} ---\n\n${page.text}`)
    .join('\n')
}

/** Render a table in GitHub-flavoured markdown without bringing in pandas. */
export function formatPdfTable(table: PdfExtractedTable): string {
  if (!table.headers.length) {
    throw new PdfExtractionError('extracted table must have at least one header column')
  }
  const headers = table.headers.map((header, index) => normalizeTableCell(header, `header ${index + 1}`))
  const width = headers.length
  const lines = [
    `| ${headers.join(' | ')} |`,
    `| ${headers.map(() => '---').join(' | ')} |`,
  ]
  for (const [index, row] of table.rows.entries()) {
    if (row.length !== width) {
      throw new PdfExtractionError(`extracted table row ${index + 1} has ${row.length} cells; expected ${width}`)
    }
    lines.push(`| ${row.map((cell, cellIndex) => normalizeTableCell(cell, `row ${index + 1}, column ${cellIndex + 1}`)).join(' | ')} |`)
  }
  return lines.join('\n')
}

/** Render every table in the user-facing shape of the original helper. */
export function formatPdfTableExtraction(extraction: PdfTableExtraction): string {
  return extraction.tables
    .map(table => `\n--- Page ${table.pageNumber}, Table ${table.tableNumber} ---\n\n${formatPdfTable(table.table)}`)
    .join('\n')
}

/** Render metadata as the indented JSON output of `--metadata`. */
export function formatPdfMetadata(metadata: PdfDocumentMetadata): string {
  return JSON.stringify(metadata, null, 2)
}

/**
 * Native orchestration for the former PyMuPDF helper. A host injects a PDF.js,
 * WASM, or other JavaScript implementation; no Python process is ever used.
 */
export class PdfDocumentExtractor {
  private readonly converter: PdfTextConversionPort

  constructor(converter: PdfTextConversionPort) {
    this.converter = converter
  }

  async extractText(path: string, pages?: readonly number[]): Promise<PdfTextExtraction> {
    return this.withDocument(path, async document => {
      const metadata = normalizePdfMetadata(document.pageCount, document.metadata)
      const selectedPages = selectPdfPages(document.pageCount, pages)
      const output: ExtractedPdfTextPage[] = []
      for (const pageIndex of selectedPages) {
        const page = await loadPdfPage(document, pageIndex)
        const text = await page.extractText()
        if (typeof text !== 'string') {
          throw new PdfExtractionError(`PDF text converter returned non-text content for page ${pageIndex + 1}`)
        }
        output.push({ pageIndex, pageNumber: pageIndex + 1, text })
      }
      return { metadata, pages: output }
    })
  }

  async extractMarkdown(path: string, pages?: readonly number[]): Promise<string> {
    const normalizedPath = requirePdfPath(path, 'path')
    if (!this.converter.toMarkdown) {
      throw new PdfExtractionError('PDF text converter does not support markdown extraction')
    }
    validatePageIndexes(pages)
    const markdown = await this.converter.toMarkdown(normalizedPath, pages)
    if (typeof markdown !== 'string') {
      throw new PdfExtractionError('PDF text converter returned non-text markdown output')
    }
    return markdown
  }

  async extractTables(path: string): Promise<PdfTableExtraction> {
    return this.withDocument(path, async document => {
      const metadata = normalizePdfMetadata(document.pageCount, document.metadata)
      const tables: ExtractedPdfTable[] = []
      for (let pageIndex = 0; pageIndex < document.pageCount; pageIndex += 1) {
        const page = await loadPdfPage(document, pageIndex)
        if (!page.extractTables) {
          throw new PdfExtractionError('PDF text converter does not support table extraction')
        }
        const pageTables = await page.extractTables()
        if (!Array.isArray(pageTables)) {
          throw new PdfExtractionError(`PDF text converter returned invalid tables for page ${pageIndex + 1}`)
        }
        for (const [tableIndex, table] of pageTables.entries()) {
          tables.push({ pageIndex, pageNumber: pageIndex + 1, table, tableNumber: tableIndex + 1 })
        }
      }
      return { metadata, tables }
    })
  }

  async extractImages(
    path: string,
    outputDirectory: string,
    filesystem: OcrDocumentFilesystemPort,
  ): Promise<PdfImageExtraction> {
    const normalizedDirectory = requirePdfPath(outputDirectory, 'outputDirectory')
    return this.withDocument(path, async document => {
      const metadata = normalizePdfMetadata(document.pageCount, document.metadata)
      await filesystem.ensureDirectory(normalizedDirectory)
      const images: WrittenPdfImage[] = []
      for (let pageIndex = 0; pageIndex < document.pageCount; pageIndex += 1) {
        const page = await loadPdfPage(document, pageIndex)
        if (!page.extractImages) {
          throw new PdfExtractionError('PDF text converter does not support embedded-image extraction')
        }
        const pageImages = await page.extractImages()
        if (!Array.isArray(pageImages)) {
          throw new PdfExtractionError(`PDF text converter returned invalid images for page ${pageIndex + 1}`)
        }
        for (const [imageIndex, image] of pageImages.entries()) {
          validateEmbeddedImage(image, pageIndex, imageIndex)
          const imagePath = filesystem.join(normalizedDirectory, `page${pageIndex + 1}_img${imageIndex + 1}.png`)
          await filesystem.writeFile(imagePath, image.bytes)
          images.push({ pageIndex, pageNumber: pageIndex + 1, path: imagePath })
        }
      }
      return { images, metadata, outputDirectory: normalizedDirectory }
    })
  }

  async metadata(path: string): Promise<PdfDocumentMetadata> {
    return this.withDocument(path, document => normalizePdfMetadata(document.pageCount, document.metadata))
  }

  private async withDocument<T>(path: string, operation: (document: PdfTextDocumentPort) => Promise<T> | T): Promise<T> {
    const document = await this.converter.open(requirePdfPath(path, 'path'))
    validatePdfDocument(document)
    try {
      return await operation(document)
    } finally {
      await document.close?.()
    }
  }
}

function requirePdfPath(value: string, name: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError(`${name} must be a non-empty string`)
  }
  return value.trim()
}

function normalizePdfMetadata(pageCount: number, metadata: PdfSourceMetadata): PdfDocumentMetadata {
  return {
    author: normalizeMetadataText(metadata.author, 'author'),
    creator: normalizeMetadataText(metadata.creator, 'creator'),
    format: normalizeMetadataText(metadata.format, 'format'),
    pages: pageCount,
    producer: normalizeMetadataText(metadata.producer, 'producer'),
    subject: normalizeMetadataText(metadata.subject, 'subject'),
    title: normalizeMetadataText(metadata.title, 'title'),
  }
}

function normalizeMetadataText(value: unknown, name: string): string {
  if (value === undefined || value === null) return ''
  if (typeof value !== 'string') {
    throw new PdfExtractionError(`PDF text converter returned non-text ${name} metadata`)
  }
  return value
}

function validatePdfDocument(document: PdfTextDocumentPort): void {
  if (!document || typeof document !== 'object') {
    throw new PdfExtractionError('PDF text converter returned no document')
  }
  if (!Number.isSafeInteger(document.pageCount) || document.pageCount < 0) {
    throw new PdfExtractionError('PDF text converter returned an invalid page count')
  }
  if (!document.metadata || typeof document.metadata !== 'object') {
    throw new PdfExtractionError('PDF text converter returned invalid metadata')
  }
}

function validatePageIndexes(pages: readonly number[] | undefined): void {
  if (pages === undefined) return
  for (const page of pages) {
    if (!Number.isSafeInteger(page) || page < 0) {
      throw new RangeError('page indexes must be non-negative safe integers')
    }
  }
}

function selectPdfPages(pageCount: number, pages: readonly number[] | undefined): readonly number[] {
  validatePageIndexes(pages)
  if (pages === undefined) return Array.from({ length: pageCount }, (_value, index) => index)
  // The Python helper silently skipped page indexes at or beyond the document length.
  return pages.filter(page => page < pageCount)
}

async function loadPdfPage(document: PdfTextDocumentPort, pageIndex: number): Promise<PdfTextPagePort> {
  const page = await document.page(pageIndex)
  if (!page || typeof page !== 'object' || typeof page.extractText !== 'function') {
    throw new PdfExtractionError(`PDF text converter returned an invalid page ${pageIndex + 1}`)
  }
  return page
}

function normalizeTableCell(value: string | number | boolean | null, label: string): string {
  if (value !== null && typeof value !== 'string' && typeof value !== 'number' && typeof value !== 'boolean') {
    throw new PdfExtractionError(`extracted table ${label} is not a printable value`)
  }
  return String(value ?? '').replaceAll('|', '\\|').replace(/\r?\n/g, '<br>')
}

function validateEmbeddedImage(image: PdfEmbeddedImage, pageIndex: number, imageIndex: number): void {
  if (!image || typeof image !== 'object' || !(image.bytes instanceof Uint8Array)) {
    throw new PdfExtractionError(`PDF text converter returned an invalid image ${imageIndex + 1} for page ${pageIndex + 1}`)
  }
}
