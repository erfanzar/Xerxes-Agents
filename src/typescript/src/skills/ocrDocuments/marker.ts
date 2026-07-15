// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type {
  DocumentJsonObject,
  OcrConversionPort,
  OcrConvertedDocument,
  OcrDocumentDiskSpacePort,
  OcrDocumentFilesystemPort,
  OcrDocumentPage,
} from './types.js'

/** marker-pdf's documented minimum free disk capacity for models and dependencies. */
export const MARKER_MINIMUM_FREE_BYTES = 5 * 1024 ** 3

/** Output modes supported by the former marker helper. */
export type MarkerOutputFormat = 'json' | 'markdown'

/** Options corresponding to marker helper flags. */
export interface MarkerExtractionOptions {
  readonly outputDirectory?: string
  readonly outputFormat?: MarkerOutputFormat
  readonly useLlm?: boolean
}

/** An image emitted by OCR and written through the explicit filesystem port. */
export interface WrittenMarkerImage {
  readonly name: string
  readonly path: string
}

/** Structured native result for marker-style OCR and layout extraction. */
export interface MarkerExtraction {
  readonly images: readonly WrittenMarkerImage[]
  readonly markdown: string
  readonly metadata: DocumentJsonObject
  readonly pages: readonly OcrDocumentPage[]
}

/** Result of the disk-space check previously performed by `--check`. */
export interface MarkerDiskSpaceCheck {
  readonly freeBytes: number
  readonly freeGigabytes: number
  readonly meetsRequirement: boolean
  readonly minimumBytes: number
  readonly path: string
}

/** Failure raised when OCR model storage is insufficient. */
export class MarkerDiskSpaceError extends Error {
  readonly check: MarkerDiskSpaceCheck

  constructor(check: MarkerDiskSpaceCheck) {
    super(formatMarkerDiskSpaceCheck(check))
    this.name = 'MarkerDiskSpaceError'
    this.check = check
  }
}

/** Failure raised when an OCR adapter returns malformed content. */
export class MarkerExtractionError extends Error {
  constructor(message: string, options: { readonly cause?: unknown } = {}) {
    super(message, options)
    this.name = 'MarkerExtractionError'
  }
}

/** Check disk capacity through an explicit host-provided adapter. */
export async function checkMarkerDiskSpace(
  diskSpace: OcrDocumentDiskSpacePort,
  path: string,
): Promise<MarkerDiskSpaceCheck> {
  const normalizedPath = requireMarkerText(path, 'path')
  const freeBytes = await diskSpace.freeBytes(normalizedPath)
  if (!Number.isFinite(freeBytes) || freeBytes < 0) {
    throw new MarkerExtractionError('disk-space adapter returned an invalid free-byte count')
  }
  return {
    freeBytes,
    freeGigabytes: freeBytes / 1024 ** 3,
    meetsRequirement: freeBytes >= MARKER_MINIMUM_FREE_BYTES,
    minimumBytes: MARKER_MINIMUM_FREE_BYTES,
    path: normalizedPath,
  }
}

/** Raise a useful error when the explicit disk check does not meet marker requirements. */
export async function assertMarkerDiskSpace(
  diskSpace: OcrDocumentDiskSpacePort,
  path: string,
): Promise<MarkerDiskSpaceCheck> {
  const check = await checkMarkerDiskSpace(diskSpace, path)
  if (!check.meetsRequirement) throw new MarkerDiskSpaceError(check)
  return check
}

/** Format the status text printed by the original `--check` mode. */
export function formatMarkerDiskSpaceCheck(check: MarkerDiskSpaceCheck): string {
  const free = check.freeGigabytes.toFixed(1)
  if (check.meetsRequirement) return `✓ ${free}GB free — sufficient for marker-pdf`
  return [
    `⚠️  Only ${free}GB free. marker-pdf needs ~5GB for PyTorch + models.`,
    'Use a text-based PDF extractor or free up disk space.',
  ].join('\n')
}

/**
 * Run a marker-equivalent conversion through an injected native OCR/layout
 * engine. The converter controls OCR and optional LLM post-processing.
 */
export async function extractMarkerDocument(
  path: string,
  converter: OcrConversionPort,
  options: MarkerExtractionOptions = {},
  filesystem?: OcrDocumentFilesystemPort,
): Promise<MarkerExtraction> {
  const normalizedPath = requireMarkerText(path, 'path')
  const outputFormat = options.outputFormat ?? 'markdown'
  if (outputFormat !== 'markdown' && outputFormat !== 'json') {
    throw new TypeError('outputFormat must be "markdown" or "json"')
  }
  if (options.outputDirectory !== undefined && filesystem === undefined) {
    throw new MarkerExtractionError('an outputDirectory requires an explicit filesystem adapter')
  }
  const rendered = await converter.convert({ path: normalizedPath, useLlm: options.useLlm ?? false })
  validateConvertedDocument(rendered)
  const extraction: MarkerExtraction = {
    images: [],
    markdown: rendered.markdown,
    metadata: rendered.metadata ?? {},
    pages: rendered.pages ?? [],
  }
  if (options.outputDirectory === undefined || filesystem === undefined || !rendered.images) return extraction

  const outputDirectory = requireMarkerText(options.outputDirectory, 'outputDirectory')
  await filesystem.ensureDirectory(outputDirectory)
  const images: WrittenMarkerImage[] = []
  for (const [name, bytes] of Object.entries(rendered.images)) {
    validateMarkerImage(name, bytes)
    const outputPath = filesystem.join(outputDirectory, name)
    await filesystem.writeFile(outputPath, bytes)
    images.push({ name, path: outputPath })
  }
  return { ...extraction, images }
}

/** Render the exact user-facing markdown or JSON body from the prior helper. */
export function formatMarkerExtraction(extraction: MarkerExtraction, outputFormat: MarkerOutputFormat = 'markdown'): string {
  if (outputFormat === 'markdown') return extraction.markdown
  if (outputFormat !== 'json') throw new TypeError('outputFormat must be "markdown" or "json"')
  return JSON.stringify({ markdown: extraction.markdown, metadata: extraction.metadata }, null, 2)
}

function requireMarkerText(value: string, name: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError(`${name} must be a non-empty string`)
  }
  return value.trim()
}

function validateConvertedDocument(document: OcrConvertedDocument): void {
  if (!document || typeof document !== 'object') {
    throw new MarkerExtractionError('OCR converter returned no document')
  }
  if (typeof document.markdown !== 'string') {
    throw new MarkerExtractionError('OCR converter returned non-text markdown')
  }
  if (document.metadata !== undefined && !isDocumentJsonObject(document.metadata)) {
    throw new MarkerExtractionError('OCR converter returned invalid metadata')
  }
  if (document.pages !== undefined) {
    if (!Array.isArray(document.pages)) throw new MarkerExtractionError('OCR converter returned invalid pages')
    for (const page of document.pages) validateOcrPage(page)
  }
  if (document.images !== undefined && (!document.images || typeof document.images !== 'object' || Array.isArray(document.images))) {
    throw new MarkerExtractionError('OCR converter returned invalid images')
  }
}

function isDocumentJsonObject(value: unknown): value is DocumentJsonObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function validateOcrPage(page: OcrDocumentPage): void {
  if (!page || typeof page !== 'object') throw new MarkerExtractionError('OCR converter returned an invalid page')
  if (!Number.isSafeInteger(page.pageNumber) || page.pageNumber < 1) {
    throw new MarkerExtractionError('OCR converter returned a page with an invalid pageNumber')
  }
  if (typeof page.markdown !== 'string') {
    throw new MarkerExtractionError(`OCR converter returned non-text markdown for page ${page.pageNumber}`)
  }
  if (page.text !== undefined && typeof page.text !== 'string') {
    throw new MarkerExtractionError(`OCR converter returned non-text content for page ${page.pageNumber}`)
  }
  if (page.metadata !== undefined && !isDocumentJsonObject(page.metadata)) {
    throw new MarkerExtractionError(`OCR converter returned invalid metadata for page ${page.pageNumber}`)
  }
}

function validateMarkerImage(name: string, bytes: Uint8Array): void {
  if (!name || name === '.' || name === '..' || name.includes('/') || name.includes('\\')) {
    throw new MarkerExtractionError(`OCR converter returned an unsafe image name: ${JSON.stringify(name)}`)
  }
  if (!(bytes instanceof Uint8Array)) {
    throw new MarkerExtractionError(`OCR converter returned invalid image data for ${name}`)
  }
}
