// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** A value that can be represented in marker-style JSON metadata. */
export type DocumentJsonValue = string | number | boolean | null | DocumentJsonObject | readonly DocumentJsonValue[]

/** Structured metadata returned by an OCR or document-conversion implementation. */
export interface DocumentJsonObject {
  readonly [key: string]: DocumentJsonValue
}

/** Values may be synchronous in tests or asynchronous in a real Bun integration. */
export type MaybePromise<T> = T | Promise<T>

/** A text table extracted from one PDF page. */
export interface PdfExtractedTable {
  readonly headers: readonly string[]
  readonly rows: readonly (readonly (string | number | boolean | null)[])[]
}

/** An embedded PDF image, already encoded in a format the caller can persist. */
export interface PdfEmbeddedImage {
  readonly bytes: Uint8Array
}

/** A single page exposed by an injected, JavaScript-native PDF text engine. */
export interface PdfTextPagePort {
  extractImages?(): MaybePromise<readonly PdfEmbeddedImage[]>
  extractTables?(): MaybePromise<readonly PdfExtractedTable[]>
  extractText(): MaybePromise<string>
}

/** Metadata fields surfaced by the original PyMuPDF helper. */
export interface PdfSourceMetadata {
  readonly author?: string | null
  readonly creator?: string | null
  readonly format?: string | null
  readonly producer?: string | null
  readonly subject?: string | null
  readonly title?: string | null
}

/** An opened PDF document owned by an injected text-conversion engine. */
export interface PdfTextDocumentPort {
  readonly metadata: PdfSourceMetadata
  readonly pageCount: number
  close?(): MaybePromise<void>
  page(index: number): MaybePromise<PdfTextPagePort>
}

/**
 * Boundary for a JavaScript/Bun PDF engine. Implement this with PDF.js, a WASM
 * parser, or a host service; this skill never shells out to PyMuPDF or Python.
 */
export interface PdfTextConversionPort {
  open(path: string): MaybePromise<PdfTextDocumentPort>
  toMarkdown?(path: string, pages?: readonly number[]): MaybePromise<string>
}

/** One structured page from a high-fidelity OCR conversion. */
export interface OcrDocumentPage {
  readonly markdown: string
  readonly metadata?: DocumentJsonObject
  readonly pageNumber: number
  readonly text?: string
}

/** Raw output returned from an injected OCR/layout engine. */
export interface OcrConvertedDocument {
  readonly images?: Readonly<Record<string, Uint8Array>>
  readonly markdown: string
  readonly metadata?: DocumentJsonObject
  readonly pages?: readonly OcrDocumentPage[]
}

/** Input sent to an injected OCR/layout engine. */
export interface OcrConversionRequest {
  readonly path: string
  readonly useLlm: boolean
}

/**
 * Boundary for a native OCR/layout implementation. The host decides whether it
 * uses WebAssembly, a remote API, or another JavaScript package.
 */
export interface OcrConversionPort {
  convert(request: OcrConversionRequest): MaybePromise<OcrConvertedDocument>
}

/** Explicit filesystem boundary for writing extracted images. */
export interface OcrDocumentFilesystemPort {
  ensureDirectory(path: string): MaybePromise<void>
  join(directory: string, name: string): string
  writeFile(path: string, bytes: Uint8Array): MaybePromise<void>
}

/** Explicit storage boundary for marker-pdf's former disk-space check. */
export interface OcrDocumentDiskSpacePort {
  freeBytes(path: string): MaybePromise<number>
}
