// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Bounds for rich file previews. A reply crosses one JSON-RPC frame (16 MiB
 * on both transports), so binary previews stay under 10 MiB (base64 adds a
 * third) and parsed previews carry at most a few MiB of text however large
 * the file is.
 */

const MiB = 1024 * 1024

/** Largest file each preview kind reads. Over it, the preview says so instead of loading. */
export const PREVIEW_FILE_LIMITS = Object.freeze({
  image: 10 * MiB,
  pdf: 10 * MiB,
  table: 32 * MiB,
  sheet: 32 * MiB,
  document: 32 * MiB,
  html: 4 * MiB,
})

/** Largest single part inflated from an XLSX or DOCX archive. */
export const PREVIEW_PART_LIMIT = 64 * MiB

/** How much of a table a preview carries. */
export interface TableLimits {
  readonly maxRows: number
  readonly maxColumns: number
  readonly maxCellChars: number
  /** Total characters across all kept cells. */
  readonly maxChars: number
}

export const TABLE_LIMITS: TableLimits = Object.freeze({
  maxRows: 1000,
  maxColumns: 100,
  maxCellChars: 1000,
  maxChars: 2 * MiB,
})

/** How much of a word-processing document a preview carries. */
export interface DocumentLimits {
  readonly maxBlocks: number
  readonly maxChars: number
  readonly maxTableRows: number
  readonly maxTableColumns: number
}

export const DOCUMENT_LIMITS: DocumentLimits = Object.freeze({
  maxBlocks: 5000,
  maxChars: 2 * MiB,
  maxTableRows: 500,
  maxTableColumns: 50,
})

/** A cell's text, cut with an ellipsis past `max` characters. */
export function capCell(value: string, max: number): string {
  return value.length > max ? value.slice(0, Math.max(0, max - 1)) + '…' : value
}
