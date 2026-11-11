// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * CSV and TSV as a table: RFC 4180 quoting (doubled quotes, delimiters and
 * line breaks inside quotes), any line ending, and the encodings spreadsheet
 * apps actually write (UTF-8 with or without a BOM, UTF-16 with a BOM, and
 * Windows-1252 when the bytes are not UTF-8). Every record is counted, but
 * only the first rows are kept, within a row, column and character budget.
 */

import { capCell, TABLE_LIMITS, type TableLimits } from './limits.js'

export interface DelimitedTable {
  readonly delimiter: string
  readonly encoding: 'utf-8' | 'utf-16le' | 'utf-16be' | 'windows-1252'
  readonly header: string[]
  readonly rows: string[][]
  /** Data records in the file, not counting the header. */
  readonly totalRows: number
  /** The widest record in the file. */
  readonly totalColumns: number
  /** Rows, columns or cell text were left out. */
  readonly truncated: boolean
}

/** Decode text the way spreadsheet apps write CSV. */
export function decodeDelimitedText(bytes: Uint8Array): { text: string; encoding: DelimitedTable['encoding'] } {
  if (bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) return { text: new TextDecoder('utf-8').decode(bytes.subarray(3)), encoding: 'utf-8' }
  if (bytes[0] === 0xff && bytes[1] === 0xfe) return { text: new TextDecoder('utf-16le').decode(bytes.subarray(2)), encoding: 'utf-16le' }
  if (bytes[0] === 0xfe && bytes[1] === 0xff) return { text: new TextDecoder('utf-16be').decode(bytes.subarray(2)), encoding: 'utf-16be' }
  try { return { text: new TextDecoder('utf-8', { fatal: true }).decode(bytes), encoding: 'utf-8' } }
  catch { return { text: new TextDecoder('windows-1252').decode(bytes), encoding: 'windows-1252' } }
}

const CANDIDATES = [',', ';', '\t', '|'] as const

/**
 * The delimiter of a `.csv`: the candidate that splits the first lines into
 * the same number (>1) of fields most often. Europe's Excel writes `;`.
 */
export function sniffDelimiter(text: string): string {
  const lines: string[] = []
  let start = 0
  let quoted = false
  for (let index = 0; index < text.length && lines.length < 20; index += 1) {
    const char = text[index]
    if (char === '"') quoted = !quoted
    else if (!quoted && (char === '\n' || char === '\r')) {
      if (index > start) lines.push(text.slice(start, index))
      start = index + 1
    }
  }
  if (lines.length < 20 && start < text.length) lines.push(text.slice(start))
  let best: string = ','
  let bestScore = 0
  for (const candidate of CANDIDATES) {
    const counts = lines.map(line => countOutsideQuotes(line, candidate))
    const first = counts[0] ?? 0
    if (first === 0) continue
    const consistent = counts.filter(count => count === first).length
    const score = consistent * 1000 + first
    if (score > bestScore) { best = candidate; bestScore = score }
  }
  return best
}

function countOutsideQuotes(line: string, delimiter: string): number {
  let count = 0
  let quoted = false
  for (const char of line) {
    if (char === '"') quoted = !quoted
    else if (!quoted && char === delimiter) count += 1
  }
  return count
}

/** Parse delimited text; `delimiter` is sniffed when omitted. */
export function parseDelimited(text: string, delimiter = sniffDelimiter(text), limits: TableLimits = TABLE_LIMITS): Omit<DelimitedTable, 'encoding'> {
  const kept: string[][] = []
  let records = 0
  let widest = 0
  let budget = limits.maxChars
  let full = false
  let truncated = false
  let field = ''
  let record: string[] = []
  let quoted = false
  let index = 0
  const keepRecord = (): void => {
    // A line with nothing on it is not a record.
    if (record.length === 1 && record[0] === '') { record = []; return }
    records += 1
    widest = Math.max(widest, record.length)
    if (!full && kept.length <= limits.maxRows) {
      const cells = record.slice(0, limits.maxColumns).map(cell => capCell(cell, limits.maxCellChars))
      const size = cells.reduce((sum, cell) => sum + cell.length, 0)
      // Rows are kept from the top only: once one does not fit, none after it is kept.
      if (size > budget) full = true
      else {
        if (record.length > limits.maxColumns || cells.some((cell, index) => cell !== record[index])) truncated = true
        budget -= size
        kept.push(cells)
      }
    }
    record = []
  }
  const length = text.length
  const delimiterCode = delimiter.charCodeAt(0)
  while (index < length) {
    const char = text[index]!
    if (quoted) {
      if (char === '"') {
        if (text[index + 1] === '"') { field += '"'; index += 2; continue }
        quoted = false
        index += 1
        continue
      }
      // Copy up to the next quote in one step.
      const next = text.indexOf('"', index)
      const stop = next < 0 ? length : next
      field += text.slice(index, stop)
      index = stop
      continue
    }
    if (char === '"' && field === '') { quoted = true; index += 1; continue }
    if (char === delimiter) { record.push(field); field = ''; index += 1; continue }
    if (char === '\n' || char === '\r') {
      record.push(field); field = ''
      keepRecord()
      index += char === '\r' && text[index + 1] === '\n' ? 2 : 1
      continue
    }
    // An unquoted run: copy to the next delimiter or line break in one step.
    // A quote inside it is a literal character, as spreadsheet apps read it.
    let stop = index + 1
    while (stop < length) {
      const code = text.charCodeAt(stop)
      if (code === delimiterCode || code === 10 || code === 13) break
      stop += 1
    }
    field += text.slice(index, stop)
    index = stop
  }
  // A last record without a line break, or an unterminated quote taken as written.
  if (field !== '' || record.length > 0) { record.push(field); keepRecord() }
  const [header = [], ...rows] = kept
  const columns = Math.min(widest, limits.maxColumns)
  const pad = (row: string[]): string[] => row.length >= columns ? row : [...row, ...Array<string>(columns - row.length).fill('')]
  return {
    delimiter,
    header: pad(header),
    rows: rows.map(pad),
    totalRows: Math.max(0, records - 1),
    totalColumns: widest,
    truncated: truncated || rows.length < records - 1,
  }
}

/** Read a CSV/TSV file's bytes as a table. `.tsv` and `.tab` always split on tabs. */
export function readDelimited(bytes: Uint8Array, path: string, limits: TableLimits = TABLE_LIMITS): DelimitedTable {
  const { text, encoding } = decodeDelimitedText(bytes)
  const delimiter = /\.(tsv|tab)$/i.test(path) ? '\t' : sniffDelimiter(text)
  return { ...parseDelimited(text, delimiter, limits), encoding }
}
