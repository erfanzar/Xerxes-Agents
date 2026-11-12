// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * An XLSX workbook as sheet tabs and one sheet's cells, read-only. Cells
 * show their cached values the way Excel displays them (shared and inline
 * strings, booleans, errors, numbers under their number format, dates as
 * ISO dates); a formula with no cached value shows as its formula. Only the
 * parts a preview needs are inflated, each under a byte limit.
 */

import { relationshipTargetPartName } from '../../skills/powerpoint/package.js'
import { decodeXmlEntities, parseOoxmlRelationships, parseXmlAttributes, xmlLocalName } from '../../skills/powerpoint/xml.js'
import type { OfficeZipReader } from '../../skills/powerpoint/zip.js'
import { capCell, PREVIEW_PART_LIMIT, TABLE_LIMITS, type TableLimits } from './limits.js'
import { formatCellNumber, formatCodeOf, generalNumber } from './numberFormat.js'

export interface SheetSummary {
  readonly name: string
  readonly hidden: boolean
}

export interface WorkbookPreview {
  readonly sheets: SheetSummary[]
  /** Zero-based index of the sheet whose cells are in `rows`. */
  readonly sheet: number
  /** Dense cells from A1, `rows[r][c]` is row r+1, column c+1; empty cells are ''. */
  readonly rows: string[][]
  /** The last row and column holding a value. */
  readonly totalRows: number
  readonly totalColumns: number
  readonly truncated: boolean
  /** Why the sheet has no cells to show (a chart sheet, say). */
  readonly notice?: string
}

/** A workbook part is missing or malformed. */
export class WorkbookFormatError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'WorkbookFormatError'
  }
}

const P = '(?:[\\w.-]+:)?'
const tag = (name: string): RegExp => new RegExp(`<${P}${name}\\b([^>]*?)(?:/>|>([\\s\\S]*?)</${P}${name}>)`, 'g')
const wrapped = (name: string): RegExp => new RegExp(`<${P}${name}\\b[^>]*>([\\s\\S]*?)</${P}${name}>`)
// matchAll clones a global pattern, so these are safe to share.
const ROW = tag('row'), CELL = tag('c'), TEXT = tag('t'), SHARED_ITEM = tag('si'), NUMBER_FORMAT = tag('numFmt')
const PHONETIC = new RegExp(`<${P}rPh\\b[\\s\\S]*?</${P}rPh>`, 'g')
const INLINE = wrapped('is'), VALUE = wrapped('v'), FORMULA = wrapped('f'), CELL_XFS = wrapped('cellXfs')
const STYLE_XF = new RegExp(`<${P}xf\\b([^>]*?)/?>`, 'g'), SHEET = new RegExp(`<${P}sheet\\b([^>]*?)/?>`, 'g')
const HAS_TEXT = new RegExp(`<${P}(?:v|t|f)\\b[^>]*>[^<]`)
const WORKBOOK_PROPERTIES = new RegExp(`<${P}workbookPr\\b([^>]*?)/?>`)

/** An attribute by local name: `id` finds `r:id`. */
function attribute(attributes: Readonly<Record<string, string>>, local: string): string | undefined {
  if (local in attributes) return attributes[local]
  for (const [name, value] of Object.entries(attributes)) if (xmlLocalName(name) === local) return value
  return undefined
}

function relsPartOf(part: string): string {
  const slash = part.lastIndexOf('/')
  return `${part.slice(0, slash + 1)}_rels/${part.slice(slash + 1)}.rels`
}

function readText(zip: OfficeZipReader, part: string, limit: number): string | undefined {
  const bytes = zip.read(part, limit)
  return bytes === undefined ? undefined : new TextDecoder().decode(bytes)
}

interface Relationship { readonly type: string; readonly part: string }

function relationships(zip: OfficeZipReader, part: string, limit: number): Map<string, Relationship> {
  const rels = relsPartOf(part)
  const xml = readText(zip, rels, limit)
  const out = new Map<string, Relationship>()
  if (xml === undefined) return out
  for (const rel of parseOoxmlRelationships(xml)) {
    if (rel.targetMode?.toLowerCase() === 'external') continue
    try { out.set(rel.id, { type: rel.type, part: relationshipTargetPartName(rels, decodeXmlEntities(rel.target)) }) }
    catch { /* a target outside the package is not a part we can read */ }
  }
  return out
}

/** Text of `<t>` runs, phonetic guides left out. */
function runText(inner: string): string {
  let text = ''
  for (const match of inner.replace(PHONETIC, '').matchAll(TEXT)) text += decodeXmlEntities(match[2] ?? '')
  return text
}

function sharedStrings(xml: string | undefined): string[] {
  if (xml === undefined) return []
  const out: string[] = []
  for (const match of xml.matchAll(SHARED_ITEM)) out.push(runText(match[2] ?? ''))
  return out
}

/** Number format code per cell style index. */
function styleFormats(xml: string | undefined): (string | undefined)[] {
  if (xml === undefined) return []
  const custom = new Map<number, string>()
  for (const match of xml.matchAll(NUMBER_FORMAT)) {
    const attributes = parseXmlAttributes(match[1] ?? '')
    const id = Number(attribute(attributes, 'numFmtId'))
    const code = attribute(attributes, 'formatCode')
    if (Number.isInteger(id) && code !== undefined) custom.set(id, decodeXmlEntities(code))
  }
  const xfs = CELL_XFS.exec(xml)?.[1] ?? ''
  const formats: (string | undefined)[] = []
  for (const match of xfs.matchAll(STYLE_XF)) {
    formats.push(formatCodeOf(Number(attribute(parseXmlAttributes(match[1] ?? ''), 'numFmtId') ?? 0), custom))
  }
  return formats
}

/** Zero-based column of a cell reference such as `BC12`. */
export function columnOf(reference: string): number | undefined {
  const letters = /^\$?([A-Za-z]{1,3})\$?\d*$/.exec(reference)?.[1]
  if (letters === undefined) return undefined
  let index = 0
  for (const letter of letters.toUpperCase()) index = index * 26 + (letter.charCodeAt(0) - 64)
  return index - 1
}

/** Spreadsheet column letters for a zero-based index: 0 is A, 26 is AA. */
export function columnName(index: number): string {
  let name = ''
  for (let value = index + 1; value > 0; value = Math.floor((value - 1) / 26)) name = String.fromCharCode(65 + ((value - 1) % 26)) + name
  return name
}

interface CellContext {
  readonly shared: readonly string[]
  readonly formats: readonly (string | undefined)[]
  readonly date1904: boolean
}

function cellText(attributes: Readonly<Record<string, string>>, inner: string | undefined, context: CellContext): string {
  if (inner === undefined) return ''
  const type = attribute(attributes, 't')
  if (type === 'inlineStr') {
    const is = INLINE.exec(inner)?.[1]
    return is === undefined ? '' : runText(is)
  }
  const raw = VALUE.exec(inner)?.[1]
  if (raw === undefined) {
    const formula = FORMULA.exec(inner)?.[1]
    return formula === undefined ? '' : '=' + decodeXmlEntities(formula)
  }
  const value = decodeXmlEntities(raw)
  switch (type) {
    case 's': return context.shared[Number.parseInt(value, 10)] ?? ''
    case 'b': return value === '1' || value.toLowerCase() === 'true' ? 'TRUE' : 'FALSE'
    case 'e':
    case 'str':
      return value
    case 'd': return value.replace(/T00:00:00(?:\.0+)?Z?$/, '')
    default: {
      const number = Number(value)
      if (value.trim() === '' || !Number.isFinite(number)) return value
      const style = Number(attribute(attributes, 's') ?? 0)
      const code = Number.isInteger(style) ? context.formats[style] : undefined
      return code === undefined ? generalNumber(number) : formatCellNumber(number, code, context.date1904)
    }
  }
}

/** Whether a cell past the kept range holds a value: enough to count it, without formatting it. */
function hasValue(attributes: Readonly<Record<string, string>>, inner: string | undefined, context: CellContext): boolean {
  if (inner === undefined || !HAS_TEXT.test(inner)) return false
  if (attribute(attributes, 't') !== 's') return true
  return Boolean(context.shared[Number.parseInt(VALUE.exec(inner)?.[1] ?? '', 10)])
}

/** Let other work on the event loop run between batches of a large sheet. */
const yieldToLoop = (): Promise<void> => new Promise(resolve => setImmediate(resolve))
const ROWS_PER_SLICE = 2000

async function sheetCells(xml: string, context: CellContext, limits: TableLimits): Promise<Omit<WorkbookPreview, 'sheets' | 'sheet'>> {
  const kept: string[][] = []
  let totalRows = 0
  let totalColumns = 0
  let budget = limits.maxChars
  let truncated = false
  let nextRow = 1
  let scanned = 0
  for (const rowMatch of xml.matchAll(ROW)) {
    if (++scanned % ROWS_PER_SLICE === 0) await yieldToLoop()
    const rowAttributes = parseXmlAttributes(rowMatch[1] ?? '')
    const declared = Number(attribute(rowAttributes, 'r'))
    const row = Number.isInteger(declared) && declared >= 1 ? declared : nextRow
    nextRow = row + 1
    let cursor = 0
    for (const cellMatch of (rowMatch[2] ?? '').matchAll(CELL)) {
      const cellAttributes = parseXmlAttributes(cellMatch[1] ?? '')
      const reference = attribute(cellAttributes, 'r')
      const column = (reference === undefined ? undefined : columnOf(reference)) ?? cursor
      cursor = column + 1
      // Past the kept range a cell only extends the totals.
      if (row > limits.maxRows || column >= limits.maxColumns || budget <= 0) {
        if (!hasValue(cellAttributes, cellMatch[2], context)) continue
        totalRows = Math.max(totalRows, row)
        totalColumns = Math.max(totalColumns, column + 1)
        truncated = true
        continue
      }
      const text = cellText(cellAttributes, cellMatch[2], context)
      if (!text) continue
      totalRows = Math.max(totalRows, row)
      totalColumns = Math.max(totalColumns, column + 1)
      const capped = capCell(text, limits.maxCellChars)
      if (capped !== text) truncated = true
      budget -= capped.length
      const cells = kept[row - 1] ?? (kept[row - 1] = [])
      cells[column] = capped
    }
  }
  const height = Math.min(totalRows, limits.maxRows)
  const width = Math.min(totalColumns, limits.maxColumns)
  const rows: string[][] = []
  for (let r = 0; r < height; r += 1) {
    const source = kept[r]
    const cells: string[] = Array<string>(width).fill('')
    if (source) for (let c = 0; c < width; c += 1) cells[c] = source[c] ?? ''
    rows.push(cells)
  }
  return { rows, totalRows, totalColumns, truncated: truncated || totalRows > height || totalColumns > width }
}

/**
 * Read the sheet list and one sheet's cells. An index past the last sheet
 * shows the last one; the reply says which sheet it is.
 */
export async function readWorkbook(zip: OfficeZipReader, sheetIndex = 0, limits: TableLimits = TABLE_LIMITS, partLimit = PREVIEW_PART_LIMIT): Promise<WorkbookPreview> {
  const root = relationships(zip, '', partLimit)
  const workbookPart = [...root.values()].find(rel => rel.type.endsWith('/officeDocument'))?.part ?? 'xl/workbook.xml'
  const workbook = readText(zip, workbookPart, partLimit)
  if (workbook === undefined) throw new WorkbookFormatError('The workbook part is missing, so this is not an Excel workbook')
  const rels = relationships(zip, workbookPart, partLimit)
  const entries: { summary: SheetSummary; rel: Relationship | undefined }[] = []
  for (const match of workbook.matchAll(SHEET)) {
    const attributes = parseXmlAttributes(match[1] ?? '')
    const name = attribute(attributes, 'name')
    if (name === undefined) continue
    const state = attribute(attributes, 'state')
    const id = attribute(attributes, 'id')
    entries.push({ summary: { name: decodeXmlEntities(name), hidden: state === 'hidden' || state === 'veryHidden' }, rel: id === undefined ? undefined : rels.get(id) })
  }
  if (!entries.length) throw new WorkbookFormatError('The workbook lists no sheets')
  const sheets = entries.map(entry => entry.summary)
  const index = Math.max(0, Math.min(entries.length - 1, Math.trunc(sheetIndex)))
  const chosen = entries[index]!
  const empty = { sheets, sheet: index, rows: [], totalRows: 0, totalColumns: 0, truncated: false }
  if (chosen.rel === undefined) return { ...empty, notice: 'This sheet has no data part in the workbook.' }
  if (chosen.rel.type.endsWith('/chartsheet')) return { ...empty, notice: 'This sheet is a chart. Open the workbook to see it.' }
  if (!chosen.rel.type.endsWith('/worksheet')) return { ...empty, notice: 'This kind of sheet cannot be previewed. Open the workbook to see it.' }
  const sheetXml = readText(zip, chosen.rel.part, partLimit)
  if (sheetXml === undefined) throw new WorkbookFormatError(`The sheet part ${chosen.rel.part} is missing`)
  const sharedPart = [...rels.values()].find(rel => rel.type.endsWith('/sharedStrings'))?.part
  const stylesPart = [...rels.values()].find(rel => rel.type.endsWith('/styles'))?.part
  const workbookPr = parseXmlAttributes(WORKBOOK_PROPERTIES.exec(workbook)?.[1] ?? '')
  const date1904 = ['1', 'true'].includes(attribute(workbookPr, 'date1904') ?? '')
  const context: CellContext = {
    shared: sharedStrings(sharedPart === undefined ? undefined : readText(zip, sharedPart, partLimit)),
    formats: styleFormats(stylesPart === undefined ? undefined : readText(zip, stylesPart, partLimit)),
    date1904,
  }
  return { sheets, sheet: index, ...(await sheetCells(sheetXml, context, limits)) }
}
