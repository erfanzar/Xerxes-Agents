// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { basename, dirname, extname } from 'node:path'

import { ValidationError } from '../core/errors.js'
import type { ToolRegistry } from '../executors/toolRegistry.js'
import { UnsupportedOfficeZipError } from '../skills/powerpoint/errors.js'
import { joinOfficePartName } from '../skills/powerpoint/package.js'
import { escapeXmlAttribute, parseOoxmlRelationships, parseXmlAttributes } from '../skills/powerpoint/xml.js'
import { createOfficeZip, readOfficeZip } from '../skills/powerpoint/zip.js'
import type { JsonObject, JsonValue, ToolDefinition } from '../types/toolCalls.js'
import { isJsonObject } from '../types/toolCalls.js'
import { parseCsv, stringifyCsv } from './dataTools.js'
import { optionalBoolean, optionalInteger, optionalString, requiredString } from './inputs.js'

/** One workbook cell as it crosses the tool boundary. `null` is an empty cell. */
export type XlsxCell = string | number | boolean | null

const MAX_SHEETS = 100
const MAX_ROWS = 1_048_576
const MAX_COLUMNS = 16_384
const MAX_CELLS = 2_000_000
const MAX_STRING_CHARS = 32_767
const MAX_SHEET_NAME = 31
const MAX_READ_BYTES = 50_000_000
const DEFAULT_READ_ROWS = 1000
const MAX_READ_ROWS = 100_000

const MAIN_NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
const REL_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
const PACKAGE_REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
const XML_HEADER = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

export const XLSX_WRITER_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'XlsxWriter',
    description: 'Write a real Excel workbook (.xlsx) with one or more sheets of rows. Use this instead of WriteFile '
      + 'for .xlsx, because WriteFile stores plain text that Excel cannot open. Numbers stay numbers, booleans stay '
      + 'booleans, strings stay strings, and null leaves the cell empty; no dates or formulas are inferred. Refuses '
      + 'an existing file unless overwrite=true.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        path: { type: 'string', description: 'Workspace-relative output path ending in .xlsx.' },
        sheets: {
          type: 'array',
          description: 'Sheets in order.',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              name: { type: 'string', description: 'Sheet name, at most 31 characters, none of []:*?/\\.' },
              rows: {
                type: 'array',
                description: 'Rows of cells; each cell is a string, number, boolean or null.',
                items: { type: 'array', items: { type: ['string', 'number', 'boolean', 'null'] } },
              },
            },
            required: ['name', 'rows'],
          },
        },
        overwrite: { type: 'boolean', default: false, description: 'Allow replacing an existing file.' },
      },
      required: ['path', 'sheets'],
    },
  },
}

export const XLSX_READER_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'XlsxReader',
    description: 'Read rows from one sheet of an .xlsx workbook. Cells come back as strings, numbers, booleans or '
      + 'null; dates are returned as the serial numbers stored in the file. Lists every sheet name; reads the '
      + 'first sheet unless `sheet` is given.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        path: { type: 'string', description: 'Workspace-relative .xlsx path.' },
        sheet: { type: 'string', description: 'Sheet name. Defaults to the first sheet.' },
        max_rows: {
          type: 'integer',
          default: DEFAULT_READ_ROWS,
          description: `Maximum rows to return (1 to ${MAX_READ_ROWS}).`,
        },
      },
      required: ['path'],
    },
  },
}

export const XLSX_CONVERT_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'XlsxConvert',
    description: 'Convert an .xlsx sheet to CSV, or a CSV to an .xlsx workbook. Use it so a script can work on a '
      + 'workbook with only its standard library. Replies with the row count and header only, never the rows. CSV '
      + 'text that is a plain decimal number becomes a number; everything else, including IDs with leading zeros, '
      + 'stays text. Refuses an existing output unless overwrite=true.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        path: { type: 'string', description: 'The .xlsx or .csv file.' },
        output_path: { type: 'string', description: 'The .csv or .xlsx file to write.' },
        sheet: { type: 'string', description: 'Sheet name. Default: the first.' },
        overwrite: { type: 'boolean', default: false },
      },
      required: ['path', 'output_path'],
    },
  },
}

export const XLSX_TOOL_DEFINITIONS: readonly ToolDefinition[] = [XLSX_WRITER_DEFINITION, XLSX_READER_DEFINITION, XLSX_CONVERT_DEFINITION]

export interface XlsxToolOptions {
  /** Resolve a tool-supplied path inside the workspace; throws when it escapes. */
  readonly resolvePath: (candidate: string) => Promise<string>
}

const READER_CAPABILITIES = Object.freeze({
  concurrencySafe: true,
  defer: false,
  destructive: false,
  openWorld: false,
  readOnly: true,
} as const)

const WRITER_CAPABILITIES = Object.freeze({
  concurrencySafe: false,
  defer: false,
  destructive: true,
  interruptBehavior: 'block',
  openWorld: false,
  readOnly: false,
} as const)

/** Register XlsxWriter, XlsxReader and XlsxConvert. */
export function registerXlsxTools(registry: ToolRegistry, options: XlsxToolOptions): void {
  registry.register(XLSX_WRITER_DEFINITION, inputs => writeWorkbook(inputs, options), 'default', WRITER_CAPABILITIES)
  registry.register(XLSX_READER_DEFINITION, inputs => readWorkbook(inputs, options), 'default', READER_CAPABILITIES)
  registry.register(XLSX_CONVERT_DEFINITION, inputs => convertWorkbook(inputs, options), 'default', WRITER_CAPABILITIES)
}

// --- writing ---------------------------------------------------------------

interface SheetInput {
  readonly name: string
  readonly rows: readonly (readonly XlsxCell[])[]
}

const INVALID_XML_CHARS = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\uFFFE\uFFFF]/u
const FORBIDDEN_SHEET_CHARS = /[\[\]:*?/\\]/u

function escapeText(value: string): string {
  return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
}

function columnLetters(index: number): string {
  let letters = ''
  for (let n = index + 1; n > 0; n = Math.floor((n - 1) / 26)) letters = String.fromCharCode(65 + ((n - 1) % 26)) + letters
  return letters
}

function parseSheets(inputs: JsonObject): SheetInput[] {
  const raw = inputs['sheets']
  if (!Array.isArray(raw) || raw.length === 0) throw new ValidationError('sheets', 'must be a non-empty array', raw)
  if (raw.length > MAX_SHEETS) throw new ValidationError('sheets', `must have at most ${MAX_SHEETS} sheets`)
  const names = new Set<string>()
  let cells = 0
  return raw.map((entry, sheetIndex) => {
    const field = `sheets[${sheetIndex}]`
    if (!isJsonObject(entry)) throw new ValidationError(field, 'must be an object with name and rows', entry)
    const name = entry['name']
    if (typeof name !== 'string') throw new ValidationError(`${field}.name`, 'must be a string', name)
    validateSheetName(name, `${field}.name`)
    if (names.has(name.toLowerCase())) throw new ValidationError(`${field}.name`, 'duplicates another sheet name (names are case-insensitive)', name)
    names.add(name.toLowerCase())
    const rawRows = entry['rows']
    if (!Array.isArray(rawRows)) throw new ValidationError(`${field}.rows`, 'must be an array of rows', rawRows)
    if (rawRows.length > MAX_ROWS) throw new ValidationError(`${field}.rows`, `must have at most ${MAX_ROWS} rows`)
    const rows = rawRows.map((rawRow, rowIndex) => {
      if (!Array.isArray(rawRow)) throw new ValidationError(`${field}.rows[${rowIndex}]`, 'must be an array of cells', rawRow)
      if (rawRow.length > MAX_COLUMNS) throw new ValidationError(`${field}.rows[${rowIndex}]`, `must have at most ${MAX_COLUMNS} cells`)
      cells += rawRow.length
      if (cells > MAX_CELLS) throw new ValidationError('sheets', `must have at most ${MAX_CELLS} cells in total`)
      return rawRow.map((cell, cellIndex) => validateCell(cell, `${field}.rows[${rowIndex}][${cellIndex}]`))
    })
    return { name, rows }
  })
}

function validateSheetName(name: string, field: string): void {
  if (name.length === 0 || name.length > MAX_SHEET_NAME) throw new ValidationError(field, `must be 1 to ${MAX_SHEET_NAME} characters`, name)
  if (FORBIDDEN_SHEET_CHARS.test(name)) throw new ValidationError(field, 'must not contain any of [ ] : * ? / \\', name)
  if (name.startsWith("'") || name.endsWith("'")) throw new ValidationError(field, 'must not start or end with an apostrophe', name)
  if (INVALID_XML_CHARS.test(name)) throw new ValidationError(field, 'must not contain control characters', name)
}

function validateCell(cell: JsonValue | undefined, field: string): XlsxCell {
  if (cell === null || typeof cell === 'boolean') return cell
  if (typeof cell === 'number') {
    if (!Number.isFinite(cell)) throw new ValidationError(field, 'must be a finite number', cell)
    return cell
  }
  if (typeof cell === 'string') {
    if (cell.length > MAX_STRING_CHARS) throw new ValidationError(field, `must be at most ${MAX_STRING_CHARS} characters`)
    if (INVALID_XML_CHARS.test(cell)) throw new ValidationError(field, 'contains a control character that XML cannot store', cell)
    return cell
  }
  throw new ValidationError(field, 'must be a string, number, boolean or null', cell)
}

/** Build the bytes of a minimal, valid workbook. Exported so tests and other tools can reuse it. */
export function buildWorkbook(sheets: readonly SheetInput[]): Uint8Array {
  const shared: string[] = []
  const sharedIndex = new Map<string, number>()
  let sharedRefs = 0
  const intern = (value: string): number => {
    sharedRefs += 1
    const existing = sharedIndex.get(value)
    if (existing !== undefined) return existing
    shared.push(value)
    sharedIndex.set(value, shared.length - 1)
    return shared.length - 1
  }

  const encoder = new TextEncoder()
  const entries: { name: string; data: Uint8Array }[] = []
  const add = (name: string, xml: string): void => {
    entries.push({ name, data: encoder.encode(XML_HEADER + xml) })
  }

  sheets.forEach((sheet, sheetIndex) => {
    const rowXml: string[] = []
    let maxColumn = 0
    sheet.rows.forEach((row, rowIndex) => {
      const cellXml: string[] = []
      row.forEach((cell, columnIndex) => {
        if (cell === null) return
        const ref = `${columnLetters(columnIndex)}${rowIndex + 1}`
        maxColumn = Math.max(maxColumn, columnIndex + 1)
        if (typeof cell === 'string') cellXml.push(`<c r="${ref}" t="s"><v>${intern(cell)}</v></c>`)
        else if (typeof cell === 'boolean') cellXml.push(`<c r="${ref}" t="b"><v>${cell ? 1 : 0}</v></c>`)
        else cellXml.push(`<c r="${ref}"><v>${String(cell)}</v></c>`)
      })
      if (cellXml.length > 0) rowXml.push(`<row r="${rowIndex + 1}">${cellXml.join('')}</row>`)
    })
    const dimension = maxColumn > 0 ? `<dimension ref="A1:${columnLetters(maxColumn - 1)}${sheet.rows.length}"/>` : ''
    add(`xl/worksheets/sheet${sheetIndex + 1}.xml`,
      `<worksheet xmlns="${MAIN_NS}">${dimension}<sheetData>${rowXml.join('')}</sheetData></worksheet>`)
  })

  const sharedItems = shared.map(value => {
    const space = value !== value.trim() || value.includes('\n') ? ' xml:space="preserve"' : ''
    return `<si><t${space}>${escapeText(value)}</t></si>`
  })
  add('xl/sharedStrings.xml',
    `<sst xmlns="${MAIN_NS}" count="${sharedRefs}" uniqueCount="${shared.length}">${sharedItems.join('')}</sst>`)

  add('xl/styles.xml',
    `<styleSheet xmlns="${MAIN_NS}"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>`
    + '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
    + '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
    + '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
    + '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
    + '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')

  const sheetEntries = sheets
    .map((sheet, index) => `<sheet name="${escapeXmlAttribute(sheet.name)}" sheetId="${index + 1}" r:id="rId${index + 1}"/>`)
    .join('')
  add('xl/workbook.xml', `<workbook xmlns="${MAIN_NS}" xmlns:r="${REL_NS}"><sheets>${sheetEntries}</sheets></workbook>`)

  const sheetRels = sheets
    .map((_, index) => `<Relationship Id="rId${index + 1}" Type="${REL_NS}/worksheet" Target="worksheets/sheet${index + 1}.xml"/>`)
    .join('')
  add('xl/_rels/workbook.xml.rels',
    `<Relationships xmlns="${PACKAGE_REL_NS}">${sheetRels}`
    + `<Relationship Id="rId${sheets.length + 1}" Type="${REL_NS}/sharedStrings" Target="sharedStrings.xml"/>`
    + `<Relationship Id="rId${sheets.length + 2}" Type="${REL_NS}/styles" Target="styles.xml"/></Relationships>`)

  add('_rels/.rels',
    `<Relationships xmlns="${PACKAGE_REL_NS}"><Relationship Id="rId1" Type="${REL_NS}/officeDocument" Target="xl/workbook.xml"/></Relationships>`)

  const sheetOverrides = sheets
    .map((_, index) => `<Override PartName="/xl/worksheets/sheet${index + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>`)
    .join('')
  add('[Content_Types].xml',
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    + '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    + '<Default Extension="xml" ContentType="application/xml"/>'
    + '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    + `${sheetOverrides}`
    + '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
    + '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
    + '</Types>')

  return createOfficeZip(entries)
}

async function writeWorkbook(inputs: JsonObject, options: XlsxToolOptions): Promise<JsonObject> {
  const path = requiredString(inputs, 'path')
  if (!path.toLowerCase().endsWith('.xlsx')) throw new ValidationError('path', 'must end in .xlsx', path)
  const overwrite = optionalBoolean(inputs, 'overwrite', false)
  const sheets = parseSheets(inputs)
  const bytes = buildWorkbook(sheets)
  await writeOutput(path, 'path', bytes, overwrite, options)
  return {
    success: true,
    file_path: path,
    sheets: sheets.length,
    rows: sheets.reduce((total, sheet) => total + sheet.rows.length, 0),
    bytes: bytes.byteLength,
  }
}

/** Write `bytes` to a workspace path, refusing an existing file unless `overwrite`. */
async function writeOutput(path: string, field: string, bytes: Uint8Array | string, overwrite: boolean, options: XlsxToolOptions): Promise<void> {
  const target = await options.resolvePath(path)
  await mkdir(dirname(target), { recursive: true })
  // Re-resolve right before the write so a symlink swapped in meanwhile is caught.
  const checked = await options.resolvePath(path)
  if (overwrite) {
    await writeFile(checked, bytes)
    return
  }
  try {
    await writeFile(checked, bytes, { flag: 'wx' })
  } catch (error) {
    if (typeof error === 'object' && error !== null && 'code' in error && error.code === 'EEXIST') {
      throw new ValidationError(field, 'already exists; pass overwrite=true to replace it', path)
    }
    throw error
  }
}

// --- reading ---------------------------------------------------------------

function decodeXml(value: string): string {
  return value.replace(/&(#x[0-9a-fA-F]+|#\d+|lt|gt|amp|quot|apos);/gu, (whole, entity: string) => {
    switch (entity) {
      case 'lt': return '<'
      case 'gt': return '>'
      case 'amp': return '&'
      case 'quot': return '"'
      case 'apos': return "'"
      default: {
        const code = entity[1] === 'x' ? Number.parseInt(entity.slice(2), 16) : Number.parseInt(entity.slice(1), 10)
        return Number.isInteger(code) && code >= 0 && code <= 0x10ffff ? String.fromCodePoint(code) : whole
      }
    }
  })
}

function textRuns(xml: string): string {
  const withoutPhonetic = xml.replace(/<rPh\b[\s\S]*?<\/rPh>/gu, '')
  let text = ''
  for (const match of withoutPhonetic.matchAll(/<t\b[^>]*?(?:\/>|>([\s\S]*?)<\/t>)/gu)) text += decodeXml(match[1] ?? '')
  return text
}

function sharedStringsOf(xml: string | undefined): string[] {
  if (xml === undefined) return []
  const items: string[] = []
  for (const match of xml.matchAll(/<si\b[^>]*?(?:\/>|>([\s\S]*?)<\/si>)/gu)) items.push(textOf(match[1]))
  return items
}

function textOf(inner: string | undefined): string {
  return inner === undefined ? '' : textRuns(inner)
}

function columnIndex(reference: string): number | undefined {
  const match = /^([A-Z]+)\d+$/u.exec(reference)
  if (match === null) return undefined
  let index = 0
  for (const letter of match[1] ?? '') index = index * 26 + (letter.charCodeAt(0) - 64)
  return index - 1
}

interface SheetRef {
  readonly name: string
  readonly part: string
}

function listSheets(files: ReadonlyMap<string, Uint8Array>, decoder: TextDecoder): SheetRef[] {
  const workbook = files.get('xl/workbook.xml')
  if (workbook === undefined) throw new ValidationError('path', 'is not a workbook: xl/workbook.xml is missing')
  const relsBytes = files.get('xl/_rels/workbook.xml.rels')
  const rels = new Map(parseOoxmlRelationships(relsBytes === undefined ? '' : decoder.decode(relsBytes)).map(rel => [rel.id, rel.target]))
  const sheets: SheetRef[] = []
  for (const match of decoder.decode(workbook).matchAll(/<sheet\b([^>]*?)\/?>/gu)) {
    const attributes = parseXmlAttributes(match[1] ?? '')
    const name = attributes['name']
    const target = attributes['r:id'] === undefined ? undefined : rels.get(attributes['r:id'])
    if (name === undefined || target === undefined) continue
    sheets.push({ name: decodeXml(name), part: joinOfficePartName('xl', target) })
  }
  if (sheets.length === 0) throw new ValidationError('path', 'is not a workbook: no sheets were found')
  return sheets
}

function parseSheetRows(xml: string, shared: readonly string[], maxRows: number): { rows: XlsxCell[][]; truncated: boolean } {
  const rows: XlsxCell[][] = []
  let nextRow = 1
  for (const rowMatch of xml.matchAll(/<row\b([^>]*?)(?:\/>|>([\s\S]*?)<\/row>)/gu)) {
    const attributes = parseXmlAttributes(rowMatch[1] ?? '')
    const rowNumber = attributes['r'] === undefined ? nextRow : Number.parseInt(attributes['r'], 10)
    if (!Number.isInteger(rowNumber) || rowNumber < 1) continue
    nextRow = rowNumber + 1
    if (rowNumber > maxRows) return { rows, truncated: true }
    const cells: XlsxCell[] = []
    let cursor = 0
    for (const cellMatch of (rowMatch[2] ?? '').matchAll(/<c\b([^>]*?)(?:\/>|>([\s\S]*?)<\/c>)/gu)) {
      const cellAttributes = parseXmlAttributes(cellMatch[1] ?? '')
      const column = cellAttributes['r'] === undefined ? cursor : columnIndex(cellAttributes['r']) ?? cursor
      if (column >= MAX_COLUMNS) continue
      cursor = column + 1
      while (cells.length < column) cells.push(null)
      cells[column] = cellValue(cellAttributes['t'], cellMatch[2], shared)
    }
    while (rows.length < rowNumber - 1) rows.push([])
    rows[rowNumber - 1] = cells
  }
  return { rows, truncated: false }
}

function cellValue(type: string | undefined, inner: string | undefined, shared: readonly string[]): XlsxCell {
  if (inner === undefined) return null
  if (type === 'inlineStr') return textRuns(inner)
  const raw = /<v\b[^>]*?>([\s\S]*?)<\/v>/u.exec(inner)?.[1]
  if (raw === undefined) return null
  const value = decodeXml(raw)
  switch (type) {
    case 's': return shared[Number.parseInt(value, 10)] ?? null
    case 'b': return value === '1' || value.toLowerCase() === 'true'
    case 'str':
    case 'e':
      return value
    default: {
      const number = Number(value)
      return value.trim() === '' || Number.isNaN(number) ? value : number
    }
  }
}

async function readWorkbook(inputs: JsonObject, options: XlsxToolOptions): Promise<JsonObject> {
  const path = requiredString(inputs, 'path')
  const wanted = optionalString(inputs, 'sheet')
  const maxRows = optionalInteger(inputs, 'max_rows', DEFAULT_READ_ROWS)
  if (maxRows < 1 || maxRows > MAX_READ_ROWS) throw new ValidationError('max_rows', `must be between 1 and ${MAX_READ_ROWS}`, maxRows)
  const loaded = await loadSheet(path, wanted, maxRows, options)
  return {
    sheet: loaded.sheet,
    sheets: loaded.sheets,
    row_count: loaded.rows.length,
    rows: loaded.rows,
    truncated: loaded.truncated,
  }
}

interface LoadedSheet {
  readonly sheet: string
  readonly sheets: string[]
  readonly rows: XlsxCell[][]
  readonly truncated: boolean
}

/** One sheet of a workbook as rectangular rows: the first sheet unless `wanted` names one. */
async function loadSheet(path: string, wanted: string | undefined, maxRows: number, options: XlsxToolOptions): Promise<LoadedSheet> {
  const target = await options.resolvePath(path)
  let size: number
  try {
    const info = await stat(target)
    if (!info.isFile()) throw new ValidationError('path', 'is not a file', path)
    size = info.size
  } catch (error) {
    if (typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT') {
      throw new ValidationError('path', 'does not exist', path)
    }
    throw error
  }
  if (size > MAX_READ_BYTES) throw new ValidationError('path', `is larger than ${MAX_READ_BYTES} bytes`, path)
  let files: ReadonlyMap<string, Uint8Array>
  try {
    files = readOfficeZip(new Uint8Array(await readFile(target)))
  } catch (error) {
    if (error instanceof UnsupportedOfficeZipError) {
      throw new ValidationError('path', `is not a readable .xlsx workbook: ${error.message}`, path)
    }
    throw error
  }
  const decoder = new TextDecoder()
  const sheets = listSheets(files, decoder)
  const chosen = wanted === undefined
    ? sheets[0]
    : sheets.find(sheet => sheet.name === wanted) ?? sheets.find(sheet => sheet.name.toLowerCase() === wanted.toLowerCase())
  if (chosen === undefined) {
    throw new ValidationError('sheet', `not found; available sheets: ${sheets.map(sheet => sheet.name).join(', ')}`, wanted)
  }
  const sheetBytes = files.get(chosen.part)
  if (sheetBytes === undefined) throw new ValidationError('path', `is damaged: ${chosen.part} is missing`, path)
  const sharedBytes = files.get('xl/sharedStrings.xml')
  const shared = sharedStringsOf(sharedBytes === undefined ? undefined : decoder.decode(sharedBytes))
  const parsed = parseSheetRows(decoder.decode(sheetBytes), shared, maxRows)
  const truncated = parsed.truncated
  // Empty cells are not stored in the file, so a trailing empty cell would otherwise
  // vanish on read-back. Pad non-empty rows to the widest row so the table stays rectangular.
  const width = parsed.rows.reduce((widest, row) => Math.max(widest, row.length), 0)
  const rows = parsed.rows.map(row => {
    if (row.length === 0 || row.length >= width) return row
    return [...row, ...Array.from<null>({ length: width - row.length }).fill(null)]
  })
  return { sheet: chosen.name, sheets: sheets.map(sheet => sheet.name), rows, truncated }
}

// --- converting ------------------------------------------------------------

/** A plain decimal: no leading zeros (so IDs such as 007 stay text), no exponent, no thousands separators. */
const PLAIN_NUMBER = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$/u
const MAX_HEADER_COLUMNS = 50
const MAX_HEADER_CHARS = 80

function csvText(cell: XlsxCell): string {
  if (cell === null) return ''
  if (typeof cell === 'boolean') return cell ? 'TRUE' : 'FALSE'
  return String(cell)
}

function csvCellValue(text: string): XlsxCell {
  if (text === '') return null
  // At most 15 significant digits, so the stored double shows the same number.
  if (PLAIN_NUMBER.test(text) && text.replace(/[-.]/gu, '').replace(/^0+/u, '').length <= 15) return Number(text)
  return text
}

/** A sheet name from a CSV file name: forbidden characters dropped, at most 31 characters. */
function sheetNameFor(path: string): string {
  const stem = basename(path, extname(path)).replace(new RegExp(FORBIDDEN_SHEET_CHARS.source, 'gu'), '').replace(new RegExp(INVALID_XML_CHARS.source, 'gu'), '').replace(/^'+|'+$/gu, '').trim()
  return stem.slice(0, MAX_SHEET_NAME) || 'Sheet1'
}

function headerOf(rows: readonly (readonly XlsxCell[])[]): string[] {
  return (rows[0] ?? []).slice(0, MAX_HEADER_COLUMNS).map(cell => csvText(cell).slice(0, MAX_HEADER_CHARS))
}

async function convertWorkbook(inputs: JsonObject, options: XlsxToolOptions): Promise<JsonObject> {
  const path = requiredString(inputs, 'path')
  const output = requiredString(inputs, 'output_path')
  const sheet = optionalString(inputs, 'sheet')
  const overwrite = optionalBoolean(inputs, 'overwrite', false)
  const from = extname(path).toLowerCase()
  const to = extname(output).toLowerCase()
  if (from === '.xlsx' && to === '.csv') {
    const loaded = await loadSheet(path, sheet, MAX_ROWS, options)
    const text = stringifyCsv(loaded.rows.map(row => row.map(csvText)), ',')
    await writeOutput(output, 'output_path', text, overwrite, options)
    return { success: true, path, output_path: output, sheet: loaded.sheet, sheets: loaded.sheets, rows: loaded.rows.length, columns: loaded.rows[0]?.length ?? 0, header: headerOf(loaded.rows) }
  }
  if (from === '.csv' && to === '.xlsx') {
    const target = await options.resolvePath(path)
    let size: number
    try {
      size = (await stat(target)).size
    } catch (error) {
      if (typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT') throw new ValidationError('path', 'does not exist', path)
      throw error
    }
    if (size > MAX_READ_BYTES) throw new ValidationError('path', `is larger than ${MAX_READ_BYTES} bytes`, path)
    const text = (await readFile(target, 'utf8')).replace(/^\uFEFF/u, '')
    const table = parseCsv(text, ',')
    const name = sheet ?? sheetNameFor(path)
    validateSheetName(name, 'sheet')
    const rows = parseSheets({ sheets: [{ name, rows: table.map(row => row.map(csvCellValue)) }] })
    await writeOutput(output, 'output_path', buildWorkbook(rows), overwrite, options)
    return { success: true, path, output_path: output, sheet: name, rows: table.length, columns: table.reduce((widest, row) => Math.max(widest, row.length), 0), header: headerOf(rows[0]?.rows ?? []) }
  }
  throw new ValidationError('output_path', 'must convert .xlsx to .csv or .csv to .xlsx', `${path} -> ${output}`)
}
