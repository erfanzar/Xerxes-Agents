// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { posix } from 'node:path'

import { readOfficeZip } from '../../src/skills/powerpoint/zip.js'

/**
 * Minimal xlsx cell reader for grading. It returns each cell's stored value: a formula cell
 * yields its cached `<v>`, which only exists when the producing tool computed it.
 */
export type SheetCell = string | number | boolean

export interface WorkbookSheet {
  readonly name: string
  readonly cells: ReadonlyMap<string, SheetCell>
}

export interface WorkbookCells {
  readonly sheets: readonly WorkbookSheet[]
}

const decoder = new TextDecoder()
const MAX_RANGE_CELLS = 100_000

export function readWorkbookCells(bytes: Uint8Array): WorkbookCells {
  const parts = readOfficeZip(bytes)
  const text = (name: string): string | undefined => {
    const data = parts.get(name)
    return data === undefined ? undefined : decoder.decode(data)
  }
  const workbook = text('xl/workbook.xml')
  if (workbook === undefined) throw new Error('not an xlsx workbook: xl/workbook.xml is missing')

  const relationships = new Map<string, string>()
  for (const match of (text('xl/_rels/workbook.xml.rels') ?? '').matchAll(/<Relationship\b([^>]*?)\/?>/g)) {
    const attributes = match[1] ?? ''
    const id = attribute(attributes, 'Id')
    const target = attribute(attributes, 'Target')
    if (id !== undefined && target !== undefined) relationships.set(id, resolvePart(target))
  }

  const shared = parseSharedStrings(text('xl/sharedStrings.xml'))
  const sheets: WorkbookSheet[] = []
  for (const match of workbook.matchAll(/<sheet\b([^>]*?)\/?>/g)) {
    const attributes = match[1] ?? ''
    const name = attribute(attributes, 'name')
    const relationshipId = attribute(attributes, 'r:id')
    if (name === undefined || relationshipId === undefined) continue
    const part = relationships.get(relationshipId)
    const xml = part === undefined ? undefined : text(part)
    if (xml === undefined) continue
    sheets.push({ cells: parseSheet(xml, shared), name: decodeEntities(name) })
  }
  return { sheets }
}

function resolvePart(target: string): string {
  return target.startsWith('/') ? target.slice(1) : posix.normalize(`xl/${target}`)
}

function attribute(attributes: string, name: string): string | undefined {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = new RegExp(`(?:^|\\s)${escaped}=(["'])(.*?)\\1`).exec(attributes)
  return match?.[2]
}

function parseSharedStrings(xml: string | undefined): string[] {
  if (xml === undefined) return []
  const strings: string[] = []
  for (const match of xml.matchAll(/<si\b[^>]*?(?:\/>|>([\s\S]*?)<\/si>)/g)) {
    strings.push(joinText(match[1] ?? ''))
  }
  return strings
}

function joinText(xml: string): string {
  const withoutPhonetic = xml.replace(/<rPh\b[\s\S]*?<\/rPh>/g, '')
  let value = ''
  for (const match of withoutPhonetic.matchAll(/<t\b[^>]*>([\s\S]*?)<\/t>/g)) value += match[1] ?? ''
  return decodeEntities(value)
}

function parseSheet(xml: string, shared: readonly string[]): Map<string, SheetCell> {
  const cells = new Map<string, SheetCell>()
  for (const match of xml.matchAll(/<c\b([^>]*?)(?:\/>|>([\s\S]*?)<\/c>)/g)) {
    const attributes = match[1] ?? ''
    const body = match[2]
    const ref = attribute(attributes, 'r')
    if (ref === undefined || body === undefined) continue
    const type = attribute(attributes, 't')
    let value: SheetCell | undefined
    if (type === 'inlineStr') {
      value = joinText(/<is\b[^>]*>([\s\S]*?)<\/is>/.exec(body)?.[1] ?? '')
    } else {
      const raw = /<v\b[^>]*>([\s\S]*?)<\/v>/.exec(body)?.[1]
      if (raw === undefined) continue
      if (type === 's') value = shared[Number(raw)]
      else if (type === 'b') value = raw.trim() === '1'
      else if (type === 'str' || type === 'e') value = decodeEntities(raw)
      else {
        const number = Number(raw)
        value = Number.isNaN(number) ? decodeEntities(raw) : number
      }
    }
    if (value === undefined || value === '') continue
    cells.set(normalizeRef(ref), value)
  }
  return cells
}

function decodeEntities(value: string): string {
  return value.replace(/&(#x[0-9a-f]+|#\d+|lt|gt|amp|quot|apos);/gi, (whole, entity: string) => {
    const lower = entity.toLowerCase()
    if (lower === 'lt') return '<'
    if (lower === 'gt') return '>'
    if (lower === 'amp') return '&'
    if (lower === 'quot') return '"'
    if (lower === 'apos') return "'"
    const codePoint = lower.startsWith('#x') ? Number.parseInt(lower.slice(2), 16) : Number.parseInt(lower.slice(1), 10)
    if (!Number.isSafeInteger(codePoint) || codePoint < 0 || codePoint > 0x10ffff) return whole
    return String.fromCodePoint(codePoint)
  })
}

export function columnIndex(letters: string): number {
  let index = 0
  for (const character of letters.toUpperCase()) index = index * 26 + (character.charCodeAt(0) - 64)
  return index
}

export function columnLetters(index: number): string {
  let letters = ''
  for (let n = index; n > 0; n = Math.floor((n - 1) / 26)) letters = String.fromCharCode(65 + ((n - 1) % 26)) + letters
  return letters
}

export function normalizeRef(ref: string): string {
  return ref.replaceAll('$', '').toUpperCase()
}

/** `A1` or `B2:D5` as the list of cell references it covers. */
export function expandRange(range: string): string[] {
  const match = /^([A-Z]+)(\d+)(?::([A-Z]+)(\d+))?$/.exec(normalizeRef(range.trim()))
  if (match === null) throw new Error(`unsupported cell range: ${JSON.stringify(range)}`)
  const startColumn = columnIndex(match[1] ?? 'A')
  const startRow = Number(match[2])
  const endColumn = match[3] === undefined ? startColumn : columnIndex(match[3])
  const endRow = match[4] === undefined ? startRow : Number(match[4])
  const columns = [Math.min(startColumn, endColumn), Math.max(startColumn, endColumn)] as const
  const rows = [Math.min(startRow, endRow), Math.max(startRow, endRow)] as const
  if ((columns[1] - columns[0] + 1) * (rows[1] - rows[0] + 1) > MAX_RANGE_CELLS) {
    throw new Error(`cell range is too large (over ${MAX_RANGE_CELLS} cells): ${range}`)
  }
  const refs: string[] = []
  for (let row = rows[0]; row <= rows[1]; row += 1) {
    for (let column = columns[0]; column <= columns[1]; column += 1) refs.push(`${columnLetters(column)}${row}`)
  }
  return refs
}

export interface PositionedRefs {
  readonly sheet?: string
  readonly refs: readonly string[]
}

/** Parse a SpreadsheetBench answer position: `A1`, `B2:D5`, `'My Sheet'!A1:B3`, comma separated. */
export function parseAnswerPositions(spec: string): PositionedRefs[] {
  const parts: string[] = []
  let quoted = false
  let current = ''
  for (const character of spec) {
    if (character === "'") quoted = !quoted
    if (character === ',' && !quoted) {
      parts.push(current)
      current = ''
    } else {
      current += character
    }
  }
  parts.push(current)

  const positions: PositionedRefs[] = []
  for (const part of parts) {
    const trimmed = part.trim()
    if (!trimmed) continue
    const match = /^(?:'((?:[^']|'')+)'|([^!']+))!(.+)$/.exec(trimmed)
    if (match === null) {
      positions.push({ refs: expandRange(trimmed) })
      continue
    }
    const sheet = match[1] !== undefined ? match[1].replaceAll("''", "'") : (match[2] ?? '').trim()
    positions.push({ refs: expandRange(match[3] ?? ''), sheet })
  }
  return positions
}

/** Equality used for grading: tolerant for numbers, trimmed for strings, empty equals missing. */
export function cellValuesEqual(expected: SheetCell | undefined, actual: SheetCell | undefined): boolean {
  if (expected === undefined || expected === '') return actual === undefined || actual === ''
  if (actual === undefined) return false
  const numericPair = (typeof expected === 'number' || typeof expected === 'string')
    && (typeof actual === 'number' || typeof actual === 'string')
    && (typeof expected === 'number' || typeof actual === 'number')
  if (numericPair) {
    const left = typeof expected === 'number' ? expected : Number(expected.trim())
    const right = typeof actual === 'number' ? actual : Number(actual.trim())
    if (Number.isFinite(left) && Number.isFinite(right)) {
      return Math.abs(left - right) <= 1e-6 * Math.max(1, Math.abs(left))
    }
    return false
  }
  if (typeof expected === 'string' && typeof actual === 'string') return expected.trim() === actual.trim()
  return expected === actual
}

export interface CellComparison {
  readonly total: number
  readonly matched: number
  /** First few mismatches, e.g. `Sheet1!B3 expected 4, got 5`. */
  readonly mismatches: readonly string[]
}

function findSheet(workbook: WorkbookCells, name: string | undefined): WorkbookSheet | undefined {
  if (name === undefined) return workbook.sheets[0]
  const lower = name.toLowerCase()
  return workbook.sheets.find(sheet => sheet.name.toLowerCase() === lower)
}

/**
 * Compare `actual` against `expected` over the cells named by `position`, or over every
 * non-empty cell of the expected first sheet when no position is given.
 */
export function compareCells(expected: WorkbookCells, actual: WorkbookCells, position?: string): CellComparison {
  const targets: { sheetName: string | undefined; refs: readonly string[] }[] = []
  if (position === undefined || !position.trim()) {
    const first = expected.sheets[0]
    if (first !== undefined) targets.push({ refs: [...first.cells.keys()].sort(), sheetName: first.name })
  } else {
    for (const entry of parseAnswerPositions(position)) targets.push({ refs: entry.refs, sheetName: entry.sheet })
  }

  let total = 0
  let matched = 0
  const mismatches: string[] = []
  for (const target of targets) {
    const expectedSheet = findSheet(expected, target.sheetName)
    const actualSheet = findSheet(actual, target.sheetName ?? expectedSheet?.name)
    const label = target.sheetName ?? expectedSheet?.name ?? 'sheet'
    for (const ref of target.refs) {
      total += 1
      const want = expectedSheet?.cells.get(ref)
      if (actualSheet === undefined) {
        if (want === undefined) matched += 1
        else if (mismatches.length < 5) mismatches.push(`${label}!${ref} sheet "${label}" is missing from the output`)
        continue
      }
      const got = actualSheet.cells.get(ref)
      if (cellValuesEqual(want, got)) matched += 1
      else if (mismatches.length < 5) mismatches.push(`${label}!${ref} expected ${String(want ?? '(empty)')}, got ${String(got ?? '(empty)')}`)
    }
  }
  return { matched, mismatches, total }
}
