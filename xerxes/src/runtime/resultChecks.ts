// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { extname, isAbsolute, resolve } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { readOfficeZip } from '../skills/powerpoint/zip.js'

/** Environment variable that selects the result-check mode. */
export const RESULT_CHECKS_ENV = 'XERXES_RESULT_CHECKS'

/** `warn` tells the user only; `enforce` lets the caller send findings back to the model. */
export type ChecksMode = 'off' | 'warn' | 'enforce'

export type CheckCode =
  | 'xlsx_invalid'
  | 'xlsx_is_text'
  | 'csv_ragged'
  | 'csv_bad_date'
  | 'input_modified'
  | 'unverified_claim'

export interface CheckFinding {
  readonly code: CheckCode
  readonly severity: 'error' | 'warn'
  readonly path?: string
  readonly message: string
}

export interface ToolRunSummary {
  readonly name: string
  readonly command?: string
  readonly ok: boolean
}

export interface ResultCheckInput {
  readonly workspaceRoot: string
  readonly changedFiles: readonly string[]
  /** Hashes taken before the run with {@link snapshotInputs}. */
  readonly inputSnapshot?: ReadonlyMap<string, string>
  readonly finalText?: string
  readonly toolRuns?: readonly ToolRunSummary[]
}

/** Resolve `runtime.checks` and `XERXES_RESULT_CHECKS` (env wins). Default `warn`. */
export function parseChecksConfig(
  runtime: Record<string, unknown>,
  env: Record<string, string | undefined>,
): ChecksMode {
  const fromEnv = env[RESULT_CHECKS_ENV]?.trim()
  if (fromEnv) return normalizeMode(fromEnv, RESULT_CHECKS_ENV)
  const fromFile = runtime['checks']
  if (fromFile === undefined || fromFile === null) return 'warn'
  return normalizeMode(fromFile, 'checks')
}

function normalizeMode(value: unknown, field: string): ChecksMode {
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase()
    if (normalized === 'off' || normalized === 'warn' || normalized === 'enforce') return normalized
  }
  throw new ValidationError(field, "must be 'off', 'warn' or 'enforce'", value)
}

/** sha256 of each existing file. A path that does not exist is left out of the map. */
export async function snapshotInputs(paths: readonly string[]): Promise<Map<string, string>> {
  const snapshot = new Map<string, string>()
  for (const path of paths) {
    const bytes = await readBytes(path)
    if (bytes !== undefined) snapshot.set(path, sha256(bytes))
  }
  return snapshot
}

/** Check the files a run changed and what it said about them. Never throws for a bad file. */
export async function runResultChecks(input: ResultCheckInput): Promise<CheckFinding[]> {
  const findings: CheckFinding[] = []
  const seen = new Set<string>()
  for (const file of input.changedFiles) {
    const absolute = absolutePath(input.workspaceRoot, file)
    if (seen.has(absolute)) continue
    seen.add(absolute)
    const extension = extname(file).toLowerCase()
    if (extension === '.xlsx') findings.push(...await checkXlsx(absolute, file))
    else if (extension === '.csv') findings.push(...await checkCsv(absolute, file))
  }
  if (input.inputSnapshot !== undefined) {
    for (const [path, before] of input.inputSnapshot) {
      const bytes = await readBytes(absolutePath(input.workspaceRoot, path))
      if (bytes === undefined) {
        findings.push({ code: 'input_modified', severity: 'error', path, message: 'An input file was deleted.' })
      } else if (sha256(bytes) !== before) {
        findings.push({ code: 'input_modified', severity: 'error', path, message: 'An input file was changed; inputs should stay read-only.' })
      }
    }
  }
  if (input.finalText !== undefined) {
    const claim = unverifiedClaim(input.finalText, input.toolRuns ?? [])
    if (claim !== undefined) findings.push(claim)
  }
  return findings
}

/** Short plain-text report, one line per finding. */
export function renderFindings(findings: readonly CheckFinding[]): string {
  if (findings.length === 0) return 'Result checks: no problems found.'
  const lines = findings.map(finding =>
    `- [${finding.severity}] ${finding.code}${finding.path === undefined ? '' : ` ${finding.path}`}: ${finding.message}`)
  return `Result checks found ${findings.length} problem${findings.length === 1 ? '' : 's'}:\n${lines.join('\n')}`
}

// --- files -----------------------------------------------------------------

function absolutePath(root: string, path: string): string {
  return isAbsolute(path) ? path : resolve(root, path)
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex')
}

async function readBytes(path: string): Promise<Uint8Array | undefined> {
  try {
    return new Uint8Array(await readFile(path))
  } catch (error) {
    if (typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT') return undefined
    throw error
  }
}

// --- xlsx ------------------------------------------------------------------

function looksLikeText(bytes: Uint8Array): boolean {
  const sample = bytes.subarray(0, 8192)
  if (sample.includes(0)) return false
  try {
    new TextDecoder('utf-8', { fatal: true }).decode(sample)
    return true
  } catch {
    return false
  }
}

async function checkXlsx(absolute: string, display: string): Promise<CheckFinding[]> {
  const bytes = await readBytes(absolute)
  if (bytes === undefined) return []
  const fail = (code: 'xlsx_invalid' | 'xlsx_is_text', message: string): CheckFinding[] =>
    [{ code, severity: 'error', path: display, message }]
  if (bytes.byteLength === 0) return fail('xlsx_invalid', 'The file is empty.')
  const isZip = bytes.byteLength >= 4 && bytes[0] === 0x50 && bytes[1] === 0x4b && bytes[2] === 0x03 && bytes[3] === 0x04
  if (!isZip) {
    return looksLikeText(bytes)
      ? fail('xlsx_is_text', 'The file holds plain text (such as CSV), not a workbook. Write it with XlsxWriter.')
      : fail('xlsx_invalid', 'The file is not a ZIP archive, so it is not a workbook.')
  }
  let names: string[]
  try {
    names = [...readOfficeZip(bytes).keys()]
  } catch (error) {
    return fail('xlsx_invalid', `The ZIP archive could not be read: ${error instanceof Error ? error.message : String(error)}`)
  }
  const missing: string[] = []
  if (!names.includes('[Content_Types].xml')) missing.push('[Content_Types].xml')
  if (!names.includes('xl/workbook.xml')) missing.push('xl/workbook.xml')
  if (!names.some(name => /^xl\/worksheets\/sheet[^/]*\.xml$/iu.test(name))) missing.push('xl/worksheets/sheet*.xml')
  return missing.length === 0 ? [] : fail('xlsx_invalid', `The workbook is missing: ${missing.join(', ')}.`)
}

// --- csv -------------------------------------------------------------------

interface CsvRecord {
  readonly line: number
  readonly cells: string[]
}

interface ParsedCsv {
  readonly records: CsvRecord[]
  readonly error?: string
}

/** Tolerant RFC-4180 reader that remembers the line each record starts on. */
function parseCsvRecords(input: string): ParsedCsv {
  const text = input.charCodeAt(0) === 0xfeff ? input.slice(1) : input
  const records: CsvRecord[] = []
  let cells: string[] = []
  let field = ''
  let quoted = false
  let fieldWasQuoted = false
  let line = 1
  let recordLine = 1
  let quoteLine = 1
  const endRecord = (): void => {
    cells.push(field)
    const blank = cells.length === 1 && cells[0] === '' && !fieldWasQuoted
    if (!blank) records.push({ line: recordLine, cells })
    cells = []
    field = ''
    fieldWasQuoted = false
  }
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index] ?? ''
    if (quoted) {
      if (character === '"') {
        if (text[index + 1] === '"') {
          field += '"'
          index += 1
        } else {
          quoted = false
        }
      } else {
        if (character === '\n') line += 1
        field += character
      }
      continue
    }
    if (character === '"' && field === '') {
      quoted = true
      fieldWasQuoted = true
      quoteLine = line
    } else if (character === ',') {
      cells.push(field)
      field = ''
      fieldWasQuoted = false
    } else if (character === '\n' || character === '\r') {
      if (character === '\r' && text[index + 1] === '\n') index += 1
      endRecord()
      line += 1
      recordLine = line
    } else {
      field += character
    }
  }
  if (quoted) return { records, error: `A quoted field that starts on line ${quoteLine} is never closed.` }
  if (field !== '' || cells.length > 0 || fieldWasQuoted) endRecord()
  return { records }
}

const MAX_LISTED = 5

function listMore(shown: number, total: number): string {
  return total > shown ? ` (and ${total - shown} more)` : ''
}

async function checkCsv(absolute: string, display: string): Promise<CheckFinding[]> {
  const bytes = await readBytes(absolute)
  if (bytes === undefined) return []
  const parsed = parseCsvRecords(new TextDecoder().decode(bytes))
  const findings: CheckFinding[] = []
  if (parsed.error !== undefined) {
    findings.push({ code: 'csv_ragged', severity: 'error', path: display, message: parsed.error })
  }
  const [header, ...rows] = parsed.records
  if (header === undefined) return findings
  const expected = header.cells.length
  const ragged = rows.filter(row => row.cells.length !== expected)
  if (ragged.length > 0) {
    const shown = ragged.slice(0, MAX_LISTED).map(row => `line ${row.line} has ${row.cells.length}`).join(', ')
    findings.push({
      code: 'csv_ragged',
      severity: 'error',
      path: display,
      message: `${ragged.length} row${ragged.length === 1 ? '' : 's'} do not have the header's ${expected} columns: ${shown}${listMore(MAX_LISTED, ragged.length)}.`,
    })
  }
  for (let column = 0; column < expected; column += 1) {
    const finding = checkDateColumn(header.cells[column] ?? '', column, rows, display)
    if (finding !== undefined) findings.push(finding)
  }
  return findings
}

const ISO_DATE = /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$/u
const ISO_SHAPE = /^\d{4}-\d{1,2}-\d{1,2}(?:[T ]|$)/u

function daysInMonth(year: number, month: number): number {
  return new Date(Date.UTC(year, month, 0)).getUTCDate()
}

function validDay(year: number, month: number, day: number): boolean {
  return month >= 1 && month <= 12 && day >= 1 && day <= daysInMonth(year, month)
}

function validIso(value: string): boolean {
  const match = ISO_DATE.exec(value)
  if (match === null) return false
  const [year, month, day] = [Number(match[1]), Number(match[2]), Number(match[3])]
  if (!validDay(year, month, day)) return false
  if (match[4] !== undefined && (Number(match[4]) > 23 || Number(match[5]) > 59)) return false
  return match[6] === undefined || Number(match[6]) <= 60
}

/** day/month/year or month/day/year with a four-digit year at either end. */
function validDayMonthYear(value: string): boolean {
  const match = /^(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})$/u.exec(value)
  if (match === null) return false
  const first = match[1] ?? ''
  const last = match[3] ?? ''
  const middle = Number(match[2])
  if (last.length === 4) {
    const [a, year] = [Number(first), Number(last)]
    return validDay(year, middle, a) || validDay(year, a, middle)
  }
  if (first.length === 4) return validDay(Number(first), middle, Number(last)) || validDay(Number(first), Number(last), middle)
  return false
}

function checkDateColumn(name: string, column: number, rows: readonly CsvRecord[], display: string): CheckFinding | undefined {
  const values = rows
    .map(row => ({ line: row.line, value: (row.cells[column] ?? '').trim() }))
    .filter(entry => entry.value !== '')
  if (values.length === 0) return undefined
  const isoShaped = values.filter(entry => ISO_SHAPE.test(entry.value)).length
  const mostlyIso = isoShaped / values.length >= 0.6
  if (!name.toLowerCase().includes('date') && !mostlyIso) return undefined
  const bad = values.filter(entry => mostlyIso
    ? !validIso(entry.value)
    : !validDayMonthYear(entry.value) && Number.isNaN(Date.parse(entry.value)))
  if (bad.length === 0) return undefined
  const shown = bad.slice(0, MAX_LISTED)
    .map(entry => `line ${entry.line} "${entry.value.length > 30 ? `${entry.value.slice(0, 29)}…` : entry.value}"`)
    .join(', ')
  return {
    code: 'csv_bad_date',
    severity: 'warn',
    path: display,
    message: `Column "${name}" (number ${column + 1}) has ${bad.length} value${bad.length === 1 ? '' : 's'} that are not valid dates: ${shown}${listMore(MAX_LISTED, bad.length)}.`,
  }
}

// --- claims ----------------------------------------------------------------

const CLAIM_PATTERNS: readonly RegExp[] = [
  /\bvalidat(?:ed|ion (?:passed|succeeded|completed?|is ok|ok))\b/iu,
  /\bverified\b/iu,
  /\bverification (?:passed|succeeded|completed?)\b/iu,
  /\b(?:all )?tests? (?:all )?(?:pass(?:ed|es)?|succeed(?:ed)?)\b/iu,
  /\b(?:all )?checks? (?:all )?(?:pass(?:ed)?|succeeded)\b/iu,
  /\b(?:i|we) (?:have )?(?:ran|run) (?:the |all )?(?:tests?|validation|validator|checks?)\b/iu,
]

const NEGATION = /\b(?:not|never|no|without|unable to|cannot|can't|couldn't|didn't|did not|n't)\b[^.\n]{0,25}$/iu

const CHECK_COMMAND = /\b(?:bun\s+(?:run\s+)?test|vitest|jest|pytest|mocha|go\s+test|cargo\s+test|(?:npm|pnpm|yarn)\s+(?:run\s+)?test|tsc|eslint|ruff|mypy|lint|validate|validator|verify|check)\b/iu

const WRITE_TOOLS: ReadonlySet<string> = new Set([
  'WriteFile', 'FileEditTool', 'XlsxWriter', 'XlsxConvert', 'AppendFile', 'MultiEdit', 'Write', 'Edit', 'NotebookEdit', 'apply_patch',
])
const READ_BACK_TOOLS: ReadonlySet<string> = new Set(['XlsxReader', 'CSVProcessor'])

function isWrite(run: ToolRunSummary): boolean {
  return WRITE_TOOLS.has(run.name) || (run.name === 'CSVProcessor' && /write/iu.test(run.command ?? ''))
}

function isVerification(run: ToolRunSummary): boolean {
  if (READ_BACK_TOOLS.has(run.name)) return !isWrite(run)
  return run.command !== undefined && CHECK_COMMAND.test(run.command)
}

function claimedVerification(text: string): string | undefined {
  for (const pattern of CLAIM_PATTERNS) {
    const global = new RegExp(pattern.source, 'giu')
    for (const match of text.matchAll(global)) {
      const before = text.slice(Math.max(0, match.index - 40), match.index)
      if (!NEGATION.test(before)) return match[0]
    }
  }
  return undefined
}

function unverifiedClaim(finalText: string, runs: readonly ToolRunSummary[]): CheckFinding | undefined {
  const claim = claimedVerification(finalText)
  if (claim === undefined) return undefined
  let lastWrite = -1
  runs.forEach((run, index) => {
    if (run.ok && isWrite(run)) lastWrite = index
  })
  const evidence = runs.some((run, index) => index > lastWrite && run.ok && isVerification(run))
  if (evidence) return undefined
  return {
    code: 'unverified_claim',
    severity: 'warn',
    message: `The reply says "${claim}", but no successful check (a test run, a validator, or reading the file back) ran after the last write.`,
  }
}
