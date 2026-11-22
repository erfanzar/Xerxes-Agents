// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { realpathSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { basename, dirname, extname, isAbsolute, resolve } from 'node:path'

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
  | 'conflict_markers'
  | 'syntax_error'
  | 'input_modified'
  | 'unverified_claim'

/** Code-level check codes subject to pre-turn baseline error filtering. */
export const CODE_CHECK_CODES: ReadonlySet<CheckCode> = new Set<CheckCode>([
  'conflict_markers',
  'syntax_error',
])

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
  /** Pre-turn fingerprints from {@link captureCodeCheckBaseline} so pre-existing code errors are not blamed on this turn. */
  readonly baselineFindings?: ReadonlySet<string>
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

/**
 * Stable identity for a code-level finding on `workspaceRoot`.
 *
 * Line numbers are normalized so an untouched pre-existing syntax or conflict
 * issue that merely shifted down when lines were inserted above it is still
 * recognized as pre-existing, while any new error code, column, message, or
 * offending line snippet produces a distinct fingerprint.
 */
export function findingFingerprint(workspaceRoot: string, finding: CheckFinding): string {
  const canonical = finding.path === undefined ? '' : canonicalPath(absolutePath(workspaceRoot, finding.path))
  const normalizedMessage = finding.message.replace(/\bline \d+\b/gu, 'line #')
  return `${canonical}\u001f${finding.code}\u001f${normalizedMessage}`
}

function canonicalPath(target: string): string {
  const abs = resolve(target)
  try {
    return realpathSync(abs)
  } catch {
    const parent = dirname(abs)
    if (parent === abs) return abs
    return resolve(canonicalPath(parent), basename(abs))
  }
}

/**
 * Capture pre-turn code-check fingerprints (`conflict_markers`, `syntax_error`)
 * over `candidatePaths` so pre-existing issues in those files are not blamed on
 * the turn unless a new error fingerprint appears.
 */
export async function captureCodeCheckBaseline(
  workspaceRoot: string,
  candidatePaths: readonly string[],
): Promise<Set<string>> {
  const fingerprints = new Set<string>()
  const seen = new Set<string>()
  for (const candidate of candidatePaths) {
    if (!candidate) continue
    const absolute = absolutePath(workspaceRoot, candidate)
    if (seen.has(absolute)) continue
    seen.add(absolute)
    const findings = await checkCodeFile(absolute, candidate)
    for (const finding of findings) {
      fingerprints.add(findingFingerprint(workspaceRoot, finding))
    }
  }
  return fingerprints
}

/**
 * Extract target file paths from a V4A (`*** Begin Patch`) or unified diff
 * (`--- a/...`, `+++ b/...`, `diff --git`) patch payload.
 */
export function extractPatchTargetPaths(patchText: string): string[] {
  const paths: string[] = []
  const seen = new Set<string>()
  const add = (raw: string): void => {
    const cleaned = cleanPatchPath(raw)
    if (!cleaned || cleaned === '/dev/null' || seen.has(cleaned)) return
    seen.add(cleaned)
    paths.push(cleaned)
  }
  for (const rawLine of patchText.split(/\r?\n/u)) {
    const v4aMatch = /^\*\*\*\s+(?:Update File|Add File|Delete File|Move to):\s+(.+?)\s*$/u.exec(rawLine)
    if (v4aMatch?.[1] !== undefined) {
      add(v4aMatch[1])
      continue
    }
    const gitHeader = /^diff --git\s+(.+?)\s+(.+?)\s*$/u.exec(rawLine)
    if (gitHeader !== null) {
      add(stripGitDiffPrefix(gitHeader[1] ?? '', 'a/'))
      add(stripGitDiffPrefix(gitHeader[2] ?? '', 'b/'))
      continue
    }
    const renameMatch = /^rename (?:from|to)\s+(.+?)\s*$/u.exec(rawLine)
    if (renameMatch?.[1] !== undefined) {
      add(renameMatch[1])
      continue
    }
    if (rawLine.startsWith('--- ') && !rawLine.startsWith('--- /dev/null')) {
      const target = rawLine.slice(4).split('\t')[0] ?? ''
      add(stripGitDiffPrefix(target, 'a/'))
      continue
    }
    if (rawLine.startsWith('+++ ') && !rawLine.startsWith('+++ /dev/null')) {
      const target = rawLine.slice(4).split('\t')[0] ?? ''
      add(stripGitDiffPrefix(target, 'b/'))
    }
  }
  return paths
}

function cleanPatchPath(raw: string): string {
  const trimmed = raw.trim()
  if (trimmed.length >= 2 && ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'")))) {
    return trimmed.slice(1, -1).trim()
  }
  return trimmed
}

function stripGitDiffPrefix(raw: string, prefix: 'a/' | 'b/'): string {
  const cleaned = cleanPatchPath(raw)
  return cleaned.startsWith(prefix) ? cleaned.slice(prefix.length) : cleaned
}

/** Check the files a run changed and what it said about them. Never throws for a bad file. */
export async function runResultChecks(input: ResultCheckInput): Promise<CheckFinding[]> {
  const findings: CheckFinding[] = []
  const seen = new Set<string>()
  for (const file of input.changedFiles) {
    if (!file) continue
    const absolute = absolutePath(input.workspaceRoot, file)
    if (seen.has(absolute)) continue
    seen.add(absolute)
    const extension = extname(file).toLowerCase()
    if (extension === '.xlsx') {
      findings.push(...await checkXlsx(absolute, file))
      continue
    }
    if (extension === '.csv') {
      findings.push(...await checkCsv(absolute, file))
    }
    const codeFindings = await checkCodeFile(absolute, file)
    for (const finding of codeFindings) {
      if (input.baselineFindings?.has(findingFingerprint(input.workspaceRoot, finding))) continue
      findings.push(finding)
    }
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
  return isAbsolute(path) ? resolve(path) : resolve(root, path)
}

function sha256(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex')
}

async function readBytes(path: string): Promise<Uint8Array | undefined> {
  try {
    return new Uint8Array(await readFile(path))
  } catch (error) {
    if (
      typeof error === 'object'
      && error !== null
      && 'code' in error
      && (error.code === 'ENOENT' || error.code === 'EISDIR' || error.code === 'ENOTDIR')
    ) {
      return undefined
    }
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

// --- code & syntax ---------------------------------------------------------

type JsLikeLoader = 'ts' | 'tsx' | 'js' | 'jsx'

const JS_LIKE_EXTENSIONS: Readonly<Record<string, JsLikeLoader>> = {
  '.ts': 'ts',
  '.mts': 'ts',
  '.cts': 'ts',
  '.tsx': 'tsx',
  '.js': 'js',
  '.mjs': 'js',
  '.cjs': 'js',
  '.jsx': 'jsx',
}

const STANDARD_TRANSPILERS = new Map<JsLikeLoader, Bun.Transpiler>()
const LEGACY_DECORATOR_TRANSPILERS = new Map<JsLikeLoader, Bun.Transpiler>()

function getTranspiler(loader: JsLikeLoader, experimentalDecorators: boolean): Bun.Transpiler {
  const cache = experimentalDecorators ? LEGACY_DECORATOR_TRANSPILERS : STANDARD_TRANSPILERS
  const existing = cache.get(loader)
  if (existing !== undefined) return existing
  const created = experimentalDecorators
    ? new Bun.Transpiler({ loader, target: 'bun', tsconfig: '{"compilerOptions":{"experimentalDecorators":true}}' })
    : new Bun.Transpiler({ loader, target: 'bun' })
  cache.set(loader, created)
  return created
}

async function checkCodeFile(absolute: string, display: string): Promise<CheckFinding[]> {
  const bytes = await readBytes(absolute)
  if (bytes === undefined) return []
  if (bytes.byteLength > 0 && !looksLikeText(bytes)) return []
  const text = new TextDecoder('utf-8').decode(bytes)
  const extension = extname(display).toLowerCase()
  const conflictFindings = checkConflictMarkers(text, display, extension)
  if (conflictFindings.length > 0) return conflictFindings
  return checkCodeSyntax(text, display, extension)
}

interface ConflictMarkerHit {
  readonly line: number
  readonly kind: 'start' | 'sep' | 'base' | 'end'
  readonly preview: string
}

function clipMarker(line: string): string {
  const trimmed = line.trimEnd()
  return trimmed.length > 36 ? `${trimmed.slice(0, 35)}…` : trimmed
}

function checkConflictMarkers(text: string, display: string, extension: string): CheckFinding[] {
  if (!text) return []
  const isMarkdown = extension === '.md' || extension === '.mdx' || extension === '.markdown'
  const isProse = isMarkdown || extension === '.rst' || extension === '.txt' || extension === '.adoc'
  const rawLines = text.split('\n')
  const hits: ConflictMarkerHit[] = []
  let fenceChar: '`' | '~' | undefined
  let fenceLength = 0

  for (let index = 0; index < rawLines.length; index += 1) {
    const raw = rawLines[index] ?? ''
    const line = raw.endsWith('\r') ? raw.slice(0, -1) : raw
    if (isMarkdown) {
      if (fenceChar === undefined) {
        const openFence = /^ {0,3}(`{3,}|~{3,})/u.exec(line)
        if (openFence?.[1] !== undefined) {
          const ch = openFence[1][0]
          if (ch === '`' || ch === '~') {
            fenceChar = ch
            fenceLength = openFence[1].length
            continue
          }
        }
      } else {
        const closePattern = fenceChar === '`'
          ? new RegExp(`^ {0,3}\`{${fenceLength},}[ \\t]*$`, 'u')
          : new RegExp(`^ {0,3}~{${fenceLength},}[ \\t]*$`, 'u')
        if (closePattern.test(line)) {
          fenceChar = undefined
          fenceLength = 0
        }
        continue
      }
    }

    const lineNumber = index + 1
    if (/^<{7}(?: .*)?$/u.test(line)) {
      hits.push({ line: lineNumber, kind: 'start', preview: clipMarker(line) })
    } else if (/^={7}$/u.test(line)) {
      hits.push({ line: lineNumber, kind: 'sep', preview: '=======' })
    } else if (/^\|{7}(?: .*)?$/u.test(line)) {
      hits.push({ line: lineNumber, kind: 'base', preview: clipMarker(line) })
    } else if (/^>{7}(?: .*)?$/u.test(line)) {
      hits.push({ line: lineNumber, kind: 'end', preview: clipMarker(line) })
    }
  }

  if (hits.length === 0) return []
  const hasStartOrEnd = hits.some(hit => hit.kind === 'start' || hit.kind === 'end')
  if (isProse && !hasStartOrEnd) return []

  const shown = hits
    .slice(0, MAX_LISTED)
    .map(hit => `line ${hit.line} (${hit.preview})`)
    .join(', ')
  return [{
    code: 'conflict_markers',
    severity: 'error',
    path: display,
    message: `Unresolved Git merge conflict marker${hits.length === 1 ? '' : 's'} at ${shown}${listMore(MAX_LISTED, hits.length)}.`,
  }]
}

function checkCodeSyntax(text: string, display: string, extension: string): CheckFinding[] {
  if (extension === '.json') return checkJsonSyntax(text, display)
  const loader = JS_LIKE_EXTENSIONS[extension]
  if (loader === undefined) return []
  return checkTranspilerSyntax(text, display, loader)
}

interface ExtractedBuildPosition {
  readonly line: number
  readonly column: number
  readonly lineText?: string
}

function extractTranspilerFailure(error: unknown): { readonly reason: string; readonly position?: ExtractedBuildPosition } {
  const candidate = unwrapFirstBuildError(error)
  const rawMessage = candidate instanceof Error
    ? candidate.message
    : typeof candidate === 'object' && candidate !== null && 'message' in candidate && typeof candidate.message === 'string'
      ? candidate.message
      : String(candidate)
  const reason = rawMessage.replace(/\s+/gu, ' ').trim() || 'Invalid syntax'
  if (typeof candidate === 'object' && candidate !== null && 'position' in candidate) {
    const pos = candidate.position
    if (typeof pos === 'object' && pos !== null) {
      const record = pos as Record<string, unknown>
      const line = typeof record['line'] === 'number' && Number.isFinite(record['line']) ? Math.max(1, Math.trunc(record['line'])) : undefined
      const column = typeof record['column'] === 'number' && Number.isFinite(record['column']) ? Math.max(1, Math.trunc(record['column'])) : undefined
      const lineText = typeof record['lineText'] === 'string' ? record['lineText'].trim() : undefined
      if (line !== undefined && column !== undefined) {
        return {
          reason,
          position: {
            line,
            column,
            ...(lineText ? { lineText: clipSnippet(lineText) } : {}),
          },
        }
      }
    }
  }
  return { reason }
}

function unwrapFirstBuildError(error: unknown): unknown {
  if (typeof error === 'object' && error !== null && 'errors' in error && Array.isArray(error.errors) && error.errors.length > 0) {
    return error.errors[0]
  }
  return error
}

function clipSnippet(snippet: string): string {
  return snippet.length > 60 ? `${snippet.slice(0, 59)}…` : snippet
}

function checkTranspilerSyntax(text: string, display: string, loader: JsLikeLoader): CheckFinding[] {
  if (!text.trim()) return []
  try {
    const standard = getTranspiler(loader, false)
    standard.scan(text)
    standard.transformSync(text)
    return []
  } catch (primaryError) {
    if ((loader === 'ts' || loader === 'tsx') && text.includes('@')) {
      try {
        const legacyDecorators = getTranspiler(loader, true)
        legacyDecorators.scan(text)
        legacyDecorators.transformSync(text)
        return []
      } catch {
        // Report the primary syntax error below.
      }
    }
    const { reason, position } = extractTranspilerFailure(primaryError)
    const location = position === undefined ? '' : ` at line ${position.line}, column ${position.column}`
    const snippet = position?.lineText ? ` ("${position.lineText}")` : ''
    return [{
      code: 'syntax_error',
      severity: 'error',
      path: display,
      message: `Syntax error${location}: ${reason}${snippet}`,
    }]
  }
}

function isJsoncConfigPath(display: string): boolean {
  const normalized = display.replaceAll('\\', '/')
  const name = basename(normalized).toLowerCase()
  if (/^(?:ts|js)config(?:\.[^/]+)?\.json$/u.test(name) || name === 'devcontainer.json') return true
  return /(?:^|\/)\.vscode\/[^/]+\.json$/iu.test(normalized)
}

function checkJsonSyntax(text: string, display: string): CheckFinding[] {
  try {
    if (isJsoncConfigPath(display)) {
      Bun.JSONC.parse(text)
    } else {
      JSON.parse(text)
    }
    return []
  } catch (error) {
    const rawReason = error instanceof Error ? error.message : String(error)
    const reason = rawReason.replace(/\s+/gu, ' ').trim() || 'Invalid JSON'
    const { line, column } = locateJsonErrorLineColumn(text, rawReason)
    const offendingLine = (text.split('\n')[line - 1] ?? '').replace(/\r$/u, '').trim()
    const snippet = offendingLine ? ` ("${clipSnippet(offendingLine)}")` : ''
    return [{
      code: 'syntax_error',
      severity: 'error',
      path: display,
      message: `Syntax error at line ${line}, column ${column}: ${reason}${snippet}`,
    }]
  }
}

function locateJsonErrorLineColumn(text: string, errorMessage: string): { readonly line: number; readonly column: number } {
  const explicitLineCol = /\bline\s+(\d+)\s+column\s+(\d+)\b/iu.exec(errorMessage)
  if (explicitLineCol !== null) {
    return {
      line: Math.max(1, Number.parseInt(explicitLineCol[1] ?? '1', 10)),
      column: Math.max(1, Number.parseInt(explicitLineCol[2] ?? '1', 10)),
    }
  }
  const explicitPos = /\bposition\s+(\d+)\b/iu.exec(errorMessage)
  if (explicitPos !== null) {
    const parsedOffset = Number.parseInt(explicitPos[1] ?? '0', 10)
    if (Number.isFinite(parsedOffset)) return offsetToLineColumn(text, parsedOffset)
  }
  try {
    return offsetToLineColumn(text, findStrictJsonErrorOffset(text))
  } catch {
    return { line: 1, column: 1 }
  }
}

function offsetToLineColumn(text: string, offset: number): { readonly line: number; readonly column: number } {
  const clamped = Math.max(0, Math.min(offset, text.length))
  let line = 1
  let lastNewline = -1
  for (let index = 0; index < clamped; index += 1) {
    if (text.charCodeAt(index) === 10) {
      line += 1
      lastNewline = index
    }
  }
  return { line, column: clamped - lastNewline }
}

/**
 * Iterative RFC-8259 scanner used only after `JSON.parse` fails so JS engines
 * that omit error coordinates (JavaScriptCore) still report exact line/column.
 */
function findStrictJsonErrorOffset(text: string): number {
  const length = text.length
  if (length === 0 || text.charCodeAt(0) === 0xfeff) return 0
  let index = 0
  const isWhitespace = (code: number): boolean => code === 32 || code === 9 || code === 10 || code === 13
  const skipWhitespace = (): void => {
    while (index < length && isWhitespace(text.charCodeAt(index))) index += 1
  }
  const scanString = (): boolean => {
    if (text.charCodeAt(index) !== 34) return false
    index += 1
    while (index < length) {
      const code = text.charCodeAt(index)
      if (code === 34) {
        index += 1
        return true
      }
      if (code < 0x20) return false
      if (code === 92) {
        index += 1
        if (index >= length) return false
        const escaped = text.charCodeAt(index)
        if (escaped === 117) {
          for (let hexIndex = 1; hexIndex <= 4; hexIndex += 1) {
            const hex = text.charCodeAt(index + hexIndex)
            const validHex = (hex >= 48 && hex <= 57) || (hex >= 65 && hex <= 70) || (hex >= 97 && hex <= 102)
            if (!validHex) {
              index += hexIndex
              return false
            }
          }
          index += 5
          continue
        }
        if (!'"\\/bfnrt'.includes(String.fromCharCode(escaped))) return false
      }
      index += 1
    }
    return false
  }
  const scanNumber = (): boolean => {
    if (text.charCodeAt(index) === 45) {
      index += 1
      if (index >= length) return false
    }
    const first = text.charCodeAt(index)
    if (first === 48) {
      index += 1
      const afterZero = text.charCodeAt(index)
      if (afterZero >= 48 && afterZero <= 57) return false
    } else if (first >= 49 && first <= 57) {
      while (index < length && text.charCodeAt(index) >= 48 && text.charCodeAt(index) <= 57) index += 1
    } else {
      return false
    }
    if (text.charCodeAt(index) === 46) {
      index += 1
      if (index >= length || text.charCodeAt(index) < 48 || text.charCodeAt(index) > 57) return false
      while (index < length && text.charCodeAt(index) >= 48 && text.charCodeAt(index) <= 57) index += 1
    }
    const exp = text.charCodeAt(index)
    if (exp === 101 || exp === 69) {
      index += 1
      if (text.charCodeAt(index) === 43 || text.charCodeAt(index) === 45) index += 1
      if (index >= length || text.charCodeAt(index) < 48 || text.charCodeAt(index) > 57) return false
      while (index < length && text.charCodeAt(index) >= 48 && text.charCodeAt(index) <= 57) index += 1
    }
    return true
  }

  type Frame = 'after_object_open' | 'after_object_key' | 'after_object_value' | 'after_array_open' | 'after_array_value' | 'expect_value'
  const stack: Frame[] = ['expect_value']
  while (stack.length > 0) {
    const frame = stack.pop()!
    skipWhitespace()
    if (frame === 'expect_value') {
      if (index >= length) return index
      const ch = text.charCodeAt(index)
      if (ch === 34) {
        if (!scanString()) return index
      } else if (ch === 123) {
        index += 1
        stack.push('after_object_open')
      } else if (ch === 91) {
        index += 1
        stack.push('after_array_open')
      } else if (ch === 116 && text.startsWith('true', index)) {
        index += 4
      } else if (ch === 102 && text.startsWith('false', index)) {
        index += 5
      } else if (ch === 110 && text.startsWith('null', index)) {
        index += 4
      } else if (!scanNumber()) {
        return index
      }
      continue
    }
    if (frame === 'after_object_open') {
      if (text.charCodeAt(index) === 125) {
        index += 1
        continue
      }
      if (!scanString()) return index
      stack.push('after_object_key')
      continue
    }
    if (frame === 'after_object_key') {
      if (text.charCodeAt(index) !== 58) return index
      index += 1
      stack.push('after_object_value')
      stack.push('expect_value')
      continue
    }
    if (frame === 'after_object_value') {
      if (text.charCodeAt(index) === 44) {
        index += 1
        skipWhitespace()
        if (!scanString()) return index
        stack.push('after_object_key')
        continue
      }
      if (text.charCodeAt(index) === 125) {
        index += 1
        continue
      }
      return index
    }
    if (frame === 'after_array_open') {
      if (text.charCodeAt(index) === 93) {
        index += 1
        continue
      }
      stack.push('after_array_value')
      stack.push('expect_value')
      continue
    }
    if (frame === 'after_array_value') {
      if (text.charCodeAt(index) === 44) {
        index += 1
        stack.push('after_array_value')
        stack.push('expect_value')
        continue
      }
      if (text.charCodeAt(index) === 93) {
        index += 1
        continue
      }
      return index
    }
  }
  skipWhitespace()
  return index
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
  'WriteFile', 'FileEditTool', 'ApplyPatchTool', 'XlsxWriter', 'XlsxConvert', 'AppendFile', 'MultiEdit', 'Write', 'Edit',
  'NotebookEdit', 'NotebookEditTool', 'append_file', 'apply_patch', 'apply_patch_tool', 'edit_file', 'file_edit_tool',
  'notebook_edit_tool', 'replace_in_file', 'write_file',
])
const READ_BACK_TOOLS: ReadonlySet<string> = new Set([
  'XlsxReader', 'CSVProcessor', 'csv_processor', 'ReadFile', 'read_file', 'FileOutlineTool', 'file_outline_tool',
])

function isWrite(run: ToolRunSummary): boolean {
  return WRITE_TOOLS.has(run.name)
    || ((run.name === 'CSVProcessor' || run.name === 'csv_processor') && /\b(?:write|convert)\b/iu.test(run.command ?? ''))
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
