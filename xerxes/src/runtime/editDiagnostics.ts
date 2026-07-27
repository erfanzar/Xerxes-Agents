// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync } from 'node:fs'
import { isAbsolute, join, resolve } from 'node:path'

import {
  analyzeWorkspaceChanges,
  changedWorkspacePaths,
  defaultCommandRunner,
  formatChangeGuardNotification,
  type CommandRunner,
} from './changeGuard.js'

const MAX_REPORT_CHARS = 3000
const MAX_FILES_REPORTED = 8
const MAX_DIAGNOSTICS_PER_FILE = 5
const MAX_MESSAGE_CHARS = 220
const SEPARATOR = '\u001f'

export type DiagnosticSeverity = 'error' | 'warning'

export interface DiagnosticRange {
  readonly startLine: number
  readonly startColumn: number
  readonly endLine: number
  readonly endColumn: number
}

/**
 * One checker finding, normalized across tsc/ruff/cargo.
 *
 * Every field participates in identity so that two turns producing the "same"
 * error on different lines are treated as different diagnostics.
 */
export interface Diagnostic {
  readonly severity: DiagnosticSeverity
  readonly source: string
  readonly code: string
  readonly range: DiagnosticRange
  readonly message: string
}

export interface CheckerSpec {
  readonly source: string
  readonly command: readonly string[]
}

/** Ordered so the first present marker wins; a polyglot repo gets one checker, not three. */
const CHECKERS: readonly { readonly marker: string, readonly spec: CheckerSpec }[] = [
  {
    marker: 'tsconfig.json',
    spec: { source: 'tsc', command: ['bunx', 'tsc', '--noEmit', '--pretty', 'false', '-p', 'tsconfig.json'] },
  },
  {
    marker: 'pyproject.toml',
    spec: { source: 'ruff', command: ['ruff', 'check', '--output-format', 'concise', '.'] },
  },
  {
    marker: 'Cargo.toml',
    spec: { source: 'cargo', command: ['cargo', 'check', '--quiet', '--message-format', 'short'] },
  },
]

export interface EditDiagnosticsOptions {
  readonly checker?: CheckerSpec
  readonly commandRunner?: CommandRunner
  readonly fileExists?: (path: string) => boolean
  readonly includeWorkspaceRisk?: boolean
  readonly maxReportChars?: number
}

export interface FileDiagnostics {
  readonly diagnostics: readonly Diagnostic[]
  readonly path: string
}

export interface EditDiagnosticsReport {
  readonly checkerSource: string
  readonly files: readonly FileDiagnostics[]
  readonly text: string
  readonly truncated: boolean
}

const EMPTY_REPORT: EditDiagnosticsReport = Object.freeze({
  checkerSource: '',
  files: Object.freeze([]),
  text: '',
  truncated: false,
})

/** Pick a checker from workspace markers; undefined means "no checker, stay silent". */
export function detectChecker(cwd: string, fileExists: (path: string) => boolean = existsSync): CheckerSpec | undefined {
  for (const candidate of CHECKERS) {
    if (fileExists(join(cwd, candidate.marker))) return candidate.spec
  }
  return undefined
}

/**
 * Turn-scoped checker-output differ.
 *
 * The map is absolute path -> diagnostic identity -> diagnostic, so "what did this
 * turn break" is a set difference rather than a text diff of two checker runs, and
 * a moved-but-identical error never registers as new.
 *
 * Paths without a baseline are never reported: the repo's pre-existing errors are
 * not this turn's business, and reporting them trains the model to ignore the channel.
 */
export class EditDiagnostics {
  readonly cwd: string
  readonly checker: CheckerSpec | undefined

  #baseline = new Map<string, Map<string, Diagnostic>>()
  #baselineCapture: Promise<void> | undefined
  #captured = false
  #runner: CommandRunner
  #includeWorkspaceRisk: boolean
  #maxReportChars: number
  #tracked = new Set<string>()

  constructor(cwd: string, options: EditDiagnosticsOptions = {}) {
    this.cwd = cwd
    this.checker = options.checker ?? detectChecker(cwd, options.fileExists ?? existsSync)
    this.#runner = options.commandRunner ?? defaultCommandRunner
    this.#includeWorkspaceRisk = options.includeWorkspaceRisk ?? true
    this.#maxReportChars = options.maxReportChars ?? MAX_REPORT_CHARS
  }

  get trackedPaths(): readonly string[] {
    return [...this.#tracked].sort((left, right) => left.localeCompare(right))
  }

  /**
   * Start the baseline run without blocking the caller.
   *
   * A whole-project typecheck costs seconds; awaiting it at turn start would add that
   * latency to every turn, including read-only ones. The promise is awaited only when
   * a report is actually produced.
   */
  begin(): void {
    if (this.#baselineCapture || !this.checker) return
    this.#baselineCapture = this.#captureBaseline()
  }

  /**
   * Record a path as this turn's business and anchor a pre-mutation baseline.
   *
   * Awaiting the returned promise guarantees the baseline predates the caller's write.
   * A caller that does not await accepts that a baseline started by this very call may
   * observe the file mid-write and absorb the error it was supposed to catch.
   */
  async noteFileWillChange(path: string): Promise<void> {
    this.#tracked.add(this.#absolute(path))
    this.begin()
    await this.#baselineCapture
  }

  reset(): void {
    this.#baseline = new Map()
    this.#baselineCapture = undefined
    this.#captured = false
    this.#tracked.clear()
  }

  /**
   * Diff the current checker output against the baseline and refresh it.
   *
   * Refreshing means an error the model chooses not to fix is reported once, not on
   * every subsequent turn.
   */
  async report(): Promise<EditDiagnosticsReport> {
    if (!this.checker || !this.#baselineCapture) return EMPTY_REPORT
    await this.#baselineCapture
    if (!this.#captured) return EMPTY_REPORT

    const tracked = await this.#trackedForReport()
    if (!tracked.length) return EMPTY_REPORT

    const current = await this.#runChecker()
    if (!current) return EMPTY_REPORT

    const files: FileDiagnostics[] = []
    for (const path of tracked) {
      const after = current.get(path)
      if (!after) continue
      // The baseline run covers the whole project, so a tracked path missing from it was
      // clean, not unmeasured — absent must mean "had no diagnostics", never "skip me".
      const before = this.#baseline.get(path) ?? new Map<string, Diagnostic>()
      const added = [...after].filter(([key]) => !before.has(key)).map(([, diagnostic]) => diagnostic)
      if (added.length) files.push({ path, diagnostics: added })
    }

    this.#baseline = current
    this.#tracked.clear()
    if (!files.length) return { ...EMPTY_REPORT, checkerSource: this.checker.source }

    const risk = this.#includeWorkspaceRisk ? await this.#workspaceRisk() : ''
    const formatted = formatEditDiagnostics(files, {
      cwd: this.cwd,
      maxChars: this.#maxReportChars,
      source: this.checker.source,
      suffix: risk,
    })
    return { checkerSource: this.checker.source, files, text: formatted.text, truncated: formatted.truncated }
  }

  async #captureBaseline(): Promise<void> {
    const parsed = await this.#runChecker()
    if (!parsed) return
    this.#baseline = parsed
    this.#captured = true
  }

  async #runChecker(): Promise<Map<string, Map<string, Diagnostic>> | undefined> {
    const checker = this.checker
    if (!checker) return undefined
    try {
      // A nonzero exit is the normal shape of "there are errors", so it is not a failure;
      // only a throw (missing binary) means the checker could not answer.
      const result = await this.#runner(checker.command, { cwd: this.cwd })
      const text = result.stdout + '\n' + (result.stderr ?? '')
      return parseDiagnostics(text, checker.source, this.cwd)
    } catch {
      return undefined
    }
  }

  async #trackedForReport(): Promise<readonly string[]> {
    if (this.#tracked.size) return this.trackedPaths
    // Nothing announced itself, so fall back to the working tree: file mutations that
    // bypassed noteFileWillChange still show up as dirty paths.
    const changed = await changedWorkspacePaths(this.cwd, { commandRunner: this.#runner })
    return changed.map(path => this.#absolute(path)).sort((left, right) => left.localeCompare(right))
  }

  async #workspaceRisk(): Promise<string> {
    const report = await analyzeWorkspaceChanges(this.cwd, { commandRunner: this.#runner })
    return report.shouldNotify ? formatChangeGuardNotification(report) : ''
  }

  #absolute(path: string): string {
    return isAbsolute(path) ? resolve(path) : resolve(this.cwd, path)
  }
}

export interface FormatEditDiagnosticsOptions {
  readonly cwd: string
  readonly maxChars?: number
  readonly source: string
  readonly suffix?: string
}

/** Render a hard-bounded report; the model gets a fact, not a wall of checker output. */
export function formatEditDiagnostics(
  files: readonly FileDiagnostics[],
  options: FormatEditDiagnosticsOptions,
): { readonly text: string, readonly truncated: boolean } {
  if (!files.length) return { text: '', truncated: false }
  const maxChars = options.maxChars ?? MAX_REPORT_CHARS
  const total = files.reduce((sum, file) => sum + file.diagnostics.length, 0)
  const shown = files.slice(0, MAX_FILES_REPORTED)
  const lines = [
    `[${options.source}] ${total} new problem(s) in ${files.length} file(s) changed this turn:`,
  ]
  let truncated = files.length > shown.length
  for (const file of shown) {
    const relative = relativePath(file.path, options.cwd)
    for (const diagnostic of file.diagnostics.slice(0, MAX_DIAGNOSTICS_PER_FILE)) {
      const code = diagnostic.code ? ' ' + diagnostic.code : ''
      lines.push(`${relative}:${diagnostic.range.startLine}:${diagnostic.range.startColumn}`
        + ` ${diagnostic.severity}${code}: ${truncate(diagnostic.message, MAX_MESSAGE_CHARS)}`)
    }
    if (file.diagnostics.length > MAX_DIAGNOSTICS_PER_FILE) {
      truncated = true
      lines.push(`  +${file.diagnostics.length - MAX_DIAGNOSTICS_PER_FILE} more in ${relative}`)
    }
  }
  if (files.length > shown.length) lines.push(`  +${files.length - shown.length} more file(s)`)
  if (options.suffix) lines.push('', options.suffix)

  let text = lines.join('\n')
  if (text.length > maxChars) {
    truncated = true
    text = text.slice(0, Math.max(0, maxChars - 3)) + '...'
  }
  return { text, truncated }
}

/** Parse checker output into absolute path -> diagnostic identity -> diagnostic. */
export function parseDiagnostics(
  output: string,
  source: string,
  cwd: string,
): Map<string, Map<string, Diagnostic>> {
  const parsed = new Map<string, Map<string, Diagnostic>>()
  for (const line of output.split(/\r?\n/)) {
    const entry = parseDiagnosticLine(line.trimEnd(), source)
    if (!entry) continue
    const absolute = isAbsolute(entry.path) ? resolve(entry.path) : resolve(cwd, entry.path)
    const bucket = parsed.get(absolute) ?? new Map<string, Diagnostic>()
    bucket.set(diagnosticKey(entry.diagnostic), entry.diagnostic)
    parsed.set(absolute, bucket)
  }
  return parsed
}

/**
 * Identity tuple: severity, source, code, range, message.
 *
 * A separator no diagnostic text can contain keeps the tuple unambiguous, so a message
 * that happens to end in a code-shaped token cannot collide with a different tuple.
 */
export function diagnosticKey(diagnostic: Diagnostic): string {
  const { startLine, startColumn, endLine, endColumn } = diagnostic.range
  return [
    diagnostic.severity,
    diagnostic.source,
    diagnostic.code,
    `${startLine}:${startColumn}-${endLine}:${endColumn}`,
    diagnostic.message,
  ].join(SEPARATOR)
}

const TSC_LINE = /^(.+?)\((\d+),(\d+)\):\s+(error|warning)\s+([A-Z]+\d+):\s+(.*)$/
const RUFF_LINE = /^(.+?):(\d+):(\d+):\s+([A-Z]+\d+)\s+(?:\[\*\]\s+)?(.*)$/
const CARGO_LINE = /^(.+?):(\d+):(\d+):\s+(error|warning)(?:\[([A-Za-z0-9]+)\])?:\s+(.*)$/

function parseDiagnosticLine(
  line: string,
  source: string,
): { readonly diagnostic: Diagnostic, readonly path: string } | undefined {
  if (!line) return undefined
  if (source === 'tsc') {
    const match = TSC_LINE.exec(line)
    if (!match) return undefined
    const [, path, lineText, columnText, severity, code, message] = match
    return build(path, lineText, columnText, severity as DiagnosticSeverity, code ?? '', message ?? '', source)
  }
  if (source === 'ruff') {
    const match = RUFF_LINE.exec(line)
    if (!match) return undefined
    const [, path, lineText, columnText, code, message] = match
    // Ruff's concise format carries no severity; its W-prefixed rules are the stylistic ones.
    const severity: DiagnosticSeverity = (code ?? '').startsWith('W') ? 'warning' : 'error'
    return build(path, lineText, columnText, severity, code ?? '', message ?? '', source)
  }
  if (source === 'cargo') {
    const match = CARGO_LINE.exec(line)
    if (!match) return undefined
    const [, path, lineText, columnText, severity, code, message] = match
    return build(path, lineText, columnText, severity as DiagnosticSeverity, code ?? '', message ?? '', source)
  }
  return undefined
}

function build(
  path: string | undefined,
  lineText: string | undefined,
  columnText: string | undefined,
  severity: DiagnosticSeverity,
  code: string,
  message: string,
  source: string,
): { readonly diagnostic: Diagnostic, readonly path: string } | undefined {
  if (!path) return undefined
  const startLine = Number.parseInt(lineText ?? '', 10)
  const startColumn = Number.parseInt(columnText ?? '', 10)
  if (!Number.isFinite(startLine) || !Number.isFinite(startColumn)) return undefined
  return {
    path: path.trim(),
    diagnostic: {
      severity,
      source,
      code,
      range: { startLine, startColumn, endLine: startLine, endColumn: startColumn },
      message: message.trim(),
    },
  }
}

const INSTANCES = new Map<string, EditDiagnostics>()

/** Shared per-workspace instance so the mutation site and the turn boundary agree. */
export function editDiagnosticsFor(cwd: string, options: EditDiagnosticsOptions = {}): EditDiagnostics {
  const key = resolve(cwd)
  const existing = INSTANCES.get(key)
  if (existing) return existing
  const created = new EditDiagnostics(key, options)
  INSTANCES.set(key, created)
  return created
}

/** Turn-start entry point: anchor the baseline before any tool mutates a file. */
export function beginEditDiagnosticsTurn(cwd: string, options: EditDiagnosticsOptions = {}): void {
  editDiagnosticsFor(cwd, options).begin()
}

/** Mutation-site entry point for file-writing tools; await it before writing. */
export async function noteEditedPath(cwd: string, path: string): Promise<void> {
  await editDiagnosticsFor(cwd).noteFileWillChange(path)
}

/** Turn-end entry point; empty string means "nothing new, say nothing". */
export async function reportEditDiagnostics(cwd: string): Promise<string> {
  const report = await editDiagnosticsFor(cwd).report()
  return report.text
}

/** Drop cached per-workspace state; used by tests and by workspace switches. */
export function resetEditDiagnostics(cwd?: string): void {
  if (cwd === undefined) {
    INSTANCES.clear()
    return
  }
  INSTANCES.delete(resolve(cwd))
}

function relativePath(path: string, cwd: string): string {
  const base = resolve(cwd)
  return path.startsWith(base + '/') ? path.slice(base.length + 1) : path
}

function truncate(value: string, limit: number): string {
  return value.length <= limit ? value : value.slice(0, Math.max(0, limit - 3)) + '...'
}
