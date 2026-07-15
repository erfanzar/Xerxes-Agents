// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const DEFAULT_HEADROOM_PREVIEW_CHARS = 4_000

const IMPORTANT_RE = /\b(error|failed?|failure|fatal|exception|traceback|warning|warn|panic|timeout|denied|segmentation|assert)\b/i
const SUMMARY_RE = /(=+ .* =+|short test summary|tests? passed|tests? failed|passed in|failed in|npm err|cargo test|all checks passed)/i

export interface HeadroomResultOptions {
  readonly compressed: string
  readonly contentType: string
  readonly notes?: readonly string[]
  readonly original: string
}

/** Bounded model-visible preview while the caller retains the full tool result elsewhere. */
export class HeadroomResult {
  readonly compressed: string
  readonly compressedChars: number
  readonly compressedLines: number
  readonly contentType: string
  readonly notes: readonly string[]
  readonly originalChars: number
  readonly originalLines: number

  constructor(options: HeadroomResultOptions) {
    const preview = options.compressed.trimEnd()
    this.contentType = options.contentType
    this.originalChars = options.original.length
    this.compressedChars = preview.length
    this.originalLines = lineCount(options.original)
    this.compressedLines = lineCount(preview)
    this.compressed = preview
    this.notes = Object.freeze([...(options.notes ?? [])])
    Object.freeze(this)
  }

  get keptRatio(): number {
    return this.originalChars <= 0 ? 1 : Math.min(1, this.compressedChars / this.originalChars)
  }

  metadataLine(): string {
    return this.contentType + ' preview: ' + formatInteger(this.originalChars) + ' -> '
      + formatInteger(this.compressedChars) + ' chars (' + (this.keptRatio * 100).toFixed(1) + '% kept)'
  }
}

export interface CompressToolResultOptions {
  readonly maxChars?: number
}

/**
 * Route noisy tool output through deterministic JSON, diff, search, log, or
 * text previews. No provider call is made; callers can persist the original
 * result independently before sending this bounded preview to a model.
 */
export function compressToolResult(
  toolName: string,
  content: unknown,
  options: CompressToolResultOptions = {},
): HeadroomResult {
  const text = stringify(content)
  if (!text) return result('empty', text, '', ['empty tool result'])
  if (looksBinary(text)) {
    return result(
      'binary',
      text,
      '[Binary-like tool result omitted. Read the stored project-memory artifact for bytes.]',
      ['binary-like content'],
    )
  }
  const maxChars = normalizedMaxChars(options.maxChars)
  const stripped = text.trimStart()
  if (stripped.startsWith('{') || stripped.startsWith('[')) {
    const compressed = compressJson(text, maxChars)
    if (compressed !== undefined) return result('json', text, compressed)
  }
  if (looksLikeDiff(text)) return result('diff', text, compressDiff(text, maxChars))
  if (looksLikeSearch(text)) return result('search', text, compressSearch(text, maxChars))
  if (looksLikeLog(text, toolName)) return result('log', text, compressLog(text, maxChars))
  return result('text', text, compressText(text, maxChars))
}

function result(contentType: string, original: string, compressed: string, notes: readonly string[] = []): HeadroomResult {
  return new HeadroomResult({ contentType, original, compressed, notes })
}

function stringify(value: unknown): string {
  if (typeof value === 'string') return value
  try {
    const rendered = JSON.stringify(sortValue(value), null, 2)
    return rendered === undefined ? String(value) : rendered
  } catch {
    return String(value)
  }
}

function looksBinary(text: string): boolean {
  if (text.includes('\u0000')) return true
  const sample = text.slice(0, 4_096)
  if (!sample) return false
  let controls = 0
  for (const character of sample) {
    const code = character.codePointAt(0) ?? 0
    if (code < 32 && character !== '\n' && character !== '\r' && character !== '\t') controls += 1
  }
  return controls / sample.length > 0.05
}

function fitText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text.trimEnd()
  const marker = '\n\n[... ' + formatInteger(text.length - maxChars) + ' chars omitted by Xerxes headroom ...]\n\n'
  const remaining = Math.max(2, maxChars - marker.length)
  const head = Math.floor(remaining / 2)
  const tail = remaining - head
  const output = text.slice(0, head).trimEnd() + marker + text.slice(-tail).trimStart()
  return output.length <= maxChars ? output.trimEnd() : output.slice(0, maxChars).trimEnd()
}

function compressJson(text: string, maxChars: number): string | undefined {
  let value: unknown
  try {
    value = JSON.parse(text)
  } catch {
    return undefined
  }
  const minified = stableJson(value)
  if (minified.length <= maxChars) return minified
  const rendered = JSON.stringify(shrinkJson(value), null, 2)
  return fitText(rendered, maxChars)
}

function shrinkJson(value: unknown, depth = 0): unknown {
  if (depth >= 4) return jsonAtomPreview(value)
  if (Array.isArray(value)) {
    const important = value.filter(jsonValueImportant)
    const sample = dedupeValues([...important.slice(0, 5), ...value.slice(0, 3), ...value.slice(-2)])
    return {
      type: 'array',
      items: value.length,
      sample: sample.map(item => shrinkJson(item, depth + 1)),
      omitted: Math.max(0, value.length - sample.length),
    }
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
    let selected = entries
    let omitted = 0
    if (entries.length > 24) {
      const important = entries.filter(([key]) => jsonKeyImportant(key))
      selected = dedupePairs([...important.slice(0, 8), ...entries.slice(0, 10), ...entries.slice(-5)])
      omitted = entries.length - selected.length
    }
    const output: Record<string, unknown> = {}
    for (const [key, item] of selected) output[key] = shrinkJson(item, depth + 1)
    if (omitted > 0) output['...'] = omitted + ' keys omitted'
    return output
  }
  return jsonAtomPreview(value)
}

function jsonKeyImportant(key: string): boolean {
  return IMPORTANT_RE.test(key)
}

function jsonValueImportant(value: unknown): boolean {
  return IMPORTANT_RE.test(stableJson(value))
}

function jsonAtomPreview(value: unknown): unknown {
  if (typeof value === 'string' && value.length > 240) {
    return value.slice(0, 160) + ' ... [' + (value.length - 220) + ' chars omitted] ... ' + value.slice(-60)
  }
  return value
}

function dedupePairs(items: readonly (readonly [string, unknown])[]): Array<[string, unknown]> {
  const seen = new Set<string>()
  const output: Array<[string, unknown]> = []
  for (const [key, value] of items) {
    if (seen.has(key)) continue
    seen.add(key)
    output.push([key, value])
  }
  return output
}

function dedupeValues(items: readonly unknown[]): unknown[] {
  const seen = new Set<string>()
  const output: unknown[] = []
  for (const item of items) {
    const marker = stableJson(item)
    if (seen.has(marker)) continue
    seen.add(marker)
    output.push(item)
  }
  return output
}

function looksLikeDiff(text: string): boolean {
  return text.startsWith('diff --git ') || text.includes('\ndiff --git ') || /^@@ -\d+/m.test(text)
}

function compressDiff(text: string, maxChars: number): string {
  const files = splitDiffFiles(text)
  if (!files.length) return compressText(text, maxChars)
  const selected = files.slice(0, 12)
  const output = ['[Xerxes headroom diff preview: ' + files.length + ' files, showing ' + selected.length + ']']
  selected.forEach((file, index) => {
    output.push('', '# File ' + (index + 1), compressDiffFile(file).trimEnd())
  })
  if (files.length > selected.length) output.push('\n[... ' + (files.length - selected.length) + ' diff files omitted ...]')
  return fitText(output.join('\n'), maxChars)
}

function splitDiffFiles(text: string): string[] {
  const starts = [...text.matchAll(/^diff --git /gm)].map(match => match.index ?? 0)
  if (!starts.length) return text.includes('@@ -') ? [text] : []
  starts.push(text.length)
  const files: string[] = []
  for (let index = 0; index < starts.length - 1; index += 1) {
    const start = starts[index]
    const end = starts[index + 1]
    if (start !== undefined && end !== undefined) files.push(text.slice(start, end).trimEnd())
  }
  return files
}

function compressDiffFile(file: string): string {
  const lines = splitLines(file)
  const hunkIndexes = lines.flatMap((line, index) => line.startsWith('@@ ') ? [index] : [])
  if (!hunkIndexes.length) return lines.slice(0, 40).join('\n')
  const first = hunkIndexes[0] ?? 0
  const header = lines.slice(0, first)
  const bounds = [...hunkIndexes, lines.length]
  const hunks: string[][] = []
  for (let index = 0; index < bounds.length - 1; index += 1) {
    const start = bounds[index]
    const end = bounds[index + 1]
    if (start !== undefined && end !== undefined) hunks.push(lines.slice(start, end))
  }
  let selected = hunks.slice(0, 4)
  if (hunks.length > 6) selected = [...selected, ...hunks.slice(-2)]
  else if (hunks.length > 4) selected = [...selected, ...hunks.slice(4)]
  const output = [...header.slice(0, 12)]
  if (header.length > 12) output.push('[... ' + (header.length - 12) + ' header lines omitted ...]')
  for (const hunk of selected) output.push(...trimDiffHunk(hunk))
  if (hunks.length > selected.length) output.push('[... ' + (hunks.length - selected.length) + ' hunks omitted ...]')
  return output.join('\n')
}

function trimDiffHunk(hunk: readonly string[], context = 2): string[] {
  if (!hunk.length) return []
  const keep = new Set<number>([0])
  hunk.forEach((line, index) => {
    if (index === 0) return
    if ((line.startsWith('+') && !line.startsWith('+++')) || (line.startsWith('-') && !line.startsWith('---'))) {
      for (let current = Math.max(1, index - context); current < Math.min(hunk.length, index + context + 1); current += 1) {
        keep.add(current)
      }
    }
  })
  const output: string[] = []
  let previous = -1
  for (const index of [...keep].sort((left, right) => left - right)) {
    if (previous !== -1 && index > previous + 1) output.push('[... ' + (index - previous - 1) + ' context lines omitted ...]')
    output.push(hunk[index] ?? '')
    previous = index
  }
  return output
}

interface SearchLine {
  readonly content: string
  readonly line: number
  readonly path: string
}

function looksLikeSearch(text: string): boolean {
  const lines = splitLines(text).slice(0, 80).filter(line => Boolean(line.trim()))
  if (lines.length < 6) return false
  const parsed = lines.filter(line => parseSearchLine(line) !== undefined).length
  return parsed >= Math.max(4, Math.floor(lines.length / 3))
}

function parseSearchLine(line: string): SearchLine | undefined {
  const pattern = /[:\-](\d+)[:\-]/g
  let match: RegExpExecArray | null
  while ((match = pattern.exec(line)) !== null) {
    const path = line.slice(0, match.index)
    if (!path) continue
    const lineNumber = Number(match[1])
    if (!Number.isSafeInteger(lineNumber)) continue
    return { path, line: lineNumber, content: line.slice((match.index ?? 0) + match[0].length) }
  }
  return undefined
}

function compressSearch(text: string, maxChars: number): string {
  const groups = new Map<string, SearchLine[]>()
  let unparsed = 0
  let total = 0
  for (const line of splitLines(text)) {
    const parsed = parseSearchLine(line)
    if (!parsed) {
      unparsed += 1
      continue
    }
    const values = groups.get(parsed.path) ?? []
    values.push(parsed)
    groups.set(parsed.path, values)
    total += 1
  }
  const output = ['[Xerxes headroom search preview: ' + total + ' matches across ' + groups.size + ' files]']
  const entries = [...groups.entries()]
  for (const [path, matches] of entries.slice(0, 20)) {
    const selected = selectFirstLast(matches, 6)
    output.push('', path)
    for (const match of selected) output.push('  ' + match.line + ': ' + match.content)
    if (matches.length > selected.length) output.push('  [... ' + (matches.length - selected.length) + ' more matches in this file ...]')
  }
  if (entries.length > 20) output.push('\n[... ' + (entries.length - 20) + ' files omitted ...]')
  if (unparsed) output.push('\n[... ' + unparsed + ' non-search lines omitted ...]')
  return fitText(output.join('\n'), maxChars)
}

function looksLikeLog(text: string, toolName: string): boolean {
  const name = toolName.toLowerCase()
  if (['exec', 'shell', 'test', 'pytest', 'build', 'npm', 'cargo'].some(hint => name.includes(hint))) return true
  const lines = splitLines(text)
  if (lines.length < 40) return false
  return lines.slice(0, 200).some(line => IMPORTANT_RE.test(line) || SUMMARY_RE.test(line))
}

function compressLog(text: string, maxChars: number): string {
  const lines = splitLines(text)
  const keep = new Set<number>()
  for (let index = 0; index < Math.min(8, lines.length); index += 1) keep.add(index)
  for (let index = Math.max(0, lines.length - 10); index < lines.length; index += 1) keep.add(index)
  lines.forEach((line, index) => {
    if (!IMPORTANT_RE.test(line) && !SUMMARY_RE.test(line) && !line.startsWith('FAILED ') && !line.startsWith('ERROR ')) return
    for (let current = Math.max(0, index - 2); current < Math.min(lines.length, index + 3); current += 1) keep.add(current)
  })
  const output = ['[Xerxes headroom log preview: kept ' + keep.size + ' of ' + lines.length + ' lines]']
  let previous = -1
  for (const index of [...keep].sort((left, right) => left - right)) {
    if (previous !== -1 && index > previous + 1) output.push('[... ' + (index - previous - 1) + ' log lines omitted ...]')
    output.push('L' + (index + 1) + ': ' + (lines[index] ?? ''))
    previous = index
  }
  return fitText(output.join('\n'), maxChars)
}

function compressText(text: string, maxChars: number): string {
  const lines = splitLines(text)
  if (lines.length <= 60) return fitText(text, maxChars)
  const selected = selectFirstLast(lines, 40)
  const omitted = lines.length - selected.length
  const head = selected.slice(0, Math.min(20, selected.length))
  const tail = selected.slice(head.length)
  const output = ['[Xerxes headroom text preview: ' + lines.length + ' lines, ' + omitted + ' omitted]', ...head]
  if (tail.length) output.push('[... ' + omitted + ' middle lines omitted ...]', ...tail)
  return fitText(output.join('\n'), maxChars)
}

function selectFirstLast<T>(items: readonly T[], limit: number): T[] {
  if (items.length <= limit) return [...items]
  const head = Math.max(1, Math.floor(limit / 2))
  const tail = Math.max(1, limit - head)
  return [...items.slice(0, head), ...items.slice(-tail)]
}

function stableJson(value: unknown): string {
  try {
    return JSON.stringify(sortValue(value)) ?? String(value)
  } catch {
    return String(value)
  }
}

function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, sortValue(item)]))
  }
  return value
}

function normalizedMaxChars(value: number | undefined): number {
  if (value === undefined) return DEFAULT_HEADROOM_PREVIEW_CHARS
  if (!Number.isFinite(value)) throw new RangeError('maxChars must be finite')
  return Math.max(512, Math.trunc(value))
}

function lineCount(value: string): number {
  return value ? splitLines(value).length : 0
}

function splitLines(value: string): string[] {
  return value.split(/\r?\n/)
}

function formatInteger(value: number): string {
  return value.toLocaleString('en-US')
}
