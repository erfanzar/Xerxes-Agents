// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { mkdir, stat } from 'node:fs/promises'
import { dirname } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import { isJsonObject, type JsonObject, type JsonValue, type ToolDefinition } from '../types/toolCalls.js'
import { optionalBoolean, optionalInteger, optionalString, requiredString } from './inputs.js'
import { WorkspacePathResolver } from './pathSafety.js'

const DEFAULT_CSV_DELIMITER = ','
const DEFAULT_TIMEZONE = 'UTC'
const HASH_ALGORITHMS = ['md5', 'sha1', 'sha256', 'sha512'] as const

export const JSON_PROCESSOR_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'JSONProcessor',
    description: 'Load, save, validate, query, and inspect JSON inside the current workspace.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        operation: { type: 'string', enum: ['load', 'save', 'validate', 'query', 'transform'] },
        data: { description: 'JSON value used by save, validate, query, and transform.' },
        file_path: { type: 'string', description: 'Workspace-relative JSON file path.' },
        query: { type: 'string', description: 'Dot and numeric-bracket path, such as users[0].name.' },
        pretty: { type: 'boolean', default: true },
        overwrite: { type: 'boolean', default: false, description: 'Allow save to replace an existing file.' },
      },
      required: ['operation'],
    },
  },
}

export const CSV_PROCESSOR_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'CSVProcessor',
    description: 'Read, write, analyze, and convert CSV files inside the current workspace.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        operation: { type: 'string', enum: ['read', 'write', 'analyze', 'convert'] },
        file_path: { type: 'string', description: 'Workspace-relative CSV file path.' },
        data: { type: 'array', items: { type: 'object' }, description: 'Rows for write.' },
        delimiter: { type: 'string', default: ',' },
        headers: { type: 'array', items: { type: 'string' } },
        has_header: { type: 'boolean', default: true },
        max_rows: { type: 'integer', minimum: 0 },
        overwrite: { type: 'boolean', default: false, description: 'Allow write to replace an existing file.' },
      },
      required: ['operation'],
    },
  },
}

export const TEXT_PROCESSOR_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'TextProcessor',
    description: 'Compute text statistics and apply safe JavaScript regular-expression transformations.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        text: { type: 'string' },
        operation: { type: 'string', enum: ['stats', 'clean', 'extract', 'replace', 'split', 'format'] },
        pattern: { type: 'string' },
        replacement: { type: 'string' },
        case_sensitive: { type: 'boolean', default: true },
      },
      required: ['text', 'operation'],
    },
  },
}

export const DATA_CONVERTER_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'DataConverter',
    description: 'Convert JSON, UTF-8 text, Base64, and hexadecimal data without external dependencies.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        data: { description: 'JSON value or encoded string to convert.' },
        from_format: { type: 'string', enum: ['json', 'text', 'raw', 'base64', 'hex'] },
        to_format: { type: 'string', enum: ['json', 'base64', 'hex', 'hash'] },
        encoding: { type: 'string', default: 'utf-8' },
      },
      required: ['data', 'from_format', 'to_format'],
    },
  },
}

export const DATE_TIME_PROCESSOR_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'DateTimeProcessor',
    description: 'Parse, format, offset, and inspect dates with a dependency-free ISO and strftime subset.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        operation: { type: 'string', enum: ['now', 'parse', 'delta', 'format'] },
        date_string: { type: 'string' },
        fmt: { type: 'string', description: 'strftime-style format for parse or format.' },
        timezone: { type: 'string', description: 'IANA timezone, such as UTC or Europe/Istanbul.' },
        delta_days: { type: 'integer', default: 0 },
        delta_hours: { type: 'integer', default: 0 },
        delta_minutes: { type: 'integer', default: 0 },
      },
      required: ['operation'],
    },
  },
}

export const DATA_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  JSON_PROCESSOR_DEFINITION,
  CSV_PROCESSOR_DEFINITION,
  TEXT_PROCESSOR_DEFINITION,
  DATA_CONVERTER_DEFINITION,
  DATE_TIME_PROCESSOR_DEFINITION,
]

/** Register dependency-free data tools that operate only within the supplied workspace. */
export function registerDataTools(registry: ToolRegistry, paths: WorkspacePathResolver): void {
  registry.register(JSON_PROCESSOR_DEFINITION, inputs => processJson(inputs, paths))
  registry.register(CSV_PROCESSOR_DEFINITION, inputs => processCsv(inputs, paths))
  registry.register(TEXT_PROCESSOR_DEFINITION, processText)
  registry.register(DATA_CONVERTER_DEFINITION, convertData)
  registry.register(DATE_TIME_PROCESSOR_DEFINITION, processDateTime)
}

export async function processJson(inputs: JsonObject, paths: WorkspacePathResolver): Promise<JsonObject> {
  const operation = requiredString(inputs, 'operation')
  const pretty = optionalBoolean(inputs, 'pretty', true)

  if (operation === 'load') {
    const filePath = requiredString(inputs, 'file_path')
    const target = await paths.resolve(filePath)
    await requireFile(target, filePath)
    try {
      return { data: JSON.parse(await Bun.file(target).text()) as JsonValue, success: true }
    } catch (error) {
      throw new ValidationError('file_path', 'must contain valid JSON: ' + errorMessage(error), filePath)
    }
  }

  if (operation === 'save') {
    const filePath = requiredString(inputs, 'file_path')
    const data = requiredJsonValue(inputs, 'data')
    const overwrite = optionalBoolean(inputs, 'overwrite', false)
    const target = await paths.resolve(filePath)
    if ((await pathExists(target)) && !overwrite) {
      throw new ValidationError('file_path', 'already exists; pass overwrite=true to replace it', filePath)
    }
    await mkdir(dirname(target), { recursive: true })
    await Bun.write(target, JSON.stringify(data, null, pretty ? 2 : undefined) + (pretty ? '\n' : ''))
    return { file_path: await paths.relative(target), success: true }
  }

  if (operation === 'validate') {
    const data = requiredJsonValue(inputs, 'data')
    try {
      if (typeof data === 'string') {
        JSON.parse(data)
      } else {
        JSON.stringify(data)
      }
      return { valid: true }
    } catch (error) {
      return { error: errorMessage(error), valid: false }
    }
  }

  if (operation === 'query') {
    const data = requiredJsonValue(inputs, 'data')
    const query = requiredString(inputs, 'query')
    return { result: queryJson(data, query) }
  }

  if (operation === 'transform') {
    const data = requiredJsonValue(inputs, 'data')
    const result: JsonObject = {
      keys: isJsonObject(data) ? Object.keys(data) : null,
      length: valueLength(data),
      type: jsonType(data),
    }
    if (pretty) {
      result.formatted = JSON.stringify(data, null, 2)
    }
    return result
  }

  throw new ValidationError('operation', 'must be load, save, validate, query, or transform', operation)
}

export async function processCsv(inputs: JsonObject, paths: WorkspacePathResolver): Promise<JsonObject> {
  const operation = requiredString(inputs, 'operation')
  const delimiter = csvDelimiter(optionalString(inputs, 'delimiter') ?? DEFAULT_CSV_DELIMITER)

  if (operation === 'read' || operation === 'convert' || operation === 'analyze') {
    const filePath = requiredString(inputs, 'file_path')
    const target = await paths.resolve(filePath)
    await requireFile(target, filePath)
    const rows = parseCsv(await Bun.file(target).text(), delimiter)

    if (operation === 'analyze') {
      const headers = rows[0] ?? []
      const dataRows = rows.slice(1)
      return {
        empty_cells: dataRows.reduce((total, row) => total + row.filter(cell => !cell.trim()).length, 0),
        headers,
        sample_data: dataRows.slice(0, 5),
        total_columns: headers.length,
        total_rows: rows.length,
      }
    }

    const hasHeader = optionalBoolean(inputs, 'has_header', true)
    const headers = csvHeaders(inputs, rows, hasHeader)
    const maxRows = optionalMaximumRows(inputs)
    const dataRows = hasHeader ? rows.slice(1) : rows
    const selectedRows = maxRows === undefined ? dataRows : dataRows.slice(0, maxRows)
    const records = selectedRows.map(row => csvRecord(headers, row))
    if (operation === 'convert') {
      return { count: records.length, json: records }
    }
    const result: JsonObject = { count: records.length, data: records }
    if (headers.length > 0) {
      result.columns = headers
    }
    return result
  }

  if (operation === 'write') {
    const filePath = requiredString(inputs, 'file_path')
    const data = requiredRows(inputs, 'data')
    const suppliedHeaders = optionalHeaders(inputs)
    const headers = suppliedHeaders ?? collectHeaders(data)
    if (headers.length === 0) {
      throw new ValidationError('data', 'must contain at least one object field or provide headers', data)
    }
    const overwrite = optionalBoolean(inputs, 'overwrite', false)
    const target = await paths.resolve(filePath)
    if ((await pathExists(target)) && !overwrite) {
      throw new ValidationError('file_path', 'already exists; pass overwrite=true to replace it', filePath)
    }
    await mkdir(dirname(target), { recursive: true })
    const rows = [headers, ...data.map(row => headers.map(header => csvCell(row[header])))]
    await Bun.write(target, stringifyCsv(rows, delimiter))
    return { file_path: await paths.relative(target), rows_written: data.length, success: true }
  }

  throw new ValidationError('operation', 'must be read, write, analyze, or convert', operation)
}

export function processText(inputs: JsonObject): JsonObject {
  const text = requiredStringAllowEmpty(inputs, 'text')
  const operation = requiredString(inputs, 'operation')
  const caseSensitive = optionalBoolean(inputs, 'case_sensitive', true)

  if (operation === 'stats') {
    const words = text.toLocaleLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? []
    return {
      characters_no_spaces: Array.from(text.replace(/\s/gu, '')).length,
      length: Array.from(text).length,
      lines: text === '' ? 0 : text.split(/\r?\n/u).length,
      most_common_chars: frequencyPairs(
        Array.from(text.toLocaleLowerCase()).filter(character => /\p{L}/u.test(character)),
        5,
      ),
      most_common_words: frequencyPairs(words, 10),
      words: words.length,
    }
  }

  if (operation === 'clean') {
    const pattern = optionalString(inputs, 'pattern')
    const cleaned = (pattern === undefined ? text : text.replace(regex(pattern, true), '')).replace(/\s+/gu, ' ').trim()
    return { cleaned_length: Array.from(cleaned).length, cleaned_text: cleaned, original_length: Array.from(text).length }
  }

  if (operation === 'extract') {
    const requestedPattern = requiredString(inputs, 'pattern')
    const source = namedPattern(requestedPattern) ?? requestedPattern
    const matches = [...text.matchAll(regex(source, caseSensitive))].map(match => match[0])
    return { count: matches.length, matches }
  }

  if (operation === 'replace') {
    const pattern = requiredString(inputs, 'pattern')
    const replacement = optionalString(inputs, 'replacement') ?? ''
    const expression = regex(pattern, caseSensitive)
    const matches = [...text.matchAll(expression)].length
    return { replaced_text: text.replace(expression, replacement), replacements_made: matches }
  }

  if (operation === 'split') {
    const pattern = optionalString(inputs, 'pattern')
    const parts = pattern === undefined ? text.trim().split(/\s+/u).filter(Boolean) : text.split(regex(pattern, caseSensitive))
    return { count: parts.length, parts }
  }

  if (operation === 'format') {
    const format = optionalString(inputs, 'pattern')
    const formatted = formatText(text, format)
    return { formatted_text: formatted }
  }

  throw new ValidationError('operation', 'must be stats, clean, extract, replace, split, or format', operation)
}

export function convertData(inputs: JsonObject): JsonObject {
  const data = requiredJsonValue(inputs, 'data')
  const fromFormat = requiredString(inputs, 'from_format')
  const toFormat = requiredString(inputs, 'to_format')
  const encoding = bufferEncoding(optionalString(inputs, 'encoding') ?? 'utf-8')
  const parsed = decodeData(data, fromFormat, encoding)

  if (toFormat === 'json') {
    return { output: JSON.stringify(parsed, null, 2), success: true }
  }
  if (toFormat === 'base64') {
    return { output: Buffer.from(textForEncoding(parsed), encoding).toString('base64'), success: true }
  }
  if (toFormat === 'hex') {
    return { output: Buffer.from(textForEncoding(parsed), encoding).toString('hex'), success: true }
  }
  if (toFormat === 'hash') {
    const source = Buffer.from(textForEncoding(parsed), encoding)
    const hashes: JsonObject = {}
    for (const algorithm of HASH_ALGORITHMS) {
      hashes[algorithm] = createHash(algorithm).update(source).digest('hex')
    }
    return { output: hashes, success: true }
  }
  throw new ValidationError('to_format', 'must be json, base64, hex, or hash', toFormat)
}

export function processDateTime(inputs: JsonObject): JsonObject {
  const operation = requiredString(inputs, 'operation')
  const timezone = validTimezone(optionalString(inputs, 'timezone') ?? DEFAULT_TIMEZONE)

  if (operation === 'now') {
    const now = new Date()
    return { datetime: now.toISOString(), formatted: defaultDateFormats(now, timezone), timestamp: now.getTime() / 1_000 }
  }

  if (operation === 'parse') {
    const dateString = requiredString(inputs, 'date_string')
    const date = parseDate(dateString, optionalString(inputs, 'fmt'))
    return {
      components: dateComponents(date, timezone),
      parsed: date.toISOString(),
      timestamp: date.getTime() / 1_000,
    }
  }

  if (operation === 'delta') {
    const dateString = optionalString(inputs, 'date_string')
    const original = dateString === undefined ? new Date() : parseDate(dateString)
    const days = optionalInteger(inputs, 'delta_days', 0)
    const hours = optionalInteger(inputs, 'delta_hours', 0)
    const minutes = optionalInteger(inputs, 'delta_minutes', 0)
    const totalSeconds = days * 86_400 + hours * 3_600 + minutes * 60
    const next = new Date(original.getTime() + totalSeconds * 1_000)
    return {
      delta: { days, hours, minutes, total_seconds: totalSeconds },
      new: next.toISOString(),
      original: original.toISOString(),
    }
  }

  if (operation === 'format') {
    const dateString = optionalString(inputs, 'date_string')
    const date = dateString === undefined ? new Date() : parseDate(dateString)
    const format = optionalString(inputs, 'fmt')
    if (format === undefined) {
      return { formats: defaultDateFormats(date, timezone) }
    }
    return { formatted: strftime(date, format, timezone) }
  }

  throw new ValidationError('operation', 'must be now, parse, delta, or format', operation)
}

function requiredJsonValue(inputs: JsonObject, name: string): JsonValue {
  const value = inputs[name]
  if (value === undefined) {
    throw new ValidationError(name, 'is required', value)
  }
  return value
}

function requiredStringAllowEmpty(inputs: JsonObject, name: string): string {
  const value = inputs[name]
  if (typeof value !== 'string') {
    throw new ValidationError(name, 'must be a string', value)
  }
  return value
}

function valueLength(value: JsonValue): number | null {
  if (typeof value === 'string' || Array.isArray(value)) {
    return value.length
  }
  if (isJsonObject(value)) {
    return Object.keys(value).length
  }
  return null
}

function jsonType(value: JsonValue): string {
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'array'
  return typeof value
}

function queryJson(data: JsonValue, query: string): JsonValue {
  let current = data
  for (const part of queryParts(query)) {
    if (Array.isArray(current)) {
      const index = Number(part)
      if (!Number.isInteger(index) || index < 0 || current[index] === undefined) {
        throw new ValidationError('query', 'does not resolve at array index ' + part, query)
      }
      current = current[index]
      continue
    }
    if (!isJsonObject(current) || !Object.hasOwn(current, part)) {
      throw new ValidationError('query', 'does not resolve at key ' + part, query)
    }
    current = current[part] as JsonValue
  }
  return current
}

function queryParts(query: string): string[] {
  const parts: string[] = []
  for (const segment of query.split('.')) {
    if (!segment) {
      throw new ValidationError('query', 'must not contain empty path segments', query)
    }
    const match = /^([^\[\]]+)((?:\[\d+\])*)$/u.exec(segment)
    if (!match) {
      throw new ValidationError('query', 'must use dot keys and numeric brackets only', query)
    }
    const key = match[1]
    if (key) parts.push(key)
    const indexes = match[2] ?? ''
    for (const index of indexes.matchAll(/\[(\d+)\]/gu)) {
      const value = index[1]
      if (value === undefined) {
        throw new ValidationError('query', 'contains an invalid array index', query)
      }
      parts.push(value)
    }
  }
  return parts
}

function csvDelimiter(value: string): string {
  if (Array.from(value).length !== 1 || value === '\n' || value === '\r' || value === '"') {
    throw new ValidationError('delimiter', 'must be one non-newline, non-quote character', value)
  }
  return value
}

function optionalMaximumRows(inputs: JsonObject): number | undefined {
  if (inputs.max_rows === undefined) return undefined
  const value = optionalInteger(inputs, 'max_rows', 0)
  if (value < 0) {
    throw new ValidationError('max_rows', 'must be zero or greater', value)
  }
  return value === 0 ? undefined : value
}

function optionalHeaders(inputs: JsonObject): string[] | undefined {
  const value = inputs.headers
  if (value === undefined) return undefined
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string' || !item)) {
    throw new ValidationError('headers', 'must be an array of non-empty strings', value)
  }
  return value as string[]
}

function csvHeaders(inputs: JsonObject, rows: string[][], hasHeader: boolean): string[] {
  const supplied = optionalHeaders(inputs)
  if (hasHeader) {
    return rows[0] ?? []
  }
  if (supplied) return supplied
  return (rows[0] ?? []).map((_, index) => 'col_' + index)
}

function csvRecord(headers: string[], row: string[]): JsonObject {
  const result: JsonObject = {}
  for (const [index, header] of headers.entries()) {
    if (header !== undefined) {
      result[header] = row[index] ?? ''
    }
  }
  return result
}

function requiredRows(inputs: JsonObject, name: string): JsonObject[] {
  const value = inputs[name]
  if (!Array.isArray(value) || value.some(row => !isJsonObject(row))) {
    throw new ValidationError(name, 'must be an array of JSON objects', value)
  }
  return value as JsonObject[]
}

function collectHeaders(rows: JsonObject[]): string[] {
  const headers = new Set<string>()
  for (const row of rows) {
    for (const key of Object.keys(row)) headers.add(key)
  }
  return [...headers]
}

function csvCell(value: JsonValue | undefined): string {
  if (value === undefined || value === null) return ''
  return typeof value === 'string' ? value : JSON.stringify(value)
}

function parseCsv(text: string, delimiter: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
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
        field += character
      }
      continue
    }
    if (character === '"') {
      if (field) throw new ValidationError('file_path', 'contains an invalid CSV quote sequence')
      quoted = true
    } else if (character === delimiter) {
      row.push(field)
      field = ''
    } else if (character === '\n' || character === '\r') {
      if (character === '\r' && text[index + 1] === '\n') index += 1
      row.push(field)
      rows.push(row)
      row = []
      field = ''
    } else {
      field += character
    }
  }
  if (quoted) throw new ValidationError('file_path', 'contains an unterminated CSV quote')
  if (field || row.length > 0) {
    row.push(field)
    rows.push(row)
  }
  return rows
}

function stringifyCsv(rows: string[][], delimiter: string): string {
  return rows.map(row => row.map(cell => csvEscape(cell, delimiter)).join(delimiter)).join('\n') + '\n'
}

function csvEscape(value: string, delimiter: string): string {
  if (!value.includes(delimiter) && !value.includes('"') && !value.includes('\n') && !value.includes('\r')) return value
  return '"' + value.replace(/"/gu, '""') + '"'
}

function namedPattern(pattern: string): string | undefined {
  if (pattern === 'emails') return '\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b'
  if (pattern === 'urls') return 'https?://[^\\s<>()]+'
  if (pattern === 'phones') return '\\+?(?:[0-9][0-9() .-]{5,}[0-9])'
  if (pattern === 'numbers') return '-?\\d+(?:\\.\\d+)?'
  return undefined
}

function regex(source: string, caseSensitive: boolean): RegExp {
  try {
    return new RegExp(source, caseSensitive ? 'gu' : 'giu')
  } catch (error) {
    throw new ValidationError('pattern', 'must be a valid JavaScript regular expression: ' + errorMessage(error), source)
  }
}

function frequencyPairs(values: string[], limit: number): JsonValue[] {
  const counts = new Map<string, number>()
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1)
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, limit)
    .map(([value, count]) => [value, count])
}

function formatText(text: string, format: string | undefined): string {
  if (format === 'title') {
    return text.replace(/\S+/gu, word => word.slice(0, 1).toLocaleUpperCase() + word.slice(1).toLocaleLowerCase())
  }
  if (format === 'upper') return text.toLocaleUpperCase()
  if (format === 'lower') return text.toLocaleLowerCase()
  if (format === 'sentence') {
    return text.replace(/(^|[.!?]\s+)([^\s])/gu, (_, prefix: string, character: string) => {
      return prefix + character.toLocaleUpperCase()
    })
  }
  if (format === 'no_punctuation') return text.replace(/[^\p{L}\p{N}_\s]/gu, '')
  return text
}

function bufferEncoding(value: string): BufferEncoding {
  if (value === 'utf-8' || value === 'utf8') return 'utf8'
  if (value === 'utf16le' || value === 'latin1' || value === 'ascii') return value
  throw new ValidationError('encoding', 'must be utf-8, utf8, utf16le, latin1, or ascii', value)
}

function decodeData(data: JsonValue, fromFormat: string, encoding: BufferEncoding): JsonValue {
  if (fromFormat === 'json') {
    if (typeof data !== 'string') return data
    try {
      return JSON.parse(data) as JsonValue
    } catch (error) {
      throw new ValidationError('data', 'must contain valid JSON: ' + errorMessage(error), data)
    }
  }
  if (fromFormat === 'text' || fromFormat === 'raw') return data
  if (fromFormat === 'base64') {
    if (typeof data !== 'string') throw new ValidationError('data', 'must be a Base64 string', data)
    return Buffer.from(data, 'base64').toString(encoding)
  }
  if (fromFormat === 'hex') {
    if (typeof data !== 'string' || data.length % 2 !== 0 || !/^[0-9a-f]*$/iu.test(data)) {
      throw new ValidationError('data', 'must be an even-length hexadecimal string', data)
    }
    return Buffer.from(data, 'hex').toString(encoding)
  }
  throw new ValidationError('from_format', 'must be json, text, raw, base64, or hex', fromFormat)
}

function textForEncoding(value: JsonValue): string {
  return typeof value === 'string' ? value : JSON.stringify(value)
}

function validTimezone(value: string): string {
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: value }).format()
    return value
  } catch (error) {
    throw new ValidationError('timezone', 'must be a valid IANA timezone: ' + errorMessage(error), value)
  }
}

function parseDate(value: string, format?: string): Date {
  if (format !== undefined) {
    return parseDateWithFormat(value, format)
  }
  const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/u.exec(value)
  if (dateOnly) return localDate(dateOnly, value)
  const slashIso = /^(\d{4})\/(\d{2})\/(\d{2})$/u.exec(value)
  if (slashIso) return localDate(slashIso, value)
  const slashEuropean = /^(\d{2})\/(\d{2})\/(\d{4})$/u.exec(value)
  if (slashEuropean) {
    return localDate([slashEuropean[0], slashEuropean[3] ?? '', slashEuropean[2] ?? '', slashEuropean[1] ?? ''], value)
  }
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    throw new ValidationError('date_string', 'could not be parsed as a date', value)
  }
  return parsed
}

function parseDateWithFormat(value: string, format: string): Date {
  const tokens: Record<string, string> = {
    '%Y': '(?<year>\\d{4})',
    '%m': '(?<month>\\d{1,2})',
    '%d': '(?<day>\\d{1,2})',
    '%H': '(?<hour>\\d{1,2})',
    '%M': '(?<minute>\\d{1,2})',
    '%S': '(?<second>\\d{1,2})',
  }
  const pieces: string[] = []
  for (let index = 0; index < format.length; index += 1) {
    const token = format.slice(index, index + 2)
    if (tokens[token] !== undefined) {
      pieces.push(tokens[token] ?? '')
      index += 1
    } else {
      pieces.push(escapeRegex(format[index] ?? ''))
    }
  }
  const match = new RegExp('^' + pieces.join('') + '$', 'u').exec(value)
  if (!match?.groups?.year || !match.groups.month || !match.groups.day) {
    throw new ValidationError('date_string', 'does not match fmt', value)
  }
  const year = Number(match.groups.year)
  const month = Number(match.groups.month)
  const day = Number(match.groups.day)
  const hour = Number(match.groups.hour ?? 0)
  const minute = Number(match.groups.minute ?? 0)
  const second = Number(match.groups.second ?? 0)
  const parsed = new Date(year, month - 1, day, hour, minute, second)
  if (
    parsed.getFullYear() !== year
    || parsed.getMonth() !== month - 1
    || parsed.getDate() !== day
    || parsed.getHours() !== hour
    || parsed.getMinutes() !== minute
    || parsed.getSeconds() !== second
  ) {
    throw new ValidationError('date_string', 'contains an invalid calendar date', value)
  }
  return parsed
}

function localDate(match: string[], original: string): Date {
  const year = Number(match[1])
  const month = Number(match[2])
  const day = Number(match[3])
  const parsed = new Date(year, month - 1, day)
  if (parsed.getFullYear() !== year || parsed.getMonth() !== month - 1 || parsed.getDate() !== day) {
    throw new ValidationError('date_string', 'contains an invalid calendar date', original)
  }
  return parsed
}

function dateComponents(date: Date, timezone: string): JsonObject {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat('en-US', {
      day: '2-digit',
      hour: '2-digit',
      hourCycle: 'h23',
      minute: '2-digit',
      month: '2-digit',
      second: '2-digit',
      timeZone: timezone,
      weekday: 'long',
      year: 'numeric',
    }).formatToParts(date).map(part => [part.type, part.value]),
  )
  return {
    day: Number(parts.day),
    hour: Number(parts.hour),
    minute: Number(parts.minute),
    month: Number(parts.month),
    second: Number(parts.second),
    weekday: parts.weekday ?? '',
    year: Number(parts.year),
  }
}

function defaultDateFormats(date: Date, timezone: string): JsonObject {
  const components = dateComponents(date, timezone)
  const year = numberPart(components, 'year')
  const month = numberPart(components, 'month')
  const day = numberPart(components, 'day')
  const hour = numberPart(components, 'hour')
  const minute = numberPart(components, 'minute')
  const second = numberPart(components, 'second')
  const dateText = year + '-' + pad(month) + '-' + pad(day)
  const timeText = pad(hour) + ':' + pad(minute) + ':' + pad(second)
  const monthLong = new Intl.DateTimeFormat('en-US', { month: 'long', timeZone: timezone }).format(date)
  const monthShort = new Intl.DateTimeFormat('en-US', { month: 'short', timeZone: timezone }).format(date)
  const hour12 = hour % 12 || 12
  const meridiem = hour < 12 ? 'AM' : 'PM'
  return {
    date: dateText,
    datetime: dateText + ' ' + timeText,
    human: monthLong + ' ' + pad(day) + ', ' + year + ' at ' + hour12 + ':' + pad(minute) + ' ' + meridiem,
    iso: date.toISOString(),
    short: monthShort + ' ' + pad(day) + ', ' + year,
    time: timeText,
    timestamp: date.getTime() / 1_000,
    us: pad(month) + '/' + pad(day) + '/' + year,
    eu: pad(day) + '/' + pad(month) + '/' + year,
  }
}

function strftime(date: Date, format: string, timezone: string): string {
  const components = dateComponents(date, timezone)
  const year = numberPart(components, 'year')
  const month = numberPart(components, 'month')
  const day = numberPart(components, 'day')
  const hour = numberPart(components, 'hour')
  const minute = numberPart(components, 'minute')
  const second = numberPart(components, 'second')
  const replacements: Record<string, string> = {
    '%Y': String(year),
    '%y': pad(year % 100),
    '%m': pad(month),
    '%d': pad(day),
    '%H': pad(hour),
    '%I': pad(hour % 12 || 12),
    '%M': pad(minute),
    '%S': pad(second),
    '%p': hour < 12 ? 'AM' : 'PM',
    '%B': new Intl.DateTimeFormat('en-US', { month: 'long', timeZone: timezone }).format(date),
    '%b': new Intl.DateTimeFormat('en-US', { month: 'short', timeZone: timezone }).format(date),
    '%A': String(components.weekday),
    '%a': new Intl.DateTimeFormat('en-US', { weekday: 'short', timeZone: timezone }).format(date),
    '%%': '%',
  }
  return format.replace(/%[%YymdHIMS pBbaA]/gu, token => replacements[token] ?? token)
}

function numberPart(value: JsonObject, key: string): number {
  const part = value[key]
  if (typeof part !== 'number') throw new Error('Missing date component ' + key)
  return part
}

function pad(value: number): string {
  return String(value).padStart(2, '0')
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^$()|[\]\\]/gu, '\\$&')
}

async function requireFile(path: string, original: string): Promise<void> {
  try {
    if (!(await stat(path)).isFile()) {
      throw new ValidationError('file_path', 'must refer to an existing regular file', original)
    }
  } catch (error) {
    if (error instanceof ValidationError) throw error
    throw new ValidationError('file_path', 'must refer to an existing regular file', original)
  }
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path)
    return true
  } catch (error) {
    if (isNotFound(error)) return false
    throw error
  }
}

function isNotFound(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
