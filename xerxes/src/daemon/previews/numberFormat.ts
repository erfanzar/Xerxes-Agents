// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Enough of Excel's number formats to show a workbook's cells as people
 * expect to read them: dates and times as ISO dates and clock times (never
 * serial numbers), percentages, fixed decimals, thousands separators and
 * currency or unit literals. Formats this does not cover (fractions,
 * scientific, conditional sections, locale-specific separators) fall back to
 * the plain number, which is still the true value.
 */

/** Excel's built-in format codes that matter for display, by id. */
const BUILT_IN: Readonly<Record<number, string>> = {
  1: '0', 2: '0.00', 3: '#,##0', 4: '#,##0.00', 9: '0%', 10: '0.00%',
  14: 'yyyy-mm-dd', 15: 'd-mmm-yy', 16: 'd-mmm', 17: 'mmm-yy',
  18: 'h:mm AM/PM', 19: 'h:mm:ss AM/PM', 20: 'h:mm', 21: 'h:mm:ss', 22: 'yyyy-mm-dd h:mm',
  37: '#,##0 ;(#,##0)', 38: '#,##0 ;(#,##0)', 39: '#,##0.00;(#,##0.00)', 40: '#,##0.00;(#,##0.00)',
  45: 'mm:ss', 46: '[h]:mm:ss', 47: 'mm:ss.0',
}

/** Built-in ids reserved for (East Asian) date formats whose codes are not stored in the file. */
const BUILT_IN_DATES = new Set([27, 28, 29, 30, 31, 34, 35, 36, 50, 51, 52, 53, 54, 57, 58])
const BUILT_IN_TIMES = new Set([32, 33, 55, 56])

/** The format code a cell style points at: a built-in id or the workbook's own `numFmt`. */
export function formatCodeOf(id: number, custom: ReadonlyMap<number, string>): string | undefined {
  if (BUILT_IN_DATES.has(id)) return 'yyyy-mm-dd'
  if (BUILT_IN_TIMES.has(id)) return 'h:mm:ss'
  return custom.get(id) ?? BUILT_IN[id]
}

/** The text of `code` without quoted literals, escapes, fills, padding and bracketed colors or locales. */
function bare(code: string): string {
  return code
    .replace(/"[^"]*"/g, '')
    .replace(/\\./g, '')
    .replace(/[_*]./g, '')
    .replace(/\[(?![hms]+\])[^\]]*\]/gi, '')
}

/** Split on `;` outside quotes. */
function sections(code: string): string[] {
  const out: string[] = []
  let current = ''
  let quoted = false
  for (let index = 0; index < code.length; index += 1) {
    const char = code[index]!
    if (char === '"') quoted = !quoted
    if (char === '\\' && !quoted) { current += char + (code[index + 1] ?? ''); index += 1; continue }
    if (char === ';' && !quoted) { out.push(current); current = ''; continue }
    current += char
  }
  out.push(current)
  return out
}

type Clock = { readonly date: boolean; readonly time: boolean; readonly seconds: boolean; readonly elapsed: boolean }

/** What a format code shows of a date, if it is a date or time format at all. */
export function clockOf(code: string): Clock | undefined {
  const plain = bare(sections(code)[0] ?? '')
  if (!plain || /^general$/i.test(plain.trim())) return undefined
  const elapsed = /\[[hms]+\]/i.test(plain)
  const letters = plain.replace(/\[[hms]+\]/gi, 'h').replace(/am\/pm|a\/p/gi, '')
  const date = /[yd]/i.test(letters) || /mmm/i.test(letters)
  const time = /[hs]/i.test(letters) || elapsed
  if (!date && !time) return undefined
  return { date, time, seconds: /s/i.test(letters), elapsed }
}

const DAY_MS = 86_400_000

/** A serial date as UTC wall-clock parts. Excel's 1900 system counts the fictional 1900-02-29. */
function serialParts(serial: number, date1904: boolean): Date {
  const seconds = Math.round(serial * 86_400)
  const days = Math.floor(seconds / 86_400)
  const base = date1904 ? Date.UTC(1904, 0, 1) : Date.UTC(1899, 11, 31)
  const shifted = date1904 || days < 60 ? days : days - 1
  return new Date(base + shifted * DAY_MS + (seconds - days * 86_400) * 1000)
}

const two = (value: number): string => String(value).padStart(2, '0')

function formatClock(serial: number, clock: Clock, date1904: boolean): string {
  if (clock.elapsed) {
    const total = Math.round(serial * 86_400)
    const hours = Math.floor(total / 3600)
    return `${hours}:${two(Math.floor(total / 60) % 60)}${clock.seconds ? ':' + two(total % 60) : ''}`
  }
  const at = serialParts(serial, date1904)
  const day = `${at.getUTCFullYear()}-${two(at.getUTCMonth() + 1)}-${two(at.getUTCDate())}`
  const time = `${two(at.getUTCHours())}:${two(at.getUTCMinutes())}${clock.seconds ? ':' + two(at.getUTCSeconds()) : ''}`
  if (clock.date && clock.time) return `${day} ${time}`
  return clock.date ? day : time
}

/** Excel's General format: up to 15 significant digits, no float noise. */
export function generalNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value)
  if (Number.isInteger(value) && Math.abs(value) < 1e15) return String(value)
  return String(Number(value.toPrecision(15)))
}

/** Literal text a section prints around its digits: quoted text, escapes and `[$€-407]` currency. */
function literal(part: string): string {
  return part
    .replace(/\[\$([^\]-]*)(?:-[^\]]*)?\]/g, '$1')
    .replace(/\[[^\]]*\]/g, '')
    .replace(/"([^"]*)"/g, '$1')
    .replace(/\\(.)/g, '$1')
    .replace(/_./g, '')
    .replace(/\*./g, '')
}

function formatFixed(value: number, section: string): string | undefined {
  const plain = bare(section)
  if (/[eE][+-]|\?|\//.test(plain) || plain.includes('@')) return undefined
  // The digit run: first to last placeholder outside quotes, brackets and escapes.
  let first = -1
  let last = -1
  for (let index = 0; index < section.length; index += 1) {
    const char = section[index]!
    if (char === '"') { const close = section.indexOf('"', index + 1); index = close < 0 ? section.length : close; continue }
    if (char === '[') { const close = section.indexOf(']', index + 1); index = close < 0 ? section.length : close; continue }
    if (char === '\\' || char === '_' || char === '*') { index += 1; continue }
    if (/[0#?]/.test(char)) { if (first < 0) first = index; last = index }
    else if (first >= 0 && /[,.]/.test(char) && last === index - 1) last = index
  }
  // A section of text only (`"zero"`) prints that text.
  if (first < 0) return literal(section).trim() || undefined
  // A `%` inside the digit run (or at its end) still multiplies by 100.
  const digits = section.slice(first, last + 1)
  const percent = plain.includes('%')
  const decimals = /\.([0#?]*)/.exec(digits)?.[1]?.length ?? 0
  const grouped = /[0#?],[0#?]/.test(digits)
  // Trailing commas scale by thousands (`#,##0,` shows 1234567 as 1235).
  const scale = /,+$/.exec(digits.replace(/\.[0#?]*$/, ''))?.[0].length ?? 0
  const shown = Math.abs(value) * (percent ? 100 : 1) / 1000 ** scale
  let text = shown.toFixed(Math.min(decimals, 20))
  if (grouped) {
    const [whole = '', fraction] = text.split('.')
    text = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ',') + (fraction === undefined ? '' : '.' + fraction)
  }
  return (literal(section.slice(0, first)) + text + literal(section.slice(last + 1))).trim()
}

/** A numeric cell as Excel would show it under `code`, or the plain number when the code is not understood. */
export function formatCellNumber(value: number, code: string | undefined, date1904: boolean): string {
  if (code === undefined || !Number.isFinite(value)) return generalNumber(value)
  const clock = clockOf(code)
  if (clock) return value >= 0 && value < 2_958_466 ? formatClock(value, clock, date1904) : generalNumber(value)
  const parts = sections(code)
  const negative = value < 0 && parts.length > 1 && parts[1]!.trim() !== ''
  const section = value === 0 && parts.length > 2 && parts[2]!.trim() !== '' ? parts[2]! : negative ? parts[1]! : parts[0]!
  if (/^\s*general\s*$/i.test(bare(section))) return generalNumber(value)
  const fixed = formatFixed(value, section)
  if (fixed === undefined) return generalNumber(value)
  // A negative section prints its own sign (usually parentheses); the first section takes a minus.
  return value < 0 && !negative && Number(fixed.replace(/[^\d.]/g, '')) !== 0 ? '-' + fixed : fixed
}
