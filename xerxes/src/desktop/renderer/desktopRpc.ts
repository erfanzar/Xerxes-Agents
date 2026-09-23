// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type RpcRecord = Record<string, unknown>
export function record(value: unknown): RpcRecord {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('Invalid daemon response')
  return value as RpcRecord
}
export function text(value: unknown): string {
  return typeof value === 'string' ? value : ''
}
const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

/**
 * A readable gloss for the common cron shapes, falling back to the raw
 * expression. Schedules were shown as `0 2 * * *`, which is precise and
 * unreadable; the original always stays available as the row's tooltip.
 */
export function cronInEnglish(expression: string): string {
  const parts = expression.trim().split(/\s+/)
  if (parts.length !== 5) return expression
  const [minute, hour, dayOfMonth, month, dayOfWeek] = parts as [string, string, string, string, string]
  const at = (): string => {
    const h = Number(hour), m = Number(minute)
    if (!Number.isInteger(h) || !Number.isInteger(m) || h < 0 || h > 23 || m < 0 || m > 59) return ''
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`
  }
  const time = at()
  if (!time) return expression
  if (dayOfMonth === '*' && month === '*' && dayOfWeek === '*') return `Every day at ${time}`
  if (dayOfMonth === '*' && month === '*' && /^[0-6]$/.test(dayOfWeek)) return `Every ${DAYS[Number(dayOfWeek)]} at ${time}`
  if (dayOfMonth === '*' && month === '*' && dayOfWeek === '1-5') return `Weekdays at ${time}`
  if (month === '*' && dayOfWeek === '*' && /^\d{1,2}$/.test(dayOfMonth)) return `Day ${dayOfMonth} of each month at ${time}`
  return expression
}

/** The schedule timezone must match the displayed wall-clock time. */
export function scheduleTime(value: string, timezone: string, locale?: string): string {
  const date = new Date(value)
  if (!Number.isFinite(date.getTime())) return 'Time unavailable'
  try { return date.toLocaleString(locale, { timeZone: timezone || 'UTC' }) + ' · ' + (timezone || 'UTC') }
  catch { return date.toISOString() + ' (schedule timezone unavailable)' }
}
export function records(value: unknown): RpcRecord[] {
  if (!Array.isArray(value)) throw new Error('Invalid daemon list response')
  return value.map(record)
}
/** Do not turn typed RPC failures into successful empty states. */
export async function desktopCall(
  bridge: { call(method: string, params?: RpcRecord): Promise<unknown> },
  key: string,
  method: string,
  params: RpcRecord = {},
): Promise<RpcRecord> {
  const result = record(
    await bridge.call(method, { ...params, ...(key ? { session_key: key } : {}) }),
  )
  if (result.ok === false || result.kind === 'error')
    throw new Error(text(result.error) || text(result.message) || `${method} failed`)
  return result
}

export interface Specialist {
  id: string
  description: string
  error: string
}
export function specialistsOf(value: unknown): Specialist[] {
  return records(value).map((row) => {
    if (!text(row.id) || typeof row.description !== 'string')
      throw new Error('Invalid specialist record')
    return { id: text(row.id), description: text(row.description), error: text(row.error) }
  })
}
export function activeBackgroundCount(value: unknown): number {
  return records(value).filter((row) =>
    ['running', 'watching', 'working', 'cancelling'].includes(text(row.state)),
  ).length
}

/** File headers delimit the daemon's diff stream, including untracked previews. */
export function diffSections(
  diff: RpcRecord,
): { path: string; start: number; end: number; untracked: boolean }[] {
  const lines = records(diff.lines ?? [])
  const untracked = new Set(Array.isArray(diff.untracked) ? diff.untracked : [])
  const sections: { path: string; start: number; end: number; untracked: boolean }[] = []
  for (const [index, line] of lines.entries()) {
    if (line.kind !== 'file') continue
    const previous = sections.at(-1)
    if (previous) previous.end = index
    const path = text(line.text)
    sections.push({ path, start: index, end: lines.length, untracked: untracked.has(path) })
  }
  for (const path of untracked) {
    if (typeof path === 'string' && !sections.some(section => section.path === path)) sections.push({ path, start: lines.length, end: lines.length, untracked: true })
  }
  return sections
}

/** Preserve the failure while omitting Electron's internal IPC wrapper. */
export function desktopError(value: unknown): string {
  const message = value instanceof Error ? value.message : String(value)
  return message.replace(/^Error invoking remote method '[^']+': (?:Error: )?/, '').replace(/^rpc -?\d+: /, '')
}
