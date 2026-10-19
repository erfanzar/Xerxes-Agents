// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs'
import { dirname } from 'node:path'

/**
 * Why a daemon process stopped, written just before it goes.
 *
 * A shared daemon serves many clients. When one of them (a TUI of another
 * build, a remote setup, the desktop app) restarts it, every other client
 * only sees its socket close — and a chat that never finished a turn is not
 * on disk, so it cannot be reopened. This record is what the next daemon and
 * `/daemon status` show, so a disconnect can be explained after the fact.
 */
export interface DaemonExitRecord {
  readonly at: string
  readonly reason: 'restart' | 'shutdown' | 'crash'
  readonly pid: number
  readonly build: string
  /** Who asked, as the requesting client described itself. */
  readonly requested_by?: string
  /** Sessions held in memory when it stopped. */
  readonly sessions: number
  /** Sessions among them with no completed turn: these are not on disk. */
  readonly unsaved_sessions: number
  readonly detail?: string
}

const MAX_TEXT = 300
const MAX_LOG_BYTES = 256 * 1024

function clean(value: string | undefined): string | undefined {
  const text = value?.replace(/[\r\n\t]+/g, ' ').trim()
  return text ? text.slice(0, MAX_TEXT) : undefined
}

/**
 * Append one record. Synchronous on purpose: it runs on the way out of the
 * process (a crash handler, a restart about to exit), where an awaited write
 * may never complete. Failures are reported, never thrown.
 */
export function recordDaemonExit(path: string, record: DaemonExitRecord): void {
  const requestedBy = clean(record.requested_by)
  const detail = clean(record.detail)
  const line = JSON.stringify({
    at: record.at,
    reason: record.reason,
    pid: record.pid,
    build: record.build,
    ...(requestedBy ? { requested_by: requestedBy } : {}),
    sessions: record.sessions,
    unsaved_sessions: record.unsaved_sessions,
    ...(detail ? { detail } : {}),
  })
  try {
    mkdirSync(dirname(path), { recursive: true })
    appendFileSync(path, `${line}\n`, { encoding: 'utf8', mode: 0o600 })
  } catch (error) {
    console.error(`Could not record why the daemon stopped: ${error instanceof Error ? error.message : String(error)}`)
  }
}

/** The most recent record, or undefined when there is none or it cannot be read. */
export function lastDaemonExit(path: string): DaemonExitRecord | undefined {
  try {
    if (!existsSync(path)) return undefined
    const text = readFileSync(path, 'utf8')
    const tail = text.length > MAX_LOG_BYTES ? text.slice(-MAX_LOG_BYTES) : text
    const lines = tail.split('\n').map(line => line.trim()).filter(Boolean)
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      try {
        const value = JSON.parse(lines[index]!) as unknown
        if (isExitRecord(value)) return value
      } catch { /* a torn last line: try the one before */ }
    }
  } catch { /* unreadable: nothing to report */ }
  return undefined
}

/** One readable line for a person: "restarted 18:02 by xerxes TUI (build …); 1 unsaved chat lost". */
export function describeDaemonExit(record: DaemonExitRecord): string {
  const verb = record.reason === 'restart' ? 'restarted' : record.reason === 'shutdown' ? 'was stopped' : 'crashed'
  const by = record.requested_by ? ` by ${record.requested_by}` : ''
  const lost = record.unsaved_sessions > 0
    ? `; ${record.unsaved_sessions} chat${record.unsaved_sessions === 1 ? '' : 's'} without a finished turn could not be kept`
    : ''
  const detail = record.reason === 'crash' && record.detail ? ` (${record.detail})` : ''
  return `The Xerxes runtime ${verb} at ${record.at}${by}${detail}${lost}.`
}

function isExitRecord(value: unknown): value is DaemonExitRecord {
  if (!value || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return typeof record.at === 'string'
    && (record.reason === 'restart' || record.reason === 'shutdown' || record.reason === 'crash')
    && typeof record.pid === 'number'
    && typeof record.build === 'string'
    && typeof record.sessions === 'number'
    && typeof record.unsaved_sessions === 'number'
}
