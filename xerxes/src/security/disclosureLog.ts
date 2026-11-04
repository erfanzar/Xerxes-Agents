// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Local, append-only record of what the data boundary let leave the machine.
 *
 * Each line is one {@link DisclosureRecord}: provider, model, trust, purpose,
 * size, a sha256 digest of the outbound text and the decision. The outbound
 * text itself is never written, so the log cannot become a second copy of the
 * private data it is meant to account for.
 */

import { createHash } from 'node:crypto'
import { appendFileSync, chmodSync, closeSync, fstatSync, mkdirSync, openSync, readSync } from 'node:fs'
import { dirname, join } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'
import type { DataTrust, DisclosureRecord, OutboundPurpose, OutboundRequestSummary } from './boundaryTypes.js'

/** Newest bytes of the log a read scans; older records are out of view, never parsed. */
export const MAX_DISCLOSURE_READ_BYTES = 4 * 1024 * 1024
export const DEFAULT_DISCLOSURE_READ_LIMIT = 50
export const MAX_DISCLOSURE_READ_LIMIT = 1000
const MAX_REASON_CHARS = 500

const TRUSTS: ReadonlySet<string> = new Set<DataTrust>(['local', 'cloud'])
const PURPOSES: ReadonlySet<string> = new Set<OutboundPurpose>(['turn', 'subagent', 'advisor', 'title', 'compaction', 'helper'])
const DECISIONS: ReadonlySet<string> = new Set(['sent', 'approved', 'denied'])

export function defaultDisclosureLogPath(environment: Record<string, string | undefined> = process.env): string {
  return join(xerxesHome(environment), 'boundary', 'disclosures.jsonl')
}

export interface DisclosureLogOptions {
  readonly path?: string
  readonly now?: () => Date
}

export class DisclosureLog {
  readonly path: string
  private readonly now: () => Date

  constructor(options: DisclosureLogOptions = {}) {
    this.path = options.path ?? defaultDisclosureLogPath()
    this.now = options.now ?? (() => new Date())
  }

  /** Append one decision. Write failures propagate: an unlogged disclosure must be visible. */
  record(summary: OutboundRequestSummary, decision: DisclosureRecord['decision'], reason?: string): DisclosureRecord {
    const record: DisclosureRecord = {
      at: this.now().toISOString(),
      ...(summary.sessionId === undefined ? {} : { sessionId: summary.sessionId }),
      ...(summary.agentId === undefined ? {} : { agentId: summary.agentId }),
      provider: summary.provider,
      model: summary.model,
      trust: summary.trust,
      purpose: summary.purpose,
      bytes: summary.bytes,
      digest: createHash('sha256').update(summary.text, 'utf8').digest('hex'),
      decision,
      ...(reason?.trim() ? { reason: reason.trim().slice(0, MAX_REASON_CHARS) } : {}),
    }
    mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 })
    appendFileSync(this.path, `${JSON.stringify(record)}\n`, { encoding: 'utf8', mode: 0o600 })
    chmodSync(this.path, 0o600)
    return record
  }

  /** Newest records first, from at most the last {@link MAX_DISCLOSURE_READ_BYTES} of the file. */
  read(options: { readonly limit?: number; readonly sessionId?: string } = {}): DisclosureRecord[] {
    const limit = Math.min(Math.max(1, Math.trunc(options.limit ?? DEFAULT_DISCLOSURE_READ_LIMIT)), MAX_DISCLOSURE_READ_LIMIT)
    const tail = readTail(this.path, MAX_DISCLOSURE_READ_BYTES)
    if (tail === undefined) return []
    const lines = tail.text.split('\n')
    // A tail read that started mid-file begins inside a line; drop that fragment.
    if (tail.truncated) lines.shift()
    const records: DisclosureRecord[] = []
    for (let index = lines.length - 1; index >= 0 && records.length < limit; index -= 1) {
      const record = parseRecord(lines[index] ?? '')
      if (!record) continue
      if (options.sessionId !== undefined && record.sessionId !== options.sessionId) continue
      records.push(record)
    }
    return records
  }
}

/** A compact fixed-width table for `/boundary log`. */
export function renderDisclosures(records: readonly DisclosureRecord[]): string {
  if (!records.length) return 'No outbound requests recorded.'
  const header = ['time', 'decision', 'trust', 'purpose', 'provider/model', 'bytes', 'digest', 'reason']
  const rows = records.map(record => [
    record.at.replace('T', ' ').replace(/\.\d+Z$/, 'Z'),
    record.decision,
    record.trust,
    record.purpose,
    clip(`${record.provider}/${record.model}`, 40),
    String(record.bytes),
    record.digest.slice(0, 12),
    clip(record.reason ?? '', 60),
  ])
  const widths = header.map((title, column) => Math.max(title.length, ...rows.map(row => row[column]?.length ?? 0)))
  const format = (row: readonly string[]) => row.map((cell, column) => cell.padEnd(widths[column] ?? 0)).join('  ').trimEnd()
  return [format(header), format(widths.map(width => '-'.repeat(width))), ...rows.map(format)].join('\n')
}

function clip(value: string, limit: number): string {
  return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`
}

function readTail(path: string, maxBytes: number): { readonly text: string; readonly truncated: boolean } | undefined {
  let fd: number
  try {
    fd = openSync(path, 'r')
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') return undefined
    throw error
  }
  try {
    const size = fstatSync(fd).size
    const length = Math.min(size, maxBytes)
    const buffer = Buffer.alloc(length)
    let offset = 0
    while (offset < length) {
      const read = readSync(fd, buffer, offset, length - offset, size - length + offset)
      if (read <= 0) break
      offset += read
    }
    return { text: buffer.subarray(0, offset).toString('utf8'), truncated: size > length }
  } finally {
    closeSync(fd)
  }
}

function parseRecord(line: string): DisclosureRecord | undefined {
  if (!line.trim()) return undefined
  let value: unknown
  try {
    value = JSON.parse(line)
  } catch {
    return undefined
  }
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined
  const raw = value as Record<string, unknown>
  if (typeof raw.at !== 'string' || typeof raw.provider !== 'string' || typeof raw.model !== 'string'
    || typeof raw.trust !== 'string' || !TRUSTS.has(raw.trust)
    || typeof raw.purpose !== 'string' || !PURPOSES.has(raw.purpose)
    || typeof raw.bytes !== 'number' || !Number.isSafeInteger(raw.bytes) || raw.bytes < 0
    || typeof raw.digest !== 'string' || !/^[0-9a-f]{64}$/.test(raw.digest)
    || typeof raw.decision !== 'string' || !DECISIONS.has(raw.decision)) return undefined
  return {
    at: raw.at,
    ...(typeof raw.sessionId === 'string' ? { sessionId: raw.sessionId } : {}),
    ...(typeof raw.agentId === 'string' ? { agentId: raw.agentId } : {}),
    provider: raw.provider,
    model: raw.model,
    trust: raw.trust as DataTrust,
    purpose: raw.purpose as OutboundPurpose,
    bytes: raw.bytes,
    digest: raw.digest,
    decision: raw.decision as DisclosureRecord['decision'],
    ...(typeof raw.reason === 'string' ? { reason: raw.reason } : {}),
  }
}
