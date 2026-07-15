// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash, randomUUID } from 'node:crypto'
import { chmodSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'

export const ApprovalScope = Object.freeze({
  ONCE: 'once',
  SESSION: 'session',
  ALWAYS: 'always',
} as const)

export type ApprovalScope = (typeof ApprovalScope)[keyof typeof ApprovalScope]

export interface ApprovalRecordInput {
  readonly argsHash?: string
  readonly createdAt?: string
  readonly granted: boolean
  readonly scope: ApprovalScope
  readonly sessionId?: string
  readonly toolName: string
}

/** Stable snake_case persistence shape shared with the original approvals file. */
export interface ApprovalRecordData {
  readonly args_hash: string
  readonly created_at: string
  readonly granted: boolean
  readonly scope: ApprovalScope
  readonly session_id: string
  readonly tool_name: string
}

/** One remembered user approval or denial. */
export class ApprovalRecord {
  readonly argsHash: string
  readonly createdAt: string
  readonly granted: boolean
  readonly scope: ApprovalScope
  readonly sessionId: string
  readonly toolName: string

  constructor(input: ApprovalRecordInput) {
    this.toolName = requiredText(input.toolName, 'toolName')
    this.scope = requireScope(input.scope)
    if (typeof input.granted !== 'boolean') throw new TypeError('granted must be a boolean')
    this.granted = input.granted
    this.sessionId = optionalText(input.sessionId)
    this.argsHash = optionalText(input.argsHash)
    this.createdAt = normalizeTimestamp(input.createdAt)
  }

  toRecord(): ApprovalRecordData {
    return {
      tool_name: this.toolName,
      scope: this.scope,
      granted: this.granted,
      session_id: this.sessionId,
      args_hash: this.argsHash,
      created_at: this.createdAt,
    }
  }

  toJSON(): ApprovalRecordData {
    return this.toRecord()
  }

  static fromRecord(value: unknown): ApprovalRecord {
    if (!isRecord(value)) throw new TypeError('approval record must be an object')
    const sessionId = optionalStringValue(value.session_id, 'session_id')
    const argsHash = optionalStringValue(value.args_hash, 'args_hash')
    const createdAt = optionalStringValue(value.created_at, 'created_at')
    return new ApprovalRecord({
      toolName: requiredValue(value.tool_name, 'tool_name'),
      scope: requiredScopeValue(value.scope),
      granted: requiredBoolean(value.granted, 'granted'),
      ...(sessionId === undefined ? {} : { sessionId }),
      ...(argsHash === undefined ? {} : { argsHash }),
      ...(createdAt === undefined ? {} : { createdAt }),
    })
  }
}

export interface ApprovalStoreOptions {
  /** Explicit persistence location; omit it for an in-memory store. */
  readonly persistencePath?: string
  /** Injectable time source for deterministic records. */
  readonly now?: () => Date
}

/**
 * Approval/denial store with once, session, and explicitly persisted-always scopes.
 *
 * JavaScript's event loop serializes mutations in one runtime; the store deliberately owns no
 * process-global singleton. Hosts that want durable approvals pass one explicit path.
 */
export class ApprovalStore {
  readonly persistencePath: string | undefined
  private readonly now: () => Date
  private records: ApprovalRecord[]

  constructor(options: ApprovalStoreOptions = {}) {
    this.persistencePath = options.persistencePath?.trim() ? resolve(options.persistencePath) : undefined
    this.now = options.now ?? (() => new Date())
    this.records = this.load()
  }

  /** Append a decision and atomically persist all ALWAYS decisions when configured. */
  add(input: ApprovalRecord | ApprovalRecordInput): ApprovalRecord {
    const record = input instanceof ApprovalRecord
      ? input
      : new ApprovalRecord({ ...input, ...(input.createdAt === undefined ? { createdAt: this.timestamp() } : {}) })
    this.records.push(record)
    if (record.scope === ApprovalScope.ALWAYS) this.flush()
    return record
  }

  /** Return a matching grant/denial, or undefined when the caller must prompt the user. */
  check(toolName: string, sessionId: string, argsHash = ''): boolean | undefined {
    const tool = requiredText(toolName, 'toolName')
    const session = optionalText(sessionId)
    const hash = optionalText(argsHash)
    for (let index = this.records.length - 1; index >= 0; index -= 1) {
      const record = this.records[index]
      if (!record || record.toolName !== tool) continue
      if (record.scope === ApprovalScope.ALWAYS) return record.granted
      if (record.scope === ApprovalScope.SESSION && record.sessionId === session) return record.granted
      if (record.scope === ApprovalScope.ONCE && record.sessionId === session && record.argsHash === hash) {
        return record.granted
      }
    }
    return undefined
  }

  /** Return defensive records in insertion order. */
  list(): readonly ApprovalRecord[] {
    return this.records.map(record => new ApprovalRecord({
      toolName: record.toolName,
      scope: record.scope,
      granted: record.granted,
      sessionId: record.sessionId,
      argsHash: record.argsHash,
      createdAt: record.createdAt,
    }))
  }

  /** Drop transient records belonging to one session. Durable approvals are retained. */
  clearSession(sessionId: string): number {
    const session = optionalText(sessionId)
    const before = this.records.length
    this.records = this.records.filter(record => record.scope === ApprovalScope.ALWAYS || record.sessionId !== session)
    return before - this.records.length
  }

  private load(): ApprovalRecord[] {
    if (!this.persistencePath) return []
    try {
      const raw = JSON.parse(readFileSync(this.persistencePath, 'utf8')) as unknown
      if (!Array.isArray(raw)) return []
      return raw.map(item => ApprovalRecord.fromRecord(item)).filter(record => record.scope === ApprovalScope.ALWAYS)
    } catch {
      return []
    }
  }

  private flush(): void {
    if (!this.persistencePath) return
    const directory = dirname(this.persistencePath)
    const temporary = this.persistencePath + '.' + randomUUID() + '.tmp'
    const durable = this.records.filter(record => record.scope === ApprovalScope.ALWAYS).map(record => record.toRecord())
    mkdirSync(directory, { recursive: true, mode: 0o700 })
    try {
      writeFileSync(temporary, JSON.stringify(durable, null, 2) + '\n', { encoding: 'utf8', mode: 0o600 })
      renameSync(temporary, this.persistencePath)
      chmodSync(this.persistencePath, 0o600)
    } finally {
      rmSync(temporary, { force: true })
    }
  }

  private timestamp(): string {
    const date = this.now()
    if (Number.isNaN(date.valueOf())) throw new RangeError('now must return a valid date')
    return date.toISOString()
  }
}

/** Produce an opaque stable hash for a tool-call argument payload. */
export function approvalArgumentsHash(value: unknown): string {
  return createHash('sha256').update(stableJson(value), 'utf8').digest('hex')
}

function requiredText(value: string, name: string): string {
  if (typeof value !== 'string') throw new TypeError(name + ' must be a string')
  const text = value.trim()
  if (!text || text.includes('\0')) throw new TypeError(name + ' must be a non-empty safe string')
  return text
}

function optionalText(value: string | undefined): string {
  if (value === undefined) return ''
  if (typeof value !== 'string' || value.includes('\0')) throw new TypeError('approval record text must be a safe string')
  return value
}

function requireScope(value: unknown): ApprovalScope {
  if (value === ApprovalScope.ONCE || value === ApprovalScope.SESSION || value === ApprovalScope.ALWAYS) return value
  throw new TypeError('scope must be once, session, or always')
}

function normalizeTimestamp(value: string | undefined): string {
  if (value === undefined) return new Date().toISOString()
  if (typeof value !== 'string' || Number.isNaN(Date.parse(value))) throw new TypeError('createdAt must be an ISO timestamp')
  return value
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function requiredValue(value: unknown, name: string): string {
  if (typeof value !== 'string') throw new TypeError(name + ' must be a string')
  return value
}

function optionalStringValue(value: unknown, name: string): string | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'string') throw new TypeError(name + ' must be a string')
  return value
}

function requiredScopeValue(value: unknown): ApprovalScope {
  return requireScope(value)
}

function requiredBoolean(value: unknown, name: string): boolean {
  if (typeof value !== 'boolean') throw new TypeError(name + ' must be a boolean')
  return value
}

function stableJson(value: unknown): string {
  if (value === null) return 'null'
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') return Number.isFinite(value) ? JSON.stringify(value) : JSON.stringify(String(value))
  if (typeof value === 'bigint') return JSON.stringify(value.toString())
  if (Array.isArray(value)) return '[' + value.map(stableJson).join(',') + ']'
  if (isRecord(value)) {
    return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + stableJson(value[key])).join(',') + '}'
  }
  return JSON.stringify(String(value))
}
