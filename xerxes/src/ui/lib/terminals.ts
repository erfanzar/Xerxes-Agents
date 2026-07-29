// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { GatewayRpc } from '../app/interfaces.js'

/** Wire view of one terminal row from the daemon `terminal.list` RPC. */
export interface TerminalSummary {
  canInterrupt: boolean
  canKill: boolean
  canWrite: boolean
  command: string
  cwd: string
  endedAt?: number
  exitCode: null | number
  id: string
  kind: 'background' | 'foreground' | 'pty'
  label: string
  outputChars: number
  pid?: number
  running: boolean
  startedAt: number
}

export interface TerminalInspection extends TerminalSummary {
  output: string
  outputTruncated: boolean
}

interface TerminalListResponse {
  error?: string
  ok?: boolean
  terminals?: unknown
}

interface TerminalDetailResponse {
  error?: string
  ok?: boolean
  terminal?: unknown
}

const KINDS = new Set(['background', 'foreground', 'pty'])

const str = (value: unknown, fallback = ''): string => (typeof value === 'string' ? value : fallback)
const num = (value: unknown): number => (typeof value === 'number' && Number.isFinite(value) ? value : 0)
const bool = (value: unknown): boolean => value === true

/**
 * Narrow one wire row.
 *
 * Returns null rather than a partially-filled row for anything without an id:
 * a terminal the panel cannot address is a row whose every action would fail.
 */
function asSummary(value: unknown): null | TerminalSummary {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const row = value as Record<string, unknown>
  const id = str(row.id).trim()
  if (!id) return null
  const kind = str(row.kind)
  const exitCode = typeof row.exitCode === 'number' && Number.isFinite(row.exitCode) ? row.exitCode : null
  const endedAt = typeof row.endedAt === 'number' && Number.isFinite(row.endedAt) ? row.endedAt : undefined

  return {
    id,
    kind: KINDS.has(kind) ? (kind as TerminalSummary['kind']) : 'foreground',
    label: str(row.label) || str(row.command) || id,
    command: str(row.command),
    cwd: str(row.cwd),
    running: bool(row.running),
    exitCode,
    startedAt: num(row.startedAt),
    outputChars: num(row.outputChars),
    canWrite: bool(row.canWrite),
    canInterrupt: bool(row.canInterrupt),
    canKill: bool(row.canKill),
    ...(typeof row.pid === 'number' ? { pid: row.pid } : {}),
    ...(endedAt === undefined ? {} : { endedAt })
  }
}

/** Every terminal the daemon is tracking, running first and newest first within each group. */
export async function listTerminals(rpc: GatewayRpc): Promise<TerminalSummary[]> {
  const response = await rpc<TerminalListResponse>('terminal.list', {})
  const rows = Array.isArray(response?.terminals) ? response.terminals : []

  return rows
    .map(asSummary)
    .filter((row): row is TerminalSummary => row !== null)
    .sort((left, right) => {
      if (left.running !== right.running) return left.running ? -1 : 1
      return (right.endedAt ?? right.startedAt) - (left.endedAt ?? left.startedAt)
    })
}

/** One terminal with the retained tail of its output. */
export async function inspectTerminal(
  rpc: GatewayRpc,
  terminalId: string,
  maxOutputChars = 60_000
): Promise<null | TerminalInspection> {
  const response = await rpc<TerminalDetailResponse>('terminal.inspect', {
    terminal_id: terminalId,
    max_output_chars: maxOutputChars
  })
  const summary = asSummary(response?.terminal)
  if (!summary) return null
  const detail = response?.terminal as Record<string, unknown>

  return { ...summary, output: str(detail.output), outputTruncated: bool(detail.outputTruncated) }
}

export type TerminalAction = 'interrupt' | 'kill' | 'write'

/**
 * Drive one live terminal.
 *
 * Resolves to an error string instead of throwing: every caller is a keypress
 * handler that must show what went wrong rather than unwind the render.
 */
export async function controlTerminal(
  rpc: GatewayRpc,
  terminalId: string,
  action: TerminalAction,
  options: { chars?: string; force?: boolean } = {}
): Promise<null | string> {
  try {
    const response = await rpc<TerminalDetailResponse>('terminal.control', {
      terminal_id: terminalId,
      action,
      ...(options.chars === undefined ? {} : { chars: options.chars }),
      ...(options.force ? { signal: 'SIGKILL' } : {})
    })
    if (response?.ok === false) return response.error?.trim() || 'the daemon rejected the request'
    return null
  } catch (error) {
    return error instanceof Error && error.message ? error.message : 'request failed'
  }
}

/** Compact "2m 14s" style age used by both the list and the detail header. */
export function terminalAge(entry: TerminalSummary, now: number): string {
  const end = entry.running ? now : (entry.endedAt ?? now)
  const seconds = Math.max(0, Math.round((end - entry.startedAt) / 1_000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`

  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`
}

/** One-word state for the row, distinguishing a clean exit from a failure. */
export function terminalState(entry: TerminalSummary): 'exited' | 'failed' | 'running' {
  if (entry.running) return 'running'

  return entry.exitCode === 0 || entry.exitCode === null ? 'exited' : 'failed'
}
