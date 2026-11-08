// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { DataBoundaryMode } from '../security/boundaryTypes.js'
import { parseDataBoundaryConfig } from '../security/dataBoundary.js'
import { DisclosureLog, renderDisclosures } from '../security/disclosureLog.js'
import { DEFAULT_SOURCE_SESSION, getSourceLog, renderSources, type SourceLog } from '../tools/sourceLog.js'

const MODES: readonly DataBoundaryMode[] = ['off', 'private', 'hybrid', 'cloud']

export const BOUNDARY_USAGE = [
  'Usage:',
  '  /boundary                     show the data boundary for this workspace',
  '  /boundary off|private|hybrid|cloud   switch the mode (off = today\'s behavior)',
  '  /boundary log [n] [--all]     what left the machine, newest first (this session unless --all)',
  'Private paths: add globs under `private_paths` in .xerxes/data-policy.yaml or runtime.boundary.private_paths.',
].join('\n')

export interface BoundaryCommandPorts {
  readonly cwd: string
  readonly sessionId?: string
  /** The daemon's merged runtime settings. */
  readonly runtime: Record<string, unknown>
  readonly env?: Record<string, string | undefined>
  /** Persist the new mode in the live runtime settings. */
  readonly setMode: (mode: DataBoundaryMode) => void
  readonly log?: DisclosureLog
}

export interface BoundaryCommandResult {
  readonly ok: boolean
  readonly report: string
  readonly mode?: DataBoundaryMode
  readonly [key: string]: unknown
}

/** `/boundary`: show, switch or audit the local/cloud data boundary. */
export function boundaryCommand(args: string, ports: BoundaryCommandPorts): BoundaryCommandResult {
  const env = ports.env ?? process.env
  const words = args.trim().split(/\s+/).filter(Boolean)
  const action = (words[0] ?? 'status').toLowerCase()
  if (action === 'help') return { ok: true, report: BOUNDARY_USAGE }
  if (action === 'status') {
    try {
      const config = parseDataBoundaryConfig(ports.runtime, env, { root: ports.cwd })
      const lines = [
        `Data boundary: ${config.mode.toUpperCase()}${env.XERXES_DATA_BOUNDARY?.trim() ? ' (set by XERXES_DATA_BOUNDARY)' : ''}`,
      ]
      if (config.mode === 'off') {
        lines.push('Nothing is gated; Xerxes behaves as it always has. Turn it on with /boundary hybrid or /boundary private.')
      } else {
        lines.push(
          `Outbound approval: ${config.approveOutbound}`,
          `Inputs read-only: ${config.readOnlyInputs ? `yes (write new files or into ${config.outputDir}/)` : 'no'}`,
          `Private paths: ${config.privatePaths.length ? config.privatePaths.join(', ') : 'none (add private_paths in .xerxes/data-policy.yaml)'}`,
        )
      }
      return { ok: true, report: lines.join('\n'), mode: config.mode }
    } catch (error) {
      return { ok: false, report: `Data boundary settings are invalid: ${error instanceof Error ? error.message : String(error)}` }
    }
  }
  if ((MODES as readonly string[]).includes(action)) {
    if (words.length > 1) return { ok: false, report: BOUNDARY_USAGE }
    const mode = action as DataBoundaryMode
    if (env.XERXES_DATA_BOUNDARY?.trim() && env.XERXES_DATA_BOUNDARY.trim().toLowerCase() !== mode) {
      return { ok: false, report: `XERXES_DATA_BOUNDARY=${env.XERXES_DATA_BOUNDARY.trim()} overrides the setting; unset it to switch with /boundary.` }
    }
    ports.setMode(mode)
    const note = mode === 'off'
      ? 'Data boundary OFF: nothing is gated.'
      : mode === 'private'
        ? 'Data boundary PRIVATE: cloud-trust models are refused; only local models may run tools on this workspace.'
        : mode === 'hybrid'
          ? 'Data boundary HYBRID: private paths stay local, risky shell commands ask first, inputs are read-only.'
          : 'Data boundary CLOUD: tools and permissions apply, but selected context may leave the machine.'
    return { ok: true, report: note, mode }
  }
  if (action === 'log') {
    const all = words.includes('--all')
    const countWord = words.slice(1).find(word => /^\d+$/.test(word))
    const limit = countWord === undefined ? 20 : Math.min(200, Math.max(1, Number(countWord)))
    const unknown = words.slice(1).filter(word => word !== '--all' && word !== countWord)
    if (unknown.length) return { ok: false, report: BOUNDARY_USAGE }
    const log = ports.log ?? new DisclosureLog()
    const records = log.read({ limit, ...(all || ports.sessionId === undefined ? {} : { sessionId: ports.sessionId }) })
    return { ok: true, report: renderDisclosures(records), count: records.length }
  }
  return { ok: false, report: BOUNDARY_USAGE }
}

/** `/sources`: the research sources the session's tools recorded. */
export function sourcesCommand(args: string, sessionId: string | undefined, log: SourceLog = getSourceLog()): BoundaryCommandResult {
  const action = args.trim().toLowerCase()
  const id = sessionId ?? DEFAULT_SOURCE_SESSION
  if (action === 'clear') {
    log.clear(id)
    return { ok: true, report: 'Cleared the recorded sources for this session.' }
  }
  if (action && action !== 'list') return { ok: false, report: 'Usage: /sources [list|clear]' }
  const list = log.list(id)
  return { ok: true, report: renderSources(list), count: list.length }
}
