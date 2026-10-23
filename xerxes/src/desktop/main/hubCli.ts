// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Runs the Xerxes runtime CLI (`hub ... --json`) as a child of the Electron
 * main process and turns its single JSON object into validated values.
 *
 * Electron's main process is not Bun, so the CLI is launched the same way the
 * daemon is: the Bun binary and the runtime entry resolved by `spawn.ts`. There
 * is no shell: the argv is built from already validated pieces.
 */

import { spawn } from 'node:child_process'
import { homedir } from 'node:os'

import {
  checkComparisonCode, checkPairingId, qrDataUrl,
  type CockpitFailure, type CockpitFailureCode, type CockpitLink, type CockpitMachine, type CockpitPairing, type CockpitProject,
} from '../hubShared.js'
import { bunBinaryOf, daemonEntryOf, type Env } from './spawn.js'

export interface CliOutcome {
  readonly exitCode: number | null
  readonly stdout: string
  readonly stderr: string
  readonly timedOut: boolean
  /** Set when the process could not be started or was stopped by this runner. */
  readonly spawnError?: string
}

export interface CliRunOptions {
  readonly timeoutMs: number
  /** Each complete stderr line as it arrives (progress for long commands). */
  readonly onStderrLine?: (line: string) => void
}

export interface CliRunner {
  /** `args` are the arguments after `hub`, for example `['remote', 'xerxes@host', 'info', '--json']`. */
  run(args: readonly string[], options: CliRunOptions): Promise<CliOutcome>
}

const MAX_STDOUT_BYTES = 1024 * 1024
const MAX_STDERR_CHARS = 16_384

export function createCliRunner(env: Env = process.env, projectDir: string = homedir()): CliRunner {
  return {
    run(args, options) {
      let binary: string
      let entry: string
      try {
        binary = bunBinaryOf(env)
        entry = daemonEntryOf(projectDir, env)
      } catch (error) {
        return Promise.resolve({ exitCode: null, stdout: '', stderr: '', timedOut: false, spawnError: error instanceof Error ? error.message : String(error) })
      }
      return new Promise<CliOutcome>(resolve => {
        let stdout = ''
        let stdoutBytes = 0
        let stderr = ''
        let pending = ''
        let timedOut = false
        let stopped: string | undefined
        let settled = false
        const child = spawn(binary, [entry, 'hub', ...args], { stdio: ['ignore', 'pipe', 'pipe'], env: env as NodeJS.ProcessEnv })
        const finish = (outcome: CliOutcome): void => {
          if (settled) return
          settled = true
          clearTimeout(timer)
          resolve(outcome)
        }
        const timer = setTimeout(() => {
          timedOut = true
          child.kill('SIGKILL')
        }, options.timeoutMs)
        child.stdout?.setEncoding('utf8')
        child.stderr?.setEncoding('utf8')
        child.stdout?.on('data', (chunk: string) => {
          stdoutBytes += Buffer.byteLength(chunk)
          if (stdoutBytes > MAX_STDOUT_BYTES) {
            stopped = 'The command printed more than 1 MiB.'
            child.kill('SIGKILL')
            return
          }
          stdout += chunk
        })
        child.stderr?.on('data', (chunk: string) => {
          stderr = (stderr + chunk).slice(-MAX_STDERR_CHARS)
          pending += chunk
          let index = pending.indexOf('\n')
          while (index >= 0) {
            const line = pending.slice(0, index).replace(/\r$/, '')
            pending = pending.slice(index + 1)
            if (line.trim() !== '') options.onStderrLine?.(line)
            index = pending.indexOf('\n')
          }
        })
        child.once('error', error => finish({ exitCode: null, stdout, stderr, timedOut, spawnError: error.message }))
        child.once('close', code => {
          if (pending.trim() !== '') options.onStderrLine?.(pending.trim())
          finish({ exitCode: code, stdout, stderr, timedOut, ...(stopped === undefined ? {} : { spawnError: stopped }) })
        })
      })
    },
  }
}

// ── output parsing ──────────────────────────────────────────────────────

type JsonObject = Readonly<Record<string, unknown>>

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function tryParse(text: string): JsonObject | undefined {
  try {
    const value: unknown = JSON.parse(text)
    return isObject(value) ? value : undefined
  } catch {
    return undefined
  }
}

/**
 * The one JSON object a `--json` command prints. Tolerant of what rides along:
 * the whole output may be (pretty-printed) JSON, or an object on its own line
 * after banner or warning lines. The LAST such line wins. Anything else is
 * `undefined`: nothing is guessed.
 */
export function parseCliJson(stdout: string): JsonObject | undefined {
  const text = stdout.trim()
  if (text === '') return undefined
  const whole = tryParse(text)
  if (whole !== undefined) return whole
  const lines = text.split('\n')
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index]!.trim()
    if (!line.startsWith('{')) continue
    const found = tryParse(line)
    if (found !== undefined) return found
  }
  return undefined
}

function failure(code: CockpitFailureCode, message: string): CockpitFailure {
  return { ok: false, code, message }
}

function oneLine(text: string, max = 300): string {
  // eslint-disable-next-line no-control-regex
  return text.replace(/[\u0000-\u001f\u007f]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, max)
}

/** What a failed process says, in one line fit for the panel. */
export function describeCliFailure(outcome: CliOutcome, action: string): CockpitFailure {
  if (outcome.spawnError !== undefined) {
    return failure('cli_unavailable', `Could not run the Xerxes runtime to ${action}: ${oneLine(outcome.spawnError)}`)
  }
  if (outcome.timedOut) return failure('timeout', `Timed out while trying to ${action}. Check that the Hub host is reachable over SSH.`)
  const parsed = parseCliJson(outcome.stdout)
  const reported = parsed === undefined ? undefined : typeof parsed.error === 'string' ? parsed.error : isObject(parsed.error) && typeof parsed.error.message === 'string' ? parsed.error.message : undefined
  if (reported !== undefined && oneLine(reported) !== '') return failure('cli_failed', oneLine(reported))
  const lines = outcome.stderr.split('\n').map(line => oneLine(line)).filter(line => line !== '')
  const last = lines.at(-1)
  return failure('cli_failed', last !== undefined ? last : `Could not ${action} (exit code ${outcome.exitCode ?? 'unknown'}).`)
}

/** The JSON object of a finished command, or the failure to show. */
export function jsonOf(outcome: CliOutcome, action: string): { readonly ok: true; readonly json: JsonObject } | CockpitFailure {
  if (outcome.spawnError !== undefined || outcome.timedOut || outcome.exitCode !== 0) return describeCliFailure(outcome, action)
  const json = parseCliJson(outcome.stdout)
  if (json === undefined) return failure('bad_output', `The Xerxes runtime answered with something this app cannot read while trying to ${action}.`)
  return { ok: true, json }
}

function text(value: unknown, max = 2048): string | undefined {
  return typeof value === 'string' && value.length > 0 && value.length <= max ? value : undefined
}

export interface HubInfo {
  readonly hubId: string | null
  readonly fingerprint: string | null
  readonly publicOrigin: string | null
  readonly version: string | null
}

export function parseHubInfo(json: JsonObject): HubInfo | CockpitFailure {
  const origin = json.public_origin
  if (origin !== undefined && origin !== null && typeof origin !== 'string') return failure('bad_output', 'The Hub reported a web address this app cannot read.')
  const fingerprint = text(json.fingerprint, 200)
  const hubId = text(json.hub_id, 200)
  if (fingerprint === undefined && hubId === undefined) return failure('bad_output', 'That did not look like a Xerxes Hub answer. Check the address.')
  return {
    hubId: hubId ?? null,
    fingerprint: fingerprint ?? null,
    publicOrigin: typeof origin === 'string' && origin.trim() !== '' ? origin.trim() : null,
    version: text(json.version, 64) ?? null,
  }
}

export function parsePairing(json: JsonObject): CockpitPairing | CockpitFailure {
  const id = checkPairingId(json.pairing_id)
  const code = checkComparisonCode(json.comparison_code)
  const url = text(json.url, 4096)
  const expires = text(json.expires_at, 64)
  if (!id.ok || !code.ok || url === undefined || expires === undefined || !/^https?:\/\//i.test(url)) {
    return failure('bad_output', 'The Hub answered the pairing request with something this app cannot read.')
  }
  return {
    pairingId: id.value,
    url,
    comparisonCode: code.value,
    expiresAt: expires,
    confirmRequired: json.confirm_required !== false,
    qrDataUrl: qrDataUrl(json.qr_svg) ?? null,
  }
}

export function parseConfirm(json: JsonObject): { readonly deviceId: string } | CockpitFailure {
  const deviceId = text(json.device_id, 200)
  return deviceId === undefined ? failure('bad_output', 'The Hub did not say which device was confirmed.') : { deviceId }
}

export function parseMachines(json: JsonObject): readonly CockpitMachine[] | CockpitFailure {
  if (!Array.isArray(json.machines)) return failure('bad_output', 'The Hub sent a machine list this app cannot read.')
  const machines: CockpitMachine[] = []
  for (const raw of json.machines.slice(0, 100) as unknown[]) {
    if (!isObject(raw)) continue
    const id = text(raw.id, 200)
    if (id === undefined) continue
    const projects: CockpitProject[] = []
    if (Array.isArray(raw.projects)) {
      for (const item of raw.projects.slice(0, 500) as unknown[]) {
        if (!isObject(item)) continue
        const workspaceId = text(item.workspace_id, 200)
        if (workspaceId === undefined) continue
        projects.push({
          workspaceId,
          projectDir: text(item.project_dir, 4096) ?? '',
          label: text(item.label, 300) ?? workspaceId,
          health: text(item.health, 40) ?? 'unknown',
        })
      }
    }
    machines.push({ id, label: text(raw.label, 300) ?? id, endpointKind: text(raw.endpoint_kind, 40) ?? 'unknown', discover: raw.discover === true, projects })
  }
  return machines
}

export function parseLink(json: JsonObject): CockpitLink | CockpitFailure {
  const machineId = text(json.machine_id, 200)
  if (machineId === undefined) return failure('bad_output', 'The Hub did not report which machine was linked.')
  if (json.enrolled !== true) return failure('cli_failed', 'The Hub was added, but this Mac did not accept it. Try linking again.')
  return { machineId, remoteSocket: text(json.remote_socket, 4096) ?? '', servicePath: text(json.service_path, 4096) ?? null }
}

export function isFailure(value: object): value is CockpitFailure {
  return (value as { ok?: unknown }).ok === false
}
