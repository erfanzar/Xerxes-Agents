// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Hub health for the desktop notice. Reads the saved Hub address
 * (`desktop-hub.json`), asks the Hub's public `/healthz` and reports one of:
 * not configured, online, degraded (some workspaces offline) or unreachable.
 * Each notice kind has its own "Don't remind me", kept in
 * `desktop-hub-notice.json`. Nothing here sends a cookie or a credential:
 * `/healthz` is the one Hub route that needs neither.
 *
 * The file, HTTP and clock access are injected, so tests drive it directly.
 */

import http from 'node:http'
import https from 'node:https'
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

import {
  HUB_HEALTH_ACTIONS, HUB_HEALTH_CHANNEL_PREFIX, asHubWorkspaceCounts, isHubHealthAction, isHubNoticeKind,
  type HubHealthAction, type HubHealthDismissed, type HubHealthResult, type HubHealthState,
} from '../hubHealthShared.js'
import { HUB_SETTINGS_FILE, parseStoredHub } from './hubStorage.js'

export const HUB_NOTICE_FILE = 'desktop-hub-notice.json'
/** How long one `/healthz` request may take before the Hub counts as unreachable. */
export const HUB_HEALTH_TIMEOUT_MS = 10_000
/** Coming back to a window checks again at most this often. */
export const HUB_FOCUS_RECHECK_MS = 30_000
const MAX_BODY_BYTES = 64 * 1024

export interface HubHealthPort {
  /** The saved Hub, or undefined when none is set up. */
  readHub(): { readonly target: string; readonly origin: string } | undefined
  /** GET `url`; rejects when the Hub does not answer. */
  get(url: string, timeoutMs: number): Promise<{ readonly status: number; readonly body: string }>
  readDismissed(): HubHealthDismissed
  writeDismissed(value: HubHealthDismissed): void
  now(): number
}

export interface HubHealth {
  handle(action: unknown, params: unknown): Promise<HubHealthResult>
  /** Check now; concurrent callers share one request. */
  check(): Promise<HubHealthState>
  /** A window regained focus: check again unless one ran recently. */
  recheckOnFocus(): void
  state(): HubHealthState
  onChange(listener: (state: HubHealthState) => void): () => void
}

function oneLine(text: string, max = 200): string {
  // eslint-disable-next-line no-control-regex
  return text.replace(/[\u0000-\u001f\u007f]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, max)
}

function bodyOf(text: string): Record<string, unknown> {
  try {
    const value: unknown = JSON.parse(text)
    return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
  } catch { return {} }
}

/** What one `/healthz` answer means. Exported for tests. */
export function readHealthz(status: number, text: string): Pick<HubHealthState, 'kind' | 'detail' | 'workspaces'> {
  if (status < 200 || status >= 300) return { kind: 'unreachable', detail: `the Hub answered HTTP ${status}`, workspaces: null }
  const body = bodyOf(text)
  const workspaces = asHubWorkspaceCounts(body.workspaces ?? null) ?? null
  if (body.status === 'identity_unavailable') return { kind: 'degraded', detail: 'the Hub is running without its sign-in identity', workspaces }
  if (body.status === 'degraded' && workspaces) {
    const offline = workspaces.total - workspaces.online
    return { kind: 'degraded', detail: offline > 0 ? `${offline} of ${workspaces.total} workspace${workspaces.total === 1 ? '' : 's'} offline` : 'some workspaces report problems', workspaces }
  }
  if (body.status === 'ok' || body.status === 'degraded') return { kind: body.status === 'ok' ? 'online' : 'degraded', detail: null, workspaces }
  return { kind: 'unreachable', detail: 'the address answered, but not as a Xerxes Hub', workspaces: null }
}

export function createHubHealth(port: HubHealthPort): HubHealth {
  const listeners = new Set<(state: HubHealthState) => void>()
  let dismissed = port.readDismissed()
  let current: HubHealthState = { kind: 'unknown', target: null, origin: null, detail: null, workspaces: null, checked_at: null, dismissed }
  let running: Promise<HubHealthState> | undefined
  let lastCheck = -Infinity

  const publish = (next: HubHealthState): HubHealthState => {
    current = next
    for (const listener of listeners) {
      try { listener(next) } catch { /* one page's fault must not stop the others */ }
    }
    return next
  }

  const run = async (): Promise<HubHealthState> => {
    lastCheck = port.now()
    const hub = port.readHub()
    const at = new Date(port.now()).toISOString()
    if (!hub) return publish({ kind: 'not_configured', target: null, origin: null, detail: null, workspaces: null, checked_at: at, dismissed })
    let verdict: Pick<HubHealthState, 'kind' | 'detail' | 'workspaces'>
    try {
      const response = await port.get(`${hub.origin.replace(/\/+$/, '')}/healthz`, HUB_HEALTH_TIMEOUT_MS)
      verdict = readHealthz(response.status, response.body)
    } catch (error) {
      verdict = { kind: 'unreachable', detail: oneLine(error instanceof Error ? error.message : String(error)) || 'no answer', workspaces: null }
    }
    return publish({ ...verdict, target: hub.target, origin: hub.origin, checked_at: new Date(port.now()).toISOString(), dismissed })
  }

  const check = (): Promise<HubHealthState> => {
    running ??= run().finally(() => { running = undefined })
    return running
  }

  const setDismissed = (kind: unknown, value: boolean): HubHealthResult => {
    if (!isHubNoticeKind(kind)) return { ok: false, code: 'invalid_request', message: 'kind must be not_configured or unreachable' }
    const next = { ...dismissed, [kind]: value }
    port.writeDismissed(next)
    dismissed = next
    return { ok: true, state: publish({ ...current, dismissed }) }
  }

  return {
    async handle(action, params) {
      if (!isHubHealthAction(action)) return { ok: false, code: 'unknown_action', message: 'Unknown Hub health action' }
      const fields = params && typeof params === 'object' && !Array.isArray(params) ? params as Record<string, unknown> : {}
      const choice: HubHealthAction = action
      switch (choice) {
        case 'status': return { ok: true, state: current.kind === 'unknown' ? await check() : current }
        case 'recheck': return { ok: true, state: await check() }
        case 'dismiss': return setDismissed(fields.kind, true)
        case 'remind': return setDismissed(fields.kind, false)
      }
    },
    check,
    recheckOnFocus() {
      if (running || port.now() - lastCheck < HUB_FOCUS_RECHECK_MS) return
      void check()
    },
    state: () => current,
    onChange(listener) {
      listeners.add(listener)
      return () => { listeners.delete(listener) }
    },
  }
}

function nodeGet(url: string, timeoutMs: number): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    let parsed: URL
    try { parsed = new URL(url) } catch { reject(new Error('the saved Hub address is not a valid URL')); return }
    const secure = parsed.protocol === 'https:'
    if (!secure && parsed.protocol !== 'http:') { reject(new Error('the saved Hub address is not http(s)')); return }
    const request = (secure ? https : http).get(parsed, { agent: false, timeout: timeoutMs, headers: { accept: 'application/json' } }, response => {
      const chunks: Buffer[] = []
      let size = 0
      response.on('data', (chunk: Buffer) => {
        size += chunk.length
        if (size > MAX_BODY_BYTES) { response.destroy(new Error('the Hub answered with too much data')); return }
        chunks.push(chunk)
      })
      response.on('error', reject)
      response.on('end', () => resolve({ status: response.statusCode ?? 0, body: Buffer.concat(chunks).toString('utf8') }))
    })
    request.on('timeout', () => request.destroy(new Error(`no answer within ${Math.round(timeoutMs / 1000)} s`)))
    request.on('error', reject)
  })
}

/** Real files and HTTP for the Electron main process. Each notice is turned off by its own "Don't remind me". */
export function createNodeHubHealthPort(options: { readonly home: string }): HubHealthPort {
  const noticeFile = join(options.home, HUB_NOTICE_FILE)
  return {
    readHub() {
      const path = join(options.home, HUB_SETTINGS_FILE)
      if (!existsSync(path)) return undefined
      try {
        const hub = parseStoredHub(readFileSync(path, 'utf8'))
        return hub ? { target: hub.target, origin: hub.origin } : undefined
      } catch { return undefined }
    },
    get: nodeGet,
    readDismissed() {
      try {
        const raw = bodyOf(readFileSync(noticeFile, 'utf8'))
        return { not_configured: raw.not_configured === true, unreachable: raw.unreachable === true }
      } catch { return { not_configured: false, unreachable: false } }
    },
    writeDismissed(value) {
      mkdirSync(dirname(noticeFile), { recursive: true })
      const temp = `${noticeFile}.tmp-${process.pid}`
      writeFileSync(temp, `${JSON.stringify({ version: 1, ...value }, null, 2)}\n`, { mode: 0o600 })
      renameSync(temp, noticeFile)
    },
    now: () => Date.now(),
  }
}

export interface HubHealthIpcMain {
  handle(channel: string, listener: (event: unknown, ...args: unknown[]) => unknown): void
}

export function hubHealthChannels(): readonly string[] {
  return HUB_HEALTH_ACTIONS.map(action => HUB_HEALTH_CHANNEL_PREFIX + action)
}

/** One channel per action; a thrown error never reaches the renderer raw. */
export function registerHubHealthIpc(ipc: HubHealthIpcMain, health: HubHealth): void {
  for (const action of HUB_HEALTH_ACTIONS) {
    ipc.handle(HUB_HEALTH_CHANNEL_PREFIX + action, async (_event, params) => {
      try {
        return await health.handle(action, params ?? {})
      } catch (error) {
        const failure: HubHealthResult = { ok: false, code: 'failed', message: oneLine(error instanceof Error ? error.message : String(error)) }
        return failure
      }
    })
  }
}
