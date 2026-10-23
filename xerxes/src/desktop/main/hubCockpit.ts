// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The desktop's Hub cockpit, in the Electron main process.
 *
 * It owns everything that must not reach the renderer: the SSH-driven CLI
 * calls, the desktop's pairing secret, the Hub session cookies, and the Hub
 * WebSocket (opened with the Hub's own `Origin` and those cookies). The
 * renderer only sees validated results and bridged text frames.
 *
 * Every Hub-host action goes through `hub remote <target> ...`; this Mac's own
 * enrollment goes through `hub link <target>`. Nothing here talks to a shell:
 * targets are validated first and arguments are passed as an argv list.
 *
 * The module has no Electron import. The CLI runner, HTTP client, socket
 * connector, storage and event sink are injected, so tests drive it directly.
 */

import {
  COCKPIT_ACTIONS, COCKPIT_MAX_FRAME_BYTES, COCKPIT_PROGRESS_EVENT, COCKPIT_SOCKET_EVENT,
  checkComparisonCode, checkDeviceLabel, checkHubTarget, checkPairingId, hubWebSocketUrlFor, parseHubOrigin, parsePairingTicket, socketIdOk,
  type CockpitAction, type CockpitFailure, type CockpitFailureCode, type CockpitMachine, type CockpitResult, type CockpitStatus, type HubSocketEvent,
} from '../hubShared.js'
import {
  isFailure, jsonOf, parseConfirm, parseHubInfo, parseLink, parseMachines, parsePairing,
  type CliRunner,
} from './hubCli.js'
import { CookieJar, HUB_REFRESH_COOKIE, HUB_SESSION_COOKIE } from './hubCookies.js'
import type { HubHttp } from './hubHttp.js'
import type { HubSocket, HubSocketConnector } from './hubSocket.js'

/** What is remembered about the Hub between launches. Nothing here is a credential. */
export interface StoredHub {
  readonly target: string
  readonly origin: string
  readonly fingerprint: string | null
  readonly hubVersion: string | null
  readonly linkedMachineId: string | null
}

export interface HubCockpitStorage {
  loadSettings(): StoredHub | undefined
  /** `undefined` forgets the Hub. Throws when it cannot be written. */
  saveSettings(value: StoredHub | undefined): void
  /** The serialized cookie jar, when the platform can keep it encrypted. */
  loadSession(): string | undefined
  saveSession(text: string | undefined): void
}

export interface HubCockpitDeps {
  readonly runner: CliRunner
  readonly http: HubHttp
  readonly connect: HubSocketConnector
  readonly storage: HubCockpitStorage
  readonly emit: (senderId: number, channel: string, payload: unknown) => void
  readonly now?: () => number
  readonly log?: (message: string) => void
}

export interface HubCockpit {
  /** One IPC call. `action` and `params` are untrusted renderer input. */
  handle(action: unknown, params: unknown, senderId: number): Promise<CockpitResult>
  /** The page that owned these sockets is gone. */
  releaseSender(senderId: number): void
  dispose(): void
}

export const CONSUME_PATH = '/api/hub/v1/pair/consume'
export const REFRESH_PATH = '/api/hub/v1/session/refresh'
export const DESKTOP_DEVICE_LABEL = 'Desktop'
export const SOCKET_REJECTED = 'did not accept this sign-in'

const INFO_TIMEOUT_MS = 30_000
const LINK_TIMEOUT_MS = 180_000
const PAIR_TIMEOUT_MS = 45_000
const HTTP_TIMEOUT_MS = 20_000
const SESSION_MARGIN_MS = 60_000
const MAX_SOCKETS = 8

function fail(code: CockpitFailureCode, message: string, denied = false): CockpitFailure {
  return denied ? { ok: false, code, message, denied: true } : { ok: false, code, message }
}

function oneLine(text: string, max = 300): string {
  // eslint-disable-next-line no-control-regex
  return text.replace(/[\u0000-\u001f\u007f]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, max)
}

type Params = Readonly<Record<string, unknown>>

/** `{}` for no params; a failure for anything but a plain object with only the allowed keys. */
function paramsOf(value: unknown, allowed: readonly string[]): Params | CockpitFailure {
  if (value === undefined || value === null) return {}
  if (typeof value !== 'object' || Array.isArray(value)) return fail('invalid_input', 'The request parameters must be an object.')
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) return fail('invalid_input', `Unexpected field "${oneLine(key, 32)}".`)
  }
  return value as Params
}

function bodyOf(text: string): Params {
  try {
    const value: unknown = JSON.parse(text)
    return typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Params) : {}
  } catch {
    return {}
  }
}

function errorMessageOf(body: Params, fallback: string): string {
  const nested = typeof body.error === 'object' && body.error !== null ? (body.error as Params).message : undefined
  if (typeof nested === 'string' && oneLine(nested) !== '') return oneLine(nested)
  return typeof body.message === 'string' && oneLine(body.message) !== '' ? oneLine(body.message) : fallback
}

export function isCockpitActionName(value: unknown): value is CockpitAction {
  return typeof value === 'string' && (COCKPIT_ACTIONS as readonly string[]).includes(value)
}

export function createHubCockpit(deps: HubCockpitDeps): HubCockpit {
  const now = deps.now ?? Date.now
  const jar = new CookieJar()
  const sockets = new Map<string, { readonly senderId: number; readonly socket: HubSocket }>()
  const running = new Set<string>()
  let settings: StoredHub | undefined
  let signingIn: Promise<CockpitResult> | undefined

  try {
    settings = deps.storage.loadSettings()
    const session = deps.storage.loadSession()
    if (session !== undefined) jar.restore(session, now())
  } catch (error) {
    deps.log?.(`Hub settings could not be read: ${error instanceof Error ? error.message : String(error)}`)
  }

  const persistSession = (): void => {
    try {
      deps.storage.saveSession(jar.serialize(now()))
    } catch (error) {
      // The session still works for this run; it just will not survive a restart.
      deps.log?.(`Hub session could not be kept: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  const statusOf = (): CockpitStatus => ({
    target: settings?.target ?? null,
    origin: settings?.origin ?? null,
    wsUrl: settings === undefined ? null : hubWebSocketUrlFor(settings.origin),
    fingerprint: settings?.fingerprint ?? null,
    hubVersion: settings?.hubVersion ?? null,
    linkedMachineId: settings?.linkedMachineId ?? null,
    hasSession: jar.has(HUB_REFRESH_COOKIE, now()) || jar.has(HUB_SESSION_COOKIE, now()),
  })

  const keep = (next: StoredHub | undefined): CockpitFailure | undefined => {
    try {
      deps.storage.saveSettings(next)
    } catch (error) {
      return fail('storage_failed', `This app could not save the Hub address: ${oneLine(error instanceof Error ? error.message : String(error))}`)
    }
    settings = next
    return undefined
  }

  const closeAllSockets = (): void => {
    for (const [, entry] of [...sockets]) {
      try { entry.socket.close(1000, 'closed') } catch { /* already closing */ }
    }
  }

  async function exclusive<T extends CockpitResult>(key: string, task: () => Promise<T>): Promise<T | CockpitFailure> {
    if (running.has(key)) return fail('busy', 'That is already running. Wait for it to finish.')
    running.add(key)
    try {
      return await task()
    } finally {
      running.delete(key)
    }
  }

  const needTarget = (): StoredHub | CockpitFailure =>
    settings ?? fail('no_target', 'Save the Hub SSH address in Settings → Hub & phone first.')

  async function machinesOf(target: string): Promise<{ readonly machines: readonly CockpitMachine[] } | CockpitFailure> {
    const outcome = await deps.runner.run(['remote', target, 'machines', 'list', '--json'], { timeoutMs: INFO_TIMEOUT_MS })
    const json = jsonOf(outcome, 'list the Hub machines')
    if (!json.ok) return json
    const machines = parseMachines(json.json)
    return isFailure(machines) ? machines : { machines }
  }

  // ── actions ───────────────────────────────────────────────────────

  async function save(raw: unknown): Promise<CockpitResult> {
    const params = paramsOf(raw, ['target'])
    if (isFailure(params)) return params
    const checked = checkHubTarget(params.target)
    if (!checked.ok) return fail('invalid_input', checked.message)
    const target = checked.value
    return exclusive('save', async () => {
      const outcome = await deps.runner.run(['remote', target, 'info', '--json'], { timeoutMs: INFO_TIMEOUT_MS })
      const json = jsonOf(outcome, 'read the Hub address')
      if (!json.ok) return json
      const info = parseHubInfo(json.json)
      if (isFailure(info)) return info
      if (info.publicOrigin === null) {
        return fail('no_origin', 'That Hub has not published a web address yet, so this app cannot open it. Finish the Hub setup on the Hub host first.')
      }
      let origin: string
      try {
        origin = parseHubOrigin(info.publicOrigin)
      } catch (error) {
        return fail('no_origin', error instanceof Error ? error.message : 'The Hub web address is not usable.')
      }
      const previous = settings
      const sameHub = previous !== undefined && previous.target === target && previous.origin === origin
      const stored: StoredHub = {
        target, origin, fingerprint: info.fingerprint, hubVersion: info.version,
        linkedMachineId: sameHub ? previous.linkedMachineId : null,
      }
      const failed = keep(stored)
      if (failed) return failed
      if (!sameHub) {
        // A session belongs to one Hub origin.
        closeAllSockets()
        jar.clear()
        persistSession()
      }
      return { ok: true as const, status: statusOf() }
    })
  }

  function forget(raw: unknown): CockpitResult {
    const params = paramsOf(raw, [])
    if (isFailure(params)) return params
    const failed = keep(undefined)
    if (failed) return failed
    closeAllSockets()
    jar.clear()
    persistSession()
    return { ok: true, status: statusOf() }
  }

  async function link(raw: unknown, senderId: number): Promise<CockpitResult> {
    const params = paramsOf(raw, ['target'])
    if (isFailure(params)) return params
    const checked = checkHubTarget(params.target)
    if (!checked.ok) return fail('invalid_input', checked.message)
    const current = needTarget()
    if (isFailure(current)) return current
    if (current.target !== checked.value) return fail('no_target', 'Save this Hub SSH address first, then link this Mac.')
    const target = current.target
    return exclusive('link', async () => {
      const outcome = await deps.runner.run(['link', target, '--json'], {
        timeoutMs: LINK_TIMEOUT_MS,
        onStderrLine: line => deps.emit(senderId, COCKPIT_PROGRESS_EVENT, { op: 'link', line: oneLine(line) }),
      })
      const json = jsonOf(outcome, 'link this Mac')
      if (!json.ok) return json
      const linked = parseLink(json.json)
      if (isFailure(linked)) return linked
      const failed = keep({ ...current, linkedMachineId: linked.machineId })
      if (failed) return failed
      const listed = await machinesOf(target)
      return {
        ok: true as const,
        link: linked,
        machines: isFailure(listed) ? [] : listed.machines,
        machinesError: isFailure(listed) ? listed.message : null,
        status: statusOf(),
      }
    })
  }

  async function pair(raw: unknown): Promise<CockpitResult> {
    const params = paramsOf(raw, ['label'])
    if (isFailure(params)) return params
    const label = checkDeviceLabel(params.label === undefined ? 'Phone' : params.label)
    if (!label.ok) return fail('invalid_input', label.message)
    const current = needTarget()
    if (isFailure(current)) return current
    return exclusive('pair', async () => {
      const outcome = await deps.runner.run(['remote', current.target, 'pair', '--label', label.value, '--json'], { timeoutMs: PAIR_TIMEOUT_MS })
      const json = jsonOf(outcome, 'create a pairing link')
      if (!json.ok) return json
      const pairing = parsePairing(json.json)
      return isFailure(pairing) ? pairing : { ok: true as const, pairing }
    })
  }

  async function confirm(raw: unknown): Promise<CockpitResult> {
    const params = paramsOf(raw, ['pairingId', 'code'])
    if (isFailure(params)) return params
    const id = checkPairingId(params.pairingId)
    if (!id.ok) return fail('invalid_input', id.message)
    const code = checkComparisonCode(params.code)
    if (!code.ok) return fail('invalid_input', code.message)
    const current = needTarget()
    if (isFailure(current)) return current
    return exclusive('confirm', async () => {
      const outcome = await deps.runner.run(['remote', current.target, 'pair', 'confirm', id.value, code.value, '--json'], { timeoutMs: PAIR_TIMEOUT_MS })
      const json = jsonOf(outcome, 'confirm the phone')
      if (!json.ok) return json
      const confirmed = parseConfirm(json.json)
      return isFailure(confirmed) ? confirmed : { ok: true as const, deviceId: confirmed.deviceId }
    })
  }

  async function machines(raw: unknown): Promise<CockpitResult> {
    const params = paramsOf(raw, [])
    if (isFailure(params)) return params
    const current = needTarget()
    if (isFailure(current)) return current
    const listed = await machinesOf(current.target)
    return isFailure(listed) ? listed : { ok: true, machines: listed.machines }
  }

  // ── sign-in ───────────────────────────────────────────────────────

  async function refresh(origin: string): Promise<'ok' | 'denied' | CockpitFailure> {
    let response
    try {
      response = await deps.http({
        url: `${origin}${REFRESH_PATH}`,
        headers: { origin, 'content-type': 'application/json', cookie: jar.header([HUB_REFRESH_COOKIE], now()) },
        body: '{}',
        timeoutMs: HTTP_TIMEOUT_MS,
      })
    } catch (error) {
      return fail('hub_unreachable', `The Hub did not answer: ${oneLine(error instanceof Error ? error.message : String(error))}`)
    }
    if (response.status === 401 || response.status === 403) {
      jar.clear()
      persistSession()
      return 'denied'
    }
    if (response.status === 202) return fail('not_signed_in', 'The Hub is waiting for this desktop to be confirmed.')
    if (response.status < 200 || response.status >= 300) {
      return fail('hub_unreachable', errorMessageOf(bodyOf(response.body), `The Hub answered ${response.status}. Try again in a moment.`))
    }
    for (const cookie of response.setCookies) jar.store(cookie, now())
    persistSession()
    return jar.has(HUB_SESSION_COOKIE, now(), SESSION_MARGIN_MS) ? 'ok' : fail('bad_output', 'The Hub signed this desktop in without a session. Try again.')
  }

  async function pairDesktop(current: StoredHub): Promise<CockpitResult> {
    const outcome = await deps.runner.run(['remote', current.target, 'pair', '--no-confirm', '--label', DESKTOP_DEVICE_LABEL, '--json'], { timeoutMs: PAIR_TIMEOUT_MS })
    const json = jsonOf(outcome, 'sign this desktop in to the Hub')
    if (!json.ok) return json
    const pairing = parsePairing(json.json)
    if (isFailure(pairing)) return pairing
    const ticket = parsePairingTicket(pairing.url)
    if (ticket === undefined) return fail('bad_output', 'The Hub pairing link had a shape this app cannot use.')
    let response
    try {
      response = await deps.http({
        url: `${current.origin}${CONSUME_PATH}`,
        headers: { origin: current.origin, 'content-type': 'application/json' },
        body: JSON.stringify({ pairing_id: ticket.pairingId, code: ticket.code, device_label: DESKTOP_DEVICE_LABEL }),
        timeoutMs: HTTP_TIMEOUT_MS,
      })
    } catch (error) {
      return fail('hub_unreachable', `The Hub did not answer: ${oneLine(error instanceof Error ? error.message : String(error))}`)
    }
    const body = bodyOf(response.body)
    if (response.status >= 200 && response.status < 300) {
      for (const cookie of response.setCookies) jar.store(cookie, now())
      if (body.status !== 'active') {
        jar.clear()
        persistSession()
        return fail('not_signed_in', 'The Hub wants this desktop confirmed before it can sign in. Confirm it on the Hub, then try again.')
      }
      persistSession()
      return jar.has(HUB_SESSION_COOKIE, now(), SESSION_MARGIN_MS)
        ? { ok: true, origin: current.origin, wsUrl: hubWebSocketUrlFor(current.origin) }
        : fail('bad_output', 'The Hub signed this desktop in without a session. Try again.')
    }
    if (response.status >= 400 && response.status < 500 && response.status !== 429) {
      return fail('not_signed_in', errorMessageOf(body, 'The Hub did not accept this desktop.'), true)
    }
    return fail('hub_unreachable', errorMessageOf(body, `The Hub answered ${response.status}. Try again in a moment.`))
  }

  async function doSignin(fresh: boolean): Promise<CockpitResult> {
    const current = needTarget()
    if (isFailure(current)) return current
    if (fresh) {
      closeAllSockets()
      jar.clear()
      persistSession()
    }
    const ready: CockpitResult = { ok: true, origin: current.origin, wsUrl: hubWebSocketUrlFor(current.origin) }
    if (jar.has(HUB_SESSION_COOKIE, now(), SESSION_MARGIN_MS)) return ready
    if (jar.has(HUB_REFRESH_COOKIE, now())) {
      const refreshed = await refresh(current.origin)
      if (refreshed === 'ok') return ready
      if (refreshed !== 'denied') return refreshed
    }
    return pairDesktop(current)
  }

  function signin(raw: unknown): Promise<CockpitResult> {
    const params = paramsOf(raw, ['fresh'])
    if (isFailure(params)) return Promise.resolve(params)
    if (params.fresh !== undefined && typeof params.fresh !== 'boolean') return Promise.resolve(fail('invalid_input', 'fresh must be true or false.'))
    if (signingIn !== undefined) return signingIn
    const run: Promise<CockpitResult> = doSignin(params.fresh === true).finally(() => {
      if (signingIn === run) signingIn = undefined
    })
    signingIn = run
    return run
  }

  // ── the bridged socket ────────────────────────────────────────────

  function socketOpen(raw: unknown, senderId: number): CockpitResult {
    const params = paramsOf(raw, ['id'])
    if (isFailure(params)) return params
    const id = params.id
    if (!socketIdOk(id)) return fail('invalid_input', 'That socket id is not valid.')
    if (sockets.has(id)) return fail('invalid_input', 'That socket id is already in use.')
    if (sockets.size >= MAX_SOCKETS) return fail('busy', 'Too many Hub connections are open.')
    const current = needTarget()
    if (isFailure(current)) return current
    // Only the session cookie rides on the socket; the refresh cookie never leaves the HTTP calls.
    const cookie = jar.header([HUB_SESSION_COOKIE], now())
    if (cookie === '') return fail('not_signed_in', 'This desktop is not signed in to the Hub.')
    const send = (event: HubSocketEvent): void => deps.emit(senderId, COCKPIT_SOCKET_EVENT, event)
    let ended = false
    let socket: HubSocket
    try {
      socket = deps.connect(
        { url: hubWebSocketUrlFor(current.origin), origin: current.origin, cookie },
        {
          open: () => send({ id, type: 'open' }),
          message: data => send({ id, type: 'message', data }),
          binary: () => send({ id, type: 'binary' }),
          error: message => {
            if (message.includes(SOCKET_REJECTED)) {
              // The Hub no longer honors this session: forget it so the next sign-in pairs anew.
              jar.clear()
              persistSession()
            }
            send({ id, type: 'error', message: oneLine(message) })
          },
          close: (code, reason) => {
            if (ended) return
            ended = true
            sockets.delete(id)
            send({ id, type: 'close', code, reason: oneLine(reason) })
          },
        },
      )
    } catch (error) {
      return fail('hub_unreachable', oneLine(error instanceof Error ? error.message : 'The Hub link could not be opened.'))
    }
    if (!ended) sockets.set(id, { senderId, socket })
    return { ok: true }
  }

  function ownedSocket(id: unknown, senderId: number): HubSocket | CockpitFailure {
    if (!socketIdOk(id)) return fail('invalid_input', 'That socket id is not valid.')
    const entry = sockets.get(id)
    return entry !== undefined && entry.senderId === senderId ? entry.socket : fail('socket_closed', 'That Hub link is closed.')
  }

  function socketSend(raw: unknown, senderId: number): CockpitResult {
    const params = paramsOf(raw, ['id', 'data'])
    if (isFailure(params)) return params
    if (typeof params.data !== 'string' || params.data === '') return fail('invalid_input', 'A Hub frame must be text.')
    if (Buffer.byteLength(params.data, 'utf8') > COCKPIT_MAX_FRAME_BYTES) return fail('invalid_input', 'That Hub frame is too large.')
    const socket = ownedSocket(params.id, senderId)
    if (isFailure(socket)) return socket
    try {
      socket.send(params.data)
    } catch (error) {
      return fail('socket_closed', oneLine(error instanceof Error ? error.message : 'The Hub link is closed.'))
    }
    return { ok: true }
  }

  function socketClose(raw: unknown, senderId: number): CockpitResult {
    const params = paramsOf(raw, ['id', 'code', 'reason'])
    if (isFailure(params)) return params
    const { code, reason } = params
    if (code !== undefined && (typeof code !== 'number' || !Number.isInteger(code) || code < 1000 || code > 4999)) return fail('invalid_input', 'That close code is not valid.')
    if (reason !== undefined && (typeof reason !== 'string' || reason.length > 120)) return fail('invalid_input', 'That close reason is not valid.')
    const socket = ownedSocket(params.id, senderId)
    if (isFailure(socket)) return socket
    try {
      socket.close(typeof code === 'number' ? code : 1000, typeof reason === 'string' ? reason : '')
    } catch {
      // Already closing: the close event is on its way.
    }
    return { ok: true }
  }

  // ── entry ─────────────────────────────────────────────────────────

  return {
    async handle(action, params, senderId) {
      if (!isCockpitActionName(action)) return fail('invalid_input', 'Unknown Hub action.')
      if (!Number.isInteger(senderId)) return fail('invalid_input', 'Unknown caller.')
      switch (action) {
        case 'status': {
          const checked = paramsOf(params, [])
          return isFailure(checked) ? checked : { ok: true, status: statusOf() }
        }
        case 'save': return save(params)
        case 'forget': return forget(params)
        case 'link': return link(params, senderId)
        case 'pair': return pair(params)
        case 'confirm': return confirm(params)
        case 'machines': return machines(params)
        case 'signin': return signin(params)
        case 'socket.open': return socketOpen(params, senderId)
        case 'socket.send': return socketSend(params, senderId)
        case 'socket.close': return socketClose(params, senderId)
      }
    },
    releaseSender(senderId) {
      for (const [id, entry] of [...sockets]) {
        if (entry.senderId !== senderId) continue
        sockets.delete(id)
        try { entry.socket.close(1000, 'page closed') } catch { /* already closing */ }
      }
    },
    dispose() {
      closeAllSockets()
      sockets.clear()
    },
  }
}
