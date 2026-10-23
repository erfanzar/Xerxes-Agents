// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { createBridge } from '../src/desktop/bridgeApi.js'
import { checkComparisonCode, checkDeviceLabel, checkHubTarget, qrDataUrl } from '../src/desktop/hubShared.js'
import {
  describeCliFailure, jsonOf, parseCliJson, parseConfirm, parseHubInfo, parseLink, parseMachines, parsePairing,
  type CliOutcome, type CliRunner,
} from '../src/desktop/main/hubCli.js'
import { createHubCockpit, type HubCockpit, type HubCockpitStorage, type StoredHub } from '../src/desktop/main/hubCockpit.js'
import { registerHubCockpitIpc, cockpitChannels } from '../src/desktop/main/hubCockpitIpc.js'
import { CookieJar } from '../src/desktop/main/hubCookies.js'
import type { HubHttp, HubHttpRequest } from '../src/desktop/main/hubHttp.js'
import type { HubSocketHandlers, HubSocketTarget } from '../src/desktop/main/hubSocket.js'
import { createCockpitSockets, type CockpitApi } from '../src/desktop/renderer/hubClient.js'
import type { SocketHandlers } from '../src/client/hubConnection.js'

const ORIGIN = 'https://hub.example.ts.net'
const TARGET = 'xerxes@209.38.102.184'
const SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="1" height="1"/></svg>'

function done(stdout: unknown, extra: Partial<CliOutcome> = {}): CliOutcome {
  return { exitCode: 0, stdout: typeof stdout === 'string' ? stdout : `${JSON.stringify(stdout)}\n`, stderr: '', timedOut: false, ...extra }
}

// ── target and input validation ─────────────────────────────────────────

test('Hub SSH addresses: user@host and aliases pass; dashes, spaces and shell syntax are refused', () => {
  for (const good of ['xerxes@209.38.102.184', 'hub-alias', 'me@hub.example.com', ' padded@host ']) {
    expect(checkHubTarget(good).ok).toBe(true)
  }
  for (const bad of ['', '   ', '-oProxyCommand=evil', '-v', 'a b', 'user@host name', 'x;rm -rf /', 'a|b', '$(id)', '`id`', 'a&b', 'user@', '@host', 'a@b@c', 'host/../x', 'a\nb', 'u ser@host', 42, null, undefined, {}]) {
    expect(checkHubTarget(bad).ok).toBe(false)
  }
  expect(checkHubTarget(`${'a'.repeat(400)}`).ok).toBe(false)
})

test('device labels, pairing ids and comparison codes are validated', () => {
  expect(checkDeviceLabel('Phone').ok).toBe(true)
  expect(checkDeviceLabel("Erfan's phone").ok).toBe(false)
  expect(checkDeviceLabel('-x').ok).toBe(false)
  expect(checkDeviceLabel('').ok).toBe(false)
  expect(checkComparisonCode('123-456').ok).toBe(true)
  expect(checkComparisonCode('12 34').ok).toBe(false)
  expect(checkComparisonCode('--json').ok).toBe(false)
})

test('a QR is shown only as an inert data URL; script, foreign content and non-svg are refused', () => {
  expect(qrDataUrl(SVG)?.startsWith('data:image/svg+xml;base64,')).toBe(true)
  expect(qrDataUrl('<svg><script>alert(1)</script></svg>')).toBeUndefined()
  expect(qrDataUrl('<svg onload="x()"></svg>')).toBeUndefined()
  expect(qrDataUrl('<svg><foreignObject/></svg>')).toBeUndefined()
  expect(qrDataUrl('<html></html>')).toBeUndefined()
  expect(qrDataUrl(null)).toBeUndefined()
})

// ── CLI JSON parsing ────────────────────────────────────────────────────

test('CLI output: a single object, pretty printed, or after banner lines parses; the last object line wins', () => {
  expect(parseCliJson('{"a":1}\n')).toEqual({ a: 1 })
  expect(parseCliJson('{\n  "a": 1,\n  "b": [1, 2]\n}\n')).toEqual({ a: 1, b: [1, 2] })
  expect(parseCliJson('warning: something\n{"a":1}\n')).toEqual({ a: 1 })
  expect(parseCliJson('{"a":1}\n{"a":2}\n')).toEqual({ a: 2 })
})

test('CLI output: empty, malformed, array and scalar output yield nothing', () => {
  expect(parseCliJson('')).toBeUndefined()
  expect(parseCliJson('   \n')).toBeUndefined()
  expect(parseCliJson('{"a":')).toBeUndefined()
  expect(parseCliJson('[1,2]')).toBeUndefined()
  expect(parseCliJson('42')).toBeUndefined()
  expect(parseCliJson('not json at all')).toBeUndefined()
})

test('jsonOf turns exit codes, timeouts, spawn errors and bad output into failures with a usable message', () => {
  const ok = jsonOf(done({ x: 1 }), 'do it')
  expect(ok.ok).toBe(true)
  const bad = jsonOf(done('garbage'), 'do it') as { ok: false; code: string }
  expect(bad.code).toBe('bad_output')
  const failed = jsonOf({ exitCode: 1, stdout: '{"error":{"message":"no such host"}}', stderr: '', timedOut: false }, 'do it') as { ok: false; code: string; message: string }
  expect(failed.code).toBe('cli_failed')
  expect(failed.message).toBe('no such host')
  const stderrOnly = describeCliFailure({ exitCode: 255, stdout: '', stderr: 'first\nssh: connect refused\n', timedOut: false }, 'do it')
  expect(stderrOnly.message).toBe('ssh: connect refused')
  expect(describeCliFailure({ exitCode: null, stdout: '', stderr: '', timedOut: true }, 'do it').code).toBe('timeout')
  expect(describeCliFailure({ exitCode: null, stdout: '', stderr: '', timedOut: false, spawnError: 'ENOENT' }, 'do it').code).toBe('cli_unavailable')
})

test('hub info, pairing, confirm, machines and link shapes are validated tolerantly', () => {
  expect(parseHubInfo({ hub_id: 'h', fingerprint: 'fp', public_origin: ORIGIN, version: '1' })).toEqual({ hubId: 'h', fingerprint: 'fp', publicOrigin: ORIGIN, version: '1' })
  expect((parseHubInfo({ hub_id: 'h', public_origin: null }) as { publicOrigin: string | null }).publicOrigin).toBeNull()
  expect((parseHubInfo({ unrelated: true }) as { ok: false }).ok).toBe(false)
  expect((parseHubInfo({ hub_id: 'h', public_origin: 7 }) as { ok: false }).ok).toBe(false)

  const pairing = parsePairing({ pairing_id: 'p1', url: `${ORIGIN}/pair#x`, comparison_code: '123-456', expires_at: 'soon', confirm_required: true, qr_svg: SVG })
  expect(pairing).toMatchObject({ pairingId: 'p1', comparisonCode: '123-456', confirmRequired: true })
  expect((pairing as { qrDataUrl: string | null }).qrDataUrl).toContain('data:image/svg+xml;base64,')
  expect((parsePairing({ pairing_id: 'p1', url: 'javascript:alert(1)', comparison_code: '1', expires_at: 'x' }) as { ok: false }).ok).toBe(false)
  expect((parsePairing({ pairing_id: 'p1', url: `${ORIGIN}/pair`, comparison_code: '1', expires_at: 'x', qr_svg: '<svg onload="x()"/>' }) as { qrDataUrl: string | null }).qrDataUrl).toBeNull()

  expect(parseConfirm({ device_id: 'd1' })).toEqual({ deviceId: 'd1' })
  expect((parseConfirm({}) as { ok: false }).ok).toBe(false)

  const machines = parseMachines({ machines: [
    { id: 'm1', label: 'Mac', endpoint_kind: 'local', discover: true, projects: [{ workspace_id: 'w1', project_dir: '/a', label: 'A', health: 'online' }, { nope: 1 }] },
    'junk',
    { id: 'm2' },
  ] })
  expect(Array.isArray(machines)).toBe(true)
  expect(machines).toEqual([
    { id: 'm1', label: 'Mac', endpointKind: 'local', discover: true, projects: [{ workspaceId: 'w1', projectDir: '/a', label: 'A', health: 'online' }] },
    { id: 'm2', label: 'm2', endpointKind: 'unknown', discover: false, projects: [] },
  ])
  expect((parseMachines({ machines: 'x' }) as { ok: false }).ok).toBe(false)

  expect(parseLink({ machine_id: 'm1', remote_socket: '/s', enrolled: true })).toEqual({ machineId: 'm1', remoteSocket: '/s', servicePath: null })
  expect((parseLink({ machine_id: 'm1', enrolled: false }) as { ok: false }).ok).toBe(false)
  expect((parseLink({ enrolled: true }) as { ok: false }).ok).toBe(false)
})

// ── cookie jar ──────────────────────────────────────────────────────────

test('the cookie jar keeps only the two Hub cookies, honors Max-Age and survives a round trip', () => {
  const jar = new CookieJar()
  jar.store('__Host-xerxes_session=abc; Max-Age=900; Path=/; Secure; HttpOnly', 1000)
  jar.store('other=evil; Max-Age=900', 1000)
  jar.store('__Host-xerxes_refresh=def; Max-Age=3600', 1000)
  expect(jar.header(['__Host-xerxes_session', 'other'], 2000)).toBe('__Host-xerxes_session=abc')
  expect(jar.has('__Host-xerxes_session', 1000 + 900_000 - 1)).toBe(true)
  expect(jar.has('__Host-xerxes_session', 1000 + 900_000 + 1)).toBe(false)
  const copy = new CookieJar()
  copy.restore(jar.serialize(2000), 2000)
  expect(copy.header(['__Host-xerxes_refresh'], 2000)).toBe('__Host-xerxes_refresh=def')
  jar.store('__Host-xerxes_session=; Max-Age=0', 3000)
  expect(jar.has('__Host-xerxes_session', 3000)).toBe(false)
  const broken = new CookieJar()
  broken.restore('not json', 0)
  expect(broken.header(['__Host-xerxes_session'], 0)).toBe('')
})

// ── the cockpit controller ──────────────────────────────────────────────

interface Rig {
  readonly cockpit: HubCockpit
  readonly runs: string[][]
  readonly requests: HubHttpRequest[]
  readonly emitted: { senderId: number; channel: string; payload: unknown }[]
  readonly connects: { target: HubSocketTarget; handlers: HubSocketHandlers; sent: string[]; closed: boolean }[]
  readonly stored: { settings?: StoredHub | undefined; session?: string | undefined }
}

function ticketUrl(id = 'pair-1', code = 'secretcode'): string {
  return `${ORIGIN}/pair#${Buffer.from(JSON.stringify({ hub_id: 'h', pairing_id: id, code })).toString('base64url')}`
}

function rig(script: (args: readonly string[]) => CliOutcome, http?: HubHttp, initial?: StoredHub): Rig {
  const runs: string[][] = []
  const requests: HubHttpRequest[] = []
  const emitted: Rig['emitted'] = []
  const connects: Rig['connects'] = []
  const stored: Rig['stored'] = { settings: initial }
  const storage: HubCockpitStorage = {
    loadSettings: () => stored.settings,
    saveSettings: value => { stored.settings = value },
    loadSession: () => stored.session,
    saveSession: text => { stored.session = text },
  }
  const runner: CliRunner = {
    async run(args, options) {
      runs.push([...args])
      const outcome = script(args)
      if (args[0] === 'link') options.onStderrLine?.('registering machine…')
      return outcome
    },
  }
  const defaultHttp: HubHttp = async request => {
    requests.push(request)
    if (request.url.endsWith('/pair/consume')) {
      return {
        status: 200,
        setCookies: ['__Host-xerxes_session=sess-token-1; Max-Age=900; Path=/; Secure; HttpOnly', '__Host-xerxes_refresh=refresh-token-1; Max-Age=86400; Path=/; Secure; HttpOnly'],
        body: JSON.stringify({ status: 'active', csrf_token: 'csrf' }),
      }
    }
    return { status: 401, setCookies: [], body: '{}' }
  }
  const cockpit = createHubCockpit({
    runner,
    http: http === undefined ? defaultHttp : async request => { requests.push(request); return http(request) },
    connect: (target, handlers) => {
      const entry = { target, handlers, sent: [] as string[], closed: false }
      connects.push(entry)
      return { send: text => { entry.sent.push(text) }, close: () => { entry.closed = true } }
    },
    storage,
    emit: (senderId, channel, payload) => { emitted.push({ senderId, channel, payload }) },
    now: () => 1_000_000,
  })
  return { cockpit, runs, requests, emitted, connects, stored }
}

const saved: StoredHub = { target: TARGET, origin: ORIGIN, fingerprint: 'fp', hubVersion: '1', linkedMachineId: null }

const standard = (args: readonly string[]): CliOutcome => {
  const key = args.join(' ')
  if (key.endsWith('info --json')) return done({ hub_id: 'h', fingerprint: 'fp', public_origin: ORIGIN, version: '1.0' })
  if (key.includes('pair confirm')) return done({ device_id: 'dev-1' })
  if (key.includes('pair --no-confirm')) return done({ pairing_id: 'pair-1', url: ticketUrl(), comparison_code: '111-222', expires_at: 'later', confirm_required: false, qr_svg: null })
  if (key.includes(' pair ')) return done({ pairing_id: 'pair-2', url: `${ORIGIN}/pair#frag`, comparison_code: '123-456', expires_at: 'later', confirm_required: true, qr_svg: SVG })
  if (key.includes('machines list')) return done({ machines: [{ id: 'm1', label: 'Mac', endpoint_kind: 'local', discover: true, projects: [{ workspace_id: 'w1', project_dir: '/p', label: 'P', health: 'online' }] }] })
  if (args[0] === 'link') return done({ machine_id: 'm1', remote_socket: '/tmp/x.sock', enrolled: true })
  return done('', { exitCode: 2, stderr: 'unexpected' })
}

async function call(cockpit: HubCockpit, action: unknown, params: unknown = {}, sender = 1): Promise<Record<string, unknown>> {
  return (await cockpit.handle(action, params, sender)) as unknown as Record<string, unknown>
}

test('IPC input: unknown actions, non-object params, extra fields and bad targets never reach the CLI', async () => {
  const { cockpit, runs } = rig(standard)
  expect((await call(cockpit, 'rm -rf')).code).toBe('invalid_input')
  expect((await call(cockpit, 42)).code).toBe('invalid_input')
  expect((await call(cockpit, 'save', 'xerxes@host')).code).toBe('invalid_input')
  expect((await call(cockpit, 'save', [TARGET])).code).toBe('invalid_input')
  expect((await call(cockpit, 'save', { target: TARGET, extra: 1 })).code).toBe('invalid_input')
  for (const target of ['-oProxyCommand=evil', 'a b', 'x;y', '$(id)', '', undefined, 7, 'user@']) {
    const result = await call(cockpit, 'save', { target })
    expect(result.ok).toBe(false)
    expect(result.code).toBe('invalid_input')
  }
  expect((await call(cockpit, 'link', { target: '-bad' })).code).toBe('invalid_input')
  expect((await call(cockpit, 'status', { x: 1 })).code).toBe('invalid_input')
  expect(runs).toEqual([])
})

test('save reads the Hub web origin over `hub remote <target> info --json` and stores no credential', async () => {
  const r = rig(standard)
  const result = await call(r.cockpit, 'save', { target: ` ${TARGET} ` })
  expect(result.ok).toBe(true)
  expect(r.runs).toEqual([['remote', TARGET, 'info', '--json']])
  expect(result.status).toMatchObject({ target: TARGET, origin: ORIGIN, wsUrl: 'wss://hub.example.ts.net/api/hub/v1/ws', hasSession: false })
  expect(r.stored.settings).toMatchObject({ target: TARGET, origin: ORIGIN })
})

test('save refuses a Hub with no web address, a non-https one, and a failing CLI', async () => {
  const none = rig(() => done({ hub_id: 'h', fingerprint: 'fp', public_origin: null }))
  expect((await call(none.cockpit, 'save', { target: TARGET })).code).toBe('no_origin')
  expect(none.stored.settings).toBeUndefined()
  const plain = rig(() => done({ hub_id: 'h', public_origin: 'http://hub.example.com' }))
  expect((await call(plain.cockpit, 'save', { target: TARGET })).code).toBe('no_origin')
  const down = rig(() => done('', { exitCode: 255, stderr: 'ssh: Could not resolve hostname' }))
  const failed = await call(down.cockpit, 'save', { target: TARGET })
  expect(failed.code).toBe('cli_failed')
  expect(failed.message).toBe('ssh: Could not resolve hostname')
})

test('link runs `hub link <target> --json`, streams progress, then lists machines through `hub remote`', async () => {
  const r = rig(standard, undefined, saved)
  expect((await call(r.cockpit, 'link', { target: 'someone@else' })).code).toBe('no_target')
  const result = await call(r.cockpit, 'link', { target: TARGET }, 7)
  expect(result.ok).toBe(true)
  expect(r.runs).toEqual([['link', TARGET, '--json'], ['remote', TARGET, 'machines', 'list', '--json']])
  expect(result.link).toEqual({ machineId: 'm1', remoteSocket: '/tmp/x.sock', servicePath: null })
  expect((result.machines as { projects: unknown[] }[])[0]!.projects).toHaveLength(1)
  expect(result.status).toMatchObject({ linkedMachineId: 'm1' })
  expect(r.emitted.some(e => e.senderId === 7 && e.channel === 'hub.cockpit.progress' && JSON.stringify(e.payload).includes('registering machine'))).toBe(true)
  expect(r.stored.settings?.linkedMachineId).toBe('m1')
})

test('pair and confirm go through `hub remote`, with validated label, id and code', async () => {
  const r = rig(standard, undefined, saved)
  const paired = await call(r.cockpit, 'pair', { label: 'Phone' })
  expect(paired.ok).toBe(true)
  expect(r.runs[0]).toEqual(['remote', TARGET, 'pair', '--label', 'Phone', '--json'])
  const pairing = paired.pairing as { pairingId: string; comparisonCode: string; qrDataUrl: string | null }
  expect(pairing.qrDataUrl).toContain('data:image/svg+xml;base64,')
  expect((await call(r.cockpit, 'pair', { label: '--no-confirm' })).code).toBe('invalid_input')
  expect((await call(r.cockpit, 'confirm', { pairingId: '-x', code: '123-456' })).code).toBe('invalid_input')
  expect((await call(r.cockpit, 'confirm', { pairingId: 'pair-2', code: '1 2' })).code).toBe('invalid_input')
  const confirmed = await call(r.cockpit, 'confirm', { pairingId: pairing.pairingId, code: pairing.comparisonCode })
  expect(confirmed).toEqual({ ok: true, deviceId: 'dev-1' })
  expect(r.runs.at(-1)).toEqual(['remote', TARGET, 'pair', 'confirm', 'pair-2', '123-456', '--json'])
  const bare = rig(standard)
  expect((await call(bare.cockpit, 'pair', {})).code).toBe('no_target')
  expect((await call(bare.cockpit, 'machines', {})).code).toBe('no_target')
})

test('sign-in pairs the desktop itself, consumes the pairing against the Hub origin and keeps cookies in main', async () => {
  const r = rig(standard, undefined, saved)
  const result = await call(r.cockpit, 'signin')
  expect(result).toEqual({ ok: true, origin: ORIGIN, wsUrl: 'wss://hub.example.ts.net/api/hub/v1/ws' })
  expect(r.runs).toEqual([['remote', TARGET, 'pair', '--no-confirm', '--label', 'Desktop', '--json']])
  expect(r.requests).toHaveLength(1)
  const request = r.requests[0]!
  expect(request.url).toBe(`${ORIGIN}/api/hub/v1/pair/consume`)
  expect(request.headers.origin).toBe(ORIGIN)
  expect(JSON.parse(request.body)).toEqual({ pairing_id: 'pair-1', code: 'secretcode', device_label: 'Desktop' })
  // Neither the reply nor the status carries a secret.
  const text = JSON.stringify([result, await call(r.cockpit, 'status')])
  for (const secret of ['sess-token-1', 'refresh-token-1', 'secretcode', 'csrf']) expect(text).not.toContain(secret)
  expect((await call(r.cockpit, 'status')).status).toMatchObject({ hasSession: true })
  // A second sign-in reuses the session: no new pairing.
  await call(r.cockpit, 'signin')
  expect(r.runs).toHaveLength(1)
  // The session is kept for the next launch (the storage layer encrypts it).
  expect(r.stored.session).toContain('sess-token-1')
})

test('sign-in reports a refusal as denied, an unreachable Hub as retryable, and a pending device as not signed in', async () => {
  const refused = rig(standard, async () => ({ status: 400, setCookies: [], body: '{"error":{"code":"invalid_request","message":"pairing expired"}}' }), saved)
  const denied = await call(refused.cockpit, 'signin')
  expect(denied).toMatchObject({ ok: false, code: 'not_signed_in', message: 'pairing expired', denied: true })
  const throwing = rig(standard, async () => { throw new Error('connect ECONNREFUSED') }, saved)
  const unreachable = await call(throwing.cockpit, 'signin')
  expect(unreachable.code).toBe('hub_unreachable')
  expect(unreachable.denied).toBeUndefined()
  const pending = rig(standard, async () => ({ status: 200, setCookies: ['__Host-xerxes_session=zzz; Max-Age=900'], body: '{"status":"pending_confirmation"}' }), saved)
  const waiting = await call(pending.cockpit, 'signin')
  expect(waiting.code).toBe('not_signed_in')
  expect((await call(pending.cockpit, 'status')).status).toMatchObject({ hasSession: false })
  const noHub = rig(standard)
  expect((await call(noHub.cockpit, 'signin')).code).toBe('no_target')
})

test('the socket opens with the Hub origin and only the session cookie, bridges frames, and is bound to its page', async () => {
  const r = rig(standard, undefined, saved)
  expect((await call(r.cockpit, 'socket.open', { id: 'abc' }, 3)).code).toBe('not_signed_in')
  await call(r.cockpit, 'signin', {}, 3)
  expect((await call(r.cockpit, 'socket.open', { id: 'bad id!' }, 3)).code).toBe('invalid_input')
  expect(await call(r.cockpit, 'socket.open', { id: 'abc' }, 3)).toEqual({ ok: true })
  expect((await call(r.cockpit, 'socket.open', { id: 'abc' }, 3)).code).toBe('invalid_input')
  const connection = r.connects[0]!
  expect(connection.target.url).toBe('wss://hub.example.ts.net/api/hub/v1/ws')
  expect(connection.target.origin).toBe(ORIGIN)
  expect(connection.target.cookie).toBe('__Host-xerxes_session=sess-token-1')
  expect(connection.target.cookie).not.toContain('refresh')

  connection.handlers.open()
  connection.handlers.message('{"hello":1}')
  connection.handlers.binary()
  const sockets = r.emitted.filter(e => e.channel === 'hub.cockpit.socket')
  expect(sockets.map(e => e.senderId)).toEqual([3, 3, 3])
  expect(sockets.map(e => (e.payload as { type: string }).type)).toEqual(['open', 'message', 'binary'])
  expect(sockets[1]!.payload).toEqual({ id: 'abc', type: 'message', data: '{"hello":1}' })

  expect(await call(r.cockpit, 'socket.send', { id: 'abc', data: '{"v":1}' }, 3)).toEqual({ ok: true })
  expect(connection.sent).toEqual(['{"v":1}'])
  // Another page cannot use this socket; bad frames are refused.
  expect((await call(r.cockpit, 'socket.send', { id: 'abc', data: 'x' }, 4)).code).toBe('socket_closed')
  expect((await call(r.cockpit, 'socket.send', { id: 'abc', data: 5 }, 3)).code).toBe('invalid_input')
  expect((await call(r.cockpit, 'socket.send', { id: 'abc', data: 'x'.repeat(1024 * 1024 + 1) }, 3)).code).toBe('invalid_input')
  expect((await call(r.cockpit, 'socket.close', { id: 'abc', code: 5 }, 3)).code).toBe('invalid_input')

  r.cockpit.releaseSender(3)
  expect(connection.closed).toBe(true)
  expect((await call(r.cockpit, 'socket.send', { id: 'abc', data: 'x' }, 3)).code).toBe('socket_closed')
})

test('a Hub that refuses the socket drops the session so the next sign-in pairs anew', async () => {
  const r = rig(standard, undefined, saved)
  await call(r.cockpit, 'signin')
  await call(r.cockpit, 'socket.open', { id: 's1' })
  r.connects[0]!.handlers.error('The Hub did not accept this sign-in.')
  expect((await call(r.cockpit, 'status')).status).toMatchObject({ hasSession: false })
  await call(r.cockpit, 'signin')
  expect(r.runs.filter(args => args.includes('--no-confirm'))).toHaveLength(2)
})

test('forgetting the Hub closes sockets and clears the session and address', async () => {
  const r = rig(standard, undefined, saved)
  await call(r.cockpit, 'signin')
  await call(r.cockpit, 'socket.open', { id: 's1' })
  const result = await call(r.cockpit, 'forget')
  expect(result.status).toMatchObject({ target: null, origin: null, hasSession: false })
  expect(r.connects[0]!.closed).toBe(true)
  expect(r.stored.settings).toBeUndefined()
})

test('IPC registration: one channel per action, a thrown error becomes a failure value, page teardown releases sockets', async () => {
  const handlers = new Map<string, (event: { sender: { id: number } }, ...args: unknown[]) => unknown>()
  const released: number[] = []
  const destroyed = new Map<number, () => void>()
  const cockpit: HubCockpit = {
    async handle(action) { if (action === 'status') throw new Error('boom\nwith detail'); return { ok: true } },
    releaseSender: id => { released.push(id) },
    dispose: () => {},
  }
  registerHubCockpitIpc({ handle: (channel, listener) => { handlers.set(channel, listener) } }, cockpit, { onDestroyed: (id, callback) => { destroyed.set(id, callback) } })
  expect([...handlers.keys()].sort()).toEqual([...cockpitChannels()].sort())
  for (const name of ['status', 'link', 'pair', 'confirm', 'machines']) expect(handlers.has(`hub.cockpit.${name}`)).toBe(true)
  const failed = await handlers.get('hub.cockpit.status')!({ sender: { id: 9 } }, {}) as { ok: boolean; message: string }
  expect(failed.ok).toBe(false)
  expect(failed.message).toContain('boom with detail')
  expect(await handlers.get('hub.cockpit.machines')!({ sender: { id: 9 } }, {})).toEqual({ ok: true })
  destroyed.get(9)!()
  expect(released).toEqual([9])
})

// ── the preload bridge and the renderer socket adapter ──────────────────

test('the bridge refuses unknown Hub actions and routes each action to its own channel', async () => {
  const calls: unknown[][] = []
  const bridge = createBridge({ invoke: async (...args) => { calls.push(args); return { ok: true } }, on: () => () => {}, setZoomFactor: () => {} })
  await expect(bridge.hubCockpit('rm')).rejects.toThrow('Unknown Hub action')
  await expect(bridge.hubCockpit('link', 'x')).rejects.toThrow()
  await bridge.hubCockpit('link', { target: TARGET })
  expect(calls).toEqual([['hub.cockpit.link', { target: TARGET }]])
})

test('the renderer socket adapter ignores the URL, routes events by id and reports a refused open as a close', async () => {
  const sent: { action: string; params: Record<string, unknown> }[] = []
  let push: (event: unknown) => void = () => {}
  const api: CockpitApi = {
    call: async (action, params) => {
      sent.push({ action, params: params ?? {} })
      return action === 'socket.open' && sent.length > 3 ? { ok: false, code: 'not_signed_in', message: 'nope' } : { ok: true }
    },
    onSocket: handler => { push = handler as (event: unknown) => void; return () => {} },
    onProgress: () => () => {},
  }
  let counter = 0
  const sockets = createCockpitSockets(api, () => `s${++counter}`)
  const log: string[] = []
  const handlers: SocketHandlers = {
    open: () => log.push('open'), message: data => log.push(`message:${data}`), binary: () => log.push('binary'),
    close: (code, reason) => log.push(`close:${code}:${reason}`), error: () => log.push('error'),
  }
  const socket = sockets.factory('wss://elsewhere.example/api', handlers)
  await Promise.resolve()
  expect(sent[0]).toEqual({ action: 'socket.open', params: { id: 's1' } })
  push({ id: 's1', type: 'open' })
  push({ id: 's1', type: 'message', data: 'hi' })
  push({ id: 'other', type: 'message', data: 'ignored' })
  push({ id: 's1', type: 'garbage' })
  socket.send('frame')
  await Promise.resolve()
  expect(sent[1]).toEqual({ action: 'socket.send', params: { id: 's1', data: 'frame' } })
  expect(log).toEqual(['open', 'message:hi'])
  push({ id: 's1', type: 'close', code: 1013, reason: 'slow' })
  expect(log.at(-1)).toBe('close:1013:slow')
  // A refused open surfaces as a close, so the connection can retry.
  sent.push({ action: 'pad', params: {} })
  sockets.factory('x', handlers)
  await new Promise(resolve => setTimeout(resolve, 0))
  expect(log.at(-1)).toBe('close:1006:nope')
  sockets.dispose()
})
