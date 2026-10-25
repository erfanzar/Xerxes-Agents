// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { createBridge } from '../src/desktop/bridgeApi.js'
import {
  FOCUS_RECHECK_MS, createMacPermissions, macPermissionsChannels, readPermissionsDismissed, registerMacPermissionsIpc,
  writePermissionsDismissed, type MacPermissionsPort, type MediaAccessAnswer,
} from '../src/desktop/main/macPermissions.js'
import {
  MAC_PERMISSION_SETTINGS_URLS, asMacPermissionsState, macPermissionNoticeNeeded,
  type MacPermissionsResult, type MacPermissionsState,
} from '../src/desktop/macPermissionsShared.js'

interface FakePort extends MacPermissionsPort {
  trusted: boolean
  screen: MediaAccessAnswer
  microphone: MediaAccessAnswer
  dismissed: boolean
  clock: number
  readonly opened: string[]
  readonly prompts: string[]
  relaunched: number
  screenRequestFails: boolean
}

function fakePort(patch: Partial<Pick<FakePort, 'trusted' | 'screen' | 'microphone' | 'dismissed' | 'platform'>> = {}): FakePort {
  const port: FakePort = {
    platform: patch.platform ?? 'darwin',
    appLabel: 'Xerxes Agents',
    trusted: patch.trusted ?? true,
    screen: patch.screen ?? 'granted',
    microphone: patch.microphone ?? 'granted',
    dismissed: patch.dismissed ?? false,
    clock: 1_000_000,
    opened: [],
    prompts: [],
    relaunched: 0,
    screenRequestFails: false,
    isTrustedAccessibilityClient(prompt) {
      if (prompt) port.prompts.push('accessibility')
      return port.trusted
    },
    getMediaAccessStatus: media => (media === 'screen' ? port.screen : port.microphone),
    askForMicrophone: async () => { port.prompts.push('microphone'); port.microphone = 'granted'; return true },
    requestScreenCapture: async () => {
      port.prompts.push('screen')
      if (port.screenRequestFails) throw new Error('capturer refused')
      port.screen = 'denied'
    },
    openExternal: async url => { port.opened.push(url) },
    relaunch: () => { port.relaunched += 1 },
    readDismissed: () => port.dismissed,
    writeDismissed: value => { port.dismissed = value },
    now: () => port.clock,
  }
  return port
}

const statusOf = (state: MacPermissionsState, id: string) => state.items.find(item => item.id === id)

test('each Electron answer maps onto one status, and a silent check never prompts', () => {
  const port = fakePort({ trusted: false, screen: 'not-determined', microphone: 'restricted' })
  const permissions = createMacPermissions(port)
  const state = permissions.check()
  expect(state.platform).toBe('darwin')
  expect(statusOf(state, 'accessibility')?.status).toBe('denied')
  expect(statusOf(state, 'screen')?.status).toBe('not_determined')
  expect(statusOf(state, 'microphone')?.status).toBe('denied')
  expect(port.prompts).toEqual([])
  expect(port.opened).toEqual([])

  port.trusted = true
  port.screen = 'granted'
  port.microphone = 'unknown'
  const later = permissions.check()
  expect(statusOf(later, 'accessibility')?.status).toBe('granted')
  expect(statusOf(later, 'screen')?.status).toBe('granted')
  expect(statusOf(later, 'microphone')?.status).toBe('unknown')
  port.screen = 'denied'
  expect(statusOf(permissions.check(), 'screen')?.status).toBe('denied')
})

test('automation, app management and full disk are unknown with a reason, never probed', () => {
  const state = createMacPermissions(fakePort()).check()
  for (const id of ['automation', 'app_management', 'full_disk']) {
    const item = statusOf(state, id)!
    expect(item.status).toBe('unknown')
    expect(item.can_request).toBe(false)
    expect(item.detail).toBeString()
    expect(item.detail!.length).toBeGreaterThan(20)
  }
  expect(statusOf(state, 'automation')?.detail).toContain('without asking another app')
  expect(statusOf(state, 'full_disk')?.detail).toContain('protected file')
  for (const item of state.items) expect(item.settings_url).toBe(MAC_PERMISSION_SETTINGS_URLS[item.id])
  expect(MAC_PERMISSION_SETTINGS_URLS.app_management).toBe('x-apple.systempreferences:com.apple.preference.security?Privacy_AppBundles')
})

test('a throwing Electron read reports unknown instead of failing the whole check', () => {
  const port = fakePort()
  port.getMediaAccessStatus = () => { throw new Error('boom') }
  const state = createMacPermissions(port).check()
  expect(statusOf(state, 'screen')?.status).toBe('unknown')
  expect(statusOf(state, 'accessibility')?.status).toBe('granted')
})

test('off macOS every item is not_applicable, nothing is requestable, and actions are refused', async () => {
  const port = fakePort({ platform: 'linux', trusted: false, screen: 'denied' })
  const permissions = createMacPermissions(port)
  const state = permissions.check()
  expect(state.platform).toBe('other')
  expect(state.notice).toBe(false)
  expect(state.items.every(item => item.status === 'not_applicable' && !item.can_request)).toBe(true)
  const open = await permissions.handle('open', { id: 'screen' })
  expect(open).toMatchObject({ ok: false, code: 'unsupported' })
  expect(port.opened).toEqual([])
})

test('the notice shows when Accessibility or Screen Recording is missing, hides when granted or dismissed', async () => {
  const port = fakePort({ trusted: false })
  const permissions = createMacPermissions(port)
  expect(permissions.check().notice).toBe(true)
  port.trusted = true
  port.screen = 'not-determined'
  expect(permissions.check().notice).toBe(true)
  port.screen = 'granted'
  // Microphone alone never raises the computer-use notice.
  port.microphone = 'denied'
  expect(permissions.check().notice).toBe(false)

  port.trusted = false
  const dismissed = await permissions.handle('dismiss', {})
  expect(dismissed).toMatchObject({ ok: true, state: { dismissed: true, notice: false } })
  expect(port.dismissed).toBe(true)
  const remind = await permissions.handle('remind', {})
  expect(remind).toMatchObject({ ok: true, state: { dismissed: false, notice: true } })

  // An unknown screen status is not "missing": no nagging on what cannot be read.
  expect(macPermissionNoticeNeeded([{ id: 'screen', label: 'Screen Recording', status: 'unknown', required_for: 'x', settings_url: MAC_PERMISSION_SETTINGS_URLS.screen, can_request: false, detail: null }], false)).toBe(false)
})

test('focus re-checks are throttled and listeners hear only real changes', () => {
  const port = fakePort({ trusted: false })
  const permissions = createMacPermissions(port)
  const heard: MacPermissionsState[] = []
  permissions.onChange(state => heard.push(state))
  permissions.check()
  expect(heard).toHaveLength(1)

  port.trusted = true
  port.clock += 1_000
  expect(permissions.recheckOnFocus()).toBeNull()
  expect(heard).toHaveLength(1)

  port.clock += FOCUS_RECHECK_MS
  const fresh = permissions.recheckOnFocus()
  expect(fresh?.notice).toBe(false)
  expect(heard).toHaveLength(2)

  // Same statuses again: no push.
  port.clock += FOCUS_RECHECK_MS
  expect(permissions.recheckOnFocus()).not.toBeNull()
  expect(heard).toHaveLength(2)
})

test('open reaches only the allowlisted pane; unknown ids, extra keys and unknown actions are refused', async () => {
  const port = fakePort()
  const permissions = createMacPermissions(port)
  expect(await permissions.handle('open', { id: 'screen' })).toMatchObject({ ok: true, opened_settings: true })
  expect(port.opened).toEqual([MAC_PERMISSION_SETTINGS_URLS.screen])

  expect(await permissions.handle('open', { id: 'camera' })).toMatchObject({ ok: false, code: 'unknown_permission' })
  expect(await permissions.handle('open', { id: 'screen', url: 'https://evil.example' })).toMatchObject({ ok: false, code: 'invalid_request' })
  expect(await permissions.handle('open', { url: 'x-apple.systempreferences:anything' })).toMatchObject({ ok: false, code: 'invalid_request' })
  expect(await permissions.handle('open', 'screen')).toMatchObject({ ok: false, code: 'invalid_request' })
  expect(await permissions.handle('erase', {})).toMatchObject({ ok: false, code: 'unknown_action' })
  expect(port.opened).toEqual([MAC_PERMISSION_SETTINGS_URLS.screen])
})

test('request shows macOS prompts only where supported and falls back to the pane once answered', async () => {
  const port = fakePort({ trusted: false, screen: 'not-determined', microphone: 'not-determined' })
  const permissions = createMacPermissions(port)
  const first = permissions.check()
  expect(statusOf(first, 'accessibility')?.can_request).toBe(true)
  expect(statusOf(first, 'screen')?.can_request).toBe(true)
  expect(statusOf(first, 'microphone')?.can_request).toBe(true)

  await permissions.handle('request', { id: 'accessibility' })
  await permissions.handle('request', { id: 'microphone' })
  await permissions.handle('request', { id: 'screen' })
  expect(port.prompts).toEqual(['accessibility', 'microphone', 'screen'])
  expect(port.opened).toEqual([])

  // Screen now answered (denied): no prompt, the pane opens instead.
  const again = await permissions.handle('request', { id: 'screen' })
  expect(again).toMatchObject({ ok: true, opened_settings: true })
  expect(port.prompts).toEqual(['accessibility', 'microphone', 'screen'])
  expect(port.opened).toEqual([MAC_PERMISSION_SETTINGS_URLS.screen])

  // Nothing to request for the unreadable ones.
  expect(await permissions.handle('request', { id: 'automation' })).toMatchObject({ ok: false, code: 'unsupported' })
  expect(await permissions.handle('request', { id: 'full_disk' })).toMatchObject({ ok: false, code: 'unsupported' })
})

test('a failed screen request opens the pane instead', async () => {
  const port = fakePort({ screen: 'not-determined' })
  port.screenRequestFails = true
  const result = await createMacPermissions(port).handle('request', { id: 'screen' })
  expect(result).toMatchObject({ ok: true, opened_settings: true })
  expect(port.opened).toEqual([MAC_PERMISSION_SETTINGS_URLS.screen])
})

test('relaunch happens only when the action is called', async () => {
  const port = fakePort({ screen: 'denied' })
  const permissions = createMacPermissions(port)
  permissions.check()
  permissions.recheckOnFocus()
  expect(port.relaunched).toBe(0)
  expect(await permissions.handle('relaunch', {})).toMatchObject({ ok: true })
  expect(port.relaunched).toBe(1)
})

test('IPC: one channel per action, failures come back as { ok: false }', async () => {
  const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>()
  const port = fakePort()
  port.openExternal = async () => { throw new Error('LaunchServices said no') }
  registerMacPermissionsIpc({ handle: (channel, listener) => { handlers.set(channel, listener) } }, createMacPermissions(port))
  expect([...handlers.keys()].sort()).toEqual([...macPermissionsChannels()].sort())
  expect([...handlers.keys()]).toContain('permissions:status')
  expect([...handlers.keys()]).toContain('permissions:open')
  expect([...handlers.keys()]).toContain('permissions:request')

  const status = await handlers.get('permissions:status')!({}, undefined) as MacPermissionsResult
  expect(status.ok).toBe(true)
  const failed = await handlers.get('permissions:open')!({}, { id: 'screen' }) as MacPermissionsResult
  expect(failed).toMatchObject({ ok: false, code: 'failed' })
  expect(!failed.ok && failed.message).toContain('LaunchServices said no')
  const unknown = await handlers.get('permissions:open')!({}, { id: 'nope' }) as MacPermissionsResult
  expect(unknown).toMatchObject({ ok: false, code: 'unknown_permission' })
})

test('bridge: unknown actions never reach the host, pushed states are validated', async () => {
  const invoked: Array<[string, unknown[]]> = []
  const listeners = new Map<string, (value: unknown) => void>()
  const bridge = createBridge({
    invoke: async (channel, ...args) => { invoked.push([channel, args]); return { ok: true } },
    on: (channel, listener) => { listeners.set(channel, listener); return () => listeners.delete(channel) },
    setZoomFactor: () => {},
  })
  await expect(bridge.macPermissions('shell', {})).rejects.toThrow('Unknown permissions action')
  await expect(bridge.macPermissions('open', 'screen')).rejects.toThrow()
  await bridge.macPermissions('open', { id: 'screen' })
  expect(invoked).toEqual([['permissions:open', [{ id: 'screen' }]]])

  const seen: MacPermissionsState[] = []
  bridge.onMacPermissions((state: MacPermissionsState) => seen.push(state))
  const push = listeners.get('permissions:state')!
  const good = createMacPermissions(fakePort({ trusted: false })).check()
  push(good)
  // A state carrying a URL other than the allowlisted one is dropped.
  push({ ...good, items: good.items.map(item => item.id === 'screen' ? { ...item, settings_url: 'https://evil.example' } : item) })
  push('garbage')
  expect(seen).toHaveLength(1)
  expect(asMacPermissionsState(JSON.parse(JSON.stringify(good)))).toEqual(good)
})

test('"Don\'t remind me" persists in its own settings file and survives a bad file', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-macperm-'))
  try {
    const file = join(directory, 'desktop-permissions.json')
    expect(readPermissionsDismissed(file)).toBe(false)
    writePermissionsDismissed(file, true)
    expect(readPermissionsDismissed(file)).toBe(true)
    writePermissionsDismissed(file, false)
    expect(readPermissionsDismissed(file)).toBe(false)
    await Bun.write(file, '{not json')
    expect(readPermissionsDismissed(file)).toBe(false)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
