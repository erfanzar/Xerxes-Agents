// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * macOS privacy permissions for the desktop app: read what can be read
 * without prompting, open the matching System Settings pane, and show
 * macOS's own prompt only when the person clicks Request.
 *
 * TCC attributes a child process's access to the app responsible for it, so
 * the daemon this app starts (and the computer use it runs) uses the grants of
 * "Xerxes Agents". Electron is reached through `MacPermissionsPort`, so tests
 * use fakes and nothing here pops a real prompt on its own.
 *
 * Automation, App Management and Full Disk Access have no API that reads the
 * state without side effects (asking another app, or touching a protected
 * file), so they report `unknown` and say why instead of probing.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { randomUUID } from 'node:crypto'
import { dirname, join } from 'node:path'

import {
  MAC_PERMISSION_IDS, MAC_PERMISSION_SETTINGS_URLS, MAC_PERMISSIONS_CHANNEL_PREFIX, MAC_PERMISSION_ACTIONS,
  MAC_REQUESTABLE_PERMISSIONS, isMacPermissionId, macPermissionNoticeNeeded,
  type MacPermissionAction, type MacPermissionId, type MacPermissionItem, type MacPermissionStatus,
  type MacPermissionsResult, type MacPermissionsState,
} from '../macPermissionsShared.js'

/** Electron's `systemPreferences.getMediaAccessStatus` answers. */
export type MediaAccessAnswer = 'not-determined' | 'granted' | 'denied' | 'restricted' | 'unknown'

/** Everything privileged, injected. Production wraps Electron; tests pass fakes. */
export interface MacPermissionsPort {
  readonly platform: string
  /** The name Privacy & Security lists for this app. */
  readonly appLabel: string
  /** `systemPreferences.isTrustedAccessibilityClient(prompt)`; `true` shows macOS's prompt. */
  isTrustedAccessibilityClient(prompt: boolean): boolean
  getMediaAccessStatus(media: 'microphone' | 'screen'): MediaAccessAnswer
  /** `systemPreferences.askForMediaAccess('microphone')`: macOS's prompt, only while not determined. */
  askForMicrophone(): Promise<boolean>
  /** Ask for the screen once (a tiny `desktopCapturer.getSources`), which makes macOS prompt and list the app. */
  requestScreenCapture(): Promise<void>
  openExternal(url: string): Promise<void>
  /** Relaunch the app; only ever called from the person's click. */
  relaunch(): void
  readDismissed(): boolean
  writeDismissed(dismissed: boolean): void
  now(): number
}

/** `{ "dismissed": true }` once the person picks "Don't remind me". Its own file, like desktop-update.json. */
export function macPermissionsSettingsFile(settingsDirectory: string): string {
  return join(settingsDirectory, 'desktop-permissions.json')
}

export function readPermissionsDismissed(file: string): boolean {
  if (!existsSync(file)) return false
  try {
    const value: unknown = JSON.parse(readFileSync(file, 'utf8'))
    return Boolean(value && typeof value === 'object' && (value as { dismissed?: unknown }).dismissed === true)
  } catch {
    return false
  }
}

export function writePermissionsDismissed(file: string, dismissed: boolean): void {
  mkdirSync(dirname(file), { recursive: true, mode: 0o700 })
  const temporary = `${file}.${randomUUID()}.tmp`
  try {
    writeFileSync(temporary, `${JSON.stringify({ dismissed })}\n`, { mode: 0o600 })
    renameSync(temporary, file)
  } finally { rmSync(temporary, { force: true }) }
}

/** A re-check on window focus runs at most this often. */
export const FOCUS_RECHECK_MS = 5_000

const LABELS: Record<MacPermissionId, string> = {
  accessibility: 'Accessibility',
  screen: 'Screen Recording',
  automation: 'Automation',
  app_management: 'App Management',
  microphone: 'Microphone',
  full_disk: 'Full Disk Access',
}

const REQUIRED_FOR: Record<MacPermissionId, string> = {
  accessibility: 'Computer use: clicking, typing, scrolling and dragging (Xerxes posts input events).',
  screen: 'Computer use: taking screenshots so the agent can see the screen.',
  automation: 'Computer use and AppleScript: controlling other apps and System Events.',
  app_management: 'Installing or updating apps in Applications, including Xerxes updating itself.',
  microphone: 'Dictation in the composer.',
  full_disk: 'Optional: tasks that read protected folders (Mail, Messages, Safari data, other apps\' containers).',
}

function unknownDetail(id: MacPermissionId, app: string): string | null {
  if (id === 'automation') return `macOS cannot report this without asking another app. It asks once per app the first time ${app} controls it; review the switches under Automation.`
  if (id === 'app_management') return `macOS does not let apps read this. Turn on ${app} under App Management if an install or update is refused.`
  if (id === 'full_disk') return `Checking would mean reading a protected file, which Xerxes does not do. Turn on ${app} only if your tasks need those folders.`
  return null
}

function mediaStatus(answer: MediaAccessAnswer): MacPermissionStatus {
  switch (answer) {
    case 'granted': return 'granted'
    case 'denied':
    case 'restricted': return 'denied'
    case 'not-determined': return 'not_determined'
    default: return 'unknown'
  }
}

function readStatus(port: MacPermissionsPort, id: MacPermissionId): MacPermissionStatus {
  try {
    if (id === 'accessibility') return port.isTrustedAccessibilityClient(false) ? 'granted' : 'denied'
    if (id === 'screen') return mediaStatus(port.getMediaAccessStatus('screen'))
    if (id === 'microphone') return mediaStatus(port.getMediaAccessStatus('microphone'))
  } catch {
    return 'unknown'
  }
  return 'unknown'
}

function canRequest(id: MacPermissionId, status: MacPermissionStatus): boolean {
  if (!MAC_REQUESTABLE_PERMISSIONS.includes(id)) return false
  // macOS shows the Accessibility prompt whenever the app is not trusted; the
  // screen and microphone prompts only appear before the first answer.
  if (id === 'accessibility') return status !== 'granted' && status !== 'not_applicable'
  return status === 'not_determined'
}

function itemDetail(id: MacPermissionId, status: MacPermissionStatus, app: string): string | null {
  const unknown = unknownDetail(id, app)
  if (unknown) return unknown
  if (status === 'unknown') return 'macOS did not report a status. Open Settings to check it.'
  if (id === 'screen' && status !== 'granted') return `After turning it on, relaunch ${app}: macOS applies Screen Recording only to a new launch.`
  if (id === 'accessibility' && status !== 'granted') return `Turn on ${app} in the list. If it is already on but still reported off, remove it with − and add it again.`
  return null
}

export interface MacPermissions {
  /** Read every status now (never prompts). */
  check(): MacPermissionsState
  /** The last state read, or a fresh one before the first read. */
  current(): MacPermissionsState
  /** Re-check after a window regains focus; throttled. Returns the new state, or null when skipped. */
  recheckOnFocus(): MacPermissionsState | null
  handle(action: unknown, params: unknown): Promise<MacPermissionsResult>
  /** Called with every state whose statuses or notice differ from the previous one. */
  onChange(listener: (state: MacPermissionsState) => void): () => void
}

function sameState(a: MacPermissionsState | null, b: MacPermissionsState): boolean {
  if (!a || a.notice !== b.notice || a.dismissed !== b.dismissed || a.items.length !== b.items.length) return false
  return a.items.every((item, index) => item.status === b.items[index]?.status && item.can_request === b.items[index]?.can_request)
}

export function createMacPermissions(port: MacPermissionsPort): MacPermissions {
  const darwin = port.platform === 'darwin'
  const app = port.appLabel
  const listeners = new Set<(state: MacPermissionsState) => void>()
  let last: MacPermissionsState | null = null
  let lastCheckAt = Number.NEGATIVE_INFINITY

  const dismissed = (): boolean => {
    try { return port.readDismissed() } catch { return false }
  }

  const read = (): MacPermissionsState => {
    const items: MacPermissionItem[] = MAC_PERMISSION_IDS.map(id => {
      const status: MacPermissionStatus = darwin ? readStatus(port, id) : 'not_applicable'
      return {
        id, label: LABELS[id], status, required_for: REQUIRED_FOR[id],
        settings_url: MAC_PERMISSION_SETTINGS_URLS[id],
        can_request: darwin && canRequest(id, status),
        detail: darwin ? itemDetail(id, status, app) : null,
      }
    })
    const off = darwin && dismissed()
    return {
      platform: darwin ? 'darwin' : 'other', app_label: app, items, dismissed: off,
      notice: darwin && macPermissionNoticeNeeded(items, off), checked_at: new Date(port.now()).toISOString(),
    }
  }

  const check = (): MacPermissionsState => {
    const next = read()
    lastCheckAt = port.now()
    const changed = !sameState(last, next)
    last = next
    if (changed) for (const listener of listeners) {
      try { listener(next) } catch { /* one window's fault must not stop the others */ }
    }
    return next
  }

  const ok = (state: MacPermissionsState, openedSettings?: boolean): MacPermissionsResult =>
    openedSettings === undefined ? { ok: true, state } : { ok: true, state, opened_settings: openedSettings }
  const fail = (code: 'unknown_action' | 'unknown_permission' | 'invalid_request' | 'unsupported' | 'failed', message: string): MacPermissionsResult => ({ ok: false, code, message })

  const openSettings = async (id: MacPermissionId): Promise<void> => {
    // Only the allowlisted pane for a known id; the renderer never supplies a URL.
    await port.openExternal(MAC_PERMISSION_SETTINGS_URLS[id])
  }

  const request = async (id: MacPermissionId): Promise<MacPermissionsResult> => {
    if (!MAC_REQUESTABLE_PERMISSIONS.includes(id)) return fail('unsupported', `${LABELS[id]} has no prompt an app can show. Use Open Settings instead.`)
    const status = readStatus(port, id)
    if (status === 'granted') return ok(check(), false)
    if (id === 'accessibility') {
      // macOS's own dialog; it also adds the app to the Accessibility list.
      port.isTrustedAccessibilityClient(true)
      return ok(check(), false)
    }
    if (status !== 'not_determined') {
      // Already answered: macOS will not prompt again, so the pane is the only way.
      await openSettings(id)
      return ok(check(), true)
    }
    if (id === 'microphone') {
      await port.askForMicrophone()
      return ok(check(), false)
    }
    try {
      await port.requestScreenCapture()
    } catch {
      await openSettings(id)
      return ok(check(), true)
    }
    return ok(check(), false)
  }

  const idOf = (params: unknown): MacPermissionId | 'invalid' | 'unknown' => {
    if (!params || typeof params !== 'object' || Array.isArray(params)) return 'invalid'
    const keys = Object.keys(params)
    if (keys.length !== 1 || keys[0] !== 'id') return 'invalid'
    const id = (params as { id?: unknown }).id
    return isMacPermissionId(id) ? id : 'unknown'
  }

  const handle = async (action: unknown, params: unknown): Promise<MacPermissionsResult> => {
    if (typeof action !== 'string' || !(MAC_PERMISSION_ACTIONS as readonly string[]).includes(action)) return fail('unknown_action', 'Unknown permissions action.')
    const verb = action as MacPermissionAction
    if (verb === 'status') return ok(check())
    if (!darwin) return fail('unsupported', 'These permissions only exist on macOS.')
    try {
      if (verb === 'open' || verb === 'request') {
        const id = idOf(params)
        if (id === 'invalid') return fail('invalid_request', 'Name exactly one permission: { id }.')
        if (id === 'unknown') return fail('unknown_permission', 'Unknown permission.')
        if (verb === 'open') {
          await openSettings(id)
          return ok(check(), true)
        }
        return await request(id)
      }
      if (verb === 'dismiss' || verb === 'remind') {
        port.writeDismissed(verb === 'dismiss')
        return ok(check())
      }
      // relaunch: only the person's click reaches here.
      port.relaunch()
      return ok(current())
    } catch (error) {
      return fail('failed', `The permissions action failed: ${(error instanceof Error ? error.message : String(error)).replace(/\s+/g, ' ').slice(0, 200)}`)
    }
  }

  const current = (): MacPermissionsState => last ?? check()

  return {
    check,
    current,
    recheckOnFocus: () => (port.now() - lastCheckAt < FOCUS_RECHECK_MS ? null : check()),
    handle,
    onChange(listener) {
      listeners.add(listener)
      return () => { listeners.delete(listener) }
    },
  }
}

export interface PermissionsIpcMain {
  handle(channel: string, listener: (event: unknown, ...args: unknown[]) => unknown): void
}

export function macPermissionsChannels(): readonly string[] {
  return MAC_PERMISSION_ACTIONS.map(action => MAC_PERMISSIONS_CHANNEL_PREFIX + action)
}

/** One channel per action (`permissions:status`, `permissions:open`, ...); a thrown error never reaches the renderer raw. */
export function registerMacPermissionsIpc(ipc: PermissionsIpcMain, permissions: MacPermissions): void {
  for (const action of MAC_PERMISSION_ACTIONS) {
    ipc.handle(MAC_PERMISSIONS_CHANNEL_PREFIX + action, async (_event, params) => {
      try {
        return await permissions.handle(action, params ?? {})
      } catch (error) {
        const failure: MacPermissionsResult = { ok: false, code: 'failed', message: (error instanceof Error ? error.message : String(error)).slice(0, 200) }
        return failure
      }
    })
  }
}
