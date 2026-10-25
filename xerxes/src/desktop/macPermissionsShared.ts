// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The macOS privacy permissions the desktop app checks, shared by the main
 * process, the preload bridge and the renderer. Main reads the statuses (no
 * prompts) and owns every System Settings link; the renderer only ever names a
 * permission id, never a URL, so it cannot open anything outside this list.
 */

export const MAC_PERMISSION_IDS = ['accessibility', 'screen', 'automation', 'app_management', 'microphone', 'full_disk'] as const
export type MacPermissionId = typeof MAC_PERMISSION_IDS[number]

export const MAC_PERMISSION_STATUSES = ['granted', 'denied', 'not_determined', 'unknown', 'not_applicable'] as const
export type MacPermissionStatus = typeof MAC_PERMISSION_STATUSES[number]

/** The only URLs `permissions:open` can reach: one Privacy & Security pane per permission. */
export const MAC_PERMISSION_SETTINGS_URLS: Readonly<Record<MacPermissionId, string>> = {
  accessibility: 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility',
  screen: 'x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture',
  automation: 'x-apple.systempreferences:com.apple.preference.security?Privacy_Automation',
  app_management: 'x-apple.systempreferences:com.apple.preference.security?Privacy_AppBundles',
  microphone: 'x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone',
  full_disk: 'x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles',
}

/** Permissions with a macOS prompt the app may ask for (only when the person clicks Request). */
export const MAC_REQUESTABLE_PERMISSIONS: readonly MacPermissionId[] = ['accessibility', 'screen', 'microphone']

/** Computer use cannot work without these two; the startup notice watches only them. */
export const MAC_NOTICE_PERMISSIONS: readonly MacPermissionId[] = ['accessibility', 'screen']

export interface MacPermissionItem {
  readonly id: MacPermissionId
  readonly label: string
  readonly status: MacPermissionStatus
  /** What in Xerxes needs it. */
  readonly required_for: string
  readonly settings_url: string
  /** Whether Request can show macOS's own prompt (or list the app) for it right now. */
  readonly can_request: boolean
  /** Why a status cannot be read, or another caveat; null when there is nothing to add. */
  readonly detail: string | null
}

export interface MacPermissionsState {
  readonly platform: 'darwin' | 'other'
  /** The name macOS lists in Privacy & Security for this app ("Electron" for a source launch). */
  readonly app_label: string
  readonly items: readonly MacPermissionItem[]
  /** "Don't remind me" was chosen; persisted in the desktop settings directory. */
  readonly dismissed: boolean
  /** Whether the startup notice should show (missing Accessibility or Screen Recording, not dismissed). */
  readonly notice: boolean
  readonly checked_at: string
}

export const MAC_PERMISSION_ACTIONS = ['status', 'open', 'request', 'dismiss', 'remind', 'relaunch'] as const
export type MacPermissionAction = typeof MAC_PERMISSION_ACTIONS[number]

/** One IPC channel per action: permissions:status, permissions:open, ... */
export const MAC_PERMISSIONS_CHANNEL_PREFIX = 'permissions:'
/** Main pushes the state here when it changes (focus re-check, dismiss, another window's request). */
export const MAC_PERMISSIONS_STATE_EVENT = 'permissions:state'

export type MacPermissionsFailureCode = 'unknown_action' | 'unknown_permission' | 'invalid_request' | 'unsupported' | 'failed'

export type MacPermissionsResult =
  | { readonly ok: true; readonly state: MacPermissionsState; readonly opened_settings?: boolean }
  | { readonly ok: false; readonly code: MacPermissionsFailureCode; readonly message: string }

export function isMacPermissionId(value: unknown): value is MacPermissionId {
  return typeof value === 'string' && (MAC_PERMISSION_IDS as readonly string[]).includes(value)
}

export function isMacPermissionAction(value: unknown): value is MacPermissionAction {
  return typeof value === 'string' && (MAC_PERMISSION_ACTIONS as readonly string[]).includes(value)
}

function isStatus(value: unknown): value is MacPermissionStatus {
  return typeof value === 'string' && (MAC_PERMISSION_STATUSES as readonly string[]).includes(value)
}

/** True when Accessibility or Screen Recording is readable and not granted, and the person has not opted out. */
export function macPermissionNoticeNeeded(items: readonly MacPermissionItem[], dismissed: boolean): boolean {
  if (dismissed) return false
  return missingNoticePermissions(items).length > 0
}

/** The computer-use permissions that are known to be off or not asked yet. */
export function missingNoticePermissions(items: readonly MacPermissionItem[]): MacPermissionItem[] {
  return items.filter(item => MAC_NOTICE_PERMISSIONS.includes(item.id) && (item.status === 'denied' || item.status === 'not_determined'))
}

function shortText(value: unknown, max: number): string | undefined {
  return typeof value === 'string' && value.length <= max ? value : undefined
}

/** Validate a state that crossed a process boundary; anything off-contract is dropped. */
export function asMacPermissionsState(value: unknown): MacPermissionsState | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  const raw = value as Record<string, unknown>
  if ((raw.platform !== 'darwin' && raw.platform !== 'other') || typeof raw.dismissed !== 'boolean' || typeof raw.notice !== 'boolean') return undefined
  const checkedAt = shortText(raw.checked_at, 64)
  const appLabel = shortText(raw.app_label, 80)
  if (checkedAt === undefined || appLabel === undefined || !Array.isArray(raw.items) || raw.items.length > MAC_PERMISSION_IDS.length) return undefined
  const items: MacPermissionItem[] = []
  for (const entry of raw.items as unknown[]) {
    if (!entry || typeof entry !== 'object') return undefined
    const item = entry as Record<string, unknown>
    if (!isMacPermissionId(item.id) || !isStatus(item.status) || typeof item.can_request !== 'boolean') return undefined
    // The renderer never trusts a URL it is handed: it must be the allowlisted one.
    if (item.settings_url !== MAC_PERMISSION_SETTINGS_URLS[item.id]) return undefined
    const label = shortText(item.label, 80)
    const requiredFor = shortText(item.required_for, 400)
    const detail = item.detail === null ? null : shortText(item.detail, 600)
    if (label === undefined || requiredFor === undefined || detail === undefined) return undefined
    if (items.some(existing => existing.id === item.id)) return undefined
    items.push({ id: item.id, label, status: item.status, required_for: requiredFor, settings_url: item.settings_url, can_request: item.can_request, detail })
  }
  return { platform: raw.platform, app_label: appLabel, items, dismissed: raw.dismissed, notice: raw.notice, checked_at: checkedAt }
}
