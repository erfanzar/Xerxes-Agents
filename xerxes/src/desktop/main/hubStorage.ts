// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Where the desktop keeps its Hub address and session between launches.
 *
 * The address file (`desktop-hub.json`) holds no credential. The session file
 * holds the Hub cookies and is written only when the platform can encrypt it
 * (Electron `safeStorage`); otherwise the session lives in memory for this run.
 */

import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

import { checkHubTarget, parseHubOrigin } from '../hubShared.js'
import type { HubCockpitStorage, StoredHub } from './hubCockpit.js'

/**
 * The platform's encryption. Every call is asynchronous: on macOS each one can
 * wait on the keychain, and a synchronous wait on the main thread kept the app
 * from ever opening a window while a keychain prompt went unanswered.
 */
export interface SecretBox {
  available(): Promise<boolean>
  encrypt(text: string): Promise<Uint8Array>
  /** `stale`: the platform asks for the text to be encrypted again (a rotated key). */
  decrypt(bytes: Uint8Array): Promise<{ readonly text: string; readonly stale: boolean }>
}

export const HUB_SETTINGS_FILE = 'desktop-hub.json'
export const HUB_SESSION_FILE = 'desktop-hub-session.bin'

function text(value: unknown, max: number): string | null {
  return typeof value === 'string' && value.length > 0 && value.length <= max ? value : null
}

/** Reads the address file. Anything that does not validate is treated as "no Hub saved". */
export function parseStoredHub(raw: string): StoredHub | undefined {
  let value: unknown
  try {
    value = JSON.parse(raw)
  } catch {
    return undefined
  }
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined
  const item = value as Record<string, unknown>
  const target = checkHubTarget(item.target)
  if (!target.ok || typeof item.origin !== 'string') return undefined
  let origin: string
  try {
    origin = parseHubOrigin(item.origin)
  } catch {
    return undefined
  }
  return {
    target: target.value,
    origin,
    fingerprint: text(item.fingerprint, 200),
    hubVersion: text(item.hub_version, 64),
    linkedMachineId: text(item.linked_machine_id, 200),
  }
}

function writePrivate(path: string, data: string | Uint8Array): void {
  mkdirSync(dirname(path), { recursive: true })
  const temp = `${path}.tmp-${process.pid}`
  writeFileSync(temp, data, { mode: 0o600 })
  try { chmodSync(temp, 0o600) } catch { /* best effort on platforms without modes */ }
  renameSync(temp, path)
}

export function createHubStorage(directory: string, box: SecretBox): HubCockpitStorage {
  const settingsPath = join(directory, HUB_SETTINGS_FILE)
  const sessionPath = join(directory, HUB_SESSION_FILE)
  return {
    loadSettings() {
      if (!existsSync(settingsPath)) return undefined
      return parseStoredHub(readFileSync(settingsPath, 'utf8'))
    },
    saveSettings(value) {
      if (value === undefined) {
        rmSync(settingsPath, { force: true })
        return
      }
      writePrivate(settingsPath, JSON.stringify({
        version: 1,
        target: value.target,
        origin: value.origin,
        fingerprint: value.fingerprint,
        hub_version: value.hubVersion,
        linked_machine_id: value.linkedMachineId,
      }, null, 2))
    },
    async loadSession() {
      // No saved session: the keychain is never asked.
      if (!existsSync(sessionPath)) return undefined
      if (!(await box.available())) return undefined
      let opened: { readonly text: string; readonly stale: boolean }
      try {
        opened = await box.decrypt(readFileSync(sessionPath))
      } catch {
        // Unreadable now is not proof it is worthless: the file stays, and the
        // next sign-in replaces it.
        return undefined
      }
      if (opened.stale) {
        try { writePrivate(sessionPath, await box.encrypt(opened.text)) } catch { /* the old copy still opens */ }
      }
      return opened.text
    },
    async saveSession(value) {
      if (value === undefined) {
        rmSync(sessionPath, { force: true })
        return
      }
      // No encryption available: keep nothing on disk.
      if (!(await box.available())) {
        rmSync(sessionPath, { force: true })
        return
      }
      writePrivate(sessionPath, await box.encrypt(value))
    },
  }
}
