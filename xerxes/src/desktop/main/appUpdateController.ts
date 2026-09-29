// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The app-wide update flow around `appUpdate.ts`: check at startup and every
 * few hours, remember a skipped version, broadcast the state to every window,
 * and install only when a renderer relays the person's click.
 */

import { execFile, spawn } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { app, ipcMain, shell, webContents } from 'electron'

import { appBundleOf, checkForAppUpdate, installAppUpdate, type AvailableRelease } from './appUpdate.js'

export type AppUpdateState =
  | { readonly phase: 'idle' }
  | { readonly phase: 'checking' }
  | { readonly phase: 'available'; readonly release: AvailableRelease; readonly installable: boolean; readonly reason?: string }
  | { readonly phase: 'downloading'; readonly release: AvailableRelease; readonly received: number; readonly total: number }
  | { readonly phase: 'restarting'; readonly release: AvailableRelease }
  | { readonly phase: 'error'; readonly message: string; readonly release?: AvailableRelease }

const FIRST_CHECK_DELAY_MS = 20_000
const CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000

export function startAppUpdates(options: { readonly settingsDirectory: string; readonly version: string }): { readonly check: () => Promise<AppUpdateState> } {
  const settingsFile = join(options.settingsDirectory, 'desktop-update.json')
  let state: AppUpdateState = { phase: 'idle' }
  let installing = false
  const broadcast = () => {
    for (const contents of webContents.getAllWebContents()) {
      if (!contents.isDestroyed()) contents.send('desktop:app-update-state', state)
    }
  }
  const set = (next: AppUpdateState) => { state = next; broadcast() }
  const skipped = async (): Promise<string | undefined> => {
    try {
      const value: unknown = JSON.parse(await readFile(settingsFile, 'utf8'))
      return value && typeof value === 'object' && typeof (value as Record<string, unknown>).skipped === 'string' ? (value as { skipped: string }).skipped : undefined
    } catch { return undefined }
  }
  const installability = (release: AvailableRelease): { installable: boolean; reason?: string } => {
    if (!app.isPackaged) return { installable: false, reason: 'This is a development build; update it from source.' }
    if (!release.asset) return { installable: false, reason: 'This release has no installer for this computer.' }
    if (!appBundleOf(app.getPath('exe'))) return { installable: false, reason: 'The app is not running from an app bundle.' }
    return { installable: true }
  }
  const check = async (manual: boolean): Promise<AppUpdateState> => {
    if (installing) return state
    if (manual) set({ phase: 'checking' })
    try {
      const result = await checkForAppUpdate({ currentVersion: options.version, platform: process.platform, arch: process.arch, fetch: (url, init) => fetch(url, init) })
      if (result.kind === 'current') { set({ phase: 'idle' }); return state }
      // A version the person chose to skip stays quiet until a newer one, or
      // until they ask explicitly.
      if (!manual && (await skipped()) === result.release.version) { set({ phase: 'idle' }); return state }
      set({ phase: 'available', release: result.release, ...installability(result.release) })
    } catch (error) {
      if (manual) set({ phase: 'error', message: error instanceof Error ? error.message : String(error) })
    }
    return state
  }
  const install = async (): Promise<AppUpdateState> => {
    if (state.phase !== 'available' || !state.installable || !state.release.asset || installing) return state
    const release = state.release
    const asset = release.asset!
    installing = true
    set({ phase: 'downloading', release, received: 0, total: asset.size })
    let lastBroadcast = 0
    try {
      await installAppUpdate({
        asset,
        appBundle: appBundleOf(app.getPath('exe'))!,
        pid: process.pid,
        ports: {
          fetch: (url, init) => fetch(url, init),
          run: argv => new Promise((resolve, reject) => execFile(argv[0]!, argv.slice(1), { maxBuffer: 4 * 1024 * 1024 }, (error, stdout, stderr) => error ? reject(new Error(String(stderr || error.message).trim())) : resolve(String(stdout)))),
          spawnDetached: argv => spawn(argv[0]!, argv.slice(1), { detached: true, stdio: 'ignore' }).unref(),
        },
        onProgress: (received, total) => {
          const now = Date.now()
          if (now - lastBroadcast < 250 && received < total) return
          lastBroadcast = now
          set({ phase: 'downloading', release, received, total })
        },
      })
      set({ phase: 'restarting', release })
      // The swap script is waiting for this process to exit.
      setTimeout(() => app.quit(), 400)
    } catch (error) {
      installing = false
      set({ phase: 'error', release, message: error instanceof Error ? error.message : String(error) })
    }
    return state
  }

  ipcMain.handle('desktop:app-update', async (_event, action: unknown) => {
    switch (action) {
      case 'state': return state
      case 'check': return check(true)
      case 'install': return install()
      case 'skip':
        if (state.phase === 'available') await writeFile(settingsFile, JSON.stringify({ skipped: state.release.version }), 'utf8')
        set({ phase: 'idle' })
        return state
      case 'dismiss':
        if (!installing) set({ phase: 'idle' })
        return state
      case 'open-release': {
        const url = 'release' in state && state.release ? state.release.pageUrl : `https://github.com/erfanzar/Xerxes-Agents/releases/latest`
        if (/^https:\/\/github\.com\//.test(url)) await shell.openExternal(url)
        return state
      }
      default: throw new Error('Unknown app update action')
    }
  })
  setTimeout(() => void check(false), FIRST_CHECK_DELAY_MS).unref?.()
  setInterval(() => void check(false), CHECK_INTERVAL_MS).unref?.()
  return { check: () => check(true) }
}
