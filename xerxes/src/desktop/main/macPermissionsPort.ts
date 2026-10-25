// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The production `MacPermissionsPort` over Electron. Kept apart from
 * macPermissions.ts so tests import the logic without Electron. Nothing here
 * runs a prompt on its own: `isTrustedAccessibilityClient(true)`,
 * `askForMediaAccess` and the screen request are reached only through the
 * Request button.
 */

import { app, desktopCapturer, shell, systemPreferences } from 'electron'

import {
  macPermissionsSettingsFile, readPermissionsDismissed, writePermissionsDismissed,
  type MacPermissionsPort, type MediaAccessAnswer,
} from './macPermissions.js'

export function electronMacPermissionsPort(options: { readonly settingsDirectory: string; readonly appName: string }): MacPermissionsPort {
  const file = macPermissionsSettingsFile(options.settingsDirectory)
  return {
    platform: process.platform,
    // A source launch runs inside Electron's own bundle, which is what
    // Privacy & Security lists (or the terminal that started it).
    appLabel: app.isPackaged ? options.appName : 'Electron (development build)',
    isTrustedAccessibilityClient: prompt => systemPreferences.isTrustedAccessibilityClient(prompt),
    getMediaAccessStatus: (media): MediaAccessAnswer => systemPreferences.getMediaAccessStatus(media),
    askForMicrophone: () => systemPreferences.askForMediaAccess('microphone'),
    requestScreenCapture: async () => {
      // A 1×1 thumbnail: the call exists only so macOS shows its prompt and lists the app.
      await desktopCapturer.getSources({ types: ['screen'], thumbnailSize: { width: 1, height: 1 }, fetchWindowIcons: false })
    },
    openExternal: url => shell.openExternal(url),
    // relaunch() then quit(), not exit(): quitting runs before-quit / will-quit,
    // so the window layout is saved and connections close cleanly. The daemon
    // is detached and keeps running; this never restarts the runtime.
    relaunch: () => { app.relaunch(); app.quit() },
    readDismissed: () => readPermissionsDismissed(file),
    writeDismissed: dismissed => writePermissionsDismissed(file, dismissed),
    now: () => Date.now(),
  }
}
