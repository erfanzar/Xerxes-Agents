// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The Hub address used to be typed into the XERXES tab and kept in
 * localStorage. It is now saved by the desktop's main process (Settings → Hub &
 * phone), which also signs in and opens the socket. Only the cleanup of the old
 * key remains here, so a stale typed address cannot linger on disk.
 */

import { displayOrigin } from '../hubShared.js'

const LEGACY_HUB_URL_KEY = 'xerxes.desktop.hub-url.v1'

/** Forget the address an earlier version stored in this page's localStorage. */
export function clearLegacyHubUrl(): void {
  try { localStorage.removeItem(LEGACY_HUB_URL_KEY) } catch { /* nothing to clear, or storage is unavailable */ }
}

/** What the XERXES tab header shows for the Hub it is connected to. */
export function hubLabelFor(origin: string): string {
  return displayOrigin(origin)
}
