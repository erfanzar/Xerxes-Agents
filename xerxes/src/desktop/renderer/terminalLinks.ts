// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { externalUrlOf } from '../chatMediaShared.js'
import { openExternalLink } from './chatLinks.js'
import type { XerxesBridge } from './types.js'

/**
 * OSC 8 labels can hide their destination, so keep a confirmation showing the
 * real URL. xterm's default opens a blank popup before assigning its location;
 * Electron denies that popup, so the destination never reaches the browser.
 * Use the same validated host handoff as chat links instead. Return an error
 * for the view to show, never write it into the shell or echo OAuth URL secrets.
 */
export async function openTerminalLink(
  uri: string,
  confirm: (message: string) => boolean,
  bridge: Pick<XerxesBridge, 'openExternal'>,
): Promise<string | null> {
  const url = externalUrlOf(uri)
  // Preserve xterm's HTTP(S)-only policy even when called outside its linkifier.
  if (!url || !/^https?:/i.test(url)) return 'Only valid HTTP and HTTPS terminal links can be opened.'
  try {
    if (!confirm(`Open this link in your default browser?\n\n${url}\n\nOnly continue if you trust this destination.`)) return null
    await openExternalLink(url, bridge)
    return null
  } catch {
    // Host errors can contain the full authorization URL; keep those out of
    // both the visible error and the terminal's retained transcript.
    return 'Could not open the link in your default browser. Copy the link and open it manually.'
  }
}
