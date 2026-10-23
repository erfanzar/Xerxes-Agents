// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Registers the `hub.cockpit.<action>` IPC channels. Electron is injected as a
 * narrow interface so the wiring can be tested without it. Every channel
 * validates the action name against the fixed list, hands the untrusted
 * params to the cockpit (which validates them again), and answers with a
 * plain `{ ok }` object; a thrown error never reaches the renderer raw.
 */

import { COCKPIT_ACTIONS, COCKPIT_CHANNEL_PREFIX, type CockpitFailure, type CockpitResult } from '../hubShared.js'
import type { HubCockpit } from './hubCockpit.js'

export interface CockpitIpcMain {
  handle(channel: string, listener: (event: { readonly sender: { readonly id: number } }, ...args: unknown[]) => unknown): void
}

export interface CockpitSenderEvents {
  /** Called when the page with this id is destroyed. */
  onDestroyed(senderId: number, callback: () => void): void
}

export function cockpitChannels(): readonly string[] {
  return COCKPIT_ACTIONS.map(action => COCKPIT_CHANNEL_PREFIX + action)
}

export function registerHubCockpitIpc(ipc: CockpitIpcMain, cockpit: HubCockpit, senders: CockpitSenderEvents): void {
  const watched = new Set<number>()
  for (const action of COCKPIT_ACTIONS) {
    ipc.handle(COCKPIT_CHANNEL_PREFIX + action, async (event, params): Promise<CockpitResult> => {
      const senderId = event.sender.id
      if (!watched.has(senderId)) {
        watched.add(senderId)
        senders.onDestroyed(senderId, () => {
          watched.delete(senderId)
          cockpit.releaseSender(senderId)
        })
      }
      try {
        return await cockpit.handle(action, params, senderId)
      } catch (error) {
        const failure: CockpitFailure = {
          ok: false,
          code: 'cli_failed',
          message: `The Hub action failed: ${(error instanceof Error ? error.message : String(error)).replace(/\s+/g, ' ').slice(0, 200)}`,
        }
        return failure
      }
    })
  }
}
