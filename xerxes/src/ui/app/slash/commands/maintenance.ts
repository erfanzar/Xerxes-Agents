// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { NO_CONFIRM_DESTRUCTIVE } from '../../../config/env.js'
import { patchOverlayState } from '../../overlayStore.js'
import type { SlashCommand } from '../types.js'

interface WipeResponse {
  ok?: boolean
  error?: string
  removed?: { bytes?: number; files?: number }
}

const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KiB', 'MiB', 'GiB']
  let value = bytes
  let unit = 'B'
  for (const next of units) {
    if (value < 1024) break
    value /= 1024
    unit = next
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${unit}`
}

const wipeCommand = (
  name: string,
  help: string,
  rpc: string,
  confirmTitle: string,
  detail: string,
  doneLabel: string
): SlashCommand => ({
  help,
  name,
  run: (_arg, ctx) => {
    const commit = () => {
      ctx.gateway
        .rpc<WipeResponse>(rpc, {})
        .then(
          ctx.guarded<WipeResponse>(r => {
            if (!r?.ok) {
              ctx.transcript.sys(`could not ${doneLabel}: ${r?.error ?? 'unknown error'}`)
              return
            }
            const removed = r.removed ?? {}
            ctx.transcript.sys(
              `${doneLabel}: ${removed.files ?? 0} file(s), ${formatBytes(removed.bytes ?? 0)} removed`
            )
          })
        )
        .catch(ctx.guardedErr)
    }

    if (NO_CONFIRM_DESTRUCTIVE) {
      return commit()
    }

    patchOverlayState({
      confirm: {
        cancelLabel: 'Cancel',
        confirmLabel: 'Yes, wipe it',
        danger: true,
        detail,
        onConfirm: commit,
        title: confirmTitle
      }
    })
  }
})

export const maintenanceCommands: SlashCommand[] = [
  wipeCommand(
    'remove-memory',
    'globally wipe all Xerxes agent memory (global, project, and SQLite stores)',
    'daemon.wipe_memory',
    'Wipe ALL Xerxes memory?',
    'This permanently deletes every Xerxes memory store — global and per-project agent memory, plus the SQLite memory databases — for every session. It cannot be undone.',
    'memory wiped'
  ),
  wipeCommand(
    'remove-history',
    'globally wipe all saved chat history and snapshots',
    'daemon.wipe_history',
    'Wipe ALL chat history?',
    'This permanently deletes every saved session transcript, the search index, and all snapshots. Open sessions keep running but their saved history is gone. It cannot be undone.',
    'history wiped'
  )
]
