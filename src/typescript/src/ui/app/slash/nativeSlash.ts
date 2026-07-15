// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SlashExecResponse } from '../../gatewayTypes.js'
import { slashExecText } from '../../lib/slashOutput.js'

import type { SlashRunCtx } from './types.js'

/**
 * Execute a command through the Bun daemon's canonical `/slash` entrypoint.
 *
 * Successful daemon commands normally publish their human-readable result as
 * a `notification` event.  This helper only renders a response body when the
 * daemon sends one directly (notably typed errors), preventing duplicate
 * transcript lines while preserving failures instead of inventing success.
 */
export function runNativeSlash(ctx: SlashRunCtx, command: string, title?: string): void {
  const normalized = command.trim().replace(/^[/!]+/, '')

  if (!normalized) {
    ctx.transcript.sys('error: native command is required')

    return
  }

  ctx.gateway.gw
    .request<SlashExecResponse>('slash.exec', { command: normalized, session_id: ctx.sid })
    .then(response => {
      if (ctx.stale()) {
        return
      }

      const text = slashExecText(response)

      if (!text) {
        return
      }

      const long = text.length > 180 || text.split('\n').filter(Boolean).length > 2

      long ? ctx.transcript.page(text, title ?? normalized.split(/\s+/, 1)[0] ?? 'Command') : ctx.transcript.sys(text)
    })
    .catch(ctx.guardedErr)
}
