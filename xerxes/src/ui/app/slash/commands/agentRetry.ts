// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { retrySubagent } from '../../../lib/agentRetry.js'
import { patchOverlayState } from '../../overlayStore.js'
import { runNativeSlash } from '../nativeSlash.js'
import type { SlashCommand } from '../types.js'

const AGENTS_USAGE =
  'usage: /agents [status]  ·  /agents retry <name-or-id> [follow-up message]  ·  bare /agents opens the live dashboard'

/**
 * Retry-aware `/agents` registration. It shadows the stock dashboard command
 * (registered earlier in the registry) and delegates every non-retry form to
 * identical behavior, adding only `/agents retry <name-or-id> [message]`.
 * Retry resumes a dead (failed/cancelled/interrupted) subagent under its
 * stable identity through the daemon `subagent.retry` RPC instead of spawning
 * a replacement agent.
 */
export const agentRetryCommands: SlashCommand[] = [
  {
    help: 'open the agents dashboard · retry a failed agent in place',
    name: 'agents',
    run: (arg, ctx) => {
      const parts = arg.trim().split(/\s+/).filter(Boolean)
      const sub = parts[0]?.toLowerCase() ?? ''

      if (sub === 'retry') {
        const target = parts[1]?.trim()

        if (!target) {
          return ctx.transcript.sys('usage: /agents retry <name-or-id> [follow-up message]')
        }

        const message = parts.slice(2).join(' ').trim()

        ctx.transcript.sys(`retrying agent \`${target}\` — same identity, prior conversation retained when available…`)

        retrySubagent(ctx.gateway.rpc, target, message || undefined)
          .then(
            ctx.guarded(response => {
              if (response.ok === false) {
                return ctx.transcript.sys(`retry failed: ${response.error?.trim() || 'the daemon rejected the retry'}`)
              }

              const label = response.agent?.title?.trim() || response.agent?.name?.trim() || target
              const status = response.agent?.status?.trim() || 'running'

              ctx.transcript.sys(`agent \`${label}\` resumed (${status}) — watch the agents panel for progress`)
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      // Non-retry forms mirror the stock dashboard command exactly.
      if (sub === 'pause' || sub === 'resume' || sub === 'unpause') {
        return ctx.transcript.sys(
          'unavailable in the native Bun daemon: per-subagent pause and resume are not exposed. ' +
            'Use /agents retry <name-or-id> to resume a dead agent, or /stop to cancel the active turn.'
        )
      }

      if (sub === 'status' || sub === 'list') {
        return runNativeSlash(ctx, 'agents', 'Agents')
      }

      if (sub) {
        return ctx.transcript.sys(AGENTS_USAGE)
      }

      patchOverlayState({ agents: true, agentsInitialHistoryIndex: 0, agentsInspectId: null })
    },
    usage: '/agents [status|retry <name-or-id> [message]]'
  }
]
