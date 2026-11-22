// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { SwarmActivateResponse, SwarmKillResponse } from '../../../gatewayTypes.js'
import {
  formatSwarmChannelsPage,
  formatSwarmStatusPage,
  normalizeSwarmStatus
} from '../../../lib/swarmFormat.js'
import { patchOverlayState } from '../../overlayStore.js'
import { applySwarmStatus, refreshSwarmSession } from '../../swarmStore.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

const SWARM_MEMBER_TARGET_RE = /^(?:all|[a-z0-9][a-z0-9_-]{0,23})$/i

function sessionParam(ctx: SlashRunCtx): { session_id?: string } {
  return ctx.sid ? { session_id: ctx.sid } : {}
}

function openSwarmBoard(ctx: SlashRunCtx): void {
  patchOverlayState({ swarm: true })
  void refreshSwarmSession(ctx.gateway.rpc, ctx.sid, { reportError: false }).catch(() => {})
}

function runSwarmKill(target: string, ctx: SlashRunCtx): void {
  ctx.gateway
    .rpc<SwarmKillResponse>('swarm.kill', { ...sessionParam(ctx), name: target })
    .then(
      ctx.guarded<SwarmKillResponse>(result => {
        if (result.ok === false) {
          ctx.transcript.sys(`error: ${result.error || `failed to kill ${target}`}`)
          return
        }
        void refreshSwarmSession(ctx.gateway.rpc, ctx.sid, { reportError: false }).catch(() => {})
        const killed = Array.isArray(result.killed) ? result.killed.filter(Boolean) : []
        if (killed.length > 0) {
          ctx.transcript.sys(
            `Killed Immortals member${killed.length === 1 ? '' : 's'}: ${killed.map(name => `@${name}`).join(', ')}`
          )
          return
        }
        ctx.transcript.sys(
          target.toLowerCase() === 'all'
            ? 'Stopped all active Immortals workers.'
            : `Killed Immortals member @${target}.`
        )
      })
    )
    .catch(ctx.guardedErr)
}

export const swarmCommands: SlashCommand[] = [
  {
    name: 'immortals',
    aliases: ['swarm'],
    group: 'activity',
    help: 'coordinate a parallel multi-agent Immortals force or open the Immortals Board [<task>|board|status|channels|kill <name|all>|stop]',
    run: (arg, ctx) => {
      const trimmed = arg.trim()
      const lower = trimmed.toLowerCase()

      if (!trimmed || lower === 'board') {
        openSwarmBoard(ctx)
        return
      }

      if (lower === 'status') {
        void refreshSwarmSession(ctx.gateway.rpc, ctx.sid, { reportError: false })
          .then(({ status, board }) => {
            if (ctx.stale()) return
            if (!status) {
              ctx.transcript.sys('error: could not inspect Immortals status')
              return
            }
            if (!status.active && status.members.length === 0 && (board?.messages.length ?? 0) === 0) {
              ctx.transcript.sys(
                'No active Immortals run in this session. Run /immortals <task> to start one or /immortals board to open the Immortals UI.'
              )
              return
            }
            ctx.transcript.page(formatSwarmStatusPage(status, board?.messages ?? []), 'Immortals')
          })
          .catch(ctx.guardedErr)
        return
      }

      if (lower === 'channels') {
        ctx.gateway
          .rpc<Record<string, unknown>>('swarm.status', sessionParam(ctx))
          .then(
            ctx.guarded(raw => {
              if (raw.ok === false) {
                ctx.transcript.sys(`error: ${String(raw.error ?? 'could not list Immortals channels')}`)
                return
              }
              const status = applySwarmStatus(normalizeSwarmStatus(raw, ctx.sid ?? ''), ctx.sid)
              const formatted = formatSwarmChannelsPage(status)
              const lineCount = formatted.split('\n').filter(Boolean).length
              lineCount > 2 ? ctx.transcript.page(formatted, 'Immortals Channels') : ctx.transcript.sys(formatted)
            })
          )
          .catch(ctx.guardedErr)
        return
      }

      if (lower === 'stop') {
        runSwarmKill('all', ctx)
        return
      }

      const tokens = trimmed.split(/\s+/)
      if (tokens[0]?.toLowerCase() === 'kill' && tokens.length <= 2) {
        const target = tokens[1]?.trim() ?? ''
        if (!target || !SWARM_MEMBER_TARGET_RE.test(target)) {
          ctx.transcript.sys('usage: /immortals kill <name|all>')
          return
        }
        runSwarmKill(target, ctx)
        return
      }

      // `/immortals <task prompt>` (or legacy `/swarm <task prompt>`): activate Immortals
      // coordinator mode for the current session and submit `<task prompt>` as the coordinator's turn.
      ctx.gateway
        .rpc<SwarmActivateResponse>('swarm.activate', { ...sessionParam(ctx), enabled: true })
        .then(
          ctx.guarded<SwarmActivateResponse>(result => {
            if (result.ok === false) {
              ctx.transcript.sys(`error: ${result.error || 'failed to activate Immortals coordinator'}`)
              return
            }
            ctx.transcript.send(trimmed)
          })
        )
        .catch(ctx.guardedErr)
    }
  }
]
