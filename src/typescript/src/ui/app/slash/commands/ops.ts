// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { BrowserManageResponse } from '../../../gatewayTypes.js'
import { getSpawnHistory, setDiffPair, type SpawnSnapshot } from '../../spawnHistoryStore.js'
import { patchOverlayState } from '../../overlayStore.js'
import { runNativeSlash } from '../nativeSlash.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

const nativeUnavailable = (ctx: SlashRunCtx, detail: string) =>
  ctx.transcript.sys(`unavailable in the native Bun daemon: ${detail}`)

export const opsCommands: SlashCommand[] = [
  {
    help: 'cancel the active native turn',
    name: 'stop',
    run: (_arg, ctx) => runNativeSlash(ctx, 'stop', 'Stop')
  },

  {
    aliases: ['reload_mcp'],
    help: 'reconnect native MCP servers',
    name: 'reload-mcp',
    run: (arg, ctx) => runNativeSlash(ctx, `reload-mcp${arg.trim() ? ` ${arg.trim()}` : ''}`, 'Reload MCP')
  },

  {
    help: 'reload native runtime configuration and rediscover skills',
    name: 'reload',
    run: (_arg, ctx) => runNativeSlash(ctx, 'reload', 'Reload')
  },

  {
    help: 'manage a live Chromium CDP connection [connect|disconnect|status]',
    name: 'browser',
    run: (arg, ctx) => {
      const [rawAction = 'status', ...rest] = arg.trim().split(/\s+/).filter(Boolean)
      const action = rawAction.toLowerCase()

      if (!['connect', 'disconnect', 'status'].includes(action)) {
        return ctx.transcript.sys('usage: /browser [connect <http(s)://host:port|ws(s)://…>|disconnect|status]')
      }

      const url = action === 'connect' ? rest.join(' ').trim() : undefined

      if (action === 'connect' && !url) {
        return ctx.transcript.sys('usage: /browser connect <http(s)://host:port|ws(s)://browser-endpoint>')
      }

      if (url) {
        ctx.transcript.sys(`checking Chromium-family browser remote debugging at ${url}...`)
      }

      ctx.gateway
        .rpc<BrowserManageResponse>('browser.manage', { action, ...(url && { url }) })
        .then(
          ctx.guarded<BrowserManageResponse>(r => {
            if (action === 'status') {
              return ctx.transcript.sys(
                r.connected
                  ? `browser connected: ${r.url || '(url unavailable)'}`
                  : 'browser not connected (run /browser connect <CDP endpoint>)'
              )
            }

            if (action === 'disconnect') {
              return ctx.transcript.sys('browser disconnected')
            }

            if (r.connected) {
              ctx.transcript.sys('Browser connected to live Chromium-family browser via CDP')
              ctx.transcript.sys(`Endpoint: ${r.url || '(url unavailable)'}`)
              ctx.transcript.sys(`Pages: ${(r.pages ?? []).length}`)
              ctx.transcript.sys('next browser tool call will use this CDP endpoint')
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'list snapshots or roll back to one',
    name: 'rollback',
    run: (arg, ctx) => {
      const parts = arg.trim().split(/\s+/).filter(Boolean)
      const [command = '', ...rest] = parts
      const lower = command.toLowerCase()

      if (!command || lower === 'list' || lower === 'ls') {
        return runNativeSlash(ctx, 'snapshots', 'Snapshots')
      }

      if (lower === 'diff') {
        return nativeUnavailable(ctx, 'snapshot diffs are not exposed; use /snapshots or /rollback <snapshot-id>.')
      }

      const snapshotId = lower === 'restore' ? rest[0] : command

      if (!snapshotId || (lower !== 'restore' && rest.length > 0) || (lower === 'restore' && rest.length > 1)) {
        return ctx.transcript.sys('usage: /rollback <snapshot-id>  (list with /snapshots)')
      }

      runNativeSlash(ctx, `rollback ${snapshotId}`, 'Rollback')
    }
  },

  {
    aliases: ['tasks'],
    help: 'open the live spawn-tree dashboard',
    name: 'agents',
    run: (arg, ctx) => {
      const sub = arg.trim().toLowerCase()

      if (sub === 'pause' || sub === 'resume' || sub === 'unpause') {
        return nativeUnavailable(
          ctx,
          'per-subagent pause and resume are not exposed. Use /stop to cancel the active turn.'
        )
      }

      if (sub === 'status' || sub === 'list') {
        return runNativeSlash(ctx, 'agents', 'Agents')
      }

      if (sub) {
        return ctx.transcript.sys('usage: /agents [status]  (open /agents with no argument for the live dashboard)')
      }

      patchOverlayState({ agents: true, agentsInitialHistoryIndex: 0 })
    }
  },

  {
    help: 'replay a completed spawn tree from this TUI session',
    name: 'replay',
    run: (arg, ctx) => {
      const history = getSpawnHistory()
      const raw = arg.trim()
      const lower = raw.toLowerCase()

      if (lower === 'list' || lower === 'ls' || lower.startsWith('load ')) {
        return nativeUnavailable(
          ctx,
          'saved spawn-tree replay is not exposed. Only completed trees from this TUI session can be replayed.'
        )
      }

      if (!history.length) {
        return ctx.transcript.sys('no completed spawn trees in this TUI session')
      }

      let index = 1

      if (raw && lower !== 'last') {
        const parsed = Number.parseInt(raw, 10)

        if (Number.isNaN(parsed) || parsed < 1 || parsed > history.length) {
          return ctx.transcript.sys(`replay: index out of range 1..${history.length}`)
        }

        index = parsed
      }

      patchOverlayState({ agents: true, agentsInitialHistoryIndex: index })
    }
  },

  {
    help: 'diff two completed spawn trees from this TUI session',
    name: 'replay-diff',
    run: (arg, ctx) => {
      const parts = arg.trim().split(/\s+/).filter(Boolean)

      if (parts.length !== 2) {
        return ctx.transcript.sys('usage: /replay-diff <a> <b>  (e.g. /replay-diff 1 2 for last two)')
      }

      const [a, b] = parts
      const history = getSpawnHistory()

      const resolve = (token: string): null | SpawnSnapshot => {
        const n = Number.parseInt(token, 10)

        return Number.isFinite(n) && n >= 1 && n <= history.length ? (history[n - 1] ?? null) : null
      }

      const baseline = resolve(a!)
      const candidate = resolve(b!)

      if (!baseline || !candidate) {
        return ctx.transcript.sys(`replay-diff: could not resolve indices · history has ${history.length} entries`)
      }

      setDiffPair({ baseline, candidate })
      patchOverlayState({ agents: true, agentsInitialHistoryIndex: 0 })
    }
  },

  {
    aliases: ['reload_skills'],
    help: 'rediscover native skills by reloading the runtime',
    name: 'reload-skills',
    run: (_arg, ctx) => runNativeSlash(ctx, 'reload', 'Reload')
  },

  {
    help: 'list native skills',
    name: 'skills',
    run: (arg, ctx) => {
      const [sub = '', ...rest] = arg.trim().split(/\s+/).filter(Boolean)
      const lower = sub.toLowerCase()

      if (!sub || lower === 'list') {
        return runNativeSlash(ctx, 'skills', 'Skills')
      }

      if (lower === 'inspect') {
        return nativeUnavailable(
          ctx,
          'skill inspection is not exposed. Use /skills to list skills or /skill <name> to activate one.'
        )
      }

      if (lower === 'search' || lower === 'install' || lower === 'browse') {
        return nativeUnavailable(
          ctx,
          'skill search, installation, and browsing are not exposed by the Bun daemon. Use /skills for discovered skills.'
        )
      }

      if (rest.length) {
        return ctx.transcript.sys('usage: /skills [list]  (activate a discovered skill with /skill <name>)')
      }

      ctx.transcript.sys('usage: /skills [list]  (activate a discovered skill with /skill <name>)')
    }
  },

  {
    help: 'list native plugins and plugin slash commands',
    name: 'plugins',
    run: (arg, ctx) => {
      const sub = arg.trim().split(/\s+/, 1)[0]?.toLowerCase()

      if (!sub || sub === 'list') {
        return runNativeSlash(ctx, 'plugins', 'Plugins')
      }

      nativeUnavailable(
        ctx,
        'plugin installation and enable/disable controls are not exposed. Use /plugins to list loaded native plugins.'
      )
    }
  },

  {
    help: 'list native tools',
    name: 'tools',
    run: (arg, ctx) => {
      const sub = arg.trim().split(/\s+/, 1)[0]?.toLowerCase()

      if (!sub || sub === 'list') {
        return runNativeSlash(ctx, 'tools', 'Tools')
      }

      if (sub === 'enable' || sub === 'disable') {
        return nativeUnavailable(
          ctx,
          'tool enable/disable configuration is not exposed. Use /tools to inspect the active native tool catalogue.'
        )
      }

      ctx.transcript.sys('usage: /tools [list]')
    }
  }
]
