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
  { name: 'loop', group: 'activity', help: 'manage bounded follow-ups for this conversation', run: (arg, ctx) => { if (!arg.trim()) patchOverlayState({ loops: true, schedules: false }); else return runNativeSlash(ctx, `loop ${arg.trim()}`, 'Follow-ups') } },
  { name: 'context', group: 'info', help: 'inspect context, pin memory and exclude stale recall', run: (arg, ctx) => { if (!arg.trim()) patchOverlayState({ contextInspector: true }); else ctx.transcript.sys('usage: /context') } },
  { name: 'mcp', help: 'inspect MCP health [status|reconnect <name>]', run: (arg, ctx) => runNativeSlash(ctx, `mcp ${arg.trim()}`, 'MCP connections') },
  { name: 'snapshots', group: 'session', help: 'browse snapshot timeline [list]', run: (arg, ctx) => { if (!arg.trim()) patchOverlayState({ snapshots: true }); else if (arg.trim() === 'list') runNativeSlash(ctx, 'snapshots', 'Snapshots'); else ctx.transcript.sys('usage: /snapshots [list]') } },
  { name: 'workspaces', group: 'activity', help: 'review retained agent workspaces [list|after <cursor>|inspect <id>]', run: (arg, ctx) => { if (!arg.trim()) patchOverlayState({ workspaces: true }); else runNativeSlash(ctx, `workspaces ${arg.trim()}`, 'Agent workspaces') } },
  { name: 'config', group: 'config', help: 'configure agents, MCP or LSP servers [agents|mcp|lsp]', run: (arg, ctx) => {
    if (!arg.trim() || arg.trim() === 'agents') patchOverlayState({ agentSettings: true })
    else if (arg.trim() === 'lsp') patchOverlayState({ lspSettings: true })
    else if (arg.trim() === 'mcp') patchOverlayState({ mcpSettings: true })
    else runNativeSlash(ctx, 'config', 'Runtime configuration')
  } },
  {
    name: 'lsp',
    help: 'inspect workspace language servers [status|release <name>]',
    run: (arg, ctx) => runNativeSlash(ctx, `lsp ${arg.trim()}`, 'Language servers')
  },
  {
    name: 'hooks',
    help: 'inspect hooks [list|preview <event> [tool-name]|failures [event]]',
    run: (arg, ctx) => runNativeSlash(ctx, `hooks ${arg.trim()}`, 'Hooks')
  },
  {
    name: 'schedules',
    group: 'activity',
    help: 'manage workspace schedules [list|add|pause|resume|run|remove|legacy|migrate]',
    run: (arg, ctx) => {
      if (!arg.trim()) { patchOverlayState({ schedules: true, loops: false }); return }
      return runNativeSlash(ctx, `schedules ${arg.trim()}`, 'Schedules')
    }
  },
  {
    name: 'activity',
    group: 'activity',
    help: 'inspect background shells, watches and schedules; stop or pause work',
    run: () => { patchOverlayState({ activity: true }) }
  },
  {
    name: 'monitors',
    group: 'activity',
    help: 'create and inspect terminal, file, WebSocket and webhook watches',
    run: (arg, ctx) => {
      if (!arg.trim()) { patchOverlayState({ monitors: true }); return }
      return runNativeSlash(ctx, `monitors ${arg.trim()}`, 'Monitors')
    }
  },
  {
    help: 'inspect durable runs and unread results [list|unread|inspect|ack]',
    name: 'runs',
    group: 'activity',
    run: (arg, ctx) => {
      if (!arg.trim()) { patchOverlayState({ runs: true }); return }
      runNativeSlash(ctx, `runs ${arg.trim()}`, 'Runs')
    }
  },
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
    help: 'list snapshots, preview with diff <id>, or restore one',
    name: 'rollback',
    run: (arg, ctx) => {
      const parts = arg.trim().split(/\s+/).filter(Boolean)
      const [command = '', ...rest] = parts
      const lower = command.toLowerCase()

      if (!command || lower === 'list' || lower === 'ls') {
        return runNativeSlash(ctx, 'snapshots', 'Snapshots')
      }

      if (lower === 'diff') {
        if (rest.length !== 1) return ctx.transcript.sys('usage: /rollback diff <snapshot-id>')
        return runNativeSlash(ctx, `rollback diff ${rest[0]}`, 'Snapshot restore preview')
      }

      if (lower === 'apply') {
        if (rest.length !== 2 || !/^[a-f0-9]{64}$/.test(rest[1]!)) return ctx.transcript.sys('usage: /rollback apply <snapshot-id> <preview-revision>')
        return runNativeSlash(ctx, `rollback apply ${rest[0]} ${rest[1]}`, 'Restore reviewed snapshot')
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

      patchOverlayState({ agents: true, agentsInitialHistoryIndex: 0, agentsInspectId: null })
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

      patchOverlayState({ agents: true, agentsInitialHistoryIndex: index, agentsInspectId: null })
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
      patchOverlayState({ agents: true, agentsInitialHistoryIndex: 0, agentsInspectId: null })
    }
  },

  {
    aliases: ['reload_skills'],
    help: 'rediscover native skills by reloading the runtime',
    name: 'reload-skills',
    run: (_arg, ctx) => runNativeSlash(ctx, 'reload', 'Reload')
  },

  {
    help: 'list, search, install local skills, inspect or trust skills and commands',
    name: 'skills',
    run: (arg, ctx) => {
      const [sub = '', ...rest] = arg.trim().split(/\s+/).filter(Boolean)
      const lower = sub.toLowerCase()

      if (!sub || lower === 'list') {
        return runNativeSlash(ctx, 'skills', 'Skills')
      }

      if (lower === 'inspect') {
        return runNativeSlash(ctx, `skills inspect ${rest.join(' ')}`, 'Skill inspection')
      }
      if (lower === 'trust') {
        return runNativeSlash(ctx, `skills trust ${rest.join(' ')}`, 'Trust skill')
      }
      if (lower === 'diagnostics') {
        return runNativeSlash(ctx, `skills diagnostics ${rest.join(' ')}`.trim(), 'Skill diagnostics')
      }

      if (lower === 'search' || lower === 'install' || lower === 'browse') {
        return runNativeSlash(ctx, `skills ${arg.trim()}`, 'Skills')
      }

      if (rest.length) {
        return ctx.transcript.sys('usage: /skills [list|inspect <name>|diagnostics|trust <name>]  (activate with /skill <name>)')
      }

      ctx.transcript.sys('usage: /skills [list|inspect <name>|diagnostics|trust <name>]  (activate with /skill <name>)')
    }
  },

  {
    help: 'list, inspect, install local tool modules; enable, disable, reload or uninstall plugins',
    name: 'plugins',
    run: (arg, ctx) => {
      const sub = arg.trim().split(/\s+/, 1)[0]?.toLowerCase()

      if (!sub || sub === 'list') {
        return runNativeSlash(ctx, 'plugins', 'Plugins')
      }
      if (sub === 'inspect') return runNativeSlash(ctx, `plugins ${arg.trim()}`, 'Plugin inspection')

      return runNativeSlash(ctx, `plugins ${arg.trim()}`, 'Plugins')
    }
  },

  {
    help: 'inspect loaded, deferred and filtered native tools',
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
