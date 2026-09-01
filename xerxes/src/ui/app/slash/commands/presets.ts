// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { RpcResult } from '../../../lib/rpc.js'
import type { SlashCommand } from '../types.js'

interface PresetRow extends RpcResult {
  id: string
  name?: string
  description?: string
  trust?: string
  is_default?: boolean
  broken?: string
}

export const presetCommands: SlashCommand[] = [
  {
    name: 'creator',
    help: 'start a fresh visible Creator mode session',
    run: (_argument, ctx) => {
      if (ctx.session.guardBusySessionSwitch('start Creator mode')) return
      ctx.session.newSession('Creator mode started — describe the agent preset you want to build.', 'Creator mode', 'creator')
    },
  },
  {
    name: 'preset',
    aliases: ['presets'],
    help: 'list, select, duplicate, or manage DSH-style agent presets',
    usage: '/preset [list|use <id>|default <id>|copy <from> <id> [name]|remove <id>|creator]',
    run: (argument, ctx) => {
      const [verb = 'list', ...parts] = argument.trim().split(/\s+/).filter(Boolean)
      if (verb === 'list') {
        void ctx.gateway.rpc('agentPreset.list', {}).then(ctx.guarded(result => {
          if (!result || result.ok === false) {
            ctx.transcript.sys(String(result?.error ?? 'could not load agent presets'))
            return
          }
          const rows = Array.isArray(result.presets) ? result.presets as PresetRow[] : []
          const lines = rows.map(row => {
            const flags = [row.trust, row.is_default ? 'default' : '', row.broken ? 'broken' : ''].filter(Boolean).join(' · ')
            return `${row.is_default ? '●' : '○'} ${row.name || row.id} \`${row.id}\`${flags ? ` — ${flags}` : ''}${row.broken ? `\n  ${row.broken}` : row.description ? `\n  ${row.description}` : ''}`
          })
          ctx.transcript.page(lines.join('\n') || 'No agent presets are available.', 'Agent presets')
        })).catch(ctx.guardedErr)
        return
      }
      const use = verb === 'creator' ? 'creator' : parts[0]
      if (verb === 'use' || verb === 'select' || verb === 'creator') {
        if (!use) return ctx.transcript.sys('usage: /preset use <id>')
        void ctx.gateway.rpc('agentPreset.select', { agent_preset: use }).then(ctx.guarded(result => {
          ctx.transcript.sys(result?.ok === false
            ? String(result.error ?? 'preset selection refused')
            : `agent preset \`${String(result?.agent_preset ?? use)}\` selected — fixed after the first turn`)
        })).catch(ctx.guardedErr)
        return
      }
      if (verb === 'default') {
        const id = parts[0]
        if (!id) return ctx.transcript.sys('usage: /preset default <id>')
        void ctx.gateway.rpc('agentPreset.setDefault', { agent_preset: id }).then(ctx.guarded(result => {
          ctx.transcript.config(result?.ok === false
            ? String(result.error ?? 'could not set default preset')
            : `default agent preset: ${id} (future sessions only)`)
        })).catch(ctx.guardedErr)
        return
      }
      if (verb === 'copy') {
        const [from, id, ...name] = parts
        if (!from || !id) return ctx.transcript.sys('usage: /preset copy <from> <id> [display name]')
        void ctx.gateway.rpc('agentPreset.copy', {
          from,
          agent_preset: id,
          ...(name.length ? { name: name.join(' ') } : {}),
        }).then(ctx.guarded(result => {
          ctx.transcript.sys(result?.ok === false
            ? String(result.error ?? 'could not duplicate preset')
            : `created agent preset \`${id}\`${typeof result?.path === 'string' && result.path ? ` at ${result.path}` : ''}`)
        })).catch(ctx.guardedErr)
        return
      }
      if (verb === 'remove') {
        const id = parts[0]
        if (!id) return ctx.transcript.sys('usage: /preset remove <id>')
        void ctx.gateway.rpc('agentPreset.remove', { agent_preset: id }).then(ctx.guarded(result => {
          ctx.transcript.sys(result?.ok === false ? String(result.error ?? 'could not remove preset') : `removed agent preset \`${id}\``)
        })).catch(ctx.guardedErr)
        return
      }
      ctx.transcript.sys('usage: /preset [list|use <id>|default <id>|copy <from> <id> [name]|remove <id>|creator]')
    },
  },
]
