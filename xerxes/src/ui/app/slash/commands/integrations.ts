// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SlashCommand } from '../types.js'

/** One channel gateway row from channel.list / channel.enable / channel.disable. */
interface ChannelRow {
  adapter_name?: string
  enabled?: boolean
  last_error?: string
  last_operation?: string
  name?: string
}

interface ChannelListResponse {
  channels?: ChannelRow[]
  channels_available?: boolean
  channels_configured?: boolean
  error?: string
  ok?: boolean
}

/** One provider profile row from provider_list. Keys never cross the wire. */
interface ProviderProfileRow {
  active?: boolean
  base_url?: string
  model?: string
  name?: string
  provider?: string
}

interface ProviderListResponse {
  error?: string
  ok?: boolean
  profiles?: ProviderProfileRow[]
}

interface ProviderTypeRow {
  api_key_env?: string
  base_url?: string
  name?: string
}

interface ProviderTypesResponse {
  ok?: boolean
  types?: ProviderTypeRow[]
}

interface CreatorTraceRow {
  action?: string
  at?: number | string
  detail?: string
  name?: string
  status?: string
  version?: string
}

interface CreatorTraceResponse {
  error?: string
  ok?: boolean
  trace?: CreatorTraceRow[]
}

interface WorktreeResponse {
  error?: string
  ok?: boolean
  path?: string
  worktree?: string
}

function renderChannels(r: ChannelListResponse): string {
  const rows = r.channels ?? []
  if (!rows.length) {
    return 'no channel gateways known to the daemon'
  }
  const lines = rows.map(row => {
    const state = row.enabled ? 'enabled' : 'disabled'
    const adapter = row.adapter_name && row.adapter_name !== row.name ? ` (${row.adapter_name})` : ''
    const error = row.last_error ? `\n    last error: ${row.last_error}` : ''
    return `${row.enabled ? '●' : '○'} ${row.name ?? '?'}${adapter} — ${state}${error}`
  })
  return [
    `channels: ${r.channels_configured ? 'configured' : 'not configured'} · manager ${r.channels_available ? 'available' : 'unavailable'}`,
    ...lines,
    '',
    'toggle with /channels enable <name> or /channels disable <name>'
  ].join('\n')
}

function renderProviders(r: ProviderListResponse): string {
  const rows = r.profiles ?? []
  if (!rows.length) {
    return 'no provider profiles saved — add one with /providers add <name> --type <type> --model <model> [--key <key>] [--base-url <url>]'
  }
  const widest = Math.max(...rows.map(row => (row.name ?? '?').length))
  return rows
    .map(row => {
      const name = (row.name ?? '?').padEnd(widest)
      const active = row.active ? '● active  ' : '          '
      const provider = row.provider ?? 'custom'
      return `${active}${name}  ${provider}  ${row.model ?? '(no model)'}  ${row.base_url ?? ''}`
    })
    .join('\n')
}

/** Parse `/providers add` flags: --type, --model, --key, --base-url. */
function parseProviderAdd(arg: string): null | {
  baseUrl?: string
  key?: string
  model?: string
  name: string
  type?: string
} {
  const words = arg.trim().split(/\s+/).filter(Boolean)
  const name = words[1]
  if (!name) return null
  const out: { baseUrl?: string; key?: string; model?: string; name: string; type?: string } = { name }
  for (let index = 2; index < words.length; index += 2) {
    const flag = words[index]
    const value = words[index + 1]
    if (!value || value.startsWith('--')) return null
    if (flag === '--type') out.type = value
    else if (flag === '--model') out.model = value
    else if (flag === '--key') out.key = value
    else if (flag === '--base-url') out.baseUrl = value
    else return null
  }
  return out
}

export const integrationCommands: SlashCommand[] = [
  {
    help: 'list, enable, or disable channel gateways [list|enable <name>|disable <name>]',
    name: 'channels',
    run: (arg, ctx) => {
      const [sub = 'list', ...rest] = arg.trim().split(/\s+/).filter(Boolean)
      const lower = sub.toLowerCase()

      if (lower === 'list' || lower === 'ls' || lower === 'status' || !arg.trim()) {
        ctx.gateway
          .rpc<ChannelListResponse>('channel.list', {})
          .then(ctx.guarded(r => ctx.transcript.page(renderChannels(r), 'Channels')))
          .catch(ctx.guardedErr)
        return
      }

      if (lower === 'enable' || lower === 'disable') {
        const name = rest.join(' ').trim()
        if (!name) {
          ctx.transcript.sys(`usage: /channels ${lower} <name>`)
          return
        }
        ctx.gateway
          .rpc<ChannelListResponse>(`channel.${lower}`, { name })
          .then(
            ctx.guarded(r => {
              if (r.ok === false) {
                ctx.transcript.sys(`channel ${lower} failed: ${r.error ?? 'unknown error'}`)
                return
              }
              ctx.transcript.page(renderChannels(r), 'Channels')
            })
          )
          .catch(ctx.guardedErr)
        return
      }

      ctx.transcript.sys('usage: /channels [list|enable <name>|disable <name>]')
    }
  },

  {
    help: 'manage provider profiles [list|use <name>|add <name> --type … --model … [--key …] [--base-url …]|remove <name>|types]',
    name: 'providers',
    run: (arg, ctx) => {
      const [sub = 'list', ...rest] = arg.trim().split(/\s+/).filter(Boolean)
      const lower = sub.toLowerCase()

      const list = () =>
        ctx.gateway
          .rpc<ProviderListResponse>('provider_list', {})
          .then(ctx.guarded(r => ctx.transcript.page(renderProviders(r), 'Providers')))
          .catch(ctx.guardedErr)

      if (lower === 'list' || lower === 'ls' || !arg.trim()) {
        list()
        return
      }

      if (lower === 'types') {
        ctx.gateway
          .rpc<ProviderTypesResponse>('provider_types', {})
          .then(
            ctx.guarded(r => {
              const rows = r.types ?? []
              ctx.transcript.page(
                rows.length
                  ? rows
                      .map(t => `${(t.name ?? '?').padEnd(16)} ${t.base_url ?? ''}${t.api_key_env ? `  (env: ${t.api_key_env})` : ''}`)
                      .join('\n')
                  : 'no provider types registered',
                'Provider types'
              )
            })
          )
          .catch(ctx.guardedErr)
        return
      }

      if (lower === 'use' || lower === 'select') {
        const name = rest.join(' ').trim()
        if (!name) {
          ctx.transcript.sys('usage: /providers use <name>')
          return
        }
        ctx.gateway
          .rpc('provider_select', { name })
          .then(ctx.guarded(() => list()))
          .catch(ctx.guardedErr)
        return
      }

      if (lower === 'remove' || lower === 'delete' || lower === 'rm') {
        const name = rest.join(' ').trim()
        if (!name) {
          ctx.transcript.sys('usage: /providers remove <name>')
          return
        }
        ctx.gateway
          .rpc('provider_delete', { name })
          .then(ctx.guarded(() => list()))
          .catch(ctx.guardedErr)
        return
      }

      if (lower === 'add') {
        const parsed = parseProviderAdd(`add ${rest.join(' ')}`)
        if (!parsed || !parsed.model) {
          ctx.transcript.sys(
            'usage: /providers add <name> --type <type> --model <model> [--key <api-key>] [--base-url <url>]'
          )
          return
        }
        ctx.gateway
          .rpc('provider_save', {
            model: parsed.model,
            name: parsed.name,
            ...(parsed.type ? { provider: parsed.type } : {}),
            ...(parsed.key ? { api_key: parsed.key } : {}),
            ...(parsed.baseUrl ? { base_url: parsed.baseUrl } : {})
          })
          .then(ctx.guarded(() => list()))
          .catch(ctx.guardedErr)
        return
      }

      ctx.transcript.sys('usage: /providers [list|use <name>|add <name> --type … --model …|remove <name>|types]')
    }
  },

  {
    help: 'create a git worktree for this session [create [name]]',
    name: 'worktree',
    run: (arg, ctx) => {
      const [sub = 'create', ...rest] = arg.trim().split(/\s+/).filter(Boolean)

      if (sub.toLowerCase() !== 'create') {
        ctx.transcript.sys('usage: /worktree create [name]')
        return
      }

      const name = rest.join(' ').trim()
      ctx.gateway
        .rpc<WorktreeResponse>('workspace.worktree', {
          action: 'create',
          session_id: ctx.sid,
          ...(name ? { name } : {})
        })
        .then(
          ctx.guarded(r => {
            if (r.ok === false) {
              ctx.transcript.sys(`worktree: ${r.error ?? 'unknown error'}`)
              return
            }
            ctx.transcript.sys(`worktree ready: ${r.path ?? r.worktree ?? '(path unavailable)'}`)
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'show the declarative-forge trace for this session',
    name: 'creator-trace',
    run: (_arg, ctx) => {
      ctx.gateway
        .rpc<CreatorTraceResponse>('creator_trace', { session_id: ctx.sid })
        .then(
          ctx.guarded(r => {
            const rows = r.trace ?? []
            if (!rows.length) {
              ctx.transcript.sys(r.ok === false ? `creator trace: ${r.error ?? 'unavailable'}` : 'no forge activity this session')
              return
            }
            ctx.transcript.page(
              rows
                .map(row =>
                  [
                    `${String(row.at ?? '').padEnd(13)} ${row.action ?? '?'} ${row.name ?? ''}${row.version ? `@${row.version}` : ''}`,
                    `  status: ${row.status ?? '?'}${row.detail ? ` — ${row.detail}` : ''}`
                  ].join('\n')
                )
                .join('\n'),
              'Creator trace'
            )
          })
        )
        .catch(ctx.guardedErr)
    }
  }
]
