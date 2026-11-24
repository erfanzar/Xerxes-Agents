// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { AcpSessionBinding } from '../../../gatewayTypes.js'
import type { CompletionItem } from '../../interfaces.js'
import { runNativeSlash } from '../nativeSlash.js'
import type { SlashCommand } from '../types.js'

export interface AcpSubcommandSpec {
  readonly arg: string
  readonly meta: string
}

export const ACP_SUBCOMMANDS: readonly AcpSubcommandSpec[] = [
  { arg: 'status', meta: 'show active ACP runtime binding and available agents' },
  { arg: 'list', meta: 'list discovered external ACP agents' },
  { arg: 'use', meta: 'connect an external ACP runtime (use <agent> [cmd ...])' },
  { arg: 'connect', meta: 'connect an external ACP runtime (connect <agent> [cmd ...])' },
  { arg: 'disconnect', meta: 'disconnect external ACP runtime and return to native Xerxes' },
  { arg: 'model', meta: 'inspect or switch the active ACP runtime model' },
  { arg: 'mode', meta: 'inspect or switch the active ACP runtime mode' },
  { arg: 'config', meta: 'inspect or update an ACP runtime config option' },
  { arg: 'catalog', meta: 'inspect ACP commands, skills, capabilities, and config options' },
  { arg: 'skills', meta: 'inspect ACP runtime skills and commands' },
  { arg: 'capabilities', meta: 'inspect ACP runtime prompt, session, MCP, and auth capabilities' },
  { arg: 'sessions', meta: 'list sessions reported by the connected ACP runtime' },
  { arg: 'fork', meta: 'fork the current ACP session' },
  { arg: 'auth', meta: 'authenticate using an ACP runtime auth method' }
] as const

const ACP_AGENT_SUGGESTIONS: readonly { id: string; meta: string }[] = [
  { id: 'antigravity', meta: 'Google Antigravity CLI (agy acp)' },
  { id: 'claude-code', meta: 'Claude Code external runtime' },
  { id: 'codex', meta: 'Codex external runtime' },
  { id: 'xerxes', meta: 'Native Xerxes runtime' }
]

/** Local `/acp <subcommand>` and second-token completions matching daemon `handleAcpSlashCommand`. */
export function acpSubcommandCompletions(
  input: string,
  binding?: AcpSessionBinding | null
): CompletionItem[] {
  const firstMatch = input.match(/^\/acp\s+(\S*)$/i)
  if (firstMatch) {
    const prefix = (firstMatch[1] ?? '').toLowerCase()
    return ACP_SUBCOMMANDS.filter(sub => !prefix || sub.arg.startsWith(prefix)).map(sub => ({
      display: `acp ${sub.arg}`,
      group: 'config',
      meta: sub.meta,
      text: `/acp ${sub.arg}`
    }))
  }

  const secondMatch = input.match(/^\/acp\s+(use|connect|model|mode|config|auth)\s+(\S*)$/i)
  if (!secondMatch) {
    return []
  }

  const sub = (secondMatch[1] ?? '').toLowerCase()
  const prefix = (secondMatch[2] ?? '').toLowerCase()

  if (sub === 'use' || sub === 'connect') {
    return ACP_AGENT_SUGGESTIONS.filter(agent => !prefix || agent.id.startsWith(prefix)).map(agent => ({
      display: `acp ${sub} ${agent.id}`,
      group: 'config',
      meta: agent.meta,
      text: `/acp ${sub} ${agent.id}`
    }))
  }

  if (sub === 'model' && binding?.models?.length) {
    return binding.models
      .filter(model => !prefix || model.id.toLowerCase().startsWith(prefix) || model.name.toLowerCase().startsWith(prefix))
      .map(model => ({
        display: `acp model ${model.id}`,
        group: 'config',
        meta: model.description || model.name,
        text: `/acp model ${model.id}`
      }))
  }

  if (sub === 'mode' && binding?.modes?.length) {
    return binding.modes
      .filter(mode => !prefix || mode.id.toLowerCase().startsWith(prefix) || mode.name.toLowerCase().startsWith(prefix))
      .map(mode => ({
        display: `acp mode ${mode.id}`,
        group: 'config',
        meta: mode.description || mode.name,
        text: `/acp mode ${mode.id}`
      }))
  }

  if (sub === 'config' && binding?.config_options?.length) {
    return binding.config_options
      .filter(opt => !prefix || opt.id.toLowerCase().startsWith(prefix) || opt.name.toLowerCase().startsWith(prefix))
      .map(opt => ({
        display: `acp config ${opt.id}`,
        group: 'config',
        meta: `${opt.name} (${String(opt.current_value)})`,
        text: `/acp config ${opt.id}`
      }))
  }

  if (sub === 'auth' && binding?.capabilities?.auth_methods?.length) {
    return binding.capabilities.auth_methods
      .filter(method => !prefix || method.id.toLowerCase().startsWith(prefix) || method.name.toLowerCase().startsWith(prefix))
      .map(method => ({
        display: `acp auth ${method.id}`,
        group: 'config',
        meta: method.description || method.name,
        text: `/acp auth ${method.id}`
      }))
  }

  return []
}

export const acpCommands: SlashCommand[] = [
  {
    name: 'acp',
    group: 'config',
    help: 'manage external ACP runtimes [status|list|use|connect|disconnect|model|mode|config|catalog|skills|capabilities|sessions|fork|auth]',
    usage:
      '/acp [status|list|use <agent>|connect <agent>|disconnect|model [id]|mode [id]|config [id value]|catalog|skills|capabilities|sessions|fork|auth [method]]',
    run: (arg, ctx) => {
      const trimmed = arg.trim()
      const command = trimmed ? `acp ${trimmed}` : 'acp status'
      runNativeSlash(ctx, command, 'ACP')
    }
  }
]
