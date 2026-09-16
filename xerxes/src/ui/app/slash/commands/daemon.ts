// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { patchOverlayState } from '../../overlayStore.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

export function formatDaemonStatus(result: Record<string, unknown>): string {
  const lines = ['Global daemon',
    `Process: ${result.pid ?? 'unknown'} · protocol ${result.daemon_protocol ?? 'unknown'}`,
    `Build: ${result.daemon_build_id ?? 'unknown'}`,
    `Provider: ${result.runtime_ready ? 'ready' : 'configuration required'}`]
  const hidden = new Set(['ok', 'pid', 'daemon_protocol', 'daemon_build_id', 'runtime_ready'])
  const append = (key: string, value: unknown, depth = 0) => {
    const label = key.replaceAll('_', ' ')
    if (Array.isArray(value)) {
      lines.push(`${'  '.repeat(depth)}${label}: ${value.length ? '' : 'none'}`)
      value.forEach((item, index) => append(String(index + 1), item, depth + 1))
    } else if (value && typeof value === 'object') {
      lines.push(`${'  '.repeat(depth)}${label}`)
      for (const [child, field] of Object.entries(value)) append(child, field, depth + 1)
    } else lines.push(`${'  '.repeat(depth)}${label}: ${value === true ? 'yes' : value === false ? 'no' : value ?? 'not set'}`)
  }
  for (const [key, value] of Object.entries(result)) if (!hidden.has(key)) append(key, value)
  return lines.join('\n')
}

function lifecycle(action: 'restart' | 'stop', ctx: SlashRunCtx) {
  patchOverlayState({ confirm: {
    title: action === 'restart' ? 'Restart the shared daemon when idle?' : 'Stop the shared daemon?',
    detail: action === 'restart'
      ? 'This affects every attached workspace. The daemon refuses while any workspace has active work. After shutdown, reconnect or run xerxes to launch the updated runtime.'
      : 'This stops the runtime for every workspace and interrupts active work. Quitting this TUI alone leaves that work running. Saved sessions remain on disk.',
    danger: true, confirmLabel: action === 'restart' ? 'Restart if idle' : 'Stop all workspaces', cancelLabel: 'Cancel',
    onConfirm: () => {
      if (ctx.stale()) return
      void ctx.gateway.rpc(action === 'restart' ? 'runtime.restart_if_idle' : 'shutdown', {}).then(ctx.guarded(result => {
        if (result?.busy) { ctx.transcript.sys('Daemon is busy in one or more workspaces. Nothing was stopped. Retry when all work is idle.'); return }
        if (result?.ok !== true) throw new Error(String(result?.error ?? 'Daemon rejected the request'))
        ctx.transcript.sys('Daemon accepted shutdown. Run xerxes to reconnect when it exits; saved sessions are retained.')
        ctx.session.die()
      })).catch(ctx.guardedErr)
    },
  } })
}

export const daemonCommands: SlashCommand[] = [
  { name: 'daemon', group: 'info', help: 'inspect or control the shared daemon [status|restart|stop]', run: (argument, ctx) => {
    const action = argument.trim() || 'status'
    if (action === 'restart' || action === 'stop') { lifecycle(action, ctx); return }
    if (action !== 'status') { ctx.transcript.sys('Usage: /daemon [status|restart|stop]'); return }
    void ctx.gateway.rpc('runtime.status', {}).then(ctx.guarded(result => {
      if (result?.ok !== true) throw new Error(String(result?.error ?? 'Cannot inspect daemon'))
      ctx.transcript.page(formatDaemonStatus(result), 'Daemon')
    })).catch(ctx.guardedErr)
  } },
  { name: 'restart', group: 'info', help: 'restart the shared daemon only when every workspace is idle', run: (_argument, ctx) => lifecycle('restart', ctx) },
]
