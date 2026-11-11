// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isRecord } from '../../protocol.js'
import { isTestRunCommand, parseTestOutput, type WatchEvent } from '../watch.js'
import type { WakeKind } from './wakes.js'

type SubscriptionEventInput = WatchEvent & { readonly event_id?: string }

/** Only typed event facts can wake subscriptions; task prose is never interpreted as an instruction. */
export function subscriptionEvent(event: SubscriptionEventInput): { id: string; workspace: string; kind: WakeKind; digest: string; value?: number } | undefined {
  let kind: WakeKind | undefined
  let value: number | undefined
  if (event.type === 'turn_end' && event.payload.error === undefined && event.payload.failed !== true) kind = 'task_idle'
  else if (event.type === 'question_request') kind = 'question_waiting'
  else if (event.type === 'status_update' && event.payload.goal_phase === 'blocked') kind = 'goal_blocked'
  else if (event.type === 'test_result' && typeof event.payload.failed === 'number' && event.payload.failed > 0) kind = 'tests_failing'
  else if (event.type === 'machine_health' && typeof event.payload.disk_used_percent === 'number' && Number.isFinite(event.payload.disk_used_percent)) {
    kind = 'disk_threshold'; value = event.payload.disk_used_percent
  }
  const id = event.event_id ?? (event.sequence === undefined ? undefined : `sequence:${event.sequence}`)
  if (kind === undefined || id === undefined) return undefined
  return { id, workspace: event.workspace_id, kind,
    digest: `${kind} for task ${JSON.stringify((event.session_id ?? 'workspace').slice(0, 200))}`,
    ...(value === undefined ? {} : { value }) }
}

/** Correlates the daemon's real command/result events without retaining task content. */
export class SubscriptionEvents {
  private readonly tests = new Set<string>()
  observe(event: SubscriptionEventInput): ReturnType<typeof subscriptionEvent> {
    const key = (id: string) => JSON.stringify([event.workspace_id, event.session_id ?? '', id])
    const payload = event.payload
    if (event.type === 'tool_call' && typeof payload.name === 'string' && /^(exec_command|ExecuteShell|Bash|shell|run_command)$/.test(payload.name)) {
      let args: unknown = payload.arguments
      if (typeof args === 'string' && args.length <= 16000) {
        try { args = JSON.parse(args) } catch { args = undefined }
      }
      const id = typeof payload.tool_call_id === 'string' ? payload.tool_call_id : typeof payload.id === 'string' ? payload.id : undefined
      if (id !== undefined && isTestRunCommand(isRecord(args) ? args : undefined, payload.arguments)) {
        if (this.tests.size >= 1000) this.tests.delete(this.tests.values().next().value ?? '')
        this.tests.add(key(id))
      }
    }
    if (event.type === 'tool_result' && typeof payload.tool_call_id === 'string' && this.tests.delete(key(payload.tool_call_id))) {
      const result = typeof payload.result === 'string' ? payload.result : ''
      const parsed = parseTestOutput(result)
      if (parsed.outcome === 'fail') return subscriptionEvent({ ...event, type: 'test_result', payload: { failed: parsed.failed ?? 1 } })
    }
    return subscriptionEvent(event)
  }
}
