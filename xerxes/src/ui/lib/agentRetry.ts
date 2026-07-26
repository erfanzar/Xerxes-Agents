// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { GatewayRpc } from '../app/interfaces.js'
import type { SubagentStatus } from '../types.js'

/** Wire view of the retried agent returned by the daemon `subagent.retry` RPC. */
export interface SubagentRetryAgent {
  readonly closed?: boolean
  readonly error?: string
  readonly history_session_id?: string
  readonly id?: string
  readonly name?: string
  readonly prompt_profile?: string
  readonly status?: string
  readonly title?: string
  readonly updated_at?: string
}

export interface SubagentRetryResponse {
  readonly agent?: SubagentRetryAgent
  readonly error?: string
  readonly ok?: boolean
}

/**
 * Agents still doing work can never be retry targets. Every terminal state —
 * including completed, since provider connection failures land there with an
 * error summary — may be resumed or sent again under the same identity.
 */
const ACTIVE_STATUSES: ReadonlySet<SubagentStatus> = new Set(['queued', 'running'])

export const subagentRetryable = (status: SubagentStatus): boolean => !ACTIVE_STATUSES.has(status)

/** Dead-agent states where retry is the primary recovery action. */
const FAILED_STATUSES: ReadonlySet<SubagentStatus> = new Set(['error', 'failed', 'interrupted', 'timeout'])

export const subagentFailed = (status: SubagentStatus): boolean => FAILED_STATUSES.has(status)

/**
 * Ask the daemon to start a new attempt for a dead subagent under its stable
 * identity (same task id/name, continued conversation when one persisted).
 * The daemon enforces terminal-only retry and idempotency; this helper only
 * normalizes the wire response and never fabricates success.
 */
export async function retrySubagent(
  rpc: GatewayRpc,
  task: string,
  message?: string
): Promise<SubagentRetryResponse> {
  const response = await rpc<SubagentRetryResponse>('subagent.retry', {
    task,
    ...(message?.trim() ? { message: message.trim() } : {})
  })

  return response ?? { ok: false, error: 'no response from the daemon' }
}
