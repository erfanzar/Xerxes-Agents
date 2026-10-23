// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'

import { canonicalJson } from '../../security/hubAuthEnvelope.js'
import type { HubAuditDecision, HubAuditOutcome, HubIdentityPort } from '../identityPort.js'
import type { HubActor } from '../protocol.js'

/**
 * Audit trail of the XERXES supervisor. Every stage of an action (intent,
 * confirm or cancel, dispatch, outcome) goes through the Hub identity audit
 * log. Only a request hash is recorded, never raw tool arguments.
 */

export const SUPERVISOR_PRINCIPAL_ID = 'xerxes-supervisor'
export const SUPERVISOR_DISPLAY_NAME = 'XERXES'

export type SupervisorAuditStage = 'intent' | 'confirm' | 'cancel' | 'dispatch' | 'outcome' | 'denied' | 'expired' | 'notify'

/** The supervisor as an actor. `requested_by` and `confirmed_by` are attribution only, never authority. */
export function supervisorActor(requestedBy: string, confirmedBy?: string): HubActor {
  return {
    principal_kind: 'agent',
    principal_id: SUPERVISOR_PRINCIPAL_ID,
    display_name: SUPERVISOR_DISPLAY_NAME,
    requested_by: requestedBy,
    ...(confirmedBy === undefined ? {} : { confirmed_by: confirmedBy }),
  }
}

/** sha256 of the canonical request. This is the only trace of the arguments that reaches the audit log. */
export function supervisorRequestHash(tool: string, workspaceId: string | undefined, sessionId: string | undefined, args: unknown): string {
  return createHash('sha256')
    .update(canonicalJson({ tool, workspace_id: workspaceId ?? null, session_id: sessionId ?? null, args }))
    .digest('hex')
}

export interface SupervisorAuditEntry {
  readonly stage: SupervisorAuditStage
  readonly actor: HubActor
  readonly actionId: string
  readonly tool: string
  readonly workspaceId?: string
  readonly sessionId?: string
  readonly requestHash: string
  readonly decision: HubAuditDecision
  readonly outcome: HubAuditOutcome
}

export class SupervisorAudit {
  constructor(
    private readonly identity: Pick<HubIdentityPort, 'audit'>,
    private readonly onError?: (error: unknown, context: string) => void,
  ) {}

  /** Rejects when the audit write fails. Callers fail closed on stages that precede a mutation. */
  async record(entry: SupervisorAuditEntry): Promise<void> {
    await this.identity.audit({
      actor: entry.actor,
      action: `supervisor.${entry.stage}:${entry.tool}`,
      ...(entry.workspaceId === undefined ? {} : { workspaceId: entry.workspaceId }),
      ...(entry.sessionId === undefined ? {} : { sessionId: entry.sessionId }),
      submissionId: entry.actionId,
      requestHash: entry.requestHash,
      decision: entry.decision,
      outcome: entry.outcome,
    })
  }

  /** For stages after the fact (a denial is refused either way; an outcome is already established). */
  async recordBestEffort(entry: SupervisorAuditEntry): Promise<void> {
    try {
      await this.record(entry)
    } catch (error) {
      this.onError?.(error, `supervisor audit ${entry.stage}`)
    }
  }
}
