// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Shared vocabulary for the local/cloud data boundary.
 *
 * Every slice of the local-agent work (boundary, local engine, advisor, evals)
 * codes against these types so they can be built in parallel. Behavior lives in
 * `dataBoundary.ts`; this file holds types and the typed error only.
 */

/** `off` keeps today's behavior exactly; the other modes enforce a boundary. */
export type DataBoundaryMode = 'off' | 'private' | 'hybrid' | 'cloud'

/** Where a provider profile may send data: `local` never leaves the machine. */
export type DataTrust = 'local' | 'cloud'

/** When an outbound cloud request needs an explicit user approval. */
export type OutboundApproval = 'off' | 'first' | 'every'

/** Which kind of model call produced an outbound request. */
export type OutboundPurpose = 'turn' | 'subagent' | 'advisor' | 'title' | 'compaction' | 'helper'

/** What a model call is about to send, as seen by the boundary gate. */
export interface OutboundRequestSummary {
  readonly provider: string
  readonly model: string
  readonly trust: DataTrust
  readonly purpose: OutboundPurpose
  readonly sessionId?: string
  readonly agentId?: string
  /** Full serialized text of the outbound messages (system + history + tools). */
  readonly text: string
  readonly bytes: number
}

export type OutboundVerdict =
  | { readonly allow: true; readonly decision: 'sent' | 'approved' }
  | { readonly allow: false; readonly reason: string }

/** Decides whether one outbound model request may leave the machine. */
export interface OutboundGate {
  check(request: OutboundRequestSummary, signal?: AbortSignal): Promise<OutboundVerdict>
}

/** One line of the local disclosure log: what went to which provider. */
export interface DisclosureRecord {
  readonly at: string
  readonly sessionId?: string
  readonly agentId?: string
  readonly provider: string
  readonly model: string
  readonly trust: DataTrust
  readonly purpose: OutboundPurpose
  readonly bytes: number
  /** sha256 hex of the serialized outbound text; the text itself is never logged. */
  readonly digest: string
  readonly decision: 'sent' | 'approved' | 'denied'
  readonly reason?: string
}

export type DataBoundaryErrorCode =
  | 'DATA_BOUNDARY_DENIED'
  | 'DATA_BOUNDARY_NO_LOCAL_ROUTE'
  | 'DATA_BOUNDARY_TIER_LOCKED'
  | 'DATA_BOUNDARY_READ_ONLY'

/** A boundary refusal. Always observable: never converted into a silent success. */
export class DataBoundaryError extends Error {
  readonly code: DataBoundaryErrorCode

  constructor(code: DataBoundaryErrorCode, message: string) {
    super(message)
    this.name = 'DataBoundaryError'
    this.code = code
  }
}
