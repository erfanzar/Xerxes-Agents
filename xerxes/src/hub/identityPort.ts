// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubActor } from './protocol.js'
import type { HubGrant, HubPrincipal } from '../security/principal.js'

/**
 * Seam between the Hub HTTP/WebSocket server and the identity subsystem
 * (pairing, devices, sessions, grants, audit). The server only ever learns who
 * a caller is through `authenticate`; it never parses tokens itself.
 */

export interface HubHttpRequestInfo {
  readonly method: string
  readonly url: string
  readonly cookie?: string
  readonly origin?: string
  readonly host?: string
  /** Header value of `x-xerxes-csrf` on mutating HTTP requests. */
  readonly csrfToken?: string
  readonly remoteAddress?: string
}

export interface AuthenticatedContext {
  readonly principal: HubPrincipal
  readonly grants: readonly HubGrant[]
  /** Browser session id; revocation closes sockets opened under it. */
  readonly browserSessionId: string
  readonly actor: HubActor
}

export type HubAuditDecision = 'allowed' | 'denied'
export type HubAuditOutcome = 'accepted' | 'rejected' | 'unknown' | 'failed' | 'n/a'

export interface HubAuditInput {
  readonly actor: HubActor
  readonly action: string
  readonly workspaceId?: string
  readonly sessionId?: string
  readonly interactionId?: string
  readonly submissionId?: string
  /** sha256 of the canonical request; never raw arguments, tokens or codes. */
  readonly requestHash: string
  readonly decision: HubAuditDecision
  readonly outcome: HubAuditOutcome
}

export type RevocationListener = (event: { readonly principalId: string; readonly deviceId?: string; readonly browserSessionId?: string; readonly reason: string }) => void

export interface HubIdentityPort {
  /** Resolve a request to an authenticated context, or null when unauthenticated/expired/revoked. */
  authenticate(request: HubHttpRequestInfo): Promise<AuthenticatedContext | null>
  /** Current grants for a principal (grants can change while a socket is open). */
  grantsFor(principalId: string): readonly HubGrant[]
  /** Fires when a device, user or browser session is revoked. */
  onRevoked(listener: RevocationListener): () => void
  /** Append to the audit log. Rejects when the write fails; callers fail closed on sensitive mutations. */
  audit(record: HubAuditInput): Promise<void>
  /**
   * Handle identity HTTP routes (pairing, refresh, logout, devices, invites,
   * passkeys). Returns undefined when the path is not an identity route.
   */
  handleHttp(request: Request, info: HubHttpRequestInfo): Promise<Response | undefined>
}
