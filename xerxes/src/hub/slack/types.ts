// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** These are Slack IDs, not Xerxes workspace/principal IDs. */
export interface SlackIdentity {
  readonly teamId: string
  readonly userId: string
}

export interface SlackPeer extends SlackIdentity {
  readonly peerId: string
}

export interface SlackDirectMessage extends SlackPeer {
  readonly channelId: string
}

export interface SlackProfile {
  readonly userId: string
  readonly displayName: string
}

export interface SlackMessage {
  readonly ts: string
  readonly userId: string
  readonly text: string
}

export type SlackPeerOperation = 'open' | 'verify' | 'read' | 'send'
/** Host must consult current approval for this exact tuple. No callback means no DM access. */
export type SlackPeerApproval = (peer: SlackPeer, operation: SlackPeerOperation) => boolean | Promise<boolean>

/** Host-only. Keep OAuth setup and profile lookup off the model-facing port. */
export interface SlackTransport {
  authTest(identity: SlackIdentity, signal?: AbortSignal): Promise<SlackIdentity>
  lookupProfile(identity: SlackIdentity, exact: { readonly userId: string } | { readonly email: string }, signal?: AbortSignal): Promise<SlackProfile>
  openPeer(peer: SlackPeer, signal?: AbortSignal): Promise<SlackDirectMessage>
  verifyDm(dm: SlackDirectMessage, signal?: AbortSignal): Promise<SlackDirectMessage>
  readDm(dm: SlackDirectMessage, limit?: number, signal?: AbortSignal): Promise<readonly SlackMessage[]>
  sendDm(dm: SlackDirectMessage, text: string, signal?: AbortSignal): Promise<{ readonly channelId: string; readonly ts: string }>
}

export type SlackErrorCode = 'invalid_request' | 'invalid_state' | 'oauth_denied' | 'oauth_unsupported' | 'not_connected'
  | 'not_found' | 'not_approved' | 'identity_mismatch' | 'not_dm' | 'storage_error'
  | 'slack_rejected' | 'rate_limited' | 'network_error' | 'invalid_response' | 'aborted' | 'timeout'

const messages: Readonly<Record<SlackErrorCode, string>> = {
  invalid_request: 'Invalid Slack request.',
  invalid_state: 'Invalid or expired Slack authorization state.',
  oauth_denied: 'Slack authorization was not granted.',
  oauth_unsupported: 'Slack must grant the required workspace user scopes without token rotation.',
  not_connected: 'This Slack account is not connected.',
  not_found: 'No exact Slack profile matched.',
  not_approved: 'This Slack peer operation is not approved.',
  identity_mismatch: 'The Slack connection does not match the expected workspace and user.',
  not_dm: 'The Slack conversation is not the expected one-to-one DM.',
  storage_error: 'Slack credential storage is unavailable or unsafe.',
  slack_rejected: 'Slack rejected the request.',
  rate_limited: 'Slack rate limited the request.',
  network_error: 'Slack could not be reached.',
  invalid_response: 'Slack returned an invalid response.',
  aborted: 'Slack request was cancelled.',
  timeout: 'Slack request timed out.',
}

/** Never include raw Slack replies, credentials, request URLs, or underlying errors. */
export class SlackTransportError extends Error {
  constructor(readonly code: SlackErrorCode, readonly retryAfterSeconds?: number) {
    super(messages[code])
    this.name = 'SlackTransportError'
  }
}
