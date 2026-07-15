// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  type GoogleAuthorizationRequest,
  type GoogleAuthorizationStatus,
  type GoogleWorkspaceOAuthClient,
  type GoogleWorkspaceToken,
} from './oauth.js'

/** User-facing, secret-free steps for integrating the native Google Workspace skill. */
export interface GoogleWorkspaceSetupGuidance {
  readonly credentials: string
  readonly security: string
  readonly steps: readonly string[]
}

/** Explain setup without reading files, installing packages, or emitting stored credential values. */
export function googleWorkspaceSetupGuidance(): GoogleWorkspaceSetupGuidance {
  return {
    credentials: 'Create a Google OAuth desktop or web client in Google Cloud and pass its values to GoogleWorkspaceOAuthClient explicitly.',
    security: 'Provide caller-owned encrypted authorization storage and a deliberate browser adapter; Xerxes does not discover token files, environment variables, or system credentials.',
    steps: [
      'Inject GoogleWorkspaceAuthorizationStorage, a fetch implementation, and GoogleWorkspaceOAuthConfig.',
      'Call beginAuthorization() to obtain a PKCE-protected browser URL, optionally using an injected browser adapter.',
      'After consent, pass the returned redirect URL or code to completeAuthorization().',
      'Inject the OAuth client into GoogleWorkspaceClient as its token provider, then dispatch structured bridge commands.',
      'Call revoke() when authorization should be removed; it revokes remotely before clearing caller-owned storage.',
    ],
  }
}

/** Small setup façade suitable for a TUI, HTTP route, or structured bridge command. */
export class GoogleWorkspaceSetup {
  private readonly oauth: GoogleWorkspaceOAuthClient

  constructor(oauth: GoogleWorkspaceOAuthClient) {
    this.oauth = oauth
  }

  guidance(): GoogleWorkspaceSetupGuidance {
    return googleWorkspaceSetupGuidance()
  }

  status(): Promise<GoogleAuthorizationStatus> {
    return this.oauth.authorizationStatus()
  }

  begin(options: { readonly openBrowser?: boolean } = {}): Promise<GoogleAuthorizationRequest> {
    return this.oauth.beginAuthorization(options)
  }

  complete(callbackOrCode: string, signal?: AbortSignal): Promise<GoogleWorkspaceToken> {
    return this.oauth.completeAuthorization(callbackOrCode, signal)
  }

  revoke(signal?: AbortSignal): Promise<boolean> {
    return this.oauth.revoke(signal)
  }
}
