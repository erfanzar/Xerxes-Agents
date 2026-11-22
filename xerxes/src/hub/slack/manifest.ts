// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

// Pure (no Node imports): shared by the Hub and the Desktop renderer.

/** The exact user scopes XERXES needs. */
export const USER_SCOPES = ['users:read', 'users:read.email', 'im:write', 'im:read', 'im:history', 'chat:write'] as const

/**
 * The Slack app manifest for the owner's own XERXES app: user scopes only (XERXES acts as you),
 * no bot, no Socket Mode, no token rotation, and this Hub's callback as the only redirect URL.
 */
export function slackAppManifest(redirectUrl: string): string {
  return JSON.stringify({
    display_information: { name: 'XERXES', description: 'Lets XERXES read and reply to direct messages you approve, as you.' },
    oauth_config: { redirect_urls: [redirectUrl], scopes: { user: [...USER_SCOPES] } },
    settings: { org_deploy_enabled: false, socket_mode_enabled: false, token_rotation_enabled: false },
  })
}

/** A Slack app configuration access token (api.slack.com/apps → Your App Configuration Tokens). */
export function isSlackConfigToken(value: unknown): value is string {
  return typeof value === 'string' && /^xoxe\.xoxp-[A-Za-z0-9._-]{1,1019}$/u.test(value)
}
