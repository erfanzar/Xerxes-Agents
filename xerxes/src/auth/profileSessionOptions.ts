// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { ProviderProfile } from '../bridge/profiles.js'
import type { LlmClientFactoryOptions } from '../llms/client.js'
import { ProviderAccountRegistry } from './providerAccountRegistry.js'

/** Account identity belongs to the client factory, never to provider wire overrides. */
export function profileSessionOptions(
  profile: Pick<ProviderProfile, 'oauth_account_id' | 'isolated_auth'> | undefined,
  accounts: Pick<ProviderAccountRegistry, 'sessionOptions'> = new ProviderAccountRegistry(),
): LlmClientFactoryOptions {
  if (profile?.oauth_account_id) return accounts.sessionOptions(profile.oauth_account_id)
  return profile?.isolated_auth ? { isolatedAuth: true } : {}
}
