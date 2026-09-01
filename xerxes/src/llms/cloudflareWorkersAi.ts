// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Cloudflare Workers AI transport (pi-ai `cloudflare-workers-ai`).
 *
 * Workers AI speaks the OpenAI-completions wire dialect on an account-scoped
 * endpoint, so the transport is the shared OpenAI-compatible client with the
 * account id materialized into the base URL and the Cloudflare API key as the
 * bearer credential — mirroring pi-ai's `cloudflareStreams` placeholder
 * substitution plus `cloudflareWorkersAIAuth`.
 */

import { ConfigurationError } from '../core/errors.js'
import { OpenAiCompatibleClient } from './client.js'

/** pi-ai CLOUDFLARE_WORKERS_AI_BASE_URL before account substitution. */
export const CLOUDFLARE_WORKERS_AI_BASE_URL_TEMPLATE
  = 'https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1'

export interface CloudflareWorkersAiConfig {
  readonly accountId: string
  readonly apiKey: string
}

function ambient(name: string, env: Readonly<Record<string, string | undefined>>): string | undefined {
  const value = env[name]?.trim()
  return value ? value : undefined
}

/**
 * Resolve the Cloudflare credential pair: the API key and account id come
 * from CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID (pi-ai
 * cloudflareWorkersAIAuth). Both are required; a missing one is a
 * configuration error naming the variable, never a guessed endpoint.
 */
export function resolveCloudflareWorkersAiConfig(
  env: Readonly<Record<string, string | undefined>> = process.env,
  overrides: { readonly apiKey?: string } = {},
): CloudflareWorkersAiConfig {
  const apiKey = overrides.apiKey?.trim() || ambient('CLOUDFLARE_API_KEY', env)
  const accountId = ambient('CLOUDFLARE_ACCOUNT_ID', env)
  if (!apiKey && !accountId) {
    throw new ConfigurationError(
      'CLOUDFLARE_API_KEY',
      'Cloudflare Workers AI needs CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID',
    )
  }
  if (!apiKey) {
    throw new ConfigurationError('CLOUDFLARE_API_KEY', 'Cloudflare Workers AI needs CLOUDFLARE_API_KEY')
  }
  if (!accountId) {
    throw new ConfigurationError('CLOUDFLARE_ACCOUNT_ID', 'Cloudflare Workers AI needs CLOUDFLARE_ACCOUNT_ID')
  }
  return { accountId, apiKey }
}

export function cloudflareWorkersAiBaseUrl(accountId: string): string {
  return CLOUDFLARE_WORKERS_AI_BASE_URL_TEMPLATE.replaceAll('{CLOUDFLARE_ACCOUNT_ID}', accountId)
}

/** Build the Workers AI client: OpenAI-compatible wire, account-scoped base URL. */
export function createCloudflareWorkersAiClient(options: {
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly fetchImplementation?: import('./client.js').FetchImplementation
  readonly overrides?: { readonly apiKey?: string }
} = {}): OpenAiCompatibleClient {
  const config = resolveCloudflareWorkersAiConfig(options.env, options.overrides)
  return new OpenAiCompatibleClient({
    apiKey: config.apiKey,
    baseUrl: cloudflareWorkersAiBaseUrl(config.accountId),
    ...(options.fetchImplementation ? { fetchImplementation: options.fetchImplementation } : {}),
    providerName: 'cloudflare-workers-ai',
  })
}
