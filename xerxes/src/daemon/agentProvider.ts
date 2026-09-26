// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { resolvedProfileContextLimit, resolvedProfileMaxOutputTokens, type ProfileStore } from '../bridge/profiles.js'
import { createLlmClient } from '../llms/client.js'
import { createHash } from 'node:crypto'
import { getProviderConfig, resolveProvider } from '../llms/providerRegistry.js'
import { wireCapability } from '../llms/modelsDev.js'
import { codexBaseUrl } from '../auth/codexAuth.js'
import type { NativeSubagentHostOptions } from './subagentHost.js'
import { unavailableAgentProfile } from '../runtime/modelInventory.js'

type AgentProfiles = Pick<ProfileStore, 'get'> & Partial<Pick<ProfileStore, 'list'>>

/** Select a child transport without changing the active parent profile. */
export function agentProviderResolver(
  profiles: AgentProfiles,
  createClient: typeof createLlmClient = createLlmClient,
): NonNullable<NativeSubagentHostOptions['resolveProviderProfile']> {
  return (name, model, expectedRoute) => {
    const profile = profiles.get(name)
    if (!profile) throw unavailableAgentProfile(name, model, profiles.list?.() ?? [])
    const route = providerRouteIdentity(model, { provider: profile.provider, baseUrl: profile.base_url })
    if (expectedRoute !== undefined && route !== expectedRoute) {
      throw new Error('Agent provider route changed; restore the original provider configuration or dispatch new work')
    }
    return {
      llm: createClient(model, { api_key: profile.api_key, base_url: profile.base_url, provider: profile.provider }),
      contextLimit: candidate => resolvedProfileContextLimit(profile, candidate),
      maxOutputTokens: candidate => resolvedProfileMaxOutputTokens(profile, candidate),
    }
  }
}

/** Routing identity only: credentials remain in the current credential store. */
export function providerRouteIdentity(
  model: string,
  connection: { readonly provider?: string; readonly baseUrl?: string; readonly responsesApi?: boolean },
): string {
  const provider = resolveProvider(model, { provider: connection.provider, base_url: connection.baseUrl })
  const config = getProviderConfig(provider)
  // A gateway model may name its own endpoint in its catalog entry (models.dev).
  const wire = wireCapability({ provider, model })
  const baseUrl = connection.baseUrl || (provider === 'openai-codex' ? codexBaseUrl() : wire.apiBaseUrl ?? config.baseUrl) || ''
  // These noncredential selectors can change a deployment even when the profile
  // itself is unchanged. OAuth account rotation is deliberately not fingerprinted.
  const environmentKeys: readonly string[] = provider === 'azure'
    ? ['AZURE_OPENAI_BASE_URL', 'AZURE_OPENAI_RESOURCE_NAME', 'AZURE_OPENAI_DEPLOYMENT_NAME_MAP', 'AZURE_OPENAI_API_VERSION']
    : provider === 'google-vertex' ? ['GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT', 'GOOGLE_CLOUD_LOCATION']
    : provider === 'amazon-bedrock' ? ['AWS_REGION', 'AWS_DEFAULT_REGION', 'AWS_PROFILE']
    : provider === 'cloudflare-ai-gateway' || provider === 'cloudflare-workers-ai'
      ? ['CLOUDFLARE_ACCOUNT_ID', 'CLOUDFLARE_GATEWAY_ID'] : []
  return createHash('sha256').update(JSON.stringify({
    version: 1, provider, baseUrl, transport: wire.api ?? config.transport,
    responsesApi: connection.responsesApi === true,
    environment: environmentKeys.map(key => [key, process.env[key] ?? '']),
  })).digest('hex')
}

export function agentProviderRouteResolver(profiles: AgentProfiles): (name: string, model: string) => string {
  return (name, model) => {
    const profile = profiles.get(name)
    if (!profile) throw unavailableAgentProfile(name, model, profiles.list?.() ?? [])
    return providerRouteIdentity(model, { provider: profile.provider, baseUrl: profile.base_url })
  }
}
