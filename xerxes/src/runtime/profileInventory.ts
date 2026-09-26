// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { CodexSession, fetchCodexModelCatalog, type CodexModel } from '../auth/codexAuth.js'
import { CopilotSession, fetchCopilotModels } from '../auth/copilotAuth.js'
import { profileQuota } from '../auth/profileUsage.js'
import type { AgentSettingsStore } from '../agents/settingsStore.js'
import { reportedModelReasoning, type ProfileStore, type ProviderProfile } from '../bridge/profiles.js'
import { claudeCodeCatalog, claudeCodeReasoningLevels } from '../llms/claudeCodeCatalog.js'
import type { RuntimeConnection } from '../daemon/runtimeConnection.js'
import { discoverModelCatalog, profileDiscoveryApiKey, sanitizeModelDiscoveryError } from '../daemon/modelDiscovery.js'
import { catalogReasoningLevels, fallbackReasoningLevels, liveReasoningLevels, providerReasoningLevels, selectableEfforts } from '../llms/reasoningLevels.js'
import { modelsDev, reportModelCapability, type LiveReasoning } from '../llms/modelsDev.js'
import { getApiKey, getProviderConfig, isProviderName, resolveProvider } from '../llms/providerRegistry.js'
import { DEFAULT_RADIUS_GATEWAY, loadRadiusGatewayConfig, normalizeRadiusGatewayUrl } from '../llms/radiusGateway.js'
import type { ModelInventoryHost } from '../tools/modelInventoryTools.js'
import { modelInventory, unavailableAgentProfile, type ModelInventoryPort } from './modelInventory.js'
import { inventoryCapabilities } from './inventoryCapabilities.js'

export interface ProfileInventoryOptions {
  /** Host-owned subscription catalog adapter; defaults to the native OAuth session. */
  readonly copilotModels?: (signal?: AbortSignal) => Promise<readonly string[]>
}
type InventoryProfiles = Pick<ProfileStore, 'list' | 'active' | 'get'>

/** Validate against the captured parent connection, never the mutable active profile. */
export function inheritedSelectionValidator(connection: RuntimeConnection) {
  const provider = resolveProvider(connection.model, { provider: connection.provider, base_url: connection.baseUrl })
  const profile: ProviderProfile = {
    name: 'inherited', provider, model: connection.model,
    api_key: connection.apiKey || getApiKey(provider),
    base_url: connection.baseUrl || getProviderConfig(provider).baseUrl || '', sampling: {},
  }
  const validate = profileSelectionValidator({ list: () => [{ ...profile, active: true }], active: () => profile, get: name => name === profile.name ? profile : undefined })
  return async (model: string, effort?: string, signal?: AbortSignal): Promise<void> => {
    signal?.throwIfAborted()
    // Reusing the established parent selection needs no catalog request.
    if (model === profile.model && effort === undefined) return
    await validate(profile.name, model, effort, signal)
  }
}

async function copilotCatalog(signal?: AbortSignal): Promise<readonly string[]> {
  const credential = await new CopilotSession().credential(signal)
  signal?.throwIfAborted()
  return fetchCopilotModels(credential, { ...(signal ? { signal } : {}) })
}

/** Catalog ownership for native hosts which do not run a daemon server. */
export function profileInventoryHost(profiles: ProfileStore, settings: AgentSettingsStore, options: ProfileInventoryOptions = {}): ModelInventoryHost {
  return async (sessionId, params, signal) => {
    if (!sessionId.trim()) throw new Error('Inventory requires a session')
    return withProfileInventory(profiles, settings, signal, port => modelInventory(port, params, signal), typeof params.provider_profile === 'string' ? params.provider_profile.trim() || undefined : undefined, options)
  }
}

/** Apply the same discovery and reasoning metadata to standalone agent launches. */
export function profileSelectionValidator(profiles: InventoryProfiles, options: ProfileInventoryOptions = {}) {
  return async (name: string, model: string, effort?: string, signal?: AbortSignal): Promise<void> => {
    signal?.throwIfAborted()
    if (!name.trim() || name.length > 512 || !model.trim() || model.length > 512 || (effort !== undefined && (!effort.trim() || effort.length > 64))) throw new Error('Invalid agent provider/model/reasoning selection')
    await withProfileInventory(profiles, undefined, signal, async port => {
      const profile = port.profiles().find(value => value.name === name)
      if (!profile) throw unavailableAgentProfile(name, model, port.profiles())
      const catalog = await port.discover(name)
      signal?.throwIfAborted()
      if (model !== profile.model && !catalog.models.some(value => value.id === model)) throw new Error('Model is not configured or discovered for agent provider ' + name + ': ' + model)
      if (effort !== undefined) {
        // Refuse only against levels something reported; unreported levels pass through to the provider.
        const reported = await port.reasoning(name, model)
        if (reported.source !== 'provider_fallback' && !reported.efforts.includes(effort)) throw new Error('Unsupported reasoning effort for agent model ' + model + ': ' + effort)
      }
    }, name, options)
  }
}

async function withProfileInventory<T>(profiles: InventoryProfiles, settings: AgentSettingsStore | undefined, signal: AbortSignal | undefined, read: (port: ModelInventoryPort) => Promise<T>, profileName?: string, options: ProfileInventoryOptions = {}): Promise<T> {
    signal?.throwIfAborted()
    const snapshot = profiles.list(), active = profiles.active()?.name
    const selected = (name: string): ProviderProfile => {
      const profile = snapshot.find(value => value.name === name)
      if (!profile) throw new Error('Unknown provider profile')
      return profile
    }
    let codexModels: readonly CodexModel[] = []
    // What each provider's /models said about reasoning, per profile and model.
    const reportedReasoning = new Map<string, LiveReasoning>()
    const result = await read({
      profiles: () => snapshot.map(profile => ({ name: profile.name, provider: profile.provider, model: profile.model, active: profile.name === active })),
      routingNotes: () => settings?.routingNotes() ?? [],
      quota: (name, signal) => profileQuota(selected(name), { ...(signal ? { signal } : {}) }),
      discover: async name => {
        const profile = selected(name), apiKey = profileDiscoveryApiKey(profile)
        const discover = async (): ReturnType<ModelInventoryPort['discover']> => {
        try {
          if (profile.provider === 'claude-code') return { source: 'provider', models: (await claudeCodeCatalog.load()).map(entry => ({ id: `claude-code/${entry.value}`, ...(entry.contextLimit ? { context_limit: entry.contextLimit, context_source: 'provider' } : {}) })) }
          if (profile.provider === 'github-copilot') {
            const models = await (options.copilotModels ?? copilotCatalog)(signal)
            signal?.throwIfAborted()
            return { source: 'provider', models: models.map(id => ({ id })) }
          }
          if (profile.provider === 'radius') {
            const catalog = await loadRadiusGatewayConfig(normalizeRadiusGatewayUrl(profile.base_url.trim() || DEFAULT_RADIUS_GATEWAY), apiKey || undefined, signal)
            return { source: 'provider', models: catalog.models.map(model => ({ id: model.id,
              ...(Number.isSafeInteger(model.contextWindow) && model.contextWindow > 0 ? { context_limit: model.contextWindow, context_source: 'provider' } : {}),
              ...(Number.isSafeInteger(model.maxTokens) && model.maxTokens > 0 ? { max_output_tokens: model.maxTokens, output_source: 'provider' } : {}),
            })) }
          }
          if (profile.provider === 'openai-codex') {
            const credential = await new CodexSession().credential(signal)
            signal?.throwIfAborted()
            codexModels = await fetchCodexModelCatalog(credential, { baseUrl: profile.base_url, ...(signal ? { signal } : {}) })
            return { source: 'provider', models: codexModels.map(model => ({ id: model.id, ...(model.contextLimit === undefined ? {} : { context_limit: model.contextLimit, context_source: 'provider' }) })) }
          }
          const catalog = await discoverModelCatalog({ provider: profile.provider, baseUrl: profile.base_url, apiKey, allowPrivateEndpoint: true, ...(signal ? { signal } : {}) })
          for (const model of catalog) {
            if (model.reasoning) reportedReasoning.set(JSON.stringify([profile.name, model.id]), model.reasoning)
            reportModelCapability(profile.provider, model.id, { ...(model.reasoning ? { reasoning: model.reasoning } : {}), ...(model.dynamicTools === undefined ? {} : { dynamicTools: model.dynamicTools }) })
          }
          return { source: 'provider', models: catalog.map(model => ({ id: model.id, ...(model.contextLimit === undefined ? {} : { context_limit: model.contextLimit, context_source: 'provider' }), ...(model.maxOutputTokens === undefined ? {} : { max_output_tokens: model.maxOutputTokens, output_source: 'provider' }) })) }
        } catch (error) {
          signal?.throwIfAborted()
          // OAuth tokens are not profile API keys; raw adapter errors cannot be
          // safely redacted using this profile's credentials.
          if (profile.provider === 'github-copilot') throw new Error('GitHub Copilot model discovery failed; check the Copilot login and connection, then retry')
          throw new Error(sanitizeModelDiscoveryError(error, { apiKey, baseUrl: profile.base_url }))
        }
        }
        const catalog = await discover()
        return { ...catalog, models: catalog.models.map(model => inventoryCapabilities(profile, model)) }
      },
      reasoning: async (name, model) => {
        const profile = selected(name), provider = isProviderName(profile.provider) ? profile.provider : undefined
        const live = codexModels.find(value => value.id === model)
        // Reported by the provider (Codex, Claude Code, its /models entry),
        // else models.dev, else nothing — the same order as the pickers.
        if (profile.provider === 'claude-code') await claudeCodeCatalog.load().catch(() => undefined)
        await modelsDev.load()
        const levels = live?.reasoningLevels.length ? providerReasoningLevels(live.reasoningLevels.map(level => ({ effort: level.effort, ...(level.description === undefined ? {} : { description: level.description }) })), live.defaultReasoningLevel)
          : profile.provider === 'claude-code' ? claudeCodeReasoningLevels(claudeCodeCatalog.find(model))
          : liveReasoningLevels(reportedReasoning.get(JSON.stringify([profile.name, model])) ?? reportedModelReasoning(profile, model), 'provider') ?? catalogReasoningLevels(model, provider, profile.base_url) ?? fallbackReasoningLevels(provider)
        return { efforts: selectableEfforts(levels), source: levels.provenance ?? 'unknown', shape: levels.shape, ...(levels.defaultEffort === undefined ? {} : { defaultEffort: levels.defaultEffort }) }
      },
    })
    signal?.throwIfAborted()
    // Reads may await network work; never attribute an old credential's result to a replaced profile.
    const identity = (value: ProviderProfile | undefined) => value ? JSON.stringify([value.name, value.provider, value.api_key, value.base_url, value.model, value.sampling, value.model_overrides]) : undefined
    if (profileName !== undefined && identity(profiles.get(profileName)) !== identity(selected(profileName))) throw new Error('Provider profile changed during discovery; retry')
    return result
}
