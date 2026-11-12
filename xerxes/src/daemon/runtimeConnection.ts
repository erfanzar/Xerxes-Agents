// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { claudeExecutable } from '../auth/claudeCodeLogin.js'
import type { ProviderProfile } from '../bridge/profiles.js'
import { nativeSamplingDefaults } from '../llms/samplingDefaults.js'
import { isOpenAiOutputTokenProvider } from '../llms/outputTokens.js'
import { resolveProvider } from '../llms/providerRegistry.js'
import { DEFAULT_PERMISSION_MODE, type PermissionMode } from '../streaming/permissions.js'
import type { DaemonConfig } from './config.js'

export interface RuntimeConnection {
  readonly apiKey?: string
  readonly baseUrl?: string
  readonly maxTokens?: number
  readonly model: string
  readonly permissionMode: PermissionMode
  readonly provider?: string
  /** Isolated OAuth account backing this profile. */
  readonly oauthAccountId?: string
  /**
   * Session default effort hint for per-turn thinking resolution. This and
   * the two thinking fields below were historically dropped from this
   * interface because the connection was treated as display-only config for
   * status surfaces; turns now consume them as session defaults, so they
   * must survive resolution.
   */
  readonly reasoningEffort?: string
  readonly responsesApi?: boolean
  /** Requested OpenAI service tier (`auto`/`default`/`flex`/`priority`); Responses-family only. */
  readonly serviceTier?: string
  readonly temperature?: number
  /**
   * Session default extended-thinking toggle. Carried through (not dropped
   * as display-only) so each turn can weigh it against ultra mode and
   * escalation keywords in the prompt.
   */
  readonly thinking?: boolean
  /** Session default thinking token budget, carried through for the same reason as `thinking`. */
  readonly thinkingBudget?: number
  readonly topK?: number
  readonly topP?: number
}

/**
 * Resolve daemon config over the active user profile. A Claude Code profile
 * is used only when someone chose it and the `claude` CLI is installed: the
 * built-in one is merely the fallback on a fresh install, which must reach
 * "configure a provider" rather than spend a Claude plan nobody picked (or
 * fail every turn without the CLI).
 */
export function runtimeConnection(
  config: DaemonConfig,
  profile: ProviderProfile | undefined,
  options: { readonly claudeCodeChosen?: boolean; readonly claudeCodeInstalled?: () => boolean } = {},
): RuntimeConnection | undefined {
  const runtime = config.runtime
  const claudeCodeInstalled = options.claudeCodeInstalled ?? (() => claudeExecutable(process.env) !== undefined)
  const useProfile = profile && (profile.provider !== 'claude-code' || (options.claudeCodeChosen === true && claudeCodeInstalled())) ? profile : undefined
  const model = stringSetting(runtime.model) || useProfile?.model || ''
  if (!model) {
    return undefined
  }
  const rawPermissionMode = stringSetting(runtime.permission_mode)
  const permissionMode = runtimePermissionMode(rawPermissionMode) ?? DEFAULT_PERMISSION_MODE
  const baseUrl = stringSetting(runtime.base_url) || useProfile?.base_url
  const apiKey = stringSetting(runtime.api_key) || useProfile?.api_key
  const provider = stringSetting(runtime.provider) || useProfile?.provider
  const openAi = openAiOutputTokens(model, { ...(provider ? { provider } : {}), ...(baseUrl ? { base_url: baseUrl } : {}) })
  const runtimeMaxTokens = numberSetting(runtime.max_tokens)
  const profileMaxTokens = numberSetting(useProfile?.sampling.max_tokens)
  // Leave defaults at the adapter boundary, rather than pinning an OpenAI
  // default onto sessions using another provider. Never leak its -1 sentinel.
  const maxTokens = (runtimeMaxTokens === -1 && !openAi ? undefined : runtimeMaxTokens)
    ?? (profileMaxTokens === -1 && !openAi ? undefined : profileMaxTokens)
  // Unset for a model whose maker publishes its sampling: the provider client
  // applies that profile per request (see nativeSamplingDefaults).
  const nativeDefaults = nativeSamplingDefaults(model)
  const temperature = numberSetting(runtime.temperature)
    ?? numberSetting(useProfile?.sampling.temperature)
    ?? nativeDefaults.temperature
  const topK = numberSetting(runtime.top_k)
    ?? numberSetting(useProfile?.sampling.top_k)
    ?? nativeDefaults.top_k
  const topP = numberSetting(runtime.top_p)
    ?? numberSetting(useProfile?.sampling.top_p)
  const responsesApi = booleanSetting(runtime.responses_api)
  // Daemon config wins over profile sampling for all three thinking knobs so
  // an explicit runtime setting is never shadowed by a stored profile. Both
  // sources feed the turn runner only as defaults; ultra mode and prompt
  // keywords can still override them per turn.
  const thinking = booleanSetting(runtime.thinking) ?? booleanSetting(useProfile?.sampling.thinking)
  const thinkingBudget = numberSetting(runtime.thinking_budget)
    ?? numberSetting(useProfile?.sampling.thinking_budget)
  const reasoningEffort = stringSetting(runtime.reasoning_effort)
    || stringSetting(useProfile?.sampling.reasoning_effort)
  const serviceTier = stringSetting(runtime.service_tier)
    || stringSetting(useProfile?.sampling.service_tier)
  return {
    model,
    permissionMode,
    ...(baseUrl ? { baseUrl } : {}),
    ...(apiKey ? { apiKey } : {}),
    ...(provider ? { provider } : {}),
    ...(useProfile?.oauth_account_id ? { oauthAccountId: useProfile.oauth_account_id } : {}),
    ...(reasoningEffort ? { reasoningEffort } : {}),
    ...(responsesApi === undefined ? {} : { responsesApi }),
    ...(serviceTier ? { serviceTier } : {}),
    ...(maxTokens !== undefined ? { maxTokens } : {}),
    ...(temperature !== undefined ? { temperature } : {}),
    ...(thinking === undefined ? {} : { thinking }),
    ...(thinkingBudget !== undefined ? { thinkingBudget } : {}),
    ...(topK !== undefined ? { topK } : {}),
    ...(topP !== undefined ? { topP } : {}),
  }
}

/** Whether this route uses the OpenAI token-limit settings, never the Codex subscription path. */
export function openAiOutputTokens(model: string, overrides: Readonly<Record<string, unknown>>): boolean {
  try { return isOpenAiOutputTokenProvider(resolveProvider(model, overrides)) }
  catch { return false } // Unconfigured models still report their normal routing error at generation.
}

export function runtimePermissionMode(value: unknown): RuntimeConnection['permissionMode'] | undefined {
  return value === 'accept-all' || value === 'auto' || value === 'manual' || value === 'plan' ? value : undefined
}

function numberSetting(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function stringSetting(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function booleanSetting(value: unknown): boolean | undefined {
  if (typeof value === 'boolean') return value
  if (typeof value !== 'string') return undefined
  if (value === 'true' || value === '1') return true
  if (value === 'false' || value === '0') return false
  return undefined
}
