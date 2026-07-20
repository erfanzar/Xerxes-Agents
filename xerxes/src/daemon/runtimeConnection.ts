// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ProviderProfile } from '../bridge/profiles.js'
import { DEFAULT_TEMPERATURE, DEFAULT_TOP_K } from '../llms/samplingDefaults.js'
import { DEFAULT_PERMISSION_MODE, type PermissionMode } from '../streaming/permissions.js'
import type { DaemonConfig } from './config.js'

export interface RuntimeConnection {
  readonly apiKey?: string
  readonly baseUrl?: string
  readonly maxTokens?: number
  readonly model: string
  readonly permissionMode: PermissionMode
  readonly provider?: string
  readonly reasoningEffort?: string
  readonly responsesApi?: boolean
  readonly temperature?: number
  readonly thinking?: boolean
  readonly thinkingBudget?: number
  readonly topK?: number
  readonly topP?: number
}

/** Resolve daemon config over the active user profile without selecting the builtin Claude Code placeholder. */
export function runtimeConnection(config: DaemonConfig, profile: ProviderProfile | undefined): RuntimeConnection | undefined {
  const runtime = config.runtime
  const useProfile = profile?.name !== 'cc' ? profile : undefined
  const model = stringSetting(runtime.model) || useProfile?.model || ''
  if (!model) {
    return undefined
  }
  const rawPermissionMode = stringSetting(runtime.permission_mode)
  const permissionMode = isPermissionMode(rawPermissionMode) ? rawPermissionMode : DEFAULT_PERMISSION_MODE
  const baseUrl = stringSetting(runtime.base_url) || useProfile?.base_url
  const apiKey = stringSetting(runtime.api_key) || useProfile?.api_key
  const provider = stringSetting(runtime.provider) || useProfile?.provider
  const maxTokens = numberSetting(runtime.max_tokens)
    ?? numberSetting(useProfile?.sampling.max_tokens)
  const temperature = numberSetting(runtime.temperature)
    ?? numberSetting(useProfile?.sampling.temperature)
    ?? DEFAULT_TEMPERATURE
  const topK = numberSetting(runtime.top_k)
    ?? numberSetting(useProfile?.sampling.top_k)
    ?? DEFAULT_TOP_K
  const topP = numberSetting(runtime.top_p)
    ?? numberSetting(useProfile?.sampling.top_p)
  const responsesApi = booleanSetting(runtime.responses_api)
  const thinking = booleanSetting(runtime.thinking) ?? booleanSetting(useProfile?.sampling.thinking)
  const thinkingBudget = numberSetting(runtime.thinking_budget)
    ?? numberSetting(useProfile?.sampling.thinking_budget)
  const reasoningEffort = stringSetting(runtime.reasoning_effort)
    || stringSetting(useProfile?.sampling.reasoning_effort)
  return {
    model,
    permissionMode,
    ...(baseUrl ? { baseUrl } : {}),
    ...(apiKey ? { apiKey } : {}),
    ...(provider ? { provider } : {}),
    ...(reasoningEffort ? { reasoningEffort } : {}),
    ...(responsesApi === undefined ? {} : { responsesApi }),
    ...(maxTokens !== undefined ? { maxTokens } : {}),
    temperature,
    ...(thinking === undefined ? {} : { thinking }),
    ...(thinkingBudget !== undefined ? { thinkingBudget } : {}),
    topK,
    ...(topP !== undefined ? { topP } : {}),
  }
}

function isPermissionMode(value: string): value is RuntimeConnection['permissionMode'] {
  return value === 'accept-all' || value === 'auto' || value === 'manual' || value === 'plan'
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
