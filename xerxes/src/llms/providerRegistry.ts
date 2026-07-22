// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError } from '../core/errors.js'

export type ProviderTransport = 'anthropic' | 'claude-code' | 'openai'

export interface ProviderConfig {
  readonly apiKeyEnv?: string
  readonly baseUrl?: string
  readonly contextLimit: number
  readonly defaultApiKey?: string
  readonly name: string
  readonly transport: ProviderTransport
}

const provider = (
  name: string,
  transport: ProviderTransport,
  options: Omit<ProviderConfig, 'name' | 'transport'>,
): ProviderConfig => ({ name, transport, ...options })

/** Provider connection and routing metadata. Model catalogs are discovered live. */
export const PROVIDERS = {
  anthropic: provider('anthropic', 'anthropic', {
    apiKeyEnv: 'ANTHROPIC_API_KEY',
    baseUrl: 'https://api.anthropic.com',
    contextLimit: 200_000,
  }),
  openai: provider('openai', 'openai', {
    apiKeyEnv: 'OPENAI_API_KEY',
    baseUrl: 'https://api.openai.com/v1',
    contextLimit: 128_000,
  }),
  openrouter: provider('openrouter', 'openai', {
    apiKeyEnv: 'OPENROUTER_API_KEY',
    baseUrl: 'https://openrouter.ai/api/v1',
    contextLimit: 1_000_000,
  }),
  'claude-code': provider('claude-code', 'claude-code', {
    baseUrl: 'claude-code://local',
    contextLimit: 200_000,
  }),
  gemini: provider('gemini', 'openai', {
    apiKeyEnv: 'GEMINI_API_KEY',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
    contextLimit: 1_000_000,
  }),
  kimi: provider('kimi', 'openai', {
    apiKeyEnv: 'MOONSHOT_API_KEY',
    baseUrl: 'https://api.moonshot.cn/v1',
    contextLimit: 128_000,
  }),
  'kimi-code': provider('kimi-code', 'openai', {
    apiKeyEnv: 'KIMI_CODE_API_KEY',
    baseUrl: 'https://api.kimi.com/coding/v1',
    contextLimit: 262_144,
  }),
  qwen: provider('qwen', 'openai', {
    apiKeyEnv: 'DASHSCOPE_API_KEY',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    contextLimit: 1_000_000,
  }),
  zhipu: provider('zhipu', 'openai', {
    apiKeyEnv: 'ZHIPU_API_KEY',
    baseUrl: 'https://api.z.ai/api/coding/paas/v4',
    contextLimit: 128_000,
  }),
  deepseek: provider('deepseek', 'openai', {
    apiKeyEnv: 'DEEPSEEK_API_KEY',
    baseUrl: 'https://api.deepseek.com/v1',
    contextLimit: 64_000,
  }),
  minimax: provider('minimax', 'openai', {
    apiKeyEnv: 'MINIMAX_API_KEY',
    baseUrl: 'https://api.minimax.io/v1',
    contextLimit: 128_000,
  }),
  ollama: provider('ollama', 'openai', {
    baseUrl: 'http://localhost:11434/v1',
    contextLimit: 128_000,
    defaultApiKey: 'ollama',
  }),
  lmstudio: provider('lmstudio', 'openai', {
    baseUrl: 'http://localhost:1234/v1',
    contextLimit: 128_000,
    defaultApiKey: 'lm-studio',
  }),
  custom: provider('custom', 'openai', {
    apiKeyEnv: 'CUSTOM_API_KEY',
    // A custom endpoint has no trustworthy static context ceiling. Its live
    // `/models` metadata is authoritative; until discovery succeeds, report
    // the limit as unknown instead of pretending it is OpenAI's default.
    contextLimit: 0,
  }),
} as const satisfies Record<string, ProviderConfig>

export type ProviderName = keyof typeof PROVIDERS

export type ProviderOverrides = Readonly<Record<string, unknown>>

/** Costs in USD per million input/output tokens. */
export const COSTS: Readonly<Record<string, readonly [number, number]>> = {
  'claude-opus-4-6': [15, 75],
  'claude-opus-4-5': [15, 75],
  'claude-sonnet-4-6': [3, 15],
  'claude-sonnet-4-5': [3, 15],
  'claude-haiku-4-5-20251001': [0.8, 4],
  'claude-3-5-sonnet-20241022': [3, 15],
  'claude-3-5-haiku-20241022': [0.8, 4],
  'gpt-4o': [2.5, 10],
  'gpt-4o-mini': [0.15, 0.6],
  'gpt-4-turbo': [10, 30],
  'gpt-4.1': [2, 8],
  'gpt-4.1-mini': [0.4, 1.6],
  'gpt-4.1-nano': [0.1, 0.4],
  'o3-mini': [1.1, 4.4],
  o3: [10, 40],
  'o4-mini': [1.1, 4.4],
  o1: [15, 60],
  'o1-mini': [3, 12],
  'gemini-2.5-pro-preview-03-25': [1.25, 10],
  'gemini-2.5-flash-preview-04-17': [0.15, 0.6],
  'gemini-2.0-flash': [0.075, 0.3],
  'gemini-2.0-flash-lite': [0.075, 0.3],
  'gemini-1.5-pro': [1.25, 5],
  'gemini-1.5-flash': [0.075, 0.3],
  'moonshot-v1-8k': [1, 3],
  'moonshot-v1-32k': [2.4, 7],
  'moonshot-v1-128k': [8, 24],
  'kimi-latest': [2.4, 7],
  'kimi-for-coding': [2.4, 7],
  'qwen-max': [2.4, 9.6],
  'qwen-plus': [0.4, 1.2],
  'qwen-turbo': [0.2, 0.6],
  'qwen-long': [0.4, 1.2],
  'qwen3-235b-a22b': [2.4, 9.6],
  'deepseek-chat': [0.27, 1.1],
  'deepseek-coder': [0.27, 1.1],
  'deepseek-reasoner': [0.55, 2.19],
  'MiniMax-M2.7-highspeed': [0, 0],
  'MiniMax-M2.7-flashspeed': [0, 0],
  'MiniMax-Text-01': [0, 0],
  'MiniMax-Text-01-MiniApp': [0, 0],
  'glm-5.2': [0.6, 2.2],
  'glm-5.1': [0.6, 2.2],
  'glm-5v-turbo': [0.3, 1.1],
  'glm-5-turbo': [0.3, 1.1],
  'glm-5': [0.6, 2.2],
  'glm-4.7': [0.5, 0.5],
  'glm-4.6': [0.5, 0.5],
  'glm-4.5': [0.3, 1.1],
  'glm-4.5-air': [0.07, 0.07],
  sonnet: [0, 0],
  opus: [0, 0],
  haiku: [0, 0],
}

const PROVIDER_ALIASES: Readonly<Record<string, ProviderName>> = { 'claude_code': 'claude-code' }

const PREFIX_MAP = [
  ['claude-code/', 'claude-code'],
  ['claude-', 'anthropic'],
  ['gpt-', 'openai'],
  ['o1', 'openai'],
  ['o3', 'openai'],
  ['o4', 'openai'],
  ['openrouter/', 'openrouter'],
  ['gemini-', 'gemini'],
  ['moonshot-', 'kimi'],
  ['kimi-for-', 'kimi-code'],
  ['kimi-', 'kimi'],
  ['qwq-', 'qwen'],
  ['qwen', 'qwen'],
  ['glm-', 'zhipu'],
  ['deepseek-', 'deepseek'],
  ['minimax-', 'minimax'],
  ['abab', 'minimax'],
  ['llama', 'ollama'],
  ['mistral', 'ollama'],
  ['mixtral', 'ollama'],
  ['phi', 'ollama'],
  ['gemma', 'ollama'],
  ['codellama', 'ollama'],
] as [string, ProviderName][]

PREFIX_MAP.sort((left, right) => right[0].length - left[0].length)

const MODEL_CONTEXT_LIMITS: Readonly<Record<string, number>> = {
  opus: 1_000_000,
  fable: 1_000_000,
  mythos: 1_000_000,
  'claude-fable-5': 1_000_000,
  'claude-mythos-5': 1_000_000,
  'claude-mythos-preview': 1_000_000,
  'claude-opus-4-8': 1_000_000,
  'claude-opus-4-7': 1_000_000,
  'claude-opus-4-6': 1_000_000,
  'claude-opus-4-5': 1_000_000,
  'claude-sonnet-4-6': 1_000_000,
  'MiniMax-M2.7-highspeed': 1_024_000,
  'MiniMax-M2.7-flashspeed': 1_024_000,
  'MiniMax-Text-01': 256_000,
  'MiniMax-Text-01-MiniApp': 256_000,
  'glm-5.2': 1_048_576,
  'moonshot-v1-8k': 8_192,
  'moonshot-v1-32k': 32_768,
  'moonshot-v1-128k': 128_000,
  'kimi-latest': 262_144,
  'kimi-for-coding': 262_144,
  'kimi-k2.5': 262_144,
  'kimi-k2.6': 262_144,
  'kimi-k2.7': 262_144,
  'kimi-k2.5-001': 262_144,
  'kimi-k2.6-001': 262_144,
  'kimi-k2.7-001': 262_144,
  'kimi-k2.7-code': 262_144,
}

export function isProviderName(value: string): value is ProviderName {
  return Object.hasOwn(PROVIDERS, value)
}

/** Honor `provider/model` routing syntax before consulting model prefixes. */
export function detectProvider(model: string): ProviderName {
  const slash = model.indexOf('/')
  if (slash >= 0) {
    const explicit = model.slice(0, slash).toLowerCase()
    const alias = PROVIDER_ALIASES[explicit]
    if (alias) {
      return alias
    }
    if (isProviderName(explicit)) {
      return explicit
    }
    // An explicit `prefix/model` is a routing decision, not a guess: silently
    // retargeting an unrecognized prefix to OpenAI would send the request to
    // the wrong provider. Plugin provider prefixes are resolved by the client
    // factory before this registry path runs.
    throw new ConfigurationError(
      'model',
      `unknown provider prefix '${explicit}' in '${model}'; use a registered provider prefix, ` +
      'a plugin provider selected through the client factory, or a bare model id',
    )
  }
  const lower = model.toLowerCase()
  return PREFIX_MAP.find(([prefix]) => lower.startsWith(prefix))?.[1] ?? 'openai'
}

export function bareModel(model: string): string {
  const slash = model.indexOf('/')
  return slash >= 0 ? model.slice(slash + 1) : model
}

export function providerModel(model: string, providerName: ProviderName): string {
  if (providerName === 'openrouter') {
    return model.toLowerCase().startsWith('openrouter/') ? bareModel(model) : model
  }
  return bareModel(model)
}

export function resolveProvider(model: string, overrides: ProviderOverrides = {}): ProviderName {
  const configured = typeof overrides.provider === 'string' ? overrides.provider : overrides.provider_type
  if (typeof configured === 'string') {
    const normalized = configured.toLowerCase().replaceAll('_', '-')
    const alias = PROVIDER_ALIASES[normalized]
    if (alias) return alias
    if (isProviderName(normalized)) return normalized
  }

  const baseUrl = typeof overrides.base_url === 'string'
    ? overrides.base_url.toLowerCase()
    : typeof overrides.custom_base_url === 'string'
      ? overrides.custom_base_url.toLowerCase()
      : ''
  if (baseUrl.startsWith('claude-code://') || model.toLowerCase().startsWith('claude-code/')) {
    return 'claude-code'
  }
  if (baseUrl.includes('openrouter.ai') || model.toLowerCase().startsWith('openrouter/')) {
    return 'openrouter'
  }
  if (baseUrl.includes('kimi.com/coding') || bareModel(model).toLowerCase().startsWith('kimi-for-')) {
    return 'kimi-code'
  }
  return detectProvider(model)
}

export function getProviderConfig(providerName: ProviderName): ProviderConfig {
  return PROVIDERS[providerName]
}

export function getApiKey(providerName: ProviderName, overrides: ProviderOverrides = {}, environment = process.env): string {
  const configured = overrides[`${providerName}_api_key`]
  if (typeof configured === 'string' && configured) {
    return configured
  }
  const providerConfig = PROVIDERS[providerName]
  if (providerConfig.apiKeyEnv) {
    const environmentValue = environment[providerConfig.apiKeyEnv]
    if (environmentValue) {
      return environmentValue
    }
  }
  return providerConfig.defaultApiKey ?? ''
}

export function providerDefaultHeaders(providerName: ProviderName): Record<string, string> {
  if (providerName !== 'kimi-code') {
    return {}
  }
  return {
    'User-Agent': 'claude-code/1.0.0',
    'X-Stainless-Lang': 'claude-code',
    'X-Client-Name': 'claude-code',
  }
}

export function calcCost(model: string, inputTokens: number, outputTokens: number): number {
  const [inputRate, outputRate] = COSTS[bareModel(model)] ?? [0, 0]
  return (inputTokens * inputRate + outputTokens * outputRate) / 1_000_000
}

export function getContextLimit(model: string, overrides: ProviderOverrides = {}): number {
  const name = bareModel(model)
  const exact = MODEL_CONTEXT_LIMITS[name]
  if (exact !== undefined) {
    return exact
  }
  return PROVIDERS[resolveProvider(model, overrides)].contextLimit
}
