// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError } from '../core/errors.js'

export type ProviderTransport = 'anthropic' | 'claude-code' | 'openai'

export interface ProviderRetryPolicy {
  /** Backoff schedule in ms; a route makes `delaysMs.length + 1` attempts. */
  readonly delaysMs: readonly number[]
  /**
   * Ceiling for provider-suggested Retry-After waits, so one bad hint cannot
   * park a turn for hours.
   */
  readonly maxSuggestedDelayMs: number
}

/** Schedule every route without an explicit policy inherits. */
export const DEFAULT_RETRY_POLICY: ProviderRetryPolicy = Object.freeze({
  // Five attempts at a fixed ten-second cadence, then the turn fails —
  // the recovery contract for a dropped provider connection.
  delaysMs: [10_000, 10_000, 10_000, 10_000],
  maxSuggestedDelayMs: 60_000,
})

export interface ProviderConfig {
  readonly apiKeyEnv?: string
  readonly baseUrl?: string
  readonly defaultApiKey?: string
  readonly name: string
  /** Route-specific transient-failure patience; absent means the default policy. */
  readonly retry?: ProviderRetryPolicy
  readonly transport: ProviderTransport
}

const provider = (
  name: string,
  transport: ProviderTransport,
  options: Omit<ProviderConfig, 'name' | 'transport'>,
): ProviderConfig => ({ name, transport, ...options })

/** Provider connection and routing metadata; model capacities live in the Pi catalog layer. */
export const PROVIDERS = {
  anthropic: provider('anthropic', 'anthropic', {
    apiKeyEnv: 'ANTHROPIC_API_KEY',
    baseUrl: 'https://api.anthropic.com',
  }),
  openai: provider('openai', 'openai', {
    apiKeyEnv: 'OPENAI_API_KEY',
    baseUrl: 'https://api.openai.com/v1',
  }),
  // Subscription-backed: a ChatGPT Plus/Pro/Business plan authorizes this
  // endpoint with an OAuth session, so it has no API-key environment variable.
  // Peak-hour 429s clear within seconds-to-minutes; the shared 5×10s default
  // gives them forty seconds of room before the turn fails.
  'openai-codex': provider('openai-codex', 'openai', {
    baseUrl: 'https://chatgpt.com/backend-api/codex',
  }),
  // Subscription-backed like Codex: the GitHub OAuth device flow mints a
  // short-lived proxy token, and the api host derives from that token's
  // proxy-ep claim (copilotApiBase) rather than this static default.
  'github-copilot': provider('github-copilot', 'openai', {
    apiKeyEnv: 'COPILOT_GITHUB_TOKEN',
    baseUrl: 'https://api.individual.githubcopilot.com',
  }),
  // Deployment-scoped: the base URL comes from AZURE_OPENAI_RESOURCE_NAME or
  // base_url and the model id maps to a deployment name, so there is no
  // meaningful static default here.
  azure: provider('azure', 'openai', {
    apiKeyEnv: 'AZURE_OPENAI_API_KEY',
  }),
  // AWS-backed: SigV4/bearer auth and the SDK credential chain resolve inside
  // the Bedrock adapter; this endpoint is only the default pin when neither
  // AWS_REGION nor AWS_PROFILE is configured (pi-ai endpoint rules).
  'amazon-bedrock': provider('amazon-bedrock', 'openai', {
    baseUrl: 'https://bedrock-runtime.us-east-1.amazonaws.com',
  }),
  groq: provider('groq', 'openai', {
    apiKeyEnv: 'GROQ_API_KEY',
    baseUrl: 'https://api.groq.com/openai/v1',
  }),
  xai: provider('xai', 'openai', {
    apiKeyEnv: 'XAI_API_KEY',
    baseUrl: 'https://api.x.ai/v1',
  }),
  cerebras: provider('cerebras', 'openai', {
    apiKeyEnv: 'CEREBRAS_API_KEY',
    baseUrl: 'https://api.cerebras.ai/v1',
  }),
  together: provider('together', 'openai', {
    apiKeyEnv: 'TOGETHER_API_KEY',
    baseUrl: 'https://api.together.ai/v1',
  }),
  baseten: provider('baseten', 'openai', {
    apiKeyEnv: 'BASETEN_API_KEY',
    baseUrl: 'https://inference.baseten.co/v1',
  }),
  huggingface: provider('huggingface', 'openai', {
    apiKeyEnv: 'HF_TOKEN',
    baseUrl: 'https://router.huggingface.co/v1',
  }),
  nvidia: provider('nvidia', 'openai', {
    apiKeyEnv: 'NVIDIA_API_KEY',
    baseUrl: 'https://integrate.api.nvidia.com/v1',
  }),
  // Moonshot's international host; the 'kimi' provider is the .cn host.
  moonshotai: provider('moonshotai', 'openai', {
    apiKeyEnv: 'MOONSHOT_API_KEY',
    baseUrl: 'https://api.moonshot.ai/v1',
  }),
  'moonshotai-cn': provider('moonshotai-cn', 'openai', {
    apiKeyEnv: 'MOONSHOT_API_KEY',
    baseUrl: 'https://api.moonshot.cn/v1',
  }),
  'zai-coding-cn': provider('zai-coding-cn', 'openai', {
    apiKeyEnv: 'ZAI_CODING_CN_API_KEY',
    baseUrl: 'https://open.bigmodel.cn/api/coding/paas/v4',
  }),
  'qwen-token-plan': provider('qwen-token-plan', 'openai', {
    apiKeyEnv: 'QWEN_TOKEN_PLAN_API_KEY',
    baseUrl: 'https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1',
  }),
  'qwen-token-plan-cn': provider('qwen-token-plan-cn', 'openai', {
    apiKeyEnv: 'QWEN_TOKEN_PLAN_CN_API_KEY',
    baseUrl: 'https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',
  }),
  'qwen-token-plan-individual': provider('qwen-token-plan-individual', 'openai', {
    apiKeyEnv: 'QWEN_TOKEN_PLAN_API_KEY',
    baseUrl: 'https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1',
  }),
  xiaomi: provider('xiaomi', 'openai', {
    apiKeyEnv: 'XIAOMI_API_KEY',
    baseUrl: 'https://api.xiaomimimo.com/v1',
  }),
  'xiaomi-token-plan-ams': provider('xiaomi-token-plan-ams', 'openai', {
    apiKeyEnv: 'XIAOMI_TOKEN_PLAN_AMS_API_KEY',
    baseUrl: 'https://token-plan-ams.xiaomimimo.com/v1',
  }),
  'xiaomi-token-plan-cn': provider('xiaomi-token-plan-cn', 'openai', {
    apiKeyEnv: 'XIAOMI_TOKEN_PLAN_CN_API_KEY',
    baseUrl: 'https://token-plan-cn.xiaomimimo.com/v1',
  }),
  'xiaomi-token-plan-sgp': provider('xiaomi-token-plan-sgp', 'openai', {
    apiKeyEnv: 'XIAOMI_TOKEN_PLAN_SGP_API_KEY',
    baseUrl: 'https://token-plan-sgp.xiaomimimo.com/v1',
  }),
  'ant-ling': provider('ant-ling', 'openai', {
    apiKeyEnv: 'ANT_LING_API_KEY',
    baseUrl: 'https://api.ant-ling.com/v1',
  }),
  // Anthropic-messages hosts (pi-ai serves these through the Anthropic protocol).
  'minimax-cn': provider('minimax-cn', 'anthropic', {
    apiKeyEnv: 'MINIMAX_CN_API_KEY',
    baseUrl: 'https://api.minimaxi.com/anthropic',
  }),
  'vercel-ai-gateway': provider('vercel-ai-gateway', 'anthropic', {
    apiKeyEnv: 'AI_GATEWAY_API_KEY',
    baseUrl: 'https://ai-gateway.vercel.sh',
  }),
  // Multi-API gateways: the transport is decided per model from the catalog
  // entry's api field (see MULTI_API_PROVIDERS in client.ts).
  fireworks: provider('fireworks', 'openai', {
    apiKeyEnv: 'FIREWORKS_API_KEY',
    baseUrl: 'https://api.fireworks.ai/inference/v1',
  }),
  opencode: provider('opencode', 'openai', {
    apiKeyEnv: 'OPENCODE_API_KEY',
    baseUrl: 'https://opencode.ai/zen/v1',
  }),
  'opencode-go': provider('opencode-go', 'openai', {
    apiKeyEnv: 'OPENCODE_API_KEY',
    baseUrl: 'https://opencode.ai/zen/go/v1',
  }),
  // Account-templated gateway: the concrete base URL is resolved from
  // CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_GATEWAY_ID (or base_url) per API family.
  'cloudflare-ai-gateway': provider('cloudflare-ai-gateway', 'openai', {
    apiKeyEnv: 'CLOUDFLARE_API_KEY',
  }),
  // Pi's own gateway: the wire protocol is pi-messages (see llms/piMessages.ts)
  // and the model catalog is live (radiusGateway.ts), not static.
  radius: provider('radius', 'openai', {
    apiKeyEnv: 'RADIUS_API_KEY',
    baseUrl: 'https://radius.pi.dev',
  }),
  openrouter: provider('openrouter', 'openai', {
    apiKeyEnv: 'OPENROUTER_API_KEY',
    baseUrl: 'https://openrouter.ai/api/v1',
  }),
  'claude-code': provider('claude-code', 'claude-code', {
    baseUrl: 'claude-code://local',
  }),
  gemini: provider('gemini', 'openai', {
    apiKeyEnv: 'GEMINI_API_KEY',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
  }),
  kimi: provider('kimi', 'openai', {
    apiKeyEnv: 'MOONSHOT_API_KEY',
    baseUrl: 'https://api.moonshot.cn/v1',
  }),
  'kimi-code': provider('kimi-code', 'openai', {
    apiKeyEnv: 'KIMI_CODE_API_KEY',
    baseUrl: 'https://api.kimi.com/coding/v1',
  }),
  // Auth is GCP Application Default Credentials (or a Vertex express API
  // key), not a registry-managed API key, so there is no apiKeyEnv here.
  'google-vertex': provider('google-vertex', 'openai', {
    baseUrl: 'https://aiplatform.googleapis.com',
  }),
  mistral: provider('mistral', 'openai', {
    apiKeyEnv: 'MISTRAL_API_KEY',
    baseUrl: 'https://api.mistral.ai',
  }),
  // The endpoint is account-scoped: the base URL is materialized from
  // CLOUDFLARE_ACCOUNT_ID at client construction, not stored statically.
  'cloudflare-workers-ai': provider('cloudflare-workers-ai', 'openai', {
    apiKeyEnv: 'CLOUDFLARE_API_KEY',
  }),
  qwen: provider('qwen', 'openai', {
    apiKeyEnv: 'DASHSCOPE_API_KEY',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  }),
  zhipu: provider('zhipu', 'openai', {
    apiKeyEnv: 'ZHIPU_API_KEY',
    baseUrl: 'https://api.z.ai/api/coding/paas/v4',
  }),
  deepseek: provider('deepseek', 'openai', {
    apiKeyEnv: 'DEEPSEEK_API_KEY',
    baseUrl: 'https://api.deepseek.com/v1',
  }),
  minimax: provider('minimax', 'openai', {
    apiKeyEnv: 'MINIMAX_API_KEY',
    baseUrl: 'https://api.minimax.io/v1',
  }),
  ollama: provider('ollama', 'openai', {
    baseUrl: 'http://localhost:11434/v1',
    defaultApiKey: 'ollama',
    // A local daemon either answers or is down; long cloud-style backoffs just
    // stall the turn in front of a user who can see the server.
    retry: { delaysMs: [250, 500], maxSuggestedDelayMs: 5_000 },
  }),
  lmstudio: provider('lmstudio', 'openai', {
    baseUrl: 'http://localhost:1234/v1',
    defaultApiKey: 'lm-studio',
    retry: { delaysMs: [250, 500], maxSuggestedDelayMs: 5_000 },
  }),
  custom: provider('custom', 'openai', {
    apiKeyEnv: 'CUSTOM_API_KEY',
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

const PROVIDER_ALIASES: Readonly<Record<string, ProviderName>> = {
  'claude_code': 'claude-code',
  codex: 'openai-codex',
  'chatgpt': 'openai-codex',
  'openai_codex': 'openai-codex',
  copilot: 'github-copilot',
  'gh-copilot': 'github-copilot',
  'github_copilot': 'github-copilot',
  'azure-openai': 'azure',
  'azure_openai': 'azure',
  'azure-openai-responses': 'azure',
  bedrock: 'amazon-bedrock',
  aws: 'amazon-bedrock',
  'amazon_bedrock': 'amazon-bedrock',
  groqcloud: 'groq',
  'x-ai': 'xai',
  'hf': 'huggingface',
  'hugging-face': 'huggingface',
  moonshot: 'moonshotai',
  'moonshot-cn': 'moonshotai-cn',
  'moonshotai_cn': 'moonshotai-cn',
  'zai-coding': 'zai-coding-cn',
  bigmodel: 'zai-coding-cn',
  vercel: 'vercel-ai-gateway',
  'vercel_gateway': 'vercel-ai-gateway',
  'minimax_cn': 'minimax-cn',
  'opencode-zen': 'opencode',
  cloudflare: 'cloudflare-ai-gateway',
  'cf-ai-gateway': 'cloudflare-ai-gateway',
  'pi-gateway': 'radius',
  vertex: 'google-vertex',
  'google_vertex': 'google-vertex',
  'workers-ai': 'cloudflare-workers-ai',
  'workers_ai': 'cloudflare-workers-ai',
  'cf-workers-ai': 'cloudflare-workers-ai',
}

const PREFIX_MAP = [
  ['claude-code/', 'claude-code'],
  ['claude-', 'anthropic'],
  ['gpt-', 'openai'],
  ['o1', 'openai'],
  ['o3', 'openai'],
  ['o4', 'openai'],
  ['openrouter/', 'openrouter'],
  ['google-vertex/', 'google-vertex'],
  ['vertex/', 'google-vertex'],
  ['workers-ai/', 'cloudflare-workers-ai'],
  ['@cf/', 'cloudflare-workers-ai'],
  ['grok-', 'xai'],
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
  ['codestral', 'mistral'],
  ['pixtral', 'mistral'],
  ['magistral', 'mistral'],
  ['devstral', 'mistral'],
  ['ministral', 'mistral'],
  ['open-mixtral', 'mistral'],
  ['mistral-', 'mistral'],
  ['mixtral', 'mistral'],
  ['llama', 'ollama'],
  ['phi', 'ollama'],
  ['gemma', 'ollama'],
  ['codellama', 'ollama'],
] as [string, ProviderName][]

PREFIX_MAP.sort((left, right) => right[0].length - left[0].length)

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
  const configKey = typeof overrides.provider === 'string' ? 'provider' : 'provider_type'
  const configured = overrides[configKey]
  // An empty string is an unset override (many callers default to ''), not an
  // unknown provider name — fall through to automatic routing like `undefined`.
  if (typeof configured === 'string' && configured.trim()) {
    const normalized = configured.toLowerCase().replaceAll('_', '-')
    const alias = PROVIDER_ALIASES[normalized]
    if (alias) return alias
    if (isProviderName(normalized)) return normalized
    throw new ConfigurationError(
      configKey,
      `unknown provider '${configured}'; omit provider/provider_type to enable automatic model routing`,
    )
  }

  const baseUrl = typeof overrides.base_url === 'string'
    ? overrides.base_url.toLowerCase()
    : typeof overrides.custom_base_url === 'string'
      ? overrides.custom_base_url.toLowerCase()
      : ''
  if (baseUrl.startsWith('claude-code://') || model.toLowerCase().startsWith('claude-code/')) {
    return 'claude-code'
  }
  // Routing to the subscription backend is explicit only — `codex/gpt-5.3-codex`
  // or a matching base URL. A `-codex` model suffix is deliberately NOT a
  // trigger: silently moving `openai/gpt-5.3-codex` off the metered API onto
  // the user's ChatGPT plan changes who pays for the turn.
  if (baseUrl.includes('/backend-api/codex')) {
    return 'openai-codex'
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

/** Transient-failure patience for the route that would serve this model. */
export function retryPolicyForModel(model: string, overrides: ProviderOverrides = {}): ProviderRetryPolicy {
  return PROVIDERS[resolveProvider(model, overrides)].retry ?? DEFAULT_RETRY_POLICY
}

export interface EffectiveContextLimitOptions {
  /** Resolved model window from profile overrides or Pi's generated catalog. */
  readonly contextLimit?: number
  /** Actual configured reply allowance for this request. */
  readonly requestedOutputTokens?: number
}

/** Prompt budget from resolved model capacity and caller-configured output. */
export function effectiveContextLimit(options: EffectiveContextLimitOptions = {}): number {
  const reported = options.contextLimit
  if (typeof reported !== 'number' || !Number.isFinite(reported) || reported <= 0) return 0
  const limit = Math.floor(reported)
  const requested = options.requestedOutputTokens
  const reserve = typeof requested === 'number' && Number.isFinite(requested) && requested > 0
    ? Math.min(limit, Math.floor(requested))
    : 0
  return limit - reserve
}
