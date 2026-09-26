// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { modelsDev, type LiveCost } from './modelsDev.js'
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
  /**
   * The models.dev provider that lists this adapter's models, when neither
   * its own id nor its endpoint is in models.dev: a subscription serving a
   * vendor's models (Codex → openai, Claude Code → anthropic). Used for a
   * model's name and description only — never its API price, which a
   * subscription does not pay.
   */
  readonly modelsDevVendor?: string
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
    modelsDevVendor: 'openai',
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
    modelsDevVendor: 'anthropic',
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

/**
 * models.dev names some providers differently from Xerxes (`google` is the
 * Gemini API, `zai`/`zhipuai` are Zhipu). A table of provider names — never
 * of models.
 */
const MODELS_DEV_PROVIDER_NAMES: Readonly<Record<string, ProviderName>> = {
  google: 'gemini',
  zai: 'zhipu',
  zhipuai: 'zhipu',
  'zai-coding-plan': 'zhipu',
  alibaba: 'qwen',
  'kimi-for-coding': 'kimi-code',
  moonshotai: 'moonshotai',
  'moonshotai-cn': 'moonshotai-cn',
}

function xerxesProviderFor(modelsDevId: string): ProviderName | undefined {
  const mapped = MODELS_DEV_PROVIDER_NAMES[modelsDevId]
  if (mapped) return mapped
  return isProviderName(modelsDevId) ? modelsDevId : undefined
}

export function isProviderName(value: string): value is ProviderName {
  return Object.hasOwn(PROVIDERS, value)
}

/** Honor `provider/model` routing syntax before consulting model prefixes. */
export function detectProvider(model: string, environment: Readonly<Record<string, string | undefined>> = process.env): ProviderName {
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
  // A bare id: whichever provider models.dev lists it under, when that is a
  // single provider Xerxes can drive — or, when several resell it, the single
  // one of those this environment holds a key for. Otherwise the
  // OpenAI-compatible default — never a guess from the model's name.
  const owners = new Set<ProviderName>()
  for (const provider of modelsDev.peek()) {
    if (!provider.models.has(model)) continue
    const name = xerxesProviderFor(provider.id)
    if (name) owners.add(name)
  }
  if (owners.size === 1) return [...owners][0]!
  const keyed = [...owners].filter(name => {
    const variable = PROVIDERS[name].apiKeyEnv
    return Boolean(variable && environment[variable])
  })
  return keyed.length === 1 ? keyed[0]! : 'openai'
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
  if (baseUrl.includes('kimi.com/coding')) {
    return 'kimi-code'
  }
  const slash = model.indexOf('/')
  const prefixed = slash > 0 && isProviderName(PROVIDER_ALIASES[model.slice(0, slash).toLowerCase()] ?? model.slice(0, slash).toLowerCase())
  return (!prefixed && providerServingBaseUrl(baseUrl)) || detectProvider(model)
}

/**
 * The one provider whose own endpoint is this base URL (same host and port),
 * e.g. the Gemini API root or Ollama's local port. Hosts several providers
 * share answer nothing.
 */
function providerServingBaseUrl(baseUrl: string): ProviderName | undefined {
  const host = urlHost(baseUrl)
  if (!host) return undefined
  const matches = (Object.keys(PROVIDERS) as ProviderName[]).filter(name => urlHost(PROVIDERS[name].baseUrl ?? '') === host)
  return matches.length === 1 ? matches[0] : undefined
}

function urlHost(value: string): string | undefined {
  try {
    const url = new URL(value)
    return url.protocol.startsWith('http') ? url.host.toLowerCase() : undefined
  } catch {
    return undefined
  }
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

export interface CostSource {
  readonly provider?: string
  readonly baseUrl?: string
  /** Prices the provider itself published for this model (its /models entry). */
  readonly reported?: LiveCost
  readonly cacheReadTokens?: number
  readonly cacheWriteTokens?: number
}

/**
 * What a run cost, in USD, from published prices only: the provider's own,
 * else models.dev's. Undefined when nobody publishes a price for the model —
 * an unknown cost, never an invented one (and never a fake $0).
 */
export function calcCost(model: string, inputTokens: number, outputTokens: number, source: CostSource = {}): number | undefined {
  const price = source.reported ?? modelsDev.find({ model, ...(source.provider ? { provider: source.provider } : {}), ...(source.baseUrl ? { baseUrl: source.baseUrl } : {}) })?.cost
  if (!price || price.input === undefined || price.output === undefined) return undefined
  const cacheRead = source.cacheReadTokens && price.cacheRead !== undefined ? source.cacheReadTokens * price.cacheRead : 0
  const cacheWrite = source.cacheWriteTokens && price.cacheWrite !== undefined ? source.cacheWriteTokens * price.cacheWrite : 0
  return (inputTokens * price.input + outputTokens * price.output + cacheRead + cacheWrite) / 1_000_000
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
