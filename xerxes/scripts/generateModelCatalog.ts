// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  getBuiltinModelDataGeneratedAt,
  getBuiltinModels,
  type BuiltinProvider,
} from '@earendil-works/pi-ai/providers/all'
import piPackage from '@earendil-works/pi-ai/package.json' with { type: 'json' }

const PROVIDERS = {
  'amazon-bedrock': 'amazon-bedrock',
  'ant-ling': 'ant-ling',
  anthropic: 'anthropic',
  'azure-openai-responses': 'azure-openai-responses',
  baseten: 'baseten',
  cerebras: 'cerebras',
  'cloudflare-ai-gateway': 'cloudflare-ai-gateway',
  'cloudflare-workers-ai': 'cloudflare-workers-ai',
  deepseek: 'deepseek',
  fireworks: 'fireworks',
  gemini: 'google',
  'github-copilot': 'github-copilot',
  google: 'google',
  'google-vertex': 'google-vertex',
  groq: 'groq',
  huggingface: 'huggingface',
  kimi: 'moonshotai-cn',
  'kimi-code': 'kimi-coding',
  'kimi-coding': 'kimi-coding',
  mistral: 'mistral',
  minimax: 'minimax',
  'minimax-cn': 'minimax-cn',
  moonshotai: 'moonshotai',
  'moonshotai-cn': 'moonshotai-cn',
  nvidia: 'nvidia',
  openai: 'openai',
  'openai-codex': 'openai-codex',
  opencode: 'opencode',
  'opencode-go': 'opencode-go',
  openrouter: 'openrouter',
  qwen: 'qwen-token-plan',
  'qwen-token-plan': 'qwen-token-plan',
  'qwen-token-plan-cn': 'qwen-token-plan-cn',
  'qwen-token-plan-individual': 'qwen-token-plan-individual',
  together: 'together',
  'vercel-ai-gateway': 'vercel-ai-gateway',
  xai: 'xai',
  xiaomi: 'xiaomi',
  'xiaomi-token-plan-ams': 'xiaomi-token-plan-ams',
  'xiaomi-token-plan-cn': 'xiaomi-token-plan-cn',
  'xiaomi-token-plan-sgp': 'xiaomi-token-plan-sgp',
  zai: 'zai',
  'zai-coding-cn': 'zai-coding-cn',
  zhipu: 'zai',
} as const satisfies Readonly<Record<string, BuiltinProvider>>

interface GeneratedModelCapabilities {
  readonly api: string
  readonly base_url?: string
  readonly compat?: Readonly<Record<string, unknown>>
  readonly context_limit: number
  readonly max_output_tokens: number
  readonly reasoning: boolean
  readonly thinking_level_map?: Readonly<Record<string, string | null>>
}

function modelCapabilities(provider: BuiltinProvider): Record<string, GeneratedModelCapabilities> {
  const capabilities: Record<string, GeneratedModelCapabilities> = Object.create(null)
  for (const model of getBuiltinModels(provider)) {
    if (!Number.isSafeInteger(model.contextWindow) || model.contextWindow <= 0) {
      throw new Error(`pi-ai ${provider}/${model.id} has an invalid context window`)
    }
    if (!Number.isSafeInteger(model.maxTokens) || model.maxTokens <= 0) {
      throw new Error(`pi-ai ${provider}/${model.id} has an invalid max output token count`)
    }
    const metadata = model as typeof model & {
      readonly compat?: Readonly<Record<string, unknown>>
      readonly thinkingLevelMap?: Readonly<Record<string, string | null>>
    }
    capabilities[model.id] = {
      api: model.api,
      context_limit: model.contextWindow,
      max_output_tokens: model.maxTokens,
      reasoning: model.reasoning,
      // Multi-API gateways (opencode, fireworks, cloudflare) pin a different
      // base URL per api family; routing needs the entry's own value.
      ...(model.baseUrl ? { base_url: model.baseUrl } : {}),
      ...(metadata.compat ? { compat: metadata.compat } : {}),
      ...(metadata.thinkingLevelMap ? { thinking_level_map: metadata.thinkingLevelMap } : {}),
    }
  }
  return Object.fromEntries(Object.entries(capabilities).sort(([left], [right]) => left.localeCompare(right)))
}

const providers = Object.fromEntries(
  Object.entries(PROVIDERS).map(([xerxesProvider, piProvider]) => [
    xerxesProvider,
    modelCapabilities(piProvider),
  ]),
)
const generatedAt = getBuiltinModelDataGeneratedAt()
const output = {
  source: {
    package: '@earendil-works/pi-ai',
    version: piPackage.version,
    ...(generatedAt === undefined ? {} : { generated_at: new Date(generatedAt).toISOString() }),
  },
  providers,
}
const destination = new URL('../src/llms/piModelCatalog.generated.json', import.meta.url)
const serialized = `${JSON.stringify(output, null, 2)}\n`
if (process.argv.includes('--check')) {
  const existing = await Bun.file(destination).text()
  if (existing !== serialized) {
    throw new Error('Pi model catalog is stale; run bun run generate:model-catalog')
  }
  console.log(`Pi model catalog matches pi-ai ${piPackage.version}`)
} else {
  await Bun.write(destination, serialized)
  console.log(`Generated ${destination.pathname} from pi-ai ${piPackage.version}`)
}
