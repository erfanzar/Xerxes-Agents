// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { exists } from 'node:fs/promises'
import { resolve } from 'node:path'

import { discoverModelIds } from '../daemon/modelDiscovery.js'
import { PROVIDERS } from '../llms/providerRegistry.js'
import { runSetupWizard, writeSetupConfig } from './setupWizard.js'

import { SETUP_PROFILES, isSetupProfileName } from './setupProfiles.js'

export interface SetupCommandOptions {
  readonly answers?: Readonly<Record<string, unknown>>
  /** Swap model discovery (tests). */
  readonly discoverModels?: (input: { provider: string; baseUrl: string; apiKey?: string }) => Promise<string[]>
  readonly profile?: string
  readonly targetPath: string
}

export async function runSetupCommand(options: SetupCommandOptions): Promise<number> {
  const target = resolve(options.targetPath)
  if (await exists(target)) {
    throw new Error(`setup config already exists at ${target}; remove it first or edit it directly`)
  }
  const profileName = options.profile
  const profileAnswers = profileName !== undefined
    ? isSetupProfileName(profileName)
      ? SETUP_PROFILES[profileName].answers
      : {}
    : {}
  const answers: Record<string, unknown> = { ...profileAnswers, ...options.answers }
  const provider = typeof answers.provider === 'string' ? answers.provider.trim() : ''
  if (!provider) {
    throw new Error(`choose a provider with --provider; supported: ${Object.keys(PROVIDERS).join(', ')}`)
  }
  if (!Object.keys(PROVIDERS).includes(provider)) {
    throw new Error(`unknown provider ${provider}; supported: ${Object.keys(PROVIDERS).join(', ')}`)
  }
  if (typeof answers.model !== 'string' || !answers.model.trim()) {
    // No built-in model: ask the provider which ones it serves.
    const listed = await (options.discoverModels ?? discoverModelIds)({
      provider,
      baseUrl: PROVIDERS[provider as keyof typeof PROVIDERS].baseUrl ?? '',
      ...(typeof answers.api_key === 'string' && answers.api_key ? { apiKey: answers.api_key } : {}),
    }).catch(() => [] as string[])
    if (!listed[0]) throw new Error(`${provider} did not list its models; pass --model with the model id to use`)
    answers.model = listed[0]
  }
  const result = runSetupWizard(answers)
  await writeSetupConfig(result.answers, target)
  return 0
}
