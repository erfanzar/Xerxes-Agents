// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { exists } from 'node:fs/promises'
import { resolve } from 'node:path'

import { PROVIDERS } from '../llms/providerRegistry.js'
import { runSetupWizard, writeSetupConfig } from './setupWizard.js'

import { SETUP_PROFILES, isSetupProfileName } from './setupProfiles.js'

export interface SetupCommandOptions {
  readonly answers?: Readonly<Record<string, unknown>>
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
  const answers = { ...profileAnswers, ...options.answers }
  const provider = String(answers.provider ?? 'anthropic')
  if (!Object.keys(PROVIDERS).includes(provider)) {
    throw new Error(`unknown provider ${provider}; supported: ${Object.keys(PROVIDERS).join(', ')}`)
  }
  const result = runSetupWizard(answers)
  await writeSetupConfig(result.answers, target)
  return 0
}
