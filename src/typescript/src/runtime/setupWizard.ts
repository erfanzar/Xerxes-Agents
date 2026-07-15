// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

export type SetupValidator = (value: unknown) => boolean

export interface SetupStep {
  readonly default?: unknown
  readonly key: string
  readonly optional?: boolean
  readonly prompt: string
  readonly validator?: SetupValidator
}

export interface WizardResult {
  readonly answers: Readonly<Record<string, unknown>>
  readonly skipped: readonly string[]
}

export interface RunWizardOptions {
  readonly steps?: readonly SetupStep[]
}

export const DEFAULT_SETUP_STEPS: readonly SetupStep[] = Object.freeze([
  {
    key: 'provider',
    prompt: 'Which LLM provider? [anthropic/openai/gemini/ollama]',
    default: 'anthropic',
  },
  {
    key: 'model',
    prompt: 'Default model id',
    default: 'claude-opus-4-6',
  },
  {
    key: 'api_key',
    prompt: 'API key (paste or skip if an environment variable is set)',
    optional: true,
  },
  {
    key: 'permission_mode',
    prompt: 'Permission mode [accept-all/auto/manual]',
    default: 'accept-all',
  },
  {
    key: 'enable_voice',
    prompt: 'Enable voice mode? [y/N]',
    default: 'n',
  },
  {
    key: 'messaging_platform',
    prompt: 'Bridge a messaging platform? [none/telegram/discord/slack]',
    default: 'none',
  },
])

/** Apply defaults and validators to pre-collected setup answers without owning an interactive UI. */
export function runSetupWizard(
  answers: Readonly<Record<string, unknown>> = {},
  options: RunWizardOptions = {},
): WizardResult {
  const resolved: Record<string, unknown> = {}
  const skipped: string[] = []
  for (const step of options.steps ?? DEFAULT_SETUP_STEPS) {
    const value = valueOrDefault(answers[step.key], step)
    if (step.optional && (value === undefined || value === null || value === '')) {
      skipped.push(step.key)
      continue
    }
    const validator = step.validator ?? acceptsAny
    if (!validator(value)) {
      throw new Error('invalid value for setup step ' + JSON.stringify(step.key))
    }
    resolved[step.key] = value
  }
  return Object.freeze({
    answers: Object.freeze({ ...resolved }),
    skipped: Object.freeze([...skipped]),
  })
}

/** Persist a minimal YAML-compatible configuration file using no third-party YAML dependency. */
export async function writeSetupConfig(
  answers: Readonly<Record<string, unknown>>,
  target: string,
): Promise<string> {
  await mkdir(dirname(target), { recursive: true })
  const lines = Object.entries(answers).map(([key, value]) => key + ': ' + yamlScalar(value))
  await writeFile(target, lines.join('\n') + '\n', 'utf8')
  return target
}

function valueOrDefault(value: unknown, step: SetupStep): unknown {
  return value === undefined || value === null || value === '' ? step.default : value
}

function acceptsAny(_value: unknown): boolean {
  return true
}

function yamlScalar(value: unknown): string {
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (value === undefined || value === null) return 'null'
  return JSON.stringify(value)
}
