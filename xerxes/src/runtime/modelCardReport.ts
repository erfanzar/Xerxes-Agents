// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * What `/bonsai` and `xerxes bonsai` show: the active model's card, and for
 * each setting the card holds, the value in effect and what set it.
 */
import { resolvedProfileModelCapabilities, type ProviderProfile } from '../bridge/profiles.js'
import { modelCard, type ModelCard, type ReplyBudget } from '../llms/modelCards.js'
import { DEFAULT_TEMPERATURE, DEFAULT_TOP_K, formatSamplingValues } from '../llms/samplingDefaults.js'
import { deriveDataTrust } from '../security/dataBoundary.js'
import { defaultEscalationMode, ESCALATION_ENV, parseEscalationConfig } from './escalationPolicy.js'
import { parsePromptProfileConfig, PROMPT_PROFILE_ENV } from './leanPrompt.js'
import { resolveReplyBudget } from './replyBudget.js'

/** The wire names of the sampling values a card publishes. */
const CARD_SAMPLING_KEYS = ['temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty'] as const

export interface ModelCardReportInput {
  readonly model: string
  readonly runtime: Record<string, unknown>
  readonly env: Record<string, string | undefined>
  readonly profile?: ProviderProfile
  /** Sampling values a setting chose (runtime, the profile's block or `/sampling`), by wire name. */
  readonly samplingSet?: Readonly<Record<string, unknown>>
  /** The window the engine says it loaded, when the daemon discovered one. */
  readonly loadedWindow?: number
}

/** One setting: the card's value, the value in effect, and where that comes from. */
export interface ModelCardSetting {
  readonly name: string
  readonly card: string
  readonly effective: string
  readonly from: string
}

export interface ModelCardReport {
  readonly model: string
  readonly card: ModelCard
  readonly settings: readonly ModelCardSetting[]
}

function count(value: number): string {
  return value.toLocaleString('en-US')
}

function samplingSetting(card: ModelCard, set: Readonly<Record<string, unknown>>): ModelCardSetting {
  const chosen = CARD_SAMPLING_KEYS.filter(key => typeof set[key] === 'number').map(key => `${key} ${String(set[key])}`)
  const cardText = card.sampling
    ? `${formatSamplingValues(card.sampling.thinking)} (thinking); ${formatSamplingValues(card.sampling.instruct)} (thinking off)`
    : '—'
  if (!card.sampling) {
    // Without a card the daemon writes Xerxes's defaults into its settings, so
    // those two values are defaults, not choices.
    const isDefault = (key: string) =>
      (key === 'temperature' && set[key] === DEFAULT_TEMPERATURE) || (key === 'top_k' && set[key] === DEFAULT_TOP_K)
    const fromSettings = CARD_SAMPLING_KEYS.filter(key => typeof set[key] === 'number' && !isDefault(key))
    const values = chosen.length ? chosen.join(', ') : `temperature ${DEFAULT_TEMPERATURE}, top_k ${DEFAULT_TOP_K}`
    return { name: 'Sampling', card: cardText, effective: values, from: fromSettings.length ? 'settings' : 'Xerxes default' }
  }
  if (!chosen.length) return { name: 'Sampling', card: cardText, effective: 'card', from: card.sampling.source }
  return { name: 'Sampling', card: cardText, effective: `${chosen.join(', ')}; the rest from the card`, from: 'settings' }
}

function outputSetting(input: ModelCardReportInput, card: ModelCard): ModelCardSetting {
  const resolved = resolvedProfileModelCapabilities(input.profile, input.model)
  const cardText = card.maxOutputTokens === undefined ? '—' : `${count(card.maxOutputTokens)} tokens`
  if (resolved.maxOutputTokens === undefined) return { name: 'Output limit', card: cardText, effective: 'none', from: '—' }
  const from = resolved.outputSource === 'override'
    ? "profile's max_output_tokens"
    : resolved.outputSource === 'provider'
      ? 'provider'
      : resolved.maxOutputTokens === card.maxOutputTokens ? 'card' : 'models.dev'
  return { name: 'Output limit', card: cardText, effective: `${count(resolved.maxOutputTokens)} tokens`, from }
}

function contextSetting(input: ModelCardReportInput, card: ModelCard): ModelCardSetting {
  const resolved = resolvedProfileModelCapabilities(input.profile, input.model)
  const cardText = card.nativeContext === undefined ? '—' : `reads ${count(card.nativeContext)} tokens`
  if (resolved.contextSource === 'override' && resolved.contextLimit !== undefined) {
    return { name: 'Context window', card: cardText, effective: `${count(resolved.contextLimit)} tokens`, from: "profile's context_limit" }
  }
  if (input.loadedWindow !== undefined) {
    return { name: 'Context window', card: cardText, effective: `${count(input.loadedWindow)} tokens`, from: 'loaded by the engine' }
  }
  if (resolved.contextLimit !== undefined) {
    return { name: 'Context window', card: cardText, effective: `${count(resolved.contextLimit)} tokens`, from: resolved.contextSource === 'provider' ? 'provider' : 'models.dev' }
  }
  return { name: 'Context window', card: cardText, effective: 'unknown', from: '—' }
}

function promptSetting(input: ModelCardReportInput, card: ModelCard): ModelCardSetting {
  const effective = parsePromptProfileConfig(input.runtime, input.env, input.profile?.prompt_profile, { model: input.model })
  const from = input.profile?.prompt_profile !== undefined
    ? "profile's prompt_profile"
    : input.env[PROMPT_PROFILE_ENV]?.trim()
      ? PROMPT_PROFILE_ENV
      : input.runtime['prompt_profile'] !== undefined && input.runtime['prompt_profile'] !== null
        ? 'runtime.prompt_profile'
        : card.promptProfile !== undefined ? 'card' : 'context window'
  return { name: 'Prompt and tools', card: card.promptProfile ?? '—', effective, from }
}

function escalationSetting(input: ModelCardReportInput, card: ModelCard): ModelCardSetting {
  const local = () => deriveDataTrust(input.profile) === 'local'
  const fallback = defaultEscalationMode({ runtime: input.runtime, env: input.env, model: input.model, local })
  const { mode } = parseEscalationConfig(input.runtime, input.env, fallback)
  const raw = input.runtime['escalation']
  const fileMode = typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>)['mode'] : undefined
  const from = input.env[ESCALATION_ENV]?.trim()
    ? ESCALATION_ENV
    : fileMode !== undefined && fileMode !== null
      ? 'runtime.escalation.mode'
      : card.escalationByDefault ? 'card' : 'local or cloud model'
  return { name: 'Result checks and escalation', card: card.escalationByDefault ? 'on' : '—', effective: mode, from }
}

function replyBudgetText(budget: ReplyBudget | null | undefined): string {
  if (!budget) return 'none'
  const time = budget.maxSeconds % 60 === 0 ? `${count(budget.maxSeconds / 60)} min` : `${count(budget.maxSeconds)} s`
  return `${count(budget.maxReasoningTokens)} reasoning tokens or ${time}`
}

function replyBudgetSetting(input: ModelCardReportInput, card: ModelCard): ModelCardSetting {
  const raw = input.runtime['reply_budget']
  const from = raw !== undefined && raw !== null ? 'runtime.reply_budget' : card.replyBudget ? 'card' : '—'
  return {
    name: 'Reply budget',
    card: card.replyBudget ? replyBudgetText(card.replyBudget) : '—',
    effective: replyBudgetText(resolveReplyBudget(input.runtime, input.model)),
    from,
  }
}

function buildsSetting(card: ModelCard): ModelCardSetting | undefined {
  const builds = [
    ...(card.llamaServer ? [`llama-server ${card.llamaServer.fileName} (needs ${count(card.llamaServer.minMemoryBytes / 1024 ** 3)} GiB${card.llamaServer.minLlamaCpp === undefined ? '' : ` and llama.cpp b${card.llamaServer.minLlamaCpp.build}+`}, launches with ${count(card.llamaServer.ctxSize)} tokens)`] : []),
    ...(card.mlx ? [`mlx_lm.server ${card.mlx.repo} (needs ${count(card.mlx.minMemoryBytes / 1024 ** 3)} GiB and mlx-lm ${card.mlx.minMlxLm}+)`] : []),
    ...(card.lmstudio ? [`LM Studio ${card.lmstudio.modelKey} (needs ${count(card.lmstudio.minMemoryBytes / 1024 ** 3)} GiB)`] : []),
  ]
  return builds.length ? { name: 'Local builds', card: builds.join('; '), effective: '—', from: '—' } : undefined
}

/** The card for `input.model` and every setting it holds, with the value in effect. */
export function modelCardReport(input: ModelCardReportInput): ModelCardReport {
  const card = modelCard(input.model)
  const builds = buildsSetting(card)
  return {
    model: input.model,
    card,
    settings: [
      samplingSetting(card, input.samplingSet ?? {}),
      outputSetting(input, card),
      contextSetting(input, card),
      promptSetting(input, card),
      escalationSetting(input, card),
      replyBudgetSetting(input, card),
      ...(builds ? [builds] : []),
    ],
  }
}

function cell(text: string): string {
  return text.replace(/\|/g, '\\|')
}

/** Markdown for the TUI and the terminal. */
export function formatModelCardReport(report: ModelCardReport): string {
  const heading = report.card.bonsai
    ? `Model card for \`${report.model}\`: ${report.card.label} (\`${report.card.id}\`)`
    : `No model card for \`${report.model}\`: Xerxes's general defaults apply.`
  return [
    heading,
    '',
    '| Setting | Card | In effect | Set by |',
    '| --- | --- | --- | --- |',
    ...report.settings.map(setting => `| ${cell(setting.name)} | ${cell(setting.card)} | ${cell(setting.effective)} | ${cell(setting.from)} |`),
    '',
    'Settings win over the card: `/sampling`, a profile\'s `max_output_tokens` or `prompt_profile`, `runtime.escalation.mode`, `runtime.reply_budget`.',
  ].join('\n')
}
