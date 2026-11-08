// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Model cards: every setting Xerxes tunes for a model, in one record. Prism's
 * Bonsai models each have a card; any other Bonsai id gets the family card,
 * and every other model gets {@link DEFAULT_CARD}, which changes nothing.
 *
 * This is the only file that matches Bonsai model names. Code that needs a
 * Bonsai-specific value asks {@link modelCard} for it.
 */
import type { SamplingValues } from './samplingDefaults.js'

const GIB = 1024 ** 3

/**
 * Sampling a model's maker publishes, for thinking and non-thinking requests.
 * Open-weight servers (MLX, llama.cpp, LM Studio) do not read the model's
 * `generation_config.json`, so a field a request leaves out is the engine's
 * own default, not the model's: a card with sampling gets it on every request.
 */
export interface PublishedSampling {
  /** Where the values come from, for status text. */
  readonly source: string
  readonly thinking: SamplingValues
  readonly instruct: SamplingValues
}

/** A llama-server (GGUF) build Xerxes can download and run. */
export interface LlamaServerBuild {
  /** File name under the local models directory. */
  readonly fileName: string
  readonly minMemoryBytes: number
  /** Launch window: what fits in this build's memory tier next to the weights, not the model's limit. */
  readonly ctxSize: number
}

/**
 * How long one reply may reason before it acts (see `runtime/replyBudget.ts`):
 * a reply still reasoning past either limit, with no tool call and no answer
 * text yet, is stopped and the model is asked again with a note.
 */
export interface ReplyBudget {
  /** Reasoning tokens, estimated at 4 characters each. */
  readonly maxReasoningTokens: number
  /** Seconds from the reply's first streamed token, so reading the prompt does not count. */
  readonly maxSeconds: number
}

/** An LM Studio build (`lms ls` key). */
export interface LmStudioBuild {
  /** LM Studio model key; also the API identifier Xerxes loads it under. */
  readonly modelKey: string
  readonly minMemoryBytes: number
}

export interface ModelCard {
  /** Stable id; for a model with a local build, the id `xerxes local setup --model` takes. */
  readonly id: string
  readonly label: string
  /** True for Prism's Bonsai models, the ones this harness is tuned for. */
  readonly bonsai: boolean
  /** Model ids this card answers to, local or hosted. */
  readonly match?: RegExp
  /** Sent on every request unless a setting or the request says otherwise. */
  readonly sampling?: PublishedSampling
  /** Output ceiling when nothing else sets one (the maker's own agent setup). */
  readonly maxOutputTokens?: number
  /** The model's own context window. */
  readonly nativeContext?: number
  /** Prompt profile when none is configured; unset means the route's window decides. */
  readonly promptProfile?: 'lean' | 'full'
  /** The cascade is on by default for this model when no setting chooses. */
  readonly escalationByDefault?: boolean
  /** The per-reply budget when `runtime.reply_budget` is unset; none means no budget. */
  readonly replyBudget?: ReplyBudget
  readonly llamaServer?: LlamaServerBuild
  readonly lmstudio?: LmStudioBuild
}

/** Bonsai 2's non-thinking sampling, the same for every size. */
const BONSAI_2_INSTRUCT: SamplingValues = Object.freeze({
  temperature: 0.7,
  topP: 0.8,
  topK: 20,
  minP: 0,
  presencePenalty: 1.5,
  repetitionPenalty: 1,
})

/** Prism's small-model guide: the 27B's thinking values with presence penalty 1.0. */
const BONSAI_2_SMALL_SAMPLING: PublishedSampling = Object.freeze({
  source: "Prism's small-model guide",
  thinking: Object.freeze({ temperature: 1, topP: 0.95, topK: 20, minP: 0, presencePenalty: 1, repetitionPenalty: 1 }),
  instruct: BONSAI_2_INSTRUCT,
})

/**
 * What every Bonsai card shares: a small model picks better among a few tools,
 * a failed check is cheap to retry, and one reply that reasons for many
 * minutes without acting should not use up the whole task. The reply budget is
 * a quarter of the 27B's 32K output ceiling, and ten minutes so a slow machine
 * does not wait out the token limit.
 */
const BONSAI_DEFAULTS = {
  bonsai: true,
  promptProfile: 'lean',
  escalationByDefault: true,
  replyBudget: Object.freeze({ maxReasoningTokens: 8_192, maxSeconds: 600 }),
} as const

/**
 * Most specific first: the first card whose `match` accepts an id wins, and
 * the last Bonsai card catches any other Bonsai id.
 */
export const MODEL_CARDS: readonly ModelCard[] = Object.freeze([
  Object.freeze({
    ...BONSAI_DEFAULTS,
    id: 'ternary-bonsai-2-27b',
    label: 'Ternary Bonsai 2 27B',
    // Served as ternary-bonsai-2-27b(-mlx), prism-ml/Ternary-Bonsai-2-27B-mlx-2bit,
    // Bonsai-2-27B-Ternary-JANG and bonsai-2-27b on Ollama.
    match: /bonsai[-_. ]?2[-_. ]?27b/i,
    sampling: Object.freeze({
      source: "Prism's model card",
      thinking: Object.freeze({ temperature: 1, topP: 0.95, topK: 20, minP: 0, presencePenalty: 0, repetitionPenalty: 1 }),
      instruct: BONSAI_2_INSTRUCT,
    }),
    // The ceiling Prism's own agent demo runs with, and the hosted completion limit.
    maxOutputTokens: 32_768,
    nativeContext: 262_144,
    // 16K is the llama-server launch default for a 24 GB machine; raise it with local.ctx_size.
    llamaServer: Object.freeze({ fileName: 'ternary-bonsai-2-27b.gguf', minMemoryBytes: 24 * GIB, ctxSize: 16_384 }),
    // The 2-bit MLX pack (8.6 GB on disk), measured on a 24 GB M4 Pro on 2026-11-05
    // with a 42,752-token window: a 16 GiB budget (20 after the OS reserve) is enough.
    lmstudio: Object.freeze({ modelKey: 'ternary-bonsai-2-27b-mlx', minMemoryBytes: 16 * GIB }),
  }),
  Object.freeze({
    ...BONSAI_DEFAULTS,
    // `bonsai-9b` is the local catalog's id for it, kept so existing configs still resolve.
    id: 'bonsai-9b',
    label: 'Bonsai 2 9B',
    match: /bonsai[-_. ]?2[-_. ]?9b|(?:^|[/:])bonsai-9b(?![0-9a-z])/i,
    sampling: BONSAI_2_SMALL_SAMPLING,
    llamaServer: Object.freeze({ fileName: 'bonsai-9b.gguf', minMemoryBytes: 12 * GIB, ctxSize: 32_768 }),
  }),
  Object.freeze({
    ...BONSAI_DEFAULTS,
    id: 'bonsai-2-4b',
    label: 'Bonsai 2 4B',
    match: /bonsai[-_. ]?2[-_. ]?4b/i,
    sampling: BONSAI_2_SMALL_SAMPLING,
  }),
  Object.freeze({
    ...BONSAI_DEFAULTS,
    id: 'bonsai-2-2b',
    label: 'Bonsai 2 2B',
    match: /bonsai[-_. ]?2[-_. ]?2b/i,
    sampling: BONSAI_2_SMALL_SAMPLING,
  }),
  // Any other Bonsai id (bonsai-8b, a fine-tune, a future size): the lean prompt
  // and the cascade, but no sampling, which Prism publishes per model.
  Object.freeze({ ...BONSAI_DEFAULTS, id: 'bonsai', label: 'Bonsai (no published card)', match: /bonsai/i }),
])

/** Every other model: no card values, so Xerxes's general defaults apply. */
export const DEFAULT_CARD: ModelCard = Object.freeze({ id: 'default', label: 'No model card', bonsai: false })

/** The card for a model id; {@link DEFAULT_CARD} when none matches or no model is set. */
export function modelCard(model: string | undefined): ModelCard {
  if (!model) return DEFAULT_CARD
  return MODEL_CARDS.find(card => card.match?.test(model)) ?? DEFAULT_CARD
}

/** Cards with a build `xerxes local setup` can run. */
export function localModelCards(): readonly ModelCard[] {
  return MODEL_CARDS.filter(card => card.llamaServer !== undefined)
}
