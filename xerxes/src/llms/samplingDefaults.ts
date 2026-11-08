// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Provider-neutral temperature used when neither runtime nor profile overrides it. */
export const DEFAULT_TEMPERATURE = 0.6

/** Provider-neutral top-k used when neither runtime nor profile overrides it. */
export const DEFAULT_TOP_K = 64

/** One complete sampling setting, as a model's maker publishes it. */
export interface SamplingValues {
  readonly temperature: number
  readonly topP: number
  readonly topK: number
  readonly minP: number
  readonly presencePenalty: number
  readonly repetitionPenalty: number
}

/** Whether the model is asked to think before answering; makers tune sampling for each. */
export type SamplingMode = 'thinking' | 'instruct'

/**
 * Sampling a model's maker publishes for it. Open-weight servers (MLX,
 * llama.cpp, LM Studio) do not read the model's `generation_config.json`, so
 * any field a request leaves out is the engine's own default, not the
 * model's. A model listed here gets its whole profile on every request.
 */
export interface PublishedSamplingProfile {
  /** Where the values come from, for status text. */
  readonly source: string
  readonly thinking: SamplingValues
  readonly instruct: SamplingValues
  /** Output ceiling the maker runs its own agent setup with, when it publishes one. */
  readonly maxOutputTokens?: number
}

/**
 * Ternary Bonsai 2 27B under the ids its packs are served as:
 * ternary-bonsai-2-27b(-mlx), prism-ml/Ternary-Bonsai-2-27B-mlx-2bit,
 * Bonsai-2-27B-Ternary-JANG, bonsai-2-27b on Ollama.
 */
const BONSAI_2_27B = /bonsai[-_. ]?2[-_. ]?27b/i
/** Bonsai 2's small sizes (2B, 4B, 9B), including the local catalog's `bonsai-9b`. */
const BONSAI_2_SMALL = /bonsai[-_. ]?2[-_. ]?[249]b|(?:^|[/:])bonsai-9b(?![0-9a-z])/i

/** Bonsai 2's non-thinking profile, the same for every size. */
const BONSAI_2_INSTRUCT: SamplingValues = Object.freeze({
  temperature: 0.7,
  topP: 0.8,
  topK: 20,
  minP: 0,
  presencePenalty: 1.5,
  repetitionPenalty: 1,
})

/**
 * Prism's model card (thinking: temperature 1.0, top_p 0.95, top_k 20, min_p
 * 0, presence 0, repetition 1.0). The 32,768-token ceiling is the one Prism's
 * own agent demo runs with, and the model's completion limit on hosted APIs.
 */
const BONSAI_2_27B_PROFILE: PublishedSamplingProfile = Object.freeze({
  source: "Prism's model card",
  thinking: Object.freeze({ temperature: 1, topP: 0.95, topK: 20, minP: 0, presencePenalty: 0, repetitionPenalty: 1 }),
  instruct: BONSAI_2_INSTRUCT,
  maxOutputTokens: 32_768,
})

/** Prism's small-model guide: as the 27B, but presence penalty 1.0 when thinking. */
const BONSAI_2_SMALL_PROFILE: PublishedSamplingProfile = Object.freeze({
  source: "Prism's small-model guide",
  thinking: Object.freeze({ temperature: 1, topP: 0.95, topK: 20, minP: 0, presencePenalty: 1, repetitionPenalty: 1 }),
  instruct: BONSAI_2_INSTRUCT,
})

/** The maker's published sampling for this model id, if Xerxes knows one. */
export function publishedSamplingProfile(model: string): PublishedSamplingProfile | undefined {
  if (BONSAI_2_27B.test(model)) return BONSAI_2_27B_PROFILE
  if (BONSAI_2_SMALL.test(model)) return BONSAI_2_SMALL_PROFILE
  return undefined
}

/** The published values for one mode, if the model has a published profile. */
export function publishedSampling(model: string, mode: SamplingMode): SamplingValues | undefined {
  return publishedSamplingProfile(model)?.[mode]
}

/** `temperature 1, top_p 0.95, top_k 20, …` for status text. */
export function formatSamplingValues(values: SamplingValues): string {
  return `temperature ${values.temperature}, top_p ${values.topP}, top_k ${values.topK}, min_p ${values.minP}, `
    + `presence_penalty ${values.presencePenalty}, repetition_penalty ${values.repetitionPenalty}`
}

/**
 * The sampling Xerxes fills in when nothing was configured. A model with a
 * published profile gets none here: its profile is applied per request, where
 * the thinking or non-thinking values can be chosen, so an unset temperature
 * must stay unset until then.
 */
export function nativeSamplingDefaults(model: string | undefined): { readonly temperature?: number; readonly top_k?: number } {
  if (model !== undefined && publishedSamplingProfile(model)) return {}
  return { temperature: DEFAULT_TEMPERATURE, top_k: DEFAULT_TOP_K }
}
